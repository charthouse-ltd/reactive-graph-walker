//! Edge Cache — in-memory hot subgraph for walker performance.
//!
//! Walkers hit RAM instead of PostgreSQL for edge lookups.
//! Edge neighborhoods are preloaded in batches. Write-back
//! batches flush to pgvector periodically.
//!
//! This is the difference between 20 hops/second and 2000+.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use sqlx::PgPool;

use crate::db::MemoryEdge;

/// Thread-safe edge cache using lock-free DashMap.
/// Walkers read without contention. Writes are atomic.
pub struct EdgeCache {
    /// node_id → outgoing/incoming edges
    edges: Arc<DashMap<i32, Vec<MemoryEdge>>>,

    /// Pending edge writes (strengthening, new edges)
    /// Batch-flushed to DB periodically
    pending_write: Arc<DashMap<i32, f32>>, // edge_id → delta

    /// Nodes currently loaded in cache
    loaded_nodes: Arc<DashMap<i32, bool>>,

    /// Node domains cached for scoring (node_id → domain)
    node_domains: Arc<DashMap<i32, String>>,

    /// DB pool for cache misses
    pool: PgPool,

    /// Cache statistics
    pub hits: Arc<std::sync::atomic::AtomicU64>,
    pub misses: Arc<std::sync::atomic::AtomicU64>,
    pub flushes: Arc<std::sync::atomic::AtomicU64>,
}

struct CacheStats {
    hits: u64,
    misses: u64,
    flushes: u64,
    loaded_nodes: usize,
    pending_writes: usize,
}

impl EdgeCache {
    /// Create a new edge cache backed by the given DB pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            edges: Arc::new(DashMap::new()),
            pending_write: Arc::new(DashMap::new()),
            loaded_nodes: Arc::new(DashMap::new()),
            node_domains: Arc::new(DashMap::new()),
            pool,
            hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            flushes: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Get edges for a node. Synchronous — lock-free DashMap read.
    /// Returns empty vec on cache miss (caller should preload).
    pub fn try_get(&self, node_id: i32) -> Vec<MemoryEdge> {
        if let Some(edges) = self.edges.get(&node_id) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return edges.clone();
        }
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Vec::new()
    }

    /// Get a node's domain synchronously from cache.
    pub fn get_domain(&self, node_id: i32) -> Option<String> {
        self.node_domains.get(&node_id).map(|d| d.clone())
    }

    /// Preload edges for a batch of node IDs.
    /// Called before a walk session to warm the cache.
    pub async fn preload_batch(&self, node_ids: &[i32]) -> Result<usize, sqlx::Error> {
        let mut loaded = 0;

        for &nid in node_ids {
            if self.loaded_nodes.contains_key(&nid) {
                continue;
            }

            let edges = crate::db::edges_from(&self.pool, nid).await?;
            self.edges.insert(nid, edges);
            self.loaded_nodes.insert(nid, true);
            loaded += 1;
        }

        if loaded > 0 {
            tracing::debug!("[cache] Preloaded {} node neighborhoods", loaded);
        }

        Ok(loaded)
    }

    /// Preload edges for seed nodes plus their 1-hop neighbors.
    /// This covers most walker steps in a session.
    pub async fn preload_radius(&self, seed_ids: &[i32]) -> Result<usize, sqlx::Error> {
        let mut to_load: HashSet<i32> = seed_ids.iter().copied().collect();
        let mut loaded = 0;

        // First pass: load seeds
        for &nid in seed_ids {
            if !self.loaded_nodes.contains_key(&nid) {
                let edges = crate::db::edges_from(&self.pool, nid).await?;
                // Also cache node domain
                if let Ok(Some(node)) = crate::db::get_node(&self.pool, nid).await {
                    if !node.domain.is_empty() {
                        self.node_domains.insert(nid, node.domain);
                    }
                }
                // Add neighbors to the load set
                for e in &edges {
                    let neighbor = if e.source_id == nid { e.target_id } else { e.source_id };
                    to_load.insert(neighbor);
                }
                self.edges.insert(nid, edges);
                self.loaded_nodes.insert(nid, true);
                loaded += 1;
            }
        }

        // Second pass: load 1-hop neighbors
        for nid in to_load {
            if nid == 0 || self.loaded_nodes.contains_key(&nid) {
                continue;
            }
            // Skip if it was a seed (already loaded)
            if seed_ids.contains(&nid) {
                continue;
            }
            let edges = crate::db::edges_from(&self.pool, nid).await?;
            self.edges.insert(nid, edges);
            self.loaded_nodes.insert(nid, true);
            loaded += 1;
        }

        if loaded > 0 {
            tracing::info!("[cache] Preloaded {} node neighborhoods (radius=1)", loaded);
        }

        Ok(loaded)
    }

    /// Queue an edge strengthening for batch flush.
    /// Non-blocking — just adds to the pending map.
    pub fn strengthen_edge(&self, edge_id: i32, delta: f32) {
        self.pending_write
            .entry(edge_id)
            .and_modify(|d| *d += delta)
            .or_insert(delta);
    }

    /// Queue multiple edge strengthenings.
    pub fn strengthen_edges(&self, edge_ids: &[i32], delta: f32) {
        for &eid in edge_ids {
            self.pending_write
                .entry(eid)
                .and_modify(|d| *d += delta)
                .or_insert(delta);
        }
    }

    /// Add a newly created edge to the cache immediately
    /// (so subsequent walks see it).
    pub fn cache_new_edge(&self, edge: &MemoryEdge) {
        // Add to source node's edge list
        self.edges
            .entry(edge.source_id)
            .and_modify(|edges| edges.push(edge.clone()))
            .or_insert_with(|| vec![edge.clone()]);

        // Also add to target node (edges are bidirectional for walkers)
        let mut reversed = edge.clone();
        std::mem::swap(&mut reversed.source_id, &mut reversed.target_id);
        self.edges
            .entry(edge.target_id)
            .and_modify(|edges| edges.push(reversed.clone()))
            .or_insert_with(|| vec![reversed]);
    }

    /// Flush pending edge writes to the database.
    /// Called periodically or at session end.
    pub async fn flush(&self) -> Result<u64, sqlx::Error> {
        let pending: Vec<(i32, f32)> = self.pending_write
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();

        if pending.is_empty() {
            return Ok(0);
        }

        let count = pending.len() as u64;

        // Batch strengthen
        let edge_ids: Vec<i32> = pending.iter().map(|(id, _)| *id).collect();
        let avg_delta: f32 = pending.iter().map(|(_, d)| *d).sum::<f32>() / pending.len() as f32;

        crate::db::strengthen_edges(&self.pool, &edge_ids, avg_delta).await?;

        // Clear pending
        self.pending_write.clear();
        self.flushes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        tracing::debug!("[cache] Flushed {} edge writes to DB", count);
        Ok(count)
    }

    /// Invalidate a node's cache entry (called when edges change externally).
    pub fn invalidate_node(&self, node_id: i32) {
        self.edges.remove(&node_id);
        self.loaded_nodes.remove(&node_id);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
            flushes: self.flushes.load(std::sync::atomic::Ordering::Relaxed),
            loaded_nodes: self.loaded_nodes.len(),
            pending_writes: self.pending_write.len(),
        }
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        self.edges.clear();
        self.loaded_nodes.clear();
        self.pending_write.clear();
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "hits={} misses={} hit_rate={:.0}% nodes={} pending_writes={} flushes={}",
            self.hits, self.misses, hit_rate, self.loaded_nodes, self.pending_writes, self.flushes
        )
    }
}
