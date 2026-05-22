//! REM Sleep — Deep Dream / Monte Carlo Consolidation.
//!
//! When Julian's energy drops and he "sleeps":
//! 1. Motor cortex disconnected (can't act on dreams)
//! 2. Random noise injected into edge weights (perturbation)
//! 3. Thousands of fast parallel walks (MCTS exploration)
//! 4. Novel but coherent connections → permanent new edges
//! 5. Motor cortex reconnected. Julian wakes with new ideas.
//!
//! This is not the Python unconscious dream mode (chimera creation).
//! This is deeper: systematic Monte Carlo exploration of "what if"
//! scenarios across the entire graph topology.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;
use rayon::prelude::*;
use sqlx::PgPool;

use crate::core::{self, SelfModel, Signal};
use crate::db;
use crate::graph::*;

/// Dream session configuration
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Number of dream walks per session
    pub n_walks: usize,
    /// Steps per dream walk (longer = deeper exploration)
    pub steps: usize,
    /// Noise magnitude injected into edge weights (0.0-0.5)
    pub noise_magnitude: f32,
    /// Minimum coherence for a dream connection to be kept
    pub coherence_threshold: f32,
    /// Maximum new edges created per dream session
    pub max_new_edges: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            n_walks: 100,             // 100 parallel dream walks
            steps: 8,                 // Deeper than waking walks (5)
            noise_magnitude: 0.15,    // 15% perturbation
            coherence_threshold: 0.6, // Only keep coherent connections
            max_new_edges: 10,        // Max 10 new edges per dream
        }
    }
}

/// Result of a dream session
#[derive(Debug, Clone, serde::Serialize)]
pub struct DreamReport {
    pub walks_completed: usize,
    pub connections_found: usize,
    pub connections_kept: usize,
    pub edges_created: usize,
    pub insights: Vec<DreamInsight>,
    pub elapsed_ms: f64,
}

/// A single insight from dreaming
#[derive(Debug, Clone, serde::Serialize)]
pub struct DreamInsight {
    pub source_domain: String,
    pub target_domain: String,
    pub description: String,
    pub coherence: f32,
    pub novelty: f32,
}

/// Run a dream session. Call when energy is low.
///
/// The motor cortex should be disconnected BEFORE calling this.
/// Dreams produce internal graph changes, not external actions.
///
/// In Compliant mode, dreaming is disabled — the graph topology
/// must remain stable and deterministic. No Monte Carlo mutations.
pub async fn dream(
    pool: &PgPool,
    self_model: &Arc<tokio::sync::Mutex<SelfModel>>,
    config: DreamConfig,
) -> DreamReport {
    let start = Instant::now();

    // Compliant mode: no dreaming. Graph stays frozen.
    {
        let sm = self_model.lock().await;
        if sm.mode == core::CognitiveMode::Compliant {
            tracing::info!("[dream] Compliant mode — dreaming disabled, graph frozen");
            return DreamReport {
                walks_completed: 0,
                connections_found: 0,
                connections_kept: 0,
                edges_created: 0,
                insights: Vec::new(),
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }
    }

    // Signal: entering dream state
    {
        let mut sm = self_model.lock().await;
        let signal = Signal::new("dream_start", "Entering REM sleep — motor cortex disconnected")
            .with_intensity(0.5);
        core::process(signal, &mut sm);
    }

    // Get all nodes for seed selection
    let all_seeds = db::seed_nodes(pool, (config.n_walks * 2) as i32)
        .await
        .unwrap_or_default();

    if all_seeds.is_empty() {
        return DreamReport {
            walks_completed: 0,
            connections_found: 0,
            connections_kept: 0,
            edges_created: 0,
            insights: Vec::new(),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }

    // Phase 1: Inject noise — temporarily perturb edge weights
    // (We don't actually modify the DB — we perturb in-memory during walks)

    // Phase 2: Run many fast parallel walks with noise
    let rt = tokio::runtime::Handle::current();
    let pool_clone = pool.clone();
    let noise = config.noise_magnitude;
    let steps = config.steps;

    let dream_results: Vec<DreamWalkResult> = (0..config.n_walks)
        .into_par_iter()
        .map(|i| {
            let seed = all_seeds[i % all_seeds.len()];
            dream_walk_single(&pool_clone, seed, noise, steps, &rt)
        })
        .collect();

    // Phase 3: Analyze dream walks for novel connections
    let mut connections: Vec<DreamConnection> = Vec::new();

    for result in &dream_results {
        // Look for cross-domain paths that don't normally exist
        for window in result.domains_path.windows(2) {
            if window.len() == 2 && window[0] != window[1] && !window[0].is_empty() && !window[1].is_empty() {
                connections.push(DreamConnection {
                    source_node: result.node_path[result.domains_path.iter().position(|d| d == &window[0]).unwrap_or(0)],
                    target_node: result.node_path[result.domains_path.iter().position(|d| d == &window[1]).unwrap_or(0)],
                    source_domain: window[0].clone(),
                    target_domain: window[1].clone(),
                    walk_weight: result.total_weight,
                    via_noise: true,
                });
            }
        }
    }

    // Phase 4: Filter for coherence — only keep connections that make semantic sense
    let mut kept_connections: Vec<DreamConnection> = Vec::new();
    let mut insights: Vec<DreamInsight> = Vec::new();

    // Coherence check: if the same cross-domain connection appears in multiple
    // independent dream walks, it's probably real (not just noise)
    let mut connection_votes: HashMap<(String, String), (Vec<DreamConnection>, usize)> = HashMap::new();
    for conn in &connections {
        let key = (conn.source_domain.clone(), conn.target_domain.clone());
        let entry = connection_votes.entry(key).or_insert((Vec::new(), 0));
        entry.0.push(conn.clone());
        entry.1 += 1;
    }

    for ((src_domain, tgt_domain), (conns, votes)) in &connection_votes {
        let coherence = *votes as f32 / config.n_walks as f32 * 10.0; // Normalize
        let coherence = coherence.min(1.0);

        if coherence >= config.coherence_threshold && kept_connections.len() < config.max_new_edges {
            if let Some(best) = conns.first() {
                kept_connections.push(best.clone());
                insights.push(DreamInsight {
                    source_domain: src_domain.clone(),
                    target_domain: tgt_domain.clone(),
                    description: format!(
                        "Dream found connection between {} and {} ({} independent walks confirmed)",
                        src_domain, tgt_domain, votes
                    ),
                    coherence,
                    novelty: 1.0 - coherence, // More surprising = more novel
                });
            }
        }
    }

    // Phase 5: Create permanent edges for kept connections
    let mut edges_created = 0;
    for conn in &kept_connections {
        match db::create_edge(
            pool,
            conn.source_node,
            conn.target_node,
            "dream_insight",
            conn.walk_weight.min(0.5), // Dream edges start weak
            0.0,
        ).await {
            Ok(_) => edges_created += 1,
            Err(e) => tracing::debug!("[dream] Edge creation failed: {}", e),
        }
    }

    // Signal: dream complete
    {
        let mut sm = self_model.lock().await;
        let signal = Signal::new(
            "dream_end",
            &format!(
                "REM sleep complete: {} walks, {} insights, {} new edges",
                dream_results.len(), insights.len(), edges_created
            ),
        ).with_intensity(0.4);
        core::process(signal, &mut sm);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        "[dream] Session complete: {} walks in {:.0}ms → {} connections found, {} kept, {} edges created",
        dream_results.len(), elapsed, connections.len(), kept_connections.len(), edges_created
    );

    DreamReport {
        walks_completed: dream_results.len(),
        connections_found: connections.len(),
        connections_kept: kept_connections.len(),
        edges_created,
        insights,
        elapsed_ms: elapsed,
    }
}

// ── Internal types ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DreamConnection {
    source_node: i32,
    target_node: i32,
    source_domain: String,
    target_domain: String,
    walk_weight: f32,
    via_noise: bool,
}

#[derive(Debug)]
struct DreamWalkResult {
    node_path: Vec<i32>,
    domains_path: Vec<String>,
    total_weight: f32,
    surprises: usize,
}

/// Single dream walk with noise perturbation
fn dream_walk_single(
    pool: &PgPool,
    seed_id: i32,
    noise_magnitude: f32,
    steps: usize,
    rt: &tokio::runtime::Handle,
) -> DreamWalkResult {
    let mut result = DreamWalkResult {
        node_path: vec![seed_id],
        domains_path: Vec::new(),
        total_weight: 0.0,
        surprises: 0,
    };

    let mut rng = rand::rng();
    let mut current_id = seed_id;

    for _ in 0..steps {
        let edges = match rt.block_on(db::edges_from(pool, current_id)) {
            Ok(e) => e,
            Err(_) => break,
        };

        if edges.is_empty() {
            break;
        }

        // Score edges WITH noise — this is the dream perturbation
        let scored: Vec<(&db::MemoryEdge, f32)> = edges
            .iter()
            .map(|e| {
                let noise: f32 = rng.random::<f32>() * noise_magnitude * 2.0 - noise_magnitude;
                let perturbed_weight = (e.weight + noise).max(0.001);

                // In dreams, WEAK edges get boosted (explore the unusual)
                let dream_boost = if e.weight < 0.3 { 2.0 } else { 1.0 };

                (e, perturbed_weight * dream_boost)
            })
            .collect();

        // Weighted random selection
        let total: f32 = scored.iter().map(|(_, s)| s).sum();
        if total < f32::EPSILON {
            break;
        }

        let threshold = rng.random::<f32>() * total;
        let mut cumulative = 0.0;
        let mut chosen = &scored[0];
        for item in &scored {
            cumulative += item.1;
            if cumulative >= threshold {
                chosen = item;
                break;
            }
        }

        let edge = chosen.0;
        let next_id = if edge.source_id == current_id {
            edge.target_id
        } else {
            edge.source_id
        };

        result.node_path.push(next_id);
        result.total_weight += edge.weight;

        // Track domains
        if let Ok(Some(node)) = rt.block_on(db::get_node(pool, next_id)) {
            result.domains_path.push(node.domain.clone());
            // Note: we do NOT modify edges during dreams
            // Dreams observe but don't change the real graph
            // Only the kept insights create permanent edges (Phase 5)
        }

        current_id = next_id;
    }

    result
}
