//! Diverger — self-propagating reactive graph engine.
//!
//! The graph drives its own computation. No loops. No timers. No events.
//! Edge changes cascade to neighbors. When accumulated energy at a node
//! crosses threshold, a spontaneous walk fires. That walk changes more
//! edges. Those changes cascade further. The topology IS the clock.
//!
//! Named "Diverger" because it finds where perspectives break apart,
//! not where they converge. The opposite of a transformer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use sqlx::PgPool;
use tokio::sync::{mpsc, RwLock};

use crate::db;
use crate::graph::*;
use crate::walker;

/// Activation energy accumulated at each node.
/// When energy crosses threshold → spontaneous walk fires.
/// The walk changes edges → cascading activation → more walks.
#[derive(Debug)]
struct NodeEnergy {
    energy: f32,
    last_fired: f64,   // unix timestamp of last spontaneous fire (refractory clock)
    last_update: f64,  // unix timestamp of last energy update (decay clock)
}

/// The Diverger engine — a self-propagating reactive graph.
pub struct Diverger {
    /// Database pool for graph access
    pool: PgPool,

    /// Per-node accumulated activation energy
    nodes: Arc<RwLock<HashMap<i32, NodeEnergy>>>,

    /// Current emotional state (modulates thresholds)
    emotion: Arc<RwLock<EmotionalState>>,

    /// Channel for edge change notifications
    /// Any system that modifies edges pushes to this channel.
    /// The Diverger reacts — no polling needed.
    edge_tx: mpsc::UnboundedSender<EdgeChange>,

    /// Running state
    alive: Arc<AtomicBool>,

    /// Performance counters
    walks_fired: Arc<AtomicU64>,
    cascades_total: Arc<AtomicU64>,
    edges_changed: Arc<AtomicU64>,

    /// Julian's Python backend URL (for motor commands)
    julian_url: String,
}

/// Notification that an edge was modified
#[derive(Debug, Clone)]
pub struct EdgeChange {
    pub edge_id: i32,
    pub source_id: i32,
    pub target_id: i32,
    pub delta: f32,        // how much the weight changed
    pub edge_type: String,
}

/// Diverger configuration
#[derive(Debug, Clone)]
pub struct DivergerConfig {
    /// Base activation threshold (higher = less spontaneous firing)
    pub base_threshold: f32,

    /// Energy decay per second (prevents infinite accumulation)
    pub energy_decay_rate: f32,

    /// Maximum concurrent spontaneous walks
    pub max_concurrent_walks: usize,

    /// Steps per spontaneous walk
    pub walk_steps: usize,

    /// Edge change → energy transfer ratio
    pub propagation_strength: f32,

    /// Minimum time between fires at same node (seconds)
    pub refractory_period: f64,

    /// Max edge changes processed per second (circuit breaker for runaway cascades)
    pub max_edges_per_second: u32,

    /// Max total walks fired per minute (global budget)
    pub max_walks_per_minute: u32,

    /// Propagation depth limit (how many cascade hops before stopping)
    pub max_cascade_depth: u32,
}

impl Default for DivergerConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.5,
            energy_decay_rate: 0.01,
            max_concurrent_walks: 8,
            walk_steps: 5,
            propagation_strength: 0.3,
            refractory_period: 0.5,
            max_edges_per_second: 100,    // Hard cap: 100 edge changes/sec
            max_walks_per_minute: 60,     // Hard cap: 1 walk/sec average
            max_cascade_depth: 3,         // Max 3 hops of propagation
        }
    }
}

/// Statistics from the Diverger
#[derive(Debug, Clone, serde::Serialize)]
pub struct DivergerStats {
    pub alive: bool,
    pub active_nodes: usize,
    pub total_energy: f32,
    pub walks_fired: u64,
    pub cascades_total: u64,
    pub edges_changed: u64,
    pub emotional_state: EmotionalState,
    pub hottest_nodes: Vec<(i32, f32)>,  // top 10 by energy
}

impl Diverger {
    /// Create a new Diverger engine with shared self-model.
    pub fn new(pool: PgPool, self_model: std::sync::Arc<tokio::sync::Mutex<crate::core::SelfModel>>, julian_url: &str) -> Self {
        let (edge_tx, edge_rx) = mpsc::unbounded_channel();
        let sm = self_model.clone();

        let diverger = Self {
            pool,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            emotion: Arc::new(RwLock::new(EmotionalState::default())),
            edge_tx,
            alive: Arc::new(AtomicBool::new(false)),
            walks_fired: Arc::new(AtomicU64::new(0)),
            cascades_total: Arc::new(AtomicU64::new(0)),
            edges_changed: Arc::new(AtomicU64::new(0)),
            julian_url: julian_url.to_string(),
        };

        // Start the reactor (consumes edge_rx) with shared self-model
        diverger.start_reactor(edge_rx, DivergerConfig::default(), sm);

        diverger
    }

    /// Start the reactive core. This is the ONLY "loop" — and it's an
    /// async channel receiver, not a timer. It only runs when edges change.
    fn start_reactor(
        &self,
        mut edge_rx: mpsc::UnboundedReceiver<EdgeChange>,
        config: DivergerConfig,
        self_model: std::sync::Arc<tokio::sync::Mutex<crate::core::SelfModel>>,
    ) {
        let pool = self.pool.clone();
        let nodes = self.nodes.clone();
        let emotion = self.emotion.clone();
        let alive = self.alive.clone();
        let walks_fired = self.walks_fired.clone();
        let cascades_total = self.cascades_total.clone();
        let edges_changed = self.edges_changed.clone();
        let julian_url_outer = self.julian_url.clone();

        alive.store(true, Ordering::SeqCst);

        tokio::spawn(async move {
            let walk_semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_walks));

            // ── Pre-fetch knowledge domains once for all diverger walks ──
            let knowledge_domains = Arc::new(
                crate::db::all_knowledge_domains(&pool).await.unwrap_or_default()
            );

            tracing::info!("[diverger] Reactor started. Waiting for edge changes...");

            // Circuit breaker state
            let mut edges_this_second: u32 = 0;
            let mut walks_this_minute: u32 = 0;
            let mut last_second_reset = std::time::Instant::now();
            let mut last_minute_reset = std::time::Instant::now();

            // This is NOT a polling loop.
            // It awaits on the channel — zero CPU when nothing changes.
            // Each edge change cascades to neighbors, potentially firing walks.
            while let Some(change) = edge_rx.recv().await {
                edges_changed.fetch_add(1, Ordering::Relaxed);

                // ── Circuit breaker: rate limiting ──
                let now_instant = std::time::Instant::now();
                if now_instant.duration_since(last_second_reset).as_secs_f64() >= 1.0 {
                    edges_this_second = 0;
                    last_second_reset = now_instant;
                }
                if now_instant.duration_since(last_minute_reset).as_secs_f64() >= 60.0 {
                    walks_this_minute = 0;
                    last_minute_reset = now_instant;
                }
                edges_this_second += 1;
                if edges_this_second > config.max_edges_per_second {
                    continue; // Drop — too many edges this second
                }

                // Get current emotional state for threshold modulation
                let emo = emotion.read().await.clone();

                // Compute dynamic threshold
                let threshold = config.base_threshold
                    * (1.0 - emo.arousal * 0.3)  // High arousal lowers threshold
                    * (1.0 + (1.0 - emo.energy) * 0.3);  // Low energy raises threshold

                // Propagate energy to both nodes of the changed edge
                let node_ids = [change.source_id, change.target_id];
                let energy_delta = change.delta.abs() * config.propagation_strength;

                let mut nodes_to_fire: Vec<i32> = Vec::new();

                {
                    let mut nodes_lock = nodes.write().await;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64();

                    for &nid in &node_ids {
                        let node = nodes_lock.entry(nid).or_insert(NodeEnergy {
                            energy: 0.0,
                            last_fired: 0.0,
                            last_update: now,
                        });

                        // Apply energy decay since this node was last updated.
                        // Decay is clocked off last_update, NOT last_fired: a node that
                        // has never fired has last_fired=0 (epoch) → dt astronomical →
                        // injected energy wiped before it can accumulate to threshold.
                        let dt = (now - node.last_update).max(0.0) as f32;
                        node.energy = (node.energy - config.energy_decay_rate * dt).max(0.0);
                        node.last_update = now;

                        // Add new energy from the edge change
                        node.energy += energy_delta;

                        // Check threshold crossing
                        if node.energy >= threshold {
                            // Refractory period check
                            if now - node.last_fired >= config.refractory_period {
                                nodes_to_fire.push(nid);
                                node.energy = 0.0;  // Reset after firing
                                node.last_fired = now;
                            }
                        }
                    }
                }

                // In Compliant mode, suppress all spontaneous walks.
                // Energy still propagates (the graph stays alive), but
                // no spontaneous behavior fires. The system only acts
                // when explicitly asked via API.
                let is_autonomous = {
                    let sm_lock = self_model.lock().await;
                    sm_lock.mode == crate::core::CognitiveMode::Autonomous
                };
                if !is_autonomous {
                    continue; // Compliant: observe but don't act
                }

                // Fire spontaneous walks from nodes that crossed threshold
                for node_id in nodes_to_fire {
                    // Walk budget check (circuit breaker)
                    if walks_this_minute >= config.max_walks_per_minute {
                        tracing::debug!("[diverger] Walk budget exhausted ({}/min) — throttling", config.max_walks_per_minute);
                        break;
                    }

                    let permit = match walk_semaphore.clone().try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => continue,  // Too many concurrent walks — skip
                    };

                    let pool = pool.clone();
                    let emo = emo.clone();
                    let walks_count = walks_fired.clone();
                    let cascade_count = cascades_total.clone();
                    let steps = config.walk_steps;

                    walks_count.fetch_add(1, Ordering::Relaxed);
                    cascade_count.fetch_add(1, Ordering::Relaxed);
                    walks_this_minute += 1;

                    let sm_walk = self_model.clone();
                    let julian_url = julian_url_outer.clone();
                        let kd = knowledge_domains.clone();
                        tokio::spawn(async move {
                        // Preload the seed's neighborhood (radius 1) before walking:
                        // walk_single is cache-only with NO DB fallback, so without a
                        // preload it dead-ends at hop 1 and never reaches the motor.
                        let cache = std::sync::Arc::new(
                            crate::edge_cache::EdgeCache::new(pool.clone())
                        );
                        let _ = cache.preload_radius(&[node_id]).await;

                        // Fire the walk on a blocking thread (true parallelism)
                        let cache_walk = cache.clone();
                        let result = tokio::task::spawn_blocking(move || {
                            let mut sm = sm_walk.blocking_lock().clone();
                            let collective = std::sync::Arc::new(std::sync::Mutex::new(
                                crate::graph::WalkerCollective::new()
                            ));
                            crate::walker::walk_single(
                                &*cache_walk, node_id,
                                WalkerBias::all()[node_id as usize % WalkerBias::all().len()],
                                &emo, steps, &mut sm, &collective, None, &kd,
                            )
                        })
                        .await;

                        if let Ok(result) = result {
                            // Hebbian: reinforce the edges this walk traversed.
                            if !result.edges_traversed.is_empty() {
                                cache.strengthen_edges(&result.edges_traversed, 0.02);
                            }

                            let hops = result.path.len();
                            let surprises = result.surprises;
                            let domains = result.domains_visited.clone();

                            tracing::debug!(
                                "[diverger] Spontaneous walk from node {}: {} hops, {} surprises, domains: {:?}",
                                node_id, hops, surprises, domains,
                            );

                            // ── MOTOR CORTEX: if walk produced actionable result, command Julian ──
                            // Only act on walks with enough substance (3+ hops, has domains)
                            if hops >= 3 && !domains.is_empty() {
                                let dead_ends = result.dead_ends;
                                let action = if dead_ends >= 2 && !domains.is_empty() {
                                    "search"  // Hit dead ends = knowledge gap → search
                                } else if surprises >= 2 {
                                    "explore"  // Cross-domain discovery
                                } else if hops >= 4 && surprises == 0 {
                                    "journal"  // Deep traversal, no surprise = reflection
                                } else {
                                    "nothing"
                                };

                                let search_query = if action == "search" {
                                    Some(format!("{} latest", domains.first().cloned().unwrap_or_default()))
                                } else {
                                    None
                                };

                                if action != "nothing" && !julian_url.is_empty() {
                                    let cmd = crate::motor::MotorCommand {
                                        action: action.to_string(),
                                        domain: domains.first().cloned().unwrap_or_default(),
                                        walker_context: format!(
                                            "Spontaneous thought: {} hops through {:?}, {} surprises, {} dead ends",
                                            hops, domains, surprises, dead_ends
                                        ),
                                        expression_seeds: Vec::new(),
                                        confidence: if surprises > 0 { 0.4 } else { 0.6 },
                                        novelty: surprises as f32 / hops.max(1) as f32,
                                        search_query,
                                        params: None,
                                        audience_id: None,  // Spontaneous walks have no specific audience
                                    };
                                    crate::motor::execute(&julian_url, cmd).await;
                                }
                            }
                        }

                        // Persist edge strengthening accumulated during the walk.
                        let _ = cache.flush().await;

                        drop(permit);  // Release semaphore
                    });
                }
            }

            alive.store(false, Ordering::SeqCst);
            tracing::info!("[diverger] Reactor stopped.");
        });
    }

    /// Notify the Diverger that an edge was changed.
    /// This is the ONLY input the Diverger needs.
    /// Call this from: walker traversal, unconscious tax, memory storage, outcome learning.
    pub fn notify_edge_change(&self, change: EdgeChange) {
        let _ = self.edge_tx.send(change);
    }

    /// Update emotional state (affects activation thresholds)
    pub async fn set_emotion(&self, state: EmotionalState) {
        *self.emotion.write().await = state;
    }

    /// Get current statistics
    pub async fn stats(&self) -> DivergerStats {
        let nodes = self.nodes.read().await;
        let emo = self.emotion.read().await;

        let total_energy: f32 = nodes.values().map(|n| n.energy).sum();

        // Top 10 hottest nodes
        let mut hot: Vec<(i32, f32)> = nodes.iter().map(|(&id, n)| (id, n.energy)).collect();
        hot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hot.truncate(10);

        DivergerStats {
            alive: self.alive.load(Ordering::Relaxed),
            active_nodes: nodes.len(),
            total_energy,
            walks_fired: self.walks_fired.load(Ordering::Relaxed),
            cascades_total: self.cascades_total.load(Ordering::Relaxed),
            edges_changed: self.edges_changed.load(Ordering::Relaxed),
            emotional_state: emo.clone(),
            hottest_nodes: hot,
        }
    }

    /// Seed the Diverger with initial energy (kickstart on boot)
    pub async fn seed_energy(&self, node_ids: Vec<i32>, initial_energy: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let mut nodes = self.nodes.write().await;
        for id in node_ids {
            nodes.entry(id).or_insert(NodeEnergy {
                energy: initial_energy,
                last_fired: 0.0,
                last_update: now,
            });
        }
        tracing::info!("[diverger] Seeded {} nodes with {:.2} energy", nodes.len(), initial_energy);
    }
}
