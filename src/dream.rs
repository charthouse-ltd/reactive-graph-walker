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

    // Phase 2: Run many fast parallel walks with noise.
    // WRAPPED in spawn_blocking: rayon's par_iter + block_on DB calls
    // must NOT run on the async runtime's worker threads (deadlocks on
    // constrained CPUs). The blocking pool absorbs the load.
    let pool_clone = pool.clone();
    let noise = config.noise_magnitude;
    let steps = config.steps;
    let seeds = all_seeds.clone();

    // ── Pre-fetch knowledge domains for dream walk cross-reference ──
    let knowledge_domains = db::all_knowledge_domains(pool).await.unwrap_or_default();
    let kd_for_walks = knowledge_domains.clone(); // moved into the walk closure; original kept for Phase 4

    let dream_results: Vec<DreamWalkResult> = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        (0..config.n_walks)
            .into_par_iter()
            .map(|i| {
                let seed = seeds[i % seeds.len()];
                dream_walk_single(&pool_clone, seed, noise, steps, &rt, &kd_for_walks)
            })
            .collect()
    })
    .await
    .unwrap_or_default();

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
        let mut coherence = (*votes as f32 / config.n_walks as f32 * 10.0).min(1.0); // Normalize

        // RAG anchoring: a connection grounded in verified compendium knowledge
        // is more credible than pure noise — boost its coherence so dreams
        // preferentially keep knowledge-anchored insights. (This is what the
        // knowledge_anchors cross-reference is FOR; previously it was unused.)
        if knowledge_domains.contains(src_domain) || knowledge_domains.contains(tgt_domain) {
            coherence = (coherence + 0.25).min(1.0);
        }

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
    /// Number of steps that landed on a domain with verified knowledge (compendium match)
    knowledge_anchors: usize,
}

/// Single dream walk with noise perturbation
fn dream_walk_single(
    pool: &PgPool,
    seed_id: i32,
    noise_magnitude: f32,
    steps: usize,
    rt: &tokio::runtime::Handle,
    knowledge_domains: &std::collections::HashSet<String>,
) -> DreamWalkResult {
    let mut result = DreamWalkResult {
        node_path: vec![seed_id],
        domains_path: Vec::new(),
        total_weight: 0.0,
        surprises: 0,
        knowledge_anchors: 0,
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

            // ── Knowledge cross-reference during dream ──
            // Even in REM sleep, check against immutable knowledge.
            // Dreams that pass through grounded domains produce higher-coherence
            // insights — the graph grows toward truth, not just noise.
            if !node.domain.is_empty() && knowledge_domains.contains(&node.domain) {
                result.knowledge_anchors += 1;
            }

            // Note: we do NOT modify edges during dreams
            // Dreams observe but don't change the real graph
            // Only the kept insights create permanent edges (Phase 5)
        }

        current_id = next_id;
    }

    result
}

// ── Concurrent Dream Loop ────────────────────────────────────────
// Not a separate "sleep" phase — a continuous background process
// that always runs alongside the reactive walker loop.
//
// Reactive (awake):   edge changes → energy → walks → strengthen → motor
// Dream (always-on):  noise walks → new edges + prune → motor DISCONNECTED
//
// Energy is a modulator, not a switch:
//   high energy → reactive dominates (less dreaming)
//   low energy → dream dominates (more dreaming, more pruning)
//
// This replaces time-sharing with concurrency — the graph is
// continuously explored AND continuously renormalized.

/// Spawn a background dream loop that runs continuously.
/// Modulated by energy: high energy → slower dream cycle,
/// low energy → faster dream cycle, more pruning.
pub fn start_dream_loop(
    pool: PgPool,
    self_model: Arc<tokio::sync::Mutex<SelfModel>>,
) {
    tokio::spawn(async move {
        let config = DreamConfig {
            n_walks: 20,              // Fewer walks per cycle (continuous, not batched)
            steps: 6,
            noise_magnitude: 0.10,    // Lower noise for continuous operation
            coherence_threshold: 0.5,
            max_new_edges: 5,
        };

        tracing::info!("[dream] Concurrent dream loop started (always-on, motor-disconnected)");

        loop {
            // ── Energy-modulated cycle time ──
            // Higher energy → longer sleep between dream cycles
            let (energy, mode) = {
                let sm = self_model.lock().await;
                (sm.energy, sm.mode.clone())
            };

            if mode == core::CognitiveMode::Compliant {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                continue;
            }

            // Energy 0.0 → dream every 3s; energy 1.0 → dream every 30s
            let cycle_secs = 3.0 + (1.0 - energy) as f64 * 27.0;
            tokio::time::sleep(std::time::Duration::from_secs_f64(cycle_secs)).await;

            // ── Run a dream cycle ──
            let report = dream(&pool, &self_model, config.clone()).await;
            if report.edges_created > 0 || report.connections_kept > 0 {
                tracing::info!(
                    "[dream] Cycle: {} edges created, {} connections kept, {} insights (energy={:.2})",
                    report.edges_created, report.connections_kept, report.insights.len(), energy
                );
            }

            // ── Homeostasis: prune weak nodes (DESTRUCTIVE, opt-in) ──
            // prune_nodes hard-DELETEs rows from memory_vectors. Running that
            // unattended in an always-on loop is data loss by default, so it is
            // gated behind RGW_ENABLE_AUTO_PRUNE (off unless explicitly set) and
            // audit-logged at WARN. Operator-triggered edge pruning via
            // POST /prune is unaffected.
            let auto_prune = std::env::var("RGW_ENABLE_AUTO_PRUNE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if auto_prune && energy < 0.3 {
                match db::prune_nodes(&pool, 0.1, 7).await {
                    Ok(prune_count) if prune_count > 0 => {
                        tracing::warn!(
                            "[dream] AUTO-PRUNE deleted {} low-importance/stale nodes (RGW_ENABLE_AUTO_PRUNE=on)",
                            prune_count
                        );
                    }
                    Err(e) => {
                        tracing::warn!("[dream] Pruning failed: {}", e);
                    }
                    _ => {}
                }
            }
        }
    });
}
