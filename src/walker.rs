//! The walker engine — parallel graph traversal with emotional biasing.
//!
//! Each walker traverses the graph independently on its own thread (via rayon).
//! The walk changes the graph: traversed edges get strengthened.
//! Multiple walkers with different biases produce convergence/divergence signals.
//!
//! Walkers now share a stigmergic collective context — they leave trails
//! for each other (visited nodes, dead ends, surprise domains) and
//! modulate edge scoring based on what other walkers have discovered.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::Rng;
use rayon::prelude::*;
use sqlx::PgPool;

use crate::core::{self, Signal, SelfModel, Noticing, safe_truncate};
use crate::db;
use crate::edge_cache::EdgeCache;
use crate::graph::*;

/// Run a single walker through the graph.
/// Now accepts a shared WalkerCollective for inter-walker stigmergy.
/// Walkers leave trails (visited nodes, dead ends, surprises) and
/// read each other's trails to diversify exploration.
pub fn walk_single(
    cache: &EdgeCache,
    seed_id: i32,
    bias: WalkerBias,
    emotion: &EmotionalState,
    steps: usize,
    self_model: &mut SelfModel,
    collective: &Arc<Mutex<WalkerCollective>>,
    learned_bias: Option<&crate::graph::LearnedBias>,
    knowledge_domains: &std::collections::HashSet<String>,
) -> WalkerResult {
    let mut result = WalkerResult {
        bias,
        path: vec![seed_id],
        domains_visited: Vec::new(),
        edge_types_used: Vec::new(),
        total_weight: 0.0,
        surprises: 0,
        dead_ends: 0,
        edges_traversed: Vec::new(),
    };

    // Signal the self-model: walk is starting
    let start_signal = Signal::new("walk_start", &format!("Walking from node {} with {:?} bias", seed_id, bias))
        .with_intensity(0.3);
    core::process(start_signal, self_model);

    let mut current_id = seed_id;
    let mut rng = rand::rng();
    let mut prev_domain = String::new();

    for step in 0..steps {
        // Get edges from current node — sync cache read (lock-free)
        let edges = cache.try_get(current_id);
        if edges.is_empty() {
            result.dead_ends += 1;
            // Write dead end to collective so other walkers avoid this node
            if let Ok(mut c) = collective.try_lock() {
                c.dead_end_nodes.insert(current_id);
            }
            // Self-model notices: dead end
            let signal = Signal::new("dead_end", &format!("No edges from node {}", current_id))
                .with_intensity(0.2);
            core::process(signal, self_model);
            break;
        }

        // Score each edge — mode determines whether emotion participates
        // In Autonomous mode: full context scoring (collective + self-model)
        // In Compliant mode: pure weight-based, no emotional/collective influence
        let scored: Vec<(&db::MemoryEdge, f32)> = if self_model.mode == core::CognitiveMode::Compliant {
            // Compliant: pure weight-based scoring, no emotional/collective influence
            edges.iter().map(|e| {
                let score = bias.score_edge_compliant(
                    &e.edge_type,
                    e.weight,
                    e.traversal_count,
                );
                (e, score)
            }).collect()
        } else {
            // Autonomous: full context scoring
            // ── Snapshot collective state (release lock BEFORE scoring loop) ──
            let collective_snapshot: Option<WalkerCollective> = collective.try_lock().ok().map(|c| c.clone());
            let current_emotion = EmotionalState {
                valence: self_model.valence,
                arousal: self_model.arousal,
                energy: self_model.energy,
            };

            // Extract working memory domains for scoring
            let wm_domains: HashSet<String> = self_model.working_memory
                .iter()
                .map(|s| s.domain.clone())
                .collect();

            // ── Batch pre-fetch node domains for all edge targets ──
            // Avoids N+1 blocking DB queries in the hot scoring loop
            let target_ids: Vec<i32> = edges.iter()
                .map(|e| if e.source_id == current_id { e.target_id } else { e.source_id })
                .collect();
            let domain_map: HashMap<i32, String> = {
                let mut map = HashMap::new();
                for &tid in &target_ids {
                    if let Some(domain) = cache.get_domain(tid) {
                        map.insert(tid, domain);
                    }
                }
                map
            };

            let current_domain = prev_domain.clone();
            let collective_ref = collective_snapshot.as_ref();
            edges.iter().map(|e| {
                let next_id = if e.source_id == current_id { e.target_id } else { e.source_id };
                let next_domain = domain_map.get(&next_id).map(|s| s.as_str()).unwrap_or("");

                let score = if let Some(lb) = learned_bias {
                    // Emergent learned bias — adapts from experience.
                    // Fix #3: goal pursuit + audience bias are Autonomous-only; Compliant
                    // threads goal_strength = 0 / no audience so scoring stays deterministic.
                    let (goal_domain, goal_strength, audience_model, audience_id) =
                        if self_model.mode == core::CognitiveMode::Autonomous {
                            (
                                self_model.active_goal_domain.as_deref(),
                                self_model.active_goal_strength,
                                Some(&self_model.audience_model),
                                self_model.active_audience_id.as_deref(),
                            )
                        } else {
                            (None, 0.0, None, None)
                        };
                    lb.score_edge(
                        &e.edge_type, e.weight, e.emotional_charge,
                        e.traversal_count, &current_emotion,
                        next_domain, &current_domain,
                        goal_domain, goal_strength, audience_model, audience_id,
                    )
                } else {
                    // Fallback: hardcoded WalkerBias with full context
                    bias.score_edge_with_context(
                        &e.edge_type,
                        e.weight,
                        e.emotional_charge,
                        e.traversal_count,
                        &current_emotion,
                        collective_ref,
                        next_id,
                        next_domain,
                        &self_model.beliefs,
                        &self_model.wounds,
                        &self_model.competencies,
                        &wm_domains,
                        self_model.active_goal_domain.as_deref(),
                        self_model.active_goal_strength,
                        Some(&self_model.audience_model),
                        self_model.active_audience_id.as_deref(),
                    )
                };
                (e, score)
            }).collect()
        };

        // Weighted random selection
        let total: f32 = scored.iter().map(|(_, s)| s).sum();
        if total < f32::EPSILON {
            result.dead_ends += 1;
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

        // Record traversal
        result.path.push(next_id);
        result.edge_types_used.push(edge.edge_type.clone());
        result.total_weight += edge.weight;
        result.edges_traversed.push(edge.id);

        // Track domain transitions + feed through self-model
        let next_domain = cache.get_domain(next_id).unwrap_or_default();
        if !next_domain.is_empty() {
                let is_surprise = !prev_domain.is_empty() && prev_domain != next_domain;

                // ── WRITE to collective: leave trails for other walkers ──
                if let Ok(mut c) = collective.try_lock() {
                    *c.visited_nodes.entry(next_id).or_insert(0) += 1;
                    *c.active_domains.entry(next_domain.clone()).or_insert(0) += 1;
                    if is_surprise {
                        c.surprise_domains.insert(next_domain.clone());
                    }
                    // Drift collective emotion toward this walker's state
                    let alpha = 0.1; // Slow drift — collective is stable
                    c.drift_valence = c.drift_valence * (1.0 - alpha) + self_model.valence * alpha;
                    c.drift_arousal = c.drift_arousal * (1.0 - alpha) + self_model.arousal * alpha;
                }

                if is_surprise {
                    result.surprises += 1;
                }
                if !result.domains_visited.contains(&next_domain) {
                    result.domains_visited.push(next_domain.clone());
                }

                // ── EVERY STEP PASSES THROUGH THE SELF-MODEL ──
                let signal_kind = if is_surprise { "surprise" } else { "walk_step" };
                let signal = Signal::new(
                    signal_kind,
                    &format!("Step {}: {} → {} via {} edge (w={:.2})",
                        step, prev_domain, next_domain, edge.edge_type, edge.weight),
                )
                .with_domain(&next_domain)
                .with_intensity(if is_surprise { 0.6 } else { 0.2 });

                core::process(signal, self_model);
                // The self-model just changed. The next step's edge scoring
                // will be different because the self-model is different.
                // THIS IS COGNITION: the walk changes the thinker,
                // the changed thinker changes the walk.

                // ── Knowledge cross-reference: ground walk in verified truth ──
                // Check the immutable knowledge store to see if this domain
                // has reference material. Anchoring cognition in the compendium
                // diversifies the graph and prevents monomania.
                if knowledge_domains.contains(&next_domain) {
                    let anchor = Signal::new(
                        "knowledge_anchor",
                        &format!(
                            "Grounded in verified knowledge: {} domain",
                            next_domain
                        ),
                    )
                    .with_domain(&next_domain)
                    .with_intensity(0.35);
                    core::process(anchor, self_model);
                }

                prev_domain = next_domain;
        }

        current_id = next_id;
    }

    // Signal the self-model: walk complete
    let end_signal = Signal::new("walk_end", &format!(
        "Walk complete: {} hops, {} surprises, domains: {:?}",
        result.path.len(), result.surprises, result.domains_visited
    )).with_intensity(0.4);
    core::process(end_signal, self_model);

    result
}

/// Run multiple walkers in parallel and aggregate results.
/// The shared self-model is passed in — all walkers feed it.
/// Returns (WalkOutput, per-walker results) for post-walk analysis.
pub async fn walk_parallel(
    pool: &PgPool,
    emotion: &EmotionalState,
    n_walkers: usize,
    steps: usize,
    self_model: &std::sync::Arc<tokio::sync::Mutex<SelfModel>>,
) -> (WalkOutput, Vec<WalkerResult>) {
    let start = Instant::now();

    // Signal: walk session starting
    {
        let mut sm = self_model.lock().await;
        let signal = Signal::new("session_start", &format!("Launching {} walkers", n_walkers))
            .with_intensity(0.4);
        core::process(signal, &mut sm);
    }

    // Get seed nodes
    let seeds = db::seed_nodes(pool, (n_walkers * 2) as i32)
        .await
        .unwrap_or_default();

    if seeds.is_empty() {
        return (empty_output(), Vec::new());
    }

    // Assign biases — mode-dependent
    let mode = {
        let sm = self_model.lock().await;
        sm.mode.clone()
    };
    let biases: &[WalkerBias] = match mode {
        core::CognitiveMode::Autonomous => WalkerBias::all(),
        core::CognitiveMode::Compliant => {
            // Compliant: only Experience (follow strong edges) and
            // Analytical (follow causal/reinforcing edges).
            // No Fear, Curiosity, Random, Contrarian.
            // The walkers find what's there, not what might be.
            &[WalkerBias::Experience, WalkerBias::Analytical]
        }
    };
    let configs: Vec<(i32, WalkerBias)> = (0..n_walkers)
        .map(|i| {
            let seed = seeds[i % seeds.len()];
            let bias = biases[i % biases.len()];
            (seed, bias)
        })
        .collect();

    // ── Edge cache: preload seed neighborhoods for fast walks ──
    let cache = EdgeCache::new(pool.clone());
    let _ = cache.preload_radius(&seeds).await;

    // Clone self-model per walker (parallel perspectives, integrate after)
    // Like the brain: process in parallel, integrate in global workspace
    let base_sm = self_model.lock().await.clone();

    // Save base counter values — merge deltas, not absolute values
    // (each walker_sm starts from base, so absolute values would count base N times)
    let base_surprise_count = base_sm.surprise_count;
    let base_dead_ends: HashMap<String, u32> = base_sm.dead_ends_by_domain.clone();
    let base_signals: HashMap<String, u32> = base_sm.signals_by_source.clone();
    let base_total_signals = base_sm.total_signals_processed;
    let learned_len = base_sm.learned_biases.len();
    let learned_rotation = base_sm.learned_bias_rotation as usize;

    // Create shared stigmergic collective — walkers leave trails for each other
    let collective = Arc::new(Mutex::new(WalkerCollective::new()));

    // ── Pre-fetch knowledge domains for walker cross-reference ──
    let knowledge_domains = db::all_knowledge_domains(pool).await.unwrap_or_default();
    tracing::debug!("[walker] Loaded {} knowledge domains for cross-reference", knowledge_domains.len());

    let walk_start = Instant::now();
    let results: Vec<(WalkerResult, SelfModel)> = configs
        .par_iter()
        .enumerate()
        .map(|(i, (seed, bias))| {
            let mut sm = base_sm.clone();  // Each walker gets its own copy
            let collective = Arc::clone(&collective);
            let learned = if learned_len > 0 {
                base_sm.learned_biases.get((learned_rotation + i) % learned_len)
            } else {
                None
            };
            let result = walk_single(&cache, *seed, *bias, &EmotionalState {
                valence: sm.valence,
                arousal: sm.arousal,
                energy: sm.energy,
            }, steps, &mut sm, &collective, learned, &knowledge_domains);
            (result, sm)
        })
        .collect();
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    // ── Flush edge cache writes to DB ──
    let _ = cache.flush().await;

    // Merge per-walker self-models back into the shared self-model
    // Each walker saw different things — integrate all perspectives
    {
        let mut sm = self_model.lock().await;
        // Proper weighted average that preserves emotional scale
        let n_walkers = results.len() as f32;
        let walker_avg_valence: f32 = results.iter().map(|(_, s)| s.valence).sum::<f32>() / n_walkers;
        let walker_avg_arousal: f32 = results.iter().map(|(_, s)| s.arousal).sum::<f32>() / n_walkers;
        let walker_avg_energy: f32 = results.iter().map(|(_, s)| s.energy).sum::<f32>() / n_walkers;
        sm.valence = sm.valence * 0.5 + walker_avg_valence * 0.5;
        sm.arousal = sm.arousal * 0.5 + walker_avg_arousal * 0.5;
        sm.energy = sm.energy * 0.5 + walker_avg_energy * 0.5;

        for (_, walker_sm) in &results {
            // Merge noticings from all walkers
            for n in &walker_sm.noticings {
                if !sm.noticings.iter().any(|existing| existing.observation == n.observation) {
                    sm.noticings.push(n.clone());
                }
            }
            // Merge attention patterns
            for (domain, &count) in &walker_sm.attention_patterns {
                *sm.attention_patterns.entry(domain.clone()).or_insert(0.0) += count;
            }
            // Merge structural tracking — add deltas, not absolutes
            sm.surprise_count += walker_sm.surprise_count.saturating_sub(base_surprise_count);
            for (domain, &count) in &walker_sm.dead_ends_by_domain {
                let delta = count.saturating_sub(*base_dead_ends.get(domain).unwrap_or(&0));
                if delta > 0 {
                    *sm.dead_ends_by_domain.entry(domain.clone()).or_insert(0) += delta;
                }
            }
            for (source, &count) in &walker_sm.signals_by_source {
                let delta = count.saturating_sub(*base_signals.get(source).unwrap_or(&0));
                if delta > 0 {
                    *sm.signals_by_source.entry(source.clone()).or_insert(0) += delta;
                }
            }
            // Merge predictions from walkers
            for (domain, pred) in &walker_sm.predictions {
                sm.predictions.entry(domain.clone()).or_insert_with(|| pred.clone());
            }
            // Merge working memory (deduplicate by content prefix)
            for wm_slot in &walker_sm.working_memory {
                let key = safe_truncate(&wm_slot.content, 30);
                if !sm.working_memory.iter().any(|s| s.content.starts_with(key)) {
                    sm.working_memory.push_back(wm_slot.clone());
                }
            }
            while sm.working_memory.len() > 5 {
                sm.working_memory.pop_front();
            }
        }
        sm.total_signals_processed += results.iter()
            .map(|(_, s)| s.total_signals_processed.saturating_sub(base_total_signals))
            .sum::<u64>();

        // Signal: integration complete
        let signal = crate::core::Signal::new("integration", &format!(
            "Integrated {} walker perspectives",
            results.len()
        )).with_intensity(0.3);
        crate::core::process(signal, &mut sm);

        // ── Update learned biases from session outcomes ──
        // Each walker's bias adapts based on what it found.
        // Walkers that found surprises → more contradiction-seeking.
        // Walkers that hit dead ends → more novelty-seeking.
        // Stage 0 (PROTOCOL-self-selection): also accumulate each profile's intrinsic
        // fitness scorecard — Autonomous-only, compute-only (nothing selects on it yet).
        let selection_weights = sm.selection_weights.clone();
        let session_repetition = (sm.consecutive_repetitions as f32 / 4.0).min(1.0);
        let accumulate_fitness = sm.mode == core::CognitiveMode::Autonomous;
        for (i, (result, _walker_sm)) in results.iter().enumerate() {
            let bias_idx = if sm.learned_biases.is_empty() {
                None
            } else {
                Some((learned_rotation + i) % sm.learned_biases.len())
            };
            if let Some(bias) = bias_idx.and_then(|idx| sm.learned_biases.get_mut(idx)) {
                let novelty = if result.path.len() > 1 {
                    result.surprises as f32 / result.path.len() as f32
                } else {
                    0.0
                };
                bias.update_from_session(
                    result.surprises as u32,
                    result.dead_ends as u32,
                    novelty,
                    result.domains_visited.len(),
                );
                if accumulate_fitness {
                    let (nov, surprise_kept, dead_end_rate) = crate::graph::fitness_inputs(result);
                    bias.scorecard.record(
                        nov, surprise_kept, dead_end_rate, session_repetition, 0.0, &selection_weights,
                    );
                }
            }
        }
        sm.learned_bias_rotation = sm.learned_bias_rotation.wrapping_add(1);
    }

    // Extract just the walker results for aggregation
    let walker_results: Vec<WalkerResult> = results.into_iter().map(|(r, _)| r).collect();

    // Aggregate results (borrows walker_results, doesn't consume)
    let output = aggregate(pool, &walker_results, walk_ms, start).await;

    // Behavioral integration metric: track novelty and repetition tendency.
    let repeated = {
        let sm = self_model.lock().await;
        sm.consecutive_repetitions > 1
    };
    crate::metrics::record_walk_session(output.novelty_score, repeated);

    // Return both output and walker results for post-walk analysis
    (output, walker_results)
}

/// Fold a single background (Diverger) walk's self-model changes back into the
/// shared self-model. Unlike [`walk_parallel`]'s N-walker integration, this folds
/// in ONE walk gently: emotion is blended (not replaced) so the high-frequency
/// spontaneous loop can't yank the waking mood around, and counters/attention are
/// merged as deltas (walked − base) so the walk's own starting snapshot isn't
/// double-counted.
///
/// Previously the Diverger walked a throwaway clone and discarded it, so the
/// always-on loop was cognitively inert — "the walk changes the thinker" only
/// held for synchronous /walk requests (audit A1). This makes it true for the
/// autonomous loop too.
pub fn integrate_background_walk(shared: &mut SelfModel, base: &SelfModel, walked: &SelfModel) {
    // Gentle emotional drift toward what the walk felt (80/20 blend).
    shared.valence = shared.valence * 0.8 + walked.valence * 0.2;
    shared.arousal = shared.arousal * 0.8 + walked.arousal * 0.2;
    shared.energy = shared.energy * 0.8 + walked.energy * 0.2;

    // New noticings the walk produced (dedup by observation text).
    for n in &walked.noticings {
        if !shared.noticings.iter().any(|e| e.observation == n.observation) {
            shared.noticings.push(n.clone());
        }
    }
    // Bound the buffer the same way core::process does (keep most significant).
    if shared.noticings.len() > 100 {
        shared.noticings.sort_by(|a, b| {
            b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal)
        });
        shared.noticings.truncate(50);
    }

    // Attention deltas (what the walk paid attention to).
    for (domain, &count) in &walked.attention_patterns {
        let prev = base.attention_patterns.get(domain).copied().unwrap_or(0.0);
        let delta = count - prev;
        if delta > 0.0 {
            *shared.attention_patterns.entry(domain.clone()).or_insert(0.0) += delta;
        }
    }

    // Structural counters: add only the deltas this walk produced.
    shared.surprise_count += walked.surprise_count.saturating_sub(base.surprise_count);
    for (domain, &count) in &walked.dead_ends_by_domain {
        let delta = count.saturating_sub(base.dead_ends_by_domain.get(domain).copied().unwrap_or(0));
        if delta > 0 {
            *shared.dead_ends_by_domain.entry(domain.clone()).or_insert(0) += delta;
        }
    }
    shared.total_signals_processed += walked
        .total_signals_processed
        .saturating_sub(base.total_signals_processed);
}

/// Classify visited nodes into (consensus, divergent, blind_spots) by how many
/// walkers landed on each. The three buckets form a COMPLETE partition of every
/// node with ≥1 vote — there is no uncategorized middle band:
///   consensus   : votes > 0.6·n   (most walkers agree)
///   divergent   : 1 < votes ≤ 0.6·n (a minority converged, short of consensus)
///   blind_spots : votes == 1       (only one walker saw it)
fn classify_votes(
    node_votes: &HashMap<i32, usize>,
    n: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let consensus_threshold = n as f32 * 0.6;
    let mut consensus = Vec::new();
    let mut divergent = Vec::new();
    let mut blind_spots = Vec::new();
    for (&node, &votes) in node_votes {
        if votes as f32 > consensus_threshold {
            consensus.push(node);
        } else if votes > 1 {
            divergent.push(node);
        } else {
            blind_spots.push(node); // votes == 1
        }
    }
    (consensus, divergent, blind_spots)
}

/// Domain-level agreement — robust to dispersed seeds, unlike node-overlap `agreement`.
/// Each walker's dominant domain is its most-visited; the score is the fraction of
/// domain-voting walkers whose dominant domain matches the modal dominant. Walkers that
/// visited no domain do not vote. Computed and logged for observability; NOT yet wired
/// into decisions — see PROTOCOL-self-selection.md §#2 (Stage-2 replacement for the
/// structurally-near-zero node-overlap `agreement`).
fn domain_agreement(results: &[WalkerResult]) -> f32 {
    let mut dominant_counts: HashMap<String, usize> = HashMap::new();
    let mut voters = 0usize;
    for r in results {
        let mut per: HashMap<&str, usize> = HashMap::new();
        for d in &r.domains_visited {
            if !d.is_empty() {
                *per.entry(d.as_str()).or_insert(0) += 1;
            }
        }
        if let Some((dom, _)) = per.into_iter().max_by_key(|(_, c)| *c) {
            *dominant_counts.entry(dom.to_string()).or_insert(0) += 1;
            voters += 1;
        }
    }
    if voters == 0 {
        return 0.0;
    }
    let modal = dominant_counts.values().copied().max().unwrap_or(0);
    modal as f32 / voters as f32
}

/// Emotional resonance: the mean edge weight the collective traversed per hop.
/// High = walks moved through strong, well-worn connections; low = weak/random.
/// Clamped to 0..1.
fn compute_resonance(results: &[WalkerResult]) -> f32 {
    let per_walker: Vec<f32> = results
        .iter()
        .filter_map(|r| {
            let hops = r.path.len().saturating_sub(1);
            (hops > 0).then(|| r.total_weight / hops as f32)
        })
        .collect();
    if per_walker.is_empty() {
        0.0
    } else {
        (per_walker.iter().sum::<f32>() / per_walker.len() as f32).clamp(0.0, 1.0)
    }
}

/// Aggregate parallel walker results into structured cognition.
async fn aggregate(
    pool: &PgPool,
    results: &[WalkerResult],
    walk_ms: f64,
    start: Instant,
) -> WalkOutput {
    let n = results.len();
    if n == 0 {
        return empty_output();
    }

    // Vote counting: which nodes did walkers visit?
    let mut node_votes: HashMap<i32, usize> = HashMap::new();
    let mut total_hops = 0;
    for r in results {
        for &node_id in &r.path {
            *node_votes.entry(node_id).or_insert(0) += 1;
        }
        total_hops += r.path.len();
    }

    // Classify nodes by agreement level (pure, complete partition — see helper).
    let (consensus, divergent, blind_spots) = classify_votes(&node_votes, n);

    // Domain distribution
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    for r in results {
        for d in &r.domains_visited {
            *domain_counts.entry(d.clone()).or_insert(0) += 1;
        }
    }
    let primary_domain = domain_counts
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k.clone())
        .unwrap_or_default();

    // Agreement & novelty scores
    let total_unique = node_votes.len().max(1);
    let agreement = consensus.len() as f32 / total_unique as f32;
    // Observability (PROTOCOL-self-selection §#2): domain-level agreement, robust to the
    // dispersed-seed structural zero in node-overlap `agreement`. Logged, not yet acted on.
    let domain_agree = domain_agreement(results);
    let total_surprises: usize = results.iter().map(|r| r.surprises).sum();
    let novelty = ((divergent.len() + total_surprises) as f32 / total_unique as f32).min(1.0);

    // Emotional resonance: how strongly the collective moved through well-worn
    // (high-weight) edges vs. weak/random ones (pure helper). Previously this
    // field was hard-coded 0.0 yet consumed by metacog.
    let resonance = compute_resonance(results);

    // Novel connections count
    let novel_connections = total_surprises;

    // Recommended action
    let action = recommend_action(agreement, novelty, &results);

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let hops_per_sec = if total_ms > 0.0 {
        total_hops as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };

    // Build expression seeds from consensus nodes
    let mut expression_seeds = Vec::new();
    for &nid in consensus.iter().take(5) {
        if let Ok(Some(node)) = db::get_node(pool, nid).await {
            expression_seeds.push(serde_json::json!({
                "id": node.id,
                "domain": node.domain,
                "votes": node_votes.get(&nid).unwrap_or(&0),
            }));
        }
    }

    tracing::info!(
        "[walker-perf] {} walkers × {} hops = {} total | \
         walk={:.0}ms total={:.0}ms | {:.0} hops/s | \
         agreement={:.0}% domain_agreement={:.0}% novelty={:.0}%",
        n,
        total_hops / n.max(1),
        total_hops,
        walk_ms,
        total_ms,
        hops_per_sec,
        agreement * 100.0,
        domain_agree * 100.0,
        novelty * 100.0,
    );

    WalkOutput {
        recommended_action: action,
        primary_domain,
        domain_distribution: domain_counts,
        agreement_score: agreement,
        novelty_score: novelty,
        emotional_resonance: resonance,
        search_query: None,
        expression_seeds,
        novel_connections,
        consensus_nodes: consensus,
        divergent_nodes: divergent,
        blind_spots,
        walker_count: n,
        total_hops,
        walk_ms,
        total_ms,
        hops_per_sec,
    }
}

fn recommend_action(agreement: f32, novelty: f32, results: &[WalkerResult]) -> String {
    let total_surprises: usize = results.iter().map(|r| r.surprises).sum();

    if agreement > 0.6 {
        "express".to_string()
    } else if novelty > 0.6 {
        "explore".to_string()
    } else if total_surprises >= 3 {
        "search".to_string()
    } else {
        "rest".to_string()
    }
}

/// Format walk output as context for LLM prompt injection.
pub fn format_walk_context(output: &WalkOutput) -> String {
    let mut lines = Vec::new();
    lines.push("=== RGW COGNITION ===".to_string());
    lines.push(format!("Action: {}", output.recommended_action));
    lines.push(format!("Domain: {}", output.primary_domain));
    lines.push(format!("Confidence: {:.0}%", output.agreement_score * 100.0));
    lines.push(format!("Novelty: {:.0}%", output.novelty_score * 100.0));

    if !output.domain_distribution.is_empty() {
        let mut sorted: Vec<_> = output.domain_distribution.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> = sorted.iter().take(5).map(|(d, c)| format!("{}({})", d, c)).collect();
        lines.push(format!("Landscape: {}", top.join(", ")));
    }

    if !output.expression_seeds.is_empty() {
        lines.push("\nRelevant:".to_string());
        for seed in output.expression_seeds.iter().take(5) {
            let domain = seed.get("domain").and_then(|v| v.as_str()).unwrap_or("?");
            lines.push(format!("  * [{}]", domain));
        }
    }

    if output.novel_connections > 0 {
        lines.push(format!("\nNovel connections: {}", output.novel_connections));
    }

    lines.push(format!(
        "\n[{} walkers, {} hops, {:.0}ms, {:.0} hops/s]",
        output.walker_count, output.total_hops, output.total_ms, output.hops_per_sec
    ));
    lines.push("=== END RGW ===".to_string());
    lines.join("\n")
}

/// Rich graph-to-text: fetch node content and describe relationships.
/// Translates raw node IDs into a narrative the LLM can understand.
pub async fn format_walk_narrative(output: &WalkOutput, pool: &PgPool) -> String {
    let mut lines = Vec::new();
    lines.push("=== JULIAN'S MIND RIGHT NOW ===".to_string());
    lines.push(format!(
        "State: {} (confidence {:.0}%, novelty {:.0}%)",
        output.recommended_action,
        output.agreement_score * 100.0,
        output.novelty_score * 100.0,
    ));

    // Fetch actual content for consensus nodes (what Julian is sure about)
    if !output.consensus_nodes.is_empty() {
        lines.push("\nStrong convictions:".to_string());
        for &nid in output.consensus_nodes.iter().take(3) {
            if let Ok(Some(node)) = db::get_node(pool, nid).await {
                // Fetch the document text (title)
                let title: Option<String> = sqlx::query_scalar(
                    "SELECT title FROM memory_vectors WHERE id = $1"
                )
                .bind(nid)
                .fetch_optional(pool)
                .await
                .unwrap_or(None);

                let label = title.unwrap_or_else(|| format!("[{}:{}]", node.domain, nid));
                lines.push(format!("  - {} (domain: {})", label, node.domain));
            }
        }
    }

    // Fetch content for divergent nodes (where perspectives disagree)
    if !output.divergent_nodes.is_empty() {
        lines.push("\nUnresolved tensions (walkers disagreed):".to_string());
        for &nid in output.divergent_nodes.iter().take(3) {
            if let Ok(Some(node)) = db::get_node(pool, nid).await {
                let title: Option<String> = sqlx::query_scalar(
                    "SELECT title FROM memory_vectors WHERE id = $1"
                )
                .bind(nid)
                .fetch_optional(pool)
                .await
                .unwrap_or(None);

                let label = title.unwrap_or_else(|| format!("[{}:{}]", node.domain, nid));
                lines.push(format!("  ? {} (domain: {})", label, node.domain));
            }
        }
    }

    // Describe walker paths (what trails of thought were followed)
    if !output.domain_distribution.is_empty() {
        let mut sorted: Vec<_> = output.domain_distribution.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        let narrative: Vec<String> = sorted.iter().take(3).map(|(d, c)| {
            format!("{} ({}x)", d, c)
        }).collect();
        lines.push(format!("\nThought trails: {}", narrative.join(" → ")));
    }

    if output.novel_connections > 0 {
        lines.push(format!(
            "\nSurprises: {} unexpected cross-domain connections found",
            output.novel_connections
        ));
    }

    lines.push(format!(
        "\n[{} perspectives, {:.0}ms thinking time]",
        output.walker_count, output.total_ms
    ));
    lines.push("=== END ===".to_string());
    lines.join("\n")
}

fn empty_output() -> WalkOutput {
    WalkOutput {
        recommended_action: "rest".to_string(),
        primary_domain: String::new(),
        domain_distribution: HashMap::new(),
        agreement_score: 0.0,
        novelty_score: 0.0,
        emotional_resonance: 0.0,
        search_query: None,
        expression_seeds: Vec::new(),
        novel_connections: 0,
        consensus_nodes: Vec::new(),
        divergent_nodes: Vec::new(),
        blind_spots: Vec::new(),
        walker_count: 0,
        total_hops: 0,
        walk_ms: 0.0,
        total_ms: 0.0,
        hops_per_sec: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wr(path: Vec<i32>, total_weight: f32) -> WalkerResult {
        WalkerResult {
            bias: WalkerBias::Curiosity,
            path,
            domains_visited: Vec::new(),
            edge_types_used: Vec::new(),
            total_weight,
            surprises: 0,
            dead_ends: 0,
            edges_traversed: Vec::new(),
        }
    }

    fn wr_dom(path: Vec<i32>, domains: &[&str]) -> WalkerResult {
        WalkerResult {
            bias: WalkerBias::Curiosity,
            path,
            domains_visited: domains.iter().map(|s| s.to_string()).collect(),
            edge_types_used: Vec::new(),
            total_weight: 1.0,
            surprises: 0,
            dead_ends: 0,
            edges_traversed: Vec::new(),
        }
    }

    #[test]
    fn domain_agreement_high_when_walkers_share_domain_despite_disjoint_paths() {
        // Disjoint node paths → the legacy node-overlap `agreement` reads ~0 here.
        // Domain-level agreement ignores node identity and must still see the convergence.
        let results = vec![
            wr_dom(vec![1, 2], &["markets", "markets", "risk"]),
            wr_dom(vec![3, 4], &["markets", "markets", "trading"]),
            wr_dom(vec![5, 6], &["markets", "markets"]),
        ];
        assert!(
            domain_agreement(&results) > 0.6,
            "all three walkers are dominated by 'markets' → high domain agreement"
        );
    }

    #[test]
    fn domain_agreement_low_when_walkers_diverge_across_domains() {
        let results = vec![
            wr_dom(vec![1, 2], &["markets"]),
            wr_dom(vec![3, 4], &["philosophy"]),
            wr_dom(vec![5, 6], &["music"]),
        ];
        assert!(
            domain_agreement(&results) < 0.5,
            "three walkers, three different dominant domains → low domain agreement"
        );
    }

    #[test]
    fn classify_votes_is_a_complete_partition() {
        let mut votes = HashMap::new();
        votes.insert(1, 4); // consensus
        votes.insert(2, 3); // consensus
        votes.insert(3, 2); // divergent (previously fell into the uncategorized gap)
        votes.insert(4, 1); // blind spot
        let (consensus, divergent, blind) = classify_votes(&votes, 4);

        assert!(consensus.contains(&1) && consensus.contains(&2));
        assert_eq!(divergent, vec![3], "2-of-4 votes must be divergent, not lost");
        assert_eq!(blind, vec![4]);
        // Every node is categorized exactly once — no gaps, no overlaps.
        assert_eq!(consensus.len() + divergent.len() + blind.len(), votes.len());
    }

    #[test]
    fn resonance_is_mean_edge_weight_per_hop() {
        // A: 2 hops, weight 1.0 → 0.5 ; B: 1 hop, weight 0.8 → 0.8 ; mean = 0.65
        let results = vec![wr(vec![1, 2, 3], 1.0), wr(vec![1, 2], 0.8)];
        let r = compute_resonance(&results);
        assert!((r - 0.65).abs() < 1e-6, "expected 0.65, got {r}");
    }

    #[test]
    fn resonance_ignores_zero_hop_walks_and_clamps() {
        assert_eq!(compute_resonance(&[]), 0.0);
        assert_eq!(compute_resonance(&[wr(vec![1], 5.0)]), 0.0); // 0 hops → skipped → empty
        assert_eq!(compute_resonance(&[wr(vec![1, 2], 9.0)]), 1.0); // clamped
    }

    #[test]
    fn background_walk_folds_deltas_not_absolutes() {
        // Regression for audit A1: the Diverger used to discard its walk's
        // self-model. Now it folds deltas back; verify counters add the delta
        // (not the absolute) and emotion blends rather than overwrites.
        let mut shared = SelfModel::new();
        shared.surprise_count = 5;
        let base = shared.clone(); // walk starts from a snapshot of shared
        let mut walked = base.clone();
        walked.surprise_count = 8; // walk found 3 new surprises
        walked.valence = 1.0; // walk "felt" strongly positive
        *walked.attention_patterns.entry("markets".into()).or_insert(0.0) += 2.0;
        walked.noticings.push(Noticing {
            kind: "surprise".into(),
            observation: "unexpected link".into(),
            domain: "markets".into(),
            significance: 0.5,
            valence: 0.1,
            timestamp: 0.0,
        });

        integrate_background_walk(&mut shared, &base, &walked);

        assert_eq!(shared.surprise_count, 8, "delta of 3 folded onto base 5");
        assert!(shared.valence > 0.0 && shared.valence < 1.0, "emotion blended, not replaced");
        assert_eq!(shared.attention_patterns.get("markets"), Some(&2.0), "attention delta merged");
        assert!(shared.noticings.iter().any(|n| n.observation == "unexpected link"));
    }
}
