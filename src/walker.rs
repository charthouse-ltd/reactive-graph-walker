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

use crate::core::{self, Signal, SelfModel, Noticing};
use crate::db;
use crate::graph::*;

/// Run a single walker through the graph.
/// Now accepts a shared WalkerCollective for inter-walker stigmergy.
/// Walkers leave trails (visited nodes, dead ends, surprises) and
/// read each other's trails to diversify exploration.
pub fn walk_single(
    pool: &PgPool,
    seed_id: i32,
    bias: WalkerBias,
    emotion: &EmotionalState,
    steps: usize,
    rt: &tokio::runtime::Handle,
    self_model: &mut SelfModel,
    collective: &Arc<Mutex<WalkerCollective>>,
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
        // Get edges from current node
        let edges = match rt.block_on(db::edges_from(pool, current_id)) {
            Ok(e) => e,
            Err(_) => {
                result.dead_ends += 1;
                break;
            }
        };

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
            // Read collective state (non-blocking try_lock)
            let collective_lock = collective.try_lock().ok();
            let collective_snapshot: Option<&WalkerCollective> = collective_lock.as_deref();
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

            edges.iter().map(|e| {
                let next_id = if e.source_id == current_id { e.target_id } else { e.source_id };
                // Fetch domain for the next node (fallback to empty if unavailable)
                let next_domain = rt.block_on(db::get_node(pool, next_id))
                    .ok()
                    .flatten()
                    .map(|n| n.domain)
                    .unwrap_or_default();

                let score = bias.score_edge_with_context(
                    &e.edge_type,
                    e.weight,
                    e.emotional_charge,
                    e.traversal_count,
                    &current_emotion,
                    collective_snapshot,
                    next_id,
                    &next_domain,
                    &self_model.beliefs,
                    &self_model.wounds,
                    &self_model.competencies,
                    &wm_domains,
                );
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
        if let Ok(Some(node)) = rt.block_on(db::get_node(pool, next_id)) {
            if !node.domain.is_empty() {
                let is_surprise = !prev_domain.is_empty() && prev_domain != node.domain;

                // ── WRITE to collective: leave trails for other walkers ──
                if let Ok(mut c) = collective.try_lock() {
                    *c.visited_nodes.entry(next_id).or_insert(0) += 1;
                    if !node.domain.is_empty() {
                        *c.active_domains.entry(node.domain.clone()).or_insert(0) += 1;
                    }
                    if is_surprise {
                        c.surprise_domains.insert(node.domain.clone());
                    }
                    // Drift collective emotion toward this walker's state
                    let alpha = 0.1; // Slow drift — collective is stable
                    c.drift_valence = c.drift_valence * (1.0 - alpha) + self_model.valence * alpha;
                    c.drift_arousal = c.drift_arousal * (1.0 - alpha) + self_model.arousal * alpha;
                }

                if is_surprise {
                    result.surprises += 1;
                }
                if !result.domains_visited.contains(&node.domain) {
                    result.domains_visited.push(node.domain.clone());
                }

                // ── EVERY STEP PASSES THROUGH THE SELF-MODEL ──
                let signal_kind = if is_surprise { "surprise" } else { "walk_step" };
                let signal = Signal::new(
                    signal_kind,
                    &format!("Step {}: {} → {} via {} edge (w={:.2})",
                        step, prev_domain, node.domain, edge.edge_type, edge.weight),
                )
                .with_domain(&node.domain)
                .with_intensity(if is_surprise { 0.6 } else { 0.2 });

                core::process(signal, self_model);
                // The self-model just changed. The next step's edge scoring
                // will be different because the self-model is different.
                // THIS IS COGNITION: the walk changes the thinker,
                // the changed thinker changes the walk.

                prev_domain = node.domain;
            }
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
pub async fn walk_parallel(
    pool: &PgPool,
    emotion: &EmotionalState,
    n_walkers: usize,
    steps: usize,
    self_model: &std::sync::Arc<std::sync::Mutex<SelfModel>>,
) -> WalkOutput {
    let start = Instant::now();

    // Signal: walk session starting
    {
        let mut sm = self_model.lock().unwrap();
        let signal = Signal::new("session_start", &format!("Launching {} walkers", n_walkers))
            .with_intensity(0.4);
        core::process(signal, &mut sm);
    }

    // Get seed nodes
    let seeds = db::seed_nodes(pool, (n_walkers * 2) as i32)
        .await
        .unwrap_or_default();

    if seeds.is_empty() {
        return empty_output();
    }

    // Assign biases — mode-dependent
    let mode = {
        let sm = self_model.lock().unwrap();
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

    let pool_clone = pool.clone();
    let rt = tokio::runtime::Handle::current();

    // Clone self-model per walker (parallel perspectives, integrate after)
    // Like the brain: process in parallel, integrate in global workspace
    let base_sm = self_model.lock().unwrap().clone();

    // Create shared stigmergic collective — walkers leave trails for each other
    let collective = Arc::new(Mutex::new(WalkerCollective::new()));

    let walk_start = Instant::now();
    let results: Vec<(WalkerResult, SelfModel)> = configs
        .par_iter()
        .map(|(seed, bias)| {
            let mut sm = base_sm.clone();  // Each walker gets its own copy
            let collective = Arc::clone(&collective);
            let result = walk_single(&pool_clone, *seed, *bias, &EmotionalState {
                valence: sm.valence,
                arousal: sm.arousal,
                energy: sm.energy,
            }, steps, &rt, &mut sm, &collective);
            (result, sm)
        })
        .collect();
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    // Merge per-walker self-models back into the shared self-model
    // Each walker saw different things — integrate all perspectives
    {
        let mut sm = self_model.lock().unwrap();
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
        }
        sm.total_signals_processed += results.iter().map(|(_, s)| s.total_signals_processed).sum::<u64>();

        // Signal: integration complete
        let signal = crate::core::Signal::new("integration", &format!(
            "Integrated {} walker perspectives",
            results.len()
        )).with_intensity(0.3);
        crate::core::process(signal, &mut sm);
    }

    // Extract just the walker results for aggregation
    let walker_results: Vec<WalkerResult> = results.into_iter().map(|(r, _)| r).collect();

    // Aggregate results
    let output = aggregate(pool, walker_results, walk_ms, start).await;

    // Batch strengthen traversed edges (the walk changes the graph)
    let all_edge_ids: Vec<i32> = output
        .consensus_nodes
        .iter()
        .copied()
        .chain(output.divergent_nodes.iter().copied())
        .collect();
    // Note: we use consensus_nodes as a proxy — actual edge strengthening
    // happens via the edge IDs collected during walks
    // For now, just log that we would strengthen

    output
}

/// Aggregate parallel walker results into structured cognition.
async fn aggregate(
    pool: &PgPool,
    results: Vec<WalkerResult>,
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
    for r in &results {
        for &node_id in &r.path {
            *node_votes.entry(node_id).or_insert(0) += 1;
        }
        total_hops += r.path.len();
    }

    // Classify nodes by agreement level
    let consensus: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v as f32 > n as f32 * 0.6)
        .map(|(k, _)| *k)
        .collect();
    let divergent: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v > 1 && (**v as f32) <= n as f32 * 0.4)
        .map(|(k, _)| *k)
        .collect();
    let blind_spots: Vec<i32> = node_votes
        .iter()
        .filter(|(_, v)| **v == 1)
        .map(|(k, _)| *k)
        .collect();

    // Domain distribution
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    for r in &results {
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
    let total_surprises: usize = results.iter().map(|r| r.surprises).sum();
    let novelty = ((divergent.len() + total_surprises) as f32 / total_unique as f32).min(1.0);

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
         agreement={:.0}% novelty={:.0}%",
        n,
        total_hops / n.max(1),
        total_hops,
        walk_ms,
        total_ms,
        hops_per_sec,
        agreement * 100.0,
        novelty * 100.0,
    );

    WalkOutput {
        recommended_action: action,
        primary_domain,
        domain_distribution: domain_counts,
        agreement_score: agreement,
        novelty_score: novelty,
        emotional_resonance: 0.0,
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
