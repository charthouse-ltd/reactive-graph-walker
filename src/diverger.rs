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

use rand::seq::SliceRandom;
use sqlx::PgPool;
use tokio::sync::{mpsc, RwLock};

use crate::db;
use crate::graph::*;

/// Activation energy accumulated at each node.
/// When energy crosses threshold → spontaneous walk fires.
/// The walk changes edges → cascading activation → more walks.
#[derive(Debug)]
struct NodeEnergy {
    energy: f32,
    last_fired: f64,   // unix timestamp of last spontaneous fire (refractory clock)
    last_update: f64,  // unix timestamp of last energy update (decay clock)
    fatigue: f32,      // rises on fire, decays with rest; raises this node's threshold
}

/// The reactor's firing threshold right now: the configured base, lowered by
/// arousal and raised by low energy (the same formula the reactor applies).
pub(crate) fn dynamic_threshold(base: f32, emo: &EmotionalState) -> f32 {
    base
        * (1.0 - emo.arousal * 0.3)  // High arousal lowers threshold
        * (1.0 + (1.0 - emo.energy) * 0.3)  // Low energy raises threshold
}

/// Read-only view of the reactor's node energies for the tension engine
/// (spec §8 `pulses_would_cross`). Shares the reactor's map; never mutates it.
#[derive(Clone)]
pub struct EnergyView {
    nodes: Arc<RwLock<HashMap<i32, NodeEnergy>>>,
    emotion: Arc<RwLock<EmotionalState>>,
    base_threshold: f32,
    energy_decay_rate: f32,
}

impl EnergyView {
    /// `(energy, threshold)` for `nid` as the reactor would see them at
    /// `now_secs`: energy decayed since the node's last update, threshold
    /// modulated by mood and raised by the node's decayed fatigue (what `live`
    /// enforces). A node the reactor has never touched is cold at the base.
    pub async fn headroom(&self, nid: i32, now_secs: f64) -> (f32, f32) {
        let base = dynamic_threshold(self.base_threshold, &*self.emotion.read().await);
        let nodes = self.nodes.read().await;
        match nodes.get(&nid) {
            Some(n) => {
                let dt = (now_secs - n.last_update).max(0.0) as f32;
                let energy = (n.energy - self.energy_decay_rate * dt).max(0.0);
                let fatigue = crate::tension::decay_fatigue(n.fatigue, dt as f64);
                (energy, crate::tension::effective_threshold(base, fatigue))
            }
            None => (0.0, base),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(base_threshold: f32, energy_decay_rate: f32) -> Self {
        // Neutral mood: arousal 0 and energy 1 leave the base threshold unmodulated.
        let emo = EmotionalState { arousal: 0.0, energy: 1.0, ..EmotionalState::default() };
        EnergyView {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            emotion: Arc::new(RwLock::new(emo)),
            base_threshold,
            energy_decay_rate,
        }
    }

    #[cfg(test)]
    async fn set_node(&self, nid: i32, node: NodeEnergy) {
        self.nodes.write().await.insert(nid, node);
    }

    #[cfg(test)]
    pub(crate) async fn set_energy(&self, nid: i32, energy: f32, now_secs: f64) {
        self.set_node(nid, NodeEnergy { energy, last_fired: 0.0, last_update: now_secs, fatigue: 0.0 }).await;
    }
}

/// The Diverger engine — a self-propagating reactive graph.
/// Reactor inbox size. Edge changes beyond this are dropped and counted in
/// `DivergerStats::edges_dropped` rather than queued without bound.
pub const EDGE_QUEUE_CAPACITY: usize = 4096;

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
    edge_tx: mpsc::Sender<Stimulus>,
    /// Edge changes dropped because the reactor queue was full (M6 backpressure).
    edges_dropped: Arc<AtomicU64>,

    /// Running state
    alive: Arc<AtomicBool>,

    /// Performance counters
    walks_fired: Arc<AtomicU64>,
    cascades_total: Arc<AtomicU64>,
    edges_changed: Arc<AtomicU64>,

    /// Julian's Python backend URL (for motor commands)
    julian_url: String,

    /// Threshold/decay the reactor runs with, for `energy_view()`.
    config: DivergerConfig,
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

/// Which stimulus started a cascade, how deep it is, and what it may still
/// spend. API callers always start at depth 0 with the full budget; feedback
/// hops (spec §6.2) carry `origin` forward and spend `HOP_COST` per hop.
#[derive(Debug, Clone)]
pub struct Stimulus {
    pub kind: StimulusKind,
    pub origin: crate::tension::Origin,
    pub depth: u32,
    pub budget: f32,
}

#[derive(Debug, Clone)]
pub enum StimulusKind {
    Edge(EdgeChange),
    Tension(crate::tension::TensionPulse),
}

impl Stimulus {
    /// An externally reported edge change: origin Edge, depth 0, full budget.
    pub fn external_edge(c: EdgeChange) -> Self {
        Stimulus {
            kind: StimulusKind::Edge(c),
            origin: crate::tension::Origin::Edge,
            depth: 0,
            budget: crate::tension::CASCADE_BUDGET,
        }
    }
}

/// What to do with a completed spontaneous walk's feedback hop, by mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Feedback {
    /// `off`: no feedback at all (today's single-hop behaviour).
    None,
    /// `observe`: a hop would have been enqueued; count it, enqueue nothing.
    WouldEnqueue,
    /// `observe`: a brake would have stopped the hop; count which one.
    WouldDrop(crate::tension::WouldHave),
    /// `brakes`/`live`: enqueue the next hop with this depth and budget.
    Enqueue(u32, f32),
    /// `brakes`/`live`: the cascade ends here (budget/depth, or pool empty).
    End { by_budget: bool },
}

/// Pure decision: `next` is `next_hop(...)` (None = budget or depth exhausted),
/// `pool_ok` whether the global pool can cover `HOP_COST`.
pub fn decide_feedback(mode: crate::tension::TensionMode, next: Option<(u32, f32)>, pool_ok: bool) -> Feedback {
    use crate::tension::{TensionMode, WouldHave};
    match mode {
        TensionMode::Off => Feedback::None,
        TensionMode::Observe => match next {
            None => Feedback::WouldDrop(WouldHave::Budget),
            Some(_) if !pool_ok => Feedback::WouldDrop(WouldHave::Pool),
            Some(_) => Feedback::WouldEnqueue,
        },
        TensionMode::Brakes | TensionMode::Live => match next {
            None => Feedback::End { by_budget: true },
            Some(_) if !pool_ok => Feedback::End { by_budget: false },
            Some((d, b)) => Feedback::Enqueue(d, b),
        },
    }
}

/// The next hop of a cascade, if it may continue: depth + 1 and the budget
/// after paying `HOP_COST` plus the energy this hop injects. `None` when the
/// budget would be exhausted or `max_depth` reached — so every cascade ends.
pub fn next_hop(s: &Stimulus, injected: f32, max_depth: u32) -> Option<(u32, f32)> {
    let depth = s.depth + 1;
    let budget = s.budget - crate::tension::HOP_COST - injected;
    if budget > 0.0 && depth < max_depth {
        Some((depth, budget))
    } else {
        None
    }
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
    pub edges_dropped: u64,
    pub emotional_state: EmotionalState,
    pub hottest_nodes: Vec<(i32, f32)>,  // top 10 by energy
}

impl Diverger {
    /// Create a new Diverger engine with shared self-model.
    pub fn new(pool: PgPool, self_model: std::sync::Arc<tokio::sync::Mutex<crate::core::SelfModel>>, julian_url: &str) -> Self {
        // Bounded: a burst of edge changes must apply backpressure (drop + count),
        // not grow the heap without limit while the reactor awaits the DB.
        let (edge_tx, edge_rx) = mpsc::channel(EDGE_QUEUE_CAPACITY);
        let sm = self_model.clone();

        let config = DivergerConfig::default();
        let diverger = Self {
            pool,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            emotion: Arc::new(RwLock::new(EmotionalState::default())),
            edge_tx,
            edges_dropped: Arc::new(AtomicU64::new(0)),
            alive: Arc::new(AtomicBool::new(false)),
            walks_fired: Arc::new(AtomicU64::new(0)),
            cascades_total: Arc::new(AtomicU64::new(0)),
            edges_changed: Arc::new(AtomicU64::new(0)),
            julian_url: julian_url.to_string(),
            config: config.clone(),
        };

        // Start the reactor (consumes edge_rx) with shared self-model
        diverger.start_reactor(edge_rx, config, sm);

        diverger
    }

    /// Start the reactive core. This is the ONLY "loop" — and it's an
    /// async channel receiver, not a timer. It only runs when edges change.
    fn start_reactor(
        &self,
        mut edge_rx: mpsc::Receiver<Stimulus>,
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
        let feedback_tx_outer = self.edge_tx.clone();
        let max_cascade_depth = config.max_cascade_depth;

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
            while let Some(stim) = edge_rx.recv().await {
                if matches!(stim.kind, StimulusKind::Edge(_)) {
                    edges_changed.fetch_add(1, Ordering::Relaxed);
                }

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

                // Read mode + emotion from the shared self-model in ONE lock.
                // This is what drives the threshold and the spontaneous walk.
                // Previously the Diverger read a separate `emotion` RwLock that
                // nothing ever wrote (set_emotion had no callers), so modulation
                // ran on constant defaults forever (audit A2). Now it tracks the
                // real mood. Energy is still propagated in Compliant mode (below);
                // only spontaneous firing is suppressed.
                let (is_autonomous, emo, goal_domain, goal_strength, active_audience_id, consecutive_repetitions) = {
                    let sm = self_model.lock().await;
                    (
                        sm.mode == crate::core::CognitiveMode::Autonomous,
                        EmotionalState {
                            valence: sm.valence,
                            arousal: sm.arousal,
                            energy: sm.energy,
                        },
                        sm.active_goal_domain.clone(),
                        sm.active_goal_strength,
                        sm.active_audience_id.clone(),
                        sm.consecutive_repetitions,
                    )
                };
                // Keep the stats cache live so GET /diverger reflects real mood.
                *emotion.write().await = emo.clone();

                // Compute dynamic threshold
                let threshold = dynamic_threshold(config.base_threshold, &emo);

                // Energy to inject: an edge change feeds both endpoints; a tension
                // pulse feeds its one target node with the pulse's own energy.
                let (node_ids, energy_delta): (Vec<i32>, f32) = match &stim.kind {
                    StimulusKind::Edge(change) => {
                        tracing::trace!(
                            "[diverger] edge stimulus #{} ({}) depth={} budget={:.2}",
                            change.edge_id, change.edge_type, stim.depth, stim.budget
                        );
                        (
                            vec![change.source_id, change.target_id],
                            change.delta.abs() * config.propagation_strength,
                        )
                    }
                    StimulusKind::Tension(p) => (vec![p.target_node], p.energy),
                };
                let enforce_brakes = crate::tension::mode().enforces_brakes();

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
                            fatigue: 0.0,
                        });

                        // Apply energy decay since this node was last updated.
                        // Decay is clocked off last_update, NOT last_fired: a node that
                        // has never fired has last_fired=0 (epoch) → dt astronomical →
                        // injected energy wiped before it can accumulate to threshold.
                        let dt = (now - node.last_update).max(0.0) as f32;
                        node.energy = (node.energy - config.energy_decay_rate * dt).max(0.0);
                        node.fatigue = crate::tension::decay_fatigue(node.fatigue, dt as f64);
                        node.last_update = now;

                        // Add new energy from the stimulus
                        node.energy += energy_delta;

                        // Threshold crossing. Fatigue raises the bar only when the
                        // tension mode enforces brakes; otherwise it is computed and
                        // the block it would have caused is counted (spec §7).
                        let fatigued = crate::tension::effective_threshold(threshold, node.fatigue);
                        let crosses_plain = node.energy >= threshold;
                        let crosses_fatigued = node.energy >= fatigued;
                        let crosses = if enforce_brakes { crosses_fatigued } else { crosses_plain };
                        if !enforce_brakes && crosses_plain && !crosses_fatigued {
                            crate::tension::would_have(crate::tension::WouldHave::Fatigue);
                        }
                        if crosses {
                            // Refractory period check
                            if now - node.last_fired >= config.refractory_period {
                                nodes_to_fire.push(nid);
                                node.energy = 0.0;  // Reset after firing
                                node.last_fired = now;
                                node.fatigue += crate::tension::FATIGUE_PER_FIRE;
                            }
                        }
                    }
                }

                // If we're repeating the same walk trajectory, inject a small
                // exploration pulse into fresh seed nodes to break attractors.
                if consecutive_repetitions >= 4 {
                    let pulse_nodes = crate::db::seed_nodes(&pool, 3).await.unwrap_or_default();
                    if !pulse_nodes.is_empty() {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64();
                        let mut nodes_lock = nodes.write().await;
                        for id in pulse_nodes {
                            let node = nodes_lock.entry(id).or_insert(NodeEnergy {
                                energy: 0.0,
                                last_fired: 0.0,
                                last_update: now,
                                fatigue: 0.0,
                            });
                            node.energy += 0.08;
                            node.last_update = now;
                        }
                    }
                    if let Ok(mut sm) = self_model.try_lock() {
                        sm.consecutive_repetitions = 0;
                    }
                }

                // Goal-directed prioritization: fire nodes in goal domain first.
                if let Some(goal) = goal_domain.as_ref() {
                    let mut scored: Vec<(i32, f32)> = Vec::with_capacity(nodes_to_fire.len());
                    for nid in nodes_to_fire {
                        let mut score = 1.0;
                        if let Ok(Some(node)) = db::get_node(&pool, nid).await
                            && node.domain == *goal {
                                score += goal_strength.clamp(0.0, 1.0) * 2.0;
                            }
                        scored.push((nid, score));
                    }
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    nodes_to_fire = scored.into_iter().map(|(nid, _)| nid).collect();
                }

                // Stuck-mode jitter: randomize firing order to improve exploration.
                if consecutive_repetitions >= 2 && nodes_to_fire.len() > 1 {
                    let mut rng = rand::rng();
                    nodes_to_fire.shuffle(&mut rng);
                }

                // In Compliant mode, suppress all spontaneous walks.
                // Energy still propagated above (the graph stays alive), but
                // no spontaneous behavior fires. The system only acts when
                // explicitly asked via API. (is_autonomous read once, above.)
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
                    let audience_id_for_walk = active_audience_id.clone();
                        let kd = knowledge_domains.clone();
                        let stim_meta = stim.clone();
                        let feedback_tx = feedback_tx_outer.clone();
                        tokio::spawn(async move {
                        // Preload the seed's neighborhood (radius 1) before walking:
                        // walk_single is cache-only with NO DB fallback, so without a
                        // preload it dead-ends at hop 1 and never reaches the motor.
                        let cache = std::sync::Arc::new(
                            crate::edge_cache::EdgeCache::new(pool.clone())
                        );
                        let _ = cache.preload_radius(&[node_id]).await;

                        // Fire the walk on a blocking thread (true parallelism).
                        // Capture the pre-walk snapshot (base) and the mutated
                        // self-model (walked) so the deltas can be folded back into
                        // the shared model — the spontaneous walk now actually
                        // changes the thinker instead of mutating a throwaway
                        // clone (audit A1).
                        let cache_walk = cache.clone();
                        let sm_blocking = sm_walk.clone();
                        let walk_out = tokio::task::spawn_blocking(move || {
                            let base = sm_blocking.blocking_lock().clone();
                            let mut walked = base.clone();
                            let collective = std::sync::Arc::new(std::sync::Mutex::new(
                                crate::graph::WalkerCollective::new()
                            ));
                            let result = crate::walker::walk_single(
                                &cache_walk, node_id,
                                WalkerBias::all()[node_id as usize % WalkerBias::all().len()],
                                &emo, steps, &mut walked, &collective, None, &kd,
                                None, None,
                            );
                            (result, base, walked)
                        })
                        .await;

                        if let Ok((result, base, walked)) = walk_out {
                            // Fold the spontaneous walk's deltas into the shared
                            // self-model (mood, noticings, attention, counters).
                            {
                                let mut shared = sm_walk.lock().await;
                                crate::walker::integrate_background_walk(&mut shared, &base, &walked);
                            }

                            // Hebbian: reinforce the edges this walk traversed.
                            if !result.edges_traversed.is_empty() {
                                cache.strengthen_edges(&result.edges_traversed, 0.02);
                            }

                            // Tension release after this spontaneous walk (spec §5.1):
                            // activity feeds activity, no clock.
                            let _ = crate::tension::spawn_after_walk(pool.clone(), sm_walk.clone());

                            // Feedback hop (spec §6.2): the last traversed edge re-enters
                            // as the next stimulus of this cascade, under the per-cascade
                            // budget, the depth limit and the global pool. Enforced in
                            // brakes/live; computed and counted in observe; absent in off.
                            {
                                use crate::tension::{engine, mode, would_have, HOP_COST};
                                let injected = 0.02 * result.edges_traversed.len() as f32;
                                let next = next_hop(&stim_meta, injected, max_cascade_depth);
                                let m = mode();
                                let n = result.path.len();
                                let can_hop = n >= 2 && !result.edges_traversed.is_empty();
                                let pool_ok = match engine() {
                                    Some(e) if next.is_some() && can_hop && m.enforces_brakes() => e.try_spend(HOP_COST).await,
                                    Some(e) if next.is_some() => e.can_spend(HOP_COST).await,
                                    _ => true,
                                };
                                // In observe nothing feeds back, so every observed cascade
                                // is exactly one hop; record that so avg_edge_cascade_hops
                                // reflects reality rather than the default.
                                if !m.enforces_brakes() && m != crate::tension::TensionMode::Off
                                    && stim_meta.origin == crate::tension::Origin::Edge
                                    && let Some(e) = engine()
                                {
                                    e.record_edge_cascade_hops(1).await;
                                }
                                match decide_feedback(m, next, pool_ok) {
                                    Feedback::None | Feedback::WouldEnqueue => {}
                                    Feedback::WouldDrop(w) => would_have(w),
                                    Feedback::Enqueue(depth, budget) => {
                                        if can_hop && let Some(&edge_id) = result.edges_traversed.last() {
                                            let hop = Stimulus {
                                                kind: StimulusKind::Edge(EdgeChange {
                                                    edge_id,
                                                    source_id: result.path[n - 2],
                                                    target_id: result.path[n - 1],
                                                    delta: 0.02,
                                                    edge_type: "feedback".into(),
                                                }),
                                                origin: stim_meta.origin,
                                                depth,
                                                budget,
                                            };
                                            if feedback_tx.try_send(hop).is_err()
                                                && let Some(e) = engine()
                                            {
                                                // Hop lost: the chain ends here. Count the real cause,
                                                // and give back the token spent on a hop that never ran.
                                                if m.enforces_brakes() {
                                                    e.refund(HOP_COST).await;
                                                }
                                                e.record_inbox_full().await;
                                                e.record_cascade_end(stim_meta.origin, false).await;
                                                if stim_meta.origin == crate::tension::Origin::Edge {
                                                    e.record_edge_cascade_hops(stim_meta.depth + 1).await;
                                                }
                                            }
                                        } else if let Some(e) = engine() {
                                            // Too short to hop (nothing was spent): the chain ends naturally.
                                            e.record_cascade_end(stim_meta.origin, false).await;
                                            if stim_meta.origin == crate::tension::Origin::Edge {
                                                e.record_edge_cascade_hops(stim_meta.depth + 1).await;
                                            }
                                        }
                                    }
                                    Feedback::End { by_budget } => {
                                        if let Some(e) = engine() {
                                            if !by_budget && next.is_some() {
                                                // Ended because the pool could not pay for the hop.
                                                e.record_hop_pool_drop().await;
                                            }
                                            e.record_cascade_end(stim_meta.origin, by_budget).await;
                                            if stim_meta.origin == crate::tension::Origin::Edge {
                                                e.record_edge_cascade_hops(stim_meta.depth + 1).await;
                                            }
                                        }
                                    }
                                }
                            }

                            let hops = result.path.len();
                            let surprises = result.surprises;
                            let domains = result.domains_visited.clone();
                            crate::metrics::record_cascade_depth(hops as u32);

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
                                        audience_id: audience_id_for_walk.clone(),
                                    };
                                    crate::motor::gate().dispatch("diverger", &julian_url, cmd).await;
                                }
                            }
                        }

                        // Persist edge strengthening accumulated during the walk.
                        if let Err(e) = cache.flush().await {
                            tracing::warn!("[diverger] edge flush failed (deltas re-queued): {}", e);
                        }

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
        self.notify_stimulus(Stimulus::external_edge(change));
    }

    /// Clone of the inbox sender, for the tension engine to emit pulses into.
    pub fn stimulus_sender(&self) -> mpsc::Sender<Stimulus> {
        self.edge_tx.clone()
    }

    /// Read-only view of node energies for the tension engine's telemetry.
    pub fn energy_view(&self) -> EnergyView {
        EnergyView {
            nodes: self.nodes.clone(),
            emotion: self.emotion.clone(),
            base_threshold: self.config.base_threshold,
            energy_decay_rate: self.config.energy_decay_rate,
        }
    }

    /// Queue any stimulus (edge change, tension pulse, or feedback hop).
    pub fn notify_stimulus(&self, s: Stimulus) {
        if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) = self.edge_tx.try_send(s) {
            let n = self.edges_dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 1 || n.is_multiple_of(1000) {
                tracing::warn!("[diverger] edge queue full ({} capacity); dropped {} changes so far", EDGE_QUEUE_CAPACITY, n);
            }
        }
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
            edges_dropped: self.edges_dropped.load(Ordering::Relaxed),
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
                fatigue: 0.0,
            });
        }
        tracing::info!("[diverger] Seeded {} nodes with {:.2} energy", nodes.len(), initial_energy);
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn energy_view_headroom_decays_energy_and_raises_threshold_by_fatigue() {
        let view = super::EnergyView::for_test(0.5, 0.01);
        view.set_node(1, super::NodeEnergy { energy: 0.3, last_fired: 0.0, last_update: 0.0, fatigue: 0.5 }).await;
        // 10 s later: energy 0.3 - 0.1 = 0.2; fatigue 0.5 - 0.1 = 0.4 -> threshold 0.5 * 1.4 = 0.7
        let (e, th) = view.headroom(1, 10.0).await;
        assert!((e - 0.2).abs() < 1e-5, "energy {e}");
        assert!((th - 0.7).abs() < 1e-5, "threshold {th}");
        // a node the reactor has never seen is cold at the base threshold
        let (e, th) = view.headroom(99, 10.0).await;
        assert_eq!(e, 0.0);
        assert!((th - 0.5).abs() < 1e-5, "threshold {th}");
    }

    use super::*;

    #[test]
    fn feedback_decision_matches_mode() {
        use crate::tension::TensionMode::*;
        assert!(matches!(decide_feedback(Off, Some((1, 0.7)), true), Feedback::None));
        assert!(matches!(decide_feedback(Observe, Some((1, 0.7)), true), Feedback::WouldEnqueue));
        assert!(matches!(decide_feedback(Observe, Some((1, 0.7)), false), Feedback::WouldDrop(crate::tension::WouldHave::Pool)));
        assert!(matches!(decide_feedback(Observe, None, true), Feedback::WouldDrop(crate::tension::WouldHave::Budget)));
        assert!(matches!(decide_feedback(Brakes, Some((1, 0.7)), true), Feedback::Enqueue(1, _)));
        assert!(matches!(decide_feedback(Live, Some((1, 0.7)), false), Feedback::End { by_budget: false }));
        assert!(matches!(decide_feedback(Live, None, true), Feedback::End { by_budget: true }));
    }

    #[test]
    fn next_hop_ends_by_budget_then_by_depth() {
        let s = Stimulus::external_edge(EdgeChange {
            edge_id: 1,
            source_id: 1,
            target_id: 2,
            delta: 0.1,
            edge_type: "t".into(),
        });
        let (d1, b1) = next_hop(&s, 0.1, 3).unwrap();
        assert_eq!(d1, 1);
        assert!((b1 - 0.7).abs() < 1e-6);
        let s2 = Stimulus { depth: d1, budget: b1, ..s.clone() };
        let (d2, b2) = next_hop(&s2, 0.1, 3).unwrap();
        assert_eq!(d2, 2);
        assert!((b2 - 0.4).abs() < 1e-6);
        let s3 = Stimulus { depth: d2, budget: b2, ..s.clone() };
        assert!(next_hop(&s3, 0.1, 3).is_none(), "depth 3 reached");
        let s4 = Stimulus { depth: 0, budget: 0.25, ..s.clone() };
        assert!(next_hop(&s4, 0.1, 10).is_none(), "budget exhausted first");
    }
}
