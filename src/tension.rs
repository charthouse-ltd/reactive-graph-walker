//! Tension-driven background activity — pure core.
//!
//! Unresolved predictions and open questions release bounded energy pulses
//! into the diverger after every walk. This file holds the decision functions
//! (mode, strength, decay, pool, fatigue, hashing) with no I/O. See
//! `docs/superpowers/specs/2026-11-04-tension-driven-activity-design.md`.

// ── Constants (spec §12). Not environment variables: the only env surface is
// RGW_TENSION_MODE. Tuning happens here, with a test. ────────────────────────
pub const PREDICTION_SATURATION_SECS: f64 = 3600.0;
pub const PREDICTION_BASE_ENERGY: f32 = 0.15;
pub const PREDICTION_HALF_LIFE_SECS: f64 = 86400.0;
pub const PREDICTION_MAX_ACTIVE: usize = 10;
pub const QUESTION_BASE_ENERGY: f32 = 0.20;
pub const QUESTION_DECAY: f32 = 0.85;
pub const QUESTION_MAX_ATTEMPTS: u32 = 12;
pub const MIN_PULSE_ENERGY: f32 = 0.02;
pub const K_PREDICTION_TARGETS: i64 = 3;
pub const K_QUESTION_TARGETS: i32 = 3;
pub const ITEM_COOLDOWN_SECS: f64 = 120.0;
pub const FATIGUE_PER_FIRE: f32 = 0.5;
pub const FATIGUE_DECAY_PER_SEC: f32 = 0.01;
pub const CASCADE_BUDGET: f32 = 1.0;
pub const HOP_COST: f32 = 0.2;
pub const POOL_CAPACITY: f32 = 20.0;
pub const POOL_REFILL_PER_SEC: f32 = 0.05;
pub const LIVE_JUDGEMENT_HOURS: u64 = 72;
pub const ABANDONED_HASHES_MAX: usize = 4096;
pub const DEFAULT_EDGE_CASCADE_HOPS: f32 = 3.0;

// ── Mode ─────────────────────────────────────────────────────────────────────

/// `RGW_TENSION_MODE`. Default and unknown values are `Observe`: compute and
/// log everything, change nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensionMode {
    Off,
    Observe,
    Brakes,
    Live,
}

impl TensionMode {
    pub fn from_env_value(v: Option<&str>) -> TensionMode {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("off") => TensionMode::Off,
            Some("observe") => TensionMode::Observe,
            Some("brakes") => TensionMode::Brakes,
            Some("live") => TensionMode::Live,
            None | Some("") => TensionMode::Observe,
            Some(other) => {
                tracing::warn!("[tension] unknown RGW_TENSION_MODE={:?}; using observe", other);
                TensionMode::Observe
            }
        }
    }

    /// Only `live` puts pulses into the diverger inbox.
    pub fn emits(self) -> bool {
        matches!(self, TensionMode::Live)
    }

    /// `brakes` and `live` enforce fatigue, cascade budget and the pool.
    pub fn enforces_brakes(self) -> bool {
        matches!(self, TensionMode::Brakes | TensionMode::Live)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TensionMode::Off => "off",
            TensionMode::Observe => "observe",
            TensionMode::Brakes => "brakes",
            TensionMode::Live => "live",
        }
    }
}

// ── Hashing ──────────────────────────────────────────────────────────────────

/// FNV-1a 64 over UTF-8 bytes. Stable across Rust versions (unlike
/// `DefaultHasher`), so ids in logs and the persisted abandoned set agree.
pub fn fnv1a64(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Log/stats id for a question; never carries the question text.
pub fn question_id(text: &str) -> String {
    format!("question:{:016x}", fnv1a64(text))
}

// ── Strength / decay ─────────────────────────────────────────────────────────

/// Pulse strength of an unresolved prediction of age `age_secs`: ramps to
/// `PREDICTION_BASE_ENERGY` over `PREDICTION_SATURATION_SECS`, then halves
/// once per full `PREDICTION_HALF_LIFE_SECS` so stale predictions cannot starve
/// everything else.
pub fn prediction_strength(age_secs: f64) -> f32 {
    let age = age_secs.max(0.0);
    let ramp = (age / PREDICTION_SATURATION_SECS).min(1.0) as f32;
    let half_lives = (age / PREDICTION_HALF_LIFE_SECS).floor() as i32;
    ramp * PREDICTION_BASE_ENERGY * 0.5f32.powi(half_lives)
}

/// Threshold a node must reach to fire, raised by its fatigue.
pub fn effective_threshold(base: f32, fatigue: f32) -> f32 {
    base * (1.0 + fatigue.max(0.0))
}

/// Fatigue after `dt_secs` of rest.
pub fn decay_fatigue(fatigue: f32, dt_secs: f64) -> f32 {
    (fatigue - FATIGUE_DECAY_PER_SEC * dt_secs.max(0.0) as f32).max(0.0)
}

// ── Global pool (token bucket) ───────────────────────────────────────────────

/// Shared budget for all self-generated work. The direct CPU bound.
#[derive(Debug, Clone)]
pub struct TokenPool {
    pub capacity: f32,
    pub tokens: f32,
    pub refill_per_sec: f32,
    last_refill: f64,
}

impl TokenPool {
    /// Starts full.
    pub fn new(now_secs: f64) -> Self {
        TokenPool {
            capacity: POOL_CAPACITY,
            tokens: POOL_CAPACITY,
            refill_per_sec: POOL_REFILL_PER_SEC,
            last_refill: now_secs,
        }
    }

    pub fn refill(&mut self, now_secs: f64) {
        let dt = (now_secs - self.last_refill).max(0.0);
        self.tokens = (self.tokens + self.refill_per_sec * dt as f32).min(self.capacity);
        self.last_refill = now_secs;
    }

    /// Refill, then spend `amount` if available. Never goes negative.
    pub fn try_spend(&mut self, amount: f32, now_secs: f64) -> bool {
        self.refill(now_secs);
        if amount <= self.tokens {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }

    /// Give back tokens spent on work that did not happen (a hop whose inbox
    /// send failed). Capped at capacity, so a refund can never mint tokens.
    pub fn refund(&mut self, amount: f32) {
        self.tokens = (self.tokens + amount.max(0.0)).min(self.capacity);
    }
}

// ── Open-question state ──────────────────────────────────────────────────────

/// Per-question release state. In-memory; the abandoned flag is mirrored into
/// `SelfModel::abandoned_question_hashes` so it survives restarts.
#[derive(Debug, Clone)]
pub struct QuestionState {
    /// The question text, kept in memory only so the embedding can be computed
    /// once. Never serialised, never logged (logs carry the hash id).
    pub text: String,
    pub strength: f32,
    pub attempts: u32,
    pub abandoned: bool,
    pub unmapped: bool,
    pub targets: Vec<i32>,
    pub last_release: f64,
}

impl Default for QuestionState {
    fn default() -> Self {
        Self::new()
    }
}

impl QuestionState {
    pub fn new() -> Self {
        QuestionState {
            text: String::new(),
            strength: QUESTION_BASE_ENERGY,
            attempts: 0,
            abandoned: false,
            unmapped: false,
            targets: Vec::new(),
            last_release: 0.0,
        }
    }

    /// Decay after a release; abandon by attempt cap or strength floor.
    pub fn after_release(&mut self, now_secs: f64) {
        self.strength *= QUESTION_DECAY;
        self.attempts += 1;
        self.last_release = now_secs;
        if self.attempts >= QUESTION_MAX_ATTEMPTS || self.strength < MIN_PULSE_ENERGY {
            self.abandoned = true;
        }
    }
}


// ── Process-wide engine ──────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

/// Admits one release at a time; a release requested while one is running is
/// coalesced away (the running one already sees the latest self-model). The
/// admission is a token that clears the flag on drop, so a release task that
/// panics (or is cancelled) cannot leave the engine wedged with every later
/// release silently skipped.
#[derive(Default)]
pub struct ReleaseGuard(Arc<std::sync::atomic::AtomicBool>);

pub struct ReleaseToken(Arc<std::sync::atomic::AtomicBool>);

impl ReleaseGuard {
    pub fn try_begin(&self) -> Option<ReleaseToken> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            .then(|| ReleaseToken(self.0.clone()))
    }
}

impl Drop for ReleaseToken {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

static ENGINE: OnceLock<Option<Arc<TensionEngine>>> = OnceLock::new();

/// One engine per process (same pattern as `motor::gate()`). `Off` installs
/// nothing, so every hook is a no-op and the diverger behaves as before.
pub struct TensionEngine {
    pub mode: TensionMode,
    inner: tokio::sync::Mutex<EngineInner>,
    inbox: OnceLock<mpsc::Sender<crate::diverger::Stimulus>>,
    /// Read-only view of the reactor's node energies, so non-emitting modes can
    /// count pulses that would actually cross a threshold, not just be emitted.
    energy_view: OnceLock<crate::diverger::EnergyView>,
    // Brake "would have" counters are atomics so the reactor can record them
    // from synchronous code without touching the async lock.
    would_fatigue: AtomicU64,
    would_budget: AtomicU64,
    would_pool: AtomicU64,
    releasing: ReleaseGuard,
}

/// Install the engine once. Returns `None` for `Off`. A second call returns the
/// already-installed engine regardless of the mode passed.
pub fn install(mode: TensionMode) -> Option<Arc<TensionEngine>> {
    ENGINE
        .get_or_init(|| {
            tracing::info!("[tension] mode={}", mode.as_str());
            if mode == TensionMode::Off {
                None
            } else {
                Some(Arc::new(TensionEngine {
                    mode,
                    inner: tokio::sync::Mutex::new(EngineInner::new(mode, crate::core::now())),
                    inbox: OnceLock::new(),
                    energy_view: OnceLock::new(),
                    would_fatigue: AtomicU64::new(0),
                    would_budget: AtomicU64::new(0),
                    would_pool: AtomicU64::new(0),
                    releasing: ReleaseGuard::default(),
                }))
            }
        })
        .clone()
}

pub fn engine() -> Option<Arc<TensionEngine>> {
    ENGINE.get().cloned().flatten()
}

/// The active tension mode. `Off` until an engine is installed.
pub fn mode() -> TensionMode {
    engine().map(|e| e.mode).unwrap_or(TensionMode::Off)
}

/// Record what a brake would have done in a non-enforcing mode. Synchronous
/// and lock-free so the reactor can call it anywhere.
pub fn would_have(what: WouldHave) {
    if let Some(e) = engine() {
        let c = match what {
            WouldHave::Fatigue => &e.would_fatigue,
            WouldHave::Budget => &e.would_budget,
            WouldHave::Pool => &e.would_pool,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// Release tension after a walk on a detached task, so the walk's caller (an
/// HTTP response, or the reactor's walk task) never waits for embedding or DB
/// lookups. Returns the task handle, or `None` when no engine is installed.
pub fn spawn_after_walk(
    pool: sqlx::PgPool,
    sm: Arc<tokio::sync::Mutex<crate::core::SelfModel>>,
) -> Option<tokio::task::JoinHandle<usize>> {
    let e = engine()?;
    // A release is already running and sees the same state: coalesce.
    let token = e.releasing.try_begin()?;
    Some(tokio::spawn(async move {
        let _release = token; // cleared on drop, panic or cancellation included
        e.after_walk(&pool, &sm).await
    }))
}

/// Release tension after a walk, inline. No engine → 0.
pub async fn after_walk(
    pool: &sqlx::PgPool,
    sm: &Arc<tokio::sync::Mutex<crate::core::SelfModel>>,
) -> usize {
    match engine() {
        Some(e) => e.after_walk(pool, sm).await,
        None => 0,
    }
}

impl TensionEngine {
    /// Give the engine the diverger's inbox so `live` can emit.
    pub fn attach_inbox(&self, tx: mpsc::Sender<crate::diverger::Stimulus>) {
        let _ = self.inbox.set(tx);
    }

    /// Give the engine a view of the reactor's node energies (observe/brakes
    /// telemetry: `pulses_would_cross`).
    pub fn attach_energy_view(&self, view: crate::diverger::EnergyView) {
        let _ = self.energy_view.set(view);
    }

    /// The whole release cycle: sample the resolution metric, plan, resolve
    /// targets, charge the pool, emit (live only). Returns pulses emitted.
    pub async fn after_walk(
        &self,
        pool: &sqlx::PgPool,
        sm: &Arc<tokio::sync::Mutex<crate::core::SelfModel>>,
    ) -> usize {
        let now = crate::core::now();
        let pending = {
            let mut sm = sm.lock().await;
            let mut inner = self.inner.lock().await;
            record_resolution_sample(&mut inner, sm.predictions_made_total, sm.predictions_resolved_total, now);
            plan_release(&mut inner, &mut sm, now)
        };
        if pending.is_empty() {
            return 0;
        }
        let charged = {
            let mut inner = self.inner.lock().await;
            let pulses = resolve_targets(pool, &mut inner, pending).await;
            charge(&mut inner, pulses, now)
        }; // `inner` released before `sm` is taken: the only lock order is sm → inner
        {
            let mut sm = sm.lock().await;
            let mut inner = self.inner.lock().await;
            settle(&mut inner, &mut sm, &charged.accepted_keys, now);
        }
        // Would any would-fire pulse have lifted its target over the threshold?
        // Needs the reactor's energies (its own lock), so it runs with no engine
        // lock held and the count is folded in below.
        let would_cross = match self.energy_view.get() {
            Some(view) if !charged.would_fire.is_empty() => count_would_cross(view, &charged.would_fire, now).await,
            _ => 0,
        };
        let mut inner = self.inner.lock().await;
        inner.pulses_would_cross += would_cross;
        let mut sent = 0;
        if let Some(tx) = self.inbox.get() {
            for p in charged.emit {
                let stim = crate::diverger::Stimulus {
                    kind: crate::diverger::StimulusKind::Tension(p),
                    origin: Origin::Tension,
                    depth: 0,
                    budget: CASCADE_BUDGET,
                };
                match tx.try_send(stim) {
                    Ok(()) => sent += 1,
                    Err(mpsc::error::TrySendError::Full(_)) => inner.dropped_inbox_full += 1,
                    Err(mpsc::error::TrySendError::Closed(_)) => {}
                }
            }
        }
        sent
    }

    pub async fn stats(&self, sm: &crate::core::SelfModel) -> TensionStats {
        let inner = self.inner.lock().await;
        let mut s = stats(&inner, sm, crate::core::now());
        s.brakes_would_have.fatigue_blocks += self.would_fatigue.load(Ordering::Relaxed);
        s.brakes_would_have.budget_cuts += self.would_budget.load(Ordering::Relaxed);
        s.brakes_would_have.pool_drops += self.would_pool.load(Ordering::Relaxed);
        s
    }

    pub async fn record_edge_cascade_hops(&self, hops: u32) {
        record_edge_cascade_hops(&mut *self.inner.lock().await, hops);
    }

    pub async fn record_cascade_end(&self, origin: Origin, by_budget: bool) {
        record_cascade_end(&mut *self.inner.lock().await, origin, by_budget);
    }

    pub async fn record_hop_pool_drop(&self) {
        record_hop_pool_drop(&mut *self.inner.lock().await, crate::core::now());
    }

    pub async fn record_inbox_full(&self) {
        record_inbox_full(&mut *self.inner.lock().await);
    }

    /// Spend `amount` from the global pool (feedback hops in enforcing modes).
    pub async fn try_spend(&self, amount: f32) -> bool {
        self.inner.lock().await.pool.try_spend(amount, crate::core::now())
    }

    /// Return tokens for work that did not happen (see `TokenPool::refund`).
    pub async fn refund(&self, amount: f32) {
        self.inner.lock().await.pool.refund(amount);
    }

    /// Would `amount` be affordable right now? Refills but does not spend
    /// (observe mode counts what the pool would have done).
    pub async fn can_spend(&self, amount: f32) -> bool {
        let mut inner = self.inner.lock().await;
        inner.pool.refill(crate::core::now());
        amount <= inner.pool.tokens
    }
}

// ── Engine types ─────────────────────────────────────────────────────────────

use std::collections::{HashMap, VecDeque};

/// Where a pulse comes from. Questions are identified by hash only.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensionSource {
    Prediction { domain: String },
    Question { id: String },
}

impl std::fmt::Display for TensionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensionSource::Prediction { domain } => write!(f, "prediction:{domain}"),
            TensionSource::Question { id } => write!(f, "{id}"),
        }
    }
}

/// A pulse ready to enter the diverger: energy aimed at one node.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensionPulse {
    pub source: TensionSource,
    pub target_node: i32,
    pub energy: f32,
}

/// A planned release before its targets are known. `key` is the domain for a
/// prediction or the 16-hex hash for a question (state lookup).
#[derive(Debug, Clone)]
pub struct PendingPulse {
    pub source: TensionSource,
    pub key: String,
    pub energy: f32,
}

/// Which stimulus started a cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Edge,
    Tension,
}

/// What a brake would have done in a mode that does not enforce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WouldHave {
    Fatigue,
    Budget,
    Pool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RecentEvent {
    pub t: u64,
    pub source: TensionSource,
    pub target: i32,
    pub energy: f32,
    pub fired: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CascadeStats {
    pub tension_origin: u64,
    pub edge_origin: u64,
    pub ended_by_budget: u64,
    pub ended_naturally: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BrakesWouldHave {
    pub fatigue_blocks: u64,
    pub budget_cuts: u64,
    pub pool_drops: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PoolStats {
    pub tokens: f32,
    pub capacity: f32,
    pub empty_fraction_1h: f32,
}

/// The `tension` block of `GET /diverger` (spec §8).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TensionStats {
    pub mode: String,
    pub open_predictions: u64,
    pub open_questions: u64,
    pub abandoned_questions: u64,
    pub unmapped: u64,
    pub pulses_would_fire: u64,
    /// Subset of `pulses_would_fire` whose target's current energy plus the
    /// pulse would have crossed its (fatigued) threshold. Observe-mode proxy
    /// for the tension-origin cascades `live` would actually start.
    pub pulses_would_cross: u64,
    pub pulses_fired: u64,
    pub dropped_pool_empty: u64,
    pub dropped_item_cooldown: u64,
    pub dropped_inbox_full: u64,
    pub dropped_compliant: u64,
    pub cascades: CascadeStats,
    pub brakes_would_have: BrakesWouldHave,
    pub pool: PoolStats,
    pub predictions_made_total: u64,
    pub predictions_resolved_total: u64,
    pub resolution_rate_1h: f32,
    /// How long `live` must hold before the stay-in-live criteria are judged (spec §9).
    pub live_judgement_hours: u64,
    pub recent: Vec<RecentEvent>,
}

const WINDOW_SECS: f64 = 3600.0;
const HOP_SAMPLES_MAX: usize = 100;
const RECENT_MAX: usize = 50;

/// All engine state. Fields are public so tests and the stats reader can see
/// them; mutation goes through the functions below.
#[derive(Debug)]
pub struct EngineInner {
    pub mode: TensionMode,
    pub pool: TokenPool,
    pub questions: HashMap<u64, QuestionState>,
    /// domain → (cached targets, last release time)
    pub prediction_targets: HashMap<String, (Vec<i32>, f64)>,
    pub avg_edge_cascade_hops: f32,
    pub hop_samples: VecDeque<u32>,
    pub pool_samples: VecDeque<(f64, bool)>,
    pub resolution_samples: VecDeque<(f64, u64, u64)>,
    pub recent: VecDeque<RecentEvent>,
    pub pulses_would_fire: u64,
    pub pulses_would_cross: u64,
    pub pulses_fired: u64,
    pub dropped_pool_empty: u64,
    pub dropped_item_cooldown: u64,
    pub dropped_inbox_full: u64,
    pub dropped_compliant: u64,
    pub unmapped: u64,
    pub cascades_tension_origin: u64,
    pub cascades_edge_origin: u64,
    pub cascades_ended_by_budget: u64,
    pub cascades_ended_naturally: u64,
    pub would_fatigue_blocks: u64,
    pub would_budget_cuts: u64,
    pub would_pool_drops: u64,
}

impl EngineInner {
    pub fn new(mode: TensionMode, now_secs: f64) -> Self {
        EngineInner {
            mode,
            pool: TokenPool::new(now_secs),
            questions: HashMap::new(),
            prediction_targets: HashMap::new(),
            avg_edge_cascade_hops: DEFAULT_EDGE_CASCADE_HOPS,
            hop_samples: VecDeque::new(),
            pool_samples: VecDeque::new(),
            resolution_samples: VecDeque::new(),
            recent: VecDeque::new(),
            pulses_would_fire: 0,
            pulses_would_cross: 0,
            pulses_fired: 0,
            dropped_pool_empty: 0,
            dropped_item_cooldown: 0,
            dropped_inbox_full: 0,
            dropped_compliant: 0,
            unmapped: 0,
            cascades_tension_origin: 0,
            cascades_edge_origin: 0,
            cascades_ended_by_budget: 0,
            cascades_ended_naturally: 0,
            would_fatigue_blocks: 0,
            would_budget_cuts: 0,
            would_pool_drops: 0,
        }
    }
}

fn trim_window<T>(q: &mut VecDeque<(f64, T)>, now: f64) {
    while let Some((t, _)) = q.front() {
        if now - *t > WINDOW_SECS {
            q.pop_front();
        } else {
            break;
        }
    }
}

// ── Release planning (pure) ──────────────────────────────────────────────────

/// Decide which tension items release energy now. Pure: no I/O. Applies the
/// Compliant freeze, per-item cooldown, the prediction cap (freshest first)
/// and half-life, question decay and abandonment (mirrored into `sm`).
pub fn plan_release(inner: &mut EngineInner, sm: &mut crate::core::SelfModel, now_secs: f64) -> Vec<PendingPulse> {
    let mut out = Vec::new();
    // Side tables track only live items, so they cannot grow past the
    // self-model's own caps (10 open questions; one entry per predicted domain).
    {
        let open: std::collections::HashSet<u64> = sm.open_questions.iter().map(|q| fnv1a64(q)).collect();
        inner.questions.retain(|h, _| open.contains(h));
        inner.prediction_targets.retain(|d, _| sm.predictions.contains_key(d));
    }
    if sm.mode == crate::core::CognitiveMode::Compliant {
        inner.dropped_compliant += 1;
        return out;
    }

    // Predictions: freshest first, at most PREDICTION_MAX_ACTIVE.
    let mut preds: Vec<(&String, f64)> = sm
        .predictions
        .iter()
        .map(|(d, p)| (d, p.predicted_at))
        .collect();
    preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (domain, predicted_at) in preds.into_iter().take(PREDICTION_MAX_ACTIVE) {
        let entry = inner
            .prediction_targets
            .entry(domain.clone())
            .or_insert((Vec::new(), f64::NEG_INFINITY));
        if now_secs - entry.1 < ITEM_COOLDOWN_SECS {
            inner.dropped_item_cooldown += 1;
            continue;
        }
        let energy = prediction_strength(now_secs - predicted_at);
        if energy < MIN_PULSE_ENERGY {
            continue;
        }
        entry.1 = now_secs; // reserve: no second release of this item within the cooldown
        out.push(PendingPulse {
            source: TensionSource::Prediction { domain: domain.clone() },
            key: domain.clone(),
            energy,
        });
    }

    // Open questions: decaying strength, attempt cap, persisted abandonment.
    let questions: Vec<String> = sm.open_questions.clone();
    for q in questions {
        let hash = fnv1a64(&q);
        if sm.is_question_abandoned(hash) {
            continue;
        }
        let state = inner.questions.entry(hash).or_default();
        if state.text.is_empty() {
            state.text = q.clone();
        }
        if state.abandoned {
            sm.note_question_abandoned(hash);
            continue;
        }
        let released_before = state.attempts > 0 || state.unmapped;
        if released_before && now_secs - state.last_release < ITEM_COOLDOWN_SECS {
            inner.dropped_item_cooldown += 1;
            continue;
        }
        state.unmapped = false; // cooldown passed: allow a retry of the mapping
        let energy = state.strength;
        if energy < MIN_PULSE_ENERGY {
            state.abandoned = true;
            sm.note_question_abandoned(hash);
            continue;
        }
        state.last_release = now_secs; // reserve (decay/attempts happen in settle)
        out.push(PendingPulse {
            source: TensionSource::Question { id: question_id(&q) },
            key: format!("{hash:016x}"),
            energy,
        });
    }
    out
}

/// Apply the release to the items the pool actually accepted: predictions get
/// their cooldown stamp, questions decay and may abandon (mirrored into `sm`).
/// Items the pool dropped keep their state untouched, so a drop never costs a
/// question one of its attempts.
pub fn settle(inner: &mut EngineInner, sm: &mut crate::core::SelfModel, accepted_keys: &[String], now_secs: f64) {
    for key in accepted_keys {
        if let Some(entry) = inner.prediction_targets.get_mut(key) {
            entry.1 = now_secs;
            continue;
        }
        if let Ok(hash) = u64::from_str_radix(key, 16)
            && let Some(state) = inner.questions.get_mut(&hash)
        {
            state.after_release(now_secs);
            if state.abandoned {
                sm.note_question_abandoned(hash);
            }
        }
    }
}

// ── Target resolution (I/O) ──────────────────────────────────────────────────

/// Embedding function, injectable so tests never depend on the global embedder.
pub type EmbedFn = std::sync::Arc<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync>;

/// Max cosine distance for a question → node match (0 = identical).
const QUESTION_MATCH_MAX_DISTANCE: f32 = 0.6;

/// Turn planned releases into node-targeted pulses. Targets are resolved once
/// per item and cached. Predictions → top nodes of the domain by importance;
/// questions → embedding similarity, with the embedding computed on the
/// blocking pool (CPU-bound). An item that maps to nothing is counted as
/// `unmapped` once; `plan_release` retries it after the cooldown.
pub async fn resolve_targets_with(
    pool: &sqlx::PgPool,
    inner: &mut EngineInner,
    pending: Vec<PendingPulse>,
    embed: EmbedFn,
) -> Vec<TensionPulse> {
    let mut out = Vec::new();
    for item in pending {
        let targets: Vec<i32> = match &item.source {
            TensionSource::Prediction { domain } => {
                let cached = inner.prediction_targets.get(domain).map(|(t, _)| t.clone()).unwrap_or_default();
                let targets = if cached.is_empty() {
                    crate::db::top_nodes_in_domain(pool, domain, K_PREDICTION_TARGETS).await.unwrap_or_default()
                } else {
                    cached
                };
                if let Some(entry) = inner.prediction_targets.get_mut(domain) {
                    entry.0 = targets.clone(); // cooldown was reserved at plan time
                }
                if targets.is_empty() {
                    inner.unmapped += 1;
                }
                targets
            }
            TensionSource::Question { .. } => {
                let hash = u64::from_str_radix(&item.key, 16).unwrap_or(0);
                let state = inner.questions.entry(hash).or_default();
                if state.unmapped {
                    continue; // already counted; plan_release lifts the flag after the cooldown
                }
                if state.targets.is_empty() {
                    let text = state.text.clone();
                    let embed = embed.clone();
                    let emb = tokio::task::spawn_blocking(move || embed(&text)).await;
                    let targets = match emb {
                        Ok(Ok(vector)) => crate::db::nearest_node_ids(pool, &vector, K_QUESTION_TARGETS as i64, QUESTION_MATCH_MAX_DISTANCE)
                            .await
                            .unwrap_or_else(|e| {
                                tracing::warn!("[tension] nearest-node lookup failed: {}", e);
                                Vec::new()
                            }),
                        _ => Vec::new(),
                    };
                    let state = inner.questions.entry(hash).or_default();
                    state.targets = targets;
                }
                let state = inner.questions.get_mut(&hash).expect("inserted above");
                if state.targets.is_empty() {
                    // Failed mapping: counts as an attempt (so a permanently unmappable
                    // question is eventually abandoned) and starts the cooldown, so it
                    // is not re-embedded on every walk.
                    state.unmapped = true;
                    state.attempts += 1;
                    if state.attempts >= QUESTION_MAX_ATTEMPTS {
                        state.abandoned = true; // mirrored into the self-model on the next plan
                    }
                    inner.unmapped += 1;
                }
                state.targets.clone()
            }
        };
        if targets.is_empty() {
            continue;
        }
        let share = item.energy / targets.len() as f32;
        for t in targets {
            out.push(TensionPulse { source: item.source.clone(), target_node: t, energy: share });
        }
    }
    out
}

/// `resolve_targets_with` using the process embedder.
pub async fn resolve_targets(pool: &sqlx::PgPool, inner: &mut EngineInner, pending: Vec<PendingPulse>) -> Vec<TensionPulse> {
    resolve_targets_with(pool, inner, pending, std::sync::Arc::new(|t: &str| crate::embed::embed_text(t))).await
}

// ── Charging against the pool ────────────────────────────────────────────────

/// Result of charging one release.
#[derive(Debug, Default)]
pub struct Charged {
    /// Pulses to put into the inbox (empty unless the mode emits).
    pub emit: Vec<TensionPulse>,
    /// Pulses the pool accepted but the mode does not emit (observe/brakes);
    /// `after_walk` checks these against the reactor's energies.
    pub would_fire: Vec<TensionPulse>,
    /// Keys of the items the pool accepted; `settle` consumes these.
    pub accepted_keys: Vec<String>,
}

/// How many of `pulses` would lift their target over its current threshold,
/// given the reactor's decayed energy and fatigue at `now_secs`. The reactor
/// resets a node to zero when it fires, so each pulse is judged on its own.
pub async fn count_would_cross(view: &crate::diverger::EnergyView, pulses: &[TensionPulse], now_secs: f64) -> u64 {
    let mut n = 0;
    for p in pulses {
        let (energy, threshold) = view.headroom(p.target_node, now_secs).await;
        if energy + p.energy >= threshold {
            n += 1;
        }
    }
    n
}

fn pulse_key(p: &TensionPulse) -> String {
    match &p.source {
        TensionSource::Prediction { domain } => domain.clone(),
        TensionSource::Question { id } => id.trim_start_matches("question:").to_string(),
    }
}

/// Charge each *item* (all of its target pulses together) to the global pool
/// and record the outcome. In non-emitting modes an item is charged its energy
/// plus ONE estimated cascade, so the pool's empty fraction forecasts `live`
/// honestly without billing a cascade per target. A dropped item is counted
/// once, logged nothing, and left for `settle` to leave untouched.
pub fn charge(inner: &mut EngineInner, pulses: Vec<TensionPulse>, now_secs: f64) -> Charged {
    let mut out = Charged::default();
    let emits = inner.mode.emits();
    // Group by item, preserving first-seen order.
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<TensionPulse>> = HashMap::new();
    for p in pulses {
        let k = pulse_key(&p);
        if !groups.contains_key(&k) {
            order.push(k.clone());
        }
        groups.entry(k).or_default().push(p);
    }
    for key in order {
        let group = groups.remove(&key).unwrap_or_default();
        let energy: f32 = group.iter().map(|p| p.energy).sum();
        let cost = if emits { energy } else { energy + inner.avg_edge_cascade_hops * HOP_COST };
        let ok = inner.pool.try_spend(cost, now_secs);
        inner.pool_samples.push_back((now_secs, !ok));
        trim_window(&mut inner.pool_samples, now_secs);
        if !ok {
            inner.dropped_pool_empty += 1;
            continue;
        }
        out.accepted_keys.push(key);
        for p in group {
            inner.recent.push_back(RecentEvent {
                t: now_secs as u64,
                source: p.source.clone(),
                target: p.target_node,
                energy: p.energy,
                fired: emits,
            });
            while inner.recent.len() > RECENT_MAX {
                inner.recent.pop_front();
            }
            if emits {
                inner.pulses_fired += 1;
                tracing::info!("[tension] fire source={} target={} energy={:.3} pool={:.1}", p.source, p.target_node, p.energy, inner.pool.tokens);
                out.emit.push(p);
            } else {
                inner.pulses_would_fire += 1;
                tracing::info!("[tension] would-fire source={} target={} energy={:.3} pool={:.1}", p.source, p.target_node, p.energy, inner.pool.tokens);
                out.would_fire.push(p);
            }
        }
    }
    out
}

// ── Recording ────────────────────────────────────────────────────────────────

pub fn record_edge_cascade_hops(inner: &mut EngineInner, hops: u32) {
    inner.hop_samples.push_back(hops);
    while inner.hop_samples.len() > HOP_SAMPLES_MAX {
        inner.hop_samples.pop_front();
    }
    let n = inner.hop_samples.len() as f32;
    inner.avg_edge_cascade_hops = inner.hop_samples.iter().map(|h| *h as f32).sum::<f32>() / n;
}

/// A feedback hop the pool could not pay for: counted like a dropped pulse and
/// sampled as an empty-pool event (spec §6.3).
pub fn record_hop_pool_drop(inner: &mut EngineInner, now_secs: f64) {
    inner.dropped_pool_empty += 1;
    inner.pool_samples.push_back((now_secs, true));
    trim_window(&mut inner.pool_samples, now_secs);
}

pub fn record_inbox_full(inner: &mut EngineInner) {
    inner.dropped_inbox_full += 1;
}

pub fn record_cascade_end(inner: &mut EngineInner, origin: Origin, by_budget: bool) {
    match origin {
        Origin::Edge => inner.cascades_edge_origin += 1,
        Origin::Tension => inner.cascades_tension_origin += 1,
    }
    if by_budget {
        inner.cascades_ended_by_budget += 1;
    } else {
        inner.cascades_ended_naturally += 1;
    }
}

/// Called from `after_walk` on every walk (never from `stats`), so the
/// resolution rate does not depend on how often `/diverger` is polled.
pub fn record_resolution_sample(inner: &mut EngineInner, made: u64, resolved: u64, now_secs: f64) {
    inner.resolution_samples.push_back((now_secs, made, resolved));
    while let Some((t, _, _)) = inner.resolution_samples.front() {
        if now_secs - *t > WINDOW_SECS {
            inner.resolution_samples.pop_front();
        } else {
            break;
        }
    }
}

// ── Stats (read-only) ────────────────────────────────────────────────────────

pub fn stats(inner: &EngineInner, sm: &crate::core::SelfModel, now_secs: f64) -> TensionStats {
    let in_window = |t: f64| now_secs - t <= WINDOW_SECS;
    let pool_in: Vec<&(f64, bool)> = inner.pool_samples.iter().filter(|(t, _)| in_window(*t)).collect();
    let empty_fraction_1h = if pool_in.is_empty() {
        0.0
    } else {
        pool_in.iter().filter(|(_, e)| *e).count() as f32 / pool_in.len() as f32
    };
    let res_in: Vec<&(f64, u64, u64)> = inner.resolution_samples.iter().filter(|(t, _, _)| in_window(*t)).collect();
    let resolution_rate_1h = match (res_in.first(), res_in.last()) {
        (Some((_, m0, r0)), Some((_, m1, r1))) if m1 > m0 => (r1 - r0) as f32 / (m1 - m0) as f32,
        _ => 0.0,
    };
    TensionStats {
        mode: inner.mode.as_str().to_string(),
        open_predictions: sm.predictions.len() as u64,
        open_questions: sm.open_questions.len() as u64,
        abandoned_questions: sm.abandoned_question_hashes.len() as u64,
        unmapped: inner.unmapped,
        pulses_would_fire: inner.pulses_would_fire,
        pulses_would_cross: inner.pulses_would_cross,
        pulses_fired: inner.pulses_fired,
        dropped_pool_empty: inner.dropped_pool_empty,
        dropped_item_cooldown: inner.dropped_item_cooldown,
        dropped_inbox_full: inner.dropped_inbox_full,
        dropped_compliant: inner.dropped_compliant,
        cascades: CascadeStats {
            tension_origin: inner.cascades_tension_origin,
            edge_origin: inner.cascades_edge_origin,
            ended_by_budget: inner.cascades_ended_by_budget,
            ended_naturally: inner.cascades_ended_naturally,
        },
        brakes_would_have: BrakesWouldHave {
            fatigue_blocks: inner.would_fatigue_blocks,
            budget_cuts: inner.would_budget_cuts,
            pool_drops: inner.would_pool_drops,
        },
        pool: PoolStats { tokens: inner.pool.tokens, capacity: inner.pool.capacity, empty_fraction_1h },
        predictions_made_total: sm.predictions_made_total,
        predictions_resolved_total: sm.predictions_resolved_total,
        resolution_rate_1h,
        live_judgement_hours: LIVE_JUDGEMENT_HOURS,
        recent: inner.recent.iter().cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CognitiveMode, ExpectedOutcome, SelfModel};

    fn sm_with(preds: &[(&str, f64)], qs: &[&str], now: f64) -> SelfModel {
        let mut sm = SelfModel::new();
        sm.mode = CognitiveMode::Autonomous;
        for (d, age) in preds {
            sm.predictions.insert(
                d.to_string(),
                ExpectedOutcome { domain: d.to_string(), predicted: "x".into(), predicted_at: now - age, confidence: 0.5 },
            );
        }
        sm.open_questions = qs.iter().map(|q| q.to_string()).collect();
        sm
    }

    #[tokio::test]
    async fn unmapped_question_is_counted_once_and_not_retried_within_cooldown() {
        let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
        let failing: EmbedFn = std::sync::Arc::new(|_| Err(anyhow::anyhow!("no embedder in this test")));
        let mut inner = EngineInner::new(TensionMode::Observe, 0.0);
        let q = || PendingPulse {
            source: TensionSource::Question { id: question_id("q") },
            key: format!("{:016x}", fnv1a64("q")),
            energy: 0.2,
        };
        assert!(resolve_targets_with(&pool, &mut inner, vec![q()], failing.clone()).await.is_empty());
        assert_eq!(inner.unmapped, 1);
        assert!(inner.questions.get(&fnv1a64("q")).unwrap().unmapped);
        assert!(resolve_targets_with(&pool, &mut inner, vec![q()], failing).await.is_empty());
        assert_eq!(inner.unmapped, 1, "not counted again");
    }

    #[tokio::test]
    async fn after_walk_without_engine_is_a_noop() {
        let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
        let sm = std::sync::Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
        assert_eq!(after_walk(&pool, &sm).await, 0); // no engine installed in this test binary
        assert_eq!(mode(), TensionMode::Off);
    }

    #[test]
    fn plan_release_respects_cooldown_cap_and_freshness() {
        let now = 1_000_000.0;
        let mut inner = EngineInner::new(TensionMode::Observe, now);
        let preds: Vec<(String, f64)> = (0..15).map(|i| (format!("d{i}"), 3600.0 + i as f64)).collect();
        let pv: Vec<(&str, f64)> = preds.iter().map(|(d, a)| (d.as_str(), *a)).collect();
        let mut sm = sm_with(&pv, &[], now);
        let p1 = plan_release(&mut inner, &mut sm, now);
        assert_eq!(p1.len(), PREDICTION_MAX_ACTIVE);
        assert!(
            p1.iter().all(|p| matches!(&p.source, TensionSource::Prediction { domain } if domain != "d14")),
            "oldest excluded"
        );
        let keys: Vec<String> = p1.iter().map(|p| p.key.clone()).collect();
        settle(&mut inner, &mut sm, &keys, now);
        let p2 = plan_release(&mut inner, &mut sm, now + 1.0);
        assert!(p2.is_empty(), "cooldown");
        assert_eq!(inner.dropped_item_cooldown, PREDICTION_MAX_ACTIVE as u64);
        let p3 = plan_release(&mut inner, &mut sm, now + ITEM_COOLDOWN_SECS + 1.0);
        assert_eq!(p3.len(), PREDICTION_MAX_ACTIVE);
    }

    #[test]
    fn plan_release_abandons_questions_and_never_revives_them() {
        let now = 0.0;
        let mut inner = EngineInner::new(TensionMode::Observe, now);
        let mut sm = sm_with(&[], &["why is the ferry late"], now);
        let mut t = now;
        let mut releases = 0;
        for _ in 0..(QUESTION_MAX_ATTEMPTS + 3) {
            let pending = plan_release(&mut inner, &mut sm, t);
            if !pending.is_empty() {
                releases += 1;
                let keys: Vec<String> = pending.iter().map(|p| p.key.clone()).collect();
                settle(&mut inner, &mut sm, &keys, t);
            }
            t += ITEM_COOLDOWN_SECS + 1.0;
        }
        assert!(releases <= QUESTION_MAX_ATTEMPTS as usize && releases >= 1, "releases={releases}");
        assert!(sm.is_question_abandoned(fnv1a64("why is the ferry late")));
        sm.open_questions = vec!["why is the ferry late".into()];
        assert!(plan_release(&mut inner, &mut sm, t + ITEM_COOLDOWN_SECS).is_empty());
    }

    #[test]
    fn side_tables_are_pruned_to_live_items() {
        let now = 0.0;
        let mut inner = EngineInner::new(TensionMode::Observe, now);
        // rotate 30 distinct questions through a self-model that keeps at most 10 open
        let mut sm = sm_with(&[("d1", 3600.0), ("d2", 3600.0)], &[], now);
        for i in 0..30 {
            sm.open_questions = (i..i + 10).map(|k| format!("q{k}")).collect();
            plan_release(&mut inner, &mut sm, now + i as f64 * (ITEM_COOLDOWN_SECS + 1.0));
        }
        assert!(inner.questions.len() <= 10, "questions side table {}", inner.questions.len());
        sm.predictions.remove("d1");
        plan_release(&mut inner, &mut sm, now + 1e6);
        assert_eq!(inner.prediction_targets.len(), 1, "stale domain pruned");
    }

    #[test]
    fn plan_release_is_silent_in_compliant_mode() {
        let now = 0.0;
        let mut inner = EngineInner::new(TensionMode::Live, now);
        let mut sm = sm_with(&[("markets", 3600.0)], &[], now);
        sm.mode = CognitiveMode::Compliant;
        assert!(plan_release(&mut inner, &mut sm, now).is_empty());
        assert_eq!(inner.dropped_compliant, 1);
    }

    #[test]
    fn charge_emits_only_in_live_and_estimates_cascade_cost_otherwise() {
        let now = 0.0;
        let pulse = |e: f32| TensionPulse {
            source: TensionSource::Prediction { domain: "m".into() },
            target_node: 1,
            energy: e,
        };
        let mut obs = EngineInner::new(TensionMode::Observe, now);
        assert!(charge(&mut obs, vec![pulse(0.1)], now).emit.is_empty());
        assert_eq!(obs.pulses_would_fire, 1);
        assert!((obs.pool.tokens - (20.0 - 0.1 - DEFAULT_EDGE_CASCADE_HOPS * HOP_COST)).abs() < 1e-5);
        let mut live = EngineInner::new(TensionMode::Live, now);
        assert_eq!(charge(&mut live, vec![pulse(0.1)], now).emit.len(), 1);
        assert_eq!(live.pulses_fired, 1);
        assert!((live.pool.tokens - 19.9).abs() < 1e-5);
        live.pool.tokens = 0.05;
        assert!(charge(&mut live, vec![pulse(0.1)], now).emit.is_empty());
        assert_eq!(live.dropped_pool_empty, 1);
    }

    #[test]
    fn charge_bills_once_per_item_and_a_dropped_item_keeps_its_attempt() {
        let now = 0.0;
        let three = |key: &str, e: f32| -> Vec<TensionPulse> {
            (1..=3)
                .map(|t| TensionPulse { source: TensionSource::Question { id: format!("question:{key}") }, target_node: t, energy: e })
                .collect()
        };
        // observe: one item with 3 targets is charged energy-sum + ONE cascade estimate
        let mut obs = EngineInner::new(TensionMode::Observe, now);
        let c = charge(&mut obs, three("a", 0.05), now);
        assert!(c.emit.is_empty());
        assert_eq!(c.accepted_keys, vec!["a".to_string()]);
        assert!((obs.pool.tokens - (20.0 - 0.15 - DEFAULT_EDGE_CASCADE_HOPS * HOP_COST)).abs() < 1e-5, "{}", obs.pool.tokens);
        assert_eq!(obs.pulses_would_fire, 3);
        // a dropped item: not accepted, counted once, and settle() leaves its attempt count alone
        let mut sm = sm_with(&[], &["q"], now);
        let key = format!("{:016x}", fnv1a64("q"));
        let pending = plan_release(&mut obs, &mut sm, now);
        assert_eq!(pending.len(), 1);
        obs.pool.tokens = 0.01;
        let pulses: Vec<TensionPulse> = pending.iter().map(|p| TensionPulse { source: p.source.clone(), target_node: 1, energy: p.energy }).collect();
        let c2 = charge(&mut obs, pulses, now);
        assert!(c2.accepted_keys.is_empty());
        assert_eq!(obs.dropped_pool_empty, 1, "one item, not one per target");
        settle(&mut obs, &mut sm, &c2.accepted_keys, now);
        assert_eq!(obs.questions[&fnv1a64("q")].attempts, 0, "attempt not consumed when the pool dropped it");
        // accepted → attempt consumed and cooldown set
        settle(&mut obs, &mut sm, &[key.clone()], now);
        assert_eq!(obs.questions[&fnv1a64("q")].attempts, 1);
        assert!(obs.questions[&fnv1a64("q")].last_release >= now);
    }

    #[test]
    fn hop_and_inbox_accounting_helpers() {
        let now = 0.0;
        let mut inner = EngineInner::new(TensionMode::Live, now);
        record_hop_pool_drop(&mut inner, now);
        assert_eq!(inner.dropped_pool_empty, 1);
        assert_eq!(inner.pool_samples.len(), 1);
        assert!(inner.pool_samples[0].1, "sampled as empty");
        record_inbox_full(&mut inner);
        assert_eq!(inner.dropped_inbox_full, 1);
    }

    #[tokio::test]
    async fn spawn_after_walk_without_engine_returns_none() {
        let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
        let sm = std::sync::Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
        assert!(spawn_after_walk(pool, sm).is_none());
    }

    #[tokio::test]
    async fn unmapped_question_is_not_re_embedded_within_cooldown() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let failing: EmbedFn = std::sync::Arc::new(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("no embedder"))
        });
        let now = 1000.0;
        let mut inner = EngineInner::new(TensionMode::Observe, now);
        let mut sm = sm_with(&[], &["unmappable"], now);
        let p1 = plan_release(&mut inner, &mut sm, now);
        assert_eq!(p1.len(), 1);
        assert!(resolve_targets_with(&pool, &mut inner, p1, failing.clone()).await.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // next walk, 1 s later: still inside the cooldown → not planned, not embedded
        let p2 = plan_release(&mut inner, &mut sm, now + 1.0);
        assert!(p2.is_empty(), "unmapped item must respect the cooldown");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // after the cooldown it may retry once
        let p3 = plan_release(&mut inner, &mut sm, now + ITEM_COOLDOWN_SECS + 1.0);
        assert_eq!(p3.len(), 1);
        let _ = resolve_targets_with(&pool, &mut inner, p3, failing).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(inner.questions[&fnv1a64("unmappable")].attempts, 2, "failed mappings count as attempts");
    }

    #[test]
    fn release_token_clears_the_flag_when_dropped_even_on_panic() {
        let g = ReleaseGuard::default();
        let token = g.try_begin().expect("first release admitted");
        assert!(g.try_begin().is_none(), "second concurrent release is coalesced away");
        drop(token);
        let token = g.try_begin().expect("released after the token dropped");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _t = token;
            panic!("release task died");
        }));
        assert!(r.is_err());
        assert!(g.try_begin().is_some(), "a panicked release must not wedge the engine");
    }

    #[test]
    fn token_pool_refund_restores_spent_tokens_up_to_capacity() {
        let mut pool = TokenPool::new(0.0);
        assert!(pool.try_spend(HOP_COST, 0.0));
        let after = pool.tokens;
        pool.refund(HOP_COST);
        assert!((pool.tokens - (after + HOP_COST)).abs() < 1e-6);
        pool.refund(100.0);
        assert_eq!(pool.tokens, pool.capacity, "refund never exceeds capacity");
    }

    fn qpulse(target: i32, energy: f32) -> TensionPulse {
        TensionPulse {
            source: TensionSource::Question { id: "question:00000000000000ab".into() },
            target_node: target,
            energy,
        }
    }

    #[test]
    fn charge_returns_would_fire_pulses_in_non_emitting_modes() {
        let mut observe = EngineInner::new(TensionMode::Observe, 0.0);
        let c = charge(&mut observe, vec![qpulse(7, 0.1)], 0.0);
        assert!(c.emit.is_empty());
        assert_eq!(c.would_fire.len(), 1);
        assert_eq!(c.would_fire[0].target_node, 7);

        let mut live = EngineInner::new(TensionMode::Live, 0.0);
        let c = charge(&mut live, vec![qpulse(7, 0.1)], 0.0);
        assert_eq!(c.emit.len(), 1);
        assert!(c.would_fire.is_empty());
    }

    #[tokio::test]
    async fn would_cross_counts_only_pulses_that_lift_a_target_over_its_threshold() {
        let view = crate::diverger::EnergyView::for_test(0.5, 0.01);
        view.set_energy(1, 0.47, 0.0).await; // 0.47 + 0.05 >= 0.5: crosses
        view.set_energy(2, 0.10, 0.0).await; // 0.15 < 0.5: does not
        // node 3 is cold (not in the map): 0.05 < 0.5
        let pulses = vec![qpulse(1, 0.05), qpulse(2, 0.05), qpulse(3, 0.05)];
        assert_eq!(count_would_cross(&view, &pulses, 0.0).await, 1);
    }

    #[test]
    fn stats_rates_and_ring_buffers_are_bounded() {
        let now = 0.0;
        let mut inner = EngineInner::new(TensionMode::Live, now);
        for i in 0..120u32 {
            record_edge_cascade_hops(&mut inner, i % 5);
        }
        assert_eq!(inner.hop_samples.len(), 100);
        for i in 0..60 {
            charge(
                &mut inner,
                vec![TensionPulse { source: TensionSource::Prediction { domain: "m".into() }, target_node: i, energy: 0.01 }],
                now,
            );
        }
        assert_eq!(inner.recent.len(), 50);
        record_resolution_sample(&mut inner, 10, 5, now);
        record_resolution_sample(&mut inner, 20, 13, now + 60.0);
        let sm = SelfModel::new();
        let s = stats(&inner, &sm, now + 60.0);
        assert_eq!(s.mode, "live");
        assert!((s.resolution_rate_1h - 0.8).abs() < 1e-6, "Δresolved 8 / Δmade 10");
        assert!(s.pool.empty_fraction_1h >= 0.0 && s.pool.empty_fraction_1h <= 1.0);
        let _s2 = stats(&inner, &sm, now + 60.0);
        assert_eq!(inner.resolution_samples.len(), 2, "stats() never records");
    }

    #[test]
    fn mode_parses_case_insensitively_and_defaults_to_observe() {
        assert_eq!(TensionMode::from_env_value(Some("LIVE")), TensionMode::Live);
        assert_eq!(TensionMode::from_env_value(Some("brakes")), TensionMode::Brakes);
        assert_eq!(TensionMode::from_env_value(None), TensionMode::Observe);
        assert_eq!(TensionMode::from_env_value(Some("")), TensionMode::Observe);
        assert_eq!(TensionMode::from_env_value(Some("banana")), TensionMode::Observe);
        assert!(TensionMode::Live.emits() && !TensionMode::Brakes.emits());
        assert!(TensionMode::Brakes.enforces_brakes() && !TensionMode::Observe.enforces_brakes());
    }

    #[test]
    fn fnv1a64_known_vectors() {
        assert_eq!(fnv1a64(""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64("a"), 0xaf63dc4c8601ec8c);
        assert_eq!(question_id("a"), "question:af63dc4c8601ec8c");
    }

    #[test]
    fn prediction_strength_saturates_and_halves_per_half_life() {
        assert!(prediction_strength(0.0).abs() < 1e-6);
        assert!((prediction_strength(1800.0) - 0.075).abs() < 1e-5);
        assert!((prediction_strength(3600.0) - 0.15).abs() < 1e-5);
        assert!((prediction_strength(7200.0) - 0.15).abs() < 1e-5);
        assert!((prediction_strength(86400.0) - 0.075).abs() < 1e-5);
        assert!((prediction_strength(2.0 * 86400.0) - 0.0375).abs() < 1e-5);
    }

    #[test]
    fn token_pool_spends_refills_and_caps() {
        let mut p = TokenPool::new(0.0);
        assert_eq!(p.tokens, 20.0);
        assert!(p.try_spend(19.5, 0.0));
        assert!(!p.try_spend(1.0, 0.0));
        assert!((p.tokens - 0.5).abs() < 1e-6);
        p.refill(10.0);
        assert!((p.tokens - 1.0).abs() < 1e-6);
        p.refill(1_000_000.0);
        assert_eq!(p.tokens, 20.0);
    }

    #[test]
    fn fatigue_raises_threshold_and_decays() {
        assert!((effective_threshold(1.0, 0.5) - 1.5).abs() < 1e-6);
        assert!((decay_fatigue(0.5, 10.0) - 0.4).abs() < 1e-6);
        assert_eq!(decay_fatigue(0.05, 100.0), 0.0);
    }

    #[test]
    fn question_decays_and_abandons() {
        let mut q = QuestionState::new();
        for _ in 0..QUESTION_MAX_ATTEMPTS {
            assert!(!q.abandoned);
            q.after_release(0.0);
        }
        assert!(q.abandoned);
        let mut q2 = QuestionState::new();
        q2.strength = 0.021;
        q2.after_release(0.0);
        assert!(q2.abandoned, "strength floor");
    }
}
