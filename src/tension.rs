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
}

// ── Open-question state ──────────────────────────────────────────────────────

/// Per-question release state. In-memory; the abandoned flag is mirrored into
/// `SelfModel::abandoned_question_hashes` so it survives restarts.
#[derive(Debug, Clone)]
pub struct QuestionState {
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

#[cfg(test)]
mod tests {
    use super::*;

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
