//! LLM cost accounting — the measurement spine for DeepSeek efficiency.
//!
//! Every generation call (DeepSeek or any cloud provider) flows through
//! `record_call`, which captures the REAL `usage` object returned by the
//! API (prompt/completion tokens) rather than a `split_whitespace()` guess.
//! Dedup hits (calls we avoided) flow through `record_dedup_hit`, so the
//! duplicate-call rate is observable. `snapshot()` renders the scorecard
//! numbers — calls/min, tokens/call, % reasoner, estimated cost — live.
//!
//! This module never logs API keys. It only counts tokens.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Serialize;

/// DeepSeek published pricing, USD per 1M tokens (cache-miss rates).
/// Used for cost ESTIMATION only; override with env if rates change.
const CHAT_INPUT_USD_PER_M: f64 = 0.27;
const CHAT_OUTPUT_USD_PER_M: f64 = 1.10;
const REASONER_INPUT_USD_PER_M: f64 = 0.55;
const REASONER_OUTPUT_USD_PER_M: f64 = 2.19;

/// One global metrics sink. Atomics for the hot counters; a small Mutex
/// only for the rolling timestamp window used to compute calls/min.
pub struct LlmMetrics {
    pub calls: AtomicU64,
    pub failed_calls: AtomicU64,
    /// Calls served from the dedup cache (i.e. API calls avoided).
    pub dedup_hits: AtomicU64,
    pub chat_calls: AtomicU64,
    pub reasoner_calls: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    /// Cost in micro-USD (USD * 1e6) to keep it an integer atomic.
    pub cost_micros: AtomicU64,
    /// Calls skipped by the spend circuit breaker (cost ceiling hit).
    pub throttled: AtomicU64,
    /// Number of completed walker sessions recorded.
    pub walk_sessions: AtomicU64,
    /// Sum of walk novelty scores scaled by 1e6.
    pub walk_novelty_micros: AtomicU64,
    /// Sessions flagged as repetitive.
    pub repeated_walks: AtomicU64,
    /// Sum of observed cascade depths.
    pub cascade_depth_total: AtomicU64,
    /// Number of cascade depth events.
    pub cascade_events: AtomicU64,
    /// Metacognitive approvals/rejections.
    pub metacog_approved: AtomicU64,
    pub metacog_rejected: AtomicU64,
    /// New emergent goals formed.
    pub goals_formed: AtomicU64,
    /// Theory-of-mind prediction outcomes.
    pub tom_predictions: AtomicU64,
    pub tom_correct: AtomicU64,
    /// Cost accrued in the current accounting day (micro-USD), reset on day roll.
    day_cost_micros: AtomicU64,
    /// Current accounting day (unix_secs / 86400). Detects the day boundary.
    day_epoch: AtomicU64,
    /// Unix-seconds timestamps of recent successful calls (rolling 60s window).
    recent: Mutex<Vec<f64>>,
    started_at: f64,
}

impl LlmMetrics {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            failed_calls: AtomicU64::new(0),
            dedup_hits: AtomicU64::new(0),
            chat_calls: AtomicU64::new(0),
            reasoner_calls: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            cost_micros: AtomicU64::new(0),
            throttled: AtomicU64::new(0),
            walk_sessions: AtomicU64::new(0),
            walk_novelty_micros: AtomicU64::new(0),
            repeated_walks: AtomicU64::new(0),
            cascade_depth_total: AtomicU64::new(0),
            cascade_events: AtomicU64::new(0),
            metacog_approved: AtomicU64::new(0),
            metacog_rejected: AtomicU64::new(0),
            goals_formed: AtomicU64::new(0),
            tom_predictions: AtomicU64::new(0),
            tom_correct: AtomicU64::new(0),
            day_cost_micros: AtomicU64::new(0),
            day_epoch: AtomicU64::new(current_day()),
            recent: Mutex::new(Vec::new()),
            started_at: now_secs(),
        }
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The current accounting day as a unix-day number.
fn current_day() -> u64 {
    (now_secs() as u64) / 86_400
}

lazy_static::lazy_static! {
    static ref METRICS: LlmMetrics = LlmMetrics::new();
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A model name is the reasoner path if it mentions "reasoner".
fn is_reasoner(model: &str) -> bool {
    model.contains("reasoner")
}

/// Estimate USD cost for one call given the model and token usage.
pub fn estimate_cost_usd(model: &str, prompt_tokens: u64, completion_tokens: u64) -> f64 {
    let (in_rate, out_rate) = if is_reasoner(model) {
        (REASONER_INPUT_USD_PER_M, REASONER_OUTPUT_USD_PER_M)
    } else {
        (CHAT_INPUT_USD_PER_M, CHAT_OUTPUT_USD_PER_M)
    };
    (prompt_tokens as f64 / 1_000_000.0) * in_rate
        + (completion_tokens as f64 / 1_000_000.0) * out_rate
}

/// Record a successful generation call with REAL usage from the API.
pub fn record_call(model: &str, prompt_tokens: u64, completion_tokens: u64) {
    METRICS.calls.fetch_add(1, Ordering::Relaxed);
    if is_reasoner(model) {
        METRICS.reasoner_calls.fetch_add(1, Ordering::Relaxed);
    } else {
        METRICS.chat_calls.fetch_add(1, Ordering::Relaxed);
    }
    METRICS.prompt_tokens.fetch_add(prompt_tokens, Ordering::Relaxed);
    METRICS.completion_tokens.fetch_add(completion_tokens, Ordering::Relaxed);

    let cost = estimate_cost_usd(model, prompt_tokens, completion_tokens);
    let cost_micros = (cost * 1_000_000.0) as u64;
    roll_day_if_needed();
    METRICS.cost_micros.fetch_add(cost_micros, Ordering::Relaxed);
    METRICS.day_cost_micros.fetch_add(cost_micros, Ordering::Relaxed);

    if let Ok(mut recent) = METRICS.recent.lock() {
        let now = now_secs();
        recent.retain(|&t| now - t < 60.0);
        recent.push(now);
    }

    tracing::info!(
        "[metrics] call model={} prompt_tok={} completion_tok={} est_cost=${:.6}",
        model, prompt_tokens, completion_tokens, cost
    );
}

/// Record a call that was avoided because the prompt hash was already seen.
pub fn record_dedup_hit() {
    METRICS.dedup_hits.fetch_add(1, Ordering::Relaxed);
}

/// Record an LLM call the spend circuit breaker refused to make.
pub fn record_throttle() {
    METRICS.throttled.fetch_add(1, Ordering::Relaxed);
}

/// Record a completed walker session's novelty and whether it looked repetitive.
pub fn record_walk_session(novelty_score: f32, repeated: bool) {
    METRICS.walk_sessions.fetch_add(1, Ordering::Relaxed);
    let novelty = novelty_score.clamp(0.0, 1.0);
    METRICS
        .walk_novelty_micros
        .fetch_add((novelty * 1_000_000.0) as u64, Ordering::Relaxed);
    if repeated {
        METRICS.repeated_walks.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record one cascade depth sample from a spontaneous diverger walk.
pub fn record_cascade_depth(depth: u32) {
    METRICS
        .cascade_depth_total
        .fetch_add(depth as u64, Ordering::Relaxed);
    METRICS.cascade_events.fetch_add(1, Ordering::Relaxed);
}

/// Record an action-level metacognitive verdict.
pub fn record_metacog_decision(approved: bool) {
    if approved {
        METRICS.metacog_approved.fetch_add(1, Ordering::Relaxed);
    } else {
        METRICS.metacog_rejected.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a newly formed emergent goal/pattern.
pub fn record_goal_formed() {
    METRICS.goals_formed.fetch_add(1, Ordering::Relaxed);
}

/// Record a theory-of-mind prediction check.
pub fn record_tom_prediction(correct: bool) {
    METRICS.tom_predictions.fetch_add(1, Ordering::Relaxed);
    if correct {
        METRICS.tom_correct.fetch_add(1, Ordering::Relaxed);
    }
}

/// Reset the per-day cost accumulator when the accounting day rolls over.
fn roll_day_if_needed() {
    let today = current_day();
    let stored = METRICS.day_epoch.load(Ordering::Relaxed);
    if stored != today
        && METRICS
            .day_epoch
            .compare_exchange(stored, today, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        METRICS.day_cost_micros.store(0, Ordering::Relaxed);
    }
}

/// Cost circuit breaker. Returns `Err(reason)` when a configured ceiling is hit
/// so the caller can skip the paid call. Both ceilings are env-tunable; set
/// either to `0` to disable that check.
///   `RGW_MAX_CALLS_PER_MIN` (default 60) — rolling 60s successful-call rate
///   `RGW_MAX_DAILY_USD`     (default 10.0) — estimated spend since midnight UTC
pub fn spend_allows() -> Result<(), &'static str> {
    spend_allows_with(
        env_u32("RGW_MAX_CALLS_PER_MIN", 60),
        env_f64("RGW_MAX_DAILY_USD", 10.0),
    )
}

/// Pure core of [`spend_allows`] — caps passed explicitly so it is testable
/// without mutating process env.
fn spend_allows_with(max_calls_per_min: u32, max_daily_usd: f64) -> Result<(), &'static str> {
    if max_calls_per_min > 0 {
        let cpm = METRICS
            .recent
            .lock()
            .map(|mut r| {
                let now = now_secs();
                r.retain(|&t| now - t < 60.0);
                r.len() as u32
            })
            .unwrap_or(0);
        if cpm >= max_calls_per_min {
            return Err("calls/min ceiling");
        }
    }
    if max_daily_usd > 0.0 {
        roll_day_if_needed();
        let spent = METRICS.day_cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        if spent >= max_daily_usd {
            return Err("daily USD ceiling");
        }
    }
    Ok(())
}

/// Record a failed generation attempt (no usage available).
pub fn record_failure() {
    METRICS.failed_calls.fetch_add(1, Ordering::Relaxed);
}

/// Serializable scorecard snapshot — the numbers PHASE 0/DELIVERABLE want.
#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub calls: u64,
    pub failed_calls: u64,
    pub dedup_hits: u64,
    pub chat_calls: u64,
    pub reasoner_calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub est_cost_usd: f64,
    pub day_cost_usd: f64,
    pub throttled: u64,
    pub uptime_secs: f64,
    pub calls_per_min: f64,
    pub calls_per_min_lifetime: f64,
    pub reasoner_pct: f64,
    pub dup_rate: f64,
    pub avg_prompt_tokens: f64,
    pub avg_completion_tokens: f64,
    pub tokens_per_call: f64,
    pub walk_sessions: u64,
    pub avg_walk_novelty: f64,
    pub repeated_walk_rate: f64,
    pub avg_cascade_depth: f64,
    pub metacog_approval_rate: f64,
    pub goals_formed: u64,
    pub tom_accuracy: f64,
}

/// Read the current metrics. Cheap; safe to call from a handler.
pub fn snapshot() -> MetricsSnapshot {
    let calls = METRICS.calls.load(Ordering::Relaxed);
    let dedup = METRICS.dedup_hits.load(Ordering::Relaxed);
    let chat = METRICS.chat_calls.load(Ordering::Relaxed);
    let reasoner = METRICS.reasoner_calls.load(Ordering::Relaxed);
    let prompt_t = METRICS.prompt_tokens.load(Ordering::Relaxed);
    let completion_t = METRICS.completion_tokens.load(Ordering::Relaxed);
    let cost = METRICS.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    roll_day_if_needed();
    let day_cost = METRICS.day_cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;

    let uptime = (now_secs() - METRICS.started_at).max(1e-6);
    let calls_per_min_recent = METRICS
        .recent
        .lock()
        .map(|mut r| {
            let now = now_secs();
            r.retain(|&t| now - t < 60.0);
            r.len() as f64
        })
        .unwrap_or(0.0);

    let attempts = calls + dedup; // calls we would have made without dedup
    let walk_sessions = METRICS.walk_sessions.load(Ordering::Relaxed);
    let walk_novelty_micros = METRICS.walk_novelty_micros.load(Ordering::Relaxed);
    let repeated_walks = METRICS.repeated_walks.load(Ordering::Relaxed);
    let cascade_events = METRICS.cascade_events.load(Ordering::Relaxed);
    let cascade_depth_total = METRICS.cascade_depth_total.load(Ordering::Relaxed);
    let metacog_approved = METRICS.metacog_approved.load(Ordering::Relaxed);
    let metacog_rejected = METRICS.metacog_rejected.load(Ordering::Relaxed);
    let goals_formed = METRICS.goals_formed.load(Ordering::Relaxed);
    let tom_predictions = METRICS.tom_predictions.load(Ordering::Relaxed);
    let tom_correct = METRICS.tom_correct.load(Ordering::Relaxed);

    MetricsSnapshot {
        calls,
        failed_calls: METRICS.failed_calls.load(Ordering::Relaxed),
        dedup_hits: dedup,
        chat_calls: chat,
        reasoner_calls: reasoner,
        prompt_tokens: prompt_t,
        completion_tokens: completion_t,
        total_tokens: prompt_t + completion_t,
        est_cost_usd: cost,
        day_cost_usd: day_cost,
        throttled: METRICS.throttled.load(Ordering::Relaxed),
        uptime_secs: uptime,
        calls_per_min: calls_per_min_recent,
        calls_per_min_lifetime: calls as f64 / (uptime / 60.0),
        reasoner_pct: if calls > 0 { reasoner as f64 / calls as f64 * 100.0 } else { 0.0 },
        dup_rate: if attempts > 0 { dedup as f64 / attempts as f64 * 100.0 } else { 0.0 },
        avg_prompt_tokens: if calls > 0 { prompt_t as f64 / calls as f64 } else { 0.0 },
        avg_completion_tokens: if calls > 0 { completion_t as f64 / calls as f64 } else { 0.0 },
        tokens_per_call: if calls > 0 { (prompt_t + completion_t) as f64 / calls as f64 } else { 0.0 },
        walk_sessions,
        avg_walk_novelty: if walk_sessions > 0 {
            (walk_novelty_micros as f64 / 1_000_000.0) / walk_sessions as f64
        } else { 0.0 },
        repeated_walk_rate: if walk_sessions > 0 {
            repeated_walks as f64 / walk_sessions as f64 * 100.0
        } else { 0.0 },
        avg_cascade_depth: if cascade_events > 0 {
            cascade_depth_total as f64 / cascade_events as f64
        } else { 0.0 },
        metacog_approval_rate: if (metacog_approved + metacog_rejected) > 0 {
            metacog_approved as f64 / (metacog_approved + metacog_rejected) as f64 * 100.0
        } else { 0.0 },
        goals_formed,
        tom_accuracy: if tom_predictions > 0 {
            tom_correct as f64 / tom_predictions as f64 * 100.0
        } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoner_is_detected_and_priced_higher() {
        assert!(is_reasoner("deepseek-reasoner"));
        assert!(!is_reasoner("deepseek-chat"));
        let chat = estimate_cost_usd("deepseek-chat", 1_000_000, 1_000_000);
        let reasoner = estimate_cost_usd("deepseek-reasoner", 1_000_000, 1_000_000);
        assert!(reasoner > chat, "reasoner must cost more than chat");
        // chat: 0.27 + 1.10 = 1.37 per 1M+1M
        assert!((chat - 1.37).abs() < 1e-9);
    }

    #[test]
    fn spend_guard_trips_on_daily_cap() {
        // A reasoner call of 1k+1k tokens costs ~$0.00274, well over a $0.0001 cap.
        record_call("deepseek-reasoner", 1000, 1000);
        // cpm check disabled (0) so the assertion is deterministic under parallel tests.
        assert!(
            spend_allows_with(0, 0.0001).is_err(),
            "daily-cost ceiling should trip once spend exceeds it"
        );
        // A ceiling of 0 disables the check entirely.
        assert!(spend_allows_with(0, 0.0).is_ok(), "a 0 ceiling disables the cap");
    }

    #[test]
    fn snapshot_reports_token_averages() {
        // Note: global state shared across parallel tests — assert only on
        // monotonic/relative behaviour, never exact counts.
        let before = snapshot().calls;
        record_call("deepseek-chat", 100, 50);
        let after = snapshot();
        assert!(after.calls >= before + 1, "a recorded call must increment the counter");
        assert!(after.total_tokens >= 150);
    }
}
