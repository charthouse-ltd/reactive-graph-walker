//! Prompt budgeting, dedup, and model routing — the three levers that
//! bound DeepSeek cost without dulling cognition.
//!
//!   * `budget_text`  — cap an enriched prompt to a token budget so graph
//!                      context can't grow unbounded into the request body.
//!   * dedup cache    — near-duplicate cognition states hash to the same
//!                      prompt; serve the cached completion instead of
//!                      paying for an identical call (idempotency).
//!   * `wants_reasoner` — route to the expensive `deepseek-reasoner` only
//!                      when difficulty genuinely warrants it.
//!
//! All thresholds are env-overridable so operators can tune without a rebuild.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

/// ~4 chars per token is the standard rough heuristic for English/code.
pub fn approx_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Max tokens allowed in an assembled prompt segment (graph context + system).
/// The user's own stimulus is budgeted separately and more generously.
pub fn max_prompt_tokens() -> usize {
    env_usize("DEEPSEEK_MAX_PROMPT_TOKENS", 2000)
}

/// Cap `s` to `max_tokens` worth of characters, preserving the head and
/// flagging the trim. Returns `s` untouched when already within budget.
pub fn budget_text(s: &str, max_tokens: usize) -> String {
    if approx_tokens(s) <= max_tokens {
        return s.to_string();
    }
    let max_chars = max_tokens.saturating_mul(4);
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("\n…[context trimmed to fit token budget]");
    out
}

/// Convenience: budget a prompt segment to the configured ceiling.
pub fn budget_prompt(s: &str) -> String {
    budget_text(s, max_prompt_tokens())
}

// ── Prompt-hash dedup / idempotency ──────────────────────────────

/// Time-to-live for a cached completion, in seconds.
fn dedup_ttl_secs() -> f64 {
    env_f64("RGW_DEDUP_TTL_SECS", 60.0)
}

lazy_static::lazy_static! {
    /// prompt_hash → (unix_secs_stored, completion_text)
    static ref DEDUP: DashMap<u64, (f64, String)> = DashMap::new();
}

fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Stable hash of everything that makes a completion request distinct.
pub fn prompt_hash(model: &str, system: &str, user: &str, max_tokens: Option<u32>) -> u64 {
    let mut h = DefaultHasher::new();
    model.hash(&mut h);
    system.hash(&mut h);
    user.hash(&mut h);
    max_tokens.hash(&mut h);
    h.finish()
}

/// Return a cached completion for this prompt if one was stored within the TTL.
/// A hit means an identical call can be skipped entirely.
pub fn dedup_lookup(hash: u64) -> Option<String> {
    let ttl = dedup_ttl_secs();
    if ttl <= 0.0 {
        return None; // dedup disabled
    }
    if let Some(entry) = DEDUP.get(&hash) {
        let (stored_at, text) = entry.value();
        if now_secs() - stored_at < ttl {
            return Some(text.clone());
        }
    }
    None
}

/// Store a completion so identical near-future prompts can reuse it.
pub fn dedup_store(hash: u64, text: &str) {
    if dedup_ttl_secs() <= 0.0 || text.is_empty() {
        return;
    }
    DEDUP.insert(hash, (now_secs(), text.to_string()));
    // Opportunistic eviction so the map can't grow without bound.
    if DEDUP.len() > 512 {
        let cutoff = now_secs() - dedup_ttl_secs();
        DEDUP.retain(|_, (ts, _)| *ts >= cutoff);
    }
}

// ── Model routing: chat vs reasoner ──────────────────────────────

/// Complexity at/above which we escalate to `deepseek-reasoner`.
pub fn reasoner_complexity_threshold() -> f32 {
    env_f32("RGW_REASONER_COMPLEXITY", 0.85)
}

/// Decide whether a task is hard enough to justify the reasoner.
/// `difficult` is a hard signal from metacog (e.g. stuck/break-loop) that
/// forces escalation regardless of the numeric complexity score.
pub fn wants_reasoner(complexity: f32, difficult: bool) -> bool {
    difficult || complexity >= reasoner_complexity_threshold()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_leaves_short_text_untouched() {
        let s = "hello world";
        assert_eq!(budget_text(s, 1000), s);
    }

    #[test]
    fn budget_trims_long_text() {
        let s = "x".repeat(10_000); // ~2500 tokens
        let out = budget_text(&s, 100); // cap ~400 chars
        assert!(out.len() < s.len());
        assert!(out.contains("context trimmed"));
        assert!(approx_tokens(&out) <= 120);
    }

    #[test]
    fn identical_prompts_hash_equal() {
        let a = prompt_hash("deepseek-chat", "sys", "user", Some(256));
        let b = prompt_hash("deepseek-chat", "sys", "user", Some(256));
        let c = prompt_hash("deepseek-chat", "sys", "user2", Some(256));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn dedup_roundtrips_within_ttl() {
        let h = prompt_hash("m", "s", "u-unique-test-key", Some(10));
        assert!(dedup_lookup(h).is_none());
        dedup_store(h, "cached answer");
        assert_eq!(dedup_lookup(h).as_deref(), Some("cached answer"));
    }

    #[test]
    fn reasoner_only_when_hard() {
        assert!(!wants_reasoner(0.5, false));
        assert!(wants_reasoner(0.5, true));   // forced by difficulty signal
        assert!(wants_reasoner(0.95, false)); // forced by high complexity
    }
}
