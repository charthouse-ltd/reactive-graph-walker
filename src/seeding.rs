//! Query-derived seeding: seed mode, pure seed planning, walk ids.
//! Spec: docs/superpowers/specs/2026-11-05-learning-walkers-design.md §5.2.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Nearest nodes that take part in the plan at full weight.
pub const SEED_TOP_K: usize = 5;
/// Further nearest nodes that take part at `SEED_WIDE_WEIGHT`.
pub const SEED_WIDE_M: usize = 20;
pub const SEED_WIDE_WEIGHT: f32 = 0.05;
/// Below this best cosine a query-mode walk does not run.
pub const SEED_FLOOR: f32 = 0.35;
/// Probability, per step, of restarting at a seed.
pub const RESTART_PROB: f32 = 0.5;
/// Merge mass discount per hop since the last (re)start.
pub const MERGE_DISCOUNT: f32 = 0.5;
/// Shorter text (after trimming) is not a cue.
pub const MIN_CUE_CHARS: usize = 3;

/// How query walks are seeded (`RGW_SEED_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMode {
    /// Today's importance/access-count seeds.
    Fixed,
    /// Walk as `Fixed`; compute the query plan for statistics only.
    Shadow,
    /// Seed from the query.
    Query,
}

impl SeedMode {
    /// `(mode, recognised)`. Unknown or empty input is `(Shadow, false)`.
    pub fn parse(s: &str) -> (SeedMode, bool) {
        match s.trim().to_ascii_lowercase().as_str() {
            "fixed" => (SeedMode::Fixed, true),
            "shadow" => (SeedMode::Shadow, true),
            "query" => (SeedMode::Query, true),
            _ => (SeedMode::Shadow, false),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SeedMode::Fixed => "fixed",
            SeedMode::Shadow => "shadow",
            SeedMode::Query => "query",
        }
    }
}

/// The process-wide seed mode, read once from `RGW_SEED_MODE`.
pub fn seed_mode() -> SeedMode {
    static MODE: OnceLock<SeedMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("RGW_SEED_MODE").unwrap_or_default();
        let (mode, recognised) = SeedMode::parse(&raw);
        if !recognised && !raw.trim().is_empty() {
            tracing::warn!("[seeding] RGW_SEED_MODE is not one of fixed|shadow|query; using shadow");
        }
        mode
    })
}

/// `RGW_EVAL == "1"`: eval arms and eval-only telemetry are allowed.
pub fn eval_enabled() -> bool {
    std::env::var("RGW_EVAL").map(|v| v == "1").unwrap_or(false)
}

/// `RGW_DETERMINISTIC == "1"`: every walk is deterministic (base seed 0 unless the request gives one).
pub fn deterministic_env() -> bool {
    std::env::var("RGW_DETERMINISTIC").map(|v| v == "1").unwrap_or(false)
}

/// A node near the cue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: i32,
    pub cos: f32,
    pub degree: u32,
}

/// How a node's degree discounts its seed weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specificity {
    /// `1 / (1 + ln(1 + degree))`: damps hubs without removing them (default).
    Log,
    /// `1 / (1 + degree)`: eval arm only.
    Linear,
}

/// Weighted seeds for one walk.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SeedPlan {
    /// `(node id, weight)`, weight descending then id ascending; weights sum to 1.
    pub seeds: Vec<(i32, f32)>,
    pub best_cos: f32,
    pub below_floor: bool,
}

pub fn specificity(kind: Specificity, degree: u32) -> f32 {
    match kind {
        Specificity::Log => 1.0 / (1.0 + (1.0 + degree as f32).ln()),
        Specificity::Linear => 1.0 / (1.0 + degree as f32),
    }
}

/// Turn similarity candidates into a seed plan (pure).
pub fn plan_seeds(cands: &[Candidate], kind: Specificity) -> SeedPlan {
    let best_cos = cands.iter().map(|c| c.cos).fold(0.0f32, f32::max);
    let mut sorted: Vec<&Candidate> = cands.iter().filter(|c| c.cos > 0.0).collect();
    sorted.sort_by(|a, b| b.cos.partial_cmp(&a.cos).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id)));

    let mut seeds: Vec<(i32, f32)> = sorted
        .iter()
        .take(SEED_TOP_K + SEED_WIDE_M)
        .enumerate()
        .map(|(rank, c)| {
            let w = if rank < SEED_TOP_K {
                c.cos * specificity(kind, c.degree)
            } else {
                SEED_WIDE_WEIGHT * c.cos
            };
            (c.id, w)
        })
        .collect();
    let total: f32 = seeds.iter().map(|s| s.1).sum();
    if total > 0.0 {
        for s in &mut seeds {
            s.1 /= total;
        }
    } else {
        seeds.clear();
    }
    seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));

    SeedPlan { seeds, best_cos, below_floor: best_cos < SEED_FLOOR }
}

/// Start node of walker `walker_index` in deterministic mode: seeds by rank, round-robin.
/// Callers only use a non-empty plan; an empty one yields node 0.
pub fn start_seed(plan: &SeedPlan, walker_index: usize) -> i32 {
    if plan.seeds.is_empty() {
        return 0;
    }
    plan.seeds[walker_index % plan.seeds.len()].0
}

/// Weighted draw from `plan.seeds`. An empty plan yields node 0.
pub fn draw_seed<R: rand::Rng>(plan: &SeedPlan, rng: &mut R) -> i32 {
    let total: f32 = plan.seeds.iter().map(|s| s.1).sum();
    let threshold = rng.random::<f32>() * total;
    let mut cumulative = 0.0;
    for (id, w) in &plan.seeds {
        cumulative += w;
        if cumulative >= threshold {
            return *id;
        }
    }
    plan.seeds.last().map(|s| s.0).unwrap_or(0)
}

/// A fresh walk id: 32 lowercase hex characters (128 random bits).
pub fn new_walk_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    format!("{:016x}{:016x}", rng.random::<u64>(), rng.random::<u64>())
}

/// The `SEED_TOP_K + SEED_WIDE_M` nodes nearest to the cue, with their degrees.
pub async fn candidates(pool: &sqlx::PgPool, cue_embedding: &[f32]) -> Result<Vec<Candidate>, sqlx::Error> {
    let near = crate::db::similar_node_ids(pool, cue_embedding, (SEED_TOP_K + SEED_WIDE_M) as i64).await?;
    let ids: Vec<i32> = near.iter().map(|n| n.0).collect();
    let degrees = crate::db::node_degrees(pool, &ids).await?;
    Ok(near
        .into_iter()
        .map(|(id, cos)| Candidate { id, cos, degree: degrees.get(&id).copied().unwrap_or(0) })
        .collect())
}

/// Embed the query text as a walk cue. `None` for text shorter than `MIN_CUE_CHARS`
/// after trimming, or when embedding fails (one WARN, without the text). The embedder
/// is CPU-bound, so it runs on the blocking pool.
pub async fn embed_cue(text: &str) -> Option<Vec<f32>> {
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_CUE_CHARS {
        return None;
    }
    let owned = trimmed.to_string();
    match tokio::task::spawn_blocking(move || crate::embed::embed_text(&owned)).await {
        Ok(Ok(embedding)) => Some(embedding),
        Ok(Err(e)) => {
            tracing::warn!("[seeding] cue embedding failed, walking with fixed seeds: {}", e);
            None
        }
        Err(e) => {
            tracing::warn!("[seeding] cue embedding task failed, walking with fixed seeds: {}", e);
            None
        }
    }
}

/// How many recent observations the medians are taken over.
const SEED_STATS_RING: usize = 512;

/// What the query seed plans looked like, walk by walk (spec §13, `seed` block).
#[derive(Debug, Default)]
pub struct SeedStats {
    walks: AtomicU64,
    fallbacks: AtomicU64,
    best_cos: Mutex<VecDeque<f32>>,
    overlap: Mutex<VecDeque<f32>>,
}

impl SeedStats {
    /// Record one plan: its best cosine, the Jaccard overlap of its first `SEED_TOP_K`
    /// ids with the fixed seeds, and whether it was (or would have been) below the floor.
    pub fn record(&self, best_cos: f32, overlap: f32, fell_back: bool) {
        self.walks.fetch_add(1, Ordering::Relaxed);
        if fell_back {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        for (ring, value) in [(&self.best_cos, best_cos), (&self.overlap, overlap)] {
            let mut ring = ring.lock().unwrap_or_else(|e| e.into_inner());
            ring.push_back(value);
            while ring.len() > SEED_STATS_RING {
                ring.pop_front();
            }
        }
    }

    pub fn to_json(&self, mode: SeedMode) -> serde_json::Value {
        let median = |ring: &Mutex<VecDeque<f32>>| -> serde_json::Value {
            let mut values: Vec<f32> = ring.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect();
            if values.is_empty() {
                return serde_json::Value::Null;
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            serde_json::json!(values[values.len() / 2])
        };
        serde_json::json!({
            "mode": mode.as_str(),
            "walks": self.walks.load(Ordering::Relaxed),
            "fallbacks": self.fallbacks.load(Ordering::Relaxed),
            "best_cos_p50": median(&self.best_cos),
            "overlap_with_fixed_p50": median(&self.overlap),
        })
    }
}

fn process_stats() -> &'static SeedStats {
    static STATS: OnceLock<SeedStats> = OnceLock::new();
    STATS.get_or_init(SeedStats::default)
}

/// Record a plan in the process-wide statistics. `Fixed` records nothing.
pub fn record_seed_observation(mode: SeedMode, best_cos: f32, overlap: f32, fell_back: bool) {
    if mode != SeedMode::Fixed {
        process_stats().record(best_cos, overlap, fell_back);
    }
}

/// The process-wide statistics as JSON, labelled with the configured mode.
pub fn seed_stats_json() -> serde_json::Value {
    process_stats().to_json(seed_mode())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(id: i32, cos: f32, degree: u32) -> Candidate { Candidate { id, cos, degree } }

    #[test]
    fn mode_parses_and_unknown_is_shadow() {
        assert_eq!(SeedMode::parse("fixed"), (SeedMode::Fixed, true));
        assert_eq!(SeedMode::parse("QUERY"), (SeedMode::Query, true));
        assert_eq!(SeedMode::parse("shadow"), (SeedMode::Shadow, true));
        assert_eq!(SeedMode::parse("banana"), (SeedMode::Shadow, false));
        assert_eq!(SeedMode::parse(""), (SeedMode::Shadow, false));
    }

    #[test]
    fn specificity_log_damps_hubs_without_removing_them() {
        assert!((specificity(Specificity::Log, 0) - 1.0).abs() < 1e-6);
        let hub = specificity(Specificity::Log, 200);
        assert!(hub > 0.15 && hub < 0.17, "1/(1+ln 201) = 0.159, got {hub}");
        assert!((specificity(Specificity::Linear, 200) - 1.0 / 201.0).abs() < 1e-6);
    }

    #[test]
    fn plan_weights_top_k_by_cos_times_specificity_and_the_rest_at_wide_weight() {
        // 6 candidates by descending cosine; the 6th is outside SEED_TOP_K.
        let cands = vec![c(1, 0.9, 0), c(2, 0.8, 0), c(3, 0.7, 0), c(4, 0.6, 0), c(5, 0.5, 0), c(6, 0.4, 0)];
        let plan = plan_seeds(&cands, Specificity::Log);
        let raw = [0.9, 0.8, 0.7, 0.6, 0.5, SEED_WIDE_WEIGHT * 0.4];
        let sum: f32 = raw.iter().sum();
        assert_eq!(plan.seeds.len(), 6);
        assert_eq!(plan.seeds[0].0, 1);
        assert!((plan.seeds[0].1 - 0.9 / sum).abs() < 1e-5);
        assert!((plan.seeds[5].1 - raw[5] / sum).abs() < 1e-5);
        assert!((plan.seeds.iter().map(|s| s.1).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(plan.best_cos, 0.9);
        assert!(!plan.below_floor);
    }

    #[test]
    fn a_hub_with_the_best_cosine_can_lose_first_place() {
        let plan = plan_seeds(&[c(1, 0.90, 200), c(2, 0.80, 2)], Specificity::Log);
        assert_eq!(plan.seeds[0].0, 2, "0.80/(1+ln 3)=0.381 beats 0.90*0.159=0.143");
    }

    #[test]
    fn below_floor_and_empty_are_flagged() {
        assert!(plan_seeds(&[c(1, SEED_FLOOR - 0.01, 0)], Specificity::Log).below_floor);
        assert!(!plan_seeds(&[c(1, SEED_FLOOR, 0)], Specificity::Log).below_floor);
        let empty = plan_seeds(&[], Specificity::Log);
        assert!(empty.below_floor && empty.seeds.is_empty() && empty.best_cos == 0.0);
    }

    #[test]
    fn negative_cosines_get_no_weight() {
        let plan = plan_seeds(&[c(1, 0.6, 0), c(2, -0.2, 0)], Specificity::Log);
        assert_eq!(plan.seeds, vec![(1, 1.0)]);
    }

    #[test]
    fn start_seed_is_round_robin_by_rank() {
        let plan = plan_seeds(&[c(7, 0.9, 0), c(8, 0.8, 0)], Specificity::Log);
        assert_eq!([start_seed(&plan, 0), start_seed(&plan, 1), start_seed(&plan, 2)], [7, 8, 7]);
    }

    #[test]
    fn draw_seed_follows_the_weights() {
        use rand::SeedableRng;
        let plan = SeedPlan { seeds: vec![(1, 0.9), (2, 0.1)], best_cos: 0.9, below_floor: false };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let ones = (0..2000).filter(|_| draw_seed(&plan, &mut rng) == 1).count();
        assert!((1700..=1900).contains(&ones), "expected about 1800, got {ones}");
    }

    #[test]
    fn walk_ids_are_32_hex_and_distinct() {
        let (a, b) = (new_walk_id(), new_walk_id());
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn short_or_blank_text_is_no_cue() {
        assert!(embed_cue("").await.is_none());
        assert!(embed_cue("   ").await.is_none());
        assert!(embed_cue("hi").await.is_none());
    }

    #[test]
    fn seed_stats_report_medians_and_counts() {
        // Uses a local SeedStats value, not the process-wide one, so tests do not interfere.
        let s = SeedStats::default();
        for (cos, ov, fb) in [(0.2, 0.0, true), (0.6, 0.2, false), (0.8, 0.4, false)] { s.record(cos, ov, fb); }
        let j = s.to_json(SeedMode::Shadow);
        assert_eq!((j["walks"].as_u64(), j["fallbacks"].as_u64()), (Some(3), Some(1)));
        assert!((j["best_cos_p50"].as_f64().unwrap() - 0.6).abs() < 1e-6);
        assert!((j["overlap_with_fixed_p50"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert!(SeedStats::default().to_json(SeedMode::Shadow)["best_cos_p50"].is_null());
    }
}
