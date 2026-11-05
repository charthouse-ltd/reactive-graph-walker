# Learning Walkers P1 (Measurement and Query Seeding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a walk's result depend on the query, and make that measurable: query-derived seeds behind `RGW_SEED_MODE`, a deterministic mode, eval arms, and a gold-node recall harness that judges gates G1 and G1b.

**Architecture:** A new pure module `seeding` turns similarity candidates into a weighted seed plan. `walker` gains `WalkOptions` and `walk_parallel_with`; when a seed plan is active the walk restarts to seeds, divides scores by fan-out, and no longer treats a cache miss as a dead end. Aggregation adds a depth-discounted merge mass and telemetry. Everything new is off the default path: the default mode `shadow` walks exactly as today and only measures.

**Tech Stack:** Rust 2024 (MSRV 1.88), tokio, rayon, sqlx/pgvector, `rand` 0.9 (`StdRng`), existing `embed::embed_text`. No new crates.

**Spec:** `docs/superpowers/specs/2026-11-05-learning-walkers-design.md`, §5 (Stages 0 and 1), §11, §12, §13 (`seed` block), §14 (G1, G1b), §17. Read §5 before Task 1.

## Global Constraints

- `rust-version = "1.88"` keeps building; no new dependencies.
- A walk with no cue, and every walk under `RGW_SEED_MODE=fixed`, behaves exactly as today apart from added output fields. `shadow` (the default) also walks exactly as today.
- New environment variables are only `RGW_SEED_MODE` (`fixed|shadow|query`, default `shadow`), `RGW_DETERMINISTIC` (`0|1`), `RGW_EVAL` (`0|1`). All other parameters are `pub const` with the exact spec values: `SEED_TOP_K = 5`, `SEED_WIDE_M = 20`, `SEED_WIDE_WEIGHT = 0.05`, `SEED_FLOOR = 0.35`, `RESTART_PROB = 0.5`, merge depth discount `0.5`.
- No schema change and no new table in this plan. The walk log is P2.
- No log line and no stored value contains cue text. Logs carry ids, cosines and counts.
- New `WalkOutput` and `WalkerResult` fields are additive; existing fields keep their meaning and are still filled.
- Every task ends with `cargo test --locked --lib --bins` green and `cargo clippy --quiet` not increasing the warning count. Integration tests need the local Postgres at `postgresql:///rgw_test?host=/tmp` (override with `DATABASE_URL`); CI does not run them.
- The repo is public. Gold sets and eval results are never committed. `gitleaks detect --log-opts=origin/main..HEAD` before any push.

## Review Focus

1. `stimulus` empty, whitespace, or shorter than 3 characters → no cue; the walk runs with fixed seeds and returns 200. (Task 5)
2. Embedder not initialised, or embedding fails → fixed-seed walk plus one WARN, never a 5xx. (Task 5)
3. No candidates at all (empty graph, or no node has an embedding) → `below_floor = true`; in `query` mode the output is well formed with `walked = false` and empty `ranked_nodes`. (Tasks 1, 4)
4. Cue embedding has a different dimension from the stored vectors → the similarity query errors → fixed-seed walk plus one WARN. (Tasks 2, 5)
5. `RGW_SEED_MODE=banana` → `shadow` with one WARN, never a panic. (Task 1)
6. `arm` sent without `RGW_EVAL=1`, or an unknown `arm` → 400 with a clear message, no walk. (Task 5)
7. A node two hops from a seed whose edges are not cached → not a dead end when a seed plan is active: no `dead_end` signal, no dead-end mark in the collective, the walker restarts. (Task 3)
8. After a restart, `path` is not a chain: `edges_traversed.len() == path.len() - segment_starts.len()`, and every hop count is taken from `edges_traversed`. (Task 3)
9. Two runs with the same `seed` differ because signal processing reads the clock → the determinism test fails; see Task 3 Step 7 for the ruling. (Task 3)

## File Structure

| File | Responsibility |
|---|---|
| `src/seeding.rs` (new) | seed mode, constants, pure seed planning, cue embedding, shadow statistics |
| `src/evalkit.rs` (new) | pure recall metrics, per-category summary, G1/G1b verdict |
| `src/walker.rs` | `WalkOptions`, `EvalArm`, `walk_parallel_with`, restart and fan division, merge mass, telemetry, PPR arm, narrative section |
| `src/graph.rs` | new fields on `WalkerResult` and `WalkOutput`, `StepStats`, `WalkTelemetry` |
| `src/db.rs` | `similar_node_ids`, `node_degrees`, `degree_at_percentile`; stable tie-break in `edges_from` |
| `src/edge_cache.rs` | `is_loaded` |
| `src/api.rs`, `src/openai.rs` | cue from `stimulus`, `seed`, `arm`, `GET /learning` |
| `tests/common/fixture.rs` (new) | schema-isolated pool and a small graph with synthetic embeddings |
| `tests/seeding_walk.rs` (new) | integration tests |
| `examples/recall_eval.rs` (new) | the harness |

---

### Task 1: Pure seeding core

**Files:**
- Create: `src/seeding.rs`
- Modify: `src/lib.rs` and `src/main.rs` (add `pub mod seeding;` / `mod seeding;`)
- Test: unit tests in `src/seeding.rs`

**Interfaces:**
- Produces:

```rust
pub const SEED_TOP_K: usize = 5;
pub const SEED_WIDE_M: usize = 20;
pub const SEED_WIDE_WEIGHT: f32 = 0.05;
pub const SEED_FLOOR: f32 = 0.35;
pub const RESTART_PROB: f32 = 0.5;
pub const MERGE_DISCOUNT: f32 = 0.5;
pub const MIN_CUE_CHARS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMode { Fixed, Shadow, Query }
impl SeedMode {
    /// (mode, recognised). Unknown or empty input → (Shadow, false).
    pub fn parse(s: &str) -> (SeedMode, bool);
    pub fn as_str(&self) -> &'static str;      // "fixed" | "shadow" | "query"
}
/// Reads RGW_SEED_MODE once per process; one WARN if set and unrecognised.
pub fn seed_mode() -> SeedMode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate { pub id: i32, pub cos: f32, pub degree: u32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specificity { Log, Linear }

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SeedPlan {
    /// (node id, weight), weight descending then id ascending; weights sum to 1.
    pub seeds: Vec<(i32, f32)>,
    pub best_cos: f32,
    pub below_floor: bool,
}

pub fn specificity(kind: Specificity, degree: u32) -> f32;
pub fn plan_seeds(cands: &[Candidate], kind: Specificity) -> SeedPlan;
/// Start node of walker `walker_index` in deterministic mode: seeds by rank, round-robin.
pub fn start_seed(plan: &SeedPlan, walker_index: usize) -> i32;
/// Weighted draw from `plan.seeds`.
pub fn draw_seed<R: rand::Rng>(plan: &SeedPlan, rng: &mut R) -> i32;
pub fn new_walk_id() -> String;                // 32 lowercase hex characters
pub fn eval_enabled() -> bool;                 // RGW_EVAL == "1"
pub fn deterministic_env() -> bool;            // RGW_DETERMINISTIC == "1"
```

- [ ] **Step 1: Write the failing tests**

```rust
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
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib seeding::`
Expected: compile errors for the missing items.

- [ ] **Step 3: Implement the items of the Interfaces block**

`plan_seeds`: sort candidates by cosine descending then id ascending; drop cosines ≤ 0; the first `SEED_TOP_K` get `cos * specificity(kind, degree)`, the next `SEED_WIDE_M` get `SEED_WIDE_WEIGHT * cos`, anything further is ignored; normalise to sum 1; sort the result by weight descending then id ascending. `best_cos` is the largest input cosine (0.0 when empty). `below_floor = best_cos < SEED_FLOOR`. `specificity(Log, d) = 1 / (1 + ln(1 + d))`, `specificity(Linear, d) = 1 / (1 + d)`. `seed_mode()` uses a `OnceLock`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib seeding::`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/seeding.rs src/lib.rs src/main.rs
git commit -m "feat(seeding): pure seed planning, seed mode, walk ids"
```

---

### Task 2: Similarity and degree queries, and an isolated test fixture

**Files:**
- Modify: `src/db.rs` (three queries; tie-break in `edges_from`), `src/seeding.rs` (`candidates`)
- Create: `tests/common/fixture.rs`, `tests/seeding_walk.rs`
- Modify: `tests/common/mod.rs` (add `pub mod fixture;`)

The existing `seed_database` helper inserts through `db::create_memory_node`, which writes a `document` column that `tests/schema.sql` does not have, and the shared local test database already holds other tests' nodes. New integration tests therefore use their own Postgres schema and insert with plain SQL.

**Interfaces:**
- Produces:

```rust
// src/db.rs
/// (id, cosine similarity), most similar first, ties by id. Selects only `id`, so it works on
/// the backends' schema and the test schema alike.
pub async fn similar_node_ids(pool: &PgPool, embedding: &[f32], limit: i64) -> Result<Vec<(i32, f32)>, sqlx::Error>;
/// Number of edges touching each id (0 for an id with none).
pub async fn node_degrees(pool: &PgPool, ids: &[i32]) -> Result<std::collections::HashMap<i32, u32>, sqlx::Error>;
/// Degree at percentile `pct` (0..1) over all nodes that have an edge; 0 for an empty graph.
pub async fn degree_at_percentile(pool: &PgPool, pct: f64) -> Result<i64, sqlx::Error>;

// src/seeding.rs
/// SEED_TOP_K + SEED_WIDE_M nearest nodes with their degrees.
pub async fn candidates(pool: &sqlx::PgPool, cue_embedding: &[f32]) -> Result<Vec<Candidate>, sqlx::Error>;

// tests/common/fixture.rs
pub const DIM: usize = 768;
/// Unit vector along `axis`.
pub fn axis(axis: usize) -> Vec<f32>;
/// Normalised weighted sum of axes.
pub fn mix(parts: &[(usize, f32)]) -> Vec<f32>;
/// A pool whose connections use `search_path = <schema>, public`. Drops and recreates the
/// schema, runs tests/schema.sql inside it, and inserts the graph below.
pub async fn fixture_pool(schema: &str) -> anyhow::Result<sqlx::PgPool>;
pub const HUB: i32 = 1; pub const A1: i32 = 2; pub const A2: i32 = 3; pub const A3: i32 = 4;
pub const B1: i32 = 5; pub const B2: i32 = 6; pub const B3: i32 = 7; pub const B4: i32 = 8;
pub const LONE: i32 = 9;
```

The fixture graph (explicit ids; `importance`, `access_count` decide today's fixed seeds, which are `HUB` then `A1`):

| id | domain | importance | access_count | embedding |
|---|---|---|---|---|
| 1 HUB | hub | 9.0 | 100 | `mix(&[(0, 0.5), (1, 0.5)])` |
| 2 A1 | alpha | 9.0 | 90 | `axis(0)` |
| 3 A2 | alpha | 3.0 | 0 | `mix(&[(0, 0.9), (3, 0.1)])` |
| 4 A3 | alpha | 3.0 | 0 | `mix(&[(0, 0.8), (3, 0.2)])` |
| 5 B1 | beta | 3.0 | 0 | `axis(1)` |
| 6 B2 | beta | 3.0 | 0 | `mix(&[(1, 0.9), (4, 0.1)])` |
| 7 B3 | beta | 3.0 | 0 | `mix(&[(1, 0.8), (4, 0.2)])` |
| 8 B4 | beta | 3.0 | 0 | `mix(&[(1, 0.7), (4, 0.3)])` |
| 9 LONE | gamma | 3.0 | 0 | `axis(2)` |

Edges, all `related` except where noted, `(source, target, weight)`: `(1,2,0.9) (1,3,0.5) (1,4,0.5) (1,5,0.3) (2,3,0.8) (3,4,0.7) (5,6,0.8) (6,7,0.8) (7,8,0.8) (5,7,0.4 reinforces) (9,4,0.2)`.

- [ ] **Step 1: Write the fixture and the failing integration test**

`fixture_pool` connects with `PgPoolOptions::after_connect` running `SET search_path TO <schema>, public`, so the unqualified table names in `src/` resolve inside the schema. Insert nodes with `INSERT INTO memory_vectors (id, content, domain, embedding, importance, access_count) VALUES ...` and edges with `INSERT INTO memory_edges (source_id, target_id, edge_type, weight)`.

```rust
// tests/seeding_walk.rs
mod common;
use common::fixture::*;

#[tokio::test]
async fn candidates_are_ranked_by_cosine_with_degrees() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_candidates").await?;
    let cands = rgw::seeding::candidates(&pool, &axis(1)).await?;
    let ids: Vec<i32> = cands.iter().take(5).map(|c| c.id).collect();
    assert_eq!(ids, vec![B1, B2, B3, B4, HUB]);
    assert!((cands[0].cos - 1.0).abs() < 1e-4);
    assert!((cands[4].cos - 0.7071).abs() < 1e-3);
    let deg = |id: i32| cands.iter().find(|c| c.id == id).unwrap().degree;
    assert_eq!((deg(B1), deg(HUB), deg(B4)), (3, 4, 1));
    Ok(())
}

#[tokio::test]
async fn degrees_and_percentile() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_degrees").await?;
    let d = rgw::db::node_degrees(&pool, &[HUB, LONE, 999]).await?;
    assert_eq!((d[&HUB], d[&LONE], d[&999]), (4, 1, 0));
    assert_eq!(rgw::db::degree_at_percentile(&pool, 1.0).await?, 4);
    Ok(())
}

#[tokio::test]
async fn a_wrong_dimension_is_an_error_not_a_panic() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_dim").await?;
    assert!(rgw::seeding::candidates(&pool, &[1.0, 0.0, 0.0]).await.is_err());
    Ok(())
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --test seeding_walk`
Expected: compile errors for the missing functions.

- [ ] **Step 3: Implement the three queries, `candidates`, and the tie-break**

`similar_node_ids`: `SELECT id, (1.0 - (embedding <=> $1::vector))::real FROM memory_vectors WHERE embedding IS NOT NULL ORDER BY embedding <=> $1::vector, id LIMIT $2`, with the embedding bound as the same `'[..]'` string `find_similar_nodes` builds. `node_degrees`: `unnest($1::int4[])` left-joined to `memory_edges` on `source_id = id OR target_id = id`, grouped by id. `degree_at_percentile`: `percentile_disc($1) WITHIN GROUP (ORDER BY d)` over per-node edge counts from `source_id UNION ALL target_id`, `COALESCE(..., 0)`. In `edges_from`, change `ORDER BY weight DESC` to `ORDER BY weight DESC, id ASC`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --test seeding_walk && cargo test --locked --lib --bins`
Expected: 3 integration tests pass; unit suite green.

- [ ] **Step 5: Commit**

```bash
git add src/db.rs src/seeding.rs tests/common tests/seeding_walk.rs
git commit -m "feat(seeding): similarity and degree queries, isolated graph fixture"
```

---

### Task 3: Walk mechanics under a seed plan

**Files:**
- Modify: `src/graph.rs` (`WalkerResult`, `StepStats`, `fitness_inputs`), `src/edge_cache.rs` (`is_loaded`), `src/walker.rs` (`WalkOptions`, `EvalArm`, `walk_single`, `walk_parallel_with`, `compute_resonance`)
- Test: unit tests in `src/walker.rs` and `src/graph.rs`; `tests/seeding_walk.rs`

**Interfaces:**
- Consumes: `seeding::{SeedPlan, SeedMode, candidates, plan_seeds, start_seed, draw_seed, RESTART_PROB, Specificity}` (Tasks 1, 2).
- Produces:

```rust
// src/graph.rs
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StepStats { pub branching_sum: u32, pub steps: u32, pub cache_miss_stops: u32 }

pub struct WalkerResult {
    /* existing fields unchanged */
    /// Depth of path[i] since the last (re)start. Empty in results built without it.
    pub depths: Vec<u8>,
    /// Indices into `path` where a segment starts. Empty means one segment starting at 0.
    pub segment_starts: Vec<usize>,
    pub step_stats: StepStats,
}
impl WalkerResult {
    pub fn hops(&self) -> usize;                    // edges_traversed.len()
    pub fn depth_at(&self, i: usize) -> u8;         // depths[i], or i (saturating at 255) when depths is empty
    pub fn segments(&self) -> Vec<&[i32]>;          // path split at segment_starts
}

// src/edge_cache.rs
pub fn is_loaded(&self, node_id: i32) -> bool;

// src/walker.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalArm { Fixed, Query, QueryLinear, Vector, Ppr, Drop(WalkerBias) }
// "fixed" | "query" | "query:linear" | "vector" | "ppr" | "drop:<bias>" (bias as serde names it, lowercase)
impl EvalArm { pub fn parse(s: &str) -> Option<EvalArm>; }

#[derive(Debug, Clone, Default)]
pub struct WalkOptions {
    pub cue_embedding: Option<Vec<f32>>,
    /// Some → deterministic walk with this base seed. None → deterministic only if RGW_DETERMINISTIC=1 (seed 0).
    pub seed: Option<u64>,
    pub arm: Option<EvalArm>,
    /// None → seeding::seed_mode().
    pub seed_mode: Option<SeedMode>,
}

pub fn walker_rng_seed(base: u64, walker_index: usize) -> u64;
/// Divide each score by the number of candidates that share its edge type.
pub fn fan_divide(scored: &mut [(&str, f32)]);
pub async fn walk_parallel_with(pool: &PgPool, emotion: &EmotionalState, n_walkers: usize, steps: usize,
    self_model: &Arc<tokio::sync::Mutex<SelfModel>>, opts: WalkOptions) -> (WalkOutput, Vec<WalkerResult>);
// walk_parallel keeps its signature and calls walk_parallel_with(.., WalkOptions::default()).
```

**Which walk runs** (decided once at the top of `walk_parallel_with`):

| Condition | Seeds | Stage 1 walk rules |
|---|---|---|
| `arm = Fixed`, or no `cue_embedding`, or mode `Fixed` | `db::seed_nodes` as today | off |
| mode `Shadow` (no arm) | as today; the plan is computed for statistics only (Task 5) | off |
| `arm = Query`, `QueryLinear` or `Drop(_)`, or mode `Query` (no arm) | the seed plan | on |
| `arm = Vector`, or a plan with `below_floor` under `Query` | no walk (Task 4) | |
| `arm = Ppr` | the seed plan, PPR push instead of walkers (Task 5) | |

A failed `candidates` call logs one WARN (ids and counts only) and falls back to the first row.
The plan uses `Specificity::Log`, except under `QueryLinear`, which uses `Specificity::Linear`.
Under `Drop(b)` walkers are assigned biases from `WalkerBias::all()` without `b`, in the same
round-robin as today; everything else is as under `Query`.

**Stage 1 walk rules** (only when "on"):
- Start: `start_seed(plan, i)` in deterministic mode, otherwise `draw_seed(plan, rng)`.
- Before each step, with probability `RESTART_PROB` (drawn from the walker's RNG): restart. A restart pushes a seed drawn with `draw_seed` onto `path` with depth 0, records its index in `segment_starts`, clears `prev_domain`, and uses up the step.
- Scores are passed through `fan_divide` before the weighted draw.
- `edges.is_empty()` and `!cache.is_loaded(current)`: increment `step_stats.cache_miss_stops`, restart, and record neither a dead end nor a `dead_end` signal nor a collective dead-end mark. A loaded node with no edges is a true dead end as today, followed by a restart instead of `break`.
- Preload: `cache.preload_radius` over the first `SEED_TOP_K` plan seeds, `cache.preload_batch` over the remaining plan seeds.

**Always** (rules on or off): `step_stats.steps` counts steps taken; `step_stats.branching_sum` adds, per step, the number of candidate edges whose weight is at least 0.1 of the largest candidate weight; `depths` is filled. In deterministic mode walkers run sequentially in index order (plain iterator in place of `par_iter`), each with `StdRng::seed_from_u64(walker_rng_seed(base, i))`, and candidate edges are sorted by edge id before scoring.

- [ ] **Step 1: Write the failing unit tests**

```rust
// src/walker.rs tests
#[test]
fn eval_arm_parses() {
    assert_eq!(EvalArm::parse("fixed"), Some(EvalArm::Fixed));
    assert_eq!(EvalArm::parse("ppr"), Some(EvalArm::Ppr));
    assert_eq!(EvalArm::parse("query:linear"), Some(EvalArm::QueryLinear));
    assert_eq!(EvalArm::parse("drop:contrarian"), Some(EvalArm::Drop(WalkerBias::Contrarian)));
    assert_eq!(EvalArm::parse("drop:nobody"), None);
    assert_eq!(EvalArm::parse("QUERY"), None, "arms are lowercase only");
}

#[test]
fn fan_divide_splits_by_edge_type() {
    let mut s = vec![("similar", 0.6), ("similar", 0.3), ("causes", 0.5)];
    fan_divide(&mut s);
    assert_eq!(s, vec![("similar", 0.3), ("similar", 0.15), ("causes", 0.5)]);
}

#[test]
fn walker_rng_seeds_differ_per_walker_and_repeat_per_base() {
    assert_ne!(walker_rng_seed(1, 0), walker_rng_seed(1, 1));
    assert_ne!(walker_rng_seed(1, 0), walker_rng_seed(2, 0));
    assert_eq!(walker_rng_seed(5, 3), walker_rng_seed(5, 3));
}

// src/graph.rs tests
#[test]
fn hops_depths_and_segments_survive_a_restart() {
    let r = WalkerResult {
        path: vec![10, 11, 12, 20, 21], edges_traversed: vec![1, 2, 3],
        depths: vec![0, 1, 2, 0, 1], segment_starts: vec![0, 3], ..legacy_result(vec![])
    };
    assert_eq!(r.hops(), 3);
    assert_eq!(r.depth_at(3), 0);
    assert_eq!(r.segments(), vec![&[10, 11, 12][..], &[20, 21][..]]);
    let legacy = legacy_result(vec![1, 2, 3]);
    assert_eq!((legacy.depth_at(2), legacy.segments().len()), (2, 1));
}
```

`legacy_result(path)` is a test helper that builds a `WalkerResult` with empty `depths` and `segment_starts`, `edges_traversed` of length `path.len() - 1` (0 for an empty path), and defaults elsewhere.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib eval_arm_parses fan_divide_splits walker_rng_seeds hops_depths`
Expected: compile errors. (Run each name separately if the filter rejects several.)

- [ ] **Step 3: Add the types, fields and pure helpers**

Add the three fields to `WalkerResult` and fix every struct literal the compiler reports (`src/walker.rs`, `src/graph.rs`, `src/metacog.rs`) with `depths: vec![], segment_starts: vec![], step_stats: StepStats::default()`. Replace `path.len().saturating_sub(1)` by `hops()` in `graph::fitness_inputs` and `walker::compute_resonance`. Leave `diverger.rs` alone: spontaneous walks never restart.

- [ ] **Step 4: Write the failing integration tests**

```rust
// tests/seeding_walk.rs
use rgw::walker::{walk_parallel_with, EvalArm, WalkOptions};
use std::sync::Arc;

fn sm() -> Arc<tokio::sync::Mutex<rgw::core::SelfModel>> {
    let mut m = rgw::core::SelfModel::new();
    m.mode = rgw::core::CognitiveMode::Autonomous;
    Arc::new(tokio::sync::Mutex::new(m))
}
fn opts(cue: Vec<f32>, arm: EvalArm) -> WalkOptions {
    WalkOptions { cue_embedding: Some(cue), seed: Some(1), arm: Some(arm), seed_mode: None }
}

#[tokio::test]
async fn query_arm_starts_in_the_cue_cluster_and_restarts_keep_the_invariant() -> anyhow::Result<()> {
    rgw::embed::init()?;   // signal processing embeds text
    let pool = fixture_pool("rgw_p1_query").await?;
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
    assert_eq!(results.len(), 4);
    let starts: Vec<i32> = results.iter().map(|r| r.path[0]).collect();
    // Plan order is cosine times specificity: B4 (degree 1) 0.543, B2 0.474, B1 0.419, B3 0.407, HUB 0.271.
    assert_eq!(starts, vec![B4, B2, B1, B3], "round-robin over the plan by rank");
    for r in &results {
        let segs = if r.segment_starts.is_empty() { 1 } else { r.segment_starts.len() };
        assert_eq!(r.edges_traversed.len(), r.path.len() - segs);
        assert_eq!(r.depths.len(), r.path.len());
        assert_eq!(r.dead_ends, 0, "B4 has one edge and its far side is cached; nothing here is a dead end");
    }
    Ok(())
}

#[tokio::test]
async fn fixed_arm_starts_where_it_does_today() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_fixed").await?;
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 2, 3, &sm(), opts(axis(1), EvalArm::Fixed)).await;
    assert_eq!(results.iter().map(|r| r.path[0]).collect::<Vec<_>>(), vec![HUB, A1]);
    assert!(results.iter().all(|r| r.segment_starts.len() <= 1));
    Ok(())
}

#[tokio::test]
async fn drop_arm_removes_one_walker_profile() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_drop").await?;
    let arm = EvalArm::Drop(rgw::graph::WalkerBias::Contrarian);
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 6, 2, &sm(), opts(axis(1), arm)).await;
    assert_eq!(results.len(), 6);
    assert!(results.iter().all(|r| r.bias != rgw::graph::WalkerBias::Contrarian));
    Ok(())
}

#[tokio::test]
async fn same_seed_gives_the_same_paths() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_determinism").await?;
    let run = || async {
        let (_o, r) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
        r.into_iter().map(|x| x.path).collect::<Vec<_>>()
    };
    assert_eq!(run().await, run().await);
    Ok(())
}
```

- [ ] **Step 5: Run to verify failure**

Run: `cargo test --test seeding_walk query_arm_starts`
Expected: compile error, `walk_parallel_with` not found.

- [ ] **Step 6: Implement `walk_parallel_with` and the walk rules**

Move the body of `walk_parallel` into `walk_parallel_with` and add the decision table above. `walk_single` gains two parameters, `plan: Option<&SeedPlan>` (rules on when `Some`) and `rng_seed: Option<u64>`, and uses one RNG type (`StdRng`) for both the seeded and the unseeded path. The `Vector` and `Ppr` rows are filled in by Tasks 4 and 5; until then they fall back to the first row.

- [ ] **Step 7: Run to verify pass**

Run: `cargo test --test seeding_walk && cargo test --locked --lib --bins`
Expected: all green.

**Ruling if `same_seed_gives_the_same_paths` fails:** signal processing reads the clock, and edge scoring reads the self-model it changes. Find the differing step by printing both path lists. If the cause is a time-dependent term, pass the walk's start time into `walk_single` and use it in place of fresh clock reads **on the deterministic path only**; do not change the non-deterministic path. Record what was frozen in the commit message.

- [ ] **Step 8: Commit**

```bash
git add src/graph.rs src/edge_cache.rs src/walker.rs src/metacog.rs tests/seeding_walk.rs
git commit -m "feat(walker): seed-plan walks with restart, fan division, deterministic mode"
```

---

### Task 4: Merge mass, output fields, telemetry, fallback and narrative

**Files:**
- Modify: `src/graph.rs` (`WalkOutput`, `WalkTelemetry`), `src/walker.rs` (`merge_mass`, `context_chains`, `telemetry`, `aggregate`, `empty_output`, vector output, `format_walk_narrative`, `render_walk_narrative`)
- Test: unit tests in `src/walker.rs`; `tests/seeding_walk.rs`

**Interfaces:**
- Consumes: `WalkerResult::{depth_at, segments, step_stats}`, `SeedPlan`, `Candidate`, `MERGE_DISCOUNT`, `new_walk_id` (Tasks 1 to 3).
- Produces:

```rust
// src/graph.rs
#[derive(Debug, Clone, Default, Serialize)]
pub struct WalkTelemetry {
    pub path_overlap: f32,        // mean pairwise Jaccard of walkers' node sets; 0 with fewer than 2 walkers
    pub hub_visit_share: Option<f32>, // filled only under RGW_EVAL=1
    pub branching_factor: f32,    // sum(branching_sum) / sum(steps); 0 with no steps
    pub duplicate_claims: u32,    // sum over nodes of (walkers that visited it - 1)
    pub cache_miss_stops: u32,
}
pub struct WalkOutput {
    /* existing fields unchanged */
    #[serde(default)] pub walk_id: String,
    #[serde(default = "default_true")] pub walked: bool,
    #[serde(default)] pub seed_mode: String,           // "fixed" | "shadow" | "query"
    #[serde(default)] pub ranked_nodes: Vec<(i32, f32)>,
    #[serde(default)] pub context_chains: Vec<Vec<i32>>,
    #[serde(default)] pub telemetry: WalkTelemetry,
}

// src/walker.rs
pub const RANKED_MAX: usize = 10;
pub const CHAINS_MAX: usize = 2;
/// m(v) = sum over walkers and visits of MERGE_DISCOUNT^depth; mass descending, then id ascending.
pub fn merge_mass(results: &[WalkerResult]) -> Vec<(i32, f32)>;
/// Up to CHAINS_MAX segments of at least 2 nodes, by summed mass descending; a node appears in one chain only.
pub fn context_chains(results: &[WalkerResult], ranked: &[(i32, f32)]) -> Vec<Vec<i32>>;
pub fn telemetry(results: &[WalkerResult]) -> WalkTelemetry;
/// The no-walk output described under "Vector output" below.
pub fn vector_output(cands: &[Candidate], first_domain: String) -> WalkOutput;
```

`aggregate` fills `ranked_nodes` (truncated to `RANKED_MAX`), `context_chains`, `telemetry` and `walk_id` for every walk; `walk_parallel_with` sets `seed_mode` and `walked`. Under `RGW_EVAL=1` it also sets `hub_visit_share`: the share of all visits that land on nodes whose degree is at least `db::degree_at_percentile(pool, 0.99)`, with that threshold cached in the process for 600 s and degrees read with `db::node_degrees`.
(Ruling: the spec lists hub share with the other telemetry; it needs a degree query per walk,
so it is filled only in eval runs and is `null` otherwise.)

**Vector output** (`arm = Vector`, or `below_floor` under `Query`): no walkers run and the self-model is not touched. `walked = false`, `ranked_nodes` = the first `RANKED_MAX` candidates as `(id, cos)`, `recommended_action = "rest"`, `walker_count = 0`, `primary_domain` = the domain of the first candidate (empty if none), every other list empty.

**Narrative.** `format_walk_narrative` also fetches text for the ids in `ranked_nodes` and `context_chains`. `render_walk_narrative` is unchanged when `seed_mode != "query"` and `walked`. Otherwise the three vote sections are replaced by:
- for `walked`: a section `Trains of thought:` with one row per chain, titles joined by ` -> `, then a section `Also relevant:` with the remaining ranked nodes in rank order, in the existing row format, up to 8 rows in total across both;
- for `!walked`: a section `Closest memories (no walk):` with up to 8 ranked nodes.

- [ ] **Step 1: Write the failing unit tests**

```rust
#[test]
fn merge_mass_discounts_by_depth_since_the_last_restart() {
    let a = WalkerResult { path: vec![1, 2, 3], depths: vec![0, 1, 2], ..wr(vec![], 0.0) };
    let b = WalkerResult { path: vec![2, 1, 9], depths: vec![0, 1, 0], segment_starts: vec![0, 2], ..wr(vec![], 0.0) };
    let m = merge_mass(&[a, b]);
    assert_eq!(m, vec![(1, 1.5), (2, 1.5), (9, 1.0), (3, 0.25)]);
}

#[test]
fn merge_mass_of_legacy_results_uses_position_as_depth() {
    assert_eq!(merge_mass(&[wr(vec![5, 6], 1.0)]), vec![(5, 1.0), (6, 0.5)]);
}

#[test]
fn context_chains_take_the_heaviest_segments_without_reusing_nodes() {
    let a = WalkerResult { path: vec![1, 2, 3], depths: vec![0, 1, 2], ..wr(vec![], 0.0) };
    let b = WalkerResult { path: vec![1, 2], depths: vec![0, 1], ..wr(vec![], 0.0) };
    let c = WalkerResult { path: vec![7], depths: vec![0], ..wr(vec![], 0.0) };
    let ranked = merge_mass(&[a.clone(), b.clone(), c.clone()]);
    assert_eq!(context_chains(&[a, b, c], &ranked), vec![vec![1, 2, 3]]);
}

#[test]
fn telemetry_counts_overlap_duplicates_branching_and_misses() {
    let mut a = wr(vec![1, 2, 3], 1.0);
    a.step_stats = StepStats { branching_sum: 6, steps: 2, cache_miss_stops: 1 };
    let mut b = wr(vec![2, 3, 4], 1.0);
    b.step_stats = StepStats { branching_sum: 2, steps: 2, cache_miss_stops: 0 };
    let t = telemetry(&[a, b]);
    assert!((t.path_overlap - 0.5).abs() < 1e-6, "Jaccard of {{1,2,3}} and {{2,3,4}} is 2/4");
    assert_eq!((t.duplicate_claims, t.cache_miss_stops), (2, 1));
    assert!((t.branching_factor - 2.0).abs() < 1e-6);
    assert_eq!(t.hub_visit_share, None);
}

#[test]
fn vector_output_with_no_candidates_is_well_formed() {
    let out = vector_output(&[], String::new());
    assert!(!out.walked && out.ranked_nodes.is_empty());
    assert_eq!((out.recommended_action.as_str(), out.walker_count), ("rest", 0));
    let one = vector_output(&[Candidate { id: 4, cos: 0.8, degree: 2 }], "d".into());
    assert_eq!((one.ranked_nodes.clone(), one.primary_domain.as_str()), (vec![(4, 0.8)], "d"));
}

#[test]
fn narrative_in_query_mode_shows_chains_then_ranked_nodes() {
    let mut texts = HashMap::new();
    for (id, title) in [(1, "One"), (2, "Two"), (3, "Three")] {
        texts.insert(id, NodeText { title: title.into(), excerpt: String::new(), domain: "d".into() });
    }
    let output = WalkOutput {
        seed_mode: "query".into(), walked: true,
        ranked_nodes: vec![(1, 2.0), (2, 1.0), (3, 0.5)], context_chains: vec![vec![1, 2]],
        consensus_nodes: vec![3], ..empty_output()
    };
    let s = render_walk_narrative(&output, &texts);
    assert!(s.contains("Trains of thought:") && s.contains("One -> Two"));
    assert!(s.contains("Also relevant:") && s.contains("Three"));
    assert!(!s.contains("Strong convictions"));
}

#[test]
fn narrative_outside_query_mode_is_unchanged() {
    let mut texts = HashMap::new();
    texts.insert(1, NodeText { title: "One".into(), excerpt: String::new(), domain: "d".into() });
    let output = WalkOutput { seed_mode: "shadow".into(), walked: true, consensus_nodes: vec![1],
                              ranked_nodes: vec![(1, 1.0)], ..empty_output() };
    let s = render_walk_narrative(&output, &texts);
    assert!(s.contains("Strong convictions (consensus)") && !s.contains("Also relevant"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib merge_mass` then `cargo test --lib narrative_`
Expected: compile errors.

- [ ] **Step 3: Implement the pure functions, the fields and the narrative sections**

Fix every `WalkOutput` literal the compiler reports; `empty_output()` sets `walked: true` and leaves the rest default.

- [ ] **Step 4: Write the failing integration tests**

```rust
#[tokio::test]
async fn query_arm_ranks_the_cue_cluster_first() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_rank").await?;
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
    assert!(out.walked);
    let top4: Vec<i32> = out.ranked_nodes.iter().take(4).map(|n| n.0).collect();
    let gold = [B1, B2, B3, B4];
    assert!(top4.iter().filter(|id| gold.contains(id)).count() >= 3, "top 4 were {top4:?}");
    assert_eq!(out.walk_id.len(), 32);
    Ok(())
}

#[tokio::test]
async fn fixed_arm_ranks_todays_seeds_first() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_rank_fixed").await?;
    // One step each: a start node holds mass 1.0, and no other node can exceed it.
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 2, 1, &sm(), opts(axis(1), EvalArm::Fixed)).await;
    assert!([HUB, A1].contains(&out.ranked_nodes[0].0));
    Ok(())
}

#[tokio::test]
async fn a_cue_far_from_everything_does_not_walk_in_query_mode() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_floor").await?;
    let model = sm();
    let before = serde_json::to_value(&*model.lock().await)?;
    let o = WalkOptions { cue_embedding: Some(axis(7)), seed: Some(1), arm: None,
                          seed_mode: Some(rgw::seeding::SeedMode::Query) };
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 4, 5, &model, o).await;
    assert!(!out.walked && results.is_empty());
    assert_eq!(out.recommended_action, "rest");
    assert_eq!(serde_json::to_value(&*model.lock().await)?, before, "no walk, no self-model change");
    Ok(())
}

#[tokio::test]
async fn vector_arm_returns_candidates_by_cosine() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_vector").await?;
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(2), EvalArm::Vector)).await;
    assert!(!out.walked);
    assert_eq!(out.ranked_nodes[0].0, LONE);
    assert!((out.ranked_nodes[0].1 - 1.0).abs() < 1e-4);
    assert_eq!(out.primary_domain, "gamma");
    Ok(())
}
```

- [ ] **Step 5: Run, implement the vector output and the wiring in `aggregate`, run again**

Also extend `same_seed_gives_the_same_paths` (Task 3) so that each run returns
`(paths, out.ranked_nodes)` and both are compared; this is invariant I6 of the spec.

Run: `cargo test --test seeding_walk && cargo test --locked --lib --bins`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/graph.rs src/walker.rs src/metacog.rs tests/seeding_walk.rs
git commit -m "feat(walker): merge mass, ranked output, telemetry, vector fallback, query narrative"
```

---

### Task 5: Modes, handlers, shadow statistics, PPR arm

**Files:**
- Modify: `src/seeding.rs` (`embed_cue`, `SeedStats`), `src/walker.rs` (`ppr_push`, shadow recording, `Ppr` row), `src/api.rs` (`WalkRequest`, `walk`, `GET /learning`), `src/openai.rs` (pass the cue)
- Test: unit tests in `src/seeding.rs`, `src/walker.rs`, `src/api.rs`; `tests/seeding_walk.rs`

**Interfaces:**
- Produces:

```rust
// src/seeding.rs
/// None for text shorter than MIN_CUE_CHARS after trimming, or when embedding fails (one WARN).
/// Runs the embedder on the blocking pool.
pub async fn embed_cue(text: &str) -> Option<Vec<f32>>;
/// Process-wide shadow statistics. `overlap` is the Jaccard overlap of the plan's first
/// SEED_TOP_K ids with the fixed seed ids.
pub fn record_seed_observation(mode: SeedMode, best_cos: f32, overlap: f32, fell_back: bool);
pub fn seed_stats_json() -> serde_json::Value;
// {"mode": "...", "walks": u64, "fallbacks": u64, "best_cos_p50": f32|null, "overlap_with_fixed_p50": f32|null}

// src/walker.rs
/// Personalized PageRank by power iteration: p0 = r; p(t+1) = RESTART_PROB * r + (1 - RESTART_PROB) * P^T p(t),
/// where a node passes its mass to neighbours in proportion to edge weight and keeps it if it has none.
/// Returns (node, mass), mass descending then id ascending.
pub fn ppr_push(plan: &SeedPlan, steps: usize, neighbours: &dyn Fn(i32) -> Vec<(i32, f32)>) -> Vec<(i32, f32)>;

// src/api.rs
pub struct WalkRequest { /* existing */ #[serde(default)] pub seed: Option<u64>, #[serde(default)] pub arm: Option<String> }
```

Handler rules:
- `/walk`: `cue = embed_cue(&req.stimulus)`. `arm` present and `!eval_enabled()` → 400 `"arm requires RGW_EVAL=1"`. `arm` present and `EvalArm::parse` is `None` → 400 `"unknown arm"`. Both are checked before any walk and before `request_self_model`.
- `/v1/chat/completions`: pass `embed_cue(&stimulus)` as the cue. The prompt it builds is not changed in this plan (see "Noted, out of scope").
- `Ppr` arm: `ranked_nodes` = `ppr_push` over the plan, with neighbours read from the preloaded cache (`cache.try_get`, effective weight = edge weight), truncated to `RANKED_MAX`; `walked = true`, `walker_count = 0`, no self-model change.
- Shadow: under mode `Shadow` with a cue, compute the plan, call `record_seed_observation`, and walk with fixed seeds. Under `Query`, record too (`fell_back` = `below_floor`). Under `Fixed`, record nothing.
- `GET /learning` (protected router): `{"seed": seed_stats_json()}`. Later plans add blocks.

- [ ] **Step 1: Write the failing unit tests**

```rust
// src/walker.rs
#[test]
fn ppr_mass_decays_along_a_chain_and_sums_to_one() {
    let plan = SeedPlan { seeds: vec![(1, 1.0)], best_cos: 1.0, below_floor: false };
    let nb = |n: i32| -> Vec<(i32, f32)> { match n { 1 => vec![(2, 1.0)], 2 => vec![(1, 1.0), (3, 1.0)], 3 => vec![(2, 1.0)], _ => vec![] } };
    let p = ppr_push(&plan, 2, &nb);
    let mass = |id: i32| p.iter().find(|x| x.0 == id).map(|x| x.1).unwrap_or(0.0);
    assert!(mass(1) > mass(2) && mass(2) > mass(3) && mass(3) > 0.0);
    assert!((p.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(p[0].0, 1);
}

// src/seeding.rs
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

// src/api.rs (uses the existing closed-pool `state()` helper: these must be rejected without a database)
#[tokio::test]
async fn arm_is_rejected_without_eval_and_when_unknown() {
    let state = state(CognitiveMode::Autonomous).await;
    let app = Router::new().route("/walk", post(walk)).with_state(state);
    let send = |body: serde_json::Value| app.clone().oneshot(Request::post("/walk")
        .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap());
    // RGW_EVAL is not set in the test process.
    assert_eq!(send(serde_json::json!({"arm": "query"})).await.unwrap().status(), StatusCode::BAD_REQUEST);
    assert_eq!(validate_arm(Some("nonsense"), true).unwrap_err().0, StatusCode::BAD_REQUEST);
    assert_eq!(validate_arm(Some("ppr"), true).unwrap(), Some(EvalArm::Ppr));
    assert_eq!(validate_arm(None, false).unwrap(), None);
}
```

`validate_arm(arm: Option<&str>, eval_enabled: bool) -> Result<Option<EvalArm>, ModeError>` is the pure check the handler calls; it lives beside `validate_request_mode`. `SeedStats` keeps two rings of the last 512 values and two counters; `record` and `to_json` are its methods, and the process-wide functions delegate to one static instance.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib ppr_mass`, then `short_or_blank`, `seed_stats_report`, `arm_is_rejected`
Expected: compile errors.

- [ ] **Step 3: Implement**

- [ ] **Step 4: Write the failing integration tests**

```rust
#[tokio::test]
async fn shadow_mode_walks_as_fixed_and_records_the_plan() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_shadow").await?;
    let before = rgw::seeding::seed_stats_json()["walks"].as_u64().unwrap_or(0);
    let o = WalkOptions { cue_embedding: Some(axis(1)), seed: Some(1), arm: None,
                          seed_mode: Some(rgw::seeding::SeedMode::Shadow) };
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 2, 3, &sm(), o).await;
    assert_eq!(out.seed_mode, "shadow");
    assert_eq!(results.iter().map(|r| r.path[0]).collect::<Vec<_>>(), vec![HUB, A1]);
    // Other tests in this binary record too, so this is a lower bound.
    assert!(rgw::seeding::seed_stats_json()["walks"].as_u64().unwrap() >= before + 1);
    Ok(())
}

#[tokio::test]
async fn no_cue_means_a_fixed_walk_in_every_mode() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_nocue").await?;
    let o = WalkOptions { cue_embedding: None, seed: Some(1), arm: None, seed_mode: Some(rgw::seeding::SeedMode::Query) };
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 2, 3, &sm(), o).await;
    assert!(out.walked);
    assert_eq!(results.iter().map(|r| r.path[0]).collect::<Vec<_>>(), vec![HUB, A1]);
    Ok(())
}

#[tokio::test]
async fn a_wrong_dimension_cue_falls_back_to_a_fixed_walk() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_baddim").await?;
    let o = WalkOptions { cue_embedding: Some(vec![1.0, 0.0]), seed: Some(1), arm: None, seed_mode: Some(rgw::seeding::SeedMode::Query) };
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 2, 3, &sm(), o).await;
    assert!(out.walked);
    assert_eq!(results[0].path[0], HUB);
    Ok(())
}

#[tokio::test]
async fn ppr_arm_ranks_the_cue_cluster_without_walkers() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_ppr").await?;
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Ppr)).await;
    assert!(results.is_empty() && out.walked);
    assert!([B1, B2, B3, B4].contains(&out.ranked_nodes[0].0));
    Ok(())
}
```

- [ ] **Step 5: Run, wire the handlers and the route, run again**

Run: `cargo test --test seeding_walk && cargo test --locked --lib --bins`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/seeding.rs src/walker.rs src/api.rs src/openai.rs tests/seeding_walk.rs
git commit -m "feat(seeding): RGW_SEED_MODE, cue from stimulus, eval arms, shadow statistics, /learning"
```

---

### Task 6: Recall metrics and the harness

**Files:**
- Create: `src/evalkit.rs`, `examples/recall_eval.rs`, `tests/fixtures/recall_fixture.json`
- Modify: `src/lib.rs`, `src/main.rs` (module), `.gitignore` (add `*.gold.json` and `recall_results*.json`)
- Test: unit tests in `src/evalkit.rs`

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GoldQuery { pub query: String, pub gold_node_ids: Vec<i32>, pub category: String }
// category is one of "lookup", "multi_hop", "temporal", "unanswerable"

#[derive(Debug, Clone)]
pub struct EvalRow { pub arm: String, pub category: String, pub ranked: Vec<i32>, pub gold: Vec<i32>,
                     pub walked: bool, pub context_chars: usize, pub ms: f64 }

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CategorySummary { pub n: usize, pub recall_at_10: f32, pub median_first_gold_rank: Option<f32>,
                             pub fallback_share: f32, pub mean_tokens: f32, pub p95_ms: f64 }

pub const RECALL_K: usize = 10;
pub fn recall_at_k(ranked: &[i32], gold: &[i32], k: usize) -> f32;          // 0.0 when gold is empty
pub fn first_gold_rank(ranked: &[i32], gold: &[i32]) -> Option<usize>;     // 1-based
/// (arm, category) -> summary; also an "overall" category per arm that excludes "unanswerable".
pub fn summarise(rows: &[EvalRow]) -> std::collections::BTreeMap<(String, String), CategorySummary>;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GateVerdict { pub g1: bool, pub g1b: bool, pub reasons: Vec<String> }
/// Spec §14. G1: query beats fixed on multi_hop recall by >= 0.10; query within 0.02 of vector on lookup;
/// query's fallback_share on unanswerable >= 0.70; p95_ms(query, overall) - p95_ms(fixed, overall) <= 50.
/// G1b: recall(ppr, overall) - recall(query, overall) <= 0.02.
/// A missing arm or category makes the corresponding gate false with a reason.
/// Thresholds are compared in f32 without tolerance; the tests stay clear of the exact boundary.
pub fn judge_gates(summary: &std::collections::BTreeMap<(String, String), CategorySummary>) -> GateVerdict;
```

`mean_tokens` is `context_chars / 4`. `fallback_share` is the share of rows with `walked == false`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn recall_and_first_rank() {
    assert_eq!(recall_at_k(&[9, 5, 7], &[5, 6], 10), 0.5);
    assert_eq!(recall_at_k(&[9, 5, 7], &[5, 6], 1), 0.0);
    assert_eq!(recall_at_k(&[1], &[], 10), 0.0);
    assert_eq!(first_gold_rank(&[9, 5, 7], &[7, 5]), Some(2));
    assert_eq!(first_gold_rank(&[9], &[5]), None);
}

fn row(arm: &str, cat: &str, ranked: Vec<i32>, gold: Vec<i32>, walked: bool, ms: f64) -> EvalRow {
    EvalRow { arm: arm.into(), category: cat.into(), ranked, gold, walked, context_chars: 400, ms }
}

#[test]
fn summary_is_per_arm_and_category_with_an_overall_that_skips_unanswerable() {
    let rows = vec![
        row("query", "multi_hop", vec![1, 2], vec![1, 2], true, 10.0),
        row("query", "multi_hop", vec![3], vec![1, 2], true, 30.0),
        row("query", "unanswerable", vec![], vec![], false, 5.0),
    ];
    let s = summarise(&rows);
    let mh = &s[&("query".to_string(), "multi_hop".to_string())];
    assert_eq!((mh.n, mh.recall_at_10), (2, 0.5));
    assert_eq!(mh.mean_tokens, 100.0);
    assert_eq!(s[&("query".to_string(), "unanswerable".to_string())].fallback_share, 1.0);
    assert_eq!(s[&("query".to_string(), "overall".to_string())].n, 2);
}

fn cs(recall: f32, fallback: f32, p95: f64) -> CategorySummary {
    CategorySummary { n: 20, recall_at_10: recall, median_first_gold_rank: Some(1.0), fallback_share: fallback, mean_tokens: 100.0, p95_ms: p95 }
}
fn table(entries: &[(&str, &str, CategorySummary)]) -> std::collections::BTreeMap<(String, String), CategorySummary> {
    entries.iter().map(|(a, c, s)| ((a.to_string(), c.to_string()), s.clone())).collect()
}

#[test]
fn gates_pass_on_the_spec_thresholds_and_fail_just_inside_them() {
    let pass = table(&[
        ("fixed", "multi_hop", cs(0.30, 0.0, 40.0)), ("query", "multi_hop", cs(0.41, 0.0, 60.0)),
        ("vector", "lookup", cs(0.80, 1.0, 5.0)),    ("query", "lookup", cs(0.79, 0.0, 60.0)),
        ("query", "unanswerable", cs(0.0, 0.70, 5.0)),
        ("fixed", "overall", cs(0.30, 0.0, 40.0)),   ("query", "overall", cs(0.60, 0.0, 90.0)),
        ("ppr", "overall", cs(0.61, 0.0, 20.0)),
    ]);
    let v = judge_gates(&pass);
    assert!(v.g1 && v.g1b, "{:?}", v.reasons);

    let mut fail = pass.clone();
    fail.insert(("query".into(), "multi_hop".into()), cs(0.39, 0.0, 60.0));
    fail.insert(("ppr".into(), "overall".into()), cs(0.63, 0.0, 20.0));
    let v = judge_gates(&fail);
    assert!(!v.g1 && !v.g1b);
    assert_eq!(v.reasons.len(), 2);
}

#[test]
fn a_missing_arm_fails_its_gate_with_a_reason() {
    let v = judge_gates(&table(&[("query", "overall", cs(0.5, 0.0, 10.0))]));
    assert!(!v.g1 && !v.g1b && !v.reasons.is_empty());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib evalkit::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/evalkit.rs`**

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib evalkit::`
Expected: 4 passed.

- [ ] **Step 5: Write `examples/recall_eval.rs` and the fixture set**

The example mirrors `examples/ab_eval.rs` (reqwest, `RGW_BASE_URL`, `RGW_API_KEY` sent as `X-RGW-Key`). Usage:

```
cargo run --release --example recall_eval -- <gold.json> [--arms fixed,query,vector,ppr[,query:linear,drop:<bias>]] [--walkers 4] [--steps 5] [--seed 1] [--out recall_results.json]
```

For each query and arm it posts `{"stimulus": query, "n_walkers": W, "steps": S, "seed": seed, "arm": arm}` to `/walk`, builds an `EvalRow` from `ranked_nodes` (ids in order), `walked`, `walker_context.len()` and its own wall-clock time, then prints the `summarise` table and the `judge_gates` verdict with its reasons, and writes rows and summary to `--out`. A 400 that mentions `RGW_EVAL` ends the run with the message "start the server with RGW_EVAL=1". The module doc comment states that gold sets and results must not be committed.

`tests/fixtures/recall_fixture.json` holds three queries over the Task 2 fixture ids, as a format example: one `multi_hop` with gold `[5, 6, 7, 8]`, one `lookup` with gold `[9]`, one `unanswerable` with gold `[]`. Its query strings are plain placeholders (`"beta cluster"`, `"lone node"`, `"nothing matches this"`); with the real embedder they do not align with the synthetic vectors, and the doc comment says so.

- [ ] **Step 6: Build and commit**

Run: `cargo build --release --example recall_eval && cargo test --locked --lib --bins`
Expected: builds; suite green.

```bash
git add src/evalkit.rs src/lib.rs src/main.rs examples/recall_eval.rs tests/fixtures/recall_fixture.json .gitignore
git commit -m "feat(eval): gold-node recall metrics, gate verdict, recall_eval harness"
```

---

### Task 7: Final checks

- [ ] **Step 1: Whole suite**

Run: `cargo test --locked --lib --bins && cargo test --test seeding_walk && cargo test --test tension_observe`
Expected: all green. `tension_observe` proves the unchanged `walk_parallel` signature still serves existing callers.

- [ ] **Step 2: Clippy**

Run: `cargo clippy --quiet 2>&1 | grep -c '^warning'` on this branch and on `main`.
Expected: the count has not increased.

- [ ] **Step 3: Default-path check**

Start the server locally with no new variable set, post `{"stimulus": "anything long enough"}` to `/walk`.
Expected: 200; `seed_mode` is `"shadow"`; `walked` is `true`; `GET /learning` shows `seed.walks` incremented; the log contains no part of the stimulus text.

- [ ] **Step 4: Secret scan**

Run: `gitleaks detect --log-opts=origin/main..HEAD`
Expected: no leaks.

---

## Gate run (after merge and deploy; not a code task)

1. The operator writes a gold set per deployment, outside the repository, from knowledge of the graph and not from current retrievals: at least 20 queries in each of `lookup`, `multi_hop`, `unanswerable`.
2. Run the harness against a server started with `RGW_EVAL=1`, arms `fixed,query,vector,ppr`, three times with seeds 1, 2, 3.
3. G1 and G1b are judged on the printed verdict of each run; all three must agree. If `SEED_FLOOR` needs tuning, tune it on half of the gold set and judge on the other half.
4. Only then set `RGW_SEED_MODE=query`. If G1b fails, stop: the next step is the decision on persona walkers, not Stage 5.

## Noted, out of scope

- The chat path (`/v1/chat/completions`) builds its prompt from `format_walk_context`, which carries walk statistics and no node text. After this plan its walk is query-seeded, but the nodes it finds still do not reach the prompt. Changing that prompt is a separate decision.
- `edges_from` returns at most 20 edges per node, by weight. Degree-based specificity uses true degrees; fan division sees at most 20 candidates.
- The cue is embedded with `embed::embed_text`, the same call and prefix used for stored documents. A query-prefix variant is not part of this plan.
- With a seed plan, radius-1 preload covers the first five seeds only. A walker that restarts at a wide seed can take one step before it meets uncached edges. `telemetry.cache_miss_stops` measures how often that happens.

---

## Execution record

Executed inline on `feat/learning-walkers-p1`, one independent whole-branch review, one fix pass.

### Decisions that deviate from the plan above

| Where | Decision | Why | Cost if wrong |
|---|---|---|---|
| Task 3 | `hops()` is nodes minus segments, not `edges_traversed.len()` | Existing tests build results with empty `edges_traversed` and expect `path.len() - 1`; for real walks the two are equal | Hop-rate metrics off for a result whose path and edges disagree |
| Task 3 | The per-step restart draw starts at the second step | A restart before the first step only re-draws the start node | First segments slightly longer |
| Task 3 | Integration tests load the embedding model once per binary | Several tests loading it at once against a cold cache failed while it downloaded | None for behaviour |
| Task 3, review | `drop:<bias>` skips the slot's learned profile as well as its label, and the omitted profile does not rotate; credit goes to the profile a walker scored with | With learned profiles in use the named bias is only a label | The arm ablates a different profile than intended |
| Task 4 | No serde default attributes on the new `WalkOutput` fields | The struct derives only `Serialize` | None until it is deserialised |
| Task 4 | The similarity floor applies to every query-seeded kind, eval arms and `ppr` included | G1 measures the query arm's fallback share | An eval arm returns no walk where one was wanted |
| Task 4 | `seed_mode` reports what decided the result | The narrative keys on it | A caller sees `fixed` although the mode is `query` and no cue was usable |
| Task 5, review | The critic does not run when the query decided the result and no walker ran, nor for any eval arm | Those are not walk sessions; eval timings must not include an LLM call | Critic counters do not advance on those requests |
| Task 5 | Shadow statistics are recorded only without an eval arm | A harness run must not move production medians | Eval traffic is invisible in `GET /learning` |
| Task 7 | `evalkit` is declared in the library only | The binary does not use it; declaring it there added dead-code warnings | None |
| Review | The `, id ASC` tie-break in `edges_from` is not applied | It changes which 20 edges every walk sees on a node with more than 20 equal-weight edges, against the "exactly as today" constraint | Two deterministic runs may see different edges on such a node |
| Review | `fixed` mode and the `fixed` arm do not embed the query; eval arms embed without the cache | The rollback mode must cost what it cost before; each harness arm must pay the real embedding | None |
| Review | Planned walks and PPR are clamped to 32 steps | A planned walk runs every step, so `steps` was an unbounded allocation | A caller asking for more gets 32 |
| Review | Chains collapse revisits, score distinct nodes, and need half their nodes in the top-ranked ten | Ping-pong segments won; the second chain left the topic | A legitimate long chain through lower-ranked nodes is not shown |
| Review | `total_hops` and the novelty denominator count the start plus traversals | Restart nodes are not hops; unchanged for walks without restarts | None |
| Review | Gate thresholds tolerate 1e-6; a category under 20 queries fails its gate | An exact 10-point gain compared as under 10 in f32; a toy set printed PASS | A gold set of 19 cannot pass |
| Review | The similarity query orders by distance only | A second sort key can keep Postgres from using a vector index | Ties in cosine are cut arbitrarily at the limit |
| Review | Shadow observation runs on a detached task | The default mode must not add queries to the request path | The statistic lags the walk by milliseconds |

### Set aside by the reviewer, and where each stands

- The chat prompt carries no node text: out of scope here; query seeding does not reach chat prompts until that prompt changes.
- `edges_from` returns at most 20 edges per node: unchanged.
- The cue is embedded with the document prefix: unchanged; `SEED_FLOOR` is tuned at the gate run.
- Radius-1 preload covers the first five seeds only: unchanged; `telemetry.cache_miss_stops` measures it.
- Spec §5.1 asks `ab_eval` to run both judge orderings: not done; no P1 gate reads `ab_eval`. Needed before it is used as evidence.
- The in-repo fixture is synthetic: the harness was run end to end on it once, which exercises the pipeline and says nothing about recall.
- `rgw_walk_id` on the chat response: plan P2.
- A harness run changes the server's self-model through the walk merge (the critic no longer runs for eval arms): run the gate on a copy, or accept the drift.
- Unbounded `n_walkers`, and `seed_nodes` without a tie-break: pre-existing, untouched.

### Deferred, not fixed

- A request `seed` alone makes the walk deterministic and sequential; the spec reads as the environment flag plus an optional seed.
- In deterministic mode the start nodes round-robin over all plan seeds, so with more than five walkers the extras start at wide seeds without a preloaded neighbourhood.
- `WalkerBias::Random` scoring uses the thread RNG in deterministic mode when no learned profile exists.
- `ppr_push` sums in hash-map order; unloaded frontier nodes keep their mass.
- Node domains are cached only for first-pass seeds; check at the gate run that `primary_domain` is not empty more often under `query`.
- Votes are per visit, so a walker that restarts at a seed votes for it again; agreement and the `express` action rise under `query`.
- `empty_output()` reports `walked: true` with an empty `walk_id`.
- Two doc comments sit on the wrong item (`WalkerBias::from_name` / `all`, `learning_stats` / `diverger_stats`).
- The README does not mention the new variables or `GET /learning`, and no startup line states the seed mode.
- Confirm with `EXPLAIN` on a deployment that the similarity query uses the vector index.
