# Tension-Driven Background Activity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let unresolved predictions and open questions release bounded energy pulses into the diverger after every walk, shipped together with three brakes (node fatigue, per-cascade budget, global refilling pool), default mode `observe` which changes no behaviour.

**Architecture:** A new `tension` module holds pure decision functions (strength, decay, pool, fatigue, hashing), a `TensionEngine` installed once per process (same `OnceLock` pattern as `motor::gate()`), and a stats block. The diverger's inbox becomes a `Stimulus` carrying origin, depth and budget; in `brakes`/`live` a completed spontaneous walk feeds its strengthened edges back as the next hop, which is what makes cascades multi-hop and makes `max_cascade_depth` and the budget meaningful. `walk_parallel` and the diverger's spontaneous walk each call one hook, `tension::after_walk`.

**Tech Stack:** Rust 2024 (MSRV 1.88), tokio, sqlx/pgvector, existing `embed::embed_text` (fastembed). No new crates.

**Spec:** `docs/superpowers/specs/2026-11-04-tension-driven-activity-design.md`

## Global Constraints

- `rust-version = "1.88"` in Cargo.toml must keep building; no new dependencies.
- `RGW_TENSION_MODE` is the **only** new environment variable; all other parameters are `pub const` in `src/tension.rs` with the exact defaults from spec §12.
- `off` must be byte-for-byte today's behaviour; `observe` (default) must change no behaviour of edge-originated activity.
- No LLM calls anywhere in this feature. No schema change. The three new `SelfModel` fields are `#[serde(default)]`.
- Log lines never contain question text; questions appear only as `question:<16 hex>` (FNV-1a 64).
- Every task ends with `cargo test` green (unit: `cargo test --lib`; integration needs the local Postgres at `postgresql:///rgw_test?host=/tmp`) and `cargo clippy --quiet` not increasing the warning count (57 at start).
- Commits go through the local gitleaks pre-push hook; the repo is public.

## Review Focus

1. `RGW_TENSION_MODE=banana` (or empty) → treated as `observe` with one WARN, never a panic or `off`. (Task 1)
2. A walk that completes before `tension::install` ran (tests, early startup) → `after_walk` is a no-op returning 0, not a panic. (Task 6)
3. The same question text re-added to `open_questions` after being abandoned → stays abandoned (hash persisted), never pulses again. (Task 3)
4. `embed_text` fails for a question (embedder not initialised) → item marked `unmapped`, counted once, retried only after `ITEM_COOLDOWN_SECS`, release of other items unaffected. (Task 3)
5. Pool refill after hours idle → tokens capped at `capacity`, never above. (Task 1)
6. Self-model in `CognitiveMode::Compliant` → release computes and logs but emits nothing (counted as `dropped_compliant`), matching the diverger's existing rule that spontaneous firing is suppressed in Compliant. (Task 3)

---

### Task 1: Pure core of `tension.rs`

**Files:**
- Create: `src/tension.rs`
- Modify: `src/lib.rs` (add `pub mod tension;`)
- Test: unit tests in `src/tension.rs`

**Interfaces:**
- Produces:
  - `pub enum TensionMode { Off, Observe, Brakes, Live }` with `pub fn from_env_value(v: Option<&str>) -> TensionMode` (case-insensitive; `None`/empty/unknown → `Observe`, unknown logs `tracing::warn!`), `pub fn emits(self) -> bool` (Live only), `pub fn enforces_brakes(self) -> bool` (Brakes|Live).
  - Constants exactly as spec §12: `PREDICTION_SATURATION_SECS: f64 = 3600.0`, `PREDICTION_BASE_ENERGY: f32 = 0.15`, `PREDICTION_HALF_LIFE_SECS: f64 = 86400.0`, `PREDICTION_MAX_ACTIVE: usize = 10`, `QUESTION_BASE_ENERGY: f32 = 0.20`, `QUESTION_DECAY: f32 = 0.85`, `QUESTION_MAX_ATTEMPTS: u32 = 12`, `MIN_PULSE_ENERGY: f32 = 0.02`, `K_PREDICTION_TARGETS: i64 = 3`, `K_QUESTION_TARGETS: i32 = 3`, `ITEM_COOLDOWN_SECS: f64 = 120.0`, `FATIGUE_PER_FIRE: f32 = 0.5`, `FATIGUE_DECAY_PER_SEC: f32 = 0.01`, `CASCADE_BUDGET: f32 = 1.0`, `HOP_COST: f32 = 0.2`, `POOL_CAPACITY: f32 = 20.0`, `POOL_REFILL_PER_SEC: f32 = 0.05`, `LIVE_JUDGEMENT_HOURS: u64 = 72`, `ABANDONED_HASHES_MAX: usize = 4096`, `DEFAULT_EDGE_CASCADE_HOPS: f32 = 3.0`.
  - `pub fn fnv1a64(text: &str) -> u64`; `pub fn question_id(text: &str) -> String` → `format!("question:{:016x}", fnv1a64(text))`.
  - `pub fn prediction_strength(age_secs: f64) -> f32` = `min(1, age/SATURATION) * BASE`, then halved once per full half-life elapsed (`age / HALF_LIFE` floored).
  - `pub struct TokenPool { pub capacity: f32, pub tokens: f32, pub refill_per_sec: f32, last_refill: f64 }` with `pub fn new(now_secs: f64) -> Self` (starts full), `pub fn refill(&mut self, now_secs: f64)`, `pub fn try_spend(&mut self, amount: f32, now_secs: f64) -> bool`.
  - `pub fn effective_threshold(base: f32, fatigue: f32) -> f32` = `base * (1.0 + fatigue)`; `pub fn decay_fatigue(fatigue: f32, dt_secs: f64) -> f32`.
  - `pub struct QuestionState { pub strength: f32, pub attempts: u32, pub abandoned: bool, pub unmapped: bool, pub targets: Vec<i32>, pub last_release: f64 }` with `pub fn new() -> Self` and `pub fn after_release(&mut self, now_secs: f64)` (multiply strength by `QUESTION_DECAY`, `attempts += 1`, set `abandoned` when `attempts >= QUESTION_MAX_ATTEMPTS || strength < MIN_PULSE_ENERGY`).

- [ ] **Step 1: Write the failing tests** in `src/tension.rs` `mod tests`:

```rust
#[test] fn mode_parses_case_insensitively_and_defaults_to_observe() {
    assert_eq!(TensionMode::from_env_value(Some("LIVE")), TensionMode::Live);
    assert_eq!(TensionMode::from_env_value(Some("brakes")), TensionMode::Brakes);
    assert_eq!(TensionMode::from_env_value(None), TensionMode::Observe);
    assert_eq!(TensionMode::from_env_value(Some("")), TensionMode::Observe);
    assert_eq!(TensionMode::from_env_value(Some("banana")), TensionMode::Observe);
    assert!(TensionMode::Live.emits() && !TensionMode::Brakes.emits());
    assert!(TensionMode::Brakes.enforces_brakes() && !TensionMode::Observe.enforces_brakes());
}
#[test] fn fnv1a64_known_vectors() {
    assert_eq!(fnv1a64(""), 0xcbf29ce484222325);
    assert_eq!(fnv1a64("a"), 0xaf63dc4c8601ec8c);
    assert_eq!(question_id("a"), "question:af63dc4c8601ec8c");
}
#[test] fn prediction_strength_saturates_and_halves_per_half_life() {
    assert!(prediction_strength(0.0).abs() < 1e-6);
    assert!((prediction_strength(1800.0) - 0.075).abs() < 1e-5);
    assert!((prediction_strength(3600.0) - 0.15).abs() < 1e-5);
    assert!((prediction_strength(7200.0) - 0.15).abs() < 1e-5);        // saturated, < 1 half-life
    assert!((prediction_strength(86400.0) - 0.075).abs() < 1e-5);      // 1 half-life
    assert!((prediction_strength(2.0 * 86400.0) - 0.0375).abs() < 1e-5);
}
#[test] fn token_pool_spends_refills_and_caps() {
    let mut p = TokenPool::new(0.0);
    assert_eq!(p.tokens, 20.0);
    assert!(p.try_spend(19.5, 0.0)); assert!(!p.try_spend(1.0, 0.0));
    assert!((p.tokens - 0.5).abs() < 1e-6);
    p.refill(10.0); assert!((p.tokens - 1.0).abs() < 1e-6);            // 0.05/s
    p.refill(1_000_000.0); assert_eq!(p.tokens, 20.0);                 // capped after a long idle
}
#[test] fn fatigue_raises_threshold_and_decays() {
    assert!((effective_threshold(1.0, 0.5) - 1.5).abs() < 1e-6);
    assert!((decay_fatigue(0.5, 10.0) - 0.4).abs() < 1e-6);
    assert_eq!(decay_fatigue(0.05, 100.0), 0.0);
}
#[test] fn question_decays_and_abandons() {
    let mut q = QuestionState::new();
    for _ in 0..QUESTION_MAX_ATTEMPTS { assert!(!q.abandoned); q.after_release(0.0); }
    assert!(q.abandoned);
    let mut q2 = QuestionState::new(); q2.strength = 0.021; q2.after_release(0.0);
    assert!(q2.abandoned, "strength floor");
}
```

- [ ] **Step 2: Run to verify failure.** Run: `cargo test --lib tension::` — Expected: compile error (module missing).
- [ ] **Step 3: Implement** the items in Interfaces in `src/tension.rs`; add `pub mod tension;` to `src/lib.rs`. FNV-1a 64: offset `0xcbf29ce484222325`, prime `0x100000001b3`, over UTF-8 bytes.
- [ ] **Step 4: Run to verify pass.** Run: `cargo test --lib tension::` — Expected: 6 passed.
- [ ] **Step 5: Commit.** `git add src/tension.rs src/lib.rs && git commit -m "feat(tension): pure core — mode, constants, fnv1a, strength, pool, fatigue, question state"`

---

### Task 2: Self-model counters and the persisted abandoned set

**Files:**
- Modify: `src/core.rs` (struct `SelfModel` ~line 199; `predict()` ~570; prediction resolution ~706; open-question push ~648)
- Test: unit tests in `src/core.rs`

**Interfaces:**
- Consumes: `tension::{fnv1a64, ABANDONED_HASHES_MAX}`.
- Produces on `SelfModel`: `#[serde(default)] pub predictions_made_total: u64`, `#[serde(default)] pub predictions_resolved_total: u64`, `#[serde(default)] pub abandoned_question_hashes: Vec<u64>`; methods `pub fn note_question_abandoned(&mut self, hash: u64)` (no duplicates; drop front when `len > ABANDONED_HASHES_MAX`), `pub fn is_question_abandoned(&self, hash: u64) -> bool`, `pub fn prune_abandoned_hashes(&mut self)` (retain only hashes whose question is still in `open_questions`).

- [ ] **Step 1: Write the failing tests** in `src/core.rs` `mod tests`:

```rust
#[test] fn self_model_loads_without_tension_fields() {
    let mut v = serde_json::to_value(SelfModel::new()).unwrap();
    let o = v.as_object_mut().unwrap();
    o.remove("predictions_made_total"); o.remove("predictions_resolved_total"); o.remove("abandoned_question_hashes");
    let sm: SelfModel = serde_json::from_value(v).unwrap();
    assert_eq!((sm.predictions_made_total, sm.predictions_resolved_total, sm.abandoned_question_hashes.len()), (0, 0, 0));
}
#[test] fn predict_counts_made_and_resolution_counts_resolved() {
    let mut sm = SelfModel::new();
    sm.predict("markets", "up", 0.6);
    assert_eq!(sm.predictions_made_total, 1);
    let mut sig = Signal::new("walk_end", "went down");
    sig.domain = "markets".into();                               // domain match → `predictions.remove()` path
    let _ = process(sig, &mut sm);                               // `pub fn process(signal, &mut SelfModel)` at core.rs:599
    assert_eq!(sm.predictions_resolved_total, 1);
    assert!(sm.predictions.is_empty());
}
#[test] fn abandoned_hashes_dedupe_cap_and_prune() {
    let mut sm = SelfModel::new();
    sm.note_question_abandoned(7); sm.note_question_abandoned(7);
    assert_eq!(sm.abandoned_question_hashes, vec![7]);
    for i in 0..(crate::tension::ABANDONED_HASHES_MAX as u64 + 5) { sm.note_question_abandoned(1000 + i); }
    assert_eq!(sm.abandoned_question_hashes.len(), crate::tension::ABANDONED_HASHES_MAX);
    assert!(!sm.abandoned_question_hashes.contains(&7), "oldest dropped first");
    sm.open_questions = vec!["keep me".into()];
    let keep = crate::tension::fnv1a64("keep me"); sm.note_question_abandoned(keep);
    sm.prune_abandoned_hashes();
    assert_eq!(sm.abandoned_question_hashes, vec![keep]);
}
```

- [ ] **Step 2: Run to verify failure.** `cargo test --lib core::tests::self_model_loads_without_tension_fields core::tests::predict_counts core::tests::abandoned_hashes` — Expected: compile errors on missing fields/methods.
- [ ] **Step 3: Implement**: add the three fields; `+= 1` in `predict()`; `+= 1` right after the `predictions.remove(&signal.domain)` succeeds (before the `age < 60.0` check, since removal is the resolution); the three methods; call `prune_abandoned_hashes()` immediately after the `open_questions.remove(0)` at ~653.
- [ ] **Step 4: Run to verify pass.** Same command — Expected: 3 passed; `cargo test --lib` all green.
- [ ] **Step 5: Commit.** `git commit -am "feat(core): prediction made/resolved counters + persisted abandoned-question hashes"`

---

### Task 3: `TensionEngine` — release planning (pure) and charging

**Files:**
- Modify: `src/tension.rs`
- Test: unit tests in `src/tension.rs`

**Interfaces:**
- Consumes: Task 1 items; `core::{SelfModel, CognitiveMode}`.
- Produces:
  - `pub enum TensionSource { Prediction { domain: String }, Question { id: String } }` (`Debug, Clone, Serialize`).
  - `pub struct TensionPulse { pub source: TensionSource, pub target_node: i32, pub energy: f32 }`.
  - `pub struct PendingPulse { pub source: TensionSource, pub key: String, pub energy: f32 }` (key = domain or question hash hex; used for state lookup).
  - `pub struct TensionStats` (`Serialize, Default, Clone`) with fields exactly as spec §8 `tension` block minus `recent`'s struct: `mode: String, open_predictions, open_questions, abandoned_questions, unmapped: u64, pulses_would_fire, pulses_fired, dropped_pool_empty, dropped_item_cooldown, dropped_inbox_full, dropped_compliant: u64, cascades: CascadeStats { tension_origin, edge_origin, ended_by_budget, ended_naturally: u64 }, brakes_would_have: BrakesWouldHave { fatigue_blocks, budget_cuts, pool_drops: u64 }, pool: PoolStats { tokens, capacity: f32, empty_fraction_1h: f32 }, predictions_made_total, predictions_resolved_total: u64, resolution_rate_1h: f32, recent: Vec<RecentEvent>` where `RecentEvent { t: u64, source: TensionSource, target: i32, energy: f32, fired: bool }`.
  - `pub struct EngineInner` with **all fields `pub`** (tests read them): `mode: TensionMode, pool: TokenPool, questions: HashMap<u64, QuestionState>, prediction_targets: HashMap<String, (Vec<i32>, f64 /*last_release*/)>, avg_edge_cascade_hops: f32, hop_samples: VecDeque<u32> /*cap 100*/, pool_samples: VecDeque<(f64, bool)> /*1h*/, resolution_samples: VecDeque<(f64, u64, u64)>, recent: VecDeque<RecentEvent> /*cap 50*/`, and the `u64` counters `pulses_would_fire, pulses_fired, dropped_pool_empty, dropped_item_cooldown, dropped_inbox_full, dropped_compliant, unmapped, cascades_tension_origin, cascades_edge_origin, cascades_ended_by_budget, cascades_ended_naturally, would_fatigue_blocks, would_budget_cuts, would_pool_drops`. Constructor `pub fn new(mode: TensionMode, now_secs: f64) -> Self` (pool full, everything else empty/zero, `avg_edge_cascade_hops = DEFAULT_EDGE_CASCADE_HOPS`).
  - `pub fn plan_release(inner: &mut EngineInner, sm: &mut SelfModel, now_secs: f64) -> Vec<PendingPulse>` — pure: applies Compliant check (returns empty, `dropped_compliant += 1`), per-item cooldown, prediction cap/order (freshest `PREDICTION_MAX_ACTIVE` by `predicted_at` desc), `prediction_strength`, question decay/abandonment (calls `sm.note_question_abandoned`, skips `sm.is_question_abandoned`), drops `energy < MIN_PULSE_ENERGY`.
  - `pub fn charge(inner: &mut EngineInner, pulses: Vec<TensionPulse>, now_secs: f64) -> Vec<TensionPulse>` — charges the pool `energy` (+ `avg_edge_cascade_hops * HOP_COST` when `!mode.emits()`), records `pool_samples`, increments `pulses_would_fire`/`pulses_fired`/`dropped_pool_empty`, pushes `recent`, logs `[tension] would-fire …` / `fire …` at INFO; returns the pulses to emit (empty unless `mode.emits()`).
  - `pub fn record_edge_cascade_hops(inner: &mut EngineInner, hops: u32)`; `pub fn record_cascade_end(inner: &mut EngineInner, origin: Origin, by_budget: bool)`; `pub fn record_would_have(inner: &mut EngineInner, what: WouldHave)` with `pub enum WouldHave { Fatigue, Budget, Pool }`; `pub enum Origin { Edge, Tension }` (`Copy, Serialize`).
  - `pub fn record_resolution_sample(inner: &mut EngineInner, made: u64, resolved: u64, now_secs: f64)` — pushes `(now, made, resolved)` and trims to 3600 s. Called from `after_walk` (Task 6) on every walk, **never** from `stats`.
  - `pub fn stats(inner: &EngineInner, sm: &SelfModel, now_secs: f64) -> TensionStats` (read-only: computes `empty_fraction_1h` and `resolution_rate_1h` from existing samples; `open_*` from `sm`).

- [ ] **Step 1: Write the failing tests**:

```rust
fn sm_with(preds: &[(&str, f64)], qs: &[&str], now: f64) -> SelfModel { /* predictions with predicted_at = now - age; open_questions */ }

#[test] fn plan_release_respects_cooldown_cap_and_freshness() {
    let now = 1_000_000.0; let mut inner = EngineInner::new(TensionMode::Observe, now);
    let preds: Vec<(String, f64)> = (0..15).map(|i| (format!("d{i}"), 3600.0 + i as f64)).collect();
    let mut sm = sm_with(&preds.iter().map(|(d,a)| (d.as_str(), *a)).collect::<Vec<_>>(), &[], now);
    let p1 = plan_release(&mut inner, &mut sm, now);
    assert_eq!(p1.len(), PREDICTION_MAX_ACTIVE);
    assert!(p1.iter().all(|p| matches!(&p.source, TensionSource::Prediction{domain} if domain != "d14")), "oldest excluded");
    let p2 = plan_release(&mut inner, &mut sm, now + 1.0);
    assert!(p2.is_empty(), "cooldown"); assert_eq!(inner.dropped_item_cooldown, PREDICTION_MAX_ACTIVE as u64);
    let p3 = plan_release(&mut inner, &mut sm, now + ITEM_COOLDOWN_SECS + 1.0);
    assert_eq!(p3.len(), PREDICTION_MAX_ACTIVE);
}
#[test] fn plan_release_abandons_questions_and_never_revives_them() {
    let now = 0.0; let mut inner = EngineInner::new(TensionMode::Observe, now);
    let mut sm = sm_with(&[], &["why is the ferry late"], now);
    let mut t = now; let mut releases = 0;
    for _ in 0..(QUESTION_MAX_ATTEMPTS + 3) { if !plan_release(&mut inner, &mut sm, t).is_empty() { releases += 1; } t += ITEM_COOLDOWN_SECS + 1.0; }
    assert!(releases <= QUESTION_MAX_ATTEMPTS as usize && releases >= 1);
    assert!(sm.is_question_abandoned(fnv1a64("why is the ferry late")));
    sm.open_questions = vec!["why is the ferry late".into()];          // re-added
    assert!(plan_release(&mut inner, &mut sm, t + ITEM_COOLDOWN_SECS).is_empty());
}
#[test] fn plan_release_is_silent_in_compliant_mode() {
    let now = 0.0; let mut inner = EngineInner::new(TensionMode::Live, now);
    let mut sm = sm_with(&[("markets", 3600.0)], &[], now); sm.mode = CognitiveMode::Compliant;
    assert!(plan_release(&mut inner, &mut sm, now).is_empty()); assert_eq!(inner.dropped_compliant, 1);
}
#[test] fn charge_emits_only_in_live_and_estimates_cascade_cost_otherwise() {
    let now = 0.0; let pulse = |e: f32| TensionPulse { source: TensionSource::Prediction{domain:"m".into()}, target_node: 1, energy: e };
    let mut obs = EngineInner::new(TensionMode::Observe, now);
    assert!(charge(&mut obs, vec![pulse(0.1)], now).is_empty());
    assert_eq!(obs.pulses_would_fire, 1);
    assert!((obs.pool.tokens - (20.0 - 0.1 - DEFAULT_EDGE_CASCADE_HOPS * HOP_COST)).abs() < 1e-5);
    let mut live = EngineInner::new(TensionMode::Live, now);
    assert_eq!(charge(&mut live, vec![pulse(0.1)], now).len(), 1);
    assert_eq!(live.pulses_fired, 1); assert!((live.pool.tokens - 19.9).abs() < 1e-5);
    live.pool.tokens = 0.05;
    assert!(charge(&mut live, vec![pulse(0.1)], now).is_empty()); assert_eq!(live.dropped_pool_empty, 1);
}
#[test] fn stats_rates_and_ring_buffers_are_bounded() {
    let now = 0.0; let mut inner = EngineInner::new(TensionMode::Live, now);
    for i in 0..120 { record_edge_cascade_hops(&mut inner, i % 5); }
    assert_eq!(inner.hop_samples.len(), 100);
    for i in 0..60 { charge(&mut inner, vec![TensionPulse{source:TensionSource::Prediction{domain:"m".into()},target_node:i,energy:0.01}], now); }
    assert_eq!(inner.recent.len(), 50);
    record_resolution_sample(&mut inner, 10, 5, now); record_resolution_sample(&mut inner, 20, 13, now + 60.0);
    let sm = SelfModel::new(); let s = stats(&inner, &sm, now + 60.0);
    assert_eq!(s.mode, "live"); assert!((s.resolution_rate_1h - 0.8).abs() < 1e-6, "Δresolved 8 / Δmade 10");
    assert!(s.pool.empty_fraction_1h >= 0.0 && s.pool.empty_fraction_1h <= 1.0);
    let s2 = stats(&inner, &sm, now + 60.0); assert_eq!(inner.resolution_samples.len(), 2, "stats() never records"); let _ = s2;
}
```

- [ ] **Step 2: Run to verify failure.** `cargo test --lib tension::` — Expected: compile errors (types missing).
- [ ] **Step 3: Implement** everything in Interfaces. `resolution_rate_1h` = Δresolved/Δmade between the oldest and newest sample in the window (0.0 when Δmade == 0); `stats()` only reads. `empty_fraction_1h`: fraction of `pool_samples` in the last 3600 s with `empty == true` (0.0 when none).
- [ ] **Step 4: Run to verify pass.** `cargo test --lib tension::` — Expected: 11 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat(tension): engine inner — release planning, charging, stats"`

---

### Task 4: Target resolution and `db::top_nodes_in_domain`

**Files:**
- Modify: `src/db.rs`, `src/tension.rs`, `src/embed.rs` (poison-recovering lock)
- Test: `tests/digester.rs` (integration, real test graph) + unit test in `src/tension.rs`

**Interfaces:**
- Consumes: `db::find_similar_nodes(pool, &[f32], limit: i32, threshold: f32) -> Result<Vec<(i32, f32, String)>>`, `embed::embed_text(&str) -> anyhow::Result<Vec<f32>>`.
- Produces:
  - `pub async fn db::top_nodes_in_domain(pool: &PgPool, domain: &str, k: i64) -> Result<Vec<i32>, sqlx::Error>` — `SELECT id FROM memory_vectors WHERE domain = $1 ORDER BY importance DESC, id ASC LIMIT $2`.
  - `pub type EmbedFn = Arc<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync>`.
  - `pub async fn resolve_targets_with(pool: &PgPool, inner: &mut EngineInner, pending: Vec<PendingPulse>, embed: EmbedFn) -> Vec<TensionPulse>` — predictions → cached `top_nodes_in_domain(domain, K_PREDICTION_TARGETS)`; questions → embed **once per question via `tokio::task::spawn_blocking`** (embedding is CPU-bound) then `find_similar_nodes(pool, &emb, K_QUESTION_TARGETS, 0.6)`; on empty/err mark `unmapped` (question) or skip (prediction) and `inner.unmapped += 1`; one `TensionPulse` per target, energy split equally across targets.
  - `pub async fn resolve_targets(pool, inner, pending) -> Vec<TensionPulse>` = `resolve_targets_with(..., Arc::new(|t| crate::embed::embed_text(t)))`.
  - `src/embed.rs`: replace every `EMBEDDER.lock().unwrap()` with a poison-recovering lock (`lock().unwrap_or_else(|e| e.into_inner())`), so one panic while embedding cannot disable every later call.

- [ ] **Step 1: Write the failing test** in `tests/digester.rs`:

```rust
#[tokio::test]
async fn targets_resolve_from_domain_and_embedding() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = rgw::db::connect(&std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".into())).await?;
    sqlx::raw_sql(include_str!("schema.sql")).execute(&pool).await?;
    if rgw::db::graph_stats(&pool).await?.nodes == 0 { seed_database(&pool).await?; }
    let domain: String = sqlx::query_scalar("SELECT domain FROM memory_vectors GROUP BY domain ORDER BY count(*) DESC LIMIT 1").fetch_one(&pool).await?;
    assert_eq!(rgw::db::top_nodes_in_domain(&pool, &domain, 3).await?.len(), 3);
    assert!(rgw::db::top_nodes_in_domain(&pool, "no-such-domain", 3).await?.is_empty());
    let mut inner = rgw::tension::EngineInner::new(rgw::tension::TensionMode::Observe, 0.0);
    let pending = vec![
        rgw::tension::PendingPulse { source: rgw::tension::TensionSource::Prediction{domain: domain.clone()}, key: domain.clone(), energy: 0.15 },
        rgw::tension::PendingPulse { source: rgw::tension::TensionSource::Question{id: rgw::tension::question_id("is moral realism defensible")}, key: format!("{:016x}", rgw::tension::fnv1a64("is moral realism defensible")), energy: 0.20 },
    ];
    let pulses = rgw::tension::resolve_targets(&pool, &mut inner, pending).await;
    assert_eq!(pulses.len(), 6, "3 targets per item");
    assert!((pulses.iter().map(|p| p.energy).sum::<f32>() - 0.35).abs() < 1e-4, "energy split, not multiplied");
    Ok(())
}
```

Also a unit test in `src/tension.rs` for Review Focus 4, with the embedder **injected** so the test never depends on global embedder state:

```rust
#[tokio::test] async fn unmapped_question_is_counted_once_and_not_retried_within_cooldown() {
    let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
    let failing: EmbedFn = Arc::new(|_| Err(anyhow::anyhow!("no embedder in this test")));
    let mut inner = EngineInner::new(TensionMode::Observe, 0.0);
    let q = || PendingPulse { source: TensionSource::Question{id: question_id("q")}, key: format!("{:016x}", fnv1a64("q")), energy: 0.2 };
    assert!(resolve_targets_with(&pool, &mut inner, vec![q()], failing.clone()).await.is_empty());
    assert_eq!(inner.unmapped, 1);
    assert!(inner.questions.get(&fnv1a64("q")).unwrap().unmapped);
    assert!(resolve_targets_with(&pool, &mut inner, vec![q()], failing).await.is_empty());
    assert_eq!(inner.unmapped, 1, "not counted again");           // plan_release skips unmapped items until cooldown passes
}
```

- [ ] **Step 2: Run to verify failure.** `cargo test --test digester targets_resolve` and `cargo test --lib tension::unmapped_question` — Expected: compile errors (functions missing).
- [ ] **Step 3: Implement** both functions plus the `embed.rs` poison-recovery change. `plan_release` (Task 3) must skip a question whose state is `unmapped` until `ITEM_COOLDOWN_SECS` since `last_release`, then clear `unmapped` and let it retry. Question embeddings are cached in `QuestionState.targets` (resolved once); prediction targets cached in `inner.prediction_targets` with the release timestamp.
- [ ] **Step 4: Run to verify pass.** `cargo test --test digester targets_resolve` and `cargo test --lib tension::` — Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -am "feat(tension): resolve pulse targets by domain importance and embedding similarity"`

---

### Task 5: Diverger `Stimulus` inbox, fatigue, cascade budget, origin tagging

**Files:**
- Modify: `src/diverger.rs` (struct ~44–62, `NodeEnergy`, `new()` ~139, `start_reactor` ~158, energy loop ~250–300, fire loop ~340–470, `notify_edge_change` ~476, `stats()` ~505)
- Test: unit tests in `src/diverger.rs` for the pure helpers; existing suite stays green.

**Interfaces:**
- Consumes: `tension::{TensionPulse, Origin, WouldHave, TensionMode, effective_threshold, decay_fatigue, CASCADE_BUDGET, HOP_COST, FATIGUE_PER_FIRE, FATIGUE_DECAY_PER_SEC}`.
- Produces:
  - `pub struct Stimulus { pub kind: StimulusKind, pub origin: Origin, pub depth: u32, pub budget: f32 }`, `pub enum StimulusKind { Edge(EdgeChange), Tension(TensionPulse) }`, `impl Stimulus { pub fn external_edge(c: EdgeChange) -> Self /* origin Edge, depth 0, budget CASCADE_BUDGET */ }`.
  - `pub fn next_hop(s: &Stimulus, injected: f32, max_depth: u32) -> Option<(u32, f32)>` — pure: returns `Some((depth+1, budget - HOP_COST - injected))` when the new budget > 0 and depth+1 < max_depth, else `None`.
  - `NodeEnergy` gains `pub fatigue: f32`.
  - `notify_edge_change(&self, change: EdgeChange)` unchanged in signature; sends `Stimulus::external_edge(change)`. New `pub fn notify_stimulus(&self, s: Stimulus)` (same `try_send` + drop counter) and `pub fn stimulus_sender(&self) -> mpsc::Sender<Stimulus>`.
  - `DivergerStats` gains nothing new here (tension stats come from the engine in Task 7).
  - Reactor behaviour: for `StimulusKind::Tension(p)` add `p.energy` to `p.target_node` (same decay-then-add path as an edge endpoint). On every node touch: `fatigue = decay_fatigue(fatigue, dt)`. Threshold check uses `effective_threshold(threshold, node.fatigue)` **only when `mode.enforces_brakes()`**; otherwise compute it, and if the node would have been blocked by fatigue but fires under the plain threshold, call `tension::record_would_have(Fatigue)`. On fire: `fatigue += FATIGUE_PER_FIRE`. The `Stimulus`'s `origin/depth/budget` are carried into the spawned walk task.

- [ ] **Step 1: Write the failing tests** in `src/diverger.rs` `mod tests`:

```rust
#[test] fn next_hop_ends_by_budget_then_by_depth() {
    let s = Stimulus::external_edge(EdgeChange{edge_id:1,source_id:1,target_id:2,delta:0.1,edge_type:"t".into()});
    let (d1, b1) = next_hop(&s, 0.1, 3).unwrap();  assert_eq!(d1, 1); assert!((b1 - 0.7).abs() < 1e-6);
    let s2 = Stimulus { depth: d1, budget: b1, ..s.clone() };
    let (d2, b2) = next_hop(&s2, 0.1, 3).unwrap();  assert_eq!(d2, 2); assert!((b2 - 0.4).abs() < 1e-6);
    let s3 = Stimulus { depth: d2, budget: b2, ..s.clone() };
    assert!(next_hop(&s3, 0.1, 3).is_none(), "depth 3 reached");
    let s4 = Stimulus { depth: 0, budget: 0.25, ..s.clone() };
    assert!(next_hop(&s4, 0.1, 10).is_none(), "budget exhausted first");
}
```
(`Stimulus` and `EdgeChange` need `Clone`.)

- [ ] **Step 2: Run to verify failure.** `cargo test --lib diverger::` — Expected: compile error.
- [ ] **Step 3: Implement** the inbox change and reactor behaviour per Interfaces. The reactor reads `mode` from `tension::mode()` (Task 6 provides it; until then use `TensionMode::Off` via a `tension::mode()` stub added now returning `Off` when no engine is installed). Keep `max_cascade_depth` as the depth limit passed to `next_hop`.
- [ ] **Step 4: Run to verify pass.** `cargo test` — Expected: all previous tests + 1 new pass; integration unchanged.
- [ ] **Step 5: Commit.** `git commit -am "feat(diverger): Stimulus inbox with origin/depth/budget, node fatigue, tension energy"`

---

### Task 6: Install the engine; the `after_walk` hook; walker call site

**Files:**
- Modify: `src/tension.rs`, `src/walker.rs` (~line 532, after `aggregate`), `src/api.rs` (`serve()` ~line 85, next to `motor::install_gate`)
- Test: unit test in `src/tension.rs`; integration in `tests/digester.rs`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `pub struct TensionEngine { inner: tokio::sync::Mutex<EngineInner>, inbox: OnceLock<mpsc::Sender<Stimulus>> }`; `pub fn install(mode: TensionMode) -> Option<Arc<TensionEngine>>` (OnceLock; `Off` installs nothing and returns `None`, so `engine()` stays `None`; a second call returns the already-installed engine); `pub fn engine() -> Option<Arc<TensionEngine>>`; `pub fn mode() -> TensionMode` (`Off` when none).
  - `impl TensionEngine { pub fn attach_inbox(&self, tx: mpsc::Sender<Stimulus>); pub async fn after_walk(&self, pool: &PgPool, sm: &Arc<tokio::sync::Mutex<SelfModel>>) -> usize /* pulses emitted */; pub async fn stats(&self, sm: &SelfModel) -> TensionStats; pub async fn record_edge_cascade_hops(&self, hops: u32); pub async fn record_cascade_end(&self, origin: Origin, by_budget: bool); pub async fn record_would_have(&self, w: WouldHave); }`
  - Free fn `pub async fn after_walk(pool: &PgPool, sm: &Arc<tokio::sync::Mutex<SelfModel>>) -> usize` — no-op returning 0 when `engine()` is `None`. Sequence inside the engine: lock `sm`, `record_resolution_sample(made, resolved, now)`, `plan_release`, unlock; `resolve_targets`; `charge`; for each returned pulse `inbox.try_send(Stimulus { kind: Tension(p), origin: Tension, depth: 0, budget: CASCADE_BUDGET })`, counting `dropped_inbox_full` on `Full`.
  - `api::serve`: `let tension_engine = crate::tension::install(TensionMode::from_env_value(std::env::var("RGW_TENSION_MODE").ok().as_deref()));` immediately after the motor gate; after `Diverger::new(...)`: `if let Some(e) = &tension_engine { e.attach_inbox(diverger.stimulus_sender()); }`. Log `[tension] mode=<mode>` at INFO at startup, in every mode including `off`.
  - `walker::walk_parallel`: directly after `let output = aggregate(...)`, call `let _ = crate::tension::after_walk(pool, self_model).await;`.

- [ ] **Step 1: Write the failing tests**:

```rust
// src/tension.rs
#[tokio::test] async fn after_walk_without_engine_is_a_noop() {
    let pool = sqlx::PgPool::connect_lazy("postgres://x:x@127.0.0.1:1/x").unwrap();
    let sm = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
    assert_eq!(after_walk(&pool, &sm).await, 0);            // engine() is None in this test binary
    assert_eq!(mode(), TensionMode::Off);
}
```
```rust
// tests/digester.rs
#[tokio::test]
async fn observe_mode_logs_would_fire_and_emits_nothing() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = rgw::db::connect(&std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".into())).await?;
    sqlx::raw_sql(include_str!("schema.sql")).execute(&pool).await?;
    if rgw::db::graph_stats(&pool).await?.nodes == 0 { seed_database(&pool).await?; }
    let engine = rgw::tension::install(rgw::tension::TensionMode::Observe);
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    engine.as_ref().unwrap().attach_inbox(tx);
    let domain: String = sqlx::query_scalar("SELECT domain FROM memory_vectors GROUP BY domain ORDER BY count(*) DESC LIMIT 1").fetch_one(&pool).await?;
    let sm = Arc::new(tokio::sync::Mutex::new(rgw::core::SelfModel::new()));
    { let mut g = sm.lock().await; g.mode = rgw::core::CognitiveMode::Autonomous; g.predict(&domain, "x", 0.5); g.predictions.get_mut(&domain).unwrap().predicted_at -= 3600.0; g.open_questions.push("what is a dead end".into()); }
    let (_out, _) = rgw::walker::walk_parallel(&pool, &rgw::core::EmotionalState::default(), 2, 3, &sm).await;
    let stats = engine.as_ref().unwrap().stats(&*sm.lock().await).await;
    assert!(stats.pulses_would_fire >= 1, "{stats:?}");
    assert_eq!(stats.pulses_fired, 0);
    assert!(rx.try_recv().is_err(), "nothing entered the inbox in observe");
    Ok(())
}
```
(`install` is process-global: this integration test must be the only one in its binary that installs, or it must run in its own `#[test]` binary; put it in a new file `tests/tension_observe.rs` that reuses the seed helper via `mod common;` if needed.)

- [ ] **Step 2: Run to verify failure.** `cargo test --lib tension::after_walk_without_engine` and `cargo test --test tension_observe` — Expected: compile errors.
- [ ] **Step 3: Implement** per Interfaces; wire `api::serve` and `walker.rs`.
- [ ] **Step 4: Run to verify pass.** `cargo test` — Expected: all green, including both new tests.
- [ ] **Step 5: Commit.** `git commit -am "feat(tension): engine install, after_walk hook, walker + serve wiring"`

---

### Task 7: Reactor feedback hop, budget/pool enforcement, stats endpoint

**Files:**
- Modify: `src/diverger.rs` (spawned walk task ~405–470: after `integrate_background_walk`), `src/api.rs` (`diverger_stats` ~1003)
- Test: unit test in `src/diverger.rs`; manual verification steps below.

**Interfaces:**
- Consumes: Task 5 `next_hop`, Task 6 engine methods, `tension::mode()`.
- Produces:
  - Note on semantics (spec §7): enqueueing feedback hops is what makes cascades multi-hop, so `brakes` means "self-propagation plus the brakes, without tension". `observe` computes the same decision and only counts it.
  - In the spawned walk task after a spontaneous walk completes: (a) `tension::after_walk(&pool, &self_model).await`; (b) for the walk's `result.edges_traversed`, compute `next_hop(&stimulus, injected = 0.02 * edges.len() as f32, config.max_cascade_depth)`. If `Some((depth, budget))` **and** `mode().enforces_brakes()` **and** the engine's pool `try_spend(HOP_COST)` succeeds: enqueue `Stimulus { kind: Edge(EdgeChange{…the **last** traversed edge, i.e. where the walk ended up…, delta: 0.02}), origin: stimulus.origin, depth, budget }` via `notify_stimulus` (the first edge would bias the cascade back to where it started), then `record_edge_cascade_hops(depth)` when the chain ends. If `None` → `record_cascade_end(origin, by_budget = budget_exhausted)`. If the mode does not enforce brakes → compute the same decision and call `record_would_have(Budget | Pool)` instead of enqueueing. In `Off` do nothing (today's behaviour).
  - `api::diverger_stats` adds `"tension": <TensionStats as JSON>` when `tension::engine()` is `Some`, else `"tension": {"mode": "off"}`.
  - README: a short "Tension-driven activity" subsection under Architecture naming the env var, the four modes, and the `/diverger.tension` block.

- [ ] **Step 1: Write the failing test** in `src/diverger.rs`:

```rust
#[test] fn feedback_decision_matches_mode() {
    // pure helper extracted for this: decide_feedback(mode, next: Option<(u32,f32)>, pool_ok: bool) -> Feedback
    use crate::tension::TensionMode::*;
    assert!(matches!(decide_feedback(Off,     Some((1, 0.7)), true),  Feedback::None));
    assert!(matches!(decide_feedback(Observe, Some((1, 0.7)), true),  Feedback::WouldEnqueue));
    assert!(matches!(decide_feedback(Observe, Some((1, 0.7)), false), Feedback::WouldDrop(crate::tension::WouldHave::Pool)));
    assert!(matches!(decide_feedback(Brakes,  Some((1, 0.7)), true),  Feedback::Enqueue(1, _)));
    assert!(matches!(decide_feedback(Live,    None,           true),  Feedback::End{by_budget: true}));
}
```
with `pub enum Feedback { None, WouldEnqueue, WouldDrop(WouldHave), Enqueue(u32, f32), End { by_budget: bool } }` and `pub fn decide_feedback(mode: TensionMode, next: Option<(u32, f32)>, pool_ok: bool) -> Feedback`.

- [ ] **Step 2: Run to verify failure.** `cargo test --lib diverger::feedback_decision` — Expected: compile error.
- [ ] **Step 3: Implement** `decide_feedback`, use it in the walk task, wire the stats endpoint, add the README subsection.
- [ ] **Step 4: Run to verify pass.** `cargo test` green; `cargo clippy --quiet 2>&1 | grep -c '^warning'` ≤ 57.
- [ ] **Step 5: Manual verification** against the local Julian graph (exact commands):

```bash
# observe (default): must log would-fire and never emit
DATABASE_URL='postgresql:///rgw_julian?host=/tmp' RGW_API_KEY=$(cat "$SCR/ab/rgw_key") RGW_MOTOR_MODE=off WALKER_PORT=11438 RUST_LOG=info ./target/debug/rgw > /tmp/rgw_obs.log 2>&1 &
# fire 3 walks, then:
curl -s -H "X-RGW-Key: $KEY" http://127.0.0.1:11438/diverger | python3 -c "import json,sys; t=json.load(sys.stdin)['tension']; print(t['mode'], t['pulses_would_fire'], t['pulses_fired'], t['brakes_would_have'])"
grep -c 'would-fire' /tmp/rgw_obs.log      # ≥ 1
grep -c '\[tension\] fire ' /tmp/rgw_obs.log  # 0
# off: /diverger.tension == {"mode":"off"} and no [tension] lines at all
```
Expected: `observe N 0 {...}` with N ≥ 1; the `off` run prints no `[tension]` lines.

- [ ] **Step 6: Commit.** `git commit -am "feat(diverger): feedback hop under budget/pool, brakes enforcement by mode, tension stats on /diverger"`

---

### Task 8: Final checks and PR

- [ ] **Step 1:** `cargo test` (unit + integration), `cargo clippy --quiet` count ≤ 57, `cargo build --release`.
- [ ] **Step 2:** Confirm `off` is byte-for-byte: run the existing `test_digest_philosophy` with and without `RGW_TENSION_MODE=off` set in the environment of a server start; `DivergerStats` JSON identical in shape except the `tension` key.
- [ ] **Step 3:** Open the PR with: the spec link, the mode table, the manual verification output, and the deploy note "ships in `observe`; promotion is a config change per spec §7/§9". Vendored sync PRs follow the merge as usual.
