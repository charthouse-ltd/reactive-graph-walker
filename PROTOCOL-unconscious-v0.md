# Protocol: Unconscious Walker v0 — Observability Mode

**Date**: 2026-04-27
**Scope**: Minimal implementation of the unconscious walker in read-only Observability mode. Ships alongside production without affecting subscriber-facing behaviour.
**Status**: Implementation plan — not yet built.
**Depends on**: `PROTOCOL-compliance-mode.md`, `PROTOCOL-unconscious-walker.md`.
**Estimated effort**: 2-4 days of focused work.

---

## Goal

Run the unconscious walker continuously in production to collect real-world data on what it would do — without any path by which it can affect subscriber-facing output. Validate the design against actual graph dynamics before promoting to any mode that mutates state or drives motor commands.

This is step 1 of the six-step graduated rollout defined in `PROTOCOL-unconscious-walker.md` § "Deployment with paying users."

## Non-goals (deliberately deferred to later stages)

- Hebbian edge strengthening (step 2)
- SHY-style proportional downscaling (step 3)
- Power-law decay replacing `/prune` exponential (step 4)
- Shadow Waking that produces logged-but-not-executed motor commands (step 5)
- Real broadcasts to production Waking (step 6)
- Successor-representation matrix
- Hippocampal-style sparse encoding
- Schema-violation salience component (requires schema centroids — defer)
- Pattern-breaking, novel edge synthesis, reconsolidation
- Federation between RGW instances

Anything in `PROTOCOL-unconscious-walker.md` not explicitly listed in "Implementation" below is out of scope for v0.

## Production safety invariants (verifiable each release)

These are the property-test contracts v0 must hold:

1. **No graph mutation.** After N unconscious cycles in Observability mode, the graph (nodes, edges, weights, traversal counts) is byte-identical to before. Diff-based test against a snapshot.
2. **No motor command emission.** No `MotorCommand` is dispatched whose source is a walker tagged `kind: unconscious_walk`. Verified by trace inspection in tests.
3. **No self-model mutation from unconscious walks.** The four-step primitive (`observe → influence → mutate → notice`) skips the `mutate` step for `kind: unconscious_walk` signals.
4. **Wake event rate within budget.** Observability would-have-woken rate must stay below configured ceiling; alarm if exceeded.
5. **Off by default.** `UNCONSCIOUS_ENABLED=false` is the default — explicit opt-in required to enable.

## Implementation

### `src/core.rs`

- Extend the unified `Signal::kind` taxonomy with a new variant `unconscious_walk`.
- In `process()`, when `signal.kind == unconscious_walk` and `mode == Observability`: execute `observe` and `notice`, **skip** `influence` and `mutate`. The signal is purely observational.
- Add `WalkerMode::Unconscious(UnconsciousMode)` enum:
  ```rust
  pub enum UnconsciousMode {
      Observability,  // v0 — read-only, log-only
      // Hebbian, SHY, ShadowWake, Live — added in later steps
  }
  ```
  For v0, only `Observability` is valid.

### `src/unconscious.rs` (new file)

The minimal long-lived task. Pseudocode:

```rust
pub async fn start_unconscious_walker(
    pool: PgPool,
    self_model: Arc<RwLock<SelfModel>>,
    config: UnconsciousConfig,
) {
    let mut interval = tokio::time::interval(config.walk_interval);
    loop {
        interval.tick().await;
        if !config.enabled { continue; }

        // 1. Pick seed: random node weighted by recent access (read-only)
        let seed = pick_seed_node(&pool, &config).await;

        // 2. Run a single walk using compliant scoring (no emotion, no mutation)
        let walk_start = Instant::now();
        let result = walk_single_readonly(
            &pool,
            seed,
            WalkerBias::Experience,
            config.steps,
            &self_model.read().await,
        ).await;

        // 3. Compute wake-score components
        let scores = compute_wake_scores(&result, &self_model.read().await, &pool).await;

        // 4. Persist observation — never mutate graph
        write_observation(&pool, seed, &result, &scores, walk_start.elapsed()).await;

        // 5. If would_have_woken, log loudly but do nothing else
        if scores.composite >= config.wake_threshold {
            log_would_have_woken(&result, &scores);
        }
    }
}
```

Key properties:

- `walk_single_readonly` is a fork of `walk_single` that does not increment edge `traversal_count` and does not call any path that strengthens weights.
- `compute_wake_scores` for v0 implements only `relevance` and `novelty`. `schema_violation = 0.0` (deferred). `cost = 0.0` (no broadcasts to penalize).
- The composite score: `composite = 0.6 * relevance + 0.4 * novelty`. Weights are config, tunable without redeploy.

### Salience scoring — v0 simplification

- **relevance**: cosine similarity between the mean embedding of nodes visited in the walk and the self-model's "active concerns" embedding.
  - "Active concerns" for v0 = mean embedding of nodes touched by Waking walks in the last hour. Cached and refreshed every 5 minutes.
- **novelty**: `1 - max_sim`, where `max_sim` is the highest cosine similarity between the current walk's path-embedding and the last 100 unconscious walks' path-embeddings.
- **schema_violation**: `0.0` for v0. Hooked into the schema, but the schema-centroid infrastructure is not built yet. Reserved column in the observation table.
- **cost**: `0.0` for v0. We never broadcast, so there is nothing to budget.

### `src/db.rs`

New table:

```sql
CREATE TABLE unconscious_observations (
    id              BIGSERIAL PRIMARY KEY,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    walk_id         UUID NOT NULL,
    seed_node_id   INT NOT NULL,
    path_node_ids  INT[] NOT NULL,
    domains_visited TEXT[] NOT NULL,
    edges_seen      INT NOT NULL,
    walk_duration_ms INT NOT NULL,

    -- Wake score components
    relevance       REAL NOT NULL,
    novelty         REAL NOT NULL,
    schema_violation REAL NOT NULL DEFAULT 0.0,
    cost_penalty    REAL NOT NULL DEFAULT 0.0,
    composite       REAL NOT NULL,
    would_have_woken BOOL NOT NULL,

    -- Self-model snapshot at walk time
    cognitive_mode  TEXT NOT NULL,
    valence         REAL,
    arousal         REAL,
    energy          REAL
);

CREATE INDEX idx_unconscious_obs_observed_at
    ON unconscious_observations (observed_at DESC);
CREATE INDEX idx_unconscious_obs_would_woken
    ON unconscious_observations (would_have_woken)
    WHERE would_have_woken = TRUE;
```

Retention: 14 days by default, configurable. Deletion runs nightly via existing pruning task.

New functions:

- `insert_observation(pool, obs) -> Result<(), Error>`
- `recent_observations(pool, limit, since) -> Result<Vec<Observation>, Error>`
- `would_have_woken_count(pool, since) -> Result<i64, Error>`

### `src/api.rs`

New endpoints (admin / local-only — no public exposure):

- `POST /unconscious/start` — idempotent, sets `enabled=true` at runtime
- `POST /unconscious/stop` — idempotent, sets `enabled=false`
- `GET /unconscious/stats` — aggregates over last 24h: walk count, mean composite score, would-have-woken count, top-5 domains visited
- `GET /unconscious/observations?limit=100&since=...` — paginated raw log
- `GET /unconscious/health` — is the walker task alive, last walk timestamp, error rate

All endpoints behind the same auth as existing admin endpoints.

### `src/main.rs`

Read config:

- `UNCONSCIOUS_ENABLED` (bool, default `false`)
- `UNCONSCIOUS_WALK_INTERVAL_MS` (default `10000`)
- `UNCONSCIOUS_STEPS` (default `5`)
- `UNCONSCIOUS_WAKE_THRESHOLD` (default `0.7` — placeholder, retune from data)
- `UNCONSCIOUS_RELEVANCE_WEIGHT` (default `0.6`)
- `UNCONSCIOUS_NOVELTY_WEIGHT` (default `0.4`)
- `UNCONSCIOUS_OBS_RETENTION_DAYS` (default `14`)

If enabled at startup, spawn `start_unconscious_walker` after the existing Diverger / Walker setup completes. If disabled, skip — code path is otherwise inert.

## What stays the same

- The existing waking walker (`walker.rs`) and dream walker (`dream.rs`) are untouched.
- The Diverger (`diverger.rs`) is untouched. The unconscious does not subscribe to its cascade in v0.
- The motor router on the Python side (`routers/rgw.py`) is untouched. v0 emits no motor commands.
- The compliance mode (`PROTOCOL-compliance-mode.md`) is untouched. Observability is additive.
- Database schema additions are forward-compatible — no existing tables modified.

## Validation plan — what the first week of data should show

Run the walker for 7 days in dev/staging, then 7 days in prod with `UNCONSCIOUS_ENABLED=true`. Acceptance criteria for promoting to step 2 (Hebbian-only):

1. **Cadence holds.** ~60K observations per week at 10s interval. Allow ±10% tolerance.
2. **Score distribution is shaped.** Composite scores show a real distribution — not all near 0 (criterion too strict), not all near 1 (criterion too loose). Aim for ~5-20% above the wake threshold.
3. **Would-have-woken rate is sensible.** Between 5 and 50 per day. Outside that range = retune weights or threshold before promotion.
4. **Domain coverage is broad.** Walks visit ≥80% of active domains over the week. If stuck in a corner, seed selection needs work.
5. **No graph mutation.** Property test passes on every deploy.
6. **CPU and memory steady.** No leaks. <5% CPU overhead measured against baseline RGW load.
7. **The would-have-woken events read sensibly.** Manual review of 20 random would-have-woken events: do they correspond to moments where Julian *should* have noticed something? Subjective but informative.

If criteria 1-6 hold and criterion 7 is plausible, promote to step 2. If any fail, fix before promotion.

## Rollback

- `POST /unconscious/stop` halts the walker immediately
- `UNCONSCIOUS_ENABLED=false` + restart — no walker on next boot
- No data migration needed for rollback — the observation table is additive
- Drop `unconscious_observations` is safe at any time (audit log only, not load-bearing)

## Open items for v0 implementation

- Exact mechanism for "active concerns" embedding cache — Postgres view, in-memory cache, or recomputed per walk? Lean toward in-memory with 5-min TTL for simplicity.
- Whether to use `tokio::time::interval` or a self-clocked loop synced to the Diverger. v0 uses `interval` for simplicity; revisit if the Diverger's cascade clock would give better timing.
- Whether observation writes batch (e.g., 10 at a time) or write-per-walk. Write-per-walk for v0; batch only if Postgres load shows up.
- Walker bias for v0: locked to `Experience`. Adding `Analytical` is a v1 concern.
- Cold-start: if there are zero recent Waking walks, what is "active concerns"? v0 default — null embedding, relevance falls back to global graph centroid.
