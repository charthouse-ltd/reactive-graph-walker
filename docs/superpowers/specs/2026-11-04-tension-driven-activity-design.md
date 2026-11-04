# Tension-driven background activity — design

**Status:** approved design, not implemented. **Date:** 2026-11-04.
**Scope:** RGW only (`reactive-graph-walker`). No backend change, no schema change.

## 1. Problem

The diverger is RGW's always-on reactor, but its only input is the HTTP API
(`notify_edge_change` is called from three API handlers and nowhere else). Between
external calls the graph does nothing. `max_cascade_depth` exists in config and is
never read, so when something does arrive there is no principled end to a cascade,
only energy decay.

We want the graph to stay active *between* stimuli, driven by what is unresolved in
its own state rather than by a timer or by random noise, and we want that activity
to be bounded by construction. An A/B on 2026-11-04 showed that unselected activity
(random nodes) makes output worse than no context at all, so "spontaneous" must mean
"purposeful", never "noisy".

## 2. Non-goals

- No change to walk scoring, fitness, selection, or dream consolidation.
- No LLM calls. Tension cascades are compute-only, like all diverger walks.
- No new persistence beyond two counters on the self-model (§7).
- No change in behaviour when `RGW_TENSION_MODE=off`; `observe` (the default)
  emits nothing into the graph.
- Wounds and the motor layer are untouched.

## 3. Sources of tension

Both already exist on `SelfModel` (`src/core.rs`).

### 3.1 Unresolved predictions
`predictions: HashMap<String, ExpectedOutcome>` keyed by domain. An entry is created
by `predict()` and removed when a `prediction_error` signal for that domain arrives.
While it exists it is tension.

Pulse strength for a prediction of age `a` seconds:

```
strength(a) = min(1.0, a / PREDICTION_SATURATION_SECS) * PREDICTION_BASE_ENERGY
```

Defaults: `PREDICTION_SATURATION_SECS = 3600`, `PREDICTION_BASE_ENERGY = 0.15`.
A prediction stops pulsing when resolved. The engine never abandons a prediction:
resolution is the only exit, and a prediction that stays open keeps counting as open
in the stats, which is itself a signal worth seeing. Two limits stop stale
predictions from starving everything else: strength halves once a prediction is
older than `PREDICTION_HALF_LIFE_SECS` (86400) and halves again every further
half-life, and at most `PREDICTION_MAX_ACTIVE` (10) predictions pulse per release,
the freshest first. (Without these, forty stale predictions at full strength on the
120 s cooldown would consume the entire 0.05/s pool refill by themselves.)

### 3.2 Open questions
`open_questions: Vec<String>`. A question has no resolving signal, so it must not
pulse forever. The engine keeps a side table keyed by the question text:

```
QuestionState { strength: f32, attempts: u32, abandoned: bool, targets: Vec<i32> }
```

Initial `strength = QUESTION_BASE_ENERGY` (0.20). Each release multiplies strength by
`QUESTION_DECAY` (0.85) and increments `attempts`. When `attempts >=
QUESTION_MAX_ATTEMPTS` (12) or `strength < MIN_PULSE_ENERGY` (0.02) the question is
marked `abandoned`, counted, and never pulses again. A question removed from
`open_questions` drops out of the side table. Strength and attempts are in-memory,
but the **abandoned set is persisted** as `abandoned_question_hashes: Vec<u64>`
(insertion-ordered, deduplicated on insert, serde default empty) on `SelfModel`, so
a container recreate does not resurrect every question the engine already gave up
on. It does not grow without bound: a hash is removed when its question leaves
`open_questions`, and the vector is capped at `ABANDONED_HASHES_MAX` (4096),
dropping the oldest first. The hash is a stable FNV-1a 64 of the
question text, implemented inline (no crate), and the same id appears in logs and
stats as `question:<16 hex>`. `std::hash::DefaultHasher` is not used anywhere for
identity because it is not stable across Rust versions.

## 4. Target mapping

Mapping is computed once per item and cached in the item's state.

- **Prediction (domain `d`)**: the top `K_PREDICTION_TARGETS` (3) nodes in domain `d`
  by `importance`, via a new `db::top_nodes_in_domain(pool, domain, k)`.
- **Open question (text `q`)**: embed `q` once with `embed::embed_text` **on the
  blocking pool** (`spawn_blocking`; embedding is CPU-bound and must not stall the
  runtime), then the top `K_QUESTION_TARGETS` (3) nodes from the existing
  `db::find_similar_nodes`. The embedder's mutex must recover from poisoning
  rather than `unwrap()`, so one panic while embedding cannot disable every later
  call.

An item whose mapping returns no nodes is marked `unmapped` and skipped (counted).

## 5. Emission: when and how

### 5.1 Trigger
Release runs **at the end of every walk**: in `walker::walk_parallel` (covers
`/walk`, the chat path and the internal caller) and after each spontaneous walk the
diverger fires. There is no timer. If the graph is quiet, tension stays latent until
the next external stimulus. This is the approved trigger; changing it is a one-line
decision at the call sites.

### 5.2 Inbox
The diverger inbox carries a `Stimulus` instead of a bare `EdgeChange`:

```rust
pub enum Stimulus {
    Edge(EdgeChange),
    Tension(TensionPulse),
}
pub struct TensionPulse {
    pub source: TensionSource,   // Prediction { domain } | Question { key }
    pub target_node: i32,
    pub energy: f32,
}
```

Same bounded channel (`EDGE_QUEUE_CAPACITY`), same `try_send` + drop counter.
`notify_edge_change` keeps its signature and wraps into `Stimulus::Edge`.

### 5.3 Reactor handling
A `Tension` stimulus adds `energy` to `target_node` exactly as an edge change adds
`delta * propagation_strength` to its endpoints. From there the existing threshold,
refractory and cascade logic applies, with the brakes in §6. A cascade started by a
tension pulse is tagged `origin = Tension` for the metrics in §8.

### 5.4 Per-item cooldown
An item releases at most once per `ITEM_COOLDOWN_SECS` (120) regardless of how many
walks complete, so a burst of walks cannot drain the pool through one question.

## 6. The brake (three layers, shipped together with §5)

### 6.1 Node fatigue
`DivergerNode` gains `fatigue: f32`. On fire: `fatigue += FATIGUE_PER_FIRE` (0.5).
On every touch: `fatigue = max(0, fatigue - FATIGUE_DECAY_PER_SEC * dt)` (0.01/s).
Effective threshold: `threshold * (1.0 + fatigue)`. Recently fired regions therefore
cost more to re-fire and cascades drift to fresh territory. This is the mechanism
that addresses the monoculture risk at the activity level.

### 6.2 Per-cascade budget
Every cascade (edge- or tension-originated) carries `budget: f32`, initialised to
`CASCADE_BUDGET` (1.0). Each hop spends `HOP_COST` (0.2) plus the energy it injects.
A cascade ends when budget ≤ 0 or depth reaches `max_cascade_depth` (the existing
config field, now read). Cascades ended by budget are counted separately from those
that died out naturally.

### 6.3 Global pool
A token bucket shared by the whole reactor:

```
GlobalPool { capacity: 20.0, tokens: f32, refill_per_sec: 0.05, last_refill: Instant }
```

Every tension pulse spends its energy from the pool before it is emitted; every
cascade hop spends `HOP_COST`. If the pool cannot cover the spend, the pulse (or hop)
is dropped and counted as `dropped_pool_empty`. Edge changes from the API are **not**
charged for their initial injection (external stimuli always land) but their
cascade hops are. The pool is the direct CPU bound: at defaults the reactor cannot sustain more
than 0.05 energy-units of self-generated work per second, about one modest cascade
per 20 s, with bursts up to 20 units.

In modes where tension is not emitted (§7), a would-be pulse is charged
`energy + avg_edge_cascade_hops * HOP_COST`, where `avg_edge_cascade_hops` is a
rolling mean over the last 100 edge-originated cascades (default 3 until there are
samples). Charging only the pulse energy would make the pool look healthier than
`live` will be, because in `live` every tension cascade also spends per hop.

## 7. Staging and configuration

`RGW_TENSION_MODE` env var: `off | observe | brakes | live`, default **`observe`**.

- `off`: the engine is not constructed. Byte-for-byte today's behaviour. Rollback.
- `observe` (default): **no behaviour change**. The engine computes every release
  and logs each would-be pulse at INFO:
  `[tension] would-fire source=prediction:markets target=4312 energy=0.11 pool=17.4`.
  Nothing enters the inbox. The three brakes are computed but **not enforced**:
  edge-originated cascades run exactly as today (single hop, no feedback), and the
  engine logs what fatigue, the cascade budget and the pool *would have* cut,
  dropped, or enqueued as a next hop. The §9.1 baseline is
  measured in this mode, so it is a true baseline.
- `brakes`: **self-propagation plus the brakes, without tension.** A completed
  spontaneous walk feeds its last traversed edge back into the inbox as the next
  hop (this is what makes cascades multi-hop at all, and is the fix for the
  "diverger is not self-propagating" finding), and the three brakes are enforced
  on those cascades. Tension is still not emitted. Note that `brakes` therefore
  changes edge-originated activity relative to `observe`; that is intended, and
  §9.3 is read with it in mind.
- `live`: brakes enforced, pulses enter the inbox.

The deploy order is `observe` → `brakes` → `live`, each a config change.

Persisted on `SelfModel` (all serde-default, so old snapshots load):
`predictions_made_total: u64` (incremented in `predict()`),
`predictions_resolved_total: u64` (incremented where the `prediction_error` path
removes an entry), and `abandoned_question_hashes: Vec<u64>` (§3.2). The two
counters exist for the metric in §9, not for the engine.

## 8. Observability

`GET /diverger` gains:

```json
"tension": {
  "mode": "observe",            // off | observe | brakes | live
  "brakes_would_have": { "fatigue_blocks": 5, "budget_cuts": 3, "pool_drops": 12 },
  "open_predictions": 4, "open_questions": 7, "abandoned_questions": 2, "unmapped": 0,
  "pulses_would_fire": 318, "pulses_fired": 0,
  "dropped_pool_empty": 12, "dropped_item_cooldown": 90, "dropped_inbox_full": 0,
  "cascades": { "tension_origin": 0, "edge_origin": 41, "ended_by_budget": 3, "ended_naturally": 38 },
  "pool": { "tokens": 17.4, "capacity": 20.0, "empty_fraction_1h": 0.01 },
  "predictions_made_total": 120, "predictions_resolved_total": 97,
  "resolution_rate_1h": 0.81,
  "recent": [ { "t": 1762264800, "source": "question:7f3a", "target": 4312, "energy": 0.11, "fired": false } ]
}
```

`recent` is a ring buffer of the last 50 events. A question is identified by
`question:<16 hex>`, the FNV-1a 64 hash of its text (§3.2), so the log never carries
the question itself and ids are stable across builds. `empty_fraction_1h` is the fraction of pool
spend attempts in the last hour that found fewer tokens than `MIN_PULSE_ENERGY`;
it is computed from a ring of (timestamp, empty: bool) samples taken on each attempt.
`resolution_rate_1h` is computed from (timestamp, made, resolved) samples that the
engine records **in `after_walk`**, i.e. on every walk, never inside the stats
read, so the rate does not depend on how often `/diverger` is polled. All counters
are in-memory except the two self-model totals.

## 9. Pre-registered promotion metrics (observe → live)

Decided now, before any data:

**Promotion `observe` → `brakes` → `live`** (judged on ≥ 48 h in each mode):

1. **Resolution rate** `resolved / made` over a rolling hour must not fall below the
   baseline measured in `observe` (where nothing is enforced). Tracking `made`
   alongside `resolved` prevents gaming by predicting less or predicting easier things.
2. **Bounded**: `empty_fraction_1h` of the global pool < 0.05 and
   `ended_by_budget / total cascades` < 0.20. In `observe` and `brakes` these are
   computed from the estimated charges (§6.3) and are a leading indicator only; they
   are confirmed in `live`.
3. **No regression** in the existing walk telemetry (agreement, novelty, hops/s)
   relative to the `observe` period. For `observe` → `brakes` this check is also
   the first look at multi-hop self-propagation; a change in diverger
   `walks_fired` is expected there and is not by itself a regression.

**Staying in `live`** (judged after `LIVE_JUDGEMENT_HOURS` = 72, then weekly):

4. **Useful edges**: the fraction of tension-originated cascades that create a
   cross-domain edge which still exists after the next dream consolidation must be
   ≥ the same fraction for edge-originated cascades over the same window. This can
   only be measured once tension cascades exist, so it is a condition for staying in
   `live`, not for entering it.
5. Criteria 1–3 continue to hold.

Any failure → drop one mode and revisit parameters. Only after `live` has held all
five for a week does the mention-reply A/B harness get rerun; it is the final arbiter
of user-visible value, not the first.

## 10. Testing

Unit tests (pure functions, no DB):
- prediction `strength(age)` saturates and is monotone;
- question decay reaches `abandoned` by attempts and by strength floor, and an
  abandoned question never pulses again;
- token bucket: spend/refill arithmetic, cannot go negative, refill capped at capacity;
- fatigue: threshold scaling and decay;
- cascade budget: a synthetic cascade terminates by budget before `max_cascade_depth`
  when `HOP_COST` dictates, and by depth otherwise;
- per-item cooldown;
- `observe` emits nothing and enforces nothing: the inbox receives zero
  `Stimulus::Tension`, `pulses_would_fire` increments, and a synthetic edge cascade
  runs identically with the engine in `observe` and with it `off`;
- `brakes` enforces: the same synthetic cascade is cut by budget / blocked by fatigue;
- stale predictions: beyond `PREDICTION_MAX_ACTIVE` the oldest do not pulse; strength
  halves after one half-life;
- abandoned set round-trips through `SelfModel` serialisation, is pruned when a
  question leaves `open_questions`, and never exceeds `ABANDONED_HASHES_MAX`;
  FNV-1a test vectors.

Integration (existing `tests/digester.rs` harness, real test graph):
- with a seeded prediction and an open question, one `walk_parallel` in `observe`
  logs ≥ 1 would-fire event, `/diverger`-equivalent stats show it, inbox untouched;
- `off` mode: `DivergerStats` identical in shape and the walk output unchanged.

## 11. Files

- new `src/tension.rs`: `TensionEngine`, item states, pure strength/decay/pool/fatigue
  functions, stats struct, ring buffer;
- `src/diverger.rs`: `Stimulus` enum, fatigue on `DivergerNode`, cascade budget and
  depth, global pool, origin tagging, stats;
- `src/core.rs`: two counters; increments in `predict()` and on resolution;
- `src/walker.rs`: one call to `tension::release_after_walk` at the end of
  `walk_parallel`;
- `src/diverger.rs` (also): the same call after each spontaneous walk the reactor
  fires, so internally triggered activity can release tension too;
- `src/db.rs`: `top_nodes_in_domain`;
- `src/api.rs`: env parsing, engine construction into `AppState`/diverger, stats
  wiring into `/diverger`;
- `README.md`: a short section under Architecture.

## 12. Defaults

One table of constants in `tension.rs`. They are not environment variables; the
only env surface is `RGW_TENSION_MODE`. Tuning happens in code with a test, so a
parameter change is reviewable.

| Name | Default |
|---|---|
| PREDICTION_SATURATION_SECS | 3600 |
| PREDICTION_BASE_ENERGY | 0.15 |
| QUESTION_BASE_ENERGY | 0.20 |
| QUESTION_DECAY | 0.85 |
| QUESTION_MAX_ATTEMPTS | 12 |
| MIN_PULSE_ENERGY | 0.02 |
| K_PREDICTION_TARGETS / K_QUESTION_TARGETS | 3 / 3 |
| ITEM_COOLDOWN_SECS | 120 |
| PREDICTION_MAX_ACTIVE | 10 |
| PREDICTION_HALF_LIFE_SECS | 86400 |
| LIVE_JUDGEMENT_HOURS | 72 |
| ABANDONED_HASHES_MAX | 4096 |
| FATIGUE_PER_FIRE / FATIGUE_DECAY_PER_SEC | 0.5 / 0.01 |
| CASCADE_BUDGET / HOP_COST | 1.0 / 0.2 |
| POOL capacity / refill_per_sec | 20.0 / 0.05 |
| EDGE_QUEUE_CAPACITY (existing) | 4096 |

## 13. Risks and how the design answers them

- **Feedback loop** (tension → walk → release → walk): bounded by the global pool
  (hard CPU cap), per-item cooldown, and fatigue. Visible via `empty_fraction_1h`.
- **Noise instead of purpose**: sources are only unresolved predictions and open
  questions; no random leaks. Target mapping is by domain importance and embedding
  similarity, the same selection signal the A/B showed to carry value.
- **Gaming the metric**: resolution rate is tracked against creation rate (§9.1).
- **Questions that never resolve**: decay + attempt cap + abandoned state (§3.2).
- **Rollback**: `off` restores today's behaviour exactly, including no brakes; and
  `observe`, the default, is already behaviour-neutral, so a deploy with the default
  setting changes nothing about how external input is handled.
- **Stale predictions starving questions**: half-life decay plus a cap on active
  pulsing predictions (§3.1).
- **Observe looking rosier than live**: would-be pulses are charged with an
  estimated cascade cost (§6.3), and §9.2 is explicitly a leading indicator.
