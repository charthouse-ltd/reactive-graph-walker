# Learning walkers: query seeding, outcome-gated learning, replay, a scratch dream layer, gated communication

**Status:** DRAFT rev 4 for review. Not implemented. Code references are to `main` at `e556039`.
Rev 2 folds in review round 1: the system's own judgment as the main learning signal
(§6.3, §6.4) and gentle learning from spontaneous walks (§7.2, §8.1.1). Rev 3 folds in
review round 2: dream signals, score weights, one-tap rating. Rev 4 folds in review
round 3: the rating push is part of plan P2 (§6.6), and the two fitness measures are
logged side by side with one decision due at G3. §19 lists what changed.
**Naming.** "Stage" here means a stage of this document. The self-selection protocol has
its own Stages 0 to 2, and its own "replay" (rule-trial replay), which are unrelated to
Stage 2 and to the replay of §8.1 here.
**Scope:** a programme of eight stages delivered as five implementation plans (§20). This
document is the design for all of them; each plan is written when its predecessor's gate
(§14) has been met.
**Not in scope:** the reactor and the tension engine. Tension is being recalibrated by
`2026-11-04-tension-amendment-1-calibration.md`; this spec adds one constraint to it
(§8.5) and changes nothing else there.

**Basis.** Two research reports (brain mechanisms for coupling parallel processes and
learning from outcomes; sleep, dreaming and speculative layers). They are kept outside
this repository. §3 lists the findings each design decision rests on, with their
evidence status. Where a constant has no source, §17 says so.

---

## 1. Problem

Verified in the code:

1. **The walk does not depend on the query.** `WalkRequest.stimulus` is accepted and
   never read. `walk_parallel` seeds every walk from `db::seed_nodes`: importance ≥ 4,
   ordered by access count. The chat path does the same.
2. **Nothing is learned from results.** Query walks strengthen no edge. Spontaneous
   walks add a flat `0.02` to every traversed edge whatever the outcome
   (`diverger.rs`). Nothing weakens an edge except the manual `/prune` endpoint. No
   route lets a consumer say whether a walk's context helped.
3. **The dream loop neither consolidates nor generates.** Every 3–30 s it runs 20
   noisy walks from the same fixed seeds and writes a `dream_insight` edge between two
   nodes that were already neighbours on the walk, directly into the real graph. Its
   "independent walks confirmed" threshold is met by one walk. It replays nothing that
   happened while awake. `episodic.rs`, which describes replay, has no callers.
4. **Walkers do not communicate in a usable way.** `WalkerCollective` is read and
   written under `try_lock`, so what a walker sees is a race. Three of its four
   signals are repulsive, it carries node ids and domain names only, `drift_valence`
   and `drift_arousal` are written and never read, each walker runs on a private
   `SelfModel` clone, and the table is dropped after the walk.
5. **Agreement is structurally near zero**, because seeds are dispersed and the shared
   signals push walkers apart.

## 2. Goals and non-goals

**Goals, in the order the evidence supports them**

1. The context returned for a query is a function of that query.
2. Every walk is judged: by the system itself on every walk, and by people or consumers
   when they say something. Those judgments are the only thing that changes weights.
3. Offline work replays real, credited walks and consolidates what they taught. A thought
   that keeps recurring in spontaneous walks may also enter slow weights, slowly and
   under a cap.
4. Free generation happens in an isolated layer whose content expires by default and
   reaches the real graph only through a gate.
5. Walkers influence each other deterministically, and only real walkers influence
   real state.

**Non-goals**

- No change to reactor thresholds, fatigue, tension pulses or their constants.
- No speculative nodes with generated text. The scratch layer holds links only.
- No LLM rewriting of stored memories during consolidation.
- No physical second database. "Separate graphs" are run-tagged rows in one table.
- No free two-way messaging between dream walkers and real walkers.
- No change to what backends receive today unless a mode is switched on. Every stage
  ships behaviour-neutral by default (§11).

## 3. Evidence basis

Status: **E** established, **D** debated, **P** preprint or single study, **N** no
direct evidence (our inference). A dagger marks a figure the researchers could not
re-open and quoted from memory or from an extraction tool; do not treat it as checked.

| # | Finding | Status | Used in |
|---|---|---|---|
| 1 | With graph and propagation fixed, the seeding method alone moved average Recall@5 between 59.6, 74.6 and 87.1 (HippoRAG 2 ablation, [Gutiérrez et al. 2025](https://arxiv.org/html/2502.14802)) | E | Stage 1 |
| 2 | Unconstrained linear spreading becomes independent of the query; decay, division by fan-out and a step limit are the fixes, in that order of effect ([Jiang et al. 2026](https://arxiv.org/abs/2601.02744)) | P | Stages 1, 5 |
| 3 | Graph retrieval helps multi-hop and temporal questions and is neutral or harmful on simple lookups and unanswerable queries ([Xiang et al. 2025](https://arxiv.org/html/2506.05690)) | E | Stage 1 fallback |
| 4 | A learning rule needs outcome minus a per-context expectation; without the expectation it degenerates into drift ([Frémaux & Gerstner 2016](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)) | E | Stages 2, 3 |
| 5 | Updating stored items with task outcomes raised average success from 0.679 to 0.772 over similarity retrieval ([Zhang et al. 2026](https://arxiv.org/html/2601.03192v2)) | P | Stage 3 |
| 6 | Every working ant-colony variant pairs reinforcement with evaporation, bounds and quality-weighted deposit; flat unbounded reinforcement stagnates (Dorigo & Stützle 2004†) | E | Stage 3 |
| 7 | Weak changes stay eligible for about an hour and become lasting only if a significant event follows ([Moncada & Viola 2007](https://doi.org/10.1523/JNEUROSCI.1083-07.2007), rat) | E | Stage 4a |
| 8 | Replay priority = gain × need ([Mattar & Daw 2018](https://www.princeton.edu/~ndaw/md18.pdf)) | E (model) | Stage 4a |
| 9 | The memory benefit of sleep tracks replay of real experience (E). In the one model that varies the schedule, generation helped only when alternated with replay ([Singh, Norman & Schapiro 2022](https://doi.org/10.1073/pnas.2123432119)) | E; schedule is a model result | Stage 4 schedule |
| 10 | Dreaming sleep removes most newly formed connections and leaves old ones alone ([Li et al. 2017](https://doi.org/10.1038/nn.4479), mouse) | E | Stage 4b pruning scope |
| 11 | No dreaming model stores the dream; generated content adjusts weights and is discarded | E | Stage 4b lifetime |
| 12 | A system that promoted on confidence and never demoted fell from 90 % to 57 % precision over 66 rounds (NELL) | E | Stage 4b demotion |
| 13 | Raw structural link proposals are right about a fifth to a third of the time at rank one | E | Stage 4b gate |
| 14 | A walker's time in a layer is proportional to that layer's total weight, and the switch from "ignored" to "dominant" is abrupt ([De Domenico et al. 2014](https://doi.org/10.1073/pnas.1318469111))† | E | Stage 4b cap |
| 15 | Generated content helped retrieval as a route to real documents, not as an answer (HyDE)† | E | Stage 4b hints |
| 16 | Stable designs let the trusted side be read freely and let the speculative side return only through a check | N (convergent designs, no direct comparison) | §8.4 |
| 17 | Sharing findings among similar agents moves them toward the majority, not toward correctness ([Choi et al. 2025](https://arxiv.org/abs/2508.17536)) | E | Stages 5–7 last |
| 18 | Workspace-style communication gave gains of tenths of a point to a few points in controlled tests | E | Stages 5–7 last |
| 19 | Brain rhythm as a communication clock is contested; the case for synchronised rounds is that a competition needs a defined set of participants | D / N | Stage 5 |
| 20 | No source defines reward for a retrieved path, and none says how alternatives are compared | N | §6, open decisions |
| 21 | Internal signals (re-use, convergence, surprise) are unsafe as a reward unless checked against an external one; one system validated its learned utility by showing that it predicted later success ([Zhang et al. 2026](https://arxiv.org/html/2601.03192v2)) | P / N | §6.4 calibration |

## 4. Architecture

```
            query text (stimulus)
                    |
            [seeding]  query-derived seeds, floor -> vector fallback
                    |
   real graph ----> [walk] N walkers, rounds, one hint hop at most ----> [merge] ranked nodes
   (memory_*)        ^   reads scratch as hints (mode-gated)               |
                     |                                                     v
              [scratch layer]                                     context + walk_id
              rgw_scratch_edges                                            |
                     ^                                                     v
                     | candidates                                   [walk log]
   [dream block] ----+                                                     |
        ^  reads real state freely                           outcome  <----+---- self-score, plus human / consumer anchors
        |                                                        |
   [replay block] <--- tagged, credited walks ------------- [learning] fast weights, node utility
        |
        +--> slow weights (existing memory_edges.weight)
```

**Stores**

| Store | Owner | Content |
|---|---|---|
| `memory_vectors`, `memory_edges` | backend | the real graph; `weight` is the slow weight |
| `rgw_walk_log`, `rgw_walk_outcome` | RGW | what each walk did and how it turned out |
| `rgw_edge_fast`, `rgw_node_utility`, `rgw_baseline` | RGW | what outcomes have taught, not yet consolidated |
| `rgw_scratch_edges` | RGW | candidate links, run-tagged, expiring |

**Invariants** (each has a test, §15)

- I1. Only four things write `memory_edges`: ingestion (`/edge`), replay capture (§8.1),
  self-capture (§8.1.1) and promotion or demotion (§8.3). Walks never do.
- I2. The dream block writes `rgw_scratch_edges` only.
- I3. Scratch rows never enter importance, access counts, traversal counts, fan
  division, need terms, reactor energy, tension pulses, the walk field of real walkers,
  workspace slots or the self-model.
- I4. A context string never asserts a link that exists only in the scratch layer.
- I5. Every persistent weight change is traceable. Each fast-weight update is recorded
  with its `walk_id` in `rgw_walk_credit.updates`; slow weights change only by capture
  of fast or self weight (§8.1) or by a promotion or demotion with its premises.
- I6. With `RGW_DETERMINISTIC=1` and a fixed seed, a walk's output is identical across
  runs.
- I7. No single spontaneous walk moves a slow weight. Spontaneous marks reach slow
  weights only through self-capture, which needs traversals on `SELF_MIN_DAYS` distinct
  days and moves any slow weight by at most `SELF_CAPTURE_CAP` per cycle.

## 5. Stages 0 and 1: measure, then seed from the query

### 5.1 Stage 0: measurement

**Deterministic mode.** `RGW_DETERMINISTIC=1` plus an optional `seed: u64` on the walk
request. Each walker's RNG is seeded from `(seed, walker_index)`. Until Stage 5 exists,
deterministic mode runs walkers sequentially in index order, so the shared table is
order-deterministic.

**Eval arms.** With `RGW_EVAL=1`, the walk request accepts `arm`:

| Arm | Behaviour |
|---|---|
| `fixed` | today's seeds |
| `query` | Stage 1 seeds and merge |
| `vector` | no walk; top-k nodes by similarity to the query |
| `ppr` | Stage 1 seeds, one deterministic Personalized PageRank push in place of the walkers, same merge |
| `drop:<bias>` | `query` without one walker profile |

**Harness.** `examples/recall_eval.rs` reads a query set
`[{query, gold_node_ids, category}]` with categories `lookup`, `multi_hop`, `temporal`,
`unanswerable`, calls `/walk` once per arm, and reports per category: recall of gold
nodes within the first 10 ranked nodes, rank of the first gold node, emitted tokens,
latency. For `unanswerable`, the metric is the share of queries where the fallback
fired. The existing `examples/ab_eval.rs` stays as the answer-level check and must run
both judge orderings.

**Gold sets.** A gold set must not be built from what the current system retrieves.
Deployment gold sets contain operator material and are never committed to this
repository. The repository carries one small fixture set over the philosophy test
graph in `tests/common`.

**Telemetry** added to the walk output under `telemetry`: mean pairwise path overlap
between walkers, share of visits on the 1 % highest-degree nodes, branching factor
(outgoing edges whose effective weight is at least 0.1 of the local maximum), duplicate
claims in the shared table.

### 5.2 Stage 1: query-derived seeds

`walk_parallel` gains `cue: Option<&str>`. `/walk` passes `stimulus`; the chat path
passes the last user message. Spontaneous walks have no cue and keep starting at the
node that fired.

```
embed:   q = embed(cue) on the blocking pool
seeds:   C = top SEED_TOP_K nodes by cos(q, emb)                         # 5
         r(i) = cos(q, emb_i) * spec(i),  spec(i) = 1 / (1 + ln(1 + deg(i)))
         r(j) += SEED_WIDE_WEIGHT * cos(q, emb_j) for the next SEED_WIDE_M nodes   # 0.05, 20
         normalise r to sum 1
floor:   if max cos < SEED_FLOOR: do not walk; return the vector arm, walked = false
start:   walker k starts at a seed drawn from r (deterministic mode: by rank, round-robin)
step:    with probability RESTART_PROB restart at a seed drawn from r             # 0.5
         divide each candidate edge's score by the count of same-type edges at the node
merge:   m(v) = sum over walkers and visits of 0.5^depth(v), depth counted from the
         last (re)start; ranked_nodes = nodes by m(v) descending
context: best intact segments as ordered chains, then remaining nodes by m(v)
```

`spec` is this system's substitute for HippoRAG's per-entity specificity. The linear
form `1/(1+deg)` is an eval arm, not the default, because it removes hubs outright.

`WalkerResult` gains `depths: Vec<u8>` and `segment_starts: Vec<usize>` because a
restart breaks adjacency. `WalkOutput` gains `walk_id`, `walked`, `seed_mode` and
`ranked_nodes: Vec<(i32, f32)>`. Existing fields are kept and still filled, so backends
that read only `walker_context` and `primary_domain` are unaffected.

**Modes** (`RGW_SEED_MODE`): `fixed` is today. `shadow` (default) computes query seeds,
logs their overlap with the fixed seeds and the best cosine, and walks with fixed
seeds. `query` uses them.

**Expected change.** Agreement stops being structurally zero, because walkers now share
a cue. The existing repulsion gets its proper job: spreading walkers out from a common
start.

**Main risk.** Harm on simple lookups and unanswerable queries. `SEED_FLOOR` is the
guard and is tuned on the gold set before `query` is enabled.

## 6. Stage 2: walk log and outcome channel

### 6.1 Walk log

Every walk gets a 128-bit random `walk_id`, returned in `WalkOutput` and in the chat
response as `rgw_walk_id`. One row per walk in `rgw_walk_log` (§10): kind, cue hash,
cue embedding, context key, seeds, per-walker ordered edges with depth and layer,
emitted nodes, modes in force, whether a hint was used. The cue text is not stored,
except for walks drawn into the rating sample (§6.4). Retention is `WALK_LOG_DAYS` or
`WALK_LOG_MAX_ROWS`, whichever is hit first. Rows of spontaneous walks are compact (no
embedding) and are kept for `SPONT_LOG_DAYS`.

`episodic.rs` is deleted. The walk log plus replay (§8.1) are its implementation.

The existing `metacog::ReplayEntry` buffer recorded at the end of `walk_parallel` is
**kept**. It is not memory replay: it feeds the self-selection protocol's rule trials
(`trial_pending_rules`), which re-run the critic's decision under a candidate rule
set, and `/selection/stats` reports its fill level. The walk log does not hold what it
captures. Its comment is renamed from "Stage 2" to "Self-selection Stage 2 (rule-trial
replay)".

### 6.2 Outcome sources

| Source | Role | How it arrives |
|---|---|---|
| `self` | the main signal, on every walk | computed by RGW after the walk (§6.3) |
| `human` | anchor | `POST /walk/{id}/outcome`; also posted automatically when a motor proposal carrying a `walk_id` is approved (1.0) or rejected (0.0) |
| `consumer` | anchor | `POST /walk/{id}/outcome` from a backend, for example whether a drafted reply was used or discarded |

```
POST /walk/{walk_id}/outcome      (X-RGW-Key)
{ "reward": 0.0..1.0, "source": "consumer" | "human",
  "used_node_ids": [int]?, "answer_text": string? }
```

Anchors are accepted until `OUTCOME_WINDOW_HOURS` after the walk. The self-score and
the anchors of a walk are stored side by side; none overwrites another.

**Attribution.** If `used_node_ids` is given, it is used. Otherwise, if an answer text
is available (posted, or produced on the chat path), a node counts as used when
`cos(embed(answer), emb_node) >= ATTRIB_SIM`. "Used" is not "useful"; attribution only
decides which walkers share the credit.

### 6.3 Self-score

After every walk, query or spontaneous, the system scores whether the walk led
anywhere. The score is algorithmic, runs beside the existing critic in `metacog.rs`,
and makes no LLM call.

```
coherence  mean cos(emb_a, emb_b) over consecutive nodes of each segment, averaged over
           walkers, rescaled from [COH_LOW, COH_HIGH] to [0, 1]
question   1 if an emitted node matches an open question with cos >= QUESTION_MATCH, else 0;
           a question counts at most once per day
pickup     1 if, within PICKUP_WINDOW_HOURS, a later walk that started from a different seed
           or fired node emits at least PICKUP_MIN_SHARED of this walk's emitted nodes, else 0
R_self     = W_COH * coherence + W_QUESTION * question + W_PICKUP * pickup     # weights sum to 1
```

`coherence` and `question` are known at once and are applied at once. `pickup` arrives
later and is applied as a second update of size `W_PICKUP` over the same traces.
Matching a question here does not remove it from `open_questions`; ending a question
belongs to the tension work.

Pickup carries half the score on purpose. A walk with low coherence that is picked up
later scores at least 0.5, so a surprising leap that leads somewhere is rewarded and
not penalised.

`pickup`, and the recurrence test of §8.1.1, are partly caused by the system's own
learning, because a strengthened edge is more likely to be walked again. Two guards
limit that. The later walk must start somewhere else. And the calibration of §6.4
measures whether the score still tracks what a person thinks.

### 6.4 Anchors and calibration

External outcomes are weighted anchors. They are not a precondition for learning.

```
anchor     R_anchor = the human rating if there is one, else the consumer value
combine    R = (ANCHOR_WEIGHT * R_anchor + R_self) / (ANCHOR_WEIGHT + 1)    with an anchor
           R = R_self                                                      without one
trust      a number in [0, 1]. A walk without an anchor learns at rate trust * ETA;
           a walk with an anchor learns at ETA.
rated set  walks that carry both a self-score and a human rating
calibrate  weekly, over the rated walks of the last CALIB_WINDOW_DAYS, if there are at
           least CALIB_MIN_N of them:
             rho   = Spearman(R_self, human rating)
             trust = clamp((rho - RHO_MIN) / (RHO_FULL - RHO_MIN), 0, 1)
stale      with fewer than CALIB_MIN_N rated walks in the window, trust is halved at each
           weekly check
start      trust = 0 until gate G2 is met
```

If the self-scores and the ratings agree, the system's own judgment is trusted for
learning. If they drift apart, or if nobody rates for a month, its weight falls
automatically.

**Rating.** A rating is one tap: good, bad or skip.

```
sample    each walk enters the rating pool with probability
          p = clamp(RATE_POOL_TARGET / walks in the last day, RATE_SAMPLE_PROB, 1),
          stratified so that low, middle and high self-scores are equally represented
kept      for pooled walks only, the cue text is kept until rated, skipped, or RATE_KEEP_DAYS
shown     the cue (for a spontaneous walk, its starting node) and up to RATE_NODES_SHOWN
          emitted nodes, as short excerpts
verdict   good -> human outcome 1.0;  bad -> human outcome 0.0;
          skip -> no outcome, and the walk is not offered again
```

Ratings are delivered where the operator already is, by the push adapter of §6.6, and
not only on request. Every rating route stays behind `X-RGW-Key`, which the adapter
holds; there is no token-gated public route.

The target is about 15 ratings a week. A light week does not drain trust: calibration
needs `CALIB_MIN_N` rated walks within `CALIB_WINDOW_DAYS`, which is about five a week.
With good and bad as the only values, `rho` is a rank correlation against a two-valued
rating; the thresholds apply unchanged.

### 6.5 Teaching signal

```
context key  k = (kind, primary_domain),  kind in {query, spontaneous}
baseline     V[k] += BASELINE_RATE * (R - V[k])
delta        = R - V[k]      (taken as 0 while context k has fewer than BASELINE_MIN_N outcomes)
per walker   delta_k = delta if walker k's path holds a used node, else 0
             with no attribution at all: delta_k = delta for every walker
```

A walk is **credited** when it has an outcome with `delta > 0`.

**Confirmation counters** for dream candidates (§8.3) are logged beside outcomes and
are never substituted for them. They do not call `SelfModel::predict()`: predictions
stay dormant under the tension amendment, and wiring them here would start prediction
pulses that spec has ruled out for v1.

### 6.6 Rating push adapter

Without a push nobody rates, trust stays at zero, and learning never goes live. The
adapter is therefore part of plan P2, not an open point.

The rating push is assumed to run on an operator messaging channel provided by the
consuming backend, as a new message type implemented there. Where a backend already
has such a channel, the push is added to it and is not a second, standalone bot that
would compete for the same channel. RGW's side is two routes; the adapter's side is
this contract.

```
daily     at an operator-set time: POST /learning/rate-sample/take {limit: RATE_PUSH_PER_DAY}
          RGW returns that many pooled walks and marks them offered; none pooled, no message
message   one per walk, to the configured operator chat only: the cue (for a spontaneous
          walk, its starting node), up to RATE_NODES_SHOWN excerpts of at most
          RATE_EXCERPT_CHARS characters, and three buttons: good, bad, skip
tap       accepted only from a configured operator; POST /walk/{walk_id}/rate {verdict};
          the message is edited to show the verdict
repeat    a later verdict replaces an earlier one while the outcome window is open;
          the same verdict twice is a no-op
expiry    an offered walk with no verdict after RATE_KEEP_DAYS becomes expired, its cue
          text is deleted, and a late tap gets an error that the adapter shows
failure   a failed push is logged and retried the next day; it never affects RGW
```

The push sends cue text and memory excerpts through the messaging provider. It is
switched on per backend, goes only to operator chats, and never to an audience
channel.

## 7. Stage 3: learning from judged walks

### 7.1 Query walks

Updates are buffered during the walk and applied after aggregation, in ascending
`(walker, step)` order: once when the self-score is known, and again as a correction
when a pickup or an anchor arrives and changes `R`.

```
effective weight  w(e) = clamp(memory_edges.weight + w_fast(e) + w_self(e), W_MIN, W_MAX)

node utility      Q[n] += Q_RATE * (R - Q[n])        for every node emitted in a query walk
ranking           rank(v) = 0.5 * z(m(v)) + 0.5 * z(Q[v])                # replaces rank by m(v) alone

rate              ETA with an anchor; trust * ETA without one (§6.4)
edge trace        tr(e) = 0.9^d,  d = hops from e to the nearest used node on that walker's
                  segment; with no attribution, d = hops to the segment's end
fast update       w_fast(e) += rate * delta_k * tr(e) * (W_MAX - w(e))     if delta_k > 0
                  w_fast(e) += rate * delta_k * tr(e) * (w(e) - W_MIN)     if delta_k < 0
                  tag_time(e) = now
decay             w_fast(e) *= 0.5^(dt / FAST_HALF_LIFE_HOURS)            applied lazily on read
```

With `ETA = 0.04` and full trust, a clearly good walk moves its last edge by about
today's `0.02`, a neutral walk by about zero, and a poor walk weakens its path. The
soft bounds mean a strong edge gains little and a weak edge loses little.

### 7.2 Spontaneous walks

Spontaneous walks keep shaping memory, gently. They are scored by the same self-score,
with their own baselines, and they write to a separate component so that retrieval
learning stays clean.

```
self update   w_self(e) += ETA_SELF * trust * delta_k * tr(e) * (W_MAX - w(e))    if delta_k > 0
              w_self(e) += ETA_SELF * trust * delta_k * tr(e) * (w(e) - W_MIN)    if delta_k < 0
              ETA_SELF = ETA / 4
decay         w_self(e) *= 0.5^(dt / SELF_HALF_LIFE_HOURS)
day mark      each spontaneous traversal of e marks (e, today) in rgw_edge_self_days
```

A thought leaves a mark that fades within a day or two unless it comes back. A
spontaneous walk never writes `w_fast`, never updates node utility, and never writes a
slow weight (I7). Dream walks write no weight of any kind; they write scratch rows
only (§8.2).

### 7.3 What stops, and modes

In `live`:
- The flat `cache.strengthen_edges(.., 0.02)` on spontaneous walks is replaced by the
  self update of §7.2.
- Nothing else changes yet. `LearnedBias::update_from_session`, the profile scorecards
  and the rule trials all keep their present intrinsic-fitness measure. Each walk's
  `R` and each profile's running mean of `delta_k` are logged beside the intrinsic
  values. Moving bias selection and rule trials to the outcome measure is one
  decision, taken for both together, and is due at G3 (§19). Until then the two cannot
  pull in different directions, because neither has moved.

**Modes** (`RGW_LEARN_MODE`): `off` computes nothing. `observe` (default) computes the
self-score, every update and the weekly calibration, logs them, and writes neither
`rgw_edge_fast` nor `rgw_node_utility`; today's flat strengthening continues. `live`
applies them.

**Main risk.** One scalar spread over every edge of a walk is a noisy teacher, and a
self-computed one can flatter itself. The per-walker tag narrows it; the baseline
centres it; anchors outweigh it; trust follows the calibration; and the fast stores
mean a wrong lesson decays unless it is confirmed.

## 8. Stage 4: consolidation and the scratch layer

One **cycle** is a replay block followed by a scratch block of at most equal budget.
Once replay exists, two scratch blocks never run in a row. A cycle starts when no query
walk has run for `CYCLE_IDLE_SECS` and at least `CYCLE_MIN_INTERVAL_SECS` have passed
since the last. Before replay exists the scratch block may run alone on the same clock,
in `shadow` only (§8.7).

### 8.1 Stage 4a: replay

Replay moves what outcomes taught from the fast store into the slow store. It never
invents a change.

```
candidates   edges with tag_time within TAG_WINDOW_HOURS and |w_fast| >= GAIN_FLOOR
gain(e)      |w_fast(e)|
need(e)      share of logged query walks in the last NEED_WINDOW_DAYS whose emitted set
             holds either endpoint of e
priority     (gain * need)^REPLAY_EXPONENT, sampled without replacement
budget       REPLAY_BUDGET edges per cycle
capture      x = CAPTURE_FRACTION * w_fast(e)
             memory_edges.weight = clamp(weight + x, W_MIN, W_MAX);  w_fast(e) -= x
stale        STALE_SHARE of the budget goes to high-need edges with the oldest capture time,
             which are re-read and left unchanged if they hold no fast weight
shortcut     a chain a -> .. -> c of 2 or 3 real edges that appears in the credited walks of
             SHORTCUT_MIN_EPISODES distinct cues becomes a scratch candidate a - c with
             generator = replay (it is not written to the real graph here)
balance      if the summed slow weight leaving a node changed this cycle and exceeds its
             value at cycle start by more than BALANCE_SLACK, scale that node's changed
             edges back to the limit
```

Replay reads `w_fast` only. It never samples spontaneous walks and never touches
`w_self`.

The capture rule is a deliberate simplification of the research proposal, which moved
slow weights toward a discounted reward target. Transferring a fraction of the fast
weight keeps slow changes bounded by what outcome-gated learning actually earned, and
needs no second learning rate.

`need` is measured from the walk log, not computed as PageRank over the backend's
graph, because a full PageRank per cycle is costly and the log already says where
queries go.

#### 8.1.1 Self-capture

Runs at the end of the replay block. It is the only path from spontaneous walks to
slow weights.

```
eligible   edges with |w_self| >= GAIN_FLOOR that spontaneous walks traversed on at least
           SELF_MIN_DAYS distinct days within the last SELF_WINDOW_DAYS
capture    x = clamp(SELF_CAPTURE_FRACTION * w_self(e), -SELF_CAPTURE_CAP, SELF_CAPTURE_CAP)
           memory_edges.weight = clamp(weight + x, W_MIN, W_MAX);  w_self(e) -= x
budget     at most SELF_CAPTURE_BUDGET edges per cycle, largest |w_self| first
```

This is how a recurring thought becomes part of the system without one stray walk
rewriting memory. The system's disposition drifts slowly, and on purpose.

### 8.2 Stage 4b: the scratch block (dreaming)

The dream loop keeps running. Its seeds, its output and its destination change.

```
seeds     each dream walk starts at a real node drawn from the waking marks:
            nodes of walks with the largest |delta|; nodes matched to open questions;
            nodes added in the last DREAM_RECENT_DAYS with degree <= 2
          paired with an older node from another domain, each weighted 0.5
walk      real edges only, at most DREAM_STEPS steps, today's noise and weak-edge boost
propose   a - c only if no real edge joins a and c and a real path of 2 or 3 hops does
          conf = product of effective weights on that path, in [0, 1]
          premises = ids of the real edges on that path
write     one row in rgw_scratch_edges: run_id, generator = dream, premises, conf,
          created_cycle, status = candidate. No validity filter runs at write time.
bound     sum of conf over candidates leaving u  <=  KAPPA * sum of real effective weight leaving u
          at most DREAM_MAX_PER_NODE new candidates per node and DREAM_MAX_PER_CYCLE per cycle
review    each scratch block starts by reviewing the candidates. One that was reused since
          the last review has its age reset to zero. The others age by one, but only if at
          least one query walk ran since the last review.
          reused = taken as a hint by a query walk, or proposed again by a later run from
                   different seeds, or both endpoints emitted together by a real walk
expire    age reaches SCRATCH_TTL_CYCLES -> status = expired; the row is kept and
          the pair is not proposed again without a new premise
retract   retracting a run_id sets every row of that run to retracted
```

"Separate graphs" are `run_id` values. Each run is a logical graph that can be read,
compared and retracted as a unit, without copying nodes.

Reuse decides only whether a candidate stays in the scratch layer. It never counts
toward promotion, which keeps its own independent test (§8.3). A candidate does not
age across a quiet period, because cycles run when the system is idle, which is
exactly when no query can reuse anything.

**Pruning** is off in this spec. The research says pruning recent provisional structure
is what dreaming sleep mostly does, but the only engineering template is from Hopfield
networks and gives no dose for graphs. It is listed in §19.

### 8.3 Return channels

Dream output reaches the real system in exactly two ways.

**Hint** (`RGW_DREAM_MODE >= hints`). A query walker standing on `u` may take at most
one scratch hop per walk:

```
score(u -> v) = KAPPA * conf * cos(q, emb_v)
```

It lands on the real node `v`, whose content may be emitted. The link `u - v` is not
stated in the context. The walk is marked `hinted`. No counter moves, no weight is
earned on the hop, and any stored artefact derived from a hinted walk carries the mark.

**Candidate promotion** (`RGW_DREAM_MODE = promote`).

```
confirm   +1 per distinct cue in which real walkers emit both a and c by paths that did
             not use the candidate
          +1 per credited walk (delta > 0) in which a hint over the candidate led to a used node
promote   confirmations from PROMOTE_MIN_CUES distinct cues
          AND check passes: no contradicts edge between a and c or on the premises;
                            every premise edge still exists; conf >= PROMOTE_MIN_CONF
          -> insert real edge (a, c, "derived", weight = W_MIN), w_fast = PROMOTE_FAST,
             status = promoted, at most PROMOTE_MAX_PER_CYCLE per cycle
demote    a promoted edge whose w_fast has decayed below GAIN_FLOOR with nothing captured,
          or whose premise was deleted, or that meets a contradicts edge
          -> delete the real edge, status = retracted
guard     stop promoting when the sampled precision of promoted edges falls between two
          consecutive audit rounds
```

A promoted edge enters with almost no slow weight. It survives only if outcome-gated
learning and replay then give it some. Deletion is the default.

Replay shortcuts (§8.1) take the same path. Apart from ingestion, **promotion is the
only way a new edge enters the real graph.**

### 8.4 Cross-layer flow

| Direction | Rule |
|---|---|
| real → dream | unrestricted read: nodes, edges, effective weights, embeddings, the walk log and its tags, open questions, a snapshot of the real walk field |
| dream → real | hints and candidates only (§8.3) |
| dream walkers among themselves | their own field and repulsion table (Stage 5), per run |

Scratch walkers never deposit in the real walk field, never add reactor energy or
tension pulses, never increment access or traversal counts, never propose to the
workspace (Stage 7), and never update the shared self-model or mood. Each of those is
a write channel into real behaviour even when no edge is added.

**Dream signals.** Today `dream.rs` sends a `dream_start` and a `dream_end` signal
through `core::process` on every cycle. Each one updates the thought embedding and
pulls the mind centroid toward the embedding of its own fixed text, every few seconds.
In `RGW_DREAM_MODE` `shadow` and above, neither signal is sent. There is no exception
for a waking-up moment: the cost is a steady drift of the self-model toward
boilerplate. In `legacy` both stay as they are.

**Selection statistics.** The dream loop also feeds its own output into the
self-selection history: `selection_observe` counts `connections_kept + edges_created`
in its `kept` term. In `shadow` and above that term counts promotions that passed the
gate in the cycle, and nothing else from the dream. The other selection calls on the
dream loop's clock (`select_biases`, `trial_pending_rules`, `check_rule_regression`)
read waking scorecards and stay; the dream loop is only their timer.

### 8.5 Constraint on the tension engine

Dream and replay activity add no energy to any reactor node and emit no tension pulse.
`spawn_after_walk` is not called for dream walks. This is the only requirement this
spec places on tension.

### 8.6 Existing `dream_insight` edges

An operator action, `POST /dream/quarantine`, dry-run unless `confirm: true`:
each `dream_insight` edge is copied to `rgw_scratch_edges` with `generator = legacy`
and deleted from `memory_edges`. If another real edge joins the same pair, the row is
stored as `expired`; otherwise as `candidate`. The action reports counts first and is
never run automatically, because it deletes rows from a backend-owned table.

### 8.7 Modes

`RGW_DREAM_MODE`: `legacy` (default) is today's loop, signals included. `shadow` runs §8.2 and writes
scratch rows that nothing reads; the legacy writes to `memory_edges` stop. `hints`
adds the hint hop. `promote` adds promotion and demotion. Replay (§8.1) runs whenever
`RGW_LEARN_MODE = live`; before that, the scratch block may run only in `shadow`.

## 9. Stages 5 to 7: communication

These come last because their measured effects are the smallest and because sharing
among similar walkers can cost diversity (§3, rows 17 and 18).

### 9.1 Stage 5: deterministic rounds and a per-walk field

The `Arc<Mutex<WalkerCollective>>` with `try_lock` is replaced by rounds.

```
round     every walker reads the same snapshot, chooses its next edge (pure function of the
          snapshot and its own RNG), then deposits are applied in walker-index order
field     deposit(i) = 1 / N for each walker standing on i
          u_i = 0.5 * a_i + 0.8 * sum_j w(j -> i) * a_j / fan_type(j) + deposit(i);  a_i = min(1, u_i)
          at most 3 propagation pulses per walk; contradicts edges carry no activation
attract   score *= 1 + FIELD_ATTRACT * a_v for a node v not yet visited by this walker
repel     score *= 1 / (1 + visits_by_other_walkers(v))
dead end  score *= DEAD_END_FACTOR (a dead end lowers a score and never forbids)
```

The field is per walk and per layer, and is dropped after the walk. **It is not the
reactor's node energy**, so it cannot shift the tension calibration. Visit counts are
kept per walker, because a pooled count cannot tell two walkers meeting from one
walker passing twice.

`RGW_FIELD_MODE`: `table` (default) is today. `observe` runs rounds, logs what the
field would have changed, and scores with the table. `rounds` uses the field.

### 9.2 Stage 6: shared live scalars (contract)

Walkers read one shared gain and one shared encode-or-retrieve scalar during the walk,
in place of private `SelfModel` copies averaged afterwards. The gain sets the
sharpness of edge choice from running means of `delta`; the second scalar mixes stored
weight against similarity to the cue and scales `ETA`. A large surprise lets walkers
re-seed. The biology behind the sign of the gain effect is contested, so each scalar's
effect is an explicit small function that is tuned, not derived. Rules and constants
are fixed in an amendment after Stage 5 results. Depends on Stage 2.

### 9.3 Stage 7: one-winner broadcast (contract)

Per round each real walker offers its current node. Offers are normalised against each
other, at most one wins, and it is broadcast only above a threshold; otherwise nothing
is broadcast. Broadcast items sit in about four decaying slots that all real walkers
read, with a signed gain per walker profile so that the contrarian is pushed away from
what the others are pulled toward. Only real walkers may propose; dream walkers may
read. The comparison that decides whether this ships is against independent walkers
merged centrally with agreement weighting. No study shows a threshold-and-ignite step
beating plain top-k selection, so top-k is built first. Rules and constants are fixed
in an amendment after Stage 5 results.

## 10. Storage

RGW creates no tables today; `memory_vectors` and `memory_edges` belong to the
backends. This spec does not alter them. All new state lives in `rgw_`-prefixed
tables that `db::ensure_rgw_schema()` creates idempotently at startup. If creation
fails (no privilege), the service starts and every mode that needs a table falls back
to its neutral value with a WARN.

```sql
rgw_walk_log      (walk_id bytea PK, created_at, kind, cue_hash bytea, cue_embedding real[],
                   context_key text, seeds int[], paths jsonb, emitted int[], modes jsonb, hinted bool,
                   rate_state text NULL, cue_text text NULL, intrinsic_fitness real)
                   -- rate_state: pooled, offered, good, bad, skipped, expired
rgw_walk_outcome  (walk_id -> rgw_walk_log, source text, reward real, used_node_ids int[],
                   received_at, PRIMARY KEY (walk_id, source))
rgw_walk_credit   (walk_id PK -> rgw_walk_log, r real, delta real, credited_walkers int[],
                   updates jsonb, applied_at)
rgw_calibration   (at PK, n int, rho real, trust real)
rgw_baseline      (context_key PK, v real, n int, updated_at)
rgw_node_utility  (node_id PK, q real, n int, updated_at)
rgw_edge_fast     (edge_id PK, w_fast real, tag_time, updated_at, captured_at,
                   w_self real, self_updated_at, self_captured_at)
rgw_edge_self_days (edge_id, day date, n int, PRIMARY KEY (edge_id, day))
rgw_scratch_edges (id PK, source_id int, target_id int, run_id bytea, generator text,
                   premises int[], conf real, created_cycle bigint, age int, last_reused_at, confirmations bytea[],
                   status text, promoted_edge_id int, updated_at,
                   UNIQUE (source_id, target_id, generator))
```

`paths` holds, per walker, the ordered `(edge_id, depth, layer)` triples. None of these
tables has a foreign key into backend tables; a deleted node or edge is detected on
read and the dependent row is dropped or retracted.

## 11. Configuration and staging

| Variable | Values | Default | Neutral value |
|---|---|---|---|
| `RGW_SEED_MODE` | `fixed`, `shadow`, `query` | `shadow` | `fixed` |
| `RGW_LEARN_MODE` | `off`, `observe`, `live` | `observe` | `off` |
| `RGW_DREAM_MODE` | `legacy`, `shadow`, `hints`, `promote` | `legacy` | `legacy` |
| `RGW_FIELD_MODE` | `table`, `observe`, `rounds` | `table` | `table` |
| `RGW_DETERMINISTIC` | `0`, `1` | `0` | |
| `RGW_EVAL` | `0`, `1` | `0` | |

Every default leaves backend-visible behaviour as it is today, apart from added output
fields and, in `SEED shadow`, the cost of one embedding and one similarity query per
walk. Each promotion is one config change and one rollback. Constants are in code
with tests, not in the environment, as in the tension spec.

Order of promotion: `SEED shadow → query`, then `LEARN observe → live`, then
`DREAM legacy → shadow → hints → promote`, then `FIELD table → observe → rounds`.
`DREAM shadow` may be enabled early, since nothing reads it.

## 12. API changes

- `POST /walk`: request reads `stimulus`; accepts `seed`, and `arm` under `RGW_EVAL=1`.
  Response adds `walk_id`, `walked`, `seed_mode`, `ranked_nodes`, `telemetry`.
- `POST /v1/chat/completions`: response adds `rgw_walk_id`.
- `POST /walk/{walk_id}/outcome`: new, protected (§6.2).
- `GET /learning`: new, protected (§13).
- `GET /learning/rate-sample?limit=N`: new, protected; read-only list of pooled walks, each
  with its cue and up to five node excerpts (§6.4).
- `POST /learning/rate-sample/take` with `{"limit": N}`: new, protected; returns pooled walks
  and marks them offered (§6.6).
- `POST /walk/{walk_id}/rate` with `{"verdict": "good" | "bad" | "skip"}`: new, protected.
- `POST /dream/quarantine`: new, protected, dry-run by default (§8.6).
- `GET /scratch?run_id=&status=`: new, protected; lists candidates for audit.
- `POST /scratch/retract`: new, protected; retracts a run.
- Motor proposals carry the `walk_id` of the walk that produced them.

All additions are additive JSON. Vendored backends keep working without change. Sending
outcomes is a change in the backends and is specified there, against §6.2.

## 13. Observability

`GET /learning`:

```json
{
  "seed":   { "mode": "shadow", "walks": 1840, "fallbacks": 212, "best_cos_p50": 0.61,
              "overlap_with_fixed_p50": 0.04 },
  "outcome":{ "walks_logged": 1840, "by_source": {"self": 1840, "human": 11, "consumer": 0},
              "rated": 9, "rho": null, "trust": 0.0, "contexts_ready": 2,
              "rating": {"offered": 31, "good": 6, "bad": 3, "skipped": 4} },
  "learn":  { "mode": "observe", "updates_would_apply": 5120, "fast_weight_sum": 0.0,
              "self_weight_sum": 0.0, "spont_coherence_mean": 0.41,
              "intrinsic_vs_r_spearman": null,
              "branching_factor_p50": 3.2, "hub_visit_share": 0.31 },
  "replay": { "cycles": 0, "captured": 0, "self_captured": 0, "shortcuts_proposed": 0 },
  "dream":  { "mode": "legacy", "runs": 0, "candidates": 0, "expired": 0, "promoted": 0, "demoted": 0,
              "hinted_walks": 0, "scratch_weight_share": 0.0, "real_edges_changed_by_dream": 0 },
  "field":  { "mode": "table" }
}
```

`real_edges_changed_by_dream` must be zero in `shadow` and `hints`, and equal to
promotions minus demotions in `promote`. `rho` is `null` until `CALIB_MIN_N` rated walks exist.

## 14. Pre-registered gates

Decided now, before data. A failed gate means the mode stays where it is.

| Gate | Promotion | Condition |
|---|---|---|
| G1 | `SEED shadow → query` | On the deployment gold set: `query` beats `fixed` on `multi_hop` recall by at least 10 points; `query` is within 2 points of `vector` on `lookup`; the fallback fires on at least 70 % of `unanswerable`; p95 added latency ≤ 50 ms |
| G1b | keep persona walkers | `query` is not worse than `ppr` by more than 2 points overall. If it is, the plan for Stage 5 is replaced by a decision on whether persona walkers stay |
| G2 | `LEARN observe → live` | At least 50 rated walks, and a Spearman correlation between self-score and human rating of at least 0.3 on them. `trust` is 0 until then and follows the weekly calibration afterwards |
| G3 | stay in `LEARN live` | On a time split, recall on later queries is not below frozen weights; branching factor has not fallen by more than 20 %; hub visit share has not risen by more than 10 points; self-capture accounts for at most a quarter of all slow-weight change in the window; mean coherence of spontaneous walks has not risen by more than 15 % over its mean in the observe period. The single-measure decision of §19 is recorded when G3 is first judged |
| G4 | `DREAM shadow → hints` | Offline precision of top-ranked candidates against held-out real edges ≥ 0.30. On the gold set, the hint arm beats a randomised scratch layer with the same edge count and degree sequence, and is not below the no-dream arm |
| G5 | `DREAM hints → promote` | A human-judged sample of at least 50 would-be promotions is ≥ 0.60 correct. Once promoting: sampled precision does not fall across 3 consecutive audit rounds |
| G6 | `FIELD observe → rounds` | Identical output across runs in deterministic mode; distinct gold nodes per walker-step not below the table; recall not lower |

If a randomised scratch layer beats the no-dream arm in G4, the graph is
under-connected for a five-step walk, and the remedy is walk length, restart and
seeding, not dreaming.

## 15. Testing

Unit (pure functions, no database):
- seed weighting, floor, round-robin start in deterministic mode
- merge mass with restarts; depth resets at a restart
- self-score: each component, the once-per-day question rule, the different-start rule for pickup
- anchor combination, learning rate with and without an anchor, calibration, stale halving
- baseline update and the withheld-until-`BASELINE_MIN_N` rule
- per-walker credit with and without attribution
- fast update: sign, soft bounds, trace by depth, lazy decay
- a spontaneous walk writes `w_self` and a day mark and nothing else
- replay priority, capture conservation (`Δweight + Δw_fast = 0`), balance limit; replay ignores `w_self` and spontaneous rows
- self-capture: needs `SELF_MIN_DAYS` distinct days, respects the cap and the budget (I7)
- scratch review: reuse resets age; age advances only if a query walk ran
- candidate rule: rejects adjacent pairs, requires a 2- or 3-hop real path
- scratch cap per node, expiry, run retraction, promotion check, demotion
- field round: deposits, fan division, soft dead end, determinism

Integration (DB-backed, `tests/schema.sql` plus `ensure_rgw_schema`):
- I1: a query walk and a spontaneous walk in `LEARN live` leave `memory_edges` unchanged
- I2/I3: a full dream block in `shadow` changes no row and no counter outside `rgw_scratch_edges`,
  and leaves the serialised self-model, mood included, byte-identical
- in `shadow`, the selection history's `kept` term ignores dream candidates
- I4: a hinted walk's context string names no scratch-only link
- I5: every edge whose `memory_edges.weight` changed in a cycle appears in the `updates` of at least one credit row, in a self-capture, or in a promotion or demotion
- I6: two runs with the same seed return byte-identical `ranked_nodes`
- outcome route: window, self-score and anchors stored side by side, unknown `walk_id`
- rating: `take` marks walks offered and never returns the same walk twice; a later verdict
  replaces an earlier one inside the outcome window; an expired walk rejects a verdict
- rating: good and bad store a human outcome, skip stores none and is not offered again;
  cue text is kept only for pooled walks and removed after a verdict or `RATE_KEEP_DAYS`
- quarantine: dry run reports and changes nothing; confirmed run moves rows
- `ensure_rgw_schema` failure degrades modes to neutral and the service starts

The CI gate stays `cargo test --locked --lib --bins`; DB-backed tests run where a
database is available, as today.

## 16. Files

| File | Change |
|---|---|
| `src/seeding.rs` | new: cue embedding, seed weights, floor, vector arm |
| `src/walklog.rs` | new: walk ids, log rows, retention |
| `src/outcome.rs` | new: outcome route logic, attribution, anchors, calibration, baseline, credit, rating sample |
| `src/metacog.rs` | self-score computed beside the algorithmic critic |
| `src/learning.rs` | new: node utility, fast weights, modes |
| `src/replay.rs` | new: priority, capture, shortcut proposals, balance |
| `src/scratch.rs` | new: candidate rules, caps, expiry, confirmation, promotion, demotion |
| `src/field.rs` | new (Stage 5): rounds and the per-walk field |
| `src/walker.rs` | cue parameter, restart, fan division, merge mass, hint hop, deterministic mode; later rounds; the `ReplayEntry` comment renamed |
| `src/graph.rs` | `WalkOutput` and `WalkerResult` fields; `WalkerCollective` removed at Stage 5 |
| `src/dream.rs` | seeds, candidate proposal, scratch writes, cycle scheduling; `dream_start` and `dream_end` signals not sent and `kept` redefined outside `legacy`; legacy path kept behind `legacy` |
| `src/diverger.rs` | flat strengthening removed in `LEARN live` |
| `src/edge_cache.rs` | effective weight read; scratch hints loaded when the mode allows |
| `src/db.rs` | `ensure_rgw_schema`, queries for the new tables, degree lookup |
| `src/api.rs`, `src/openai.rs`, `src/motor.rs` | routes and fields of §12 |
| `src/episodic.rs` | deleted |
| `examples/recall_eval.rs`, `examples/link_precision.rs` | new harnesses |
| `tests/` | fixture gold set; integration tests of §15 |

## 17. Defaults

Source: **R** read from the cited source; **U** quoted by the researchers from memory or
through an extraction tool, check before relying on it; **N** no source, a starting
point for tuning.

| Name | Default | Source |
|---|---|---|
| SEED_TOP_K | 5 | R (HippoRAG 2) |
| SEED_WIDE_M / SEED_WIDE_WEIGHT | 20 / 0.05 | N / R (HippoRAG 2 passage weight) |
| RESTART_PROB | 0.5 | R (HippoRAG 2 damping) |
| SEED_FLOOR | 0.35 | N |
| merge depth discount | 0.5 | N |
| WALK_LOG_DAYS / WALK_LOG_MAX_ROWS | 30 / 50 000 | N |
| OUTCOME_WINDOW_HOURS | 24 | N |
| ATTRIB_SIM | 0.6 | N |
| COH_LOW / COH_HIGH | 0.2 / 0.7 | N |
| QUESTION_MATCH | 0.6 | N |
| PICKUP_WINDOW_HOURS / PICKUP_MIN_SHARED | 24 / 2 | review round 1 / N |
| W_COH / W_QUESTION / W_PICKUP | 0.3 / 0.2 / 0.5 | review round 2 |
| ANCHOR_WEIGHT | 8 | review round 1 (range 5–10) |
| CALIB_WINDOW_DAYS / CALIB_MIN_N | 30 / 20 | N / review round 1 |
| RHO_MIN / RHO_FULL | 0.1 / 0.5 | N |
| RATE_SAMPLE_PROB / RATE_KEEP_DAYS | 0.05 / 14 | N |
| RATE_POOL_TARGET | 12 | N |
| RATE_NODES_SHOWN / RATE_PUSH_PER_DAY | 5 / 4 | review round 2 |
| RATE_EXCERPT_CHARS | 200 | N |
| SPONT_LOG_DAYS | 2 | N |
| BASELINE_RATE / BASELINE_MIN_N | 0.1 / 10 | R (range 0.05–0.1) / N |
| Q_RATE | 0.1 | N |
| ETA | 0.04 | N (chosen to match today's 0.02 step) |
| trace discount | 0.9 | R (Mattar & Daw) |
| W_MIN / W_MAX | 0.01 / 1.0 | existing prune floor and cap |
| FAST_HALF_LIFE_HOURS | 24 | N |
| ETA_SELF | ETA / 4 | review round 1 |
| SELF_HALF_LIFE_HOURS | 36 | review round 1 (a day or two) |
| SELF_MIN_DAYS / SELF_WINDOW_DAYS | 3 / 7 | N |
| SELF_CAPTURE_FRACTION / SELF_CAPTURE_CAP / SELF_CAPTURE_BUDGET | 0.25 / 0.01 / 10 | N |
| TAG_WINDOW_HOURS | 24 | N (biology: about 1 h, does not transfer) |
| GAIN_FLOOR | 0.001 | N |
| NEED_WINDOW_DAYS | 7 | N |
| REPLAY_EXPONENT | 0.6 | U |
| REPLAY_BUDGET | 20 | R (Mattar & Daw) |
| CAPTURE_FRACTION | 0.5 | N |
| STALE_SHARE | 0.15 | N |
| SHORTCUT_MIN_EPISODES | 3 | N |
| BALANCE_SLACK | 0.10 | N |
| CYCLE_IDLE_SECS / CYCLE_MIN_INTERVAL_SECS | 60 / 600 | N |
| DREAM_STEPS | 6 | existing |
| DREAM_RECENT_DAYS | 7 | N |
| KAPPA | 0.1 (sweep 0.01–0.5) | N |
| DREAM_MAX_PER_NODE / DREAM_MAX_PER_CYCLE | 3 / 50 | N |
| SCRATCH_TTL_CYCLES | 5 | N |
| PROMOTE_MIN_CUES / PROMOTE_MIN_CONF | 3 / 0.2 | N |
| PROMOTE_FAST / PROMOTE_MAX_PER_CYCLE | 0.05 / 5 | N / existing cap |
| field retention / spread / pulses | 0.5 / 0.8 / 3 | R (preprint) |
| FIELD_ATTRACT / DEAD_END_FACTOR | 0.5 / 0.25 | N |

## 18. Risks

- **A wrong outcome signal is learned faithfully.** Guards: anchors outweigh
  self-scores, trust starts at zero and follows the weekly calibration, the baseline
  per context, the fast stores that decay.
- **Self-judgment reinforcing itself.** A strengthened edge is walked again more
  often, which raises `pickup` and recurrence. Guards: the different-start rule, the
  quarter rate and short half-life of `w_self`, the capture cap (I7), the G3 limit on
  self-capture's share, and the calibration, which fails if the score stops tracking
  what a person thinks. If nobody rates, trust decays to zero and learning stops.
  Pickup now carries half the score and is the component most exposed to this loop,
  which is why G3 also watches whether spontaneous walks are becoming more coherent,
  that is, more predictable.
- **Harm on simple questions.** Guard: `SEED_FLOOR`, the `vector` arm, G1 per category.
- **Promoting noise.** Guards: default expiry, two-signal promotion, near-zero entry
  weight, demotion, the per-round precision guard, G5.
- **Attention capture by the scratch layer.** Guards: `KAPPA`, one hint hop per walk,
  the `scratch_weight_share` counter.
- **Circular confirmation.** A candidate's own traffic never counts toward its
  promotion; confirmations count once per distinct cue.
- **Laundering through the LLM.** A hinted walk is marked, and the mark follows any
  stored artefact derived from it.
- **Memory content leaves through the rating push.** Cue text and excerpts pass through
  the messaging provider. Guards: opt-in per backend, operator chats only, excerpts
  capped at `RATE_EXCERPT_CHARS`, at most `RATE_PUSH_PER_DAY` walks a day.
- **Cost.** One embedding and one vector query per walk; one log row per walk; a
  predicate on traversal when hints are on. Measured in G1's latency condition.
- **The layer does nothing.** G4's randomised arm exists to detect exactly that.
- **Persona walkers may not earn their place.** G1b decides before Stage 5 is planned.
- **Schema privilege.** If RGW may not create tables, nothing here runs; §10 degrades
  to neutral.

## 19. Assumptions made and open decisions

**Assumptions made while writing this spec.** Each can be reversed; say which.

1. The system's own score is the main learning signal. Human ratings, motor approve or
   reject, and backend reports are anchors weighted `ANCHOR_WEIGHT` times a self-score
   (review round 1).
2. RGW may create its own `rgw_` tables in the backends' databases.
3. Learning uses both node utility and edge weights.
4. The context key for the baseline is the walk's primary domain.
5. Persona walkers stay until G1b says otherwise.
6. Spontaneous walks write a separate, quarter-rate, short-lived weight, and reach slow
   weights only through capped self-capture. Dream walks write scratch rows only
   (review round 1).
7. Scratch content is links only, in one RGW-owned table with run tags.
8. Consolidation transfers fast weight to slow weight; it does not learn a separate target.
9. Need is measured from the walk log, not from PageRank.
10. Clocks: wall time for decay and windows; idle-triggered cycles.
11. Stages 6 and 7 are contracts only and get an amendment after Stage 5.
12. The rating push runs on an operator messaging channel assumed to be provided by the
    consuming backend, as a new message type implemented there (review round 3).

**Changed in rev 2 (review round 1).** Assumptions 1 and 6 above; gate G2; the
self-score, anchors and calibration of §6.3 and §6.4; `w_self`, self-capture and
invariant I7; reuse-based review of scratch candidates. Three points go beyond the
review text and are this spec's additions: spontaneous marks are stored apart from
query-walk marks so that replay can ignore them; a later walk counts as a pickup only
if it started somewhere else; and a scratch candidate ages only across cycles between
which a query ran, since a literal "next dream" rule would expire nearly every
candidate unseen. The earlier answer-similarity proxy is removed; the self-score
replaces it.

**Changed in rev 3 (review round 2).** Accepted: neither dream signal is sent outside
`legacy` (§8.4), with no waking-up exception, because both signals also drift the mind
centroid; score weights 0.3 / 0.2 / 0.5 and the coherence clause in G3; one-tap
rating with skip, pushed to the operator, target about 15 a week. This spec's
additions: `dream_start` is removed along with `dream_end`; the dream's `kept` term in
the selection history counts only gated promotions; rating routes stay behind the API
key. **Not accepted:** removing `metacog::ReplayEntry`. It serves the self-selection
protocol's rule trials, not memory replay, and nothing in this spec replaces it
(§6.1). Its comment is renamed as asked.

**Changed in rev 4 (review round 3).** The rating push is specified (§6.6) and is in
P2's scope, with its contract here and its implementation in each backend. Profile
scorecards and bias nudging no longer switch measure when learning goes live (§7.3);
the intrinsic and the outcome measure are logged side by side, and one decision for
bias selection and rule trials together is due at G3.

**Open decisions.**

| Decision | Default in this spec | Why open |
|---|---|---|
| Which outcome the backends report, and when | none assumed beyond the route | It is a product decision per deployment. Anchors are no longer required, but more of them make learning faster and safer |
| Which measure bias selection and rule trials use: intrinsic fitness or the walk outcome | both stay on intrinsic fitness; both measures are logged side by side | They must end up on the same measure or they pull in different directions. **Due at G3**, decided on the logged data once G2 has been met |
| The meaning of `kept` in the self-selection protocol | gated promotions only, outside `legacy` | It changes a statistic that protocol defines; its document needs the same edit |
| Where deployment gold sets come from | operator-written, outside the repo | Cost; a set built from current retrievals inherits the seed bias |
| What to do with weights inflated by the old flat rule | left as they are | Resetting or decaying touches backend-owned rows broadly |
| Automatic evaporation of slow weights | none | Same reason; negative outcomes already weaken walked edges |
| The independent check for promotion | structural only | An LLM verifier or a human sample costs money or time; structural checks may be too weak for G5 |
| Dream pruning of recent provisional weights | off | No dose for graphs |
| Speculative nodes with generated text | excluded | Worst measured record of any stored content |
| How an open question is marked resolved | not addressed | Would give tension pulses an end, and belongs with the tension work |
| Whether real walkers may read reactor energy as a prior | not addressed | Would couple this spec to the tension calibration |

## 20. Delivery

Five implementation plans, each written only after the previous gate.

| Plan | Stages | Depends on | Ends at |
|---|---|---|---|
| P1 | 0, 1 | nothing | G1, G1b |
| P2 | 2, including the rating push adapter in both backends (§6.6) | P1 (walk ids ride on the new output) | self-scores and anchors logged; a week of daily pushes delivered and answered; calibration running; G2 |
| P3 | 3, 4a | P2 and G2 | G3 |
| P4 | 4b | P1 for `shadow`; P3 for `hints` and `promote` | G4, G5 |
| P5 | 5, then amendment for 6 and 7 | P1 and G1b | G6 |

The offline link-precision harness of P4 needs neither seeding nor outcomes and may be
built alongside P1.
