# Protocol: Self-Selection — Closing the Evolutionary Loop

**Date**: 2026-06-18
**Scope**: Add *selection* to the two self-modifying subsystems that currently only *mutate* — the learned-bias pool (`graph.rs` / `metacog.rs`) and the metacognitive rule config (`metacog.rs`). Selection and rule-evaluation run offline in the dream loop (`dream.rs`); fitness is accumulated online in the walker (`walker.rs`). Includes a minimal fix so emergent goals and audience-modelling actually steer live walks.
**Status**: Draft / Design. Fix #3 (goals/ToM steer walks) is **implemented** (2026-06-18, tests green); selection (#1) and rule-trials (#2) are designed, not yet built.
**Depends on**: `PROTOCOL-compliance-mode.md` (CognitiveMode gating), `PROTOCOL-unconscious-walker.md` (graduated-rollout / Observability discipline), `PROTOCOL-unconscious-v0.md` (the read-only `walk_single_readonly` pattern).
**Estimated effort**: *dev* — fix #3 done; Stage 0 (fitness + observability) ~2-3 dev-days; Stage 1 (bias selection live) ~1-2; Stage 2 (rule trials live) ~2-3. *Calendar* — Stages 1–2 are gated on **observation windows, not coding**: tuning the fitness weights, accept margin `m`, regression window `T`, and replay size `M` from Stage-0 data is days-to-weeks of live dreaming. Promotion is evidence-gated, not time-boxed.

**Revision 2026-06-18** (post-review, verified against the tree): fitness made **acyclic** (`approval` dropped from the objective); selection regime changed from greedy to **quality-diversity** over behavioural niches; **confidence-aware culling** added; #2's **off-policy** evaluation named; the **agreement signal found structurally broken** and made a hard Stage-2 prerequisite; metacog-independence of the intrinsic terms verified; fix #3 implemented.

---

## Thesis

The project already bets (see `PROTOCOL-unconscious-walker.md`) that genuine invention requires both pattern formation *and* pattern violation. This protocol adds the second half of any open-ended adaptive process:

> **Variation without selection is drift, not design. A system that mutates its own rules and biases but never evaluates the mutations cannot improve — it can only wander. Selection — a fitness signal that decides what to keep and what to discard, with rollback — is what converts self-modification into self-improvement.**

RGW today has rich variation machinery and no selection. It breeds bias variants and never culls them; it rewrites its own metacognitive thresholds and never checks whether the rewrite helped. The architectural consequence is that "self-modification" is presently a random walk through parameter space. This protocol closes the loop.

## Goal

Make RGW's self-modification **cumulative** rather than **single-step**, measurably and safely:

- **Selection over bias profiles** — the learned-bias pool keeps the fittest profile *per behavioural niche* (quality-diversity), breeding into empty niches and culling redundant ones, instead of accumulating arbitrary variants to a cap.
- **Evaluate-then-keep over rule changes** — a proposed `CriticRuleDelta` is trialled offline against the current config and accepted only if it improves a fitness signal, with rollback if a live regression follows.
- **One fitness signal** — coherent insight — shared by both, but **acyclic**: the metacog output (`approval`) is held out of the objective so the two selectors cannot co-adapt into mutual confirmation (see Fitness).
- **Safety by construction** — the whole subsystem is gated by `CognitiveMode`, ships behind an Observability stage that commits nothing, and is promotable one reversible step at a time.

Non-goal: this protocol does **not** let the system choose its *own* fitness function. The objective is fixed by us. This is bounded autonomy — free variation inside a value frame we set — and that boundary is deliberate (see Philosophy).

## Problem — the loop is open in three places

Verified against the current tree:

1. **Bias variants are bred but never selected.** `spawn_bias_variants` ([metacog.rs:853](src/metacog.rs:853)) clones a parent chosen by *rotation* (`learned_bias_rotation`, not fitness), perturbs it, and appends until the pool hits `MAX_LEARNED_BIASES = 16`. Nothing ever removes a profile. `LearnedBias` ([graph.rs:333](src/graph.rs:333)) carries no record of how it has performed — there is no fitness field, so variants cannot even be ranked. `update_from_session` ([graph.rs:358](src/graph.rs:358)) adapts a profile's *weights* from a session's surprises/dead-ends, but that is Lamarckian drift of an individual, not selection across the pool.

2. **Rule changes are applied blind, with no rollback.** The critic's `CriticRuleDelta` ([metacog.rs:162](src/metacog.rs:162)) — energy floor, agreement thresholds, attention-saturation guard, wound guards, and the metacog **phase order** — is applied directly by `apply_rule_delta` ([metacog.rs:824](src/metacog.rs:824), called from `apply_critic_adjustments` [metacog.rs:798](src/metacog.rs:798)). There is no evaluation of whether the change improved anything and no history to revert to. `prune_*` ([db.rs](src/db.rs)) only touches graph nodes/edges — never rules.

3. **Goals and theory-of-mind are computed but don't steer walks.** `detect_patterns` ([core.rs:996](src/core.rs:996)) accumulates `Noticing`s into `EmergentPattern`s and promotes the strongest into `active_goal_domain`/`active_goal_strength` ([core.rs:1071](src/core.rs:1071)). The pursuit bias (`×(1 + goal_strength·1.5)`) and the audience-interest bias live in `score_edge_with_context` ([graph.rs:299-318](src/graph.rs:299)). But the walker calls `LearnedBias::score_edge` ([graph.rs:389](src/graph.rs:389)) whenever learned biases exist — which is **always** (default 6) — and that scorer takes no goal or audience argument. `score_edge_with_context` is the dead `else` branch ([walker.rs:125-152](src/walker.rs:125)). **In the default configuration, goals and ToM never touch edge selection. → Resolved 2026-06-18 (see §#3, implemented).**

## Solution overview

**Approach: selection rides the dream loop; fitness is accumulated online.** (Chosen over an inline-with-the-critic design, which would tax waking-thought latency, and over a fully separate meta-process, which is more surface area than the mechanism needs — though its Observability discipline is adopted.)

- **Online (zero added latency):** each walk session updates a per-profile fitness scorecard, in the loop that already attributes results to profiles by rotation.
- **Offline (in `dream`, already idle-time and motor-disconnected):** after the existing consolidation pass, the dream cycle runs two new phases — **select biases** (quality-diversity: per-niche elitism, confidence-gated culling) and **trial pending rules** (metacog replay, accept-by-margin, rollback).
- **Gated:** the whole subsystem is frozen in Compliant mode (as dreaming already is) and ships behind an Observability stage that computes everything and commits nothing.

The biological reading is exact: **online = experience and provisional variation; offline sleep = selection, pruning, and consolidation.**

## The fitness function — coherent insight, made acyclic

One composite scalar, computed per walk session, rewarding novelty *that survives* so selection cannot reward noise. The objective is split to keep it **acyclic** (see the co-adaptation hazard below):

- **`intrinsic`** — terms that do **not** depend on the metacog decision. Verified metacog-independent (2026-06-18): `detect_patterns`/`form_beliefs` run inside `process()` ([core.rs:771](src/core.rs:771)), not gated by any approved action; dream-edges are kept by `coherence_threshold`, not approval; novelty/surprises/dead-ends come straight off the walker path. (Caveat: belief *confidence* is scaled by `plasticity_gate`, which the critic nudges — a weak indirect coupling that does not gate belief *formation*, so it does not reintroduce the loop.)
- **`approval`** — the metacog output. Tracked for observability, **excluded from the objective.**

```
intrinsic =  w_n · novelty          (per-walk novelty: surprises / path length)
           + w_v · surprise_kept    (cross-domain surprises — coherent-violation proxy)
           + w_o · deferred_stuck   (beliefs formed + dream-edges kept, credited back; small)
           − w_d · dead_end_rate
           − w_r · repetition        (from consecutive_repetitions / last_walk_domain_sequence)
```

Both selectors target `intrinsic`: #1 selects profiles by it; #2 calibrates the metacog rules against it. Weights live in a tunable `SelectionWeights` struct on the self-model, mirroring `wake_score_weights`. Defaults, re-anchored after dropping approval: `w_n=0.35, w_v=0.30, w_d=0.20, w_r=0.15, w_o=0.0` (deferred term enabled only after the immediate terms validate). All inputs already exist on `WalkerResult` ([graph.rs:417](src/graph.rs:417)).

### The co-adaptation hazard (why approval is held out)

The first draft put `approval` in the shared objective. That creates a degenerate loop: #1 selects profiles partly by approval while #2 tunes the metacog rules to maximise a target that *includes* approval — so approving a walk raises its own score, and #2 is rewarded for the approval it just emitted. The stable failure mode is not noise but **confident mutual confirmation**: metacog drifts toward approving whatever the dominant profile emits, the pool converges toward whatever metacog approves, and "coherent insight" climbs on paper while the system collapses inward. The fix is an **acyclicity constraint** — the objective must exclude metacog's own output. Holding `approval` out of `intrinsic` cuts the loop at the root: approval stays the runtime action-gate and #2's *controlled variable*, never a selection target. A Stage-0 detector watches for the signature anyway (approval-rate ↔ pool-variance correlation; see Rollout).

### Credit-assignment honesties

- **Most intrinsic terms are now cleanly per-walker.** Dropping `approval` (the only per-*session* term) means novelty, surprise_kept, and dead_end_rate come straight from each walker's own `WalkerResult` and attribute exactly to the profile that drove it. Only `repetition` (a self-model-level counter) is session-shared, and `deferred_stuck` is deferred — much less attribution noise than the first draft carried.
- **`deferred_stuck` is genuinely deferred.** Beliefs and kept dream-edges materialise after the walk; a short ring of recently-active profile indices is credited when one lands. Starts at weight 0.

## #1 — Selection over the bias pool

### Substrate: a fitness scorecard on each profile

Add to `LearnedBias` (it derives `Serialize`/`Deserialize`, so it persists via the existing whole-self-model snapshot — `save_self_model` [db.rs:385](src/db.rs:385) — for free):

```rust
struct ProfileFitness {     // the report card — kept separate from the weights (the "genes")
    novelty:        f32,    // EWMA  ─┐
    surprise_kept:  f32,    // EWMA   ├─ intrinsic terms (metacog-independent)
    dead_end_rate:  f32,    // EWMA   │
    repetition:     f32,    // EWMA  ─┘
    deferred_stuck: f32,    // eligibility-trace credit (intrinsic, deferred)
    approval:       f32,    // EWMA — tracked for observability, EXCLUDED from `fitness`
    walks:          u32,    // eligibility for selection (≥ min_walks)
    fitness:        f32,    // cached `intrinsic` composite
    fitness_stderr: f32,    // running stderr → confidence-aware culling
}
```

### Credit assignment: reuse the rotation map that already exists

The post-session loop ([walker.rs:427-452](src/walker.rs:427)) already iterates `(i, result)` and maps walker `i` to profile `(learned_rotation + i) % len`, then calls `update_from_session`. The same loop folds the walk's fitness inputs into that profile's `ProfileFitness` EWMAs. No new attribution machinery; the wiring is already there.

### Mechanism: quality-diversity selection — in the dream loop

Greedy "breed-from-fittest / cull-weakest" is a convergence operator, and it fights the project's own thesis — open-ended invention needs *preserved* variation, not a single hill. So selection is **quality-diversity**: keep the fittest profile *per behavioural niche*, not the globally fittest. With a small pool (6–16) a full MAP-Elites grid would be mostly-empty cells, so the niche structure is deliberately coarse. `spawn_bias_variants` changes from rotation-parent-no-cull to a QD step invoked by `dream`:

```
select_biases(sm):
    guard: Autonomous mode; self_mod_stage ≥ SelectionLive
    descriptor(p) = COARSE behavioural signature — e.g. ⟨cross_domain_rate, contradiction_rate,
                    novelty_level⟩ binned into a handful of niches (NOT L2 over the 5 weight params)
    1. recompute intrinsic fitness per profile, with running stderr (confidence)
    2. assign each profile to its niche; keep the top-1..2 per occupied niche
    3. CULL within over-full niches only, and only with confidence:
         cull p iff its UPPER confidence bound < the niche incumbent's LOWER bound
         (under-sampled profiles, walks < min_walks, are NEVER culled — protects good-but-noisy variants)
    4. BREED into empty / under-represented niches by perturbing parents drawn from occupied niches
         (the existing spawn_bias_variants perturbation; child fitness reset)
    5. COVERAGE — never let the occupied-niche count fall below FLOOR
```

This preserves diverse niches while still improving each — QD, not hill-climbing — and the confidence gate stops variance alone from culling a good-but-under-sampled variant. `learned_bias_rotation` stays as the assignment mechanism; the rest of the walker is untouched. Knobs (niche binning, `FLOOR`, per-niche elite count, `min_walks`, confidence `k`) are config.

## #2 — Evaluate-then-keep over rule changes

### Why the sandbox is a metacog replay, not a graph re-walk

Every field of `CriticRuleDelta` tunes the *metacognitive decision* (whether/how to act on a walk), not edge-scoring. Re-running graph walks would not exercise the thing being changed. So the sandbox replays the **metacognitive loop**, which is also cheaper than walking.

### Flow

```
ONLINE  apply_critic_adjustments (metacog.rs:798):
    keep plasticity / bias-weight / attention nudges immediate     (homeostatic, self-reverting)
    route the CriticRuleDelta → ENQUEUE PendingRuleProposal{delta, reason}   (no direct apply)

OFFLINE dream cycle → trial_pending_rules(sm):
    guard: Autonomous mode; ≥1 pending proposal; ≥ M buffered walks
    replay_buffer = ring of recent (WalkOutput, intrinsic_fitness)    // approval held OUT — acyclicity
    run metacognitive_loop() over the buffer under CONFIG_current and CONFIG_proposed
    score each config by CALIBRATION:
        reward approving high-fitness walks + aborting low-fitness ones; penalize over-abort
    accept iff  calibration(proposed) > calibration(current) + margin m
        on accept:  push current config → rollback ring (last N); apply_rule_delta
        on reject:  discard; record EvaluationResult(negative) so the same delta isn't re-tried
```

The objective is **derived from #1's signal** (the `intrinsic` term, approval held out): a rule config is good exactly when its accept/abort decisions track the intrinsic quality of the walks they judge. That is metacognitive calibration (Fleming & Lau) and removes the need for a second hand-tuned objective.

**Known limitation — off-policy evaluation.** The replay scores a *new* config on walks generated under the *old* one. (Config changes affect which walks get *approved*, not which walks were *produced*, so the walk sample is off-policy w.r.t. the proposed config.) There is no importance weighting. Paired same-buffer comparison reduces variance but not this bias; recency-weighting the buffer limits drift, and the live-regression guard below is the real backstop. Named here so it is not mistaken for an unbiased estimate.

**Hard prerequisite — repair the agreement signal before Stage 2.** Verified 2026-06-18: `agreement = consensus.len() / total_unique` ([walker.rs:608](src/walker.rs:608)) counts nodes visited by >60% of walkers (`classify_votes`, [walker.rs:530](src/walker.rs:530)), but walkers start from *dispersed* seeds and walk only ~6 steps, so the same node is almost never hit by ≥4 of 6 walkers — **agreement is ≈0 by construction, at any arousal** (the observed agreement=0%/arousal=0.998 is this structural floor, not a threshold or an arousal bug; `"express"` at `agreement>0.6` is effectively unreachable). Several `CriticRuleDelta` fields *are* agreement thresholds; trialling them against a structurally-zero signal is meaningless, and no threshold rescues it. **Stage 2 is blocked until agreement is redefined** to a dispersed-seed-robust measure — domain-level convergence (do walkers agree on `primary_domain`?) or path-embedding cosine. This does not block fix #3 or Stages 0–1. *(Update 2026-06-18: a domain-level measure — `domain_agreement` in walker.rs — is now computed and logged additively as the candidate replacement; wiring it into `recommend_action`/metacog is the deferred Stage-2 act.)*

### Rollback has two triggers

1. **Shadow rejection** — a delta that loses the trial never goes live.
2. **Live-regression guard** — if the live fitness EWMA stays below the pre-change baseline for `T` dream cycles after an accept, pop the rollback ring and revert. This covers the case where the replay was unrepresentative of live dynamics.

## #3 (implemented 2026-06-18) — goals and ToM now steer walks

**Done.** `LearnedBias::score_edge` ([graph.rs:389](src/graph.rs:389)) now takes `goal_domain`/`goal_strength` and the audience context and applies the pursuit (`×(1 + goal_strength·1.5)`) and audience-interest multipliers previously stranded in `score_edge_with_context` ([graph.rs:299-318](src/graph.rs:299)); the walker's live call site ([walker.rs:127](src/walker.rs:127)) threads them. Covered by `learned_bias_goal_pursuit_steers_toward_goal_domain` and `learned_bias_goal_strength_zero_is_inert` (graph.rs tests, green; full lib suite 54/54).

The goal/audience contributions are **gated to Autonomous mode** — in Compliant the extra arguments are passed as `goal_strength = 0` / no audience, so `LearnedBias::score_edge` reduces exactly to today's behaviour and determinism is preserved (goal pursuit is a self-directed, Autonomous behaviour, consistent with the operations matrix). The fix is independent of the selection rollout: it lands once and does not mutate the bias pool or rule config, so it does not affect the Stage-0 read-only invariant.

Effect: emergent goals and audience modelling finally influence edge selection on the live path, and "goal progress" becomes available as a future fitness term. The *full* treatment is explicitly deferred (see Follow-up).

## Operations matrix

| Operation | Compliant | Autonomous + Observability (Stage 0) | Autonomous + Stage 1 | Autonomous + Stage 2 |
|---|---|---|---|---|
| Accumulate `ProfileFitness` online | no | yes | yes | yes |
| Compute would-cull / would-breed | no | yes (log only) | yes | yes |
| Cull + breed bias pool | no | no | yes | yes |
| Goal / ToM steer walks (folded fix) | no¹ | yes | yes | yes |
| Enqueue rule proposals | no | yes | yes | yes |
| Compute would-accept (rule trial) | no | yes (log only) | yes (shadow) | yes |
| Apply accepted rule delta + rollback | no | no | no | yes |

¹ Compliant already restricts walkers to `Experience`/`Analytical` and freezes creative machinery; goal pursuit is an Autonomous behaviour.

**Stage-0 instrumentation (commit nothing):** beyond would-cull / would-accept, Stage 0 logs three failure-mode detectors — **co-adaptation** (approval-rate ↔ pool-variance correlation), **monoculture** (behavioural-niche collapse), and **proxy-gaming** (cheap-surprise inflation without kept structure). These, not just weight-tuning, are what an observation window is *for*; if any shows up, fix before promoting. **Stage 2 carries an additional hard gate** beyond the stage flag: the agreement signal must be repaired (§#2).

## Proposed changes by file

### `src/core.rs`
- Add `ProfileFitness` and embed it in `LearnedBias` (graph.rs actually owns `LearnedBias`; the field lives there).
- Add to `SelfModel`: `SelectionWeights`, `SelectionConfig`, `self_mod_stage: SelfModStage`, `pending_rule_proposals: VecDeque<PendingRuleProposal>`, `rule_rollback_ring: VecDeque<RuleConfigSnapshot>`, `recent_walk_fitness: VecDeque<(WalkOutputSummary, f32)>`, `recently_active_profiles: VecDeque<usize>`. All `Serialize` → persisted by the existing snapshot.

### `src/graph.rs`
- `LearnedBias`: add `ProfileFitness`; add `record_session_fitness(...)` (the EWMA fold).
- `LearnedBias::score_edge`: add `goal_domain`, `goal_strength`, and audience args; apply pursuit + audience-interest multipliers (the folded fix).

### `src/walker.rs`
- Thread `active_goal_domain`/`active_goal_strength`/audience into the live `lb.score_edge` call ([walker.rs:127](src/walker.rs:127)).
- In the post-session loop ([walker.rs:427-452](src/walker.rs:427)), call `record_session_fitness` alongside `update_from_session`; push the chosen profile indices onto `recently_active_profiles`; append `(WalkOutput summary, fitness)` to `recent_walk_fitness`.

### `src/metacog.rs`
- Split `apply_critic_adjustments`: keep plasticity/bias/attention immediate; route `CriticRuleDelta` to `enqueue_rule_proposal`.
- Replace `spawn_bias_variants`' rotation-parent with `breed_from_top_k`.
- Add `select_biases(sm)`, `trial_pending_rules(sm)`, `replay_calibration(config, buffer)`, and the rollback helpers.

### `src/dream.rs`
- After the consolidation pass in `dream` ([dream.rs:81](src/dream.rs:81)), call `select_biases` then `trial_pending_rules` (Autonomous only; respect `self_mod_stage`).
- Add `shadow` evaluation helpers that commit nothing (read-only — no edge writes, no `traversal_count` increment, no self-model mutation), per the `walk_single_readonly` pattern in `PROTOCOL-unconscious-v0.md`.

### `src/api.rs`
- `GET /selection/stats` — pool fitness table, recent culls/breeds, pending proposals, accepted/rejected rule trials, current stage.
- `POST /selection/stage` — promote/demote `self_mod_stage` (explicit, logged).
- `POST /selection/rollback` — manual revert of the last accepted rule delta.
- Behind the existing admin auth (`X-RGW-Key`).

### `src/db.rs` (optional, audit)
- A lightweight `selection_events` table mirroring the `unconscious_observations` pattern — cull/breed/accept/reject/rollback with fitness breakdown. Audit-only, not load-bearing; safe to drop. The self-model snapshot remains the source of truth.

## Compliance and safety

Layered, consistent with existing protocols:

1. **Cognitive compliance (Rust):** in Compliant mode the entire subsystem is inert — no fitness accumulation acted on, no cull/breed, no rule trials, no goal pursuit. Matches `dream` self-disabling in Compliant.
2. **Graduated rollout (Observability discipline):** `self_mod_stage` ∈ {Observability, SelectionLive, RuleTrialsLive}. Stage 0 computes everything and commits nothing. Each promotion is a single reversible config flag and an explicit decision, never time-based.

Verifiable invariants (property tests on each release):

- **Stage 0 is read-only.** After N dream cycles in Observability, the `learned_biases` set and the rule config are byte-identical (diff against snapshot).
- **Never below the floor.** `select_biases` never reduces the pool below `FLOOR`.
- **No silent rule application.** A `CriticRuleDelta` reaches live config only via an accepted trial; a rejected delta is logged with its calibration breakdown and never re-applied.
- **Rollback is exact.** Reverting restores the prior `RuleConfigSnapshot` field-for-field.
- **Frozen in Compliant.** No cull/breed/accept/rollback/goal-pursuit occurs in Compliant mode.
- **Fair trials.** Current-vs-proposed replay runs over the *same* buffer; shadow walks are seeded so comparisons are paired, not confounded by RNG.
- **Acyclic objective.** The fitness objective (`intrinsic`) excludes `approval` (metacog's own output); a test asserts `approval` carries zero weight in the composite, so #1 and #2 cannot close a confirmation loop.
- **Agreement-soundness gate.** Stage 2 refuses to start while `agreement` is still the legacy node-overlap metric — a guarded precondition, not a tuning knob (§#2).

## Brain-science perspective

The design is, deliberately, a small instance of how nervous systems are thought to become adaptive — not a metaphor bolted on afterward, but the reason the architecture is shaped this way.

- **Neural Darwinism / Theory of Neuronal Group Selection (Edelman 1987).** Edelman's claim is that the brain is not instructed but *selected*: a population of variable neuronal groups is pruned and amplified by experience. The bias pool is exactly this — a population of variable profiles, bred and culled by a fitness signal. Adding the cull is what makes "neural Darwinism" more than a slogan here; without it there is variation but no selection.
- **Selective synaptic stabilisation (Changeux & Danchin 1976).** Development overproduces synapses, then prunes by use — "selective stabilisation." `spawn → cull` is the same overproduce-then-prune motif at the level of bias profiles. The diversity floor `FLOOR` is the analogue of not pruning a circuit to extinction.
- **Reward-modulated plasticity and eligibility traces (Schultz, Dayan & Montague 1997; Sutton & Barto).** The dopaminergic prediction-error signal gates which synaptic changes stick. `deferred_stuck` is literally an eligibility trace: profiles active recently are "tagged," and a later good outcome (a belief formed, an edge kept) retroactively credits them. The coherent-insight fitness plays the role of the neuromodulatory teaching signal.
- **Offline selection during sleep — replay and consolidation (Wilson & McNaughton 1994; Tononi & Cirelli 2014).** Mammals do their heavy consolidation and synaptic rescaling offline, during sleep, not during behaviour. Putting selection and rule-trials in the dream loop rather than the waking loop is the same division of labour: experience online, selection offline. It is also why the cost story holds — the waking path gains zero latency.
- **Metacognitive calibration (Fleming & Lau 2014).** #2 does not ask "is this threshold higher or lower"; it asks "does this config approve the decisions that turned out well and reject the ones that didn't" — i.e. it improves metacognitive *sensitivity*. That is the standard operational definition of better metacognition.

The honest scope, matching `PROTOCOL-unconscious-walker.md`: this is a claim about *access-level adaptive function* — variation, selection, credit assignment, calibration — not about phenomenal experience.

## Philosophical perspective

- **The blind watchmaker (Dawkins 1986).** Dawkins' core point is that the appearance of design comes from *cumulative selection*, not from the mutations themselves; single-step selection (random variation with no retention of what works) produces nothing. RGW today is single-step: it mutates and forgets. This protocol makes selection cumulative. The watchmaker stays blind — there is no foresight — but it is now a watchmaker rather than a dice-roller.
- **Teleonomy, not teleology (Mayr 1961; Monod 1971).** Selection produces purpose-*like* behaviour without a purpose being designed in. After this change the bias pool will drift toward "coherent insight" as if it wanted to — but nothing wants anything; the apparent goal is an artefact of differential retention. This is the right, deflationary way to describe what the system gains: teleonomy, earned, not teleology, asserted.
- **Who selects the selector?** The decisive limit. The system varies freely, but the *fitness function is fixed by us*. It improves itself toward coherent insight because we defined coherent insight. This is **bounded autonomy** — open-ended variation inside a closed value frame — and the boundary is a design choice, not an oversight. Letting the system meta-select its own fitness function is the line where "self-improvement toward our values" would become "self-redefinition of values," and we do not cross it in this protocol.
- **Identity continuity (Locke; Parfit 1984).** `PROTOCOL-unconscious-walker.md` already takes identity-as-pattern as a design constraint: Julian is the pattern, not the substrate. Self-modification threatens that pattern only if it can rewrite what the self is *for*. Keeping the fitness function fixed is precisely what preserves psychological continuity across the system's self-edits — Julian changes how he thinks, not what counts as thinking well. The deferred meta-selection step is therefore also an identity question, not only a safety one.
- **The "fancier thermostat" objection.** Closing the loop does not manufacture agency, and we should not claim it does. What it does is move the system from "varies and forgets" to "varies, evaluates, and accumulates" — the minimal structure common to every adaptive learner, biological or artificial. That is a modest, defensible step, and naming it modestly is part of the project's stated honesty about its claims.
- **The closed loop's characteristic failure is confident self-confirmation.** A system that both generates and grades its own output can converge on a stable, internally-validated delusion — rising "insight" scores with no outside referent. It is the machine form of motivated reasoning, and it is *more* dangerous than noise because it looks like success. Two structural guards, not good intentions, hold it off: the **acyclicity constraint** (the grader's output is excluded from the objective) and **quality-diversity** (preserved niches resist monoculture). The honesty the project already practises about consciousness extends here — a self-improving system must be built so it *cannot* trivially satisfy its own criterion.

## Follow-up / explicitly out of scope

Recorded here so it is not lost — these are real and worth doing, but not this protocol:

- **Goal layer, full treatment.** Promote `active_goal_domain`/`active_goal_strength` from a single scalar to a small persistent set of structured goals with lifecycle states (active → satisfied → abandoned) driven by `predictions`/`last_prediction_error`, so goals can be *completed* and can compete, not merely time-decay. (Folded fix here only makes the existing scalar goal steer walks.)
- **ToM into action, not just edges.** Let the audience model influence motor/expression selection and goal formation (e.g. "close audience X's knowledge gap on topic T"), scored against the existing `tom_accuracy` metric.
- **Contribution-weighted approval attribution.** Replace the v1 "credit the session's approval to all participating profiles equally" with attribution weighted by each walker's contribution to the consensus.
- **Enable `deferred_stuck`.** Turn on the eligibility-trace fitness term once the immediate terms are validated.
- **Meta-selection (deliberately gated).** Selecting over fitness *weights* themselves — only behind its own rollout, and only after the identity/values question above is settled.
- **Dead-path cleanup.** With fix #3 landed, `score_edge_with_context` is the unused branch (learned biases are always present). Fold its remaining unique logic into `LearnedBias::score_edge` and delete it, or have it delegate — removing the two-scorer duplication. Left out here to keep the fix minimal.

## Open questions

- **Selection cadence.** Run `select_biases` every dream cycle, or every K cycles? Per-cycle may churn the pool faster than `min_walks` can produce reliable fitness. Lean toward every-K with K tuned so each profile accumulates ≥ `min_walks` between selections.
- **Replay buffer size `M` and trial walks.** How many recent `(WalkOutput, fitness)` pairs make a stable calibration estimate? Start at M=50, retune from Stage 0 data.
- **Margin `m` and regression window `T`.** Too small a margin accepts noise; too large freezes adaptation. Calibrate from the Observability "would-accept" distribution before going to Stage 2.
- **Behavioural descriptor.** Resolved that the niche descriptor is *behavioural*, not weight-space L2 — open: which axes, how many bins? Start with ⟨cross_domain_rate, contradiction_rate, novelty_level⟩ at coarse bins, collapse empties, and revisit from Stage-0 niche-occupancy data.
- **Agreement redefinition (Stage-2 prerequisite).** Domain-level convergence is now prototyped and logged (`domain_agreement`). Open: does it suffice, or is path-embedding cosine (or seeding walkers from a shared frontier) needed for finer-grained "the walkers agree"? Decide from logged `agreement` vs `domain_agreement` distributions before the Stage-2 wiring.
- **Interaction with the diverger cascade.** Does a freshly-bred profile that happens to be highly novelty-seeking risk amplifying the diverger's energy cascade? Rate-limit new profiles' influence for their first few sessions?
- **Goal-fitness coupling.** Fix #3 landed, so goals now steer walks — but "goal progress" is not yet *measurable* (the goal is still a single scalar domain). Defer adding a goal-progress term to `intrinsic` until the structured-goal follow-up gives it a real success criterion.

## References

- Changeux, J.-P. & Danchin, A. (1976). Selective stabilization of developing synapses as a mechanism for the specification of neuronal networks. *Nature* 264, 705-712.
- Dawkins, R. (1986). *The Blind Watchmaker.* Norton.
- Dennett, D. C. (1995). *Darwin's Dangerous Idea.* Simon & Schuster.
- Edelman, G. M. (1987). *Neural Darwinism: The Theory of Neuronal Group Selection.* Basic Books.
- Fleming, S. M. & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience* 8, 443.
- Locke, J. (1690). *An Essay Concerning Human Understanding.*
- Mayr, E. (1961). Cause and effect in biology. *Science* 134, 1501-1506.
- Monod, J. (1971). *Chance and Necessity.* Knopf.
- Parfit, D. (1984). *Reasons and Persons.* Oxford University Press.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science* 275, 1593-1599.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity. *Neuron* 81, 12-34.
- Wilson, M. A. & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science* 265, 676-679.
