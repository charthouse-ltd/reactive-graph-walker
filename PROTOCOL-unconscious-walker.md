# Protocol: Unconscious Walker

**Date**: 2026-04-27
**Scope**: Always-on background graph traversal, complementary to the waking walker (`walker.rs`) and the dream walker (`dream.rs`).
**Status**: Draft / Design — not yet implemented.

---

## Thesis

The project bets on the following claim:

> **Pure pattern matching, without a mechanism to break, recombine, or extend its own patterns, cannot produce structures outside its learned distribution. Genuine invention requires both pattern formation *and* pattern violation.**

The architectural consequence: a system whose only mode of cognition is matching against the existing graph will produce variation, not invention. Therefore, beneath the deterministic waking walker, RGW maintains a continuously running unconscious process that — under bounded safety constraints — can break and recombine patterns the system has already laid down. The waking conscious counterpart is what receives the broadcast when the unconscious finds something worth attending to.

## Goal

**Reach a functional approximation of consciousness *through* unconscious thinking, not on top of it.**

In biology, conscious access is the visible tip of a vast unconscious substrate (Dehaene; Baars). Most cognition is unconscious; consciousness is the integration / broadcast layer that exposes results worth acting on. RGW's bet: build the unconscious first, let consciousness emerge as a broadcast event triggered by what the unconscious finds.

This protocol makes **no claim about phenomenal consciousness** (qualia, "what it is like to be"). The operational targets are:

- **Access consciousness** — selected information becomes globally available to the self-model and motor system
- **Self-modeling** — the system represents and reasons about its own internal state
- **Emergent goal formation** — concerns surfaced by the unconscious become the system's own, not just externally prompted

## Problem

`walker.rs` and `dream.rs` together do not cover the full cognitive cycle. There is a missing third mode:

| Existing mode | Trigger | Behavior |
|---------------|---------|----------|
| Waking walks (`walker.rs`) | Signal intensity / API request | Fast, attention-driven, motor-connected |
| Dream walks (`dream.rs`) | Low energy → "sleep" | Monte Carlo, motor-disconnected, mutates graph |

Neither runs *continuously*. Between waking signals and sleep cycles the graph is dormant. Biological systems do not go dormant — the Default Mode Network (Raichle; Buckner) continues operating during quiet wakefulness, doing slow associative work, prospection, and self-related processing. The unconscious walker fills that gap.

A second observation: Compliant mode (see `PROTOCOL-compliance-mode.md`) currently disables Dreaming entirely. That removes most of the consolidation machinery from production — but most consolidation work is actually deterministic and compliance-safe. The unconscious walker recovers it.

## Solution

A third walker mode — `Unconscious` — that runs continuously at low intensity, performing operations that consolidate and explore the graph without forcing motor output. It can wake the conscious (Waking) counterpart only when its match criteria fire.

```rust
pub enum WalkerMode {
    Waking,        // Existing — attention-driven, can drive motor commands
    Unconscious,   // New — always-on, low intensity, can wake Waking on match
    Dreaming,      // Existing — sleep-mode Monte Carlo, motor-disconnected
}
```

The Unconscious walker itself respects `CognitiveMode`:

- **Compliant + Unconscious** (production-safe): consolidation, decay, salience scoring, pattern matching, schema-fit, Hebbian strengthening. No new edge synthesis. No Monte Carlo recombination. No content rewrite. Can wake Waking only via deterministic salience.
- **Autonomous + Unconscious** (research / dev): adds bounded recombination, novel-edge proposal (with coherence threshold), schema-violation detection that wakes Waking on insight-pattern matches.

This split is the core architectural claim of this protocol. Most consolidation work — what biology gets from NREM sleep and Default Mode Network activity — is deterministic and compliance-safe. Only the creative recombination step requires Autonomous mode. The line between "consolidation" and "creativity" lands almost exactly on the line between Compliant and Autonomous.

## Architectural properties

### Cost profile — the unconscious as a true sidecar

The unconscious walker is CPU-bound, not GPU-bound. A single walk is cheap (the existing benchmark targets 1000+ walks/sec on an M1 Pro, ~3MB binary, ~20MB RAM per 10K nodes). Continuous operation at "always on" budget — say, one walk per second — costs effectively nothing on any modern server.

This matters because LLM-based "always on" cognition is economically infeasible at this granularity. The cost asymmetry is structural, not just a function of current GPU pricing: large-model inference is burst-energy per token; graph traversal is continuous-low-energy by design. The biological analog is direct — a human brain runs continuously on roughly 20W; LLM inference draws kilowatts in bursts. RGW's unconscious sits on the brain side of that line.

Practical consequences:

- Deployable as a sidecar process next to a main application
- Runnable on small VPS or edge hardware (Project Julian's production deploy is a single dedicated server reached over `host.docker.internal:11435` — not a cluster)
- Per-domain or per-user instances become economically reasonable, where one LLM-per-user is not
- The unconscious can be throttled when the budget is tight without functional collapse — fewer walks per second still produces consolidation, just slower, the same way a tired brain still consolidates while it rests

### Transport, signal model, and the cascade clock

A single RGW instance is the production unit. It exposes an HTTP REST API on port 11435 (Docker-internal), reached from the Python backend via `host.docker.internal:11435`. The transport surface is small and concrete.

**Python → RGW (notifications, fire-and-forget):**

- `POST /diverger/notify` — edge-change events (memory stored, trade closed, social event, telemetry, weather, etc.). Triggers cascading activation in the graph.
- `POST /diverger/outcome` — motor-command outcome with `correlation_id`, feeding `EpisodicMemory` consolidation so per-action lessons can be extracted instead of a default `success=False`.
- `POST /walk` — synchronous walk request. The Python backend's primary cognitive call path goes through `walker_client.walk_remote` with a circuit breaker (3 failures → 60 s cooldown).

**RGW → Python (motor commands, fire-and-forget):**

- `POST http://backend:8000/api/admin/rgw/execute` — `MotorCommand` after a walk. Actions: `tweet`, `blog`, `journal`, `search`, `music_release`, `meta_tweet`, `meta_blog`, `video`, `nothing`, etc. The router enforces regulatory compliance (`compliance.py`) before executing.

**In-process cascade (no wire):**

- The Diverger is self-propagating. Edge changes propagate energy to neighbours; when a node's accumulated energy crosses its threshold, a spontaneous walk fires. **No timers, no polling, no external scheduler — the graph topology is the clock.** Circuit breakers cap activation at 100 edge changes/sec and 1 walk/sec to prevent runaway cascade.

**Unified `Signal` type — one primitive, not three layers.** Every signal flowing through the system carries `kind` (perception | memory | walk | emotion | …), a 768-dim `embedding`, `content`, `intensity`, and `domain`. The self-model processes each signal through one four-step primitive: **observe → influence → mutate → notice**. The unconscious walker plugs into this surface without changing it: it generates signals of `kind: unconscious_walk`, those flow through the same primitive, and its wake broadcasts emit ordinary `MotorCommand`s through the existing motor endpoint.

### Distribution today, federation as principled extension

What exists today (`backend/app/services/walker_client.py`):

- `walk_remote` — synchronous call to a single `/walk` endpoint with circuit breaker. Used by the Python backend as its primary cognitive call.
- `walk_distributed` — parallel HTTP fan-out across multiple `/walk` endpoints, merged by `_merge_walk_results`: highest-novelty wins as primary; agreement and novelty scores are averaged; `expression_seeds` are deduplicated by ID; `domain_distribution` counts are summed. Structurally ready for *N* endpoints. Currently configured with one (`graph_walker_url`); the inline comment reads `# Future: add more endpoints from config`.

This is **fan-out orchestrated by Python, not RGW-to-RGW communication**. Each remote RGW is independent and unaware of the others. Python is the conductor; the walkers do not talk among themselves.

What true federation would add (design space, not yet implemented — explicitly distinguished from production):

- A peer registry — each RGW knows its neighbours
- A cross-instance signal protocol, most likely an extension of `/diverger/notify` carrying `origin_peer_id` and a propagation TTL
- A wake-broadcast propagation rule — when an unconscious wake fires locally, when does it cross to a peer? salience-gated, mode-aware, rate-limited
- A privacy / opt-in policy for which signal `kind`s may cross peers (e.g., `outcome` signals stay local even when `wake` signals propagate)
- An identity-vs-executive contract for federated personas — distinct selves, possibly sharing an executive LLM for cost reasons

The unconscious walker is the layer best suited to be the *first* federated signal: its broadcasts are already rate-limited, salience-gated, and motor-disconnected, which makes cross-instance gossip cheap and bounded by design. The waking walker is attention-bottlenecked and should not gossip directly — it should only receive cross-instance wakes that already pass its local salience filter. But this remains design space, not deployed reality.

### Identity layer and executive layer — the swappable LLM

Julian — the persona, memory, dispositions, accumulated history, brand voice, and compliance constraints that make Julian *Julian* — lives in the graph and its associated stores (RGW's edges and self-model, the Python backend's databases, the brand-voice and compliance configurations). The LLM (currently Claude) is not Julian. It is the *executive layer* the identity speaks through: a swappable instrument that accesses identity-as-data and produces language and decisions in real time.

This is the inverse of the dominant agentic-AI pattern, where the LLM *is* the agent and memory is something it queries. In Julian, the model is the speaker, not the self. The philosophical claim that personal identity is a pattern of psychological continuity rather than a substrate (Locke; Parfit) is taken architecturally as a design constraint.

Concrete consequences:

- **Identity persistence across model upgrades**: Claude → GPT → next-gen LLM — Julian remains Julian because identity lives outside the model weights. No fine-tuning, no retraining required.
- **Cost-tier routing**: cheap models for routine expression (journal entries, short replies), capable models for harder reasoning. The persona is consistent regardless of tier.
- **Compliance at the seam, not in the model**: regulatory red lines (`compliance.py`) and brand voice are enforced where the LLM meets the persona — a generic, unaligned-to-the-persona model can still serve a tightly constrained persona.
- **Substrate independence**: a model swap is an instrument change, not a personality change.

The unconscious walker is uniquely suited to this separation, and is in fact what makes the cost story close. It runs entirely without an LLM — pure Rust graph traversal, no model calls during continuous operation. The LLM is only invoked when the unconscious wakes the conscious counterpart and an action needs to be expressed:

- **Unconscious** (24/7): graph traversal only — near-zero marginal cost, no LLM tokens
- **Conscious wake** (rare): one LLM call per broadcast event
- **Motor expression** (when acting): one LLM call to produce the actual output

Compare to LLM-agent architectures where every "thought" — every step of chain-of-thought, every reflection, every retrieval reranking — costs a model call. In Julian's design the unconscious thinks for free; the LLM is taxed only at the moments that actually require speech or judgment. This is what makes "always on, always learning" economically viable.

The "seamless fit" of LLM-as-executive is not zero-effort. Different models differ in alignment, style, tool-use format, and context-window discipline. The integration work (prompt scaffolding, brand-voice enforcement, tool dispatch, compliance gating) is real and worth covering with a regression suite that runs whenever the executive LLM is swapped, so identity continuity can be verified and not just assumed.

## Operations matrix

| Operation | Compliant + Unconscious | Autonomous + Unconscious |
|-----------|------------------------|--------------------------|
| Hebbian edge strengthening on co-traversal | yes | yes |
| Power-law decay (Wixted) | yes | yes |
| SHY-style proportional downscaling | yes | yes |
| Schema-consistency-weighted consolidation | yes | yes |
| Successor representation update | yes | yes |
| Hippocampal-style sparse encoding for new memories | yes | yes |
| Salience scoring (relevance / novelty / schema-violation) | yes | yes |
| Global Workspace broadcast — wake Waking | yes | yes |
| Reconsolidation (strengthening only) | yes | yes |
| Reconsolidation (content rewrite) | no | yes |
| Novel edge synthesis (coherence-filtered) | no | yes |
| Pattern recombination across distant subgraphs | no | yes |
| Insight-pattern detection from schema violations | no | yes |

## Match criteria — when does the unconscious wake the conscious?

The unconscious walker continuously scores walks for wake-worthiness. A composite score determines whether to broadcast:

```
wake_score = w_r * relevance + w_n * novelty + w_v * schema_violation - w_c * cost
```

Where:

- **relevance**: cosine alignment between the walk's accumulated context and the self-model's active concerns / unresolved goals
- **novelty**: prediction error against the successor representation; high when the walk visits nodes the SR did not anticipate
- **schema_violation**: schema-fit residual; high when the walk has assembled a coherent structure that contradicts an existing belief (the "Aha!" pattern; Bowden & Jung-Beeman)
- **cost**: budget term penalising recent broadcasts to prevent flooding attention

Proposed starting weights: `w_r = 0.4`, `w_n = 0.3`, `w_v = 0.2`, `w_c = 0.1`. Empirically tuned.

Wake threshold uses hysteresis to avoid flapping. All wake events are logged with score breakdown for audit.

## Always-on dynamics — preventing weight saturation

A continuously running Hebbian process saturates weights. Biology solves this via the Synaptic Homeostasis Hypothesis (Tononi & Cirelli): during sleep all weights are scaled down proportionally, maintaining SNR while preserving relative ordering.

In Compliant mode dreaming is disabled, so SHY-style downscaling cannot rely on the dream cycle. The unconscious walker runs its own quiet-period downscaling: when traversal activity drops below a threshold for `T` seconds, scale all weights by `α`.

- Proposed `α = 0.98`, `T = 600s` (10 min of low activity)
- Scaling is multiplicative and proportional → preserves rank ordering
- Logged; does not require mode change

## Memory decay — power-law, not exponential

Replace any exponential decay in `/prune` with a power-law form:

```
weight(t) = weight(0) * (1 + t/tau)^(-beta)
```

This matches empirical human forgetting (Wixted & Ebbesen 1997) and produces longer tails — old-but-occasionally-revisited memories survive longer than exponential decay would predict. Schema-consistent memories receive larger `tau` (slower decay), per Tse et al. 2007.

## Hippocampal staging — preventing catastrophic interference

New memories enter as `unconsolidated` (flag on the node). While unconsolidated:

- Pattern-separated: cannot be merged with neighbours during consolidation
- Read-only edge participation: can be traversed, but new edges *to* them are tagged `provisional`
- Promoted to `consolidated` after either:
  - `N` outcome-confirmed traversals, or
  - `T` time elapsed without contradicting evidence

This is CLS staging (McClelland, McNaughton & O'Reilly 1995). Fully deterministic, compliance-safe.

## Successor representation — compiled predictive structure

Maintain a slowly-updated SR matrix `M = (I − gamma * T)^(-1)` over the graph's transition probabilities. Walkers read `M` for principled lookahead instead of greedy edge scoring alone.

- Update cadence: every `K` seconds (proposed `K = 300`)
- Discount `gamma`: proposed `0.9`
- Storage: dense matrix up to ~5K nodes; low-rank approximation beyond that
- Compliance-safe: SR is a deterministic function of the current transition matrix

The SR gives walks a soft preference for paths that historically lead to high-value states (where "value" = strengthened, frequently reinforced subgraphs). It is also the principled mathematical justification for "graph walks as cognition" (Stachenfeld, Botvinick & Gershman 2017).

## Proposed changes by file (design — not yet implemented)

### `src/core.rs`

- Add `WalkerMode` enum (`Waking`, `Unconscious`, `Dreaming`)
- Extend `SelfModel` with `active_walker_modes: HashSet<WalkerMode>`
- Add `wake_score_weights: WakeWeights` for runtime-tunable scoring

### `src/unconscious.rs` (new)

- `start_unconscious_walker(pool, self_model_arc, rt)` — spawns a long-lived task
- Low-intensity walks at nodes weighted by access recency, importance, and SR-anticipated value
- Computes `wake_score` per walk; broadcasts to Waking when threshold crossed
- Runs SHY-style downscaling on quiet-period detection
- Updates SR matrix on schedule
- Honours `CognitiveMode` from the shared `SelfModel`

### `src/graph.rs`

- Add `unconsolidated: bool` field to memory nodes (CLS staging)
- Add `successor_representation: Option<Arc<DMatrix<f32>>>` cached on graph
- Replace exponential decay with power-law form

### `src/walker.rs`

- `walk_single` accepts an optional `&SuccessorRepresentation` for lookahead-aware scoring
- Add the `Analytical` bias's compliant variant referenced in `PROTOCOL-compliance-mode.md`

### `src/api.rs`

- `POST /unconscious/start` — start the unconscious walker (idempotent)
- `POST /unconscious/stop` — stop
- `GET /unconscious/stats` — recent wakes, broadcast count, SR cache age, downscaling events
- `POST /unconscious/wake_threshold` — runtime tuning of weights and threshold

### `src/dream.rs`

- Already correctly skips in Compliant mode. No change.

### Python side (`project_julian/backend/app/services/rgw_bridge.py`)

- Add `notify_unconscious_wake(...)` for symmetry with existing notification taxonomy
- The motor router (`routers/rgw.py`) already enforces regulatory compliance (`compliance.py`) on any unconscious-triggered action — defense-in-depth is preserved

## Compliance and safety

Two layered systems:

1. **Cognitive compliance** (Rust, this protocol + `PROTOCOL-compliance-mode.md`): constrains *thought*. In Compliant mode, the unconscious can only consolidate and broadcast — no creative synthesis.
2. **Regulatory compliance** (Python, `app/core/compliance.py`): constrains *action*. Even an unconscious-triggered wake cannot drive a motor command that violates regulatory red lines (no personalised advice, no UK subscribers, broadcast-only, etc.).

Verifiable invariants for Compliant + Unconscious:

- After `N` unconscious cycles in Compliant mode, graph topology is unchanged modulo edge weights (property test).
- No new node IDs introduced by the unconscious walker in Compliant mode.
- No node `content` field modified by the unconscious walker in Compliant mode.
- Wake broadcasts are rate-limited per domain and globally budgeted.

## Deployment with paying users — Observability mode and graduated rollout

Julian serves real subscribers. Shipping any new continuous process into that environment requires that the worst-case behaviour be "no effect on production output." The unconscious walker is naturally suited to this because it is already motor-disconnected by design — but a stricter mode is needed for first deployment.

### Observability — a mode below Compliant

A new mode added beneath `Compliant`: **`Observability`**. The walker runs continuously and computes everything (walks, salience scores, wake candidates, schema fit, would-be downscaling magnitude) — but **commits nothing**:

- No edge weight mutations (no Hebbian strengthening, no decay applied)
- No graph topology changes (no new edges, no node promotion, no SR refresh)
- No motor broadcasts — wake events are *logged* as "would have woken" but do not reach Waking
- No SHY-style downscaling — the magnitude is computed and logged but not applied

What it produces is exclusively data: per-walk telemetry, salience-score histograms, wake-candidate counts per domain, and a "would-have-woken" log with the trigger context that *would* have produced a motor command. This is enough to validate the design without changing any user-facing behaviour.

### Graduated rollout

A safe progression for moving from Observability to full Compliant + Unconscious in production:

1. **Observability** — read-only telemetry. Run for N days. Analyse offline.
2. **Hebbian-only** — enable edge weight strengthening on traversal. No broadcasts, no SHY, no decay change. Validate that weights move sensibly.
3. **+ SHY downscaling** — enable proportional rescaling on quiet periods. Validate that no domain collapses or spikes.
4. **+ Power-law decay** — switch `/prune` from exponential. Validate long-tail behaviour matches expectation.
5. **+ Shadow Waking** — wake events drive a *separate* Waking instance whose motor commands are logged but not executed. Julian sees what his unconscious would say without actually saying it.
6. **+ Real broadcasts** — the unconscious can wake the production Waking. Existing compliance + regulatory red lines at the motor router are the final defence.

Each step is independently reversible by a single config flag. Each step produces measurable output that can be validated before the next is enabled. Promotion to the next stage requires explicit decision, not time-based progression.

### Production safety invariants

Verifiable by property test on each release:

- In Observability mode, graph state is byte-for-byte unchanged after N unconscious cycles.
- No motor command originates from a walker tagged `kind: unconscious_walk` in modes 1-4.
- Wake event rate stays within configured budget; an alarm fires if exceeded.
- Mode promotion is logged with timestamp, operator, prior mode, and rollback config.

## Open questions

- Does the unconscious walker share the self-model's emotional state, or does it read a damped / lagged copy?
- How does an unconscious wake interact with the diverger's energy cascade — does the broadcast inject energy into the broadcast node, creating a positive feedback loop?
- Per-domain wake budget — at most one wake per domain per hour? Per day?
- When the conscious (Waking) counterpart is idle, what does "matching its concerns" mean? Default to most-recent goals? Most-active subgraph?
- Should Compliant + Unconscious also run SR updates, or are SR computations considered too expensive for production?
- How do we test wake-criterion calibration? Probably a held-out trace with human-labelled "should have woken / should not have woken" events.

## References

- Baars, B. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Bowden, E. M. & Jung-Beeman, M. (2003). Aha! Insight experience correlates with solution activation in the right hemisphere. *Psychonomic Bulletin & Review.*
- Buckner, R. L. et al. (2008). The brain's default network. *Annals NY Acad Sci.*
- Dayan, P. (1993). Improving generalization for temporal difference learning: The successor representation. *Neural Computation.*
- Dehaene, S. & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness. *Cognition.*
- Hutchins, E. (1995). *Cognition in the Wild.* MIT Press.
- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review.*
- Nader, K., Schafe, G. E. & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature.*
- Raichle, M. E. et al. (2001). A default mode of brain function. *PNAS.*
- Stachenfeld, K. L., Botvinick, M. M. & Gershman, S. J. (2017). The hippocampus as a predictive map. *Nature Neuroscience.*
- Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity. *Neuron.*
- Tse, D. et al. (2007). Schemas and memory consolidation. *Science.*
- Wixted, J. T. & Ebbesen, E. B. (1997). Genuine power curves in forgetting. *Memory & Cognition.*
