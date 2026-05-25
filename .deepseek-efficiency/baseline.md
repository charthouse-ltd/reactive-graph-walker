# PHASE 0 — DeepSeek cost baseline

_Captured on the `deepseek-efficiency` branch, against `main` @ e6cda15._

## Method

The brief asks for a measured run (10 min / N ticks). That is **not feasible in
this environment**: the binary needs `DATABASE_URL` (a live Postgres) and
`DEEPSEEK_API_KEY`, and a real loop would bill the DeepSeek account. So the
baseline below is **analytical** — derived directly from the code paths — and is
paired with **new instrumentation** (`src/metrics.rs`, `GET /metrics`) so the
same numbers can be captured from a live run later, with no further code work.

Every figure here is traceable to a specific line of the pre-change code.

## 1. Models and the `usage` object

- `src/llm.rs` is a direct DeepSeek HTTP client (`POST /v1/chat/completions`),
  models `deepseek-chat` / `deepseek-reasoner` (`src/llm.rs:8-10`).
- **The `usage` object was discarded.** `DeepSeekResponse` deserialized only
  `choices`; prompt/completion tokens were thrown away. `provider.rs` then
  *estimated* output via `text.split_whitespace().count()` and **never tracked
  prompt tokens or cost at all** (`provider.rs:128`, `:185`).
  → Conclusion: **tokens/call and $/run were unobservable before this PR.**

## 2. What actually drives LLM calls

Traced every call site of `provider.generate` / `llm.chat`:

| Path | Trigger | Notes |
|---|---|---|
| `POST /v1/chat/completions` (`openai.rs:214`) | external request | 1 LLM call per request (request-driven, not a loop) |
| `POST /walk` metacog critic (`api.rs:224`) | after each walk | LLM fires when `escalate_to_llm \|\| critic_sessions_since_llm > 10` |
| dream loop (`dream.rs:start_dream_loop`) | always-on | **0 LLM calls** — pure graph Monte-Carlo + DB |

Two findings that correct the brief's premise:

1. **The always-on dream loop makes no LLM calls.** It perturbs edge weights and
   runs parallel graph walks; it never touches `provider`/`llm`. So the "always-on
   loop racking up DeepSeek calls" risk does not exist as described.
2. **DeepSeek was not wired into the running system at all.** `api.rs:90` builds
   the provider from `ProviderConfig::default()`, which has `local_model_path:
   None` and `cloud_models: []`. With that config `provider.generate` does
   nothing and returns empty, so `openai.rs` and `api.rs` both **fell back to
   Ollama**. The DeepSeek engine existed but was unreachable from `main`.
   → Baseline DeepSeek calls in the default runtime: **0**.

## 3. The real cost risks (code-derived)

Even though DeepSeek was dormant, the call-shaping logic that *would* govern cost
once wired had three concrete problems:

### (a) A forced cadence floor, independent of need
`api.rs` escalated to the LLM whenever `critic_sessions_since_llm > 10`. That is a
**guaranteed LLM call every 11 walks** even on `Normal`/`Rest`/healthy sessions
that the algorithmic critic already handled.
- Forced floor: `100 / 11 ≈ 9.09` LLM calls per 100 walks, regardless of cognition.

### (b) Unbounded prompt growth
- Critic prompt: `format_critic_prompt` did `summary.domains_visited.join(", ")` —
  the domain list grows with the graph, dumping an unbounded subgraph into the
  prompt (`metacog.rs:561`).
- Chat prompt: `openai.rs` concatenated user `system` + `stimulus` + walker
  context with **no token cap**.
- → No token budget anywhere on the prompt-building path.

### (c) No dedup, no tiered routing
- Identical cognition states each produced a fresh paid call (no idempotency).
- Routing was complexity → local/cloud only; **no `deepseek-chat` vs
  `deepseek-reasoner` distinction**, so the expensive reasoner tier was never
  gated to genuine difficulty.

## 4. Baseline numbers

| Metric | Baseline value | Source |
|---|---|---|
| DeepSeek calls in default runtime | **0** (unwired → Ollama fallback) | `api.rs:90`, `provider.rs` default |
| Token accounting | **none** (`usage` discarded; output guessed) | `llm.rs`, `provider.rs:185` |
| Cost accounting | **none** | — |
| Forced critic-LLM floor | **≥ 9.09 calls / 100 walks** (`>10` rule) | `api.rs:219` (pre-change) |
| Prompt token ceiling | **unbounded** | `metacog.rs:561`, `openai.rs:191` |
| Dedup / idempotency | **none** | — |
| `deepseek-chat` vs `reasoner` routing | **none** | `provider.rs:142` |
| Dream-loop LLM calls | **0** | `dream.rs` |

These are the "before" columns of `SCORECARD.md`.
