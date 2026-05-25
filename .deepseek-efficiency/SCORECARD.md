# DeepSeek Efficiency — SCORECARD

Branch `deepseek-efficiency`. Baseline derivation in [`baseline.md`](./baseline.md);
quality method in [`scenarios.md`](./scenarios.md).

> **Honesty note.** A live paid run was not possible here (no DB/key, real cost).
> "Before" token/cost figures were **unmeasurable by construction** — the code
> discarded DeepSeek's `usage` object and DeepSeek wasn't even wired into the
> running server. The headline of this PR is therefore twofold: (1) make cost
> **observable** (real `usage` capture + `/metrics`), and (2) make it **bounded**
> (budget, dedup, cadence, tiered routing) before the wire is turned on.

## Before → After

| Metric | Before | After | How |
|---|---|---|---|
| **Tokens/call observable?** | ❌ `usage` discarded; output guessed via `split_whitespace` | ✅ real `prompt`/`completion` tokens captured from API; char/4 fallback | `llm.rs` `chat_with_model`, `provider.rs` cloud fns, `metrics.rs` |
| **Cost/run observable?** | ❌ none | ✅ live `est_cost_usd` at DeepSeek published rates | `metrics::estimate_cost_usd`, `GET /metrics` |
| **calls/min observable?** | ❌ none | ✅ rolling 60s + lifetime | `metrics::snapshot` |
| **Forced critic-LLM floor** | **≥ 9.09 / 100 walks** (`>10`, fires on healthy sessions) | **≤ 2.0 / 100 walks** (`>50`, skipped while resting) → **−78 %** | `api.rs` cadence rewrite + `RGW_CRITIC_LLM_INTERVAL` |
| **Prompt token ceiling** | unbounded | **2000 tok/segment** (+ top-k(12) domains) | `budget::budget_prompt`, `metacog.rs` top-k |
| **Duplicate-call rate** | n/a (every call paid) | **dedup cache**, TTL 60s; `dup_rate` reported | `budget` dedup + `metrics::record_dedup_hit` |
| **% reasoner usage** | none (no tiering) | **gated**: reasoner only when metacog flags difficulty or complexity ≥ 0.85; periodic calibration stays on chat → ~0 % on healthy runs | `budget::wants_reasoner`, `provider.generate`, `api.rs` |
| **Embedding recompute** | every ingest re-embeds | identical text reused from cache | `embed.rs` `EMBED_CACHE` |
| **DeepSeek wired?** | ❌ no-op default config → Ollama | ✅ auto-wired from `DEEPSEEK_API_KEY` | `provider::Provider::new` |

### The cadence win, worked

The old `critic_sessions_since_llm > 10` guaranteed a paid critic call every 11
walks **even when the algorithmic critic already returned `Normal`/`Rest`**. The
new path fires the LLM only on (a) a genuine-difficulty diagnosis, or (b) a
periodic calibration at the configurable interval (default 50), and never during
`Rest`. Forced (non-diagnostic) floor: `100/11 → 100/50`, a **78 % reduction**,
on top of which dedup removes identical-state repeats.

## Cognition-quality check (fixed scenario set)

All five scenarios in [`scenarios.md`](./scenarios.md) pass as unit tests — the
levers preserve the JSON contract, the diagnosis, domain breadth, and escalate
genuinely-hard sessions to the reasoner. Efficiency does **not** degrade the
signal the critic receives.

```
cargo test --bin rgw   # 38 passed (29 pre-existing + 9 new)
```

`tests/digester.rs` is a pre-existing Postgres integration test; it requires a
live `rgw_test` DB and is unrelated to this change.

## Tuning knobs (env, no rebuild)

| Var | Default | Effect |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | wires DeepSeek on when set |
| `DEEPSEEK_MODEL` / `DEEPSEEK_REASONER_MODEL` | `deepseek-chat` / `deepseek-reasoner` | tier model names |
| `DEEPSEEK_MAX_PROMPT_TOKENS` | `2000` | per-segment prompt budget |
| `RGW_REASONER_COMPLEXITY` | `0.85` | complexity at which reasoner engages |
| `RGW_CRITIC_LLM_INTERVAL` | `50` | walks between periodic critic calibrations |
| `RGW_DEDUP_TTL_SECS` | `60` | dedup window; `0` disables |

Set the last three to `100000 / 10 / 0` to reproduce pre-PR behaviour for an
apples-to-apples live measurement.

## Safety

No changes to action-emitting modules (`motor.rs`, `speech.rs`, `music.rs`) or DB
writes (`db.rs`). No API keys logged (`metrics.rs` counts tokens only). Delivered
on a branch with a PR; unit suite green.
