# Fixed cognition-quality scenario set

Efficiency must not lobotomize the walker. Because a live DeepSeek run isn't
available here, the quality check is **structural**: on a fixed set of cognitive
states we assert that the cost-bounding levers preserve the essential signal
(the JSON contract, the diagnosis, the breadth of domains) and that routing still
escalates genuinely-hard sessions. Each scenario is backed by a unit test that
runs in `cargo test --bin rgw`.

| # | Scenario | Cognitive expectation | Efficiency behaviour | Test (guard) |
|---|---|---|---|---|
| S1 | **Normal session** (healthy walk, `escalate_to_llm=false`) | algorithmic critic suffices; no LLM needed | stays on cheap `deepseek-chat`; no reasoner | `metacog::tests::efficiency_routing_escalates_only_hard_sessions` |
| S2 | **Stuck / BreakLoop** (`consecutive_repetitions=4`) | genuine difficulty; LLM should weigh in | escalates to `deepseek-reasoner` | `metacog::tests::efficiency_routing_escalates_only_hard_sessions` |
| S3 | **Pathological context growth** (100 domains visited) | critic must still see diagnosis + JSON contract + domain breadth | top-k(12) cap + 2000-token wire budget; `Output ONLY JSON`, `Diagnosis:` and the unique-count survive | `metacog::tests::efficiency_budget_preserves_critic_signal` |
| S4 | **Duplicate cognition state** (identical prompt within TTL) | same input ⇒ same output | served from dedup cache; paid call skipped; identical text returned | `budget::tests::dedup_roundtrips_within_ttl` |
| S5 | **Cost asymmetry** | reasoner is the expensive path | priced strictly higher than chat per token | `metrics::tests::reasoner_is_detected_and_priced_higher` |

## How to run a live quality + cost check

With real credentials, the same scenarios can be driven end-to-end and the
numbers read from the new endpoint:

```bash
export DATABASE_URL=postgres://…
export DEEPSEEK_API_KEY=sk-…
cargo run --release &                      # DeepSeek now wired automatically
# drive a fixed scenario set, e.g. N walks + a few /v1/chat/completions calls
curl -s localhost:11435/metrics | jq       # tokens/call, calls/min, $cost, %reasoner, dup_rate
```

`GET /metrics` returns the live scorecard (`MetricsSnapshot`): `calls`,
`tokens_per_call`, `calls_per_min`, `est_cost_usd`, `reasoner_pct`, `dup_rate`.
Compare against the same run with `RGW_CRITIC_LLM_INTERVAL=10`,
`DEEPSEEK_MAX_PROMPT_TOKENS=100000`, `RGW_DEDUP_TTL_SECS=0` to reproduce
pre-PR behaviour and quantify the delta on real traffic.
