# reactive-graph-walker (rgw) — DeepSeek Efficiency (for Claude Code)

Run in **Claude Code** inside `reactive-graph-walker` (Rust; use CodeGraph — it indexes
Rust). This engine is a **self-propagating cognitive loop**: the walker produces
cognition in memory, builds an enriched prompt, and `src/llm.rs` (a direct DeepSeek
HTTP client — `deepseek-chat` vs `deepseek-reasoner`) generates from it. The cost risk
here is different from a request/response app: an always-on walk/dream loop can rack up
calls and oversized prompts. Goal: bound call volume, prompt size, and reasoner usage
without dulling the cognition. Quantify to `.deepseek-efficiency/`.

## Hard safety rules
- Action-emitting modules (`src/motor.rs`, `src/speech.rs`, `src/music.rs`) and DB
  writes (`src/db.rs`) change only on a branch + PR. `cargo test` must pass. No API
  keys in logs.

### PHASE 0 — Cost baseline
1. Read `src/llm.rs` and `src/provider.rs` / `src/openai.rs`: confirm the DeepSeek
   models used, and capture the `usage` object DeepSeek returns (prompt/completion
   tokens) — log it per call if not already.
2. Identify the loop that drives calls: `src/walker.rs`, `src/core.rs`, `src/dream.rs`,
   `src/diverger.rs`, `src/metacog.rs`. Measure, over a fixed run (e.g. 10 min or N
   ticks): calls, tokens in/out, % using `deepseek-reasoner`, est. cost, calls/min.
3. Save `.deepseek-efficiency/baseline.md`.

### PHASE 1 — Where cost accumulates
Two suspects: (a) the **enriched prompt** the walker assembles from graph context can
grow unbounded; (b) the **loop cadence** (dream/diverger) may call the LLM more often
than needed. Quantify both: avg prompt tokens, and call frequency by trigger.

### PHASE 2 — Bound the loop (waste removal)
1. **Token-budget the enriched prompt** built before `llm.rs`: cap graph-context size
   to a top-k / token budget (lean on `src/edge_cache.rs` and `src/embed.rs` to select
   the most relevant nodes instead of dumping subgraphs). Metric: avg prompt tokens.
2. **Rate-limit / coalesce** dream/walk ticks so near-duplicate cognition states don't
   each trigger a call; add a dedup/idempotency check on the prompt hash. Metric:
   calls/min and duplicate-call rate.
3. **Cache**: ensure `edge_cache.rs` results and embeddings are reused, not recomputed.

### PHASE 3 — Model + token efficiency
1. **Route `deepseek-chat` vs `deepseek-reasoner`** by task: only escalate to reasoner
   when metacog/diverger signals genuine difficulty. `reasoner` is the expensive path.
   Metric: % reasoner calls before vs after.
2. **Trim system/enrichment templates** in the prompt-building path (verify cognition
   quality on a fixed scenario set so you don't lobotomize the walker).
3. **Stream + early-stop** where full completions aren't needed.

### DELIVERABLE
`.deepseek-efficiency/SCORECARD.md`: tokens/call + calls/min + cost/run before vs
after, % reasoner usage delta, with a qualitative cognition-quality check on a fixed
scenario set (efficiency must not degrade output). Branch + PR; `cargo test` green.
