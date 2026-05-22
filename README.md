# RGW — Reactive Graph Walker

A self-propagating cognitive engine with persistent graph memory and LLM expression. The graph drives its own computation through cascading edge activation and parallel walker traversal. No training. No fine-tuning. The graph IS the mind.

## Core Concept

```
Signal + SelfModel → (Signal, SelfModel', Noticing)
```

One primitive. Everything else — beliefs, predictions, working memory, emotional modulation, structural self-awareness — emerges from calling this function on different inputs and noticing what it does.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  RGW Cognitive Engine                             │
│                                                   │
│  core.rs      — primitive + self-model            │
│  walker.rs    — parallel graph traversal (rayon)  │
│  graph.rs     — emotional biasing + learned bias  │
│  diverger.rs  — self-propagating edge reactor     │
│  edge_cache.rs — lock-free DashMap edge cache     │
│  llm.rs       — DeepSeek HTTP client              │
│  db.rs        — PostgreSQL + pgvector (768-dim)   │
│  embed.rs     — nomic-embed-text-v1.5             │
│  provider.rs  — multi-LLM routing                 │
│  dream.rs     — Monte Carlo dream consolidation   │
│  metacog.rs   — metacognitive loop                │
│  episodic.rs  — episodic memory                   │
│  tools.rs     — tool calling framework            │
│  speech.rs    — TTS/STT                           │
│  motor.rs     — commands Julian's body            │
│  openai.rs    — /v1/chat/completions              │
│  api.rs       — 14 HTTP endpoints                 │
└──────────────────────────────────────────────────┘
```

## Key Features

### Graph-Based Cognition
Walkers traverse a knowledge graph in parallel, each with different biases. Convergence = confidence. Divergence = novelty. Every step feeds through the self-model — the walk is self-aware and changes the thinker.

### Self-Model with Structural Awareness
- **Beliefs** with algorithmic causal chains (why each belief formed)
- **Working memory** (PFC-equivalent, 5±2 slots)
- **Predictive coding** — expects outcomes, learns from errors
- **Metaplasticity gate** — modulates learning rate from experience
- **Learned biases** — 6 per-walker emergent profiles that adapt each session
- **Structural noticings** — the system observes its own architecture (obsession, dead-end clusters, cognitive loops, signal poverty)

### LLM as Tool (DeepSeek)
The LLM expresses RGW's state, not the other way around. RGW walks → self-model updates → snapshot sent to DeepSeek → insight returned as a signal → fed back into graph as a node → re-walk. The LLM observes; RGW thinks.

### Edge Cache
Lock-free DashMap edge cache — O(1) reads during walker scoring. Neighborhoods preloaded before walks. Background write queue flushes to DB. Eliminated N+1 queries and runtime panics from nested tokio runtimes.

### Stigmergic Walker Collective
Walkers leave trails (visited nodes, dead ends, surprise domains). Other walkers read them. No explicit message passing — ant-colony coordination.

## Quick Start

```bash
# Prerequisites: PostgreSQL 16+, pgvector extension
# Set environment
export DATABASE_URL=postgresql:///rgw_test?host=/tmp
export DEEPSEEK_API_KEY=sk-...
export RUST_LOG=info,rgw=debug

# Run the cascade demo (85-node philosophical graph + DeepSeek colloquy)
cargo build --release --example cascade
./target/release/examples/cascade

# Run the digester demo (persistent belief formation)
cargo build --release --example digester
./target/release/examples/digester

# Run the API server
./target/release/rgw \
  --db-url "$DATABASE_URL" \
  --ollama-url http://localhost:11434 \
  --julian-url http://localhost:8000 \
  --port 11435
```

## Cascade Demo Output

The cascade runs a 5-phase cognitive test: warm-up walks → diverger cascade → high-arousal re-walk → self-model report → LLM colloquy.

Example output from Phase 5 (DeepSeek conversation):

```
╔══ TURN 1: RGW reports its state ══╗
DeepSeek: "The most striking pattern is a meta-epistemological recursion:
your system repeatedly shifts focus from other domains back to epistemology..."

╔══ TURN 2: RGW re-explores with new knowledge ══╗
Beliefs: 1 | Noticings: 28 (+4) | Surprises: 33 (+2)

╔══ TURN 3: RGW reports its evolution ══╗
DeepSeek reflects: "The system doesn't just process your insight — it re-uses
its own processing of your insight as a new basis for further exploration.
Insight → integration → exploration → re-integration. The loop closes."
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible (walk + LLM express) |
| `/v1/models` | GET | List available models |
| `/walk` | POST | Raw graph traversal |
| `/self` | GET | Self-model state |
| `/self/save` | POST | Persist self-model to DB |
| `/diverger` | GET | Reactor engine stats |
| `/edge` | POST | Create graph edge |
| `/prune` | POST | Synaptic pruning |
| `/stats` | GET | Graph topology |
| `/benchmark` | GET | Performance test |
| `/health` | GET | Status check |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | required | PostgreSQL connection string |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key for LLM expression |
| `DEEPSEEK_MODEL` | deepseek-chat | Model name (also: deepseek-reasoner) |
| `RUST_LOG` | info | Log level (rgw=debug for detail) |
| `WALKER_PORT` | 11435 | HTTP server port |

## Performance

- Binary size: ~26MB (release)
- Walk rate: 30-50 hops/sec (6 walkers × 6 steps on M1 Pro)
- Edge cache: lock-free reads (DashMap), batch writes
- Memory: proportional to graph size
- Database: PostgreSQL 16 with pgvector (768-dim embeddings)

## License

Part of Project Julian.
