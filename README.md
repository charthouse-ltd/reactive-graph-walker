# RGW — Reactive Graph Walker

RGW traverses Julian's PostgreSQL memory graph with parallel Rust walkers, maintains a persistent self-model, and can use an LLM to express the result. One process owns one self-model. Julian owns memory documents, embeddings and database migrations.

## Running

Requires Rust 1.88+, PostgreSQL with pgvector, and Julian's schema. The embedding model may download on first startup.

```sh
export DATABASE_URL='postgresql://test-user:test-password@127.0.0.1:5432/test-db'
export RGW_API_KEY='test-admin-key' # placeholder; use a strong secret outside source control
cargo build --locked
cargo run --locked --bin rgw
```

The default bind address is `127.0.0.1:11435`. Send credentials in `X-RGW-Key`. Only `/health` and `/ready` are public. CORS is closed.

Startup restores the saved self-model before starting writers. A database or snapshot decoding error stops startup; it never substitutes a fresh model for an unreadable saved row. An absent row permits a fresh model. The configured cognitive mode overrides the saved mode at startup. Self-state is saved periodically and on graceful shutdown.

Production defaults are **compliant cognition**, **dream loop disabled**, **critic application disabled**, **motor shadow**, and **tension observe**. These prevent autonomous background graph writes. Explicit requests such as walks, edge creation and pruning can still mutate graph dynamics; compliant mode is not a database read-only role. Walk edge deltas are flushed at the end of that walk.

## HTTP API and scopes

The existing `RGW_API_KEY` remains a full-access **admin** key. It also approves motor proposals unless `RGW_APPROVE_KEY` is set. Optional read, ingest and approve keys grant only their named scope. Missing or unknown credentials return 401; a recognized key with the wrong scope returns 403. Empty optional keys are disabled. Admin credentials remain mandatory at startup even when scoped keys are configured.

**Approval policy:** when `RGW_APPROVE_KEY` is set, approval/rejection accepts **only** that key; the admin key gets 403, so approval is separated from everything else. When it is unset, the admin key approves as before (single-key deployments keep working). Read and ingest keys can never approve.

| Method | Path | Scope | Purpose |
|---|---|---|---|
| GET | `/health` | public | Liveness, always 200; truthful `db_connected` |
| GET | `/ready` | public | 200 when DB/schema are ready, otherwise 503; includes `schema_version` |
| GET | `/self` | read | Full self-model, including audience state |
| GET | `/stats` | read | Graph statistics |
| GET | `/metrics` | read | Runtime and LLM metrics |
| GET | `/recall` | read | Episodes by date/place |
| GET | `/diverger` | read | Reactor and tension statistics |
| GET | `/learning` | read | Seeding statistics |
| GET | `/selection/stats` | read | Selection/trial state |
| GET | `/metacog/config` | read | Metacognitive configuration |
| GET | `/tools` | read | Tool catalogue |
| GET | `/music`, `/music/midi`, `/music/prompt` | read | Mood-derived music parameters, MIDI or prompt |
| GET | `/v1/models` | read | OpenAI-compatible model listing |
| GET | `/motor/proposals` | read | Pending/decided proposals, shadow records and recent outcomes |
| POST | `/ingest/signal` | ingest | External cognitive signal |
| POST | `/diverger/notify` | ingest | Edge-change notification |
| POST | `/tom/record` | ingest | Audience interaction |
| POST | `/diverger/outcome`, `/outcomes` | ingest | Motor outcome callback; 202 |
| POST | `/motor/proposals/{id}/approve` | approve | Send held proposal; requires `{"confirm":"yes"}` |
| POST | `/motor/proposals/{id}/reject` | approve | Reject held proposal; optional note |
| POST | `/walk` | admin | Walk and optional critic |
| POST | `/v1/chat/completions` | admin | Walk plus LLM expression; non-streaming |
| GET | `/benchmark` | admin | Runs multiple walks; intentionally not read-scoped |
| POST | `/edge`, `/prune` | admin | Create edges or decay/delete edges |
| POST | `/self/mode`, `/self/save` | admin | Change mode or save self-model |
| POST | `/metacog/config` | admin | Update critic configuration |
| POST | `/selection/stage`, `/selection/rollback` | admin | Selection controls |
| POST | `/tools/execute` | admin | Execute a tool; node creation is disabled |
| POST | `/dream` | admin | Explicit dream cycle; compliant mode suppresses activity |
| POST | `/music/generate` | admin | Requires separately installed Python/MusicGen |
| POST | `/speak` | admin | Currently returns 503 |

Only the existing OpenAI routes have a `/v1` prefix; general `/v1` aliases are not implemented.

## Motor channel and outcomes

Every command receives a UUID `correlation_id`, exposed on its proposal and inside the command sent to `/api/admin/rgw/execute`.

| Motor mode | Behavior |
|---|---|
| `off` | Drop actions |
| `shadow` (default) | Retain/log intent without sending |
| `propose` | Queue up to 100 proposals, expiring after 24 hours; explicit approval sends |
| `live` | Send immediately |

Outbound authentication is optional and separate from inbound API keys:

- `RGW_JULIAN_BEARER` sends `Authorization: Bearer …`.
- `RGW_JULIAN_KEY` sends `X-RGW-Key: …`.
- Both may be configured; neither is sent by default. Julian must support the chosen header.

Julian's existing callback and the shorter `/outcomes` route accept:

```json
{
  "correlation_id": "f0f5b293-7f08-4d2b-8461-5fd76c8ae371",
  "outcome": "completed",
  "success": true,
  "lesson": "Use the result on the next attempt."
}
```

`success` and `lesson` are optional. The UUID must be non-nil; outcome must contain 1–500 characters and lesson at most 500. Accepted callbacks return 202. They are retained in a bounded, process-local ring (100 entries, latest callback per UUID), visible in `/motor/proposals`. They are **not durable** and do not yet update episodic learning: the existing episodic buffer has no command correlation mapping. Unknown UUIDs are accepted for compatibility with delayed callbacks.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | required | Explicit PostgreSQL connection URL |
| `RGW_API_KEY` | required | Full-access admin key; blank/unset fails closed |
| `RGW_READ_KEY` | unset | GET read routes |
| `RGW_INGEST_KEY` | unset | Signal, event, audience and outcome writes |
| `RGW_APPROVE_KEY` | unset | Motor approve/reject |
| `RGW_COGNITIVE_MODE` | `compliant` | `compliant` or `autonomous`; invalid values stop startup |
| `RGW_DREAM_ENABLED` | `false` | Start background dream loop; also requires autonomous mode to act |
| `RGW_CRITIC_APPLY` | `false` | Allow critic mutations; autonomous selection-live/rule-trials-live stages also enable application |
| `RGW_CRITIC_LLM_INTERVAL` | `50` | Periodic LLM critic calibration interval |
| `RGW_MOTOR_MODE` | `shadow` | `off`, `shadow`, `propose`, `live` |
| `JULIAN_URL` | `http://localhost:8000` | Diverger motor target |
| `RGW_MOTOR_URL` | unset | Motor tool target |
| `RGW_JULIAN_BEARER` | unset | Outgoing Bearer credential, for both motor targets |
| `RGW_JULIAN_KEY` | unset | Outgoing X-RGW-Key credential, for both motor targets |
| `RGW_TENSION_MODE` | `observe` | `off`, `observe`, `brakes`, `live` |
| `RGW_SEED_MODE` | `shadow` | `fixed`, `shadow`, `query` |
| `RGW_ENABLE_AUTO_PRUNE` | off | Opt-in destructive node pruning in the dream loop |
| `RGW_EVAL` | off | `1` enables per-request evaluation arms |
| `RGW_DETERMINISTIC` | off | `1` enables deterministic walker execution |
| `RGW_NARRATIVE_EXCLUDE_SOURCE_TYPES` | built-in private-source exclusions | Comma-separated narrative source filters |
| `WALKER_HOST` / `WALKER_PORT` | `127.0.0.1` / `11435` | Bind address |
| `WALKER_THREADS` | `0` | Auto-detect CPU threads |
| `OLLAMA_URL` / `EXPRESSION_MODEL` | `http://localhost:11434` / `qwen3:14b` | Local expression fallback |
| `DEEPSEEK_API_KEY` | unset | Cloud expression credential |
| `DEEPSEEK_MODEL` / `DEEPSEEK_REASONER_MODEL` | `deepseek-chat` / `deepseek-reasoner` | Cloud models |
| `RGW_MAX_CALLS_PER_MIN` | `60` | Process-local successful-call budget |
| `RGW_MAX_DAILY_USD` | `10` | Process-local estimated spend budget |
| `DEEPSEEK_MAX_PROMPT_TOKENS` | `2000` | Approximate prompt budget |
| `RGW_DEDUP_TTL_SECS` | `60` | Completion cache TTL |
| `RGW_REASONER_COMPLEXITY` | `0.85` | Reasoner routing threshold |
| `MUSICGEN_PYTHON` | user MusicGen virtualenv | Music generation Python executable |
| `RUST_LOG` | `info` | Logging filter |

Dream/critic booleans accept `true`/`false` or `1`/`0`; invalid values stop startup. Spend limits are estimates within one process, not distributed hard quotas.

## Schema contract and tests

Julian's Alembic migrations remain authoritative. RGW does not migrate production databases. `RGW_SCHEMA_CONTRACT = "julian-rgw-1"` identifies the subset checked by readiness, not an entire Julian migration revision.

Required tables are `memory_vectors` (including `document` and `embedding vector(768)`), `memory_edges`, `runtime_settings` (`value TEXT`, `description`, `category`), and `rgw_episodes`. `tests/schema.sql` mirrors the relevant Julian baseline and episode migration, including NOT NULL columns and cascading edge foreign keys. ORM-only defaults are deliberately absent. Optional knowledge grounding reads `knowledge_vectors`.

The `memory_store` tool and legacy `db::create_memory_node` fail cleanly: node creation must go through Julian's content/embedding pipeline.

For a disposable local PostgreSQL instance only:

```sh
export DATABASE_URL='postgresql://rgw:rgwtest@127.0.0.1:55439/rgw_test'
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f tests/schema.sql
cargo build --locked
cargo test --locked --lib --bins
cargo test --locked --tests
```

The PR-only CI job uses pgvector/PostgreSQL 16 and the same schema. Synthetic graph tests run offline. Four tests are explicitly ignored because they need downloaded embedding weights: `test_digest_philosophy`, `targets_resolve_from_domain_and_embedding`, `observe_mode_logs_would_fire_and_emits_nothing`, and `a_fresh_embedding_bypasses_the_cache`. Run those individually with `--ignored` when the model is available and a disposable database is explicitly configured.

## Speech: not implemented (TTS macOS-only stub, no STT)

The speech module has a macOS `say` adapter, but the API uses disabled configuration and returns 503. Other speech engines are unimplemented. There is no speech-to-text route.

## License

MIT — see [LICENSE](LICENSE).
