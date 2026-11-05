//! HTTP API — serves the same endpoints as the Python walker_service.
//! Drop-in replacement: walker_client.py doesn't know it's Rust.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tower_http::cors::CorsLayer;

use crate::db;
use crate::core::{CognitiveMode, MetacogPhase, SelfModel};
use crate::diverger::{Diverger, EdgeChange};
use crate::graph::*;
use crate::openai;
use crate::walker;

/// Shared application state — one entity, one mind
pub struct AppState {
    pub pool: PgPool,
    pub diverger: Diverger,
    pub self_model: std::sync::Arc<tokio::sync::Mutex<SelfModel>>,
    pub ollama_url: String,
    pub expression_model: String,
    pub julian_url: String,
    /// Intelligent LLM router (local + cloud). If None, falls back to Ollama.
    pub provider: Option<crate::provider::Provider>,
    /// The single gate for outbound motor actions (RGW_MOTOR_MODE).
    pub motor_gate: Arc<crate::motor::MotorGate>,
}

/// Walk request (matches Python WalkRequest)
#[derive(Deserialize)]
pub struct WalkRequest {
    #[serde(default)]
    pub stimulus: String,
    #[serde(default)]
    pub emotional_state: EmotionalState,
    #[serde(default = "default_walkers")]
    pub n_walkers: usize,
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Per-request cognitive mode override.
    /// May lower autonomy for this walk, never exceed the configured global mode.
    #[serde(default)]
    pub rgw_mode: Option<String>,
    /// Makes this walk deterministic with the given base seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Evaluation arm (`fixed`, `query`, `query:linear`, `vector`, `ppr`, `drop:<bias>`).
    /// Accepted only when the server runs with `RGW_EVAL=1`.
    #[serde(default)]
    pub arm: Option<String>,
}

fn default_walkers() -> usize { 4 }
fn default_steps() -> usize { 5 }

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub cpu_cores: usize,
    pub db_connected: bool,
    pub version: String,
    pub runtime: String,
    pub motor_mode: String,
}

/// Start the HTTP server with the Diverger engine
pub async fn serve(
    pool: PgPool,
    addr: &str,
    ollama_url: &str,
    expression_model: &str,
    julian_url: &str,
) -> anyhow::Result<()> {
    // Create the self-model FIRST — the continuous state of self-awareness
    let self_model = std::sync::Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
    tracing::info!("[rgw] Self-model initialized. Consciousness online.");

    // Motor gate FIRST: every outbound action goes through it. Default shadow.
    let motor_mode = crate::motor::MotorMode::from_env_value(std::env::var("RGW_MOTOR_MODE").ok().as_deref());
    let motor_gate = crate::motor::install_gate(motor_mode);
    // Tension-driven background activity (spec: docs/superpowers/specs/2026-11-04-…).
    // Default observe: compute + log, change nothing. See /diverger "tension".
    let tension_engine = crate::tension::install(crate::tension::TensionMode::from_env_value(
        std::env::var("RGW_TENSION_MODE").ok().as_deref(),
    ));
    tracing::info!("[motor] Motor gate mode: {}", motor_gate.mode().as_str());

    // Create the Diverger with shared self-model + Julian URL for motor commands
    let diverger = Diverger::new(pool.clone(), self_model.clone(), julian_url);
    if let Some(e) = &tension_engine {
        e.attach_inbox(diverger.stimulus_sender());
        e.attach_energy_view(diverger.energy_view());
    }

    // ── Start concurrent dream loop (always-on, motor-disconnected) ──
    crate::dream::start_dream_loop(pool.clone(), self_model.clone());

    // Seed initial energy from high-importance nodes
    let seeds = db::seed_nodes(&pool, 50).await.unwrap_or_default();
    diverger.seed_energy(seeds, 0.3).await;

    // Try to initialize the intelligent provider router
    let provider = {
        let config = crate::provider::ProviderConfig::default();
        match crate::provider::Provider::new(config) {
            Ok(p) => {
                tracing::info!("[rgw] Provider router initialized");
                Some(p)
            }
            Err(e) => {
                tracing::info!("[rgw] Provider router unavailable ({}), using Ollama fallback", e);
                None
            }
        }
    };

    let state = Arc::new(AppState {
        pool,
        diverger,
        self_model,
        ollama_url: ollama_url.to_string(),
        expression_model: expression_model.to_string(),
        julian_url: julian_url.to_string(),
        provider,
        motor_gate: motor_gate.clone(),
    });

    // RGW_API_KEY is validated in main() before startup (refuse to start without
    // it); require_auth additionally rejects every request if it is missing.

    // ── Public endpoint (no auth) — health only ──
    // Strict posture: ONLY /health is open (liveness probes work without a key).
    // Everything else — including read-only introspection — always requires X-RGW-Key.
    let public = Router::new()
        .route("/health", get(health));

    // ── Protected endpoints (always require X-RGW-Key) ──
    // Everything except /health: read-only introspection AND mutations / actions
    // / LLM-spend. Fail-closed: no key configured → 401. See audit C1/S3.
    let protected = Router::new()
        // read-only introspection
        .route("/stats", get(stats))
        .route("/recall", get(recall))
        .route("/metrics", get(metrics_endpoint))
        .route("/diverger", get(diverger_stats))
        .route("/learning", get(learning_stats))
        .route("/self", get(self_state))
        .route("/metacog/config", get(get_metacog_config).post(set_metacog_config))
        .route("/selection/stage", post(set_selection_stage))
        .route("/selection/stats", get(selection_stats))
        .route("/selection/rollback", post(selection_rollback))
        .route("/tools", get(list_tools))
        .route("/music", get(music_endpoint))
        .route("/music/midi", get(music_midi_endpoint))
        .route("/music/prompt", get(music_prompt_endpoint))
        .route("/v1/models", get(openai::list_models))
        .route("/benchmark", get(benchmark))
        // mutations / actions / cost
        .route("/walk", post(walk))
        .route("/prune", post(prune))
        .route("/edge", post(create_edge_endpoint))
        .route("/diverger/notify", post(diverger_notify))
        .route("/self/mode", post(set_mode_endpoint))
        .route("/self/save", post(save_self_model_endpoint))
        .route("/tools/execute", post(execute_tool_endpoint))
        .route("/speak", post(speak_endpoint))
        .route("/dream", post(dream_endpoint))
        .route("/music/generate", post(music_generate_endpoint))
        // Signal ingest — external events feed into the cognitive loop
        .route("/ingest/signal", post(ingest_signal_endpoint))
        .route("/tom/record", post(tom_record_endpoint))
        // OpenAI-compatible chat (triggers a walk + paid LLM expression)
        .route("/v1/chat/completions", post(openai::chat_completions))
        // Motor gate: inspect / approve / reject held actions (human only)
        .merge(motor_routes(motor_gate))
        .route_layer(middleware::from_fn(require_auth));

    let app = public
        .merge(protected)
        .with_state(state.clone())
        // Closed CORS. Callers are server-side and do not send Origin.
        // A permissive layer echoed any Origin.
        .layer(CorsLayer::new());

    // Auto-save self-model every 60 seconds (consciousness persistence)
    {
        let save_state = state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let sm = save_state.self_model.lock().await.clone();
                if sm.total_signals_processed > 0 {
                    match db::save_self_model(&save_state.pool, &sm).await {
                        Ok(_) => tracing::debug!("[rgw] Self-model saved ({} signals)", sm.total_signals_processed),
                        Err(e) => tracing::debug!("[rgw] Self-model save failed: {}", e),
                    }
                }
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Final persistence: without this, every container recreate threw away up
    // to 60 s of self-model evolution (the periodic save is the only other
    // writer). Pending edge deltas live in per-walk caches that flush at the
    // end of each walk, so the self-model is the state that needs saving here.
    let sm = state.self_model.lock().await.clone();
    match db::save_self_model(&state.pool, &sm).await {
        Ok(_) => tracing::info!("[rgw] shutdown: self-model saved ({} signals)", sm.total_signals_processed),
        Err(e) => tracing::error!("[rgw] shutdown: self-model save FAILED: {}", e),
    }
    Ok(())
}

/// Resolve on SIGINT (Ctrl-C) or SIGTERM (what `docker stop` / compose recreate
/// send). axum then stops accepting, drains in-flight requests, and returns so
/// `serve` can persist state before the process exits.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!("[rgw] cannot install SIGTERM handler: {}", e);
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("[rgw] SIGINT received, shutting down"),
        _ = terminate => tracing::info!("[rgw] SIGTERM received, shutting down"),
    }
}

/// Minimum recommended RGW_API_KEY length (shorter keys are accepted with a WARN).
const MIN_STRONG_KEY_LEN: usize = 32;

/// Validate the RGW_API_KEY value at startup. Unset, empty or whitespace → Err.
/// Never echoes the value. Short keys are accepted with a WARN (existing
/// deployments' key lengths are unknown, so they are not rejected).
pub fn validate_api_key(v: Option<&str>) -> Result<(), String> {
    match v.map(str::trim) {
        None | Some("") => Err(
            "RGW_API_KEY is not set (or is empty). RGW refuses to start without it. \
             Generate one with `openssl rand -hex 32`, set RGW_API_KEY, and send it \
             as the X-RGW-Key header from callers."
                .to_string(),
        ),
        Some(k) => {
            if k.len() < MIN_STRONG_KEY_LEN {
                tracing::warn!(
                    "[rgw] RGW_API_KEY is shorter than {} characters; consider `openssl rand -hex 32`",
                    MIN_STRONG_KEY_LEN
                );
            }
            Ok(())
        }
    }
}

/// Auth middleware for every route except /health.
/// Convention shared with the deployed backends: the secret travels as the
/// `X-RGW-Key` header and matches the `RGW_API_KEY` env var. Fail-closed: if
/// `RGW_API_KEY` is unset or empty here, every request gets 401 (main() also
/// refuses to start in that case; this is the second line of defence).
async fn require_auth(req: Request, next: Next) -> Result<Response, StatusCode> {
    let Some(expected) = std::env::var("RGW_API_KEY").ok().filter(|s| !s.trim().is_empty()) else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    let presented = req
        .headers()
        .get("X-RGW-Key")
        .and_then(|v| v.to_str().ok());

    match presented {
        Some(key) if constant_time_eq(key.as_bytes(), expected.trim().as_bytes()) => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Length-checked constant-time byte comparison (avoids token-timing leaks).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// GET /health
async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let db_ok = sqlx::query("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .is_ok();

    Json(HealthResponse {
        status: if db_ok { "ok".into() } else { "db_disconnected".into() },
        cpu_cores: rayon::current_num_threads(),
        db_connected: db_ok,
        version: env!("CARGO_PKG_VERSION").into(),
        runtime: "rust".into(),
        motor_mode: state.motor_gate.mode().as_str().into(),
    })
}

/// Motor gate routes. Mounted inside the protected router (behind require_auth).
fn motor_routes<S: Clone + Send + Sync + 'static>(gate: Arc<crate::motor::MotorGate>) -> Router<S> {
    Router::new()
        .route("/motor/proposals", get(motor_proposals))
        .route("/motor/proposals/{id}/approve", post(motor_approve))
        .route("/motor/proposals/{id}/reject", post(motor_reject))
        .with_state(gate)
}

fn gate_error_response(e: crate::motor::GateError) -> (StatusCode, Json<serde_json::Value>) {
    use crate::motor::GateError;
    let (code, msg) = match e {
        GateError::NotFound => (StatusCode::NOT_FOUND, "proposal not found"),
        GateError::AlreadyDecided => (StatusCode::CONFLICT, "proposal already decided"),
        GateError::Expired => (StatusCode::GONE, "proposal expired"),
    };
    (code, Json(serde_json::json!({"error": msg})))
}

/// GET /motor/proposals
async fn motor_proposals(State(gate): State<Arc<crate::motor::MotorGate>>) -> Json<serde_json::Value> {
    Json(gate.snapshot())
}

/// POST /motor/proposals/{id}/approve — human only. Body must be exactly {"confirm":"yes"}.
async fn motor_approve(
    State(gate): State<Arc<crate::motor::MotorGate>>,
    Path(id): Path<u64>,
    body: axum::body::Bytes,
) -> (StatusCode, Json<serde_json::Value>) {
    let confirmed = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("confirm").and_then(|c| c.as_str()).map(|c| c == "yes"))
        .unwrap_or(false);
    if !confirmed {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "approve requires the JSON body {\"confirm\": \"yes\"}; nothing was sent"})),
        );
    }
    match gate.approve(id).await {
        Ok(p) => {
            tracing::warn!(
                "[motor] APPROVED proposal #{} action={} domain={} result={:?}",
                p.id, p.command.action, p.command.domain, p.send_result
            );
            (StatusCode::OK, Json(serde_json::to_value(&p).unwrap_or_default()))
        }
        Err(e) => gate_error_response(e),
    }
}

#[derive(Deserialize, Default)]
struct RejectBody {
    #[serde(default)]
    note: Option<String>,
}

/// POST /motor/proposals/{id}/reject — optional body {"note": "..."}.
async fn motor_reject(
    State(gate): State<Arc<crate::motor::MotorGate>>,
    Path(id): Path<u64>,
    body: axum::body::Bytes,
) -> (StatusCode, Json<serde_json::Value>) {
    let note = if body.is_empty() {
        None
    } else {
        match serde_json::from_slice::<RejectBody>(&body) {
            Ok(b) => b.note,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "body must be JSON like {\"note\": \"...\"}"})),
                );
            }
        }
    };
    match gate.reject(id, note) {
        Ok(p) => {
            tracing::warn!(
                "[motor] REJECTED proposal #{} action={} domain={}",
                p.id, p.command.action, p.command.domain
            );
            (StatusCode::OK, Json(serde_json::to_value(&p).unwrap_or_default()))
        }
        Err(e) => gate_error_response(e),
    }
}

/// POST /walk — main cognitive endpoint
async fn walk(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WalkRequest>,
) -> Result<Json<WalkOutput>, ModeError> {
    let n = if req.n_walkers == 0 {
        rayon::current_num_threads()
    } else {
        req.n_walkers
    };

    // Rejected before any walk and before the self-model is touched.
    let arm = validate_arm(req.arm.as_deref(), crate::seeding::eval_enabled())?;

    // A lowered mode belongs to this request, including its critic updates. It is
    // never exposed to concurrent walks, the diverger or the dream loop.
    let self_model = request_self_model(&state.self_model, req.rgw_mode.as_deref()).await?;

    // The query, embedded. Too short or not embeddable means a fixed-seed walk. The rollback
    // mode and the `fixed` arm do not embed at all; an eval arm embeds without the cache, so
    // every arm of a harness run pays the real cost.
    let cue_embedding = if walker::cue_needed(crate::seeding::seed_mode(), arm) {
        crate::seeding::embed_cue(&req.stimulus, arm.is_some()).await
    } else {
        None
    };
    let opts = walker::WalkOptions {
        cue_embedding,
        seed: req.seed,
        arm,
        seed_mode: None,
    };
    let (mut output, walker_results) = walker::walk_parallel_with(
        &state.pool,
        &req.emotional_state,
        n,
        req.steps,
        &self_model,
        opts,
    )
    .await;

    // Prompt-ready narrative for the caller's LLM. Measured 2026-11-04 on the
    // Julian graph: injecting this beat no-context (61% vs 50%) while the old
    // one-liner callers synthesised from action/novelty did not (45%).
    output.walker_context = walker::format_walk_narrative(&output, &state.pool).await;

    // ── Metacognitive Critic: run after walk session ──
    // Algorithmic diagnosis always runs. LLM escalates only when warranted.
    // Not for an eval arm, and not when the query decided the result and no walker ran
    // (below the floor): neither is a walk session.
    if critic_applies(arm.is_some(), &output.seed_mode, walker_results.len()) {
        let mut sm = self_model.lock().await;
        let diagnosis = crate::metacog::run_critic(&output, &walker_results, &mut sm);

        // Escalate to the LLM critic only when it earns its cost:
        //   * the diagnosis flags genuine difficulty (stuck / explore / refine), or
        //   * a periodic calibration is due — but not while resting, and at a far
        //     larger interval than the old every-11-sessions default (cadence waste).
        let escalate = diagnosis.algorithmic_adjustments.escalate_to_llm;
        let interval: u32 = std::env::var("RGW_CRITIC_LLM_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        let periodic = sm.critic_sessions_since_llm > interval
            && diagnosis.primary_diagnosis != crate::metacog::DiagnosisLabel::Rest;
        if (escalate || periodic)
            && let Some(ref provider) = state.provider {
                let summary = crate::metacog::build_session_summary(&output, &walker_results, &sm);
                let (system, user) = crate::metacog::build_critic_llm_prompt(&diagnosis, &summary);

                // Genuine difficulty routes to the reasoner; periodic calibration stays on chat.
                let result = provider.generate(&system, &user, 0.5, escalate, Some(256), 0.3).await;
                crate::metacog::run_llm_critic(&diagnosis, &summary, &mut sm, &result.text).await;
            }
    }

    Ok(Json(output))
}

/// GET /self — current self-model state (consciousness introspection)
async fn self_state(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await;
    Json(serde_json::to_value(&*sm).unwrap_or_default())
}

/// GET /metacog/config — runtime metacognitive configuration.
/// This endpoint is auth-protected via the protected router's middleware.
async fn get_metacog_config(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await;
    Json(metacog_config_json(&sm))
}

#[derive(Deserialize, Default)]
struct MetacogConfigPatch {
    min_energy: Option<f32>,
    agreement_threshold_autonomous: Option<f32>,
    agreement_threshold_compliant: Option<f32>,
    max_attention_saturation: Option<f32>,
    wound_guard_threshold: Option<f32>,
    wound_confidence_floor: Option<f32>,
    phase_order: Option<Vec<MetacogPhase>>,
    learned_bias_rotation: Option<u64>,
    /// Spawn additional learned-bias profiles (bounded).
    spawn_bias_variants: Option<u8>,
    /// "legacy" | "domain" — which agreement signal drives decisions (default legacy).
    agreement_metric: Option<crate::core::AgreementMetric>,
}

/// POST /metacog/config — mutate metacognitive runtime config.
/// This endpoint is auth-protected via the protected router's middleware.
async fn set_metacog_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MetacogConfigPatch>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate before taking the lock or mutating anything: an invalid patch applies nothing.
    if let Err((status, body)) = validate_agreement_metric(req.agreement_metric) {
        return (status, Json(body));
    }

    let mut sm = state.self_model.lock().await;
    let rules_before = crate::metacog::RuleConfigSnapshot::of(&sm);

    if let Some(metric) = req.agreement_metric {
        let prev = sm.agreement_metric;
        sm.agreement_metric = metric;
        tracing::warn!(
            "[selection] agreement_metric {:?} -> {:?} (operator via /metacog/config)",
            prev, sm.agreement_metric
        );
        crate::metacog::on_metric_change(&mut sm, prev);
    }

    if let Some(v) = req.min_energy {
        sm.critic_rules.min_energy = v.clamp(0.05, 0.6);
    }
    if let Some(v) = req.agreement_threshold_autonomous {
        sm.critic_rules.agreement_threshold_autonomous = v.clamp(0.05, 0.9);
    }
    if let Some(v) = req.agreement_threshold_compliant {
        sm.critic_rules.agreement_threshold_compliant = v.clamp(0.05, 0.95);
    }
    if let Some(v) = req.max_attention_saturation {
        sm.critic_rules.max_attention_saturation = v.clamp(0.3, 0.95);
    }
    if let Some(v) = req.wound_guard_threshold {
        sm.critic_rules.wound_guard_threshold = v.clamp(0.1, 0.95);
    }
    if let Some(v) = req.wound_confidence_floor {
        sm.critic_rules.wound_confidence_floor = v.clamp(0.1, 0.99);
    }

    if let Some(phases) = req.phase_order {
        let has_draft = phases.iter().any(|p| p == &MetacogPhase::Draft);
        let has_critique = phases.iter().any(|p| p == &MetacogPhase::Critique);
        if !has_draft || !has_critique {
            // Fields earlier in this patch may already be applied (pre-existing partial apply);
            // still log them and clear the watch like any operator rule edit.
            crate::metacog::note_operator_rule_edit(&mut sm, &rules_before);
            return (StatusCode::OK, Json(serde_json::json!({
                "status": "error",
                "error": "phase_order must include both 'draft' and 'critique'"
            })));
        }
        sm.metacog_phase_order = phases;
    }

    if let Some(rot) = req.learned_bias_rotation {
        sm.learned_bias_rotation = rot;
    }

    if let Some(spawn) = req.spawn_bias_variants {
        spawn_bias_variants_api(&mut sm, spawn);
    }

    // Operator rule edits bypass Stage-2 trials by design (explicit human action): WARN-log
    // them and clear any regression watch (its baseline no longer belongs to the accept).
    crate::metacog::note_operator_rule_edit(&mut sm, &rules_before);

    (StatusCode::OK, Json(serde_json::json!({
        "status": "ok",
        "config": metacog_config_json(&sm),
    })))
}

/// Validate the agreement_metric field of a /metacog/config patch. Only "legacy" and "domain"
/// are settable; anything else (deserialized as Unknown) is a 400 with the error body.
fn validate_agreement_metric(
    metric: Option<crate::core::AgreementMetric>,
) -> Result<(), (StatusCode, serde_json::Value)> {
    match metric {
        Some(crate::core::AgreementMetric::Unknown) => Err((
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "status": "error",
                "error": "agreement_metric must be 'legacy' or 'domain'"
            }),
        )),
        _ => Ok(()),
    }
}

/// GET /selection/stats — Stage 0-2 self-selection state (read-only). Auth-protected.
async fn selection_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await;
    Json(selection_stats_json(&sm))
}

fn selection_stats_json(sm: &SelfModel) -> serde_json::Value {
    const HISTORY_SHOWN: usize = 20;
    let pool: Vec<serde_json::Value> = sm
        .learned_biases
        .iter()
        .enumerate()
        .map(|(i, b)| serde_json::json!({
            "index": i,
            "walks": b.scorecard.walks,
            "fitness": b.scorecard.fitness,
            "niche": crate::graph::behavioural_niche(&b.scorecard),
        }))
        .collect();
    let history: Vec<&crate::metacog::RuleTrialRecord> =
        sm.rule_trial_history.iter().rev().take(HISTORY_SHOWN).collect();
    serde_json::json!({
        "self_mod_stage": sm.self_mod_stage,
        "agreement_metric": sm.agreement_metric,
        "trials_live": crate::metacog::trials_live(sm),
        "pool": pool,
        "live_pool_fitness": crate::metacog::live_pool_fitness(sm),
        "pending_rule_proposals": sm.pending_rule_proposals,
        "rule_trial_history_latest_first": history,
        "rollback_depth": sm.rule_rollback_entries.len(),
        "newest_rollback_changed_fields": sm.rule_rollback_entries.back().map(|e| e.changed_field_names()),
        "regression_watch": sm.rule_regression_watch,
        "replay_buffer": {"len": sm.recent_walk_fitness.len(), "required": crate::metacog::REPLAY_M},
        "trial_config": crate::metacog::RULE_TRIAL_CONFIG,
    })
}

/// POST /selection/rollback — restore the rule config replaced by the last accepted trial.
/// 409 when the rollback ring is empty. Auth-protected, operator action, WARN-logged.
async fn selection_rollback(State(state): State<Arc<AppState>>) -> (StatusCode, Json<serde_json::Value>) {
    let mut sm = state.self_model.lock().await;
    match crate::metacog::rollback_last_rule(&mut sm) {
        Some(out) => {
            tracing::warn!(
                "[selection] ROLLBACK by operator via /selection/rollback; restored {:?}, skipped {:?} (depth now {})",
                out.restored, out.skipped, sm.rule_rollback_entries.len()
            );
            (StatusCode::OK, Json(serde_json::json!({
                "status": "ok",
                "restored_fields": out.restored,
                "skipped_fields": out.skipped,
                "entry": out.entry,
                "rollback_depth": sm.rule_rollback_entries.len(),
            })))
        }
        None => (StatusCode::CONFLICT, Json(serde_json::json!({
            "status": "error",
            "error": "rollback ring is empty",
        }))),
    }
}

#[derive(Deserialize)]
struct SelectionStagePatch {
    /// "observability" | "selection_live" | "rule_trials_live"
    stage: crate::core::SelfModStage,
}

/// POST /selection/stage — promote/demote the self-modification rollout stage.
/// Auth-protected. Default Observability = compute-only; SelectionLive enables cull/breed.
/// Operator action only — explicit and logged at WARN, never time-based.
async fn set_selection_stage(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SelectionStagePatch>,
) -> Json<serde_json::Value> {
    let mut sm = state.self_model.lock().await;
    let prev = sm.self_mod_stage;
    sm.self_mod_stage = req.stage;
    tracing::warn!("[selection] self_mod_stage {:?} -> {:?} (operator via /selection/stage)", prev, sm.self_mod_stage);
    crate::metacog::on_stage_change(&mut sm, prev);
    Json(serde_json::json!({
        "status": "ok",
        "previous": format!("{:?}", prev),
        "stage": format!("{:?}", sm.self_mod_stage),
    }))
}

fn spawn_bias_variants_api(sm: &mut SelfModel, count: u8) {
    const MAX_LEARNED_BIASES: usize = 16;

    for _ in 0..count {
        if sm.learned_biases.len() >= MAX_LEARNED_BIASES {
            break;
        }

        if sm.learned_biases.is_empty() {
            sm.learned_biases.push(crate::graph::LearnedBias::default());
        }

        let parent_idx = (sm.learned_bias_rotation as usize) % sm.learned_biases.len();
        let mut child = sm.learned_biases[parent_idx].clone();
        child.sessions_learned = 0;
        child.novelty_seeking = (child.novelty_seeking + 0.08).clamp(0.05, 0.95);
        child.cross_domain_curiosity = (child.cross_domain_curiosity + 0.06).clamp(0.05, 0.95);
        child.experience_reliance = (child.experience_reliance - 0.05).clamp(0.05, 0.95);
        sm.learned_biases.push(child);
    }
}

fn metacog_config_json(sm: &SelfModel) -> serde_json::Value {
    serde_json::json!({
        "critic_rules": {
            "min_energy": sm.critic_rules.min_energy,
            "agreement_threshold_autonomous": sm.critic_rules.agreement_threshold_autonomous,
            "agreement_threshold_compliant": sm.critic_rules.agreement_threshold_compliant,
            "max_attention_saturation": sm.critic_rules.max_attention_saturation,
            "wound_guard_threshold": sm.critic_rules.wound_guard_threshold,
            "wound_confidence_floor": sm.critic_rules.wound_confidence_floor,
        },
        "phase_order": sm.metacog_phase_order,
        "agreement_metric": sm.agreement_metric,
        "learned_bias_count": sm.learned_biases.len(),
        "learned_bias_rotation": sm.learned_bias_rotation,
        "active_goal_domain": sm.active_goal_domain,
        "active_goal_strength": sm.active_goal_strength,
        "active_audience_id": sm.active_audience_id,
    })
}

/// POST /self/save — persist self-model to database
async fn save_self_model_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await.clone();
    match db::save_self_model(&state.pool, &sm).await {
        Ok(_) => Json(serde_json::json!({"status": "saved"})),
        Err(e) => Json(serde_json::json!({"status": "error", "error": e.to_string()})),
    }
}

/// POST /tom/record — record a Theory-of-Mind interaction with an audience.
///
/// Lets the audience model populate even though the backends express through
/// their OWN LLM and never hit `/v1/chat/completions` (where ToM is wired
/// inline). They call this AFTER posting: `message` is what Julian said, and
/// optional `response` is the audience's prior reply (drives ToM prediction
/// error). Without this, A3 would be inert for those consumers.
#[derive(Deserialize)]
struct TomRecordRequest {
    audience_id: String,
    /// What Julian just said to this audience.
    message: String,
    #[serde(default)]
    domain: String,
    /// The audience's prior message, if any — compared against expectation.
    #[serde(default)]
    response: Option<String>,
}

async fn tom_record_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TomRecordRequest>,
) -> Json<serde_json::Value> {
    if req.audience_id.is_empty() {
        return Json(serde_json::json!({"status": "error", "error": "audience_id required"}));
    }
    let mut sm = state.self_model.lock().await;
    // Their prior message is a response to our previous output (ToM prediction error).
    if let Some(resp) = req.response.as_deref().filter(|r| !r.is_empty())
        && let Some(n) = crate::metacog::record_audience_response(&mut sm, &req.audience_id, resp)
    {
        sm.noticings.push(n);
    }
    crate::metacog::record_audience_output(&mut sm, &req.audience_id, &req.message, &req.domain);
    let interactions = sm
        .audience_model
        .get(&req.audience_id)
        .map(|a| a.interaction_count)
        .unwrap_or(0);
    Json(serde_json::json!({
        "status": "recorded",
        "audience_id": req.audience_id,
        "interactions": interactions,
        "audiences_tracked": sm.audience_model.len(),
    }))
}

/// GET /tools — list available tools
async fn list_tools() -> Json<Vec<crate::tools::Tool>> {
    Json(crate::tools::available_tools())
}

/// POST /tools/execute — execute a tool by name
#[derive(Deserialize)]
struct ToolExecRequest {
    tool: String,
    params: serde_json::Value,
}

async fn execute_tool_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ToolExecRequest>,
) -> Json<crate::tools::ToolResult> {
    let result = crate::tools::execute_tool(&req.tool, req.params, &state.pool).await;

    // Feed tool result through self-model
    {
        let mut sm = state.self_model.lock().await;
        // A motor hold (gate not live) is not a failure.
        let held = result.metadata.get("held").and_then(|v| v.as_bool()) == Some(true);
        let signal = crate::core::Signal::new(
            if result.success { "tool_success" } else if held { "motor_held" } else { "tool_failure" },
            &format!("{}: {}", result.tool, crate::core::safe_truncate(&result.content, 100)),
        ).with_intensity(if result.success { 0.3 } else { 0.2 });
        crate::core::process(signal, &mut sm);
    }

    Json(result)
}

/// POST /speak — text-to-speech
#[derive(Deserialize)]
struct SpeakRequest {
    text: String,
}

async fn speak_endpoint(
    Json(req): Json<SpeakRequest>,
) -> Result<axum::body::Bytes, StatusCode> {
    let config = crate::speech::SpeechConfig::default();
    match crate::speech::speak(&req.text, &config).await {
        Ok(audio) => Ok(axum::body::Bytes::from(audio)),
        Err(e) => {
            tracing::warn!("[speech] TTS failed: {}", e);
            Err(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

/// GET /music — current emotional state as music parameters
async fn music_endpoint(
    State(state): State<Arc<AppState>>,
) -> Json<crate::music::MusicParams> {
    let sm = state.self_model.lock().await;
    Json(crate::music::emotion_to_music(&sm))
}

/// GET /music/midi — generate MIDI from current emotional state
async fn music_midi_endpoint(
    State(state): State<Arc<AppState>>,
) -> (axum::http::StatusCode, [(axum::http::header::HeaderName, &'static str); 2], axum::body::Bytes) {
    let sm = state.self_model.lock().await;
    let params = crate::music::emotion_to_music(&sm);
    let midi = crate::music::generate_midi(&params);

    (
        axum::http::StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "audio/midi"),
            (axum::http::header::CONTENT_DISPOSITION, "attachment; filename=\"julian_mood.mid\""),
        ],
        axum::body::Bytes::from(midi),
    )
}

/// GET /music/prompt — see what MusicGen prompt Julian's mood would produce
async fn music_prompt_endpoint(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await;
    let prompt = crate::music::emotion_to_prompt(&sm);
    let params = crate::music::emotion_to_music(&sm);
    Json(serde_json::json!({
        "prompt": prompt,
        "params": params,
    }))
}

/// POST /music/generate — generate music via MusicGen from current emotional state
#[derive(Deserialize)]
struct MusicGenRequest {
    #[serde(default = "default_duration")]
    duration_secs: u32,
    /// Override prompt (if empty, auto-generated from emotional state)
    #[serde(default)]
    prompt: String,
}
fn default_duration() -> u32 { 15 }

async fn music_generate_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MusicGenRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let prompt = if req.prompt.is_empty() {
        let sm = state.self_model.lock().await;
        crate::music::emotion_to_prompt(&sm)
    } else {
        req.prompt
    };

    tracing::info!("[music] Generating via MusicGen: '{}'", crate::core::safe_truncate(&prompt, 80));

    match crate::music::generate_musicgen(&prompt, req.duration_secs).await {
        Ok(path) => {
            // Notify self-model: Julian created music
            {
                let mut sm = state.self_model.lock().await;
                let signal = crate::core::Signal::new("music_created", &format!(
                    "Composed: {}", crate::core::safe_truncate(&prompt, 60)
                )).with_intensity(0.5);
                crate::core::process(signal, &mut sm);
            }

            Ok(Json(serde_json::json!({
                "status": "ok",
                "path": path,
                "prompt": prompt,
                "duration_secs": req.duration_secs,
            })))
        }
        Err(e) => {
            // Friction: music generation failed
            {
                let mut sm = state.self_model.lock().await;
                crate::friction::motor_friction("music_generate", &e, &mut sm);
            }
            Ok(Json(serde_json::json!({
                "status": "error",
                "error": e,
                "prompt": prompt,
            })))
        }
    }
}

/// POST /dream — enter REM sleep, run Monte Carlo graph exploration
async fn dream_endpoint(
    State(state): State<Arc<AppState>>,
) -> Json<crate::dream::DreamReport> {
    let config = crate::dream::DreamConfig::default();
    let report = crate::dream::dream(&state.pool, &state.self_model, config).await;
    Json(report)
}

/// GET /metrics — live DeepSeek cost scorecard.
/// tokens/call, calls/min, est. cost, % reasoner, and dedup (duplicate-call) rate.
async fn metrics_endpoint() -> Json<crate::metrics::MetricsSnapshot> {
    Json(crate::metrics::snapshot())
}

/// GET /stats — graph topology
async fn stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match db::detailed_stats(&state.pool).await {
        Ok(stats) => Ok(Json(stats)),
        Err(e) => {
            tracing::error!("Stats query failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Query params for `GET /recall`.
#[derive(Deserialize)]
struct RecallParams {
    from: Option<String>,
    to: Option<String>,
    place: Option<String>,
    limit: Option<i32>,
}

/// GET /recall — read-only autobiographical recall (§7 of the
/// autobiographical-timeline design).
///
/// `?from&to&place&limit` → `{ where, where_since, episodes, texture }`.
/// `where`/`where_since` are the location in effect at-or-before `to` (the
/// most-recent `kind='location'` episode); `episodes` are events in `[from, to]`
/// (optionally filtered by `place`), chronological, capped at `limit`
/// (default 20). `texture` is reserved for a future associative `/walk`
/// colouring — null in v1. Strictly read-only: reads only the prune-exempt
/// `rgw_episodes` table; never touches the awake-loop, energy, walk cadence, or
/// selection. Query/DB errors degrade to an empty answer rather than 500.
async fn recall(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecallParams>,
) -> Json<serde_json::Value> {
    // Open-ended defaults: `infinity` for `to` means "as of now" (no episode is
    // future-dated); `-infinity` for `from` means "since the beginning".
    let to = params.to.filter(|s| !s.is_empty()).unwrap_or_else(|| "infinity".to_string());
    let from = params.from.filter(|s| !s.is_empty()).unwrap_or_else(|| "-infinity".to_string());
    let place = params.place.filter(|s| !s.is_empty());
    let limit = params.limit.unwrap_or(20);

    let location = db::location_as_of(&state.pool, &to).await.unwrap_or_else(|e| {
        tracing::debug!("[recall] location_as_of failed: {}", e);
        None
    });
    let episodes = db::recall_window(&state.pool, &from, &to, place.as_deref(), limit)
        .await
        .unwrap_or_else(|e| {
            tracing::debug!("[recall] recall_window failed: {}", e);
            Vec::new()
        });

    Json(serde_json::json!({
        "where": location.as_ref().and_then(|l| l.location.clone()),
        "where_since": location.as_ref().map(|l| l.occurred_at.clone()),
        "episodes": serde_json::to_value(&episodes).unwrap_or_else(|_| serde_json::json!([])),
        "texture": serde_json::Value::Null,
    }))
}

/// GET /diverger — self-propagating reactive graph stats
/// GET /learning — what the learning stages have measured. Plan P1 fills the `seed` block.
async fn learning_stats() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "seed": crate::seeding::seed_stats_json() }))
}

async fn diverger_stats(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let stats = state.diverger.stats().await;
    let tension = match crate::tension::engine() {
        Some(e) => {
            let sm = state.self_model.lock().await.clone();
            serde_json::to_value(e.stats(&sm).await).unwrap_or_default()
        }
        None => serde_json::json!({ "mode": "off" }),
    };
    Json(serde_json::json!({
        "tension": tension,
        "status": if stats.alive { "alive" } else { "stopped" },
        "active_nodes": stats.active_nodes,
        "total_energy": stats.total_energy,
        "walks_fired": stats.walks_fired,
        "cascades_total": stats.cascades_total,
        "edges_changed": stats.edges_changed,
        "emotional_state": stats.emotional_state,
        "hottest_nodes": stats.hottest_nodes,
    }))
}

/// Notification that an edge changed — the Diverger reacts
#[derive(Deserialize)]
struct NotifyRequest {
    edge_id: i32,
    source_id: i32,
    target_id: i32,
    delta: f32,
    #[serde(default)]
    edge_type: String,
}

/// POST /diverger/notify — external systems notify edge changes
async fn diverger_notify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NotifyRequest>,
) -> StatusCode {
    state.diverger.notify_edge_change(EdgeChange {
        edge_id: req.edge_id,
        source_id: req.source_id,
        target_id: req.target_id,
        delta: req.delta,
        edge_type: req.edge_type,
    });
    StatusCode::ACCEPTED
}

/// POST /edge — create a new edge (node genesis / graph expansion)
#[derive(Deserialize)]
struct CreateEdgeRequest {
    source_id: i32,
    target_id: i32,
    edge_type: String,
    #[serde(default = "default_weight")]
    weight: f32,
    #[serde(default)]
    emotional_charge: f32,
}
fn default_weight() -> f32 { 0.5 }

async fn create_edge_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateEdgeRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match db::create_edge(
        &state.pool,
        req.source_id,
        req.target_id,
        &req.edge_type,
        req.weight,
        req.emotional_charge,
    ).await {
        Ok(edge_id) => {
            // Notify the Diverger about the new edge
            state.diverger.notify_edge_change(EdgeChange {
                edge_id,
                source_id: req.source_id,
                target_id: req.target_id,
                delta: req.weight,
                edge_type: req.edge_type,
            });
            Ok(Json(serde_json::json!({"status": "ok", "edge_id": edge_id})))
        }
        Err(e) => {
            tracing::error!("[edge] Creation failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// POST /prune — synaptic pruning: decay + delete neglected edges
async fn prune(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let decay_per_day = 0.02;  // Lose 0.02 weight per day of neglect
    let min_weight = 0.01;      // Below this = forgotten = deleted

    match db::prune_edges(&state.pool, decay_per_day, min_weight).await {
        Ok((decayed, deleted)) => {
            tracing::info!("[prune] Decayed {} edges, deleted {} dead edges", decayed, deleted);
            Json(serde_json::json!({
                "status": "ok",
                "decayed": decayed,
                "deleted": deleted,
            }))
        }
        Err(e) => {
            Json(serde_json::json!({
                "status": "error",
                "error": e.to_string(),
            }))
        }
    }
}

/// POST /ingest/signal — inject an external signal into the cognitive loop.
///
/// Use this for RSS feed items, inbound emails, webhooks, calendar events,
/// or any external data source that should enter the agent's awareness.
///
/// The signal flows through core::process(), meaning the self-model observes it,
/// and in Autonomous mode, it can trigger emotional shifts, focus changes,
/// and downstream diverger activity.
#[derive(Deserialize)]
struct IngestSignalRequest {
    /// Signal kind: "rss", "email", "webhook", "calendar", "social_mention", "analytics", etc.
    kind: String,
    /// The content (human-readable text that the agent will process)
    content: String,
    /// Domain tag for attention tracking
    #[serde(default)]
    domain: String,
    /// Signal strength (0.0-1.0, default 0.5)
    #[serde(default = "default_intensity")]
    intensity: f32,
}
fn default_intensity() -> f32 { 0.5 }

async fn ingest_signal_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestSignalRequest>,
) -> Json<serde_json::Value> {
    let domain = if req.domain.is_empty() { &req.kind } else { &req.domain };

    // ── Awake driver: content → embedding → find similar nodes → inject energy ──
    // This is the perception→walk→motor loop: external input activates
    // matching graph nodes, which cascades to spontaneous walks.
    let matched_nodes: Vec<(i32, f32)> = if !req.content.is_empty() && req.content.len() > 10 {
        match crate::embed::embed_text(&req.content) {
            Ok(embedding) => {
                match crate::db::find_similar_nodes(&state.pool, &embedding, 5, 0.7).await {
                    Ok(nodes) => {
                        for (node_id, similarity, _content) in &nodes {
                            state.diverger.notify_edge_change(EdgeChange {
                                edge_id: 0,
                                source_id: *node_id,
                                target_id: *node_id, // self-loop: activate the node itself
                                delta: *similarity * req.intensity,
                                edge_type: "activated_by".into(),
                            });
                        }
                        tracing::debug!(
                            "[ingest] Content matched {} nodes (best sim={:.2}), injected energy",
                            nodes.len(),
                            nodes.first().map(|(_, s, _)| *s).unwrap_or(0.0)
                        );
                        nodes.iter().map(|(id, sim, _)| (*id, (*sim))).collect()
                    }
                    Err(e) => {
                        tracing::debug!("[ingest] find_similar_nodes failed: {}", e);
                        Vec::new()
                    }
                }
            }
            Err(e) => {
                tracing::debug!("[ingest] embed_text failed: {}", e);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // ── Self-model update ──
    let mut sm = state.self_model.lock().await;

    let signal = crate::core::Signal::new(&req.kind, &req.content)
        .with_domain(domain)
        .with_intensity(req.intensity.clamp(0.0, 1.0));

    let (output, noticing) = crate::core::process(signal, &mut sm);

    Json(serde_json::json!({
        "status": "ingested",
        "kind": req.kind,
        "output_intensity": output.intensity,
        "noticing": noticing.as_ref().map(|n| serde_json::json!({
            "kind": n.kind,
            "observation": n.observation,
            "significance": n.significance,
        })),
        "matched_nodes": matched_nodes.len(),
        "self_model": {
            "valence": sm.valence,
            "arousal": sm.arousal,
            "energy": sm.energy,
            "focus": sm.current_focus,
            "mode": format!("{:?}", sm.mode),
        },
    }))
}

/// Fixed errors never echo caller-controlled values or request bodies.
pub(crate) type ModeError = (StatusCode, &'static str);

/// Whether the metacognitive critic runs after this walk. It does not run for an eval arm
/// (a measurement, which must neither change the self-model nor be timed with an LLM call),
/// nor when the query decided the result and no walker ran. An empty fixed walk still
/// reaches the critic, as before.
pub(crate) fn critic_applies(arm_present: bool, seed_mode: &str, n_results: usize) -> bool {
    !arm_present && !(seed_mode == "query" && n_results == 0)
}

/// The eval arm of a request: `Ok(None)` when none was sent. An arm sent without
/// `RGW_EVAL=1`, or one that does not parse, is a 400.
pub(crate) fn validate_arm(arm: Option<&str>, eval_enabled: bool) -> Result<Option<walker::EvalArm>, ModeError> {
    let Some(arm) = arm else { return Ok(None); };
    if !eval_enabled {
        return Err((StatusCode::BAD_REQUEST, "arm requires RGW_EVAL=1"));
    }
    match walker::EvalArm::parse(arm) {
        Some(parsed) => Ok(Some(parsed)),
        None => Err((StatusCode::BAD_REQUEST, "unknown arm")),
    }
}

pub(crate) fn validate_request_mode(
    requested: Option<&str>, configured: &CognitiveMode,
) -> Result<Option<CognitiveMode>, ModeError> {
    let Some(requested) = requested else { return Ok(None); };
    let Some(target) = parse_mode(requested) else {
        tracing::warn!("[rgw] rejected invalid rgw_mode");
        return Err((StatusCode::BAD_REQUEST, "rgw_mode must be autonomous or compliant"));
    };
    if *configured == CognitiveMode::Compliant && target == CognitiveMode::Autonomous {
        tracing::warn!("[rgw] rejected rgw_mode autonomy escalation");
        return Err((StatusCode::FORBIDDEN, "rgw_mode exceeds the configured instance mode"));
    }
    Ok(Some(target))
}

/// The self-model a request walks against. No mode, or the instance's own mode,
/// uses the shared model so the walk's learning persists. A lower mode runs on a
/// request-local snapshot that nothing else can observe; nothing is restored
/// afterwards, so a cancelled request or a concurrent operator change cannot be
/// overwritten. Raising autonomy is rejected by `validate_request_mode`.
pub(crate) async fn request_self_model(
    shared: &Arc<tokio::sync::Mutex<SelfModel>>, requested: Option<&str>,
) -> Result<Arc<tokio::sync::Mutex<SelfModel>>, ModeError> {
    let sm = shared.lock().await;
    match validate_request_mode(requested, &sm.mode)? {
        Some(target) if target != sm.mode => {
            let mut snapshot = sm.clone();
            snapshot.mode = target;
            Ok(Arc::new(tokio::sync::Mutex::new(snapshot)))
        }
        _ => Ok(Arc::clone(shared)),
    }
}

/// Parse a mode string ("autonomous", "compliant") into CognitiveMode
fn parse_mode(s: &str) -> Option<CognitiveMode> {
    match s.to_lowercase().as_str() {
        "autonomous" => Some(CognitiveMode::Autonomous),
        "compliant" => Some(CognitiveMode::Compliant),
        _ => None,
    }
}

/// POST /self/mode — switch cognitive mode globally
#[derive(Deserialize)]
struct SetModeRequest {
    mode: String,
}

async fn set_mode_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetModeRequest>,
) -> Json<serde_json::Value> {
    match parse_mode(&req.mode) {
        Some(mode) => {
            let prev = {
                let mut sm = state.self_model.lock().await;
                let prev = sm.mode.clone();
                sm.mode = mode.clone();
                prev
            };
            tracing::info!("[rgw] Cognitive mode: {:?} → {:?}", prev, mode);
            Json(serde_json::json!({
                "status": "ok",
                "previous": format!("{:?}", prev),
                "current": format!("{:?}", mode),
            }))
        }
        None => {
            Json(serde_json::json!({
                "status": "error",
                "error": format!("Unknown mode '{}'. Use 'autonomous' or 'compliant'", req.mode),
            }))
        }
    }
}

/// GET /benchmark — run multiple walk configs and report performance
async fn benchmark(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let emotion = EmotionalState::default();
    let mut results = Vec::new();

    for n_walkers in [1, 2, 4, rayon::current_num_threads().min(8)] {
        let mut times = Vec::new();

        for _ in 0..3 {
            let start = Instant::now();
            let (output, _) = walker::walk_parallel(
                &state.pool,
                &emotion,
                n_walkers,
                5,
                &state.self_model,
            )
            .await;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            times.push((elapsed, output.total_hops));
        }

        let avg_ms = times.iter().map(|(t, _)| t).sum::<f64>() / times.len() as f64;
        let total_hops = n_walkers * 5;
        let hops_per_sec = total_hops as f64 / (avg_ms / 1000.0);

        results.push(serde_json::json!({
            "walkers": n_walkers,
            "steps": 5,
            "total_hops": total_hops,
            "avg_ms": (avg_ms * 10.0).round() / 10.0,
            "min_ms": (times.iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min) * 10.0).round() / 10.0,
            "max_ms": (times.iter().map(|(t, _)| *t).fold(0.0_f64, f64::max) * 10.0).round() / 10.0,
            "hops_per_sec": hops_per_sec.round(),
            "ticks_per_sec": (1000.0 / avg_ms * 10.0).round() / 10.0,
        }));
    }

    let fastest = results.iter().min_by(|a, b| {
        a["avg_ms"].as_f64().unwrap().partial_cmp(&b["avg_ms"].as_f64().unwrap()).unwrap()
    }).cloned();

    Json(serde_json::json!({
        "cpu_cores": rayon::current_num_threads(),
        "runtime": "rust",
        "results": results,
        "summary": {
            "fastest": fastest,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::http::StatusCode;
    use axum::routing::get;

    /// Serialises tests that mutate RGW_API_KEY (process env is shared across test threads).
    static API_KEY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct ApiKeyGuard {
        previous: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl ApiKeyGuard {
        fn set(value: Option<&str>) -> Self {
            let lock = API_KEY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let previous = std::env::var("RGW_API_KEY").ok();
            match value {
                Some(v) => {
                    // Safety: tests set process env in a narrow scope and restore on Drop.
                    unsafe { std::env::set_var("RGW_API_KEY", v) };
                }
                None => {
                    // Safety: tests set process env in a narrow scope and restore on Drop.
                    unsafe { std::env::remove_var("RGW_API_KEY") };
                }
            }
            Self { previous, _lock: lock }
        }
    }

    impl Drop for ApiKeyGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => {
                    // Safety: restore prior value after test completes.
                    unsafe { std::env::set_var("RGW_API_KEY", v) };
                }
                None => {
                    // Safety: restore prior unset state after test completes.
                    unsafe { std::env::remove_var("RGW_API_KEY") };
                }
            }
        }
    }

    async fn spawn_auth_test_app() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/metacog/config", get(|| async { StatusCode::OK }))
            .route_layer(middleware::from_fn(require_auth));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn auth_normalizes_configured_key_like_backend_sender() {
        let _guard = ApiKeyGuard::set(Some("  test-key  "));
        let (base, server) = spawn_auth_test_app().await;
        let client = reqwest::Client::new();
        let response = client.get(format!("{base}/metacog/config"))
            .header("X-RGW-Key", "test-key").send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let wrong = client.get(format!("{base}/metacog/config"))
            .header("X-RGW-Key", "wrong-key").send().await.unwrap();
        assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
        server.abort();
    }

    #[tokio::test]
    async fn metacog_config_auth_respects_rgw_api_key() {
        let guard = ApiKeyGuard::set(Some("test-key"));
        let (base_url, server) = spawn_auth_test_app().await;
        let client = reqwest::Client::new();

        // Missing key should be rejected when RGW_API_KEY is set.
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // Wrong key should also be rejected.
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .header("X-RGW-Key", "wrong")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // Correct key should be accepted.
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .header("X-RGW-Key", "test-key")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // Fail-closed when the key is unset: 401 even with a header.
        // Safety: scoped mutation restored by guard drop.
        unsafe { std::env::remove_var("RGW_API_KEY") };
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .header("X-RGW-Key", "")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // Empty key configured → also 401.
        // Safety: scoped mutation restored by guard drop.
        unsafe { std::env::set_var("RGW_API_KEY", "") };
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .header("X-RGW-Key", "")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        server.abort();
        drop(guard);
    }

    #[test]
    fn validate_api_key_table() {
        assert!(validate_api_key(None).is_err());
        assert!(validate_api_key(Some("")).is_err());
        assert!(validate_api_key(Some("   ")).is_err());
        assert!(validate_api_key(Some("short")).is_ok(), "short keys warn, not reject");
        let dummy = "0123456789abcdef".repeat(4); // 64-hex dummy
        assert!(validate_api_key(Some(&dummy)).is_ok());

        let msg = validate_api_key(Some("  ")).unwrap_err();
        assert!(msg.contains("RGW_API_KEY"));
        assert!(msg.contains("openssl rand -hex 32"));
    }

    #[tokio::test]
    async fn t5_motor_routes_auth_and_confirm() {
        use crate::motor::{DispatchOutcome, MotorGate, MotorMode, test_support};
        use std::sync::atomic::Ordering;

        let guard = ApiKeyGuard::set(Some("test-key"));
        let (capture_url, hits, capture) = test_support::spawn_capture().await;

        let gate = Arc::new(MotorGate::new(MotorMode::Propose));
        let DispatchOutcome::Queued(id) = gate.dispatch("diverger", &capture_url, test_support::cmd("journal")).await
        else {
            panic!("expected Queued")
        };

        let app: Router = motor_routes(gate.clone()).route_layer(middleware::from_fn(require_auth));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let client = reqwest::Client::new();

        // No key → 401.
        let res = client.get(format!("{}/motor/proposals", base)).send().await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // With key → snapshot shows the pending proposal.
        let res = client
            .get(format!("{}/motor/proposals", base))
            .header("X-RGW-Key", "test-key")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let snap: serde_json::Value = res.json().await.unwrap();
        assert_eq!(snap["mode"], "propose");
        assert_eq!(snap["pending"].as_array().unwrap().len(), 1);

        let approve = format!("{}/motor/proposals/{}/approve", base, id);
        // Missing body, wrong value, wrong case → 400, nothing sent.
        for body in [None, Some(r#"{"confirm":"y"}"#), Some(r#"{"confirm":"YES"}"#), Some("not json")] {
            let mut req = client.post(&approve).header("X-RGW-Key", "test-key");
            if let Some(b) = body {
                req = req.header("content-type", "application/json").body(b);
            }
            let res = req.send().await.unwrap();
            assert_eq!(res.status(), StatusCode::BAD_REQUEST, "body {:?}", body);
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0);

        // Approve with confirm but without the key → 401, nothing sent.
        let res = client.post(&approve).json(&serde_json::json!({"confirm": "yes"})).send().await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(hits.load(Ordering::SeqCst), 0);

        // Proper approve → 200, exactly one POST.
        let res = client
            .post(&approve)
            .header("X-RGW-Key", "test-key")
            .json(&serde_json::json!({"confirm": "yes"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let p: serde_json::Value = res.json().await.unwrap();
        assert_eq!(p["status"], "approved_sent");
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // Again → 409; unknown → 404; neither sends.
        let res = client
            .post(&approve)
            .header("X-RGW-Key", "test-key")
            .json(&serde_json::json!({"confirm": "yes"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CONFLICT);
        let res = client
            .post(format!("{}/motor/proposals/9999/reject", base))
            .header("X-RGW-Key", "test-key")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        server.abort();
        capture.abort();
        drop(guard);
    }

    #[test]
    fn selection_stats_json_reports_stage2_state() {
        let sm = SelfModel::new();
        let v = selection_stats_json(&sm);
        assert_eq!(v["self_mod_stage"], "observability");
        assert_eq!(v["agreement_metric"], "legacy");
        assert_eq!(v["trials_live"], false);
        assert_eq!(v["rollback_depth"], 0);
        assert_eq!(v["replay_buffer"]["required"], crate::metacog::REPLAY_M);
        assert!(v["regression_watch"].is_null());
    }

    #[test]
    fn validate_agreement_metric_accepts_legacy_domain_rejects_unknown_with_400() {
        use crate::core::AgreementMetric;
        assert!(validate_agreement_metric(None).is_ok(), "field absent: nothing to validate");
        assert!(validate_agreement_metric(Some(AgreementMetric::Legacy)).is_ok());
        assert!(validate_agreement_metric(Some(AgreementMetric::Domain)).is_ok());
        let (status, body) = validate_agreement_metric(Some(AgreementMetric::Unknown)).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["status"], "error");

        // The patch's own deserialization maps an unknown string to Unknown (→ 400), not a 422.
        let patch: MetacogConfigPatch = serde_json::from_str(r#"{"agreement_metric":"domian"}"#).unwrap();
        assert!(validate_agreement_metric(patch.agreement_metric).is_err());
        let ok: MetacogConfigPatch = serde_json::from_str(r#"{"agreement_metric":"domain"}"#).unwrap();
        assert!(validate_agreement_metric(ok.agreement_metric).is_ok());
    }
}

#[cfg(test)]
mod mode_ceiling_tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    async fn state(mode: CognitiveMode) -> Arc<AppState> {
        // Closed pool proves rejection requires no database or running services.
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgresql://localhost/unused").unwrap();
        pool.close().await;
        let mut sm = SelfModel::new();
        sm.mode = mode;
        let self_model = Arc::new(tokio::sync::Mutex::new(sm));
        Arc::new(AppState {
            motor_gate: Arc::new(crate::motor::MotorGate::new(crate::motor::MotorMode::Propose)),
            diverger: Diverger::new(pool.clone(), self_model.clone(), ""),
            pool, self_model, ollama_url: String::new(),
            expression_model: String::new(), julian_url: String::new(), provider: None,
        })
    }

    #[test]
    fn mode_matrix() {
        for configured in [CognitiveMode::Compliant, CognitiveMode::Autonomous] {
            assert_eq!(validate_request_mode(None, &configured), Ok(None));
            for input in ["compliant", "COMPLIANT", "Compliant"] {
                assert_eq!(validate_request_mode(Some(input), &configured), Ok(Some(CognitiveMode::Compliant)));
            }
            for input in ["autonomous", "AUTONOMOUS", "Autonomous", "AuToNoMoUs"] {
                let result = validate_request_mode(Some(input), &configured);
                if configured == CognitiveMode::Compliant {
                    assert_eq!(result.unwrap_err().0, StatusCode::FORBIDDEN);
                } else {
                    assert_eq!(result, Ok(Some(CognitiveMode::Autonomous)));
                }
            }
            for input in ["", "unknown", " autonomous ", "autonomous\n", "compliant\0autonomous"] {
                assert_eq!(validate_request_mode(Some(input), &configured).unwrap_err().0, StatusCode::BAD_REQUEST);
            }
        }
    }

    #[tokio::test]
    async fn both_routes_reject_before_execution_including_bypass_and_bad_types() {
        let state = state(CognitiveMode::Compliant).await;
        let app = Router::new()
            .route("/walk", post(walk))
            .route("/v1/chat/completions", post(openai::chat_completions))
            .with_state(state.clone());
        let before = serde_json::to_value(&*state.self_model.lock().await).unwrap();
        for path in ["/walk", "/v1/chat/completions"] {
            for bypass in [false, true] {
                for (value, expected) in [
                    (serde_json::json!("autonomous"), StatusCode::FORBIDDEN),
                    (serde_json::json!("AuToNoMoUs"), StatusCode::FORBIDDEN),
                    (serde_json::json!("unknown"), StatusCode::BAD_REQUEST),
                    (serde_json::json!(["autonomous"]), StatusCode::UNPROCESSABLE_ENTITY),
                    (serde_json::json!({"mode": "autonomous"}), StatusCode::UNPROCESSABLE_ENTITY),
                    (serde_json::json!(true), StatusCode::UNPROCESSABLE_ENTITY),
                    (serde_json::json!(1), StatusCode::UNPROCESSABLE_ENTITY),
                ] {
                    let body = serde_json::json!({"messages": [], "rgw_mode": value, "rgw_bypass": bypass});
                    let response = app.clone().oneshot(Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string())).unwrap()).await.unwrap();
                    assert_eq!(response.status(), expected, "{path}: {body}");
                    assert_eq!(serde_json::to_value(&*state.self_model.lock().await).unwrap(), before);
                }
            }
        }
    }

    #[tokio::test]
    async fn both_handlers_keep_shared_mode_and_learn_only_at_the_instance_mode() {
        for configured in [CognitiveMode::Compliant, CognitiveMode::Autonomous] {
            for requested in [None, Some("compliant"), Some(match configured {
                CognitiveMode::Compliant => "compliant", CognitiveMode::Autonomous => "autonomous",
            })] {
                for chat in [false, true] {
                    let state = state(configured.clone()).await;
                    let mut body = serde_json::json!({"messages": []});
                    if let Some(requested) = requested {
                        body["rgw_mode"] = serde_json::json!(requested);
                    }
                    if chat {
                        let req = serde_json::from_value(body).unwrap();
                        assert!(openai::chat_completions(State(state.clone()), Json(req)).await.is_ok());
                    } else {
                        let req = serde_json::from_value(body).unwrap();
                        assert!(walk(State(state.clone()), Json(req)).await.is_ok());
                    }
                    let sm = state.self_model.lock().await;
                    assert_eq!(sm.mode, configured);
                    // A lowered mode walks a request-local snapshot; the shared mind learns
                    // only from walks at its own mode (absent or same, e.g. Renu's walks).
                    let lowered = requested == Some("compliant") && configured == CognitiveMode::Autonomous;
                    assert_eq!(sm.total_signals_processed > 0, !lowered, "{configured:?} {requested:?} chat={chat}");
                }
            }
        }
    }

    #[tokio::test]
    async fn lowered_overrides_are_request_local_and_never_restored() {
        let shared = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
        let (first, second) = tokio::join!(
            request_self_model(&shared, Some("compliant")),
            request_self_model(&shared, Some("compliant")),
        );
        let (first, second) = (first.unwrap(), second.unwrap());
        assert!(!Arc::ptr_eq(&first, &shared) && !Arc::ptr_eq(&first, &second));
        assert_eq!(first.lock().await.mode, CognitiveMode::Compliant);
        assert_eq!(shared.lock().await.mode, CognitiveMode::Autonomous);
        // An operator change while the override is in flight survives the request.
        shared.lock().await.mode = CognitiveMode::Compliant;
        drop((first, second)); // cancellation: there is no stale mode to restore
        assert_eq!(shared.lock().await.mode, CognitiveMode::Compliant);
    }

    #[tokio::test]
    async fn absent_or_same_mode_walks_the_shared_model() {
        let shared = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
        for requested in [None, Some("autonomous"), Some("AUTONOMOUS")] {
            assert!(Arc::ptr_eq(&request_self_model(&shared, requested).await.unwrap(), &shared));
        }
    }

    #[tokio::test]
    async fn rejected_overrides_leave_the_shared_model_untouched() {
        let shared = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
        shared.lock().await.mode = CognitiveMode::Compliant;
        let before = serde_json::to_value(&*shared.lock().await).unwrap();
        assert_eq!(request_self_model(&shared, Some("autonomous")).await.unwrap_err().0, StatusCode::FORBIDDEN);
        assert_eq!(request_self_model(&shared, Some("unknown")).await.unwrap_err().0, StatusCode::BAD_REQUEST);
        assert_eq!(serde_json::to_value(&*shared.lock().await).unwrap(), before);
    }

    use crate::walker::EvalArm;

    #[tokio::test]
    async fn arm_is_rejected_without_eval_and_when_unknown() {
        let state = state(CognitiveMode::Autonomous).await;
        let app = Router::new().route("/walk", post(walk)).with_state(state);
        let send = |body: serde_json::Value| app.clone().oneshot(Request::post("/walk")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap());
        // RGW_EVAL is not set in the test process.
        assert_eq!(send(serde_json::json!({"arm": "query"})).await.unwrap().status(), StatusCode::BAD_REQUEST);
        assert_eq!(validate_arm(Some("nonsense"), true).unwrap_err().0, StatusCode::BAD_REQUEST);
        assert_eq!(validate_arm(Some("ppr"), true).unwrap(), Some(EvalArm::Ppr));
        assert_eq!(validate_arm(None, false).unwrap(), None);
    }

    #[test]
    fn critic_runs_only_for_real_walk_sessions() {
        assert!(critic_applies(false, "shadow", 2));
        assert!(critic_applies(false, "query", 3));
        assert!(critic_applies(false, "fixed", 0), "an empty fixed walk reaches the critic as before");
        assert!(!critic_applies(false, "query", 0), "no walker ran: below the floor");
        assert!(!critic_applies(true, "query", 4), "eval arms are measurements, not sessions");
        assert!(!critic_applies(true, "fixed", 4));
    }
}
