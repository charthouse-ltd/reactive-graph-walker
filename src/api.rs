//! HTTP API — serves the same endpoints as the Python walker_service.
//! Drop-in replacement: walker_client.py doesn't know it's Rust.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Query, Request, State},
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
    /// If set, temporarily switches mode for this walk only.
    #[serde(default)]
    pub rgw_mode: Option<String>,
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

    // Create the Diverger with shared self-model + Julian URL for motor commands
    let diverger = Diverger::new(pool.clone(), self_model.clone(), julian_url);

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
    });

    if std::env::var("RGW_API_KEY").ok().filter(|s| !s.is_empty()).is_none() {
        tracing::warn!(
            "[rgw] RGW_API_KEY not set — protected endpoints are UNAUTHENTICATED (fail-open). \
             Set RGW_API_KEY (and send X-RGW-Key on callers) to require auth on \
             /walk, /v1/chat/completions, /ingest/signal, etc."
        );
    }

    // ── Public endpoint (no auth) — health only ──
    // Strict posture: ONLY /health is open (liveness probes work without a key).
    // Everything else — including read-only introspection — requires X-RGW-Key
    // when RGW_API_KEY is set.
    let public = Router::new()
        .route("/health", get(health));

    // ── Protected endpoints (require X-RGW-Key when RGW_API_KEY is set) ──
    // Everything except /health: read-only introspection AND mutations / actions
    // / LLM-spend. Fail-open until the key is set on both sides. See audit C1/S3.
    let protected = Router::new()
        // read-only introspection
        .route("/stats", get(stats))
        .route("/recall", get(recall))
        .route("/metrics", get(metrics_endpoint))
        .route("/diverger", get(diverger_stats))
        .route("/self", get(self_state))
        .route("/metacog/config", get(get_metacog_config).post(set_metacog_config))
        .route("/selection/stage", post(set_selection_stage))
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
        .route_layer(middleware::from_fn(require_auth));

    let app = public
        .merge(protected)
        .with_state(state.clone())
        .layer(CorsLayer::permissive());

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
    axum::serve(listener, app).await?;
    Ok(())
}

/// Auth middleware for mutating / cost endpoints.
/// Convention shared with the deployed backends: the secret travels as the
/// `X-RGW-Key` header and matches the `RGW_API_KEY` env var. Fail-open when
/// `RGW_API_KEY` is unset on this side (preserves behaviour during graceful
/// rollout — the callers send nothing until the key is set on both sides);
/// fail-closed once it is configured. Only /health bypasses this (liveness).
async fn require_auth(req: Request, next: Next) -> Result<Response, StatusCode> {
    let Some(expected) = std::env::var("RGW_API_KEY").ok().filter(|s| !s.is_empty()) else {
        // Fail-open: no key configured → allow (a startup WARN already fired).
        return Ok(next.run(req).await);
    };

    let presented = req
        .headers()
        .get("X-RGW-Key")
        .and_then(|v| v.to_str().ok());

    match presented {
        Some(key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => Ok(next.run(req).await),
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
    })
}

/// POST /walk — main cognitive endpoint
async fn walk(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WalkRequest>,
) -> Result<Json<WalkOutput>, StatusCode> {
    let n = if req.n_walkers == 0 {
        rayon::current_num_threads()
    } else {
        req.n_walkers
    };

    // Per-request mode override: temporarily switch, walk, restore
    let mode_target = req.rgw_mode.as_ref().and_then(|m| parse_mode(m));
    let prev_mode = if let Some(target) = mode_target {
        let mut sm = state.self_model.lock().await;
        let prev = sm.mode.clone();
        sm.mode = target;
        Some(prev)
    } else {
        None
    };

    let (output, walker_results) = walker::walk_parallel(
        &state.pool,
        &req.emotional_state,
        n,
        req.steps,
        &state.self_model,
    )
    .await;

    // ── Metacognitive Critic: run after walk session ──
    // Algorithmic diagnosis always runs. LLM escalates only when warranted.
    {
        let mut sm = state.self_model.lock().await;
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
        if escalate || periodic {
            if let Some(ref provider) = state.provider {
                let summary = crate::metacog::build_session_summary(&output, &walker_results, &sm);
                let (system, user) = crate::metacog::build_critic_llm_prompt(&diagnosis, &summary);

                // Genuine difficulty routes to the reasoner; periodic calibration stays on chat.
                let result = provider.generate(&system, &user, 0.5, escalate, Some(256), 0.3).await;
                crate::metacog::run_llm_critic(&diagnosis, &summary, &mut sm, &result.text).await;
            }
        }
    }

    // Restore previous mode if we overrode it
    if let Some(prev) = prev_mode {
        let mut sm = state.self_model.lock().await;
        sm.mode = prev;
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
}

/// POST /metacog/config — mutate metacognitive runtime config.
/// This endpoint is auth-protected via the protected router's middleware.
async fn set_metacog_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MetacogConfigPatch>,
) -> Json<serde_json::Value> {
    let mut sm = state.self_model.lock().await;

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
            return Json(serde_json::json!({
                "status": "error",
                "error": "phase_order must include both 'draft' and 'critique'"
            }));
        }
        sm.metacog_phase_order = phases;
    }

    if let Some(rot) = req.learned_bias_rotation {
        sm.learned_bias_rotation = rot;
    }

    if let Some(spawn) = req.spawn_bias_variants {
        spawn_bias_variants_api(&mut sm, spawn);
    }

    Json(serde_json::json!({
        "status": "ok",
        "config": metacog_config_json(&sm),
    }))
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
        let signal = crate::core::Signal::new(
            if result.success { "tool_success" } else { "tool_failure" },
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
async fn diverger_stats(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let stats = state.diverger.stats().await;
    Json(serde_json::json!({
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
                                delta: *similarity as f32 * req.intensity,
                                edge_type: "activated_by".into(),
                            });
                        }
                        tracing::debug!(
                            "[ingest] Content matched {} nodes (best sim={:.2}), injected energy",
                            nodes.len(),
                            nodes.first().map(|(_, s, _)| *s as f32).unwrap_or(0.0)
                        );
                        nodes.iter().map(|(id, sim, _)| (*id, *sim as f32)).collect()
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

    struct ApiKeyGuard {
        previous: Option<String>,
    }

    impl ApiKeyGuard {
        fn set(value: Option<&str>) -> Self {
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
            Self { previous }
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

        // Fail-open behavior when key is unset.
        // Safety: scoped mutation restored by guard drop.
        unsafe { std::env::remove_var("RGW_API_KEY") };
        let res = client
            .get(format!("{}/metacog/config", base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        server.abort();
        drop(guard);
    }
}
