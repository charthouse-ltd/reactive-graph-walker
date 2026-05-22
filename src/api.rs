//! HTTP API — serves the same endpoints as the Python walker_service.
//! Drop-in replacement: walker_client.py doesn't know it's Rust.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tower_http::cors::CorsLayer;

use crate::db;
use crate::core::{CognitiveMode, SelfModel};
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

    let app = Router::new()
        // RGW native endpoints
        .route("/health", get(health))
        .route("/walk", post(walk))
        .route("/stats", get(stats))
        .route("/benchmark", get(benchmark))
        .route("/diverger", get(diverger_stats))
        .route("/prune", post(prune))
        .route("/edge", post(create_edge_endpoint))
        .route("/diverger/notify", post(diverger_notify))
        .route("/self", get(self_state))
        .route("/self/mode", post(set_mode_endpoint))
        .route("/self/save", post(save_self_model_endpoint))
        .route("/tools", get(list_tools))
        .route("/tools/execute", post(execute_tool_endpoint))
        .route("/speak", post(speak_endpoint))
        .route("/dream", post(dream_endpoint))
        .route("/music", get(music_endpoint))
        .route("/music/midi", get(music_midi_endpoint))
        .route("/music/generate", post(music_generate_endpoint))
        .route("/music/prompt", get(music_prompt_endpoint))
        // Signal ingest — external events feed into the cognitive loop
        .route("/ingest/signal", post(ingest_signal_endpoint))
        // OpenAI-compatible endpoints (drop-in replacement for Ollama)
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
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

        // Optionally escalate to LLM if diagnosis warrants it
        if diagnosis.algorithmic_adjustments.escalate_to_llm || sm.critic_sessions_since_llm > 10 {
            if let Some(ref provider) = state.provider {
                let summary = crate::metacog::build_session_summary(&output, &walker_results, &sm);
                let (system, user) = crate::metacog::build_critic_llm_prompt(&diagnosis, &summary);

                let result = provider.generate(&system, &user, 0.5, Some(256), 0.3).await;
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

/// POST /self/save — persist self-model to database
async fn save_self_model_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let sm = state.self_model.lock().await.clone();
    match db::save_self_model(&state.pool, &sm).await {
        Ok(_) => Json(serde_json::json!({"status": "saved"})),
        Err(e) => Json(serde_json::json!({"status": "error", "error": e.to_string()})),
    }
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
            &format!("{}: {}", result.tool, &result.content[..result.content.len().min(100)]),
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

    tracing::info!("[music] Generating via MusicGen: '{}'", &prompt[..prompt.len().min(80)]);

    match crate::music::generate_musicgen(&prompt, req.duration_secs).await {
        Ok(path) => {
            // Notify self-model: Julian created music
            {
                let mut sm = state.self_model.lock().await;
                let signal = crate::core::Signal::new("music_created", &format!(
                    "Composed: {}", &prompt[..prompt.len().min(60)]
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
    let mut sm = state.self_model.lock().await;

    let signal = crate::core::Signal::new(&req.kind, &req.content)
        .with_domain(if req.domain.is_empty() { &req.kind } else { &req.domain })
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
