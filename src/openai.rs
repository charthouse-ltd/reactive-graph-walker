//! OpenAI-compatible API layer.
//!
//! Makes RGW a drop-in replacement for Ollama/OpenAI. Julian's code
//! calls POST /v1/chat/completions and gets a response. Internally:
//! 1. Parse the prompt → extract stimulus
//! 2. Graph walk (parallel, emotionally biased)
//! 3. Forward to Qwen via Ollama for text expression
//! 4. Return in OpenAI format
//!
//! From outside: looks like any LLM.
//! Inside: graph traversal + LLM expression.

use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::graph::EmotionalState;
use crate::walker::{self, format_walk_context};

// ── OpenAI-compatible request/response types ────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    // RGW extensions
    #[serde(default)]
    pub rgw_emotion: Option<EmotionalState>,
    #[serde(default = "default_walkers")]
    pub rgw_walkers: usize,
    #[serde(default = "default_steps")]
    pub rgw_steps: usize,
    /// Per-request cognitive mode override ("autonomous" or "compliant")
    #[serde(default)]
    pub rgw_mode: Option<String>,
}

fn default_temp() -> f32 { 0.7 }
fn default_walkers() -> usize { 4 }
fn default_steps() -> usize { 5 }

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    // RGW metadata (non-standard but useful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgw_metadata: Option<RgwMetadata>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct RgwMetadata {
    pub walker_action: String,
    pub walker_domain: String,
    pub walker_agreement: f32,
    pub walker_novelty: f32,
    pub walk_ms: f64,
    pub expression_ms: f64,
    pub total_ms: f64,
}

// ── Models list (Ollama compatibility) ──────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// GET /v1/models — list available models (Ollama compat)
pub async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo {
            id: "rgw".into(),
            object: "model".into(),
            created: 0,
            owned_by: "diverger".into(),
        }],
    })
}

// ── Main chat completions endpoint ──────────────────────────────

/// POST /v1/chat/completions — the main endpoint.
/// RGW thinks via graph walk, then forwards to Qwen for expression.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let start = Instant::now();

    // 1. Extract stimulus from the last user message
    let stimulus = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // Extract system prompt (forwarded to Qwen)
    let system_prompt = req
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // 2. Graph walk — RGW thinks
    let emotion = req.rgw_emotion.unwrap_or_default();
    let walk_start = Instant::now();

    // Per-request mode override (extract first, lock later)
    let mode_target = req.rgw_mode.as_ref().and_then(|m| match m.to_lowercase().as_str() {
        "autonomous" => Some(crate::core::CognitiveMode::Autonomous),
        "compliant" => Some(crate::core::CognitiveMode::Compliant),
        _ => None,
    });
    let prev_mode = if let Some(target) = mode_target {
        let mut sm = state.self_model.lock().await;
        let prev = sm.mode.clone();
        sm.mode = target;
        Some(prev)
    } else {
        None
    };

    let (walk_output, _walker_results) = walker::walk_parallel(
        &state.pool,
        &emotion,
        req.rgw_walkers,
        req.rgw_steps,
        &state.self_model,
    )
    .await;

    // Restore mode after walk
    if let Some(prev) = prev_mode {
        let mut sm = state.self_model.lock().await;
        sm.mode = prev;
    }

    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Build enriched prompt for Qwen
    let walker_context = format_walk_context(&walk_output);
    let enriched_prompt = format!(
        "{}\n\n{}\n\nUser: {}",
        walker_context,
        if system_prompt.is_empty() { "" } else { &system_prompt },
        stimulus,
    );

    // 4. Generate text — route through provider or fall back to Ollama
    //
    // Complexity is derived from walker output:
    //   - High novelty + low agreement = complex (needs strong model)
    //   - Low novelty + high agreement = simple (cheap/local model)
    let complexity = (walk_output.novelty_score * 0.6 + (1.0 - walk_output.agreement_score) * 0.4)
        .clamp(0.0, 1.0);

    let expr_start = Instant::now();
    let text = if let Some(ref provider) = state.provider {
        // Intelligent routing: complexity determines local vs cloud
        let system_with_context = if system_prompt.is_empty() {
            walker_context.clone()
        } else {
            format!("{}\n\n{}", system_prompt, walker_context)
        };
        // No external difficulty signal here; let the complexity score (derived
        // from walker novelty/disagreement) decide chat vs reasoner via its threshold.
        let result = provider.generate(
            &system_with_context,
            &stimulus,
            complexity,
            false,
            req.max_tokens,
            req.temperature,
        ).await;

        if !result.text.is_empty() {
            tracing::info!(
                "[rgw] Provider routed: {:?} (complexity={:.2})",
                result.model_used, complexity
            );
            result.text
        } else {
            // Provider returned nothing — fall back to Ollama
            tracing::warn!("[rgw] Provider returned empty, falling back to Ollama");
            forward_to_ollama(
                &state.ollama_url, &state.expression_model,
                &req.messages, &walker_context, req.temperature, req.max_tokens,
            ).await.unwrap_or_else(|e| {
                tracing::warn!("Ollama fallback also failed: {}", e);
                walker_context.clone()
            })
        }
    } else {
        // No provider configured — direct Ollama (backward compatible)
        forward_to_ollama(
            &state.ollama_url, &state.expression_model,
            &req.messages, &walker_context, req.temperature, req.max_tokens,
        ).await.unwrap_or_else(|e| {
            tracing::warn!("Ollama forward failed: {}. Returning walker context as fallback.", e);
            walker_context.clone()
        })
    };
    let expr_ms = expr_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Notify diverger about the walk (edges were changed)
    // The walk itself already strengthened edges — diverger will cascade

    tracing::info!(
        "[rgw] Chat: walk={:.0}ms expr={:.0}ms total={:.0}ms domain={} action={}",
        walk_ms, expr_ms, total_ms,
        walk_output.primary_domain,
        walk_output.recommended_action,
    );

    // 5. Return OpenAI-formatted response
    let response = ChatResponse {
        id: format!("rgw-{}", uuid_simple()),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "rgw".into(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: stimulus.len() as u32 / 4,  // Rough estimate
            completion_tokens: 0,  // Filled by Qwen
            total_tokens: 0,
        },
        rgw_metadata: Some(RgwMetadata {
            walker_action: walk_output.recommended_action.clone(),
            walker_domain: walk_output.primary_domain.clone(),
            walker_agreement: walk_output.agreement_score,
            walker_novelty: walk_output.novelty_score,
            walk_ms,
            expression_ms: expr_ms,
            total_ms,
        }),
    };

    Ok(Json(response))
}

// ── Ollama forwarding ───────────────────────────────────────────

/// Forward the enriched prompt to Ollama (Qwen) for text generation.
/// RGW did the thinking. Qwen does the writing.
async fn forward_to_ollama(
    ollama_url: &str,
    model: &str,
    original_messages: &[Message],
    walker_context: &str,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    // Build messages: inject walker context into system message
    let mut messages = Vec::new();

    // Find original system message and enrich it
    let has_system = original_messages.iter().any(|m| m.role == "system");
    if has_system {
        for msg in original_messages {
            if msg.role == "system" {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": format!("{}\n\n{}", msg.content, walker_context),
                }));
            } else {
                messages.push(serde_json::json!({
                    "role": msg.role,
                    "content": msg.content,
                }));
            }
        }
    } else {
        // No system message — add walker context as system
        messages.push(serde_json::json!({
            "role": "system",
            "content": walker_context,
        }));
        for msg in original_messages {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content,
            }));
        }
    }

    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens.unwrap_or(1024),
        "stream": false,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", ollama_url))
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Ollama returned {}", resp.status()));
    }

    let data: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Ollama response parse failed: {}", e))?;

    data["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No content in Ollama response".into())
}

// ── Helpers ─────────────────────────────────────────────────────

fn uuid_simple() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    format!("{:016x}", rng.random::<u64>())
}
