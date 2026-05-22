//! Motor Cortex — RGW commands the execution backend to act.
//!
//! When the Diverger fires a walk and the result says "express" or
//! "explore" or "search", the motor cortex translates that into an
//! HTTP command to the Python backend API.
//!
//! Rust brain → Python body. The brain decides, the body executes.
//!
//! The `params` field carries action-specific payloads, enabling
//! multimodal commands (image prompts, social targets, blog metadata)
//! without changing the MotorCommand struct for each new capability.

use serde::Serialize;

/// Command that RGW sends to the execution backend.
///
/// Actions: "tweet", "blog", "journal", "search", "explore",
///          "image_generate", "post_social", "email", "nothing"
///
/// The `params` field carries action-specific structured data:
///   - image_generate: {"prompt": "...", "style": "...", "size": "1024x1024", "provider": "comfyui"}
///   - post_social:    {"platform": "twitter", "text": "...", "image_url": "...", "schedule_at": "..."}
///   - blog:           {"title": "...", "body_md": "...", "tags": [...], "publish": true}
///   - tweet:          {"text": "...", "reply_to": "...", "media_ids": [...]}
///   - email:          {"to": "...", "subject": "...", "body": "..."}
///
/// The `audience_id` field enables theory of mind: Julian records
/// what was said and to whom, building an audience model over time.
#[derive(Debug, Clone, Serialize)]
pub struct MotorCommand {
    /// What to do
    pub action: String,
    /// Which domain this relates to
    pub domain: String,
    /// Walker context (enriched prompt for the LLM)
    pub walker_context: String,
    /// Expression seeds (memory fragments to express)
    pub expression_seeds: Vec<serde_json::Value>,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Novelty (0-1)
    pub novelty: f32,
    /// Search query (if action = "search")
    pub search_query: Option<String>,
    /// Action-specific parameters (flexible JSON payload for multimodal commands)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Who is the audience for this output? (Theory of Mind)
    /// e.g. "twitter:followers", "user:alice", "dm:bob"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience_id: Option<String>,
}

/// Send a motor command to Julian's Python backend.
/// Fire-and-forget: never blocks the Diverger.
pub async fn execute(julian_url: &str, command: MotorCommand) {
    if julian_url.is_empty() {
        return;
    }

    let url = format!("{}/api/admin/rgw/execute", julian_url);

    match reqwest::Client::new()
        .post(&url)
        .json(&command)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                tracing::info!(
                    "[motor] Sent: {} (domain={}, confidence={:.0}%)",
                    command.action, command.domain, command.confidence * 100.0
                );
            } else {
                tracing::warn!("[motor] Julian returned {}", resp.status());
            }
        }
        Err(e) => {
            tracing::debug!("[motor] Failed to reach Julian: {}", e);
        }
    }
}
