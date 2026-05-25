//! DeepSeek LLM — direct HTTP integration.
//!
//! Talks to DeepSeek's API (OpenAI-compatible) over HTTPS.
//! No Ollama proxy needed. No llama.cpp linking issues.
//! The graph walker produces cognition in memory → DeepSeek
//! reads the enriched prompt → text comes out.
//!
//! DeepSeek models:
//!   - deepseek-chat     — general purpose, fast, cheap
//!   - deepseek-reasoner — chain-of-thought reasoning, slower, smarter
//!
//! API: POST https://api.deepseek.com/v1/chat/completions
//! Auth: Bearer $DEEPSEEK_API_KEY

use serde::{Deserialize, Serialize};

/// The DeepSeek LLM engine — a real HTTP client.
pub struct LlmEngine {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

/// DeepSeek chat message
#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// DeepSeek request body (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    temperature: f32,
    stream: bool,
}

/// DeepSeek response
#[derive(Debug, Deserialize)]
struct DeepSeekResponse {
    choices: Vec<DeepSeekChoice>,
    /// Token accounting returned by DeepSeek. Captured for cost metrics.
    #[serde(default)]
    usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Deserialize)]
struct DeepSeekChoice {
    message: DeepSeekMessage,
}

#[derive(Debug, Deserialize)]
struct DeepSeekMessage {
    content: String,
}

/// The `usage` object DeepSeek returns (OpenAI-compatible shape).
#[derive(Debug, Deserialize, Default)]
struct DeepSeekUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Token usage for one completion — real if the API reported it,
/// otherwise a char/4 estimate so cost accounting is never blind.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl LlmEngine {
    /// Create a new DeepSeek engine.
    ///
    /// api_key: DeepSeek API key (env: DEEPSEEK_API_KEY)
    /// model: "deepseek-chat" or "deepseek-reasoner"
    pub fn new(api_key: &str, model: &str) -> anyhow::Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("DEEPSEEK_API_KEY not set");
        }

        let engine = Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            client: reqwest::Client::new(),
        };

        tracing::info!("[llm] DeepSeek engine ready: model={}", model);
        Ok(engine)
    }

    /// Backward-compatible constructor (ignores legacy llama.cpp params).
    /// Called from provider.rs with the old signature.
    pub fn load(model_path: &str, _n_gpu_layers: u32, _n_ctx: u32) -> anyhow::Result<Self> {
        // model_path is repurposed: if it looks like a DeepSeek model name, use it.
        // Otherwise fall back to DEEPSEEK_MODEL env or default.
        let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            anyhow::bail!("DEEPSEEK_API_KEY not set — cannot initialize DeepSeek engine");
        }

        let model = if model_path.contains("deepseek") {
            model_path.to_string()
        } else {
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".into())
        };

        Self::new(&api_key, &model)
    }

    /// Send a chat completion request to DeepSeek.
    ///
    /// This is the main entry point. Called by provider.rs when routing
    /// decides to use the local (now DeepSeek) path for generation.
    pub async fn chat(
        &self,
        system: &str,
        user: &str,
        max_tokens: Option<u32>,
        temperature: f32,
    ) -> anyhow::Result<String> {
        let model = self.model.clone();
        let (text, _usage) = self
            .chat_with_model(&model, system, user, max_tokens, temperature)
            .await?;
        Ok(text)
    }

    /// Send a completion using an explicit model name (e.g. "deepseek-chat"
    /// or "deepseek-reasoner"), returning both the text and the REAL token
    /// usage. One engine instance can therefore serve both routing tiers.
    ///
    /// Records the call into the global cost metrics on success.
    pub async fn chat_with_model(
        &self,
        model: &str,
        system: &str,
        user: &str,
        max_tokens: Option<u32>,
        temperature: f32,
    ) -> anyhow::Result<(String, ChatUsage)> {
        let messages = vec![
            ChatMessage { role: "system".into(), content: system.to_string() },
            ChatMessage { role: "user".into(), content: user.to_string() },
        ];

        let body = DeepSeekRequest {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
            stream: false,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        tracing::info!("[llm] POST {} model={} max_tokens={:?} temp={:.2} ({} chars user)",
            url, model, max_tokens, temperature, user.len());

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .inspect_err(|_| crate::metrics::record_failure())?;

        tracing::info!("[llm] DeepSeek response: HTTP {} ({} bytes)",
            resp.status().as_u16(), resp.content_length().unwrap_or(0));

        if !resp.status().is_success() {
            crate::metrics::record_failure();
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let preview: String = text.chars().take(300).collect();
            anyhow::bail!("DeepSeek {} — {}", status, preview);
        }

        let data: DeepSeekResponse = resp.json().await.inspect_err(|_| crate::metrics::record_failure())?;
        let reported = data.usage.unwrap_or_default();
        let content = data
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        if content.is_empty() {
            crate::metrics::record_failure();
            anyhow::bail!("DeepSeek returned empty response");
        }

        // Prefer the API's real token counts; fall back to a char/4 estimate
        // so cost accounting is never blind even if `usage` is absent.
        let usage = ChatUsage {
            prompt_tokens: if reported.prompt_tokens > 0 {
                reported.prompt_tokens
            } else {
                (crate::budget::approx_tokens(system) + crate::budget::approx_tokens(user)) as u32
            },
            completion_tokens: if reported.completion_tokens > 0 {
                reported.completion_tokens
            } else {
                crate::budget::approx_tokens(&content) as u32
            },
        };

        crate::metrics::record_call(model, usage.prompt_tokens as u64, usage.completion_tokens as u64);

        Ok((content, usage))
    }

    /// Return the model name (for logging/routing visibility)
    pub fn model_name(&self) -> &str {
        &self.model
    }
}
