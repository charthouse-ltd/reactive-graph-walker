//! LLM Provider — unified interface for local + cloud models.
//!
//! Routes between:
//!   - Local: Qwen via embedded llama.cpp (fast, free, in-process)
//!   - Cloud: Gemini 3 Pro via API (powerful, costs money, network hop)
//!
//! The graph walker decides task complexity. Simple expression → local.
//! Complex reasoning, tool use, multimodal → cloud.

use serde::{Deserialize, Serialize};

use crate::llm::LlmEngine;

/// Provider configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// Path to local GGUF model
    pub local_model_path: Option<String>,
    /// GPU layers for local model (0 = CPU only)
    pub local_gpu_layers: u32,
    /// Context window for local model
    pub local_ctx_size: u32,

    /// Cloud model configurations (multiple providers)
    pub cloud_models: Vec<CloudModelConfig>,

    /// Routing: complexity threshold (0-1). Above this → cloud.
    pub cloud_threshold: f32,
}

/// Configuration for a cloud LLM provider
#[derive(Debug, Clone, Deserialize)]
pub struct CloudModelConfig {
    /// Provider name: "gemini", "anthropic", "openai", "groq"
    pub provider: String,
    /// Model name (e.g. "gemini-3-pro", "claude-sonnet-4-6", "gpt-4o")
    pub model: String,
    /// API key
    pub api_key: String,
    /// API base URL (provider-specific)
    pub api_url: Option<String>,
    /// What this model is best at: "reasoning", "creative", "fast", "code"
    pub strength: String,
    /// Cost per 1M tokens (for routing decisions)
    pub cost_per_million: f32,
    /// Priority (lower = preferred when strengths match)
    pub priority: u8,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            local_model_path: None,
            local_gpu_layers: 99,
            local_ctx_size: 4096,
            cloud_models: Vec::new(),
            cloud_threshold: 0.7,
        }
    }
}

/// Unified LLM provider
pub struct Provider {
    local: Option<LlmEngine>,
    /// DeepSeek engine, wired from DEEPSEEK_API_KEY when present. This is the
    /// primary path: one engine that routes between chat and reasoner tiers.
    deepseek: Option<LlmEngine>,
    /// Model name for the cheap/fast tier (default "deepseek-chat").
    chat_model: String,
    /// Model name for the expensive reasoning tier (default "deepseek-reasoner").
    reasoner_model: String,
    config: ProviderConfig,
}

/// Which model handled the request
#[derive(Debug, Serialize)]
pub enum ModelUsed {
    Local(String),
    Cloud(String),
    Fallback,
}

/// Generation result
#[derive(Debug, Serialize)]
pub struct GenerateResult {
    pub text: String,
    pub model_used: ModelUsed,
    pub tokens_generated: u32,
    pub elapsed_ms: f64,
}

impl Provider {
    /// Initialize the provider — load local model if path given
    pub fn new(config: ProviderConfig) -> anyhow::Result<Self> {
        let local = if let Some(ref path) = config.local_model_path {
            match LlmEngine::load(path, config.local_gpu_layers, config.local_ctx_size) {
                Ok(engine) => {
                    tracing::info!("[provider] Local model loaded: {}", path);
                    Some(engine)
                }
                Err(e) => {
                    tracing::warn!("[provider] Local model failed to load: {}. Cloud-only mode.", e);
                    None
                }
            }
        } else {
            tracing::info!("[provider] No local model configured. Cloud-only mode.");
            None
        };

        // Wire DeepSeek on by default from the environment. This is the
        // primary generation path; chat vs reasoner is chosen per call.
        let chat_model =
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
        let reasoner_model = std::env::var("DEEPSEEK_REASONER_MODEL")
            .unwrap_or_else(|_| "deepseek-reasoner".into());
        let deepseek = match std::env::var("DEEPSEEK_API_KEY") {
            Ok(key) if !key.is_empty() => match LlmEngine::new(&key, &chat_model) {
                Ok(engine) => {
                    tracing::info!(
                        "[provider] DeepSeek wired: chat={} reasoner={} (routed by difficulty)",
                        chat_model, reasoner_model
                    );
                    Some(engine)
                }
                Err(e) => {
                    tracing::warn!("[provider] DeepSeek init failed: {}. Falling back.", e);
                    None
                }
            },
            _ => {
                tracing::info!("[provider] DEEPSEEK_API_KEY not set — DeepSeek path disabled.");
                None
            }
        };

        Ok(Self { local, deepseek, chat_model, reasoner_model, config })
    }

    /// Generate text — routes between local and cloud based on complexity.
    ///
    /// complexity: 0.0 = trivial (journal entry, short response)
    ///             1.0 = hard (blog post, analysis, multi-step reasoning)
    /// difficult:  hard signal from metacog (stuck / break-loop) that forces
    ///             reasoner escalation regardless of the complexity score.
    pub async fn generate(
        &self,
        system: &str,
        user: &str,
        complexity: f32,
        difficult: bool,
        max_tokens: Option<u32>,
        temperature: f32,
    ) -> GenerateResult {
        let start = std::time::Instant::now();

        // ── Token budget: graph context (carried in `system`) and the user
        // stimulus can both grow unbounded. Cap each before they hit the wire.
        let system = crate::budget::budget_prompt(system);
        let user = crate::budget::budget_prompt(user);

        // ── Cost circuit breaker: hard stop when a spend ceiling is hit ──
        // Dedup/cadence reduce calls but don't cap absolute spend. This does:
        // above RGW_MAX_CALLS_PER_MIN or RGW_MAX_DAILY_USD, skip the paid call
        // entirely and return empty (callers fall back to Ollama / walker text).
        if let Err(reason) = crate::metrics::spend_allows() {
            crate::metrics::record_throttle();
            tracing::warn!("[provider] LLM call throttled ({}); returning empty.", reason);
            return GenerateResult {
                text: String::new(),
                model_used: ModelUsed::Fallback,
                tokens_generated: 0,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        // ── DeepSeek primary path: route, dedup, call, account ──
        if let Some(ref ds) = self.deepseek {
            let model = if crate::budget::wants_reasoner(complexity, difficult) {
                self.reasoner_model.as_str()
            } else {
                self.chat_model.as_str()
            };

            // Idempotency: identical cognition states hash equal — reuse the
            // cached completion instead of paying for the same call again.
            let hash = crate::budget::prompt_hash(model, &system, &user, max_tokens, temperature);
            if let Some(cached) = crate::budget::dedup_lookup(hash) {
                crate::metrics::record_dedup_hit();
                tracing::info!("[provider] DeepSeek dedup hit (model={}) — call skipped", model);
                return GenerateResult {
                    tokens_generated: crate::budget::approx_tokens(&cached) as u32,
                    text: cached,
                    model_used: ModelUsed::Cloud(format!("deepseek:{} (dedup)", model)),
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                };
            }

            match ds.chat_with_model(model, &system, &user, max_tokens, temperature).await {
                Ok((text, usage)) => {
                    crate::budget::dedup_store(hash, &text);
                    tracing::info!(
                        "[provider] DeepSeek {} (complexity={:.2} difficult={})",
                        model, complexity, difficult
                    );
                    return GenerateResult {
                        tokens_generated: usage.completion_tokens,
                        text,
                        model_used: ModelUsed::Cloud(format!("deepseek:{}", model)),
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    };
                }
                Err(e) => {
                    tracing::warn!("[provider] DeepSeek failed: {}. Trying fallbacks.", e);
                }
            }
        }

        let system = system.as_str();
        let user = user.as_str();

        // Route: simple → local, complex → cloud
        if complexity < self.config.cloud_threshold {
            // Try local first
            if let Some(ref llm) = self.local {
                match llm.chat(system, user, max_tokens, temperature).await {
                    Ok(text) => {
                        return GenerateResult {
                            tokens_generated: text.split_whitespace().count() as u32,
                            text,
                            model_used: ModelUsed::Local(llm.model_name().into()),
                            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        };
                    }
                    Err(e) => {
                        tracing::warn!("[provider] Local generation failed: {}. Falling back to cloud.", e);
                    }
                }
            }
        }

        // Cloud: try models in priority order, matching strength to task
        let task_strength = if complexity > 0.9 { "reasoning" }
            else if complexity > 0.7 { "creative" }
            else { "fast" };

        let mut models: Vec<&CloudModelConfig> = self.config.cloud_models.iter().collect();
        // Sort: matching strength first, then by priority, then by cost
        models.sort_by(|a, b| {
            let a_match = (a.strength == task_strength) as u8;
            let b_match = (b.strength == task_strength) as u8;
            b_match.cmp(&a_match)
                .then(a.priority.cmp(&b.priority))
                .then(a.cost_per_million.partial_cmp(&b.cost_per_million).unwrap_or(std::cmp::Ordering::Equal))
        });

        for model_cfg in &models {
            let result = match model_cfg.provider.as_str() {
                "gemini" => generate_gemini(&model_cfg.api_key, &model_cfg.model, system, user, max_tokens, temperature).await,
                "anthropic" => generate_anthropic(&model_cfg.api_key, &model_cfg.model, system, user, max_tokens, temperature).await,
                "openai" => generate_openai_compat(
                    model_cfg.api_url.as_deref().unwrap_or("https://api.openai.com"),
                    &model_cfg.api_key, &model_cfg.model, system, user, max_tokens, temperature,
                ).await,
                "groq" => generate_openai_compat(
                    model_cfg.api_url.as_deref().unwrap_or("https://api.groq.com/openai"),
                    &model_cfg.api_key, &model_cfg.model, system, user, max_tokens, temperature,
                ).await,
                "deepseek" => generate_openai_compat(
                    model_cfg.api_url.as_deref().unwrap_or("https://api.deepseek.com"),
                    &model_cfg.api_key, &model_cfg.model, system, user, max_tokens, temperature,
                ).await,
                other => {
                    tracing::warn!("[provider] Unknown provider: {}", other);
                    continue;
                }
            };

            match result {
                Ok(text) => {
                    tracing::info!(
                        "[provider] Cloud: {}:{} (strength={}, task={})",
                        model_cfg.provider, model_cfg.model, model_cfg.strength, task_strength
                    );
                    return GenerateResult {
                        tokens_generated: text.split_whitespace().count() as u32,
                        text,
                        model_used: ModelUsed::Cloud(format!("{}:{}", model_cfg.provider, model_cfg.model)),
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    };
                }
                Err(e) => {
                    tracing::warn!("[provider] {}:{} failed: {}. Trying next.", model_cfg.provider, model_cfg.model, e);
                    continue;
                }
            }
        }

        // Fallback: local even if complex (better than nothing)
        if let Some(ref llm) = self.local {
            if let Ok(text) = llm.chat(system, user, max_tokens, temperature).await {
                return GenerateResult {
                    tokens_generated: text.split_whitespace().count() as u32,
                    text,
                    model_used: ModelUsed::Fallback,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                };
            }
        }

        GenerateResult {
            text: String::new(),
            model_used: ModelUsed::Fallback,
            tokens_generated: 0,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

/// Simple per-provider rate limiter
struct RateLimiter {
    calls: std::sync::Mutex<Vec<f64>>,  // timestamps of recent calls
    max_per_minute: u32,
}

impl RateLimiter {
    fn new(max_per_minute: u32) -> Self {
        Self {
            calls: std::sync::Mutex::new(Vec::new()),
            max_per_minute,
        }
    }

    fn try_acquire(&self) -> bool {
        let mut calls = self.calls.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Remove calls older than 60s
        calls.retain(|&t| now - t < 60.0);

        if calls.len() < self.max_per_minute as usize {
            calls.push(now);
            true
        } else {
            false  // Rate limited
        }
    }
}

// Global rate limiters per provider
lazy_static::lazy_static! {
    static ref RATE_LIMITERS: std::sync::Mutex<std::collections::HashMap<String, RateLimiter>> =
        std::sync::Mutex::new(std::collections::HashMap::new());
}

fn check_rate_limit(provider: &str, model: &str, rpm: u32) -> bool {
    let key = format!("{}:{}", provider, model);
    let mut limiters = RATE_LIMITERS.lock().unwrap();
    let limiter = limiters.entry(key).or_insert_with(|| RateLimiter::new(rpm));
    limiter.try_acquire()
}

/// Call Anthropic Claude API
async fn generate_anthropic(
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: Option<u32>,
    temperature: f32,
) -> Result<String, String> {
    if !check_rate_limit("anthropic", model, 50) {
        return Err("Rate limited".into());
    }

    let body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens.unwrap_or(1024),
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {}", e))?;

    if !resp.status().is_success() {
        crate::metrics::record_failure();
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Anthropic {} — {}", status, &body[..200.min(body.len())]));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| {
        crate::metrics::record_failure();
        format!("Parse failed: {}", e)
    })?;
    let text = data["content"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            crate::metrics::record_failure();
            "No text in Anthropic response".to_string()
        })?;
    let pt = data["usage"]["input_tokens"].as_u64()
        .unwrap_or_else(|| (crate::budget::approx_tokens(system) + crate::budget::approx_tokens(user)) as u64);
    let ct = data["usage"]["output_tokens"].as_u64()
        .unwrap_or_else(|| crate::budget::approx_tokens(&text) as u64);
    crate::metrics::record_call(model, pt, ct);
    Ok(text)
}

/// Call any OpenAI-compatible API (OpenAI, Groq, Together, etc.)
async fn generate_openai_compat(
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: Option<u32>,
    temperature: f32,
) -> Result<String, String> {
    if !check_rate_limit("openai-compat", model, 50) {
        return Err("Rate limited".into());
    }

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens.unwrap_or(1024),
        "temperature": temperature,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("OpenAI-compat request failed: {}", e))?;

    if !resp.status().is_success() {
        crate::metrics::record_failure();
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI-compat {} — {}", status, &body[..200.min(body.len())]));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| {
        crate::metrics::record_failure();
        format!("Parse failed: {}", e)
    })?;
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            crate::metrics::record_failure();
            "No content in response".to_string()
        })?;
    let pt = data["usage"]["prompt_tokens"].as_u64()
        .unwrap_or_else(|| (crate::budget::approx_tokens(system) + crate::budget::approx_tokens(user)) as u64);
    let ct = data["usage"]["completion_tokens"].as_u64()
        .unwrap_or_else(|| crate::budget::approx_tokens(&text) as u64);
    crate::metrics::record_call(model, pt, ct);
    Ok(text)
}

/// Call Gemini API directly
async fn generate_gemini(
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: Option<u32>,
    temperature: f32,
) -> Result<String, String> {
    if !check_rate_limit("gemini", model, 60) {
        return Err("Rate limited".into());
    }

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let body = serde_json::json!({
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [{
            "parts": [{"text": user}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens.unwrap_or(1024),
        }
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("Gemini request failed: {}", e))?;

    if !resp.status().is_success() {
        crate::metrics::record_failure();
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini {} — {}", status, &body[..200.min(body.len())]));
    }

    let data: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| {
            crate::metrics::record_failure();
            format!("Gemini parse failed: {}", e)
        })?;

    let text = data["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            crate::metrics::record_failure();
            "No text in Gemini response".to_string()
        })?;
    let pt = data["usageMetadata"]["promptTokenCount"].as_u64()
        .unwrap_or_else(|| (crate::budget::approx_tokens(system) + crate::budget::approx_tokens(user)) as u64);
    let ct = data["usageMetadata"]["candidatesTokenCount"].as_u64()
        .unwrap_or_else(|| crate::budget::approx_tokens(&text) as u64);
    crate::metrics::record_call(model, pt, ct);
    Ok(text)
}
