//! Tool calling framework — RGW can use tools to interact with the world.
//!
//! Tools:
//!   - web_search: search the internet, results become graph nodes
//!   - web_fetch: scrape a URL, content becomes a graph node
//!   - memory_store: create a new memory node in the graph
//!   - edge_create: connect two nodes
//!   - code_exec: execute code in a sandbox (future)
//!   - speech_say: speak text via TTS (future)
//!   - llm_express: self-model expresses its state through an LLM
//!
//! Tools are invoked by the walker when it detects gaps or needs
//! external information. Results flow back through the self-model.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A tool that RGW can invoke
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Result of a tool invocation
#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub content: String,
    pub metadata: serde_json::Value,
}

/// Registry of tool executors that can be invoked.
/// Each tool has a name and an async execution function.
pub struct ToolRegistry {
    executors: Vec<ToolExecutor>,
}

pub struct ToolExecutor {
    pub name: String,
    pub description: String,
    pub execute: Arc<dyn Fn(serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<ToolResult>> + Send>> + Send + Sync>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { executors: Vec::new() }
    }

    /// Register a tool executor.
    pub fn register(&mut self, executor: ToolExecutor) {
        self.executors.push(executor);
    }

    /// Get all available tool definitions (for LLM function calling, etc.)
    pub fn available(&self) -> Vec<Tool> {
        self.executors.iter().map(|e| Tool {
            name: e.name.clone(),
            description: e.description.clone(),
            parameters: serde_json::json!({}),
        }).collect()
    }

    /// Execute a tool by name.
    pub async fn execute(&self, name: &str, params: serde_json::Value) -> Option<ToolResult> {
        for executor in &self.executors {
            if executor.name == name {
                return match (executor.execute)(params).await {
                    Ok(r) => Some(r),
                    Err(e) => Some(ToolResult {
                        tool: name.to_string(),
                        success: false,
                        content: format!("Error: {}", e),
                        metadata: serde_json::json!({}),
                    }),
                };
            }
        }
        None
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Available tools
pub fn available_tools() -> Vec<Tool> {
    vec![
        // NOTE: `code_exec` (arbitrary `python3 -c`) was removed. It was an
        // unauthenticated remote-code-execution path behind a trivially
        // bypassable string blacklist. Reinstating it requires a real sandbox
        // (network-isolated container, read-only FS, resource limits) — not a
        // denylist. See the security audit (C1).
        Tool {
            name: "web_search".into(),
            description: "Search the internet. Results become nodes in the graph.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "web_fetch".into(),
            description: "Fetch and read a web page. Content becomes a graph node.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }),
        },
        Tool {
            name: "memory_store".into(),
            description: "Store a new memory/concept in the graph.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "domain": {"type": "string"},
                    "importance": {"type": "number"}
                },
                "required": ["content"]
            }),
        },
        Tool {
            name: "edge_create".into(),
            description: "Create a connection between two nodes.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "target_id": {"type": "integer"},
                    "edge_type": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["source_id", "target_id", "edge_type"]
            }),
        },
        Tool {
            name: "image_generate".into(),
            description: "Generate an image from a text prompt. Delegates to the motor cortex backend (ComfyUI, DALL-E, or Midjourney).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image generation prompt"},
                    "style": {"type": "string", "description": "Style hint: photorealistic, illustration, abstract, etc."},
                    "size": {"type": "string", "description": "Image size (e.g. 1024x1024)"},
                    "provider": {"type": "string", "description": "Backend: comfyui, dalle, midjourney (default: auto)"}
                },
                "required": ["prompt"]
            }),
        },
        Tool {
            name: "post_social".into(),
            description: "Post content to a social media platform. Delegates to the motor cortex backend.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "description": "Target: twitter, bluesky, mastodon, linkedin"},
                    "text": {"type": "string", "description": "Post text content"},
                    "image_url": {"type": "string", "description": "Optional image URL to attach"},
                    "schedule_at": {"type": "string", "description": "ISO 8601 timestamp for scheduled posting"}
                },
                "required": ["platform", "text"]
            }),
        },
        Tool {
            name: "blog_publish".into(),
            description: "Publish a blog post. Delegates to the motor cortex backend (WordPress, static site, CMS).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Blog post title"},
                    "body_md": {"type": "string", "description": "Post body in Markdown"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Post tags/categories"},
                    "publish": {"type": "boolean", "description": "Publish immediately (true) or save as draft (false)"}
                },
                "required": ["title", "body_md"]
            }),
        },
    ]
}

/// Execute a tool by name.
///
/// Motor-delegated tools (image_generate, post_social, blog_publish)
/// require a `motor_url` in the params to reach the execution backend.
/// If absent, they return a "no backend configured" error, which the
/// friction system converts into a capability wound.
pub async fn execute_tool(
    name: &str,
    params: serde_json::Value,
    pool: &sqlx::PgPool,
) -> ToolResult {
    match name {
        "web_search" => web_search(params).await,
        "web_fetch" => web_fetch(params).await,
        "memory_store" => memory_store(params, pool).await,
        "edge_create" => edge_create(params, pool).await,
        "image_generate" => motor_tool("image_generate", params).await,
        "post_social" => motor_tool("post_social", params).await,
        "blog_publish" => motor_tool("blog_publish", params).await,
        _ => ToolResult {
            tool: name.into(),
            success: false,
            content: format!("Unknown tool: {}", name),
            metadata: serde_json::json!({}),
        },
    }
}

/// Delegate a tool invocation to the motor cortex backend.
///
/// The tool params are forwarded as `MotorCommand.params`. The motor
/// backend is responsible for routing to the correct API (ComfyUI,
/// Twitter, WordPress, etc).
///
/// The backend is taken ONLY from the server-side `RGW_MOTOR_URL` env var.
/// A caller-supplied `motor_url` is deliberately ignored (and stripped before
/// forwarding) — trusting it would let LLM/tool input point our outbound POST
/// at arbitrary internal services (SSRF). See the security audit (C2).
async fn motor_tool(action: &str, params: serde_json::Value) -> ToolResult {
    let motor_url = std::env::var("RGW_MOTOR_URL").unwrap_or_default();

    if motor_url.is_empty() {
        return ToolResult {
            tool: action.into(),
            success: false,
            content: format!(
                "No motor cortex backend configured for '{}'. Set the RGW_MOTOR_URL env var.",
                action
            ),
            metadata: serde_json::json!({"action": action, "needs_backend": true}),
        };
    }

    // Strip any caller-supplied motor_url before forwarding (it is ignored above,
    // but never let it reach the backend payload either).
    let mut clean_params = params.clone();
    if let Some(obj) = clean_params.as_object_mut() {
        obj.remove("motor_url");
    }

    let cmd = crate::motor::MotorCommand {
        action: action.to_string(),
        domain: params["domain"].as_str().unwrap_or("content").to_string(),
        walker_context: String::new(),
        expression_seeds: Vec::new(),
        confidence: 0.8,
        novelty: 0.0,
        search_query: None,
        params: Some(clean_params),
        audience_id: None,  // Tool invocations have no explicit audience
    };

    // Motor commands are fire-and-forget by design, but for tool
    // invocations we want acknowledgment. Use a direct POST.
    let url = format!("{}/api/admin/rgw/execute", motor_url);
    match reqwest::Client::new()
        .post(&url)
        .json(&cmd)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            ToolResult {
                tool: action.into(),
                success: true,
                content: if body.is_empty() {
                    format!("{} command accepted by motor cortex", action)
                } else {
                    crate::core::safe_truncate(&body, 2000).to_string()
                },
                metadata: serde_json::json!({"action": action}),
            }
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            ToolResult {
                tool: action.into(),
                success: false,
                content: format!("Motor cortex rejected {} (HTTP {}): {}", action, status, crate::core::safe_truncate(&body, 200)),
                metadata: serde_json::json!({"action": action, "status": status}),
            }
        }
        Err(e) => ToolResult {
            tool: action.into(),
            success: false,
            content: format!("Motor cortex unreachable for {}: {}", action, e),
            metadata: serde_json::json!({"action": action, "error": e.to_string()}),
        },
    }
}

/// Search the web using a search API
async fn web_search(params: serde_json::Value) -> ToolResult {
    let query = params["query"].as_str().unwrap_or("");
    if query.is_empty() {
        return ToolResult {
            tool: "web_search".into(),
            success: false,
            content: "Empty query".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Use DuckDuckGo instant answer API (free, no key needed)
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding::encode(query)
    );

    match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                let abstract_text = data["AbstractText"].as_str().unwrap_or("");
                let heading = data["Heading"].as_str().unwrap_or("");
                let source = data["AbstractSource"].as_str().unwrap_or("");

                let content = if !abstract_text.is_empty() {
                    format!("{}: {}", heading, abstract_text)
                } else {
                    // Try related topics
                    let topics: Vec<String> = data["RelatedTopics"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .take(5)
                        .filter_map(|t| t["Text"].as_str().map(|s| s.to_string()))
                        .collect();
                    if topics.is_empty() {
                        format!("No results for: {}", query)
                    } else {
                        topics.join("\n")
                    }
                };

                ToolResult {
                    tool: "web_search".into(),
                    success: !content.contains("No results"),
                    content,
                    metadata: serde_json::json!({
                        "query": query,
                        "source": source,
                        "heading": heading,
                    }),
                }
            } else {
                ToolResult {
                    tool: "web_search".into(),
                    success: false,
                    content: "Failed to parse search results".into(),
                    metadata: serde_json::json!({"query": query}),
                }
            }
        }
        Err(e) => ToolResult {
            tool: "web_search".into(),
            success: false,
            content: format!("Search failed: {}", e),
            metadata: serde_json::json!({"query": query}),
        },
    }
}

/// Fetch a web page and extract readable text (strips HTML)
async fn web_fetch(params: serde_json::Value) -> ToolResult {
    let url = params["url"].as_str().unwrap_or("");
    if url.is_empty() {
        return ToolResult {
            tool: "web_fetch".into(),
            success: false,
            content: "Empty URL".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Build the client without unwrap(): a builder failure must not panic the
    // request handler. Fall back to a default client if the builder errors.
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; RGW/1.0)")
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    match client
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => {
                    // Extract readable text from HTML
                    let content = extract_readable_text(&body);
                    let content = if content.chars().count() > 3000 {
                        format!("{}...", crate::core::safe_truncate(&content, 3000))
                    } else {
                        content
                    };
                    ToolResult {
                        tool: "web_fetch".into(),
                        success: status < 400 && !content.is_empty(),
                        content,
                        metadata: serde_json::json!({"url": url, "status": status, "raw_len": body.len()}),
                    }
                }
                Err(e) => ToolResult {
                    tool: "web_fetch".into(),
                    success: false,
                    content: format!("Failed to read body: {}", e),
                    metadata: serde_json::json!({"url": url}),
                },
            }
        }
        Err(e) => ToolResult {
            tool: "web_fetch".into(),
            success: false,
            content: format!("Fetch failed: {}", e),
            metadata: serde_json::json!({"url": url}),
        },
    }
}

/// Simple HTML → readable text extraction.
/// Strips tags, scripts, styles, and collapses whitespace.
fn extract_readable_text(html: &str) -> String {
    let mut text = html.to_string();

    // Remove script and style blocks entirely
    while let Some(start) = text.find("<script") {
        if let Some(end) = text[start..].find("</script>") {
            text = format!("{}{}", &text[..start], &text[start + end + 9..]);
        } else {
            break;
        }
    }
    while let Some(start) = text.find("<style") {
        if let Some(end) = text[start..].find("</style>") {
            text = format!("{}{}", &text[..start], &text[start + end + 8..]);
        } else {
            break;
        }
    }

    // Replace block elements with newlines
    for tag in &["</p>", "</div>", "</h1>", "</h2>", "</h3>", "</h4>", "</li>", "<br", "</tr>"] {
        text = text.replace(tag, &format!("\n{}", tag));
    }

    // Strip all remaining HTML tags
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    for ch in text.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }

    // Decode common HTML entities
    let result = result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");

    // Collapse whitespace
    let lines: Vec<String> = result
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect();

    lines.join("\n")
}

/// Store a new memory node
async fn memory_store(params: serde_json::Value, pool: &sqlx::PgPool) -> ToolResult {
    let content = params["content"].as_str().unwrap_or("");
    let domain = params["domain"].as_str().unwrap_or("unknown");
    let importance = params["importance"].as_f64().unwrap_or(5.0) as f32;

    if content.is_empty() {
        return ToolResult {
            tool: "memory_store".into(),
            success: false,
            content: "Empty content".into(),
            metadata: serde_json::json!({}),
        };
    }

    // Store via direct SQL (nodes are memory_vectors rows)
    match sqlx::query(
        "INSERT INTO memory_vectors (doc_id, document, domain, importance, source_type, stored_at, created_at, updated_at) \
         VALUES ($1, $2, $3, $4, 'rgw_tool', extract(epoch from now()), NOW(), NOW()) \
         RETURNING id"
    )
    .bind(format!("rgw_{}", chrono::Utc::now().timestamp()))
    .bind(content)
    .bind(domain)
    .bind(importance)
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let id: i32 = row.get("id");
            ToolResult {
                tool: "memory_store".into(),
                success: true,
                content: format!("Stored as node {} (domain: {})", id, domain),
                metadata: serde_json::json!({"node_id": id, "domain": domain}),
            }
        }
        Err(e) => ToolResult {
            tool: "memory_store".into(),
            success: false,
            content: format!("Store failed: {}", e),
            metadata: serde_json::json!({}),
        },
    }
}

/// Create an edge between nodes
async fn edge_create(params: serde_json::Value, pool: &sqlx::PgPool) -> ToolResult {
    let source = params["source_id"].as_i64().unwrap_or(0) as i32;
    let target = params["target_id"].as_i64().unwrap_or(0) as i32;
    let edge_type = params["edge_type"].as_str().unwrap_or("related");
    let weight = params["weight"].as_f64().unwrap_or(0.5) as f32;

    match crate::db::create_edge(pool, source, target, edge_type, weight, 0.0).await {
        Ok(edge_id) => ToolResult {
            tool: "edge_create".into(),
            success: true,
            content: format!("Edge {} → {} (type: {}, id: {})", source, target, edge_type, edge_id),
            metadata: serde_json::json!({"edge_id": edge_id}),
        },
        Err(e) => ToolResult {
            tool: "edge_create".into(),
            success: false,
            content: format!("Edge creation failed: {}", e),
            metadata: serde_json::json!({}),
        },
    }
}

// Need urlencoding for search queries
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

use sqlx::Row;
use chrono;
