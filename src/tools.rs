//! Tool calling framework — RGW can use tools to interact with the world.
//!
//! Tools:
//!   - web_search: search the internet, results become graph nodes
//!   - web_fetch: scrape a URL, content becomes a graph node
//!   - memory_store: disabled (Julian owns node content)
//!   - edge_create: connect two nodes
//!   - code_exec: REMOVED — was an unauthenticated RCE (python3 -c). Needs a real sandbox before it can return.
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

pub type ToolFuture = std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<ToolResult>> + Send>>;
pub type ToolCallback = dyn Fn(serde_json::Value) -> ToolFuture + Send + Sync;

pub struct ToolExecutor {
    pub name: String,
    pub description: String,
    pub execute: Arc<ToolCallback>,
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
            description: "Disabled: Julian owns memory node content and embeddings.".into(),
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
    motor_tool_via(crate::motor::gate(), &motor_url, action, params).await
}

/// Testable core of [`motor_tool`]: takes the gate and the server-side URL.
pub(crate) async fn motor_tool_via(
    gate: &crate::motor::MotorGate,
    motor_url: &str,
    action: &str,
    params: serde_json::Value,
) -> ToolResult {
    use crate::motor::DispatchOutcome;

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
        correlation_id: uuid::Uuid::new_v4(),
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
    // invocations we want acknowledgment: 30 s timeout, body returned.
    let source = format!("tool:{}", action);
    let mode = gate.mode().as_str();
    match gate
        .dispatch_with_timeout(&source, motor_url, cmd, std::time::Duration::from_secs(30))
        .await
    {
        DispatchOutcome::Sent(_, body) => ToolResult {
            tool: action.into(),
            success: true,
            content: if body.is_empty() {
                format!("{} command accepted by motor cortex", action)
            } else {
                crate::core::safe_truncate(&body, 2000).to_string()
            },
            metadata: serde_json::json!({"action": action}),
        },
        DispatchOutcome::SendFailed(err) => ToolResult {
            tool: action.into(),
            success: false,
            content: format!("Motor cortex failed for {}: {}", action, err),
            metadata: serde_json::json!({"action": action, "error": err}),
        },
        DispatchOutcome::Queued(id) => ToolResult {
            tool: action.into(),
            success: false,
            content: format!("Queued for approval as proposal #{}; nothing was sent", id),
            metadata: serde_json::json!({"action": action, "held": true, "mode": mode, "proposal_id": id}),
        },
        DispatchOutcome::Dropped | DispatchOutcome::Shadowed => ToolResult {
            tool: action.into(),
            success: false,
            content: format!("Held: motor mode is {}; nothing was sent", mode),
            metadata: serde_json::json!({"action": action, "held": true, "mode": mode, "proposal_id": null}),
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

    // Block SSRF: only http/https to publicly-routable hosts, on every hop.
    let resp = match guarded_get(url, system_resolve, |addr| ip_is_internal(addr.ip())).await {
        Ok(resp) => resp,
        Err(FetchError::Blocked(reason)) => {
            return ToolResult {
                tool: "web_fetch".into(),
                success: false,
                content: format!("Blocked URL: {}", reason),
                metadata: serde_json::json!({"url": url, "blocked": true}),
            };
        }
        Err(FetchError::Failed(reason)) => {
            return ToolResult {
                tool: "web_fetch".into(),
                success: false,
                content: format!("Fetch failed: {}", reason),
                metadata: serde_json::json!({"url": url}),
            };
        }
    };

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

/// Redirect hops `web_fetch` follows before giving up.
const FETCH_MAX_REDIRECTS: usize = 5;

/// Why a guarded fetch did not produce a response.
#[derive(Debug, PartialEq, Eq)]
enum FetchError {
    /// The SSRF guard refused a hop; nothing was sent to that hop.
    Blocked(String),
    /// Transport or client failure.
    Failed(String),
}

/// Production resolver: the system's DNS answers for `host:port`.
async fn system_resolve(host: String, port: u16) -> Result<Vec<std::net::SocketAddr>, String> {
    tokio::net::lookup_host((host.as_str(), port))
        .await
        .map(|addrs| addrs.collect())
        .map_err(|e| e.to_string())
}

/// SSRF-guarded GET. For the first URL and for every redirect hop:
///
/// 1. the scheme must be http/https and the name must not be an internal one;
/// 2. the host is resolved **once**, and every answer must pass `blocked`;
/// 3. the connection is pinned to exactly those vetted answers, so a second DNS
///    answer (rebinding) cannot change where the request goes.
///
/// Redirects are never followed by the HTTP client; each `Location` goes through
/// the same three steps, so a public name that resolves to an internal address
/// on a later hop is refused before any request is sent to it. Proxies are not
/// used: a proxy would resolve the name itself and defeat the pin.
///
/// `resolve` and `blocked` are parameters so tests can prove this without DNS.
async fn guarded_get<R, Fut, B>(raw: &str, resolve: R, blocked: B) -> Result<reqwest::Response, FetchError>
where
    R: Fn(String, u16) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<std::net::SocketAddr>, String>>,
    B: Fn(&std::net::SocketAddr) -> bool,
{
    let mut url = reqwest::Url::parse(raw).map_err(|e| FetchError::Blocked(format!("invalid URL ({})", e)))?;
    for _hop in 0..=FETCH_MAX_REDIRECTS {
        let pin = vet_hop(&url, &resolve, &blocked).await.map_err(FetchError::Blocked)?;
        let mut builder = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; RGW/1.0)")
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy();
        if let Some((host, addrs)) = &pin {
            builder = builder.resolve_to_addrs(host, addrs);
        }
        // No unwrap() and no fallback to Client::new(): a default client would
        // follow redirects unguarded and resolve names itself.
        let client = builder.build().map_err(|e| FetchError::Failed(format!("HTTP client build failed: {}", e)))?;
        let resp = client
            .get(url.clone())
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| FetchError::Failed(e.to_string()))?;
        let location = if resp.status().is_redirection() {
            resp.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok()).map(str::to_owned)
        } else {
            None
        };
        match location {
            Some(next) => {
                url = url.join(&next).map_err(|e| FetchError::Blocked(format!("invalid redirect target ({})", e)))?;
            }
            None => return Ok(resp),
        }
    }
    Err(FetchError::Failed("too many redirects".into()))
}

/// Vet one hop. `Ok(Some((host, addrs)))` is a name with the vetted answers to
/// pin; `Ok(None)` is a literal public IP (nothing to resolve). `Err` is the reason.
async fn vet_hop<R, Fut, B>(
    url: &reqwest::Url,
    resolve: &R,
    blocked: &B,
) -> Result<Option<(String, Vec<std::net::SocketAddr>)>, String>
where
    R: Fn(String, u16) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<std::net::SocketAddr>, String>>,
    B: Fn(&std::net::SocketAddr) -> bool,
{
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(format!("scheme '{}' not allowed", other)),
    }
    let host = url.host_str().ok_or_else(|| "URL has no host".to_string())?;
    let name = host.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    if name == "localhost" || name.ends_with(".localhost") || name.ends_with(".internal") {
        return Err("internal host blocked".into());
    }
    let port = url.port_or_known_default().unwrap_or(80);
    if let Ok(ip) = name.parse::<std::net::IpAddr>() {
        return if blocked(&std::net::SocketAddr::new(ip, port)) {
            Err("private/loopback address blocked".into())
        } else {
            Ok(None)
        };
    }
    let addrs = resolve(host.to_string(), port)
        .await
        .map_err(|e| format!("DNS resolution failed ({})", e))?;
    if addrs.is_empty() {
        return Err("host did not resolve".into());
    }
    // One internal answer is enough to refuse: the client could pick any of them.
    if addrs.iter().any(blocked) {
        return Err("resolves to a private/loopback address".into());
    }
    Ok(Some((host.to_string(), addrs)))
}

/// True if the IP is not publicly routable. IPv4 is a denylist of the special
/// ranges (loopback, private, link-local, unspecified, multicast, reserved,
/// benchmarking). IPv6 is an allowlist: only global unicast `2000::/3` outside
/// its special-purpose sub-ranges is public; IPv4-mapped, NAT64 and 6to4
/// addresses are judged by the IPv4 address they embed.
fn ip_is_internal(ip: std::net::IpAddr) -> bool {
    use std::net::{IpAddr, Ipv4Addr};
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast() // 224.0.0.0/4
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // CGNAT 100.64/10
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // IETF protocol assignments 192.0.0.0/24
                || (o[0] == 198 && (o[1] & 0xfe) == 18) // benchmarking 198.18.0.0/15
                || o[0] >= 240 // reserved 240.0.0.0/4
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return ip_is_internal(IpAddr::V4(v4));
            }
            let s = v6.segments();
            let embedded = |hi: u16, lo: u16| IpAddr::V4(Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8));
            // Well-known NAT64 64:ff9b::/96 reaches the embedded IPv4 address.
            if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                return ip_is_internal(embedded(s[6], s[7]));
            }
            // Allowlist: only global unicast 2000::/3 can be public. Everything
            // else (::/128, ::1, IPv4-compatible, local-use NAT64 64:ff9b:1::/48,
            // unique-local, site-local, link-local, multicast, unassigned) is internal.
            if (s[0] & 0xe000) != 0x2000 {
                return true;
            }
            // Special-purpose ranges inside 2000::/3.
            if s[0] == 0x2002 {
                return ip_is_internal(embedded(s[1], s[2])); // 6to4 2002:a.b.c.d::/48
            }
            (s[0] == 0x2001 && (s[1] & 0xfe00) == 0) // IETF protocol assignments 2001::/23 (Teredo, ORCHID, ...)
                || (s[0] == 0x2001 && s[1] == 0xdb8) // documentation 2001:db8::/32
                || (s[0] == 0x3fff && (s[1] & 0xf000) == 0) // documentation 3fff::/20
        }
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

/// Node content is owned by Julian and must go through its ingestion pipeline.
async fn memory_store(_params: serde_json::Value, _pool: &sqlx::PgPool) -> ToolResult {
    ToolResult {
        tool: "memory_store".into(),
        success: false,
        content: "memory_store is disabled; Julian owns memory node content and embeddings".into(),
        metadata: serde_json::json!({"disabled": true}),
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


#[cfg(test)]
mod tests {
    use super::*;

    /// C1 regression: the RCE primitive must never be advertised via GET /tools.
    #[test]
    fn code_exec_is_not_advertised() {
        assert!(available_tools().iter().all(|t| t.name != "code_exec"));
    }

    #[test]
    fn ssrf_ip_classification() {
        use std::net::IpAddr;
        for s in ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1",
                  "169.254.169.254", "0.0.0.0", "100.64.0.1", "::1",
                  "fd00::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254"] {
            assert!(ip_is_internal(s.parse::<IpAddr>().unwrap()), "{s} should be internal");
        }
        for s in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"] {
            assert!(!ip_is_internal(s.parse::<IpAddr>().unwrap()), "{s} should be public");
        }
    }

    #[test]
    fn ssrf_ip_classification_covers_reserved_and_embedded_ranges() {
        use std::net::IpAddr;
        for s in ["224.0.0.1", "239.255.255.250", "192.0.0.8", "198.18.0.1", "198.19.255.254",
                  "240.0.0.1", "255.255.255.255", "::", "::10.0.0.1", "::127.0.0.1", "::8.8.8.8",
                  "64:ff9b::a00:1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b:1::808:808",
                  "2002:a00:1::1", "2002:7f00:1::", "ff02::1", "ff05::2", "fec0::1", "fc00::1",
                  "fe80::1", "2001::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", "2001:db8::1",
                  "2001:10::1", "2001:20::1", "2001:1ff::1", "3fff::1", "100::1", "4000::1", "e000::1"] {
            assert!(ip_is_internal(s.parse::<IpAddr>().unwrap()), "{s} should be internal");
        }
        for s in ["192.0.1.1", "198.17.255.255", "198.20.0.1", "223.255.255.254", "100.128.0.1",
                  "64:ff9b::808:808", "2002:808:808::1", "2a00:1450:4001::200e",
                  "2606:4700:4700::1111", "2001:4860:4860::8888", "2001:200::1", "3ffe::1"] {
            assert!(!ip_is_internal(s.parse::<IpAddr>().unwrap()), "{s} should be public");
        }
    }

    /// A resolver that must not be reached.
    async fn no_dns(host: String, _port: u16) -> Result<Vec<std::net::SocketAddr>, String> {
        panic!("{host} must be refused before any DNS lookup")
    }

    #[tokio::test]
    async fn guarded_get_rejects_bad_targets_without_network() {
        for u in ["file:///etc/passwd", "ftp://example.com/", "http://localhost/",
                  "http://x.localhost/", "http://metadata.google.internal/",
                  "http://127.0.0.1:11435/self", "http://169.254.169.254/latest/meta-data/",
                  "http://[::1]/", "http://[::ffff:10.0.0.1]/", "http://[64:ff9b::a9fe:a9fe]/", "not a url"] {
            let result = guarded_get(u, no_dns, |addr| ip_is_internal(addr.ip())).await;
            assert!(matches!(result, Err(FetchError::Blocked(_))), "{u} should be blocked");
        }
    }

    /// One loopback server per test. `/start` redirects to `location`; every
    /// other path answers 200 "reached". Returns its port and a hit counter.
    async fn spawn_site(location: Option<String>) -> (u16, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let app = axum::Router::new().fallback(move |uri: axum::http::Uri| {
            let h = h.clone();
            let location = location.clone();
            async move {
                use axum::response::IntoResponse;
                h.fetch_add(1, Ordering::SeqCst);
                match location {
                    Some(to) if uri.path() == "/start" => {
                        (axum::http::StatusCode::FOUND, [(axum::http::header::LOCATION, to)]).into_response()
                    }
                    _ => "reached".into_response(),
                }
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (port, hits)
    }

    fn loopback(port: u16) -> std::net::SocketAddr {
        std::net::SocketAddr::from(([127, 0, 0, 1], port))
    }

    #[tokio::test]
    async fn redirect_to_a_name_that_resolves_internally_is_refused_before_any_request() {
        use std::sync::atomic::Ordering;
        // `inner` stands for an internal service; the policy below marks its address internal.
        let (inner_port, inner_hits) = spawn_site(None).await;
        let (public_port, public_hits) =
            spawn_site(Some(format!("http://intranet.test:{inner_port}/secret"))).await;
        let lookups = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen = lookups.clone();
        // Both names are ordinary hostnames; only DNS says where they lead.
        let resolve = move |host: String, port: u16| {
            seen.lock().unwrap().push(host);
            async move { Ok(vec![loopback(port)]) }
        };
        let result = guarded_get(
            &format!("http://public.test:{public_port}/start"),
            resolve,
            |addr| addr.port() == inner_port,
        )
        .await;
        assert_eq!(result.unwrap_err(), FetchError::Blocked("resolves to a private/loopback address".into()));
        assert_eq!(public_hits.load(Ordering::SeqCst), 1, "the first hop is fetched");
        assert_eq!(inner_hits.load(Ordering::SeqCst), 0, "the internal hop must never receive a request");
        assert_eq!(*lookups.lock().unwrap(), ["public.test", "intranet.test"], "every hop is resolved and vetted");
    }

    #[tokio::test]
    async fn the_vetted_dns_answer_is_pinned_for_the_connection() {
        use std::sync::atomic::Ordering;
        let (port, hits) = spawn_site(None).await;
        let (inner_port, inner_hits) = spawn_site(None).await;
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = calls.clone();
        // A rebinding name: the first answer is the vetted address, every later
        // answer is the internal one. `.test` never resolves through real DNS, so
        // the request can only succeed over the pinned first answer.
        let resolve = move |_host: String, _port: u16| {
            let first = c.fetch_add(1, Ordering::SeqCst) == 0;
            async move { Ok(vec![loopback(if first { port } else { inner_port })]) }
        };
        let resp = guarded_get(&format!("http://rebind.test:{port}/page"), resolve, |addr| addr.port() == inner_port)
            .await
            .expect("the vetted address is reachable");
        assert_eq!(resp.text().await.unwrap(), "reached");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "resolved once; the client must not resolve again");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(inner_hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn public_redirects_are_followed_hop_by_hop_and_bounded() {
        use std::sync::atomic::Ordering;
        let (final_port, final_hits) = spawn_site(None).await;
        let (first_port, _) = spawn_site(Some(format!("http://second.test:{final_port}/page"))).await;
        let resolve = |_host: String, port: u16| async move { Ok(vec![loopback(port)]) };
        let resp = guarded_get(&format!("http://first.test:{first_port}/start"), resolve, |_| false)
            .await
            .expect("a redirect between public hosts is followed");
        assert_eq!(resp.text().await.unwrap(), "reached");
        assert_eq!(final_hits.load(Ordering::SeqCst), 1);

        // A relative Location is resolved against the current hop; a loop ends.
        let (loop_port, loop_hits) = spawn_site(Some("/start".into())).await;
        let result = guarded_get(&format!("http://loop.test:{loop_port}/start"), resolve, |_| false).await;
        assert_eq!(result.unwrap_err(), FetchError::Failed("too many redirects".into()));
        assert_eq!(loop_hits.load(Ordering::SeqCst), FETCH_MAX_REDIRECTS + 1);

        // A redirect to a literal internal address or a non-http scheme is refused.
        for target in ["http://169.254.169.254/latest/meta-data/", "file:///etc/passwd", "http://localhost/"] {
            let (port, _) = spawn_site(Some(target.into())).await;
            let result = guarded_get(&format!("http://hop.test:{port}/start"), resolve, |addr| ip_is_internal(addr.ip()) && addr.port() != port).await;
            assert!(matches!(result, Err(FetchError::Blocked(_))), "{target}");
        }
    }
}
