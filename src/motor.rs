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
//!
//! Every outbound action goes through [`MotorGate`] (RGW_MOTOR_MODE:
//! off | shadow | propose | live, default shadow). Only the gate sends.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

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

// ── Motor gate ──────────────────────────────────────────────────────
//
// Every outbound motor action goes through one gate. The mode is fixed at
// startup from RGW_MOTOR_MODE and defaults to Shadow: nothing leaves the
// process until a human says so.

/// Max proposals held in the queue (pending + decided).
pub const PROPOSAL_CAP: usize = 100;
/// Max recent shadow records kept for inspection.
pub const SHADOW_RING_CAP: usize = 50;
/// Pending proposals older than this become expired.
pub const PROPOSAL_TTL_SECS: u64 = 86_400;
/// Timeout for diverger-originated sends (unchanged from the old execute()).
pub const DEFAULT_SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// How outbound motor actions are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MotorMode {
    /// Drop the action (debug log, counted).
    Off,
    /// Don't send; log at INFO and keep in the shadow ring. Default.
    Shadow,
    /// Don't send; queue for a human to approve or reject.
    Propose,
    /// Send immediately.
    Live,
}

impl MotorMode {
    /// Parse RGW_MOTOR_MODE. Unset, empty or unrecognised → Shadow (never Live by accident).
    pub fn from_env_value(v: Option<&str>) -> MotorMode {
        let raw = v.unwrap_or("");
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => MotorMode::Off,
            "shadow" | "" => MotorMode::Shadow,
            "propose" => MotorMode::Propose,
            "live" => MotorMode::Live,
            other => {
                tracing::warn!(
                    "[motor] Unrecognised RGW_MOTOR_MODE {:?}; using shadow (valid: off|shadow|propose|live)",
                    other
                );
                MotorMode::Shadow
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            MotorMode::Off => "off",
            MotorMode::Shadow => "shadow",
            MotorMode::Propose => "propose",
            MotorMode::Live => "live",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    ApprovedSent,
    ApprovedSendFailed,
    Rejected,
    Expired,
}

/// A motor action waiting for (or past) a human decision.
#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    pub id: u64,
    pub created_at: u64,
    /// "diverger" | "tool:<name>"
    pub source: String,
    /// Server-side target base URL (JULIAN_URL / RGW_MOTOR_URL), never request input.
    pub target: String,
    pub command: MotorCommand,
    pub status: ProposalStatus,
    pub decided_at: Option<u64>,
    pub note: Option<String>,
    /// Outcome of the send on approve (e.g. "HTTP 200" or an error).
    pub send_result: Option<String>,
    #[serde(skip)]
    timeout: Duration,
}

/// What the gate would have sent while in shadow mode.
#[derive(Debug, Clone, Serialize)]
pub struct ShadowRecord {
    pub at: u64,
    pub source: String,
    pub target: String,
    pub command: MotorCommand,
}

/// Result of dispatching a command through the gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    Dropped,
    Shadowed,
    Queued(u64),
    /// HTTP status and response body (2xx).
    Sent(u16, String),
    SendFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    NotFound,
    AlreadyDecided,
    Expired,
}

#[derive(Debug, Default)]
struct Counters {
    off_dropped: AtomicU64,
    shadowed: AtomicU64,
    proposed: AtomicU64,
    sent: AtomicU64,
    send_failed: AtomicU64,
    approved: AtomicU64,
    rejected: AtomicU64,
    expired: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> serde_json::Value {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        serde_json::json!({
            "off_dropped": g(&self.off_dropped),
            "shadowed": g(&self.shadowed),
            "proposed": g(&self.proposed),
            "sent": g(&self.sent),
            "send_failed": g(&self.send_failed),
            "approved": g(&self.approved),
            "rejected": g(&self.rejected),
            "expired": g(&self.expired),
        })
    }
}

/// The single gate for outbound motor actions.
#[derive(Debug)]
pub struct MotorGate {
    mode: MotorMode,
    proposals: Mutex<VecDeque<Proposal>>,
    shadow: Mutex<VecDeque<ShadowRecord>>,
    next_id: AtomicU64,
    counters: Counters,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl MotorGate {
    pub fn new(mode: MotorMode) -> Self {
        Self {
            mode,
            proposals: Mutex::new(VecDeque::new()),
            shadow: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
            counters: Counters::default(),
        }
    }

    pub fn mode(&self) -> MotorMode {
        self.mode
    }

    /// Dispatch with the default (diverger) send timeout.
    pub async fn dispatch(&self, source: &str, base_url: &str, cmd: MotorCommand) -> DispatchOutcome {
        self.dispatch_with_timeout(source, base_url, cmd, DEFAULT_SEND_TIMEOUT).await
    }

    /// Dispatch a command. Only the Live branch sends.
    pub async fn dispatch_with_timeout(
        &self,
        source: &str,
        base_url: &str,
        cmd: MotorCommand,
        timeout: Duration,
    ) -> DispatchOutcome {
        match self.mode {
            MotorMode::Off => {
                self.counters.off_dropped.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("[motor] OFF: dropped {} from {} (domain={})", cmd.action, source, cmd.domain);
                DispatchOutcome::Dropped
            }
            MotorMode::Shadow => {
                self.counters.shadowed.fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    "[motor] SHADOW: would send {} from {} (domain={}, confidence={:.0}%)",
                    cmd.action, source, cmd.domain, cmd.confidence * 100.0
                );
                let mut ring = self.shadow.lock().unwrap_or_else(|e| e.into_inner());
                if ring.len() >= SHADOW_RING_CAP {
                    ring.pop_front();
                }
                ring.push_back(ShadowRecord {
                    at: now_secs(),
                    source: source.to_string(),
                    target: base_url.to_string(),
                    command: cmd,
                });
                DispatchOutcome::Shadowed
            }
            MotorMode::Propose => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                self.counters.proposed.fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    "[motor] PROPOSE: queued #{} {} from {} (domain={})",
                    id, cmd.action, source, cmd.domain
                );
                let mut q = self.proposals.lock().unwrap_or_else(|e| e.into_inner());
                if q.len() >= PROPOSAL_CAP {
                    // Prefer evicting a decided entry; otherwise drop the oldest pending.
                    let idx = q
                        .iter()
                        .position(|p| p.status != ProposalStatus::Pending)
                        .unwrap_or(0);
                    if let Some(old) = q.remove(idx)
                        && old.status == ProposalStatus::Pending
                    {
                        tracing::warn!(
                            "[motor] Proposal queue full ({}); dropped oldest pending #{} ({})",
                            PROPOSAL_CAP, old.id, old.command.action
                        );
                    }
                }
                q.push_back(Proposal {
                    id,
                    created_at: now_secs(),
                    source: source.to_string(),
                    target: base_url.to_string(),
                    command: cmd,
                    status: ProposalStatus::Pending,
                    decided_at: None,
                    note: None,
                    send_result: None,
                    timeout,
                });
                DispatchOutcome::Queued(id)
            }
            MotorMode::Live => match send(base_url, &cmd, timeout).await {
                Ok((status, body)) if (200..300).contains(&status) => {
                    self.counters.sent.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        "[motor] Sent: {} from {} (domain={}, confidence={:.0}%)",
                        cmd.action, source, cmd.domain, cmd.confidence * 100.0
                    );
                    DispatchOutcome::Sent(status, body)
                }
                Ok((status, body)) => {
                    self.counters.send_failed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("[motor] Backend returned {} for {} from {}", status, cmd.action, source);
                    DispatchOutcome::SendFailed(format!(
                        "HTTP {}: {}",
                        status,
                        crate::core::safe_truncate(&body, 200)
                    ))
                }
                Err(e) => {
                    self.counters.send_failed.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!("[motor] Failed to reach backend for {}: {}", cmd.action, e);
                    DispatchOutcome::SendFailed(e)
                }
            },
        }
    }

    /// Mark pending proposals past their TTL as expired.
    fn expire_locked(&self, q: &mut VecDeque<Proposal>, now: u64) {
        for p in q.iter_mut() {
            if p.status == ProposalStatus::Pending && now.saturating_sub(p.created_at) >= PROPOSAL_TTL_SECS {
                p.status = ProposalStatus::Expired;
                p.decided_at = Some(now);
                self.counters.expired.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Approve a pending proposal: sends it exactly once and records the result.
    pub async fn approve(&self, id: u64) -> Result<Proposal, GateError> {
        // Claim the proposal under the lock (status flips away from Pending so a
        // concurrent approve sees AlreadyDecided), then send without holding it.
        let (target, cmd, timeout) = {
            let mut q = self.proposals.lock().unwrap_or_else(|e| e.into_inner());
            let now = now_secs();
            self.expire_locked(&mut q, now);
            let p = q.iter_mut().find(|p| p.id == id).ok_or(GateError::NotFound)?;
            match p.status {
                ProposalStatus::Pending => {}
                ProposalStatus::Expired => return Err(GateError::Expired),
                _ => return Err(GateError::AlreadyDecided),
            }
            p.status = ProposalStatus::ApprovedSendFailed; // provisional until the send returns
            p.decided_at = Some(now);
            p.send_result = Some("sending".into());
            (p.target.clone(), p.command.clone(), p.timeout)
        };
        self.counters.approved.fetch_add(1, Ordering::Relaxed);

        let (status, result) = match send(&target, &cmd, timeout).await {
            Ok((code, _)) if (200..300).contains(&code) => {
                self.counters.sent.fetch_add(1, Ordering::Relaxed);
                (ProposalStatus::ApprovedSent, format!("HTTP {}", code))
            }
            Ok((code, body)) => {
                self.counters.send_failed.fetch_add(1, Ordering::Relaxed);
                (
                    ProposalStatus::ApprovedSendFailed,
                    format!("HTTP {}: {}", code, crate::core::safe_truncate(&body, 200)),
                )
            }
            Err(e) => {
                self.counters.send_failed.fetch_add(1, Ordering::Relaxed);
                (ProposalStatus::ApprovedSendFailed, e)
            }
        };

        let mut q = self.proposals.lock().unwrap_or_else(|e| e.into_inner());
        match q.iter_mut().find(|p| p.id == id) {
            Some(p) => {
                p.status = status;
                p.send_result = Some(result);
                Ok(p.clone())
            }
            // Evicted while sending (queue overflow); report what happened anyway.
            None => Err(GateError::NotFound),
        }
    }

    /// Reject a pending proposal. Nothing is sent.
    pub fn reject(&self, id: u64, note: Option<String>) -> Result<Proposal, GateError> {
        let mut q = self.proposals.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_secs();
        self.expire_locked(&mut q, now);
        let p = q.iter_mut().find(|p| p.id == id).ok_or(GateError::NotFound)?;
        match p.status {
            ProposalStatus::Pending => {}
            ProposalStatus::Expired => return Err(GateError::Expired),
            _ => return Err(GateError::AlreadyDecided),
        }
        p.status = ProposalStatus::Rejected;
        p.decided_at = Some(now);
        p.note = note;
        self.counters.rejected.fetch_add(1, Ordering::Relaxed);
        Ok(p.clone())
    }

    /// Mode, counters, pending proposals, the last 50 decided, and the shadow ring.
    pub fn snapshot(&self) -> serde_json::Value {
        let (pending, decided) = {
            let mut q = self.proposals.lock().unwrap_or_else(|e| e.into_inner());
            self.expire_locked(&mut q, now_secs());
            let pending: Vec<Proposal> =
                q.iter().filter(|p| p.status == ProposalStatus::Pending).cloned().collect();
            let decided: Vec<Proposal> = q
                .iter()
                .rev()
                .filter(|p| p.status != ProposalStatus::Pending)
                .take(50)
                .cloned()
                .collect();
            (pending, decided)
        };
        let shadow: Vec<ShadowRecord> =
            self.shadow.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect();
        serde_json::json!({
            "mode": self.mode,
            "counters": self.counters.snapshot(),
            "pending": pending,
            "decided": decided,
            "shadow": shadow,
        })
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.proposals
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.status == ProposalStatus::Pending)
            .count()
    }

    #[cfg(test)]
    fn queue_len(&self) -> usize {
        self.proposals.lock().unwrap().len()
    }

    #[cfg(test)]
    fn shadow_len(&self) -> usize {
        self.shadow.lock().unwrap().len()
    }

    #[cfg(test)]
    fn backdate(&self, id: u64, secs: u64) {
        let mut q = self.proposals.lock().unwrap();
        if let Some(p) = q.iter_mut().find(|p| p.id == id) {
            p.created_at = p.created_at.saturating_sub(secs);
        }
    }

    #[cfg(test)]
    fn counter(&self, name: &str) -> u64 {
        self.counters.snapshot()[name].as_u64().unwrap()
    }
}

static GATE: OnceLock<Arc<MotorGate>> = OnceLock::new();
static FALLBACK_GATE: OnceLock<MotorGate> = OnceLock::new();

/// Install the process-wide gate (once, from api::serve). Returns the installed gate.
pub fn install_gate(mode: MotorMode) -> Arc<MotorGate> {
    GATE.get_or_init(|| Arc::new(MotorGate::new(mode))).clone()
}

/// The process-wide gate. Uninitialised (examples, tests) → a Shadow gate.
pub fn gate() -> &'static MotorGate {
    match GATE.get() {
        Some(g) => g.as_ref(),
        None => FALLBACK_GATE.get_or_init(|| MotorGate::new(MotorMode::Shadow)),
    }
}

/// POST a motor command. Crate-private: only MotorGate calls this.
/// Returns the HTTP status and body for any HTTP response, or an error string.
pub(crate) async fn send(
    base_url: &str,
    cmd: &MotorCommand,
    timeout: Duration,
) -> Result<(u16, String), String> {
    if base_url.is_empty() {
        return Err("no motor backend URL configured".into());
    }
    let url = format!("{}/api/admin/rgw/execute", base_url);
    let resp = reqwest::Client::new()
        .post(&url)
        .json(cmd)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    Ok((status, body))
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Tiny capture server counting POSTs to /api/admin/rgw/execute.
    pub async fn spawn_capture() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let app = axum::Router::new().route(
            "/api/admin/rgw/execute",
            axum::routing::post(move || {
                let h = h.clone();
                async move {
                    h.fetch_add(1, Ordering::SeqCst);
                    "ok"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{}", addr), hits, handle)
    }

    pub fn cmd(action: &str) -> super::MotorCommand {
        super::MotorCommand {
            action: action.into(),
            domain: "test".into(),
            walker_context: String::new(),
            expression_seeds: Vec::new(),
            confidence: 0.5,
            novelty: 0.0,
            search_query: None,
            params: None,
            audience_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{cmd, spawn_capture};
    use super::*;

    #[test]
    fn t1_mode_parsing() {
        assert_eq!(MotorMode::from_env_value(None), MotorMode::Shadow);
        assert_eq!(MotorMode::from_env_value(Some("")), MotorMode::Shadow);
        assert_eq!(MotorMode::from_env_value(Some("  ")), MotorMode::Shadow);
        assert_eq!(MotorMode::from_env_value(Some("LIVE")), MotorMode::Live);
        assert_eq!(MotorMode::from_env_value(Some("live")), MotorMode::Live);
        assert_eq!(MotorMode::from_env_value(Some("propose")), MotorMode::Propose);
        assert_eq!(MotorMode::from_env_value(Some("off")), MotorMode::Off);
        assert_eq!(MotorMode::from_env_value(Some("yes")), MotorMode::Shadow);
    }

    #[tokio::test]
    async fn t2_each_mode_against_capture_server() {
        let (url, hits, server) = spawn_capture().await;

        let off = MotorGate::new(MotorMode::Off);
        assert_eq!(off.dispatch("diverger", &url, cmd("journal")).await, DispatchOutcome::Dropped);
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        assert_eq!(off.queue_len(), 0);
        assert_eq!(off.counter("off_dropped"), 1);

        let shadow = MotorGate::new(MotorMode::Shadow);
        assert_eq!(shadow.dispatch("diverger", &url, cmd("journal")).await, DispatchOutcome::Shadowed);
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        assert_eq!(shadow.shadow_len(), 1);
        assert_eq!(shadow.queue_len(), 0);

        let propose = MotorGate::new(MotorMode::Propose);
        assert!(matches!(
            propose.dispatch("diverger", &url, cmd("journal")).await,
            DispatchOutcome::Queued(_)
        ));
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        assert_eq!(propose.pending_len(), 1);

        let live = MotorGate::new(MotorMode::Live);
        assert!(matches!(
            live.dispatch("diverger", &url, cmd("journal")).await,
            DispatchOutcome::Sent(200, _)
        ));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(live.counter("sent"), 1);

        server.abort();
    }

    #[tokio::test]
    async fn t3_propose_lifecycle() {
        let (url, hits, server) = spawn_capture().await;
        let gate = MotorGate::new(MotorMode::Propose);

        let DispatchOutcome::Queued(a) = gate.dispatch("diverger", &url, cmd("search")).await else {
            panic!("expected Queued")
        };
        let p = gate.approve(a).await.unwrap();
        assert_eq!(p.status, ProposalStatus::ApprovedSent);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        assert_eq!(gate.approve(a).await.unwrap_err(), GateError::AlreadyDecided);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        let DispatchOutcome::Queued(b) = gate.dispatch("diverger", &url, cmd("explore")).await else {
            panic!("expected Queued")
        };
        let r = gate.reject(b, Some("no".into())).unwrap();
        assert_eq!(r.status, ProposalStatus::Rejected);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        let DispatchOutcome::Queued(c) = gate.dispatch("diverger", &url, cmd("journal")).await else {
            panic!("expected Queued")
        };
        gate.backdate(c, PROPOSAL_TTL_SECS + 1);
        assert_eq!(gate.approve(c).await.unwrap_err(), GateError::Expired);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(gate.counter("expired"), 1);

        // Closed port: bind, take the address, drop the listener.
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", l.local_addr().unwrap())
        };
        let DispatchOutcome::Queued(d) = gate.dispatch("diverger", &closed, cmd("journal")).await else {
            panic!("expected Queued")
        };
        let p = gate.approve(d).await.unwrap();
        assert_eq!(p.status, ProposalStatus::ApprovedSendFailed);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        assert_eq!(gate.approve(999).await.unwrap_err(), GateError::NotFound);
        server.abort();
    }

    #[tokio::test]
    async fn t4_bounds() {
        let propose = MotorGate::new(MotorMode::Propose);
        for _ in 0..(PROPOSAL_CAP * 10) {
            propose.dispatch("diverger", "http://127.0.0.1:9", cmd("journal")).await;
        }
        assert!(propose.queue_len() <= PROPOSAL_CAP);

        let shadow = MotorGate::new(MotorMode::Shadow);
        for _ in 0..(SHADOW_RING_CAP * 10) {
            shadow.dispatch("diverger", "http://127.0.0.1:9", cmd("journal")).await;
        }
        assert!(shadow.shadow_len() <= SHADOW_RING_CAP);
    }

    #[tokio::test]
    async fn t6_tool_path_held_under_shadow() {
        let (url, hits, server) = spawn_capture().await;
        let gate = MotorGate::new(MotorMode::Shadow);
        let res = crate::tools::motor_tool_via(
            &gate,
            &url,
            "post_social",
            serde_json::json!({"text": "hi", "motor_url": "http://evil.invalid"}),
        )
        .await;
        assert!(!res.success);
        assert_eq!(res.metadata["held"], true);
        assert_eq!(res.metadata["mode"], "shadow");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        // Shadow record carries the stripped params and the server-side target.
        let snap = gate.snapshot();
        assert_eq!(snap["shadow"][0]["source"], "tool:post_social");
        assert_eq!(snap["shadow"][0]["target"], url.as_str());
        assert!(snap["shadow"][0]["command"]["params"].get("motor_url").is_none());

        // Under propose the tool reports the proposal id; under live it sends once.
        let propose = MotorGate::new(MotorMode::Propose);
        let res = crate::tools::motor_tool_via(&propose, &url, "blog_publish", serde_json::json!({})).await;
        assert_eq!(res.metadata["held"], true);
        assert_eq!(res.metadata["proposal_id"], 1);
        assert_eq!(hits.load(Ordering::SeqCst), 0);

        let live = MotorGate::new(MotorMode::Live);
        let res = crate::tools::motor_tool_via(&live, &url, "image_generate", serde_json::json!({})).await;
        assert!(res.success);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[test]
    fn uninitialised_global_gate_is_shadow() {
        // Tests never install the global; the fallback must be Shadow.
        if GATE.get().is_none() {
            assert_eq!(gate().mode(), MotorMode::Shadow);
        }
    }
}
