//! Process-local HTTP admission and resource ceilings.

use std::collections::{HashMap, hash_map::RandomState};
use std::hash::BuildHasher;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{ConnectInfo, DefaultBodyLimit, FromRequest, Request, State},
    http::{StatusCode, header::RETRY_AFTER},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Debug)]
pub struct Limits {
    pub concurrent_walks: usize,
    pub concurrent_work: usize,
    pub concurrent_maintenance: usize,
    pub concurrent_benchmarks: usize,
    pub rate_rps: usize,
    pub rate_burst: usize,
    pub rate_clients: usize,
    pub walkers: usize,
    pub steps: usize,
    pub body_bytes: usize,
    pub recall: usize,
    pub energy_nodes: usize,
    pub music_seconds: usize,
    pub completion_tokens: usize,
    pub metacog_phases: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            concurrent_walks: 2,
            concurrent_work: 2,
            concurrent_maintenance: 1,
            concurrent_benchmarks: 1,
            rate_rps: 10,
            rate_burst: 30,
            rate_clients: 1024,
            walkers: 8,
            steps: 32,
            body_bytes: 1024 * 1024,
            recall: 200,
            energy_nodes: 10_000,
            music_seconds: 30,
            completion_tokens: 4096,
            metacog_phases: 8,
        }
    }
}

impl Limits {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(|name| std::env::var(name).ok())
    }

    fn parse(get: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let mut limits = Self::default();
        for (name, field) in [
            ("RGW_MAX_CONCURRENT_WALKS", &mut limits.concurrent_walks),
            ("RGW_MAX_CONCURRENT_WORK", &mut limits.concurrent_work),
            (
                "RGW_MAX_CONCURRENT_MAINTENANCE",
                &mut limits.concurrent_maintenance,
            ),
            (
                "RGW_MAX_CONCURRENT_BENCHMARKS",
                &mut limits.concurrent_benchmarks,
            ),
            ("RGW_RATE_LIMIT_RPS", &mut limits.rate_rps),
            ("RGW_RATE_LIMIT_BURST", &mut limits.rate_burst),
            ("RGW_RATE_LIMIT_CLIENTS", &mut limits.rate_clients),
            ("RGW_MAX_WALKERS", &mut limits.walkers),
            ("RGW_MAX_STEPS", &mut limits.steps),
            ("RGW_MAX_BODY_BYTES", &mut limits.body_bytes),
            ("RGW_MAX_RECALL_LIMIT", &mut limits.recall),
            ("RGW_MAX_ENERGY_NODES", &mut limits.energy_nodes),
            ("RGW_MAX_MUSIC_SECONDS", &mut limits.music_seconds),
            ("RGW_MAX_COMPLETION_TOKENS", &mut limits.completion_tokens),
            ("RGW_MAX_METACOG_PHASES", &mut limits.metacog_phases),
        ] {
            if let Some(value) = get(name) {
                *field = value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0 && n <= i32::MAX as usize && n <= Semaphore::MAX_PERMITS)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{name} must be a positive integer no greater than {}",
                            i32::MAX
                        )
                    })?;
            }
        }
        Ok(limits)
    }

    pub fn walk(&self, walkers: usize, steps: usize) -> (usize, usize) {
        let walkers = if walkers == 0 {
            rayon::current_num_threads()
        } else {
            walkers
        };
        (walkers.clamp(1, self.walkers), steps.clamp(1, self.steps))
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Identity {
    Key(u64),
    Ip(IpAddr),
    Unknown,
}

struct Bucket {
    tokens: f64,
    updated: Instant,
}

pub struct Admission {
    pub limits: Limits,
    walks: Arc<Semaphore>,
    work: Arc<Semaphore>,
    maintenance: Arc<Semaphore>,
    benchmarks: Arc<Semaphore>,
    buckets: Mutex<HashMap<Identity, Bucket>>,
    key_hash: RandomState,
}

impl Admission {
    pub fn new(limits: Limits) -> Self {
        Self {
            walks: Arc::new(Semaphore::new(limits.concurrent_walks)),
            work: Arc::new(Semaphore::new(limits.concurrent_work)),
            maintenance: Arc::new(Semaphore::new(limits.concurrent_maintenance)),
            benchmarks: Arc::new(Semaphore::new(limits.concurrent_benchmarks)),
            limits,
            buckets: Mutex::new(HashMap::new()),
            key_hash: RandomState::new(),
        }
    }

    fn identity(&self, req: &Request) -> Identity {
        if let Some(key) = req.headers().get("X-RGW-Key") {
            // Retain neither plaintext credentials nor user-controlled header strings.
            Identity::Key(self.key_hash.hash_one(key.as_bytes()))
        } else if let Some(ConnectInfo(peer)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
            Identity::Ip(peer.ip())
        } else {
            Identity::Unknown
        }
    }

    fn take_token(&self, identity: Identity, now: Instant) -> Result<(), u64> {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        let burst = self.limits.rate_burst as f64;
        let rps = self.limits.rate_rps as f64;
        if !buckets.contains_key(&identity) && buckets.len() >= self.limits.rate_clients {
            // Evict only fully replenished buckets: churning identities cannot reset debt.
            buckets.retain(|_, b| {
                b.tokens + now.saturating_duration_since(b.updated).as_secs_f64() * rps < burst
            });
            if buckets.len() >= self.limits.rate_clients {
                return Err(1);
            }
        }
        let bucket = buckets.entry(identity).or_insert(Bucket {
            tokens: burst,
            updated: now,
        });
        bucket.tokens = (bucket.tokens
            + now.saturating_duration_since(bucket.updated).as_secs_f64() * rps)
            .min(burst);
        bucket.updated = now.max(bucket.updated);
        if bucket.tokens < 1.0 {
            Err(((1.0 - bucket.tokens) / rps).ceil().max(1.0) as u64)
        } else {
            bucket.tokens -= 1.0;
            Ok(())
        }
    }

    fn acquire(&self, path: &str) -> Result<Vec<OwnedSemaphorePermit>, ()> {
        let class = match path {
            "/walk" | "/v1/chat/completions" | "/benchmark" | "/dream" => Some(&self.walks),
            "/ingest/signal" | "/tools/execute" => Some(&self.work),
            "/prune" | "/music/generate" | "/speak" => Some(&self.maintenance),
            p if p.starts_with("/motor/proposals/") && p.ends_with("/approve") => Some(&self.work),
            _ => None,
        };
        let mut permits = Vec::new();
        if let Some(class) = class {
            permits.push(class.clone().try_acquire_owned().map_err(|_| ())?);
        }
        if path == "/benchmark" {
            permits.push(
                self.benchmarks
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| ())?,
            );
        }
        Ok(permits)
    }
}

fn rejection(status: StatusCode, message: &str, retry: Option<u64>) -> Response {
    let mut response = (status, Json(serde_json::json!({"error": message}))).into_response();
    if let Some(seconds) = retry {
        response
            .headers_mut()
            .insert(RETRY_AFTER, seconds.to_string().parse().unwrap());
    }
    response
}

/// Mounted inside authentication: unknown keys cannot allocate buckets or reserve work.
/// No semaphore waiters are created. Permits cover body reading and the full handler.
pub async fn admit(
    State(admission): State<Arc<Admission>>,
    mut req: Request,
    next: Next,
) -> Response {
    let permits = if req.method() == axum::http::Method::POST {
        if let Err(retry) = admission.take_token(admission.identity(&req), Instant::now()) {
            crate::metrics::ADMISSION_RATE_REJECTED.fetch_add(1, Ordering::Relaxed);
            return rejection(
                StatusCode::TOO_MANY_REQUESTS,
                "request rate exceeded",
                Some(retry),
            );
        }
        match admission.acquire(req.uri().path()) {
            Ok(permits) => permits,
            Err(()) => {
                crate::metrics::ADMISSION_CONCURRENCY_REJECTED.fetch_add(1, Ordering::Relaxed);
                return rejection(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "route capacity exhausted",
                    Some(1),
                );
            }
        }
    } else {
        Vec::new()
    };

    // Count bytes actually received, including chunked bodies and handlers that ignore bodies.
    DefaultBodyLimit::max(admission.limits.body_bytes).apply(&mut req);
    let (parts, body) = req.into_parts();
    let bytes = match Bytes::from_request(Request::from_parts(parts.clone(), body), &()).await {
        Ok(bytes) => bytes,
        Err(error) => {
            let oversized = error.status() == StatusCode::PAYLOAD_TOO_LARGE;
            if oversized {
                crate::metrics::ADMISSION_BODY_REJECTED.fetch_add(1, Ordering::Relaxed);
                return rejection(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request body too large",
                    None,
                );
            }
            return rejection(StatusCode::BAD_REQUEST, "cannot read request body", None);
        }
    };
    let request = Request::from_parts(parts, Body::from(bytes));
    // An abandoned HTTP request must not free capacity while its blocking work continues.
    // The admitted task keeps its permits until the handler actually finishes.
    if permits.is_empty() {
        return next.run(request).await;
    }
    tokio::spawn(async move {
        let _permits = permits;
        next.run(request).await
    })
    .await
    .unwrap_or_else(|_| {
        rejection(
            StatusCode::INTERNAL_SERVER_ERROR,
            "request task failed",
            None,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::Request,
        routing::{get, post},
    };
    use std::time::Duration;
    use tower::ServiceExt;

    fn request(path: &str, key: Option<&str>, body: &str) -> Request<Body> {
        let mut request = Request::post(path).header("content-type", "application/json");
        if let Some(key) = key {
            request = request.header("X-RGW-Key", key);
        }
        request.body(Body::from(body.to_owned())).unwrap()
    }

    fn app(admission: Arc<Admission>) -> Router {
        Router::new()
            .route("/walk", post(|| async { StatusCode::OK }))
            .route("/ingest/signal", post(|| async { StatusCode::OK }))
            .route(
                "/self/save",
                post(|Json(_): Json<serde_json::Value>| async { StatusCode::OK }),
            )
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/ready", get(|| async { StatusCode::OK }))
            .route("/metrics", get(|| async { StatusCode::OK }))
            .layer(axum::middleware::from_fn_with_state(admission, admit))
    }

    #[tokio::test]
    async fn saturated_class_rejects_immediately_and_releases_permits() {
        let admission = Arc::new(Admission::new(Limits {
            concurrent_walks: 1,
            ..Default::default()
        }));
        let permit = admission.walks.clone().acquire_owned().await.unwrap();
        let app = app(admission);
        let response = tokio::time::timeout(
            Duration::from_millis(100),
            app.clone()
                .oneshot(request("/walk", Some("test-key"), "{}")),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "1");
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(
            app.clone()
                .oneshot(request("/ingest/signal", Some("test-key"), "{}"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        for path in ["/health", "/ready", "/metrics"] {
            assert_eq!(
                app.clone()
                    .oneshot(Request::get(path).body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
        }
        drop(permit);
        for _ in 0..2 {
            assert_eq!(
                app.clone()
                    .oneshot(request("/walk", Some("test-key"), "{}"))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
        }
    }

    #[tokio::test]
    async fn rate_bucket_is_shared_across_routes_but_isolated_by_key() {
        let admission = Arc::new(Admission::new(Limits {
            rate_burst: 1,
            rate_rps: 1,
            ..Default::default()
        }));
        let app = app(admission);
        assert_eq!(
            app.clone()
                .oneshot(request("/walk", Some("first"), "{}"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let response = app
            .clone()
            .oneshot(request("/self/save", Some("first"), "{}"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "1");
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(
            app.clone()
                .oneshot(request("/walk", Some("second"), "{}"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            app.oneshot(
                Request::get("/health")
                    .header("X-RGW-Key", "first")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn all_expensive_classes_and_benchmark_share_their_caps() {
        let admission = Arc::new(Admission::new(Limits {
            concurrent_walks: 2,
            concurrent_work: 1,
            concurrent_maintenance: 1,
            concurrent_benchmarks: 1,
            ..Default::default()
        }));
        let benchmark = admission.acquire("/benchmark").unwrap();
        assert!(admission.acquire("/benchmark").is_err());
        let walk = admission.acquire("/walk").unwrap();
        for path in ["/walk", "/v1/chat/completions", "/dream"] {
            assert!(admission.acquire(path).is_err(), "{path}");
        }
        let work = admission.acquire("/ingest/signal").unwrap();
        for path in [
            "/ingest/signal",
            "/tools/execute",
            "/motor/proposals/1/approve",
        ] {
            assert!(admission.acquire(path).is_err(), "{path}");
        }
        let prune = admission.acquire("/prune").unwrap();
        for path in ["/prune", "/music/generate", "/speak"] {
            assert!(admission.acquire(path).is_err());
        }
        drop((benchmark, walk, work, prune));
        assert!(admission.acquire("/benchmark").is_ok());
    }

    #[tokio::test]
    async fn keyless_clients_share_by_peer_ip_not_port_or_forwarded_headers() {
        let admission = Arc::new(Admission::new(Limits {
            rate_burst: 1,
            rate_rps: 1,
            ..Default::default()
        }));
        let app = app(admission);
        let peer = std::net::Ipv4Addr::LOCALHOST;
        for (port, forwarded, expected) in [
            (1, "first", StatusCode::OK),
            (2, "second", StatusCode::TOO_MANY_REQUESTS),
        ] {
            let mut req = request("/self/save", None, "{}");
            req.extensions_mut()
                .insert(ConnectInfo(SocketAddr::from((peer, port))));
            req.headers_mut()
                .insert("x-forwarded-for", forwarded.parse().unwrap());
            assert_eq!(app.clone().oneshot(req).await.unwrap().status(), expected);
        }
        let mut req = request("/self/save", Some("separate-key"), "{}");
        req.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from((peer, 1))));
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::OK
        );
        let mut req = request("/self/save", None, "{}");
        req.extensions_mut().insert(ConnectInfo(SocketAddr::from((
            std::net::Ipv4Addr::BROADCAST,
            1,
        ))));
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn disconnect_keeps_capacity_until_admitted_work_finishes() {
        let admission = Arc::new(Admission::new(Limits {
            concurrent_walks: 1,
            ..Default::default()
        }));
        let entered = Arc::new(tokio::sync::Notify::new());
        let finish = Arc::new(tokio::sync::Notify::new());
        let app = Router::new()
            .route(
                "/walk",
                post({
                    let entered = entered.clone();
                    let finish = finish.clone();
                    move || {
                        let entered = entered.clone();
                        let finish = finish.clone();
                        async move {
                            entered.notify_one();
                            finish.notified().await;
                            StatusCode::OK
                        }
                    }
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                admission.clone(),
                admit,
            ));
        let first = tokio::spawn(app.clone().oneshot(request("/walk", None, "{}")));
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        first.abort();
        let _ = first.await;
        assert_eq!(
            app.clone()
                .oneshot(request("/walk", None, "{}"))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        finish.notify_one();
        // Acquiring the returned permit proves it survives cancellation, then is released.
        let _permit = tokio::time::timeout(Duration::from_secs(1), admission.walks.acquire())
            .await
            .unwrap()
            .unwrap();
    }

    #[test]
    fn bucket_refills_and_does_not_evict_active_clients() {
        let admission = Admission::new(Limits {
            rate_burst: 2,
            rate_rps: 2,
            rate_clients: 1,
            ..Default::default()
        });
        let now = Instant::now();
        let first = Identity::Key(1);
        assert!(admission.take_token(first, now).is_ok());
        assert!(admission.take_token(first, now).is_ok());
        assert_eq!(admission.take_token(first, now), Err(1));
        assert_eq!(admission.take_token(Identity::Key(2), now), Err(1));
        assert!(
            admission
                .take_token(first, now + Duration::from_millis(500))
                .is_ok()
        );
        assert!(
            admission
                .take_token(Identity::Key(2), now + Duration::from_secs(2))
                .is_ok()
        );
    }

    #[tokio::test]
    async fn body_limit_counts_bytes_and_can_exceed_axums_default() {
        let small_app = app(Arc::new(Admission::new(Limits {
            body_bytes: 4,
            ..Default::default()
        })));
        assert_eq!(
            small_app
                .clone()
                .oneshot(request("/self/save", None, "1234"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let response = small_app
            .oneshot(request("/self/save", None, "12345"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(response.headers()["content-type"], "application/json");
        let large_app = app(Arc::new(Admission::new(Limits {
            body_bytes: 3 * 1024 * 1024,
            ..Default::default()
        })));
        let large_json = serde_json::to_string(&"x".repeat(2 * 1024 * 1024)).unwrap();
        assert_eq!(
            large_app
                .oneshot(request("/self/save", None, &large_json))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }

    #[test]
    fn clamps_bound_extreme_values_and_auto_walkers() {
        let limits = Limits {
            walkers: 2,
            steps: 3,
            ..Default::default()
        };
        assert_eq!(limits.walk(usize::MAX, usize::MAX), (2, 3));
        assert_eq!(limits.walk(1, 2), (1, 2));
        let (walkers, steps) = limits.walk(0, 0);
        assert!((1..=2).contains(&walkers));
        assert_eq!(steps, 1);
    }

    #[test]
    fn invalid_configuration_fails_closed_without_echoing_values() {
        for value in [
            "0",
            "-1",
            "NaN",
            "",
            "sensitive-value",
            "999999999999999999999",
        ] {
            let error = Limits::parse(|name| (name == "RGW_MAX_STEPS").then(|| value.to_owned()))
                .unwrap_err();
            assert!(error.to_string().contains("RGW_MAX_STEPS"));
            assert!(!error.to_string().contains("sensitive-value"));
        }
    }
}
