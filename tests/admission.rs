//! Actual router and walker regression coverage against the disposable fixture DB.
mod common;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rgw::{
    admission::{Admission, Limits},
    api::{AppState, app_router},
    core::{CognitiveMode, SelfModel},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

fn request(method: &str, path: &str, key: Option<&str>, body: Value) -> Request<Body> {
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(key) = key {
        builder = builder.header("X-RGW-Key", key);
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

async fn state(pool: sqlx::PgPool, mode: CognitiveMode, limits: Limits) -> Arc<AppState> {
    let mut model = SelfModel::new();
    model.mode = mode;
    let self_model = Arc::new(tokio::sync::Mutex::new(model));
    let diverger = rgw::diverger::Diverger::with_energy_limit(
        pool.clone(),
        self_model.clone(),
        "",
        limits.energy_nodes,
    );
    Arc::new(AppState {
        admission: Arc::new(Admission::new(limits)),
        pool,
        diverger,
        self_model,
        ollama_url: String::new(),
        expression_model: String::new(),
        julian_url: String::new(),
        provider: None,
        motor_gate: Arc::new(rgw::motor::MotorGate::new(rgw::motor::MotorMode::Off)),
    })
}

#[tokio::test]
async fn router_enforces_limits_scopes_and_real_walk_clamps() -> anyhow::Result<()> {
    // This integration binary contains one test, so these test-only env settings are isolated.
    unsafe {
        std::env::set_var("RGW_API_KEY", "test-admin-key");
        std::env::set_var("RGW_READ_KEY", "test-read-key");
        std::env::set_var("RGW_INGEST_KEY", "test-ingest-key");
        std::env::set_var("RGW_SEED_MODE", "fixed");
    }
    let pool = common::fixture::fixture_pool("admission_router").await?;
    for mode in [CognitiveMode::Compliant, CognitiveMode::Autonomous] {
        let state = state(
            pool.clone(),
            mode,
            Limits {
                walkers: 2,
                steps: 1,
                ..Default::default()
            },
        )
        .await;
        let app = app_router(state.clone());
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/walk",
                Some("test-admin-key"),
                json!({"n_walkers": usize::MAX, "steps": usize::MAX, "seed": 1}),
            ))
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let output: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
        assert_eq!(output["walker_count"], 2);
        assert!(
            output["total_hops"].as_u64().unwrap() <= 4,
            "at most seed plus one hop per walker: {output}"
        );
        // Existing chat path must enforce the same caps before its LLM fallback.
        let before = state.self_model.lock().await.total_signals_processed;
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                Some("test-admin-key"),
                json!({"messages": [], "rgw_walkers": usize::MAX, "rgw_steps": usize::MAX}),
            ))
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let chat: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
        let context = chat["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(context.contains("[2 walkers,"), "{context}");
        assert!(
            context.contains("2 hops,")
                || context.contains("3 hops,")
                || context.contains("4 hops,"),
            "{context}"
        );
        assert!(state.self_model.lock().await.total_signals_processed > before);
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/benchmark",
                Some("test-admin-key"),
                json!({}),
            ))
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let benchmark: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
        for result in benchmark["results"].as_array().unwrap() {
            assert!(result["walkers"].as_u64().unwrap() <= 2);
            assert_eq!(result["steps"], 1);
        }
    }

    let state = state(
        pool,
        CognitiveMode::Compliant,
        Limits {
            concurrent_walks: 1,
            body_bytes: 256,
            rate_burst: 3,
            rate_rps: 1,
            energy_nodes: 2,
            ..Default::default()
        },
    )
    .await;
    state.diverger.seed_energy((1..100).collect(), 0.3).await;
    assert_eq!(state.diverger.stats().await.active_nodes, 2);
    let app = app_router(state.clone());
    for key in ["test-read-key", "test-ingest-key"] {
        assert_eq!(
            app.clone()
                .oneshot(request("POST", "/benchmark", Some(key), json!({})))
                .await?
                .status(),
            StatusCode::FORBIDDEN
        );
    }
    assert_eq!(
        app.clone()
            .oneshot(request(
                "GET",
                "/benchmark",
                Some("test-admin-key"),
                json!({})
            ))
            .await?
            .status(),
        StatusCode::METHOD_NOT_ALLOWED
    );
    // Auth failures neither reserve capacity nor consume the real key's bucket.
    assert_eq!(
        app.clone()
            .oneshot(request("POST", "/dream", Some("invalid-key"), json!({})))
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let oversized = app
        .clone()
        .oneshot(request(
            "POST",
            "/self/save",
            Some("test-admin-key"),
            json!({"extra": "x".repeat(300)}),
        ))
        .await?;
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);

    // Poll a real dream request while the model lock is held: admission has succeeded,
    // but its handler cannot finish. The second expensive request must fail immediately.
    let lock = state.self_model.lock().await;
    let mut held =
        Box::pin(
            app.clone()
                .oneshot(request("POST", "/dream", Some("test-admin-key"), json!({}))),
        );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut held)
            .await
            .is_err()
    );
    let response = app
        .clone()
        .oneshot(request("POST", "/walk", Some("test-admin-key"), json!({})))
        .await?;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["retry-after"], "1");
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/self/save",
            Some("test-admin-key"),
            json!({}),
        ))
        .await?;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "1");
    for (path, key) in [
        ("/health", None),
        ("/ready", None),
        ("/metrics", Some("test-read-key")),
    ] {
        let response = app
            .clone()
            .oneshot(request("GET", path, key, json!({})))
            .await?;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        assert!(!response.headers().contains_key("retry-after"));
    }
    drop(lock);
    assert_eq!(held.await?.status(), StatusCode::OK);
    let metrics = app
        .oneshot(request("GET", "/metrics", Some("test-read-key"), json!({})))
        .await?;
    let metrics: Value = serde_json::from_slice(&to_bytes(metrics.into_body(), usize::MAX).await?)?;
    assert!(metrics["admission_rate_rejected"].as_u64().unwrap() >= 1);
    assert!(metrics["admission_body_rejected"].as_u64().unwrap() >= 1);
    assert!(metrics["admission_energy_rejected"].as_u64().unwrap() >= 1);
    Ok(())
}
