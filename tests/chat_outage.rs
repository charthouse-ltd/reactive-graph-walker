//! When no language model answers, a chat completion is a 503. Walk context
//! and retrieved memory are prompt material and must not come back as the reply.
mod common;

use axum::{Json, extract::State, http::StatusCode};
use common::fixture;
use rgw::{api::AppState, core::SelfModel, motor::{MotorGate, MotorMode}, openai};
use serde_json::json;
use std::sync::Arc;

/// Nothing listens here, so a model call fails at connect without leaving the machine.
const NO_MODEL_URL: &str = "http://127.0.0.1:1";

fn state(pool: sqlx::PgPool, ollama_url: &str) -> Arc<AppState> {
    let self_model = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
    Arc::new(AppState {
        admission: Arc::new(rgw::admission::Admission::new(Default::default())),
        diverger: rgw::diverger::Diverger::new(pool.clone(), self_model.clone(), ""),
        pool,
        self_model,
        motor_gate: Arc::new(MotorGate::new(MotorMode::Off)),
        ollama_url: ollama_url.to_string(),
        expression_model: "test-model".into(),
        julian_url: String::new(),
        provider: None,
    })
}

fn request(bypass: bool) -> openai::ChatRequest {
    serde_json::from_value(json!({
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "What do you remember about the hub?"}
        ],
        "rgw_audience": "user:test",
        "rgw_bypass": bypass,
    }))
    .unwrap()
}

#[tokio::test]
async fn no_model_is_a_503_and_nothing_is_recorded_as_said() -> anyhow::Result<()> {
    let pool = fixture::fixture_pool("chat_outage").await?;
    for bypass in [false, true] {
        let state = state(pool.clone(), NO_MODEL_URL);
        let (status, message) = openai::chat_completions(State(state.clone()), Json(request(bypass)))
            .await
            .expect_err("an outage must not be a completion");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "bypass={bypass}");
        for internal in ["RGW COGNITION", "Relevant memory", "fixture node"] {
            assert!(!message.contains(internal), "error text leaked {internal}");
        }
        let model = state.self_model.lock().await;
        assert!(
            model.audience_model.values().all(|a| a.last_message_sent.is_empty()),
            "a failed completion was recorded as output to the audience (bypass={bypass})"
        );
    }
    Ok(())
}

#[tokio::test]
async fn an_empty_model_reply_is_a_503_and_a_real_reply_is_returned() -> anyhow::Result<()> {
    let pool = fixture::fixture_pool("chat_reply").await?;
    for (content, expected) in [("   ", None), ("hello from the model", Some("hello from the model"))] {
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || async move {
                Json(json!({"choices": [{"message": {"role": "assistant", "content": content}}]}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}", listener.local_addr()?);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        for bypass in [false, true] {
            let state = state(pool.clone(), &url);
            let result = openai::chat_completions(State(state), Json(request(bypass))).await;
            match expected {
                None => assert_eq!(result.unwrap_err().0, StatusCode::SERVICE_UNAVAILABLE),
                Some(text) => {
                    let Json(response) = result.expect("a model reply is a completion");
                    assert_eq!(response.choices[0].message.content, text);
                    assert_eq!(response.choices[0].finish_reason, "stop");
                }
            }
        }
        server.abort();
    }
    Ok(())
}

#[tokio::test]
async fn bypass_retrieval_skips_excluded_source_types() -> anyhow::Result<()> {
    let pool = fixture::fixture_pool("chat_retrieval").await?;
    let cue = fixture::axis(0);
    let excluded = vec!["trade".to_string(), "brain_dump".to_string()];
    let ids = |rows: Vec<(i32, f32, String)>| rows.into_iter().map(|r| r.0).collect::<Vec<_>>();

    let before = ids(rgw::db::find_similar_documents(&pool, &cue, 3, 0.8, &excluded).await?);
    assert_eq!(before[0], fixture::A1, "the nearest node is served while its type is allowed");

    // Stored case and padding must not let a private row through.
    sqlx::query("UPDATE memory_vectors SET source_type = ' Trade ' WHERE id = $1")
        .bind(fixture::A1).execute(&pool).await?;
    let after = ids(rgw::db::find_similar_documents(&pool, &cue, 3, 0.8, &excluded).await?);
    assert!(!after.contains(&fixture::A1), "an excluded source type reached prompt text");
    assert!(!after.is_empty(), "other rows are still retrieved");

    let unfiltered = ids(rgw::db::find_similar_documents(&pool, &cue, 3, 0.8, &[]).await?);
    assert_eq!(unfiltered[0], fixture::A1, "an empty list disables the filter, as for the narrative");
    Ok(())
}
