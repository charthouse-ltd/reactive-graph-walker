//! Motor callbacks use Julian's existing episode schema and the real HTTP router.
mod common;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rgw::{
    api::{AppState, app_router},
    core::SelfModel,
    motor::{MotorCommand, MotorGate, MotorMode},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

async fn state(pool: sqlx::PgPool) -> Arc<AppState> {
    let self_model = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
    let gate = Arc::new(MotorGate::new(MotorMode::Propose));
    gate.restore_outcomes(&pool).await;
    Arc::new(AppState {
        admission: Arc::new(rgw::admission::Admission::new(Default::default())),
        diverger: rgw::diverger::Diverger::new(pool.clone(), self_model.clone(), ""),
        pool,
        self_model,
        motor_gate: gate,
        ollama_url: String::new(),
        expression_model: String::new(),
        julian_url: String::new(),
        provider: None,
    })
}

async fn callback(state: Arc<AppState>, path: &str, value: Value) -> anyhow::Result<()> {
    let response = app_router(state)
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("X-RGW-Key", "test-outcome-key")
                .body(Body::from(value.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    Ok(())
}

async fn listed(state: Arc<AppState>) -> anyhow::Result<Value> {
    let response = app_router(state)
        .oneshot(
            Request::get("/motor/proposals")
                .header("X-RGW-Key", "test-outcome-key")
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    Ok(serde_json::from_slice(
        &to_bytes(response.into_body(), 100_000).await?,
    )?)
}

#[tokio::test]
async fn callbacks_persist_update_restore_and_degrade() -> anyhow::Result<()> {
    // One router test in this binary owns the authentication environment.
    unsafe {
        std::env::set_var("RGW_API_KEY", "test-outcome-key");
    }
    let pool = common::fixture::fixture_pool("motor_outcomes").await?;
    let state = state(pool.clone()).await;
    let id = uuid::Uuid::new_v4();
    state
        .motor_gate
        .dispatch(
            "test",
            "/execution",
            MotorCommand {
                correlation_id: id,
                action: "search".into(),
                domain: "science".into(),
                walker_context: String::new(),
                expression_seeds: vec![],
                confidence: 0.8,
                novelty: 0.5,
                search_query: None,
                params: None,
                audience_id: None,
            },
        )
        .await;
    let busy_model = state.self_model.lock().await;
    let before = serde_json::to_value(&*busy_model)?;
    callback(
        state.clone(),
        "/diverger/outcome",
        json!({
            "correlation_id": id, "outcome": "completed", "success": true, "lesson": "keep result"
        }),
    )
    .await?;
    drop(busy_model); // Callback persistence must not wait for a walk holding this lock.
    let (headline, detail): (String, String) = sqlx::query_as(
        "SELECT headline, detail FROM rgw_episodes WHERE source_type = 'motor_outcome' AND source_ref = $1")
        .bind(id.to_string()).fetch_one(&pool).await?;
    assert_eq!(headline, "completed");
    let row: Value = serde_json::from_str(&detail)?;
    assert_eq!(
        row["origin"],
        json!({"action":"search", "domain":"science", "target":"/execution"})
    );
    assert_eq!(row["success"], true);
    assert_eq!(row["lesson"], "keep result");
    assert_eq!(
        serde_json::to_value(&*state.self_model.lock().await)?,
        before,
        "compliant is observe only"
    );

    callback(state.clone(), "/outcomes", json!({
        "correlation_id": id, "outcome": "corrected", "success": false, "lesson": "retry differently"
    })).await?;
    let count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM rgw_episodes WHERE source_type = 'motor_outcome' AND source_ref = $1",
    )
    .bind(id.to_string())
    .fetch_one(&pool)
    .await?;
    assert_eq!(count, 1);
    let restored = self::state(pool.clone()).await;
    let listing = listed(restored.clone()).await?;
    assert_eq!(listing["outcomes"][0]["outcome"], "corrected");
    assert_eq!(listing["outcomes"][0]["success"], false);
    assert_eq!(listing["outcomes"][0]["lesson"], "retry differently");
    assert_eq!(listing["outcomes"][0]["origin"], row["origin"]);

    // Restored correlation metadata permits learning without a pending proposal.
    {
        let mut model = restored.self_model.lock().await;
        model.mode = rgw::core::CognitiveMode::Autonomous;
        model.self_mod_stage = rgw::core::SelfModStage::SelectionLive;
    }
    let payload = json!({"correlation_id":id,"outcome":"failed","success":false});
    let (a, b) = tokio::join!(
        callback(restored.clone(), "/outcomes", payload.clone()),
        callback(restored.clone(), "/diverger/outcome", payload.clone())
    );
    a?;
    b?;
    assert_eq!(restored.self_model.lock().await.total_signals_processed, 1);
    let restarted_again = self::state(pool.clone()).await;
    {
        let mut model = restarted_again.self_model.lock().await;
        model.mode = rgw::core::CognitiveMode::Autonomous;
        model.self_mod_stage = rgw::core::SelfModStage::SelectionLive;
    }
    callback(restarted_again.clone(), "/outcomes", payload).await?;
    assert_eq!(
        restarted_again
            .self_model
            .lock()
            .await
            .total_signals_processed,
        0,
        "no replay after restart"
    );

    let unknown = uuid::Uuid::new_v4();
    callback(
        restored.clone(),
        "/outcomes",
        json!({"correlation_id":unknown,"outcome":"late"}),
    )
    .await?;
    let detail: String = sqlx::query_scalar(
        "SELECT detail FROM rgw_episodes WHERE source_type = 'motor_outcome' AND source_ref = $1",
    )
    .bind(unknown.to_string())
    .fetch_one(&pool)
    .await?;
    let unmatched: Value = serde_json::from_str(&detail)?;
    assert!(unmatched["origin"].is_null());
    assert!(unmatched["success"].is_null());
    assert_eq!(restored.self_model.lock().await.total_signals_processed, 1);

    // Isolate search_path so public's table cannot hide a missing fixture table.
    pool.close().await;
    let missing = sqlx::postgres::PgPoolOptions::new()
        .after_connect(|conn, _| {
            Box::pin(async move {
                sqlx::query("SET search_path TO motor_outcomes")
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;
    sqlx::query("ALTER TABLE rgw_episodes RENAME TO missing_episodes")
        .execute(&missing)
        .await?;
    let unavailable = self::state(missing.clone()).await;
    callback(
        unavailable.clone(),
        "/outcomes",
        json!({"correlation_id":unknown,"outcome":"still accepted"}),
    )
    .await?;
    assert_eq!(
        listed(unavailable).await?["outcomes"][0]["outcome"],
        "still accepted"
    );
    // Episodes were already required for /recall; preserve that readiness contract.
    assert_eq!(
        rgw::api::readiness(&missing).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    Ok(())
}

#[tokio::test]
async fn restore_is_bounded_and_skips_malformed_motor_rows() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("motor_restore").await?;
    for i in 0..105 {
        let id = uuid::Uuid::new_v4();
        sqlx::query("INSERT INTO rgw_episodes (occurred_at, kind, headline, detail, source_type, source_ref)
            VALUES (NOW() + $1 * interval '1 second', 'motor_outcome', 'completed', $2, 'motor_outcome', $3)")
            .bind(i as f64).bind(json!({"correlation_id":id,"outcome":"completed","success":true,
                "lesson":null,"origin":null,"received_at":i,"reflected":false}).to_string())
            .bind(id.to_string()).execute(&pool).await?;
    }
    sqlx::query("INSERT INTO rgw_episodes (occurred_at, kind, headline, detail, source_type, source_ref)
        VALUES (NOW() + interval '1 day', 'motor_outcome', 'bad', 'invalid JSON', 'motor_outcome', 'bad')")
        .execute(&pool).await?;
    let gate = MotorGate::new(MotorMode::Off);
    gate.restore_outcomes(&pool).await;
    let snapshot = gate.snapshot();
    let outcomes = snapshot["outcomes"].as_array().unwrap();
    assert_eq!(
        outcomes.len(),
        99,
        "read at most 100 rows, skip the malformed one"
    );
    assert_eq!(outcomes.last().unwrap()["received_at"], 104);
    Ok(())
}
