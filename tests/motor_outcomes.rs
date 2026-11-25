//! Motor callbacks use Julian's existing episode schema and the real HTTP router.
mod common;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rgw::{
    api::{AppState, app_router},
    core::SelfModel,
    motor::{DispatchOutcome, MotorCommand, MotorGate, MotorMode, ProposalStatus},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

// An isolated fake execution backend; never contact the running RGW service.
async fn backend() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new().route("/api/admin/rgw/execute",
        axum::routing::post(|| async { "ok" }));
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    (url, task)
}

async fn approve_pending(gate: &MotorGate) {
    let pending = gate.snapshot()["pending"].as_array().unwrap().clone();
    for p in pending {
        assert_eq!(gate.approve(p["id"].as_u64().unwrap()).await.unwrap().status,
            ProposalStatus::ApprovedSent);
    }
}

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
    callback_status(state, path, value, StatusCode::ACCEPTED).await
}

async fn callback_status(state: Arc<AppState>, path: &str, value: Value, expected: StatusCode) -> anyhow::Result<()> {
    let response = app_router(state)
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("X-RGW-Key", "test-outcome-key")
                .body(Body::from(value.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), expected);
    let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 100_000).await?)?;
    if expected != StatusCode::ACCEPTED { assert!(body["error"].is_string()); }
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
    let (url, server) = backend().await;
    let id = uuid::Uuid::new_v4();
    state
        .motor_gate
        .dispatch(
            "test",
            &url,
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
    approve_pending(&state.motor_gate).await;
    server.abort();
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
        json!({"action":"search", "domain":"science", "target":url})
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
    // Cold callback path must consult durable history, even without restore_outcomes.
    let cold = MotorGate::new(MotorMode::Off);
    cold.record_outcome(&pool, &state.self_model, rgw::motor::MotorOutcome {
        correlation_id: id, outcome: "corrected".into(), success: Some(false),
        lesson: Some("retry differently".into()),
    }).await?;
    assert!(cold.snapshot()["outcomes"][0]["origin"].is_object());
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
    for path in ["/outcomes", "/diverger/outcome"] {
        callback_status(restored.clone(), path,
            json!({"correlation_id":unknown,"outcome":"late"}), StatusCode::NOT_FOUND).await?;
    }
    let count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM rgw_episodes WHERE source_ref = $1")
        .bind(unknown.to_string()).fetch_one(&pool).await?;
    assert_eq!(count, 0, "unknown callbacks must not grow the prune-exempt table");
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
    callback_status(unavailable.clone(), "/outcomes",
        json!({"correlation_id":unknown,"outcome":"not accepted"}),
        StatusCode::SERVICE_UNAVAILABLE).await?;
    assert!(listed(unavailable).await?["outcomes"].as_array().unwrap().is_empty());
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
    assert!(gate.consolidate_episodes().episodes.is_empty(), "unmatched callbacks are not verified action evidence");
    Ok(())
}

#[tokio::test]
async fn recall_serves_autobiography_but_never_motor_outcome_rows() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("motor_recall").await?;
    sqlx::query("INSERT INTO rgw_episodes (occurred_at, kind, location, headline, source_type, source_ref)
        VALUES (NOW() AT TIME ZONE 'UTC', 'post', 'Lisbon, Portugal', 'Wrote about tides', 'post', '1')")
        .execute(&pool).await?;
    let gate = MotorGate::new(MotorMode::Propose);
    let (url, server) = backend().await;
    let model = tokio::sync::Mutex::new(SelfModel::new());
    let id = uuid::Uuid::new_v4();
    gate.dispatch("test", &url, MotorCommand {
        correlation_id: id, action: "search".into(), domain: "science".into(),
        walker_context: String::new(), expression_seeds: vec![], confidence: 0.8, novelty: 0.5,
        search_query: None, params: None, audience_id: None,
    }).await;
    approve_pending(&gate).await;
    server.abort();
    gate.record_outcome(&pool, &model, rgw::motor::MotorOutcome {
        correlation_id: id, outcome: "completed".into(), success: Some(true),
        lesson: Some("internal lesson".into()),
    }).await?;
    let stored: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM rgw_episodes WHERE kind = 'motor_outcome'").fetch_one(&pool).await?;
    assert_eq!(stored, 1, "the outcome is still stored for inspection and idempotency");

    sqlx::query("INSERT INTO rgw_episodes (occurred_at, kind, location, headline, source_type, source_ref) VALUES (NOW(), 'action_outcome', 'Lisbon', 'skipped', 'action_outcome', 'walk:test')").execute(&pool).await?;

    for place in [None, Some("Lisbon")] {
        let episodes = rgw::db::recall_window(&pool, "-infinity", "infinity", place, 50).await?;
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].kind, "post");
        let served = serde_json::to_string(&episodes)?;
        for internal in [url.as_str(), "internal lesson", &id.to_string()] {
            assert!(!served.contains(internal), "recall leaked {internal}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn verified_memory_survives_restart_and_dreams_never_write_episodes() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("episodic_dream").await?;
    let gate = Arc::new(MotorGate::new(MotorMode::Propose));
    let model = Arc::new(tokio::sync::Mutex::new(SelfModel::new()));
    let (url, server) = backend().await;
    for i in 0..3 {
        let id = uuid::Uuid::new_v4();
        gate.dispatch("test", &url, MotorCommand {
            correlation_id: id, action: "search".into(), domain: "science".into(),
            walker_context: String::new(), expression_seeds: vec![], confidence: 0.8,
            novelty: 0.5, search_query: None, params: None, audience_id: None,
        }).await;
        approve_pending(&gate).await;
        let callback = rgw::motor::MotorOutcome {
            correlation_id: id, outcome: format!("verified {i}"), success: Some(false),
            lesson: Some("retry differently".into()),
        };
        gate.record_outcome(&pool, &model, callback.clone()).await?;
        gate.record_outcome(&pool, &model, callback).await?;
    }
    server.abort();
    let context = gate.consolidate_episodes();
    assert_eq!(context.episodes.len(), 3);
    assert_eq!(context.lessons.len(), 1);
    assert_eq!(context.lessons[0].evidence_count, 3);
    assert!(context.lessons[0].confidence > 0.9);
    let restored = Arc::new(MotorGate::new(MotorMode::Off));
    restored.restore_outcomes(&pool).await;
    assert_eq!(restored.consolidate_episodes().episodes.len(), 3);
    model.lock().await.mode = rgw::core::CognitiveMode::Autonomous;
    let before: i64 = sqlx::query_scalar("SELECT count(*) FROM rgw_episodes").fetch_one(&pool).await?;
    let report = rgw::dream::dream(&pool, &model, &restored, rgw::dream::DreamConfig { n_walks: 2, steps: 1, noise_magnitude: 0.0, ..Default::default() }).await;
    assert_eq!(report.walks_completed, 2);
    let after: i64 = sqlx::query_scalar("SELECT count(*) FROM rgw_episodes").fetch_one(&pool).await?;
    assert_eq!(before, after);
    assert_eq!(restored.consolidate_episodes().lessons[0].evidence_count, 3);
    assert!(rgw::db::recall_window(&pool, "-infinity", "infinity", None, 50).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn unsent_and_legacy_unmatched_callbacks_are_rejected() -> anyhow::Result<()> {
    use rgw::motor::{MotorOutcome, OutcomeError};
    let pool = common::fixture::fixture_pool("motor_rejections").await?;
    let model = tokio::sync::Mutex::new(SelfModel::new());
    for mode in [MotorMode::Off, MotorMode::Shadow, MotorMode::Propose, MotorMode::Live] {
        let gate = MotorGate::new(mode);
        let id = uuid::Uuid::new_v4();
        let result = gate.dispatch("test", "not-a-url", MotorCommand {
            correlation_id: id, action: "search".into(), domain: "science".into(),
            walker_context: String::new(), expression_seeds: vec![], confidence: 0.8,
            novelty: 0.5, search_query: None, params: None, audience_id: None,
        }).await;
        let callback = MotorOutcome { correlation_id: id, outcome: "claimed success".into(), success: Some(true), lesson: None };
        assert!(matches!(gate.record_outcome(&pool, &model, callback.clone()).await, Err(OutcomeError::Unmatched)));
        if let DispatchOutcome::Queued(p) = result {
            gate.reject(p, None).unwrap();
            assert!(matches!(gate.record_outcome(&pool, &model, callback.clone()).await, Err(OutcomeError::Unmatched)));
        }
        // A legacy persisted row with no origin must not authorize a retry, even after restore.
        sqlx::query("INSERT INTO rgw_episodes(occurred_at, kind, headline, detail, source_type, source_ref) VALUES (NOW(), 'motor_outcome', 'legacy', $1, 'motor_outcome', $2)")
            .bind(json!({"correlation_id":id,"outcome":"legacy","success":null,"lesson":null,"origin":null,"received_at":0,"reflected":false}).to_string())
            .bind(id.to_string()).execute(&pool).await?;
        let restarted = MotorGate::new(MotorMode::Off);
        restarted.restore_outcomes(&pool).await;
        assert!(matches!(restarted.record_outcome(&pool, &model, callback).await, Err(OutcomeError::Unmatched)));
        let headline: String = sqlx::query_scalar("SELECT headline FROM rgw_episodes WHERE source_ref = $1").bind(id.to_string()).fetch_one(&pool).await?;
        assert_eq!(headline, "legacy");
        assert!(gate.snapshot()["outcomes"].as_array().unwrap().is_empty());
        assert!(restarted.consolidate_episodes().episodes.is_empty());
    }
    assert_eq!(sqlx::query_scalar::<_, i64>("SELECT count(*) FROM rgw_episodes").fetch_one(&pool).await?, 4, "only the seeded legacy rows exist");
    Ok(())
}
