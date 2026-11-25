mod common;

#[tokio::test]
async fn restart_restores_saved_state_and_rejects_corruption() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("phase1_restore").await?;
    let mut model = rgw::core::SelfModel::new();
    model.total_signals_processed = 123;
    model.current_focus = "restart regression".into();
    rgw::db::save_self_model(&pool, &model).await?;
    let restored = rgw::db::restore_self_model(&pool).await?;
    assert_eq!(serde_json::to_value(&model)?, serde_json::to_value(&restored)?);
    sqlx::query("UPDATE runtime_settings SET value = 'invalid json' WHERE key = 'rgw_self_model'")
        .execute(&pool).await?;
    assert!(rgw::db::restore_self_model(&pool).await.is_err());
    let saved: String = sqlx::query_scalar("SELECT value FROM runtime_settings WHERE key = 'rgw_self_model'")
        .fetch_one(&pool).await?;
    assert_eq!(saved, "invalid json");
    Ok(())
}

#[tokio::test]
async fn readiness_checks_contract_and_decoding_errors_are_results() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("phase1_contract").await?;
    assert!(rgw::db::schema_ready(&pool).await.is_ok());
    sqlx::query("ALTER TABLE runtime_settings DROP COLUMN description").execute(&pool).await?;
    assert!(rgw::db::schema_ready(&pool).await.is_err());
    sqlx::query("ALTER TABLE memory_edges ALTER COLUMN edge_type TYPE integer USING 1").execute(&pool).await?;
    assert!(rgw::db::edges_from(&pool, 1).await.is_err());
    Ok(())
}

#[tokio::test]
async fn readiness_rejects_missing_tables_columns_and_wrong_settings_type() -> anyhow::Result<()> {
    common::fixture::fixture_pool("phase1_missing").await?.close().await;
    // Do not fall through to public's valid tables when a fixture table is renamed.
    let pool = sqlx::postgres::PgPoolOptions::new()
        .after_connect(|conn, _| Box::pin(async move {
            sqlx::query("SET search_path TO phase1_missing").execute(conn).await?;
            Ok(())
        }))
        .connect(&std::env::var("DATABASE_URL")?).await?;
    assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::OK);
    for table in ["memory_vectors", "memory_edges", "runtime_settings", "rgw_episodes"] {
        sqlx::raw_sql(&format!("ALTER TABLE {table} RENAME TO temporarily_missing")).execute(&pool).await?;
        assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        sqlx::raw_sql(&format!("ALTER TABLE temporarily_missing RENAME TO {table}")).execute(&pool).await?;
    }
    for (table, column) in [("memory_vectors", "document"), ("memory_vectors", "embedding"),
        ("runtime_settings", "description"), ("runtime_settings", "category")] {
        sqlx::raw_sql(&format!("ALTER TABLE {table} RENAME COLUMN {column} TO temporarily_missing")).execute(&pool).await?;
        assert!(rgw::db::schema_ready(&pool).await.is_err());
        sqlx::raw_sql(&format!("ALTER TABLE {table} RENAME COLUMN temporarily_missing TO {column}")).execute(&pool).await?;
    }
    sqlx::raw_sql("ALTER TABLE runtime_settings ALTER COLUMN value TYPE jsonb USING value::jsonb").execute(&pool).await?;
    assert!(rgw::db::schema_ready(&pool).await.is_err());
    Ok(())
}

#[tokio::test]
async fn julian_node_ownership_and_foreign_keys_are_enforced() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("phase1_ownership").await?;
    assert!(rgw::db::create_memory_node(&pool, "test", "test", &[], 1.0, 0.0, 0.0).await.is_err());
    assert!(sqlx::query("INSERT INTO memory_vectors (document) VALUES ('test')").execute(&pool).await.is_err());
    assert!(rgw::db::create_edge(&pool, 1, 999, "related", 0.5, 0.0).await.is_err());
    sqlx::query("DELETE FROM memory_vectors WHERE id = 1").execute(&pool).await?;
    let edges: i64 = sqlx::query_scalar("SELECT count(*) FROM memory_edges WHERE source_id = 1 OR target_id = 1").fetch_one(&pool).await?;
    assert_eq!(edges, 0);
    Ok(())
}

#[tokio::test]
async fn outcome_schema_contract_guards_readiness_and_writes() -> anyhow::Result<()> {
    use rgw::motor::{DispatchOutcome, MotorCommand, MotorGate, MotorMode, MotorOutcome, OutcomeError};
    let pool = common::fixture::fixture_pool("phase1_outcome_contract").await?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move {
        axum::serve(listener, axum::Router::new().route("/api/admin/rgw/execute",
            axum::routing::post(|| async { "ok" }))).await.unwrap();
    });
    let gate = MotorGate::new(MotorMode::Live);
    let id = uuid::Uuid::new_v4();
    assert!(matches!(gate.dispatch("test", &url, MotorCommand {
        correlation_id: id, action: "search".into(), domain: "science".into(),
        walker_context: String::new(), expression_seeds: vec![], confidence: 0.8,
        novelty: 0.5, search_query: None, params: None, audience_id: None,
    }).await, DispatchOutcome::Sent(..)));
    server.abort();
    let mut model = rgw::core::SelfModel::new();
    model.mode = rgw::core::CognitiveMode::Autonomous;
    model.self_mod_stage = rgw::core::SelfModStage::SelectionLive;
    let model = tokio::sync::Mutex::new(model);
    let callback = MotorOutcome { correlation_id: id, outcome: "done".into(), success: Some(true), lesson: None };
    for column in ["source_type", "source_ref"] {
        sqlx::raw_sql(&format!("ALTER TABLE rgw_episodes RENAME COLUMN {column} TO missing_column")).execute(&pool).await?;
        assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(gate.record_outcome(&pool, &model, callback.clone()).await, Err(OutcomeError::Storage(_))));
        sqlx::raw_sql(&format!("ALTER TABLE rgw_episodes RENAME COLUMN missing_column TO {column}")).execute(&pool).await?;
    }
    sqlx::query("ALTER TABLE rgw_episodes DROP CONSTRAINT uq_rgw_episodes_source").execute(&pool).await?;
    for index in [
        "",
        "CREATE INDEX candidate ON rgw_episodes(source_type, source_ref)",
        "CREATE UNIQUE INDEX candidate ON rgw_episodes(source_type, source_ref) WHERE source_type = 'motor_outcome'",
        "CREATE UNIQUE INDEX candidate ON rgw_episodes(source_type, headline)",
        "CREATE UNIQUE INDEX candidate ON rgw_episodes(source_type, source_ref, headline)",
        "CREATE UNIQUE INDEX candidate ON rgw_episodes(source_type) INCLUDE(source_ref)",
        "CREATE UNIQUE INDEX candidate ON rgw_episodes(lower(source_type), source_ref)",
        "ALTER TABLE rgw_episodes ADD CONSTRAINT candidate UNIQUE(source_type, source_ref) DEFERRABLE",
    ] {
        if !index.is_empty() { sqlx::raw_sql(index).execute(&pool).await?; }
        assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::SERVICE_UNAVAILABLE, "{index}");
        assert!(matches!(gate.record_outcome(&pool, &model, callback.clone()).await, Err(OutcomeError::Storage(_))), "{index}");
        assert!(gate.snapshot()["outcomes"].as_array().unwrap().is_empty());
        assert!(gate.consolidate_episodes().episodes.is_empty());
        assert_eq!(model.lock().await.total_signals_processed, 0);
        assert_eq!(sqlx::query_scalar::<_, i64>("SELECT count(*) FROM rgw_episodes").fetch_one(&pool).await?, 0);
        sqlx::raw_sql("ALTER TABLE rgw_episodes DROP CONSTRAINT IF EXISTS candidate; DROP INDEX IF EXISTS candidate").execute(&pool).await?;
    }
    // A failed concurrent build leaves a real invalid index, without editing catalogs.
    sqlx::query("INSERT INTO rgw_episodes(occurred_at, kind, headline, source_type, source_ref) VALUES (NOW(), 'test', 'test', 'test', 'duplicate'), (NOW(), 'test', 'test', 'test', 'duplicate')").execute(&pool).await?;
    assert!(sqlx::query("CREATE UNIQUE INDEX CONCURRENTLY candidate ON rgw_episodes(source_type, source_ref)").execute(&pool).await.is_err());
    sqlx::query("DELETE FROM rgw_episodes").execute(&pool).await?;
    assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert!(gate.record_outcome(&pool, &model, callback.clone()).await.is_err());
    sqlx::query("DROP INDEX candidate").execute(&pool).await?;
    // PostgreSQL accepts reversed keys and INCLUDE columns for conflict inference.
    sqlx::query("CREATE UNIQUE INDEX candidate ON rgw_episodes(source_ref, source_type) INCLUDE(headline)").execute(&pool).await?;
    assert_eq!(rgw::api::readiness(&pool).await.0, axum::http::StatusCode::OK);
    gate.record_outcome(&pool, &model, callback.clone()).await?;
    assert_eq!(model.lock().await.total_signals_processed, 1);
    assert_eq!(gate.snapshot()["outcomes"][0]["reflected"], true);
    let before = gate.snapshot()["outcomes"].clone();
    sqlx::query("DROP INDEX candidate").execute(&pool).await?;
    let correction = MotorOutcome { outcome: "failed correction".into(), ..callback };
    assert!(gate.record_outcome(&pool, &model, correction).await.is_err());
    assert_eq!(gate.snapshot()["outcomes"], before, "failed correction must not replace a durable record");
    assert_eq!(model.lock().await.total_signals_processed, 1);
    Ok(())
}
