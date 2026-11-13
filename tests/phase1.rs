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
