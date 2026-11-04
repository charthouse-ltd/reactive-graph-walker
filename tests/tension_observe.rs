//! Observe mode must compute and log would-fire pulses and emit nothing.
//! Lives in its own test binary because `tension::install` is process-global.
mod common;
use common::seed_database;
use std::sync::Arc;

#[tokio::test]
async fn observe_mode_logs_would_fire_and_emits_nothing() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = rgw::db::connect(
        &std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".into()),
    )
    .await?;
    sqlx::raw_sql(include_str!("schema.sql")).execute(&pool).await?;
    if rgw::db::graph_stats(&pool).await?.nodes == 0 {
        seed_database(&pool).await?;
    }
    let engine = rgw::tension::install(rgw::tension::TensionMode::Observe).expect("observe installs an engine");
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    engine.attach_inbox(tx);

    let domain: String = sqlx::query_scalar(
        "SELECT domain FROM memory_vectors GROUP BY domain ORDER BY count(*) DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await?;
    let sm = Arc::new(tokio::sync::Mutex::new(rgw::core::SelfModel::new()));
    {
        let mut g = sm.lock().await;
        g.mode = rgw::core::CognitiveMode::Autonomous;
        g.predict(&domain, "x", 0.5);
        g.predictions.get_mut(&domain).unwrap().predicted_at -= 3600.0;
        g.open_questions.push("what is a dead end".into());
    }
    let (_out, _) = rgw::walker::walk_parallel(&pool, &rgw::graph::EmotionalState::default(), 2, 3, &sm).await;

    let stats = engine.stats(&*sm.lock().await).await;
    assert!(stats.pulses_would_fire >= 1, "{stats:?}");
    assert_eq!(stats.pulses_fired, 0);
    assert!(rx.try_recv().is_err(), "nothing entered the inbox in observe");
    assert_eq!(rgw::tension::mode(), rgw::tension::TensionMode::Observe);
    Ok(())
}
