//! The process-wide shadow statistics. This file is its own test binary, and holds a single
//! test, because the counters are global to the process.
mod common;
use common::fixture::*;
use rgw::seeding::SeedMode;
use rgw::walker::{walk_parallel_with, EvalArm, WalkOptions};
use std::sync::Arc;

fn walks() -> u64 {
    rgw::seeding::seed_stats_json()["walks"].as_u64().unwrap_or(0)
}

/// The observation runs off the request path; wait for the counter to reach `n`.
async fn wait_for_walks(n: u64) -> u64 {
    for _ in 0..200 {
        if walks() >= n {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    walks()
}

#[tokio::test]
async fn statistics_record_mode_walks_and_never_eval_arms() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = fixture_pool("rgw_p1_stats").await?;
    let sm = || {
        let mut m = rgw::core::SelfModel::new();
        m.mode = rgw::core::CognitiveMode::Autonomous;
        Arc::new(tokio::sync::Mutex::new(m))
    };
    let run = |cue: Vec<f32>, arm: Option<EvalArm>, mode: SeedMode| {
        let pool = pool.clone();
        let model = sm();
        async move {
            let o = WalkOptions { cue_embedding: Some(cue), seed: Some(1), arm, seed_mode: Some(mode) };
            walk_parallel_with(&pool, &Default::default(), 2, 2, &model, o).await.0
        }
    };

    // An eval arm is a measurement of the system, not of its traffic.
    run(axis(1), Some(EvalArm::Query), SeedMode::Shadow).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(walks(), 0, "eval arms must not move the statistics");

    // Shadow: walks as today, records the plan.
    let out = run(axis(1), None, SeedMode::Shadow).await;
    assert_eq!(out.seed_mode, "shadow");
    assert_eq!(wait_for_walks(1).await, 1);

    // Query mode with a cue far from everything: recorded as a fallback.
    let out = run(axis(7), None, SeedMode::Query).await;
    assert!(!out.walked);
    assert_eq!(wait_for_walks(2).await, 2);
    let stats = rgw::seeding::seed_stats_json();
    assert_eq!(stats["fallbacks"].as_u64(), Some(1));

    // Fixed mode records nothing.
    run(axis(1), None, SeedMode::Fixed).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(walks(), 2);
    Ok(())
}
