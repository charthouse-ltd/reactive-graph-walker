//! Query seeding against an isolated graph with synthetic embeddings.
//! Needs the local Postgres (see tests/common/fixture.rs); CI does not run this file.
// tests/seeding_walk.rs
mod common;
use common::fixture::*;

/// Load the embedding model once per test binary. `rgw::embed::init` replaces the global
/// model on every call, and several tests initialising at once against a cold model cache
/// fail while the files are still downloading; `OnceLock` makes the others wait.
fn embedder() -> anyhow::Result<()> {
    static INIT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| rgw::embed::init().map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

#[tokio::test]
async fn candidates_are_ranked_by_cosine_with_degrees() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_candidates").await?;
    let cands = rgw::seeding::candidates(&pool, &axis(1)).await?;
    let ids: Vec<i32> = cands.iter().take(5).map(|c| c.id).collect();
    assert_eq!(ids, vec![B1, B2, B3, B4, HUB]);
    assert!((cands[0].cos - 1.0).abs() < 1e-4);
    assert!((cands[4].cos - 0.7071).abs() < 1e-3);
    let deg = |id: i32| cands.iter().find(|c| c.id == id).unwrap().degree;
    assert_eq!((deg(B1), deg(HUB), deg(B4)), (3, 4, 1));
    Ok(())
}

#[tokio::test]
async fn degrees_and_percentile() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_degrees").await?;
    let d = rgw::db::node_degrees(&pool, &[HUB, LONE, 999]).await?;
    assert_eq!((d[&HUB], d[&LONE], d[&999]), (4, 1, 0));
    assert_eq!(rgw::db::degree_at_percentile(&pool, 1.0).await?, 4);
    Ok(())
}

#[tokio::test]
async fn a_wrong_dimension_is_an_error_not_a_panic() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_dim").await?;
    assert!(rgw::seeding::candidates(&pool, &[1.0, 0.0, 0.0]).await.is_err());
    Ok(())
}

use rgw::walker::{walk_parallel_with, EvalArm, WalkOptions};
use std::sync::Arc;

fn sm() -> Arc<tokio::sync::Mutex<rgw::core::SelfModel>> {
    let mut m = rgw::core::SelfModel::new();
    m.mode = rgw::core::CognitiveMode::Autonomous;
    Arc::new(tokio::sync::Mutex::new(m))
}
fn opts(cue: Vec<f32>, arm: EvalArm) -> WalkOptions {
    WalkOptions { cue_embedding: Some(cue), seed: Some(1), arm: Some(arm), seed_mode: None }
}

#[tokio::test]
async fn query_arm_starts_in_the_cue_cluster_and_restarts_keep_the_invariant() -> anyhow::Result<()> {
    embedder()?; // signal processing embeds text
    let pool = fixture_pool("rgw_p1_query").await?;
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
    assert_eq!(results.len(), 4);
    let starts: Vec<i32> = results.iter().map(|r| r.path[0]).collect();
    // Plan order is cosine times specificity: B4 (degree 1) 0.543, B2 0.474, B1 0.419, B3 0.407, HUB 0.271.
    assert_eq!(starts, vec![B4, B2, B1, B3], "round-robin over the plan by rank");
    for r in &results {
        let segs = if r.segment_starts.is_empty() { 1 } else { r.segment_starts.len() };
        assert_eq!(r.edges_traversed.len(), r.path.len() - segs);
        assert_eq!(r.depths.len(), r.path.len());
        assert_eq!(r.dead_ends, 0, "B4 has one edge and its far side is cached; nothing here is a dead end");
    }
    Ok(())
}

#[tokio::test]
async fn fixed_arm_starts_where_it_does_today() -> anyhow::Result<()> {
    embedder()?;
    let pool = fixture_pool("rgw_p1_fixed").await?;
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 2, 3, &sm(), opts(axis(1), EvalArm::Fixed)).await;
    assert_eq!(results.iter().map(|r| r.path[0]).collect::<Vec<_>>(), vec![HUB, A1]);
    assert!(results.iter().all(|r| r.segment_starts.len() <= 1));
    Ok(())
}

#[tokio::test]
async fn drop_arm_removes_one_walker_profile() -> anyhow::Result<()> {
    embedder()?;
    let pool = fixture_pool("rgw_p1_drop").await?;
    let arm = EvalArm::Drop(rgw::graph::WalkerBias::Contrarian);
    let (_out, results) = walk_parallel_with(&pool, &Default::default(), 6, 2, &sm(), opts(axis(1), arm)).await;
    assert_eq!(results.len(), 6);
    assert!(results.iter().all(|r| r.bias != rgw::graph::WalkerBias::Contrarian));
    Ok(())
}

#[tokio::test]
async fn same_seed_gives_the_same_paths() -> anyhow::Result<()> {
    embedder()?;
    let pool = fixture_pool("rgw_p1_determinism").await?;
    // Paths and the ranked list: invariant I6 of the spec.
    let run = || async {
        let (o, r) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
        (r.into_iter().map(|x| x.path).collect::<Vec<_>>(), o.ranked_nodes)
    };
    assert_eq!(run().await, run().await);
    Ok(())
}

#[tokio::test]
async fn query_arm_ranks_the_cue_cluster_first() -> anyhow::Result<()> {
    embedder()?;
    let pool = fixture_pool("rgw_p1_rank").await?;
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(1), EvalArm::Query)).await;
    assert!(out.walked);
    let top4: Vec<i32> = out.ranked_nodes.iter().take(4).map(|n| n.0).collect();
    let gold = [B1, B2, B3, B4];
    assert!(top4.iter().filter(|id| gold.contains(id)).count() >= 3, "top 4 were {top4:?}");
    assert_eq!(out.walk_id.len(), 32);
    Ok(())
}

#[tokio::test]
async fn fixed_arm_ranks_todays_seeds_first() -> anyhow::Result<()> {
    embedder()?;
    let pool = fixture_pool("rgw_p1_rank_fixed").await?;
    // One step each: a start node holds mass 1.0, and no other node can exceed it.
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 2, 1, &sm(), opts(axis(1), EvalArm::Fixed)).await;
    assert!([HUB, A1].contains(&out.ranked_nodes[0].0));
    Ok(())
}

#[tokio::test]
async fn a_cue_far_from_everything_does_not_walk_in_query_mode() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_floor").await?;
    let model = sm();
    let before = serde_json::to_value(&*model.lock().await)?;
    let o = WalkOptions { cue_embedding: Some(axis(7)), seed: Some(1), arm: None,
                          seed_mode: Some(rgw::seeding::SeedMode::Query) };
    let (out, results) = walk_parallel_with(&pool, &Default::default(), 4, 5, &model, o).await;
    assert!(!out.walked && results.is_empty());
    assert_eq!(out.recommended_action, "rest");
    assert_eq!(serde_json::to_value(&*model.lock().await)?, before, "no walk, no self-model change");
    Ok(())
}

#[tokio::test]
async fn vector_arm_returns_candidates_by_cosine() -> anyhow::Result<()> {
    let pool = fixture_pool("rgw_p1_vector").await?;
    let (out, _) = walk_parallel_with(&pool, &Default::default(), 4, 5, &sm(), opts(axis(2), EvalArm::Vector)).await;
    assert!(!out.walked);
    assert_eq!(out.ranked_nodes[0].0, LONE);
    assert!((out.ranked_nodes[0].1 - 1.0).abs() < 1e-4);
    assert_eq!(out.primary_domain, "gamma");
    Ok(())
}
