//! Query seeding against an isolated graph with synthetic embeddings.
//! Needs the local Postgres (see tests/common/fixture.rs); CI does not run this file.
// tests/seeding_walk.rs
mod common;
use common::fixture::*;

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
