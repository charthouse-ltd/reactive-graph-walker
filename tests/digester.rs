//! Integration test: RGW digesting philosophical knowledge.
//!
//! Seeds the graph with key ideas from great thinkers (Plato, Nietzsche,
//! Darwin, Buddha, Kant, Turing), creates edges between related concepts,
//! then runs parallel walkers and observes:
//!   - Inter-walker stigmergy (collective trails)
//!   - Structural noticings (dead_end_cluster, surprise_density, etc.)
//!   - Working memory push
//!   - Predictive coding
//!   - Belief formation from pattern detection
//!
//! Usage:
//!   DATABASE_URL=postgresql://user@localhost/rgw_test cargo test --test digester -- --nocapture

use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Instant;

use sqlx::PgPool;

// ── Philosophical seed data ─────────────────────────────────────
// Each entry: (content, domain, importance) — key ideas from great thinkers.
// These are curated to have rich cross-domain relationships.

mod common;
use common::seed_database;

// ── Walking ─────────────────────────────────────────────────────

async fn run_walks(pool: &PgPool) -> anyhow::Result<()> {
    use rgw::core::{self, SelfModel};
    use rgw::graph::EmotionalState;
    use rgw::walker;

    let self_model = Arc::new(Mutex::new(SelfModel::new()));
    let emotion = EmotionalState::default();

    println!("=== WALKER SESSION 1: Curiosity-biased exploration ===\n");

    let (output, _) = walker::walk_parallel(
        pool,
        &emotion,
        4,    // 4 walkers
        6,    // 6 steps each
        &self_model,
    ).await;

    println!("Walk complete:");
    println!("  agreement={:.2}  novelty={:.2}  resonance={:.2}",
        output.agreement_score, output.novelty_score, output.emotional_resonance);
    println!("  consensus nodes: {:?}", &output.consensus_nodes[..output.consensus_nodes.len().min(5)]);
    println!("  divergent nodes: {:?}", &output.divergent_nodes[..output.divergent_nodes.len().min(5)]);
    println!("  blind spots:     {:?}", &output.blind_spots[..output.blind_spots.len().min(5)]);

    // /walk's walker_context is built from this narrative: it must always carry
    // the state line, and node rows whenever the walk surfaced known nodes.
    let narrative = walker::format_walk_narrative(&output, pool).await;
    assert!(narrative.contains("State: "), "narrative missing state line: {narrative}");
    if !output.consensus_nodes.is_empty() {
        assert!(narrative.contains("Strong convictions"), "consensus nodes but no rows: {narrative}");
    }
    println!("\n{narrative}\n");
    println!("  total hops: {}  walk_ms: {:.0}  hops/sec: {:.0}",
        output.total_hops, output.walk_ms, output.hops_per_sec);

    // ── Self-model state after walk ──
    let sm = self_model.lock().await;
    println!("\n=== SELF-MODEL STATE ===");
    println!("  mode: {:?}", sm.mode);
    println!("  valence={:.3}  arousal={:.3}  energy={:.3}  plasticity_gate={:.3}",
        sm.valence, sm.arousal, sm.energy, sm.plasticity_gate);
    println!("  focus: {} (intensity={:.2})", sm.current_focus, sm.focus_intensity);
    println!("  signals processed: {}  noticings: {}", sm.total_signals_processed, sm.total_noticings);
    println!("  beliefs: {}", sm.beliefs.len());
    for b in &sm.beliefs {
        println!("    - [{}] {} (confidence={:.2}, evidence={})",
            b.domain, &b.statement[..b.statement.len().min(60)], b.confidence, b.evidence_count);
    }
    println!("  working memory: {} items", sm.working_memory.len());
    for wm in &sm.working_memory {
        println!("    - [{}] {} (activation={:.0}%)",
            wm.domain, &wm.content[..wm.content.len().min(50)], wm.activation * 100.0);
    }

    // ── Structural noticings ──
    println!("\n=== NOTICINGS ===");
    let structural_kinds = ["dead_end_cluster", "surprise_density", "cognitive_loop",
        "signal_poverty", "belief_stagnation", "prediction_error"];
    for n in &sm.noticings {
        if structural_kinds.contains(&n.kind.as_str()) {
            println!("  🔍 [{}] {} (significance={:.2}, valence={:.2})",
                n.kind, n.observation, n.significance, n.valence);
        }
    }

    // ── Attention patterns ──
    println!("\n=== ATTENTION PATTERNS ===");
    let mut patterns: Vec<_> = sm.attention_patterns.iter().collect();
    patterns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (domain, count) in patterns.iter().take(10) {
        println!("  {}: {:.2}", domain, count);
    }

    // ── Surprise count + dead ends ──
    println!("\n=== STRUCTURAL TRACKING ===");
    println!("  surprise_count: {}", sm.surprise_count);
    println!("  consecutive_repetitions: {}", sm.consecutive_repetitions);
    println!("  dead_ends_by_domain:");
    for (domain, count) in &sm.dead_ends_by_domain {
        println!("    {}: {}", domain, count);
    }
    println!("  signals_by_source:");
    for (source, count) in &sm.signals_by_source {
        println!("    {}: {}", source, count);
    }

    // ── Predictions ──
    println!("\n=== PREDICTIONS ===");
    println!("  active predictions: {}", sm.predictions.len());
    for (domain, pred) in &sm.predictions {
        println!("    {}: '{}' (confidence={:.2})", domain, pred.predicted, pred.confidence);
    }

    Ok(())
}

// ── Main ────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires downloaded embedding model; run explicitly with --ignored"]
async fn test_digest_philosophy() -> anyhow::Result<()> {
    // Initialize tracing for visibility
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rgw=debug"))
        )
        .try_init()
        .ok(); // Ignore if already initialized

    // Initialize embedder
    rgw::embed::init()?;

    // Connect to database
    let db_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must explicitly select a throwaway test database");
    println!("Connecting to: {}", db_url);
    let pool = rgw::db::connect(&db_url).await?;

    // Ensure schema exists
    let schema = include_str!("schema.sql");
    // schema.sql contains multiple statements; sqlx::query() uses the prepared-
    // statement protocol, which Postgres forbids for multi-command strings.
    // raw_sql() runs them via the simple query protocol instead.
    sqlx::raw_sql(schema).execute(&pool).await?;
    println!("Schema ensured.\n");

    // Check if already seeded
    let stats = rgw::db::graph_stats(&pool).await?;
    if stats.nodes > 0 {
        println!("Database already has {} nodes, {} edges — skipping seed.\n", stats.nodes, stats.edges);
    } else {
        seed_database(&pool).await?;
    }

    // Run walks
    run_walks(&pool).await?;

    Ok(())
}

/// Edge-cache flush must apply each edge's OWN accumulated delta, not the mean
/// of all pending deltas (which both over- and under-strengthens, and is not
/// Hebbian). Uses two real edges well below the 1.0 cap.
#[tokio::test]
async fn edge_flush_applies_per_edge_deltas() -> anyhow::Result<()> {
    let pool = common::fixture::fixture_pool("phase1_edge_flush").await?;
    let rows: Vec<(i32, f32)> = sqlx::query_as(
        "SELECT id, weight::float4 FROM memory_edges WHERE weight < 0.8 ORDER BY id LIMIT 2",
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(rows.len(), 2, "need two edges below the cap");
    let (a, wa) = rows[0];
    let (b, wb) = rows[1];

    let cache = rgw::edge_cache::EdgeCache::new(pool.clone());
    cache.strengthen_edge(a, 0.05);
    cache.strengthen_edge(a, 0.05); // accumulates: a gets +0.10
    cache.strengthen_edge(b, 0.01); // b gets +0.01 (mean would be 0.055)
    let flushed = cache.flush().await?;
    assert_eq!(flushed, 2);

    let after: Vec<(i32, f32)> = sqlx::query_as("SELECT id, weight::float4 FROM memory_edges WHERE id = ANY($1) ORDER BY id")
        .bind(vec![a, b])
        .fetch_all(&pool)
        .await?;
    let get = |id: i32| after.iter().find(|r| r.0 == id).map(|r| r.1).unwrap();
    assert!((get(a) - (wa + 0.10)).abs() < 1e-4, "edge {a}: {} -> {} (expected +0.10)", wa, get(a));
    assert!((get(b) - (wb + 0.01)).abs() < 1e-4, "edge {b}: {} -> {} (expected +0.01)", wb, get(b));
    // nothing left pending after a successful flush
    assert_eq!(cache.flush().await?, 0);
    Ok(())
}

/// Tension targets: a prediction maps to the top nodes of its domain by
/// importance; an open question maps by embedding similarity. Energy is split
/// across an item's targets, never multiplied.
#[tokio::test]
#[ignore = "requires downloaded embedding model; run explicitly with --ignored"]
async fn targets_resolve_from_domain_and_embedding() -> anyhow::Result<()> {
    rgw::embed::init()?;
    let pool = rgw::db::connect(
        &std::env::var("DATABASE_URL").expect("DATABASE_URL must explicitly select a throwaway test database"),
    )
    .await?;
    sqlx::raw_sql(include_str!("schema.sql")).execute(&pool).await?;
    if rgw::db::graph_stats(&pool).await?.nodes == 0 {
        seed_database(&pool).await?;
    }
    let domain: String = sqlx::query_scalar(
        "SELECT domain FROM memory_vectors GROUP BY domain ORDER BY count(*) DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(rgw::db::top_nodes_in_domain(&pool, &domain, 3).await?.len(), 3);
    assert!(rgw::db::top_nodes_in_domain(&pool, "no-such-domain", 3).await?.is_empty());

    let mut inner = rgw::tension::EngineInner::new(rgw::tension::TensionMode::Observe, 0.0);
    let q = "is moral realism defensible";
    // plan_release normally records the text; this test bypasses it.
    inner.questions.entry(rgw::tension::fnv1a64(q)).or_default().text = q.to_string();
    let pending = vec![
        rgw::tension::PendingPulse {
            source: rgw::tension::TensionSource::Prediction { domain: domain.clone() },
            key: domain.clone(),
            energy: 0.15,
        },
        rgw::tension::PendingPulse {
            source: rgw::tension::TensionSource::Question { id: rgw::tension::question_id(q) },
            key: format!("{:016x}", rgw::tension::fnv1a64(q)),
            energy: 0.20,
        },
    ];
    let pulses = rgw::tension::resolve_targets(&pool, &mut inner, pending).await;
    assert_eq!(pulses.len(), 6, "3 targets per item: {pulses:?}");
    assert!((pulses.iter().map(|p| p.energy).sum::<f32>() - 0.35).abs() < 1e-4, "energy split, not multiplied");
    Ok(())
}
