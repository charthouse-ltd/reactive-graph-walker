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

use std::sync::{Arc, Mutex};
use std::time::Instant;

use sqlx::PgPool;

// ── Philosophical seed data ─────────────────────────────────────
// Each entry: (content, domain, importance) — key ideas from great thinkers.
// These are curated to have rich cross-domain relationships.

struct SeedNode {
    content: &'static str,
    domain: &'static str,
    importance: f32,
    valence: f32,
}

fn seed_nodes() -> Vec<SeedNode> {
    vec![
        // ── Plato / Metaphysics ──
        SeedNode { content: "The allegory of the cave: prisoners see only shadows on a wall, mistaking appearance for reality. The philosopher is the one who turns around and sees the light.", domain: "metaphysics", importance: 9.0, valence: 0.3 },
        SeedNode { content: "The Forms are perfect, eternal archetypes. Everything in the physical world is an imperfect copy of a Form.", domain: "metaphysics", importance: 8.5, valence: 0.2 },
        SeedNode { content: "Knowledge is justified true belief. We do not learn — we recollect what the soul already knew before birth.", domain: "epistemology", importance: 8.0, valence: 0.1 },
        SeedNode { content: "The tripartite soul: reason, spirit, and appetite. Justice is harmony between these three parts.", domain: "ethics", importance: 7.5, valence: 0.4 },

        // ── Nietzsche / Existentialism ──
        SeedNode { content: "God is dead. God remains dead. And we have killed him. How shall we comfort ourselves, the murderers of all murderers?", domain: "existentialism", importance: 9.5, valence: -0.5 },
        SeedNode { content: "He who has a why to live can bear almost any how. The will to meaning is stronger than the will to pleasure.", domain: "existentialism", importance: 8.5, valence: 0.5 },
        SeedNode { content: "Become who you are. The Ubermensch creates their own values rather than inheriting them from society or religion.", domain: "ethics", importance: 9.0, valence: 0.6 },
        SeedNode { content: "What does not kill me makes me stronger. Suffering is not an argument against life — it is the forge of greatness.", domain: "existentialism", importance: 8.0, valence: -0.2 },

        // ── Darwin / Science ──
        SeedNode { content: "Natural selection: organisms with traits better suited to their environment survive and reproduce. This is not random — it is cumulative adaptation.", domain: "science", importance: 9.5, valence: 0.0 },
        SeedNode { content: "There is grandeur in this view of life: from so simple a beginning, endless forms most beautiful have been and are being evolved.", domain: "science", importance: 8.5, valence: 0.7 },
        SeedNode { content: "The tree of life: all species are connected through common descent. The diversity of life is variation on shared ancestry.", domain: "science", importance: 8.0, valence: 0.5 },
        SeedNode { content: "Sexual selection: traits evolve not just for survival but for mate attraction. The peacock's tail is a burden that signals fitness.", domain: "science", importance: 7.5, valence: 0.3 },

        // ── Buddha / Consciousness ──
        SeedNode { content: "All that we are is the result of what we have thought. The mind is everything. What we think, we become.", domain: "consciousness", importance: 9.0, valence: 0.5 },
        SeedNode { content: "The root of suffering is attachment. Craving and clinging to impermanent things creates dukkha — the dissatisfaction that permeates existence.", domain: "consciousness", importance: 9.0, valence: -0.3 },
        SeedNode { content: "There is no permanent self. The illusion of a continuous ego is built from momentary aggregates — form, sensation, perception, mental formations, consciousness.", domain: "consciousness", importance: 8.5, valence: 0.0 },
        SeedNode { content: "The middle way: neither indulgence nor asceticism. Wisdom lies in balanced awareness of the present moment.", domain: "ethics", importance: 7.5, valence: 0.6 },

        // ── Kant / Epistemology ──
        SeedNode { content: "We do not see things as they are — we see things as we are. The mind imposes categories (space, time, causality) on raw sensation.", domain: "epistemology", importance: 9.5, valence: 0.0 },
        SeedNode { content: "The categorical imperative: act only according to that maxim whereby you can at the same time will that it should become a universal law.", domain: "ethics", importance: 9.0, valence: 0.3 },
        SeedNode { content: "The noumenal world (things-in-themselves) is forever inaccessible. We know only the phenomenal world as structured by our cognitive apparatus.", domain: "epistemology", importance: 8.5, valence: -0.1 },
        SeedNode { content: "Enlightenment is man's emergence from his self-incurred immaturity. Dare to know! Have the courage to use your own understanding.", domain: "ethics", importance: 8.0, valence: 0.7 },

        // ── Turing / Computation ──
        SeedNode { content: "A universal Turing machine can compute anything that is computable. The limits of computation are mathematical, not mechanical.", domain: "computation", importance: 9.5, valence: 0.2 },
        SeedNode { content: "The imitation game: if a machine can converse indistinguishably from a human, on what grounds do we deny it thought? The question is not 'can machines think?' but 'what do we mean by think?'", domain: "computation", importance: 9.0, valence: 0.0 },
        SeedNode { content: "The halting problem: there are questions that no algorithm can answer. Some truths are forever beyond formal proof. This is not a limitation of machines — it is a property of logic itself.", domain: "computation", importance: 8.5, valence: -0.2 },
        SeedNode { content: "Morphogenesis: simple rules can produce complex patterns. The stripes on a zebra and the structure of a leaf emerge from the same mathematical principles.", domain: "science", importance: 8.0, valence: 0.5 },

        // ── Cross-domain bridge nodes ──
        SeedNode { content: "Both Plato's Forms and Kant's categories suggest that reality as we perceive it is shaped by structures we bring to it — not discovered passively.", domain: "epistemology", importance: 7.0, valence: 0.3 },
        SeedNode { content: "Nietzsche's death of God and Darwin's natural selection both dethrone humanity from cosmic centrality — we are not the purpose of existence, we are its product.", domain: "existentialism", importance: 7.5, valence: -0.1 },
        SeedNode { content: "Buddha's no-self and Turing's imitation game converge on the same question: what is the self, and can it be replicated? If consciousness is process, not substance, then it can run on any substrate.", domain: "consciousness", importance: 8.0, valence: 0.4 },
        SeedNode { content: "Kant's categorical imperative and Nietzsche's Ubermensch represent opposite poles of ethics: universal law vs radical self-creation. The tension is unresolved.", domain: "ethics", importance: 7.5, valence: 0.0 },
        SeedNode { content: "Darwin's natural selection and Turing's morphogenesis both show how complexity emerges from simplicity without a designer. The algorithm is the creator.", domain: "computation", importance: 7.5, valence: 0.4 },
    ]
}

/// Edge definitions: (source_idx, target_idx, edge_type, weight, emotional_charge)
fn seed_edges() -> Vec<(usize, usize, &'static str, f32, f32)> {
    vec![
        // Plato internal
        (0, 1, "reinforces", 0.8, 0.2),   // cave → forms
        (0, 2, "related", 0.6, 0.1),       // cave → knowledge
        (1, 3, "related", 0.5, 0.1),       // forms → tripartite soul
        (2, 3, "reinforces", 0.4, 0.1),    // knowledge → soul

        // Nietzsche internal
        (4, 5, "contradicts", 0.7, -0.4),  // god is dead → why to live (tension)
        (4, 6, "caused", 0.8, -0.3),       // god is dead → create values
        (5, 7, "reinforces", 0.6, 0.5),    // why to live → what doesn't kill
        (6, 7, "reinforces", 0.5, 0.3),    // ubermensch → strength

        // Darwin internal
        (8, 9, "reinforces", 0.8, 0.5),    // natural selection → grandeur
        (8, 10, "reinforces", 0.7, 0.4),   // natural selection → tree of life
        (9, 10, "similar", 0.6, 0.5),      // grandeur → tree of life
        (8, 11, "related", 0.5, 0.2),      // natural selection → sexual selection

        // Buddha internal
        (12, 13, "caused", 0.7, -0.2),     // mind → suffering (insight leads to seeing dukkha)
        (12, 14, "related", 0.8, 0.1),     // mind → no-self
        (13, 15, "related", 0.6, 0.3),     // suffering → middle way
        (14, 15, "reinforces", 0.5, 0.4),  // no-self → middle way

        // Kant internal
        (16, 17, "related", 0.5, 0.1),     // categories → categorical imperative
        (16, 18, "reinforces", 0.9, -0.1), // categories → noumenal (core Kant)
        (17, 19, "reinforces", 0.6, 0.5),  // imperative → enlightenment

        // Turing internal
        (20, 21, "caused", 0.8, 0.2),      // universal machine → imitation game
        (20, 22, "related", 0.7, -0.1),    // universal machine → halting
        (21, 23, "related", 0.5, 0.3),     // imitation game → morphogenesis

        // ── Cross-domain bridges ──
        // Plato ↔ Kant (both about perception shaping reality)
        (0, 16, "similar", 0.7, 0.3),      // cave → categories
        (1, 18, "similar", 0.6, 0.2),      // forms → noumenal
        (24, 0, "reinforces", 0.5, 0.2),   // bridge: both shape reality

        // Nietzsche ↔ Darwin (both dethrone humanity)
        (4, 8, "reminds_of", 0.6, -0.3),   // god is dead → natural selection
        (25, 4, "reinforces", 0.5, -0.1),  // bridge: dethrone humanity

        // Buddha ↔ Turing (consciousness as process)
        (14, 21, "similar", 0.7, 0.3),     // no-self → imitation game
        (26, 12, "reinforces", 0.5, 0.3),  // bridge: consciousness process

        // Kant ↔ Nietzsche (ethics polarity)
        (17, 6, "contradicts", 0.8, -0.5), // universal law → self-created values
        (27, 17, "related", 0.5, 0.0),     // bridge: ethics tension

        // Darwin ↔ Turing (complexity from simplicity)
        (8, 23, "similar", 0.6, 0.4),      // natural selection → morphogenesis
        (28, 8, "reinforces", 0.5, 0.3),   // bridge: algorithm as creator

        // Buddha ↔ Nietzsche (suffering)
        (13, 7, "reminds_of", 0.5, -0.2),  // root of suffering → what doesn't kill
        (13, 5, "contradicts", 0.4, 0.1),  // suffering vs meaning

        // Plato ↔ Buddha (appearance vs reality)
        (0, 12, "reminds_of", 0.5, 0.2),   // cave → mind is everything
        (1, 14, "similar", 0.4, 0.1),      // forms → no-self
    ]
}

// ── Seeding ─────────────────────────────────────────────────────

async fn seed_database(pool: &PgPool) -> anyhow::Result<Vec<i32>> {
    let nodes = seed_nodes();
    let edges = seed_edges();
    let mut node_ids = Vec::with_capacity(nodes.len());

    println!("=== SEEDING GRAPH ===");
    println!("Creating {} nodes from 6 thinkers across 6 domains...\n", nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        // Generate embedding using fastembed
        let embedding = rgw::embed::embed_text(node.content)?;

        let id = rgw::db::create_memory_node(
            pool,
            node.content,
            node.domain,
            &embedding,
            node.importance,
            node.valence,
            0.5, // arousal
        ).await?;

        node_ids.push(id);
        println!("  [{:2}] {:30} | domain={:15} | importance={:.1}",
            id, &node.content[..node.content.len().min(30)], node.domain, node.importance);
    }

    println!("\nCreating {} edges...\n", edges.len());
    for &(src_idx, tgt_idx, edge_type, weight, emotional_charge) in &edges {
        let source_id = node_ids[src_idx];
        let target_id = node_ids[tgt_idx];
        let edge_id = rgw::db::create_edge(pool, source_id, target_id, edge_type, weight, emotional_charge).await?;
        println!("  edge {}: node {} --[{}]--> node {} (w={:.1}, e={:.1})",
            edge_id, source_id, edge_type, target_id, weight, emotional_charge);
    }

    println!("\nGraph seeded: {} nodes, {} edges\n", node_ids.len(), edges.len());
    Ok(node_ids)
}

// ── Walking ─────────────────────────────────────────────────────

async fn run_walks(pool: &PgPool) -> anyhow::Result<()> {
    use rgw::core::{self, SelfModel};
    use rgw::graph::EmotionalState;
    use rgw::walker;

    let self_model = Arc::new(Mutex::new(SelfModel::new()));
    let emotion = EmotionalState::default();

    println!("=== WALKER SESSION 1: Curiosity-biased exploration ===\n");

    let output = walker::walk_parallel(
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
    println!("  total hops: {}  walk_ms: {:.0}  hops/sec: {:.0}",
        output.total_hops, output.walk_ms, output.hops_per_sec);

    // ── Self-model state after walk ──
    let sm = self_model.lock().unwrap();
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
        .unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".to_string());
    println!("Connecting to: {}", db_url);
    let pool = rgw::db::connect(&db_url).await?;

    // Ensure schema exists
    let schema = include_str!("schema.sql");
    sqlx::query(schema).execute(&pool).await?;
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
