//! RGW Digester — seeds the graph with philosophical knowledge and walks it.
//!
//! Standalone binary that:
//!   1. Creates DB schema (if needed)
//!   2. Seeds with ideas from Plato, Nietzsche, Darwin, Buddha, Kant, Turing
//!   3. Runs parallel walkers with inter-walker stigmergy
//!   4. Logs structural noticings, working memory, beliefs, predictions
//!
//! Build for VPS:
//!   cargo build --release --target x86_64-unknown-linux-gnu --example digester
//!
//! Build for Mac:
//!   cargo build --release --example digester
//!
//! Run:
//!   DATABASE_URL=postgresql://user:pass@host/rgw_test \
//!   RUST_LOG=info,rgw=debug \
//!   ./target/release/examples/digester

use std::sync::{Arc, Mutex};

use sqlx::PgPool;

// ── Philosophical seed data ─────────────────────────────────────

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

fn seed_edges() -> Vec<(usize, usize, &'static str, f32, f32)> {
    vec![
        // ── Internal domain edges ──
        (0, 1, "reinforces", 0.8, 0.2),
        (0, 2, "related", 0.6, 0.1),
        (1, 3, "related", 0.5, 0.1),
        (4, 5, "contradicts", 0.7, -0.4),
        (4, 6, "caused", 0.8, -0.3),
        (5, 7, "reinforces", 0.6, 0.5),
        (8, 9, "reinforces", 0.8, 0.5),
        (8, 10, "reinforces", 0.7, 0.4),
        (8, 11, "related", 0.5, 0.2),
        (12, 13, "caused", 0.7, -0.2),
        (12, 14, "related", 0.8, 0.1),
        (13, 15, "related", 0.6, 0.3),
        (16, 18, "reinforces", 0.9, -0.1),
        (17, 19, "reinforces", 0.6, 0.5),
        (20, 21, "caused", 0.8, 0.2),
        (20, 22, "related", 0.7, -0.1),
        (21, 23, "related", 0.5, 0.3),

        // ── Cross-domain bridges ──
        (0, 16, "similar", 0.7, 0.3),     // Plato cave → Kant categories
        (1, 18, "similar", 0.6, 0.2),     // Plato forms → Kant noumenal
        (24, 0, "reinforces", 0.5, 0.2),  // Bridge: reality shaped by mind
        (4, 8, "reminds_of", 0.6, -0.3),  // Nietzsche → Darwin (dethrone humanity)
        (25, 4, "reinforces", 0.5, -0.1), // Bridge: dethrone humanity
        (14, 21, "similar", 0.7, 0.3),    // Buddha no-self → Turing imitation
        (26, 12, "reinforces", 0.5, 0.3), // Bridge: consciousness as process
        (17, 6, "contradicts", 0.8, -0.5),// Kant imperative vs Nietzsche ubermensch
        (27, 17, "related", 0.5, 0.0),    // Bridge: ethics tension
        (8, 23, "similar", 0.6, 0.4),     // Darwin selection → Turing morphogenesis
        (28, 8, "reinforces", 0.5, 0.3),  // Bridge: algorithm as creator
        (0, 12, "reminds_of", 0.5, 0.2),  // Plato cave → Buddha mind
    ]
}

// ── Database setup ──────────────────────────────────────────────

async fn ensure_schema(pool: &PgPool) -> anyhow::Result<()> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector").execute(pool).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_vectors (
            id SERIAL PRIMARY KEY, content TEXT NOT NULL, domain VARCHAR(100) DEFAULT '',
            embedding vector(768), importance REAL DEFAULT 5.0, valence REAL DEFAULT 0.0,
            arousal REAL DEFAULT 0.5, access_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )"
    ).execute(pool).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_edges (
            id SERIAL PRIMARY KEY,
            source_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
            target_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
            edge_type VARCHAR(50) NOT NULL DEFAULT 'related', weight REAL DEFAULT 0.5,
            emotional_charge REAL DEFAULT 0.0, traversal_count INTEGER DEFAULT 0,
            last_traversed TIMESTAMPTZ DEFAULT NOW(), created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(source_id, target_id, edge_type)
        )"
    ).execute(pool).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS runtime_settings (
            key VARCHAR(100) PRIMARY KEY, value JSONB NOT NULL, updated_at TIMESTAMPTZ DEFAULT NOW()
        )"
    ).execute(pool).await?;

    Ok(())
}

async fn seed_database(pool: &PgPool) -> anyhow::Result<Vec<i32>> {
    let nodes = seed_nodes();
    let edges = seed_edges();
    let mut node_ids = Vec::with_capacity(nodes.len());

    println!("\n╔══════════════════════════════════════════════╗");
    println!("║   RGW DIGESTER — Philosophical Knowledge     ║");
    println!("║   Plato · Nietzsche · Darwin · Buddha         ║");
    println!("║   Kant · Turing                               ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("Seeding {} nodes across 6 domains...\n", nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        let embedding = rgw::embed::embed_text(node.content)?;
        let id = rgw::db::create_memory_node(
            pool, node.content, node.domain, &embedding,
            node.importance, node.valence, 0.5,
        ).await?;
        node_ids.push(id);
        println!("  [node {:2}] {:15} | {:45}",
            id, node.domain, &node.content[..node.content.len().min(45)]);
    }

    println!("\nCreating {} edges...\n", edges.len());
    for &(src_idx, tgt_idx, edge_type, weight, emotional_charge) in &edges {
        let source_id = node_ids[src_idx];
        let target_id = node_ids[tgt_idx];
        let _edge_id = rgw::db::create_edge(
            pool, source_id, target_id, edge_type, weight, emotional_charge
        ).await?;
    }

    println!("  ✓ {} nodes, {} edges seeded\n", node_ids.len(), edges.len());
    Ok(node_ids)
}

// ── Walking ─────────────────────────────────────────────────────

async fn run_walks(pool: &PgPool) -> anyhow::Result<()> {
    use rgw::core::{self, SelfModel};
    use rgw::graph::EmotionalState;
    use rgw::walker;

    let self_model = Arc::new(Mutex::new(SelfModel::new()));

    // ── Session 1: Curiosity-dominant ──
    println!("╔══════════════════════════════════════════════╗");
    println!("║   SESSION 1 — Initial Exploration             ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let output = walker::walk_parallel(pool, &EmotionalState::default(), 4, 6, &self_model).await;
    print_walk_summary(&output);

    // ── Session 2: Higher arousal (simulating engagement) ──
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║   SESSION 2 — Engaged Re-walk                 ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let excited = EmotionalState { valence: 0.3, arousal: 0.7, energy: 0.8 };
    let output2 = walker::walk_parallel(pool, &excited, 6, 5, &self_model).await;
    print_walk_summary(&output2);

    // ── Self-model state ──
    let sm = self_model.lock().unwrap();
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║   SELF-MODEL STATE                            ║");
    println!("╚══════════════════════════════════════════════╝\n");

    println!("  Cognitive mode:    {:?}", sm.mode);
    println!("  Valence:           {:+.3}  (emotional valence)", sm.valence);
    println!("  Arousal:           {:.3}   (activation level)", sm.arousal);
    println!("  Energy:            {:.3}   (capacity)", sm.energy);
    println!("  Plasticity gate:   {:.3}   (learning rate)", sm.plasticity_gate);
    println!("  Focus:             {} (intensity={:.2})", sm.current_focus, sm.focus_intensity);
    println!("  Signals processed: {}", sm.total_signals_processed);
    println!("  Noticings:         {}", sm.total_noticings);
    println!("  Surprise count:    {}", sm.surprise_count);

    // ── Beliefs ──
    println!("\n  ── Beliefs ({}) ──", sm.beliefs.len());
    for b in &sm.beliefs {
        println!("  [{}] {} (conf={:.2}, ev={})",
            b.domain, &b.statement[..b.statement.len().min(55)], b.confidence, b.evidence_count);
    }

    // ── Working Memory ──
    println!("\n  ── Working Memory ({}) ──", sm.working_memory.len());
    for wm in &sm.working_memory {
        println!("  [{}] {} (act={:.0}%)",
            wm.domain, &wm.content[..wm.content.len().min(50)], wm.activation * 100.0);
    }

    // ── Structural Noticings ──
    println!("\n  ── Structural Noticings ──");
    let structural_kinds = ["dead_end_cluster", "surprise_density", "cognitive_loop",
        "signal_poverty", "belief_stagnation", "prediction_error"];
    let mut found = false;
    for n in &sm.noticings {
        if structural_kinds.contains(&n.kind.as_str()) {
            println!("  🔍 [{}] {} (sig={:.2})", n.kind, n.observation, n.significance);
            found = true;
        }
    }
    if !found { println!("  (none — graph is healthy)"); }

    // ── Attention Patterns ──
    println!("\n  ── Attention Distribution ──");
    let mut patterns: Vec<_> = sm.attention_patterns.iter().collect();
    patterns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (domain, count) in patterns.iter().take(8) {
        let bar = "█".repeat((*count * 10.0).min(30.0) as usize);
        println!("  {:15} {:.2}  {}", domain, count, bar);
    }

    // ── Signals by source ──
    println!("\n  ── Module Activity ──");
    for (source, count) in &sm.signals_by_source {
        println!("  {:15} {} signals", source, count);
    }

    // ── Open Questions ──
    if !sm.open_questions.is_empty() {
        println!("\n  ── Open Questions ──");
        for q in &sm.open_questions {
            println!("  ? {}", &q[..q.len().min(80)]);
        }
    }

    println!("\n╔══════════════════════════════════════════════╗");
    println!("║   DIGESTION COMPLETE                          ║");
    println!("╚══════════════════════════════════════════════╝\n");

    Ok(())
}

fn print_walk_summary(output: &rgw::graph::WalkOutput) {
    println!("  Walkers: {}  |  Hops: {}  |  Time: {:.0}ms  |  Rate: {:.0} hops/s",
        output.walker_count, output.total_hops, output.walk_ms, output.hops_per_sec);
    println!("  Agreement: {:.2}  |  Novelty: {:.2}  |  Resonance: {:.2}",
        output.agreement_score, output.novelty_score, output.emotional_resonance);
    println!("  Domain: {}  |  Action: {}",
        output.primary_domain, output.recommended_action);
    println!("  Consensus: {:?}  |  Divergent: {:?}",
        &output.consensus_nodes[..output.consensus_nodes.len().min(4)],
        &output.divergent_nodes[..output.divergent_nodes.len().min(4)]);
}

// ── Main ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rgw=debug"))
        )
        .init();

    // Initialize embedder (downloads model on first run)
    println!("Loading embedding model...");
    rgw::embed::init()?;

    // Connect to database
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".to_string());
    println!("Connecting to: {}\n", db_url);
    let pool = rgw::db::connect(&db_url).await?;

    // Setup
    ensure_schema(&pool).await?;
    println!("Schema ensured.");

    let stats = rgw::db::graph_stats(&pool).await?;
    if stats.nodes > 0 {
        println!("Database has {} nodes, {} edges — using existing data.\n", stats.nodes, stats.edges);
    } else {
        seed_database(&pool).await?;
    }

    // Walk
    run_walks(&pool).await?;

    Ok(())
}
