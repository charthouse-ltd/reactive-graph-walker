//! RGW Cascade — full self-propagating cognition test.
//!
//! Seeds 100+ philosophical nodes with intentional topology:
//!   - Dense clusters (well-connected hubs)
//!   - Sparse bridges (few edges between clusters)
//!   - Dead-end zones (isolated nodes)
//!   - Contradiction zones (high-tension edges)
//!
//! Then runs the Diverger reactor — edge changes cascade, thresholds
//! fire spontaneous walks. Optionally connects DeepSeek for expression.
//!
//! Logs: structural noticings, prediction errors, beliefs, working memory,
//! collective walker trails, attention patterns, module activity.
//!
//! Build:
//!   cargo build --release --example cascade
//!
//! Run (local DB):
//!   DATABASE_URL=postgresql:///rgw_test?host=/tmp RUST_LOG=info,rgw=debug ./target/release/examples/cascade
//!
//! Run (with DeepSeek):
//!   DEEPSEEK_API_KEY=sk-... DATABASE_URL=... RUST_LOG=info,rgw=debug ./target/release/examples/cascade

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, Instant};

use sqlx::PgPool;

// ── Expanded seed data: 108 nodes, 12 thinkers, 10 domains ──────

struct SeedNode {
    content: &'static str,
    domain: &'static str,
    importance: f32,
    valence: f32,
}

fn seed_nodes() -> Vec<SeedNode> {
    let mut nodes = Vec::new();

    // ═══ DENSE CLUSTER 1: Epistemology hub (Plato↔Kant↔Descartes↔Hume) ═══
    let cluster1 = vec![
        // Plato
        ("The allegory of the cave: prisoners see only shadows, mistaking appearance for reality.", "epistemology", 9.0, 0.3),
        ("The Forms are perfect eternal archetypes. Physical objects are imperfect copies.", "metaphysics", 8.5, 0.2),
        ("Knowledge is justified true belief — recollection of what the soul knew before birth.", "epistemology", 8.0, 0.1),
        // Kant
        ("We do not see things as they are — the mind imposes categories on raw sensation.", "epistemology", 9.5, 0.0),
        ("The noumenal world is forever inaccessible. We know only phenomena.", "epistemology", 8.5, -0.1),
        ("The categorical imperative: act so your maxim could become universal law.", "ethics", 9.0, 0.3),
        // Descartes
        ("I think, therefore I am. The one indubitable truth is the existence of the thinking self.", "epistemology", 9.0, 0.4),
        ("Mind-body dualism: the mental and physical are fundamentally different substances.", "metaphysics", 8.0, 0.1),
        ("Systematic doubt: demolish all beliefs and rebuild from indubitable foundations.", "epistemology", 7.5, 0.0),
        // Hume
        ("All knowledge comes from experience. There are no innate ideas — the mind is a blank slate.", "epistemology", 8.5, -0.1),
        ("Causation is not rationally justified — we infer it from constant conjunction, not reason.", "epistemology", 9.0, -0.2),
        ("The self is a bundle of perceptions. There is no underlying substance called 'I'.", "consciousness", 8.5, -0.3),
    ];
    nodes.extend(cluster1);

    // ═══ DENSE CLUSTER 2: Meaning cluster (Nietzsche↔Schopenhauer↔Camus) ═══
    let cluster2 = vec![
        ("God is dead. We have killed him. How shall we comfort ourselves?", "existentialism", 9.5, -0.5),
        ("Become who you are. The Ubermensch creates their own values.", "ethics", 9.0, 0.6),
        ("What does not kill me makes me stronger.", "existentialism", 8.0, -0.2),
        // Schopenhauer
        ("The world is my representation. Reality is Will — blind, striving, insatiable.", "metaphysics", 9.0, -0.6),
        ("Life swings like a pendulum between pain and boredom.", "existentialism", 8.5, -0.7),
        ("Aesthetic contemplation offers temporary escape from the tyranny of Will.", "aesthetics", 7.5, 0.3),
        // Camus
        ("The absurd: humanity's need for meaning meets the universe's silent indifference.", "existentialism", 9.0, -0.4),
        ("One must imagine Sisyphus happy. Revolt against meaninglessness IS meaning.", "existentialism", 8.5, 0.5),
        ("Suicide is the only serious philosophical problem — is life worth living?", "existentialism", 8.0, -0.8),
    ];
    nodes.extend(cluster2);

    // ═══ DENSE CLUSTER 3: Science cluster (Darwin↔Feynman↔Gödel↔Turing) ═══
    let cluster3 = vec![
        ("Natural selection: cumulative adaptation, not random chance.", "science", 9.5, 0.0),
        ("There is grandeur in this view of life: endless forms most beautiful.", "science", 8.5, 0.7),
        ("The tree of life connects all species through common descent.", "science", 8.0, 0.5),
        // Feynman
        ("The double-slit experiment: observation changes reality at the quantum level.", "science", 9.0, 0.0),
        ("I think I can safely say nobody understands quantum mechanics.", "science", 8.0, -0.1),
        ("What I cannot create, I do not understand. Understanding requires construction.", "epistemology", 8.5, 0.3),
        // Gödel
        ("Any consistent formal system contains true statements it cannot prove.", "computation", 9.5, -0.2),
        ("The limits of mathematics are not practical — they are logical and fundamental.", "computation", 8.5, -0.1),
        ("Truth extends beyond provability. There is more to reality than formal systems.", "epistemology", 8.0, 0.2),
        // Turing
        ("A universal machine can compute anything computable — limits are mathematical, not mechanical.", "computation", 9.5, 0.2),
        ("The imitation game: if a machine talks like a human, on what grounds do we deny it thought?", "computation", 9.0, 0.0),
        ("Morphogenesis: simple rules produce complex patterns — stripes, leaves, galaxies.", "science", 8.0, 0.5),
    ];
    nodes.extend(cluster3);

    // ═══ DENSE CLUSTER 4: Consciousness cluster (Buddha↔Spinoza↔Wittgenstein) ═══
    let cluster4 = vec![
        ("All that we are is the result of what we have thought. The mind is everything.", "consciousness", 9.0, 0.5),
        ("The root of suffering is attachment to impermanent things.", "consciousness", 9.0, -0.3),
        ("There is no permanent self — only momentary aggregates of form and sensation.", "consciousness", 8.5, 0.0),
        ("The middle way: neither indulgence nor asceticism. Balanced present-moment awareness.", "ethics", 7.5, 0.6),
        // Spinoza
        ("God and Nature are one substance. Deus sive Natura. Pantheism, not atheism.", "metaphysics", 9.0, 0.5),
        ("Free will is an illusion born of ignorance of causes. All things follow necessarily.", "metaphysics", 8.5, -0.3),
        ("The highest good is the intellectual love of God — understanding the necessity of all things.", "ethics", 8.0, 0.4),
        // Wittgenstein
        ("The limits of my language are the limits of my world.", "language", 9.5, 0.1),
        ("Whereof one cannot speak, thereof one must be silent.", "language", 9.0, 0.0),
        ("Philosophy is a battle against the bewitchment of our intelligence by means of language.", "language", 8.5, -0.2),
        ("Meaning is use. A word's meaning is its function in the language-game, not a fixed definition.", "language", 8.0, 0.3),
    ];
    nodes.extend(cluster4);

    // ═══ SPARSE BRIDGE NODES (intentional few connections between clusters) ═══
    let bridges = vec![
        ("Plato's Forms and Kant's categories both suggest reality is shaped by mind, not discovered.", "epistemology", 7.0, 0.3),
        ("Nietzsche and Darwin both dethrone humanity from cosmic centrality.", "existentialism", 7.5, -0.1),
        ("Buddha's no-self and Hume's bundle theory converge: the self is process, not substance.", "consciousness", 8.0, 0.2),
        ("Kant's noumenal and Wittgenstein's 'whereof one cannot speak' mark the same boundary.", "epistemology", 7.5, 0.1),
        ("Gödel's incompleteness and Hume's skepticism: reason has limits that reason itself discovers.", "epistemology", 8.0, -0.1),
        ("Turing's morphogenesis and Spinoza's necessity: complexity emerges from simple deterministic rules.", "science", 7.5, 0.4),
        ("Schopenhauer's Will and Darwin's natural selection: the blind striving beneath all life.", "metaphysics", 7.0, -0.4),
        ("Camus' absurd and Wittgenstein's silence: the ethical and the mystical show themselves, cannot be said.", "existentialism", 7.5, 0.0),
        ("Feynman's quantum observer and Buddha's mind-is-everything: consciousness shapes reality.", "science", 8.0, 0.5),
        ("Descartes' cogito and Buddhism's meditation: both turn inward to find certainty.", "consciousness", 7.0, 0.3),
    ];
    nodes.extend(bridges);

    // ═══ DEAD-END ZONE: Isolated nodes with few connections ═══
    let dead_ends = vec![
        ("Rashomon effect: the same event perceived differently by different observers. Truth is perspectival.", "epistemology", 6.0, 0.0),
        ("The Ship of Theseus: if all parts are replaced, is it still the same ship? Identity over time.", "metaphysics", 6.5, 0.1),
        ("The Trolley Problem: is it moral to sacrifice one to save five? Ethics without easy answers.", "ethics", 6.0, -0.2),
        ("Chinese Room argument: syntax is not semantics. A computer following rules does not understand.", "computation", 7.0, -0.1),
        ("Mary's Room: does someone who knows all facts about color learn something new when seeing red?", "consciousness", 7.0, 0.2),
    ];
    nodes.extend(dead_ends);

    // ═══ CONTRADICTION ZONE: High-tension opposing ideas ═══
    let tensions = vec![
        ("Determinism: every event is causally determined by prior events. Free will is an illusion.", "metaphysics", 8.0, -0.4),
        ("Libertarian free will: consciousness can initiate causal chains, not merely transmit them.", "metaphysics", 7.5, 0.5),
        ("Moral realism: ethical truths exist independently of human opinion — like mathematical truths.", "ethics", 7.5, 0.2),
        ("Moral relativism: ethics are culturally constructed. There are no universal moral facts.", "ethics", 7.5, 0.1),
        ("Scientific realism: the entities described by science (electrons, genes) really exist.", "science", 8.0, 0.1),
        ("Scientific instrumentalism: scientific theories are useful fictions that predict, not describe reality.", "science", 7.5, 0.0),
        ("Strong AI: a sufficiently advanced computer would literally be conscious, not just simulate it.", "computation", 8.0, 0.5),
        ("Weak AI: machines can simulate intelligence but never possess genuine phenomenal consciousness.", "computation", 7.5, 0.0),
    ];
    nodes.extend(tensions);

    // ═══ WILD CARDS: Unexpected connections ═══
    let wildcards = vec![
        ("Heraclitus: You cannot step into the same river twice. Everything flows, nothing stands still.", "metaphysics", 8.5, 0.2),
        ("The Dao that can be spoken is not the eternal Dao. The named is not the unnameable.", "language", 8.0, 0.3),
        ("Pascal: The heart has reasons that reason knows nothing of.", "epistemology", 7.5, 0.4),
        ("Arendt: The banality of evil — monstrous acts committed by ordinary people following orders.", "ethics", 8.5, -0.7),
        ("McLuhan: The medium is the message. The form of communication shapes society more than content.", "language", 8.0, 0.1),
        ("Epicurus: Is God willing to prevent evil but not able? Then he is not omnipotent. Is he able but not willing? Then he is malevolent.", "metaphysics", 8.5, -0.5),
        ("Foucault: Knowledge is power. What counts as truth is determined by social power structures.", "epistemology", 8.0, -0.3),
        ("Kuhn: Scientific revolutions are paradigm shifts — not cumulative progress but radical reconceptualization.", "science", 8.5, 0.0),
        ("de Beauvoir: One is not born a woman, but becomes one. Gender is constructed, not biological destiny.", "ethics", 8.0, 0.3),
        ("Laozi: The softest thing in the world overcomes the hardest. Water wears down stone — not through force but persistence.", "ethics", 7.5, 0.5),
        ("Nagel: What is it like to be a bat? There is something it is like — consciousness has a subjective character science cannot capture.", "consciousness", 9.0, 0.1),
        ("Popper: A theory is scientific only if it is falsifiable. Unfalsifiable claims are not science but metaphysics.", "science", 8.5, 0.0),
        ("Chomsky: Language is an innate biological capacity — universal grammar is hardwired in the human brain.", "language", 8.0, 0.3),
        ("Sartre: Existence precedes essence. We are thrown into existence and must create our own meaning.", "existentialism", 8.5, -0.1),
        ("Heidegger: Being-toward-death. Awareness of mortality is what gives life its urgency and authenticity.", "existentialism", 8.5, -0.4),
        ("Rawls: Justice as fairness — design society behind a veil of ignorance where you don't know your position.", "ethics", 8.0, 0.3),
        ("Quine: The web of belief — knowledge is a network where any statement can be revised, but not all at once.", "epistemology", 8.0, 0.1),
        ("Dennett: Consciousness is an illusion — a user-interface the brain presents to itself, not a magical substance.", "consciousness", 8.0, -0.2),
    ];
    nodes.extend(wildcards);

    nodes.into_iter().map(|(c, d, i, v)| SeedNode { content: c, domain: d, importance: i, valence: v }).collect()
}

// ── Edge definitions ─────────────────────────────────────────────
// (source_idx, target_idx, edge_type, weight, emotional_charge)

fn seed_edges() -> Vec<(usize, usize, &'static str, f32, f32)> {
    let mut e = Vec::new();

    // ── Cluster 1 internal (nodes 0-11): dense epistemology hub ──
    let c1: Vec<(usize, usize, &str, f32, f32)> = vec![
        (0,1,"reinforces",0.8,0.2), (0,2,"related",0.6,0.1), (0,3,"similar",0.7,0.3),
        (1,7,"related",0.5,0.1), (2,8,"similar",0.4,0.1), (3,4,"reinforces",0.9,-0.1),
        (3,5,"related",0.5,0.1), (4,6,"similar",0.6,0.2), (5,8,"contradicts",0.5,0.0),
        (6,7,"related",0.5,0.1), (6,8,"reinforces",0.7,0.2), (8,9,"contradicts",0.6,-0.1),
        (9,10,"reinforces",0.8,-0.1), (10,11,"similar",0.7,-0.2), (7,11,"related",0.4,0.1),
        (0,6,"similar",0.5,0.3), (1,4,"similar",0.6,0.1), (2,10,"contradicts",0.5,-0.2),
    ];
    e.extend(c1);

    // ── Cluster 2 internal (nodes 12-20): meaning/purpose hub ──
    let c2 = vec![
        (12,13,"caused",0.8,-0.3), (12,14,"reinforces",0.5,-0.1), (13,14,"related",0.4,0.3),
        (15,16,"reinforces",0.7,-0.5), (15,17,"caused",0.6,-0.3), (16,18,"related",0.5,0.3),
        (17,18,"similar",0.6,0.4), (18,19,"reinforces",0.8,-0.4), (12,15,"reminds_of",0.6,-0.4),
        (13,17,"similar",0.5,0.2), (14,18,"related",0.4,-0.1),
    ];
    e.extend(c2);

    // ── Cluster 3 internal (nodes 21-32): science/computation hub ──
    let c3 = vec![
        (21,22,"reinforces",0.8,0.5), (21,23,"related",0.7,0.4), (22,23,"similar",0.6,0.5),
        (24,25,"related",0.6,0.0), (24,26,"reinforces",0.5,0.2), (25,28,"similar",0.4,0.1),
        (27,28,"reinforces",0.9,-0.1), (27,29,"related",0.6,0.1), (28,29,"similar",0.7,0.0),
        (30,31,"caused",0.8,0.2), (30,32,"related",0.7,-0.1), (31,33,"related",0.5,0.3),
        (21,30,"similar",0.5,0.3), (23,33,"similar",0.5,0.2), (27,26,"related",0.4,-0.1),
    ];
    e.extend(c3);

    // ── Cluster 4 internal (nodes 33-43): consciousness/language hub ──
    let c4 = vec![
        (33,34,"caused",0.7,-0.2), (33,35,"related",0.8,0.1), (34,36,"related",0.6,0.3),
        (35,40,"similar",0.5,0.1), (37,38,"reinforces",0.9,0.4), (37,39,"related",0.6,0.3),
        (40,41,"reinforces",0.8,0.1), (40,42,"caused",0.6,-0.1), (41,42,"similar",0.5,0.0),
        (41,43,"related",0.4,0.2), (33,37,"reminds_of",0.5,0.3), (35,39,"similar",0.4,0.2),
    ];
    e.extend(c4);

    // ── Bridge nodes (44-53): sparse inter-cluster connections ──
    let bridges = vec![
        // Epistemology ↔ Meaning
        (0,12,"reminds_of",0.4,0.0),
        (8,17,"similar",0.3,0.1),
        // Epistemology ↔ Science
        (3,21,"similar",0.5,0.2),
        (10,27,"related",0.4,-0.1),
        // Meaning ↔ Science
        (15,21,"contradicts",0.5,-0.3),
        (16,22,"reminds_of",0.3,0.0),
        // Consciousness ↔ Epistemology
        (33,0,"reminds_of",0.5,0.2),
        (35,11,"similar",0.6,0.1),
        // Consciousness ↔ Computation
        (35,30,"similar",0.5,0.3),
        (34,31,"related",0.4,0.0),
        // Language ↔ Epistemology
        (40,4,"similar",0.5,0.1),
        (41,8,"reminds_of",0.4,0.0),
        // Meaning ↔ Consciousness
        (18,34,"similar",0.4,0.1),
        (19,36,"reminds_of",0.3,0.1),
    ];
    e.extend(bridges);

    // ── Dead-end edges (54-58): minimal connections ──
    let deads = vec![
        (54,0,"related",0.2,0.0),   // Ship of Theseus → Plato cave (weak)
        (55,5,"reminds_of",0.2,0.0), // Trolley → Kant imperative (weak)
        (56,30,"contradicts",0.3,0.0), // Chinese Room → Turing (weak)
        (57,35,"related",0.2,0.1),  // Mary's Room → Buddha no-self (weak)
        (58,53,"reminds_of",0.1,0.0), // Rashomon → bridge (very weak)
    ];
    e.extend(deads);

    // ── Contradiction zone edges (59-66): high tension ──
    let tensions = vec![
        (59,60,"contradicts",0.9,-0.8),  // Determinism vs Free Will
        (61,62,"contradicts",0.8,-0.6),  // Moral realism vs relativism
        (63,64,"contradicts",0.7,-0.4),  // Scientific realism vs instrumentalism
        (65,66,"contradicts",0.9,-0.7),  // Strong AI vs Weak AI
        (59,37,"related",0.4,-0.2),      // Determinism ↔ Spinoza
        (60,13,"similar",0.4,0.3),       // Free will ↔ Ubermensch
        (65,31,"reminds_of",0.5,0.1),    // Strong AI ↔ Turing
        (66,56,"reinforces",0.5,0.0),    // Weak AI ↔ Chinese Room
    ];
    e.extend(tensions);

    // ── Wild card edges (67-84): unexpected bridges ──
    let wilds = vec![
        (67,1,"similar",0.4,0.1),    // Heraclitus flux ↔ Plato Forms
        (68,40,"similar",0.6,0.3),   // Dao ↔ Wittgenstein silence
        (69,8,"reminds_of",0.4,0.2), // Pascal heart ↔ Descartes doubt
        (70,18,"related",0.3,-0.4),  // Arendt banality ↔ Camus absurd
        (71,42,"similar",0.5,0.2),   // McLuhan medium ↔ Wittgenstein meaning
        (72,12,"contradicts",0.6,-0.5), // Epicurus evil ↔ Nietzsche God is dead
        (73,10,"reminds_of",0.5,-0.2), // Foucault power ↔ Hume causation
        (74,21,"similar",0.6,0.1),   // Kuhn paradigm ↔ Darwin selection
        (75,13,"related",0.4,0.3),   // de Beauvoir gender ↔ Nietzsche create values
        (76,36,"similar",0.4,0.4),   // Laozi soft ↔ Buddha middle way
        (77,24,"related",0.5,0.0),   // Nagel bat ↔ Feynman quantum
        (78,63,"reinforces",0.5,0.0), // Popper falsify ↔ Scientific realism
        (79,43,"similar",0.5,0.1),   // Chomsky grammar ↔ Wittgenstein language-game
        (80,19,"similar",0.6,0.0),   // Sartre existence ↔ Camus Sisyphus
        (81,17,"related",0.5,-0.3),  // Heidegger death ↔ Schopenhauer pain
        (82,5,"reminds_of",0.4,0.2), // Rawls veil ↔ Kant imperative
        (83,10,"contradicts",0.3,0.0), // Quine web ↔ Hume bundle
        (84,35,"contradicts",0.4,-0.1), // Dennett illusion ↔ Buddha no-self
    ];
    e.extend(wilds);

    e
}

// ── Database ─────────────────────────────────────────────────────

async fn ensure_schema(pool: &PgPool) -> anyhow::Result<()> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector").execute(pool).await?;
    sqlx::query("CREATE TABLE IF NOT EXISTS memory_vectors (id SERIAL PRIMARY KEY, content TEXT NOT NULL, domain VARCHAR(100) DEFAULT '', embedding vector(768), importance REAL DEFAULT 5.0, valence REAL DEFAULT 0.0, arousal REAL DEFAULT 0.5, access_count INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())").execute(pool).await?;
    sqlx::query("CREATE TABLE IF NOT EXISTS memory_edges (id SERIAL PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE, target_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE, edge_type VARCHAR(50) NOT NULL DEFAULT 'related', weight REAL DEFAULT 0.5, emotional_charge REAL DEFAULT 0.0, traversal_count INTEGER DEFAULT 0, last_traversed TIMESTAMPTZ DEFAULT NOW(), created_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(source_id, target_id, edge_type))").execute(pool).await?;
    Ok(())
}

async fn seed_graph(pool: &PgPool) -> anyhow::Result<Vec<i32>> {
    let nodes = seed_nodes();
    let edges = seed_edges();
    let mut ids = Vec::with_capacity(nodes.len());

    println!("Seeding {} nodes...", nodes.len());
    for node in &nodes {
        let emb = rgw::embed::embed_text(node.content)?;
        let id = rgw::db::create_memory_node(pool, node.content, node.domain, &emb, node.importance, node.valence, 0.5).await?;
        ids.push(id);
    }

    println!("Creating {} edges...", edges.len());
    for &(s, t, ty, w, ec) in &edges {
        if s < ids.len() && t < ids.len() {
            let _ = rgw::db::create_edge(pool, ids[s], ids[t], ty, w, ec).await;
        }
    }

    let stats = rgw::db::graph_stats(pool).await?;
    println!("Graph: {} nodes, {} edges\n", stats.nodes, stats.edges);
    Ok(ids)
}

// ── Cascade Engine ───────────────────────────────────────────────

async fn run_cascade(pool: &PgPool, node_ids: &[i32]) -> anyhow::Result<()> {
    use rgw::core::{self, SelfModel, CognitiveMode, Signal};
    use rgw::graph::{EmotionalState, WalkerBias};
    use rgw::walker;
    use rgw::diverger::{Diverger, DivergerConfig, EdgeChange};

    let self_model = Arc::new(Mutex::new(SelfModel::new()));
    let julian_url = "";

    // Initialize Diverger
    let diverger = Diverger::new(pool.clone(), self_model.clone(), julian_url);

    // Seed initial energy into the graph to kickstart cascades
    let seed_nodes: Vec<i32> = node_ids.iter().take(20).copied().collect();
    diverger.seed_energy(seed_nodes, 0.3).await;

    // Set emotional state
    diverger.set_emotion(EmotionalState { valence: 0.0, arousal: 0.4, energy: 0.8 }).await;

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║   RGW CASCADE — Self-Propagating Cognition          ║");
    println!("║   108 nodes, 10 domains, intentional topology        ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ── PHASE 1: Manual walk sessions to warm the graph ──
    println!("─── PHASE 1: Warm-up walks ───\n");
    for session in 0..3 {
        let arousal = 0.3 + session as f32 * 0.15;
        diverger.set_emotion(EmotionalState { valence: 0.1, arousal, energy: 0.8 }).await;
        let (output, _) = walker::walk_parallel(pool, &EmotionalState { valence: 0.1, arousal, energy: 0.8 }, 6, 6, &self_model).await;
        println!("  Session {}: {:2} hops | agreement={:.2} novelty={:.2} | domain={} action={}",
            session + 1, output.total_hops, output.agreement_score, output.novelty_score,
            output.primary_domain, output.recommended_action);
    }

    // ── PHASE 2: Inject edge changes to trigger diverger cascades ──
    println!("\n─── PHASE 2: Diverger cascade (spontaneous walks) ───\n");

    // Pick some edges from warm walks and notify the diverger
    let edge_sample = vec![
        EdgeChange { edge_id: 1, source_id: node_ids[0], target_id: node_ids[1], delta: 0.3, edge_type: "reinforces".into() },
        EdgeChange { edge_id: 5, source_id: node_ids[3], target_id: node_ids[4], delta: 0.4, edge_type: "reinforces".into() },
        EdgeChange { edge_id: 20, source_id: node_ids[12], target_id: node_ids[13], delta: 0.5, edge_type: "caused".into() },
        EdgeChange { edge_id: 35, source_id: node_ids[21], target_id: node_ids[22], delta: 0.3, edge_type: "reinforces".into() },
        EdgeChange { edge_id: 50, source_id: node_ids[33], target_id: node_ids[34], delta: 0.4, edge_type: "caused".into() },
        EdgeChange { edge_id: 59, source_id: node_ids[59], target_id: node_ids[60], delta: 0.6, edge_type: "contradicts".into() },
        EdgeChange { edge_id: 65, source_id: node_ids[65], target_id: node_ids[66], delta: 0.7, edge_type: "contradicts".into() },
    ];

    for change in &edge_sample {
        diverger.notify_edge_change(change.clone());
    }

    println!("  Injected {} edge changes. Waiting for cascades...\n", edge_sample.len());

    // Let the diverger process for a few seconds
    tokio::time::sleep(Duration::from_secs(3)).await;

    // ── PHASE 3: High-arousal re-walk ──
    println!("─── PHASE 3: High-arousal re-walk ───\n");

    // Set a prediction: expect to find connections in epistemology
    {
        let mut sm = self_model.lock().await;
        sm.predict("epistemology", "connection between rationalists and empiricists", 0.6);
    }

    diverger.set_emotion(EmotionalState { valence: 0.3, arousal: 0.8, energy: 0.7 }).await;
    let (output3, _) = walker::walk_parallel(pool, &EmotionalState { valence: 0.3, arousal: 0.8, energy: 0.7 }, 8, 5, &self_model).await;
    println!("  {:2} walkers × {} hops | agreement={:.2} novelty={:.2} | domain={} action={}",
        8, output3.total_hops, output3.agreement_score, output3.novelty_score,
        output3.primary_domain, output3.recommended_action);

    // ── PHASE 4: Final diverger stats ──
    tokio::time::sleep(Duration::from_secs(1)).await;
    let d_stats = diverger.stats().await;
    println!("\n─── DIVERGER STATE ───");
    println!("  Alive: {} | Active nodes: {} | Total energy: {:.2}", d_stats.alive, d_stats.active_nodes, d_stats.total_energy);
    println!("  Walks fired: {} | Cascades: {} | Edges changed: {}", d_stats.walks_fired, d_stats.cascades_total, d_stats.edges_changed);
    if !d_stats.hottest_nodes.is_empty() {
        println!("  Hottest nodes: {:?}", &d_stats.hottest_nodes[..d_stats.hottest_nodes.len().min(5)]);
    }

    // ── SELF-MODEL REPORT ──
    let sm_snapshot = {
        let sm = self_model.lock().await;
        sm.clone()  // Clone and release the lock immediately
    }; // Lock released here — critical: Phase 5 also needs the lock!
    let sm = &sm_snapshot; // Use the snapshot for the report
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║   SELF-MODEL REPORT                                  ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    println!("  Mode: {:?} | Plasticity: {:.3}", sm.mode, sm.plasticity_gate);
    println!("  Valence={:+.3} Arousal={:.3} Energy={:.3}", sm.valence, sm.arousal, sm.energy);
    println!("  Signals: {} | Noticings: {} | Surprises: {}", sm.total_signals_processed, sm.total_noticings, sm.surprise_count);
    println!("  Focus: {} (intensity={:.2})", sm.current_focus, sm.focus_intensity);

    // ── Beliefs ──
    println!("\n  ── Beliefs ({}) ──", sm.beliefs.len());
    if sm.beliefs.is_empty() {
        println!("  (none yet — need more walks for pattern accumulation)");
    }
    for b in &sm.beliefs {
        println!("  [{}] {} (conf={:.2} ev={})", b.domain, core::safe_truncate(&b.statement, 60), b.confidence, b.evidence_count);
    }

    // ── Working Memory ──
    println!("\n  ── Working Memory ({}) ──", sm.working_memory.len());
    for wm in &sm.working_memory {
        println!("  [{}] {} (act={:.0}%)", wm.domain, core::safe_truncate(&wm.content, 55), wm.activation * 100.0);
    }

    // ── Structural Noticings ──
    println!("\n  ── Structural Noticings ──");
    let structural_kinds = [
        "dead_end_cluster", "surprise_density", "cognitive_loop",
        "signal_poverty", "belief_stagnation", "prediction_error",
        "emotional_shift", "focus_shift", "obsession", "wound_activated",
        "exhaustion", "activation",
    ];
    let mut found = 0;
    for n in &sm.noticings {
        if structural_kinds.contains(&n.kind.as_str()) {
            let emoji = match n.kind.as_str() {
                "dead_end_cluster" => "🕳️", "surprise_density" => "✨", "cognitive_loop" => "🔄",
                "signal_poverty" => "📡", "belief_stagnation" => "📊", "prediction_error" => "❗",
                "emotional_shift" => "💭", "focus_shift" => "👁️", "obsession" => "🔁",
                "wound_activated" => "💢", "exhaustion" => "🪫", "activation" => "⚡",
                _ => "•",
            };
            println!("  {} [{:<18}] {} (sig={:.2} val={:+.2})",
                emoji, n.kind, core::safe_truncate(&n.observation, 65), n.significance, n.valence);
            found += 1;
        }
    }
    if found == 0 { println!("  (none — more walks or larger graph needed)"); }
    println!("  Total noticings across all kinds: {}", sm.noticings.len());

    // ── Attention ──
    println!("\n  ── Attention Distribution ──");
    let mut patterns: Vec<_> = sm.attention_patterns.iter().collect();
    patterns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (domain, count) in patterns.iter().take(12) {
        let bar = "█".repeat((*count * 2.0).min(40.0) as usize);
        println!("  {:15} {:6.2}  {}", domain, count, bar);
    }

    // ── Module Activity ──
    println!("\n  ── Module Activity ──");
    for (source, count) in &sm.signals_by_source {
        let bar = "▬".repeat(*count as usize);
        println!("  {:15} {:4}  {}", source, count, bar);
    }

    // ── Predictions ──
    println!("\n  ── Active Predictions ──");
    for (domain, pred) in &sm.predictions {
        println!("  [{}] '{}' (conf={:.2})", domain, pred.predicted, pred.confidence);
    }

    // ── Emergent Patterns ──
    if !sm.emergent_patterns.is_empty() {
        println!("\n  ── Emergent Patterns ──");
        for p in &sm.emergent_patterns {
            println!("  [{}] {} (str={:.2} ev={})", p.domain, p.description, p.strength, p.evidence_count);
        }
    }

    // ── Graph stats ──
    let gs = rgw::db::graph_stats(pool).await?;
    println!("\n  ── Graph ──");
    println!("  Nodes: {} | Edges: {} | Avg edges/node: {:.1}", gs.nodes, gs.edges, gs.edges as f64 / gs.nodes.max(1) as f64);

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║   CASCADE COMPLETE                                   ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ── PHASE 5: RGW ↔ LLM Colloquy ──
    // The LLM is called directly from the cognitive loop —
    // RGW reports its state, the LLM reflects, the insight feeds
    // back as a signal and a graph node, then RGW re-walks.
    if let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") {
        if !api_key.is_empty() {
            println!("─── PHASE 5: RGW ↔ DeepSeek Colloquy ───\n");
            let llm = rgw::llm::LlmEngine::new(&api_key, "deepseek-chat")?;

            // ── TURN 1: RGW → LLM ──
            println!("  ╔══ TURN 1: RGW reports its state ══╗\n");
            let sm1 = self_model.lock().await.clone();
            let wm1 = sm1.format_working_memory();
            let top_domains = {
                let mut pats: Vec<_> = sm1.attention_patterns.iter().collect();
                pats.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                pats.iter().take(5).map(|(d, _)| d.as_str()).collect::<Vec<_>>()
            };
            let belief_ctx: String = sm1.beliefs.iter()
                .map(|b| format!("- [{}] {}", b.domain, b.statement)).collect::<Vec<_>>().join("\n");
            let noticing_ctx: String = sm1.noticings.iter()
                .map(|n| format!("  [{}] {}", n.kind, n.observation))
                .collect::<Vec<_>>().join("\n");

            let system1 = "You are RGW's expression layer. You receive the cognitive state of a graph-based engine \
                           that has just traversed a philosophical knowledge graph. \
                           Respond with a concise, insightful analysis (2-4 sentences). \
                           Be specific about what patterns or tensions you see.";
            let user1 = format!(
                "SELF-MODEL STATE:\nSurprises={} Noticings={} Beliefs={}\nTop domains: {:?}\n\
                 Working memory:\n{}\nBeliefs:\n{}\nNoticings:\n{}\n\nWhat is the most interesting pattern?",
                sm1.surprise_count, sm1.noticings.len(), sm1.beliefs.len(),
                top_domains,
                if wm1.is_empty() { "  (empty)" } else { &wm1 },
                if belief_ctx.is_empty() { "  (none)" } else { &belief_ctx },
                if noticing_ctx.is_empty() { "  (none)" } else { &noticing_ctx },
            );

            let response1 = llm.chat(system1, &user1, Some(512), 0.7).await?;
            println!("  DeepSeek: \"{}\"\n", core::safe_truncate(&response1, 200));

            // Feed LLM insight back → Signal → core::process()
            {
                let mut sm = self_model.lock().await;
                let insight_signal = Signal::new("llm_insight", &response1)
                    .with_domain("integration")
                    .with_intensity(0.7);
                core::process(insight_signal, &mut sm);
                sm.predict("integration", "synthesis of philosophical domains", 0.5);
            }
            // Create graph node from insight
            let insight_embedding = rgw::embed::embed_text(core::safe_truncate(&response1, 500))?;
            let insight_node_id = rgw::db::create_memory_node(
                pool, core::safe_truncate(&response1, 500),
                "integration", &insight_embedding, 7.0, 0.3, 0.5
            ).await?;
            for &bridge_idx in &[74, 76, 77, 78] {
                if bridge_idx < node_ids.len() {
                    let _ = rgw::db::create_edge(pool, insight_node_id, node_ids[bridge_idx],
                        "synthesizes", 0.3, 0.2).await;
                }
            }
            println!("  → Created node {} from LLM insight + 4 edges\n", insight_node_id);

            // ── TURN 2: Re-walk with LLM insight ──
            println!("  ╔══ TURN 2: RGW re-explores with new knowledge ══╗\n");
            diverger.set_emotion(EmotionalState { valence: 0.2, arousal: 0.6, energy: 0.7 }).await;
            let (_, _) = walker::walk_parallel(pool,
                &EmotionalState { valence: 0.2, arousal: 0.6, energy: 0.7 },
                6, 5, &self_model).await;
            let sm2 = self_model.lock().await.clone();
            println!("  Beliefs: {} (was {}) | Noticings: {} (was {}) | Surprises: {} (was {})",
                sm2.beliefs.len(), sm1.beliefs.len(),
                sm2.noticings.len(), sm1.noticings.len(),
                sm2.surprise_count, sm1.surprise_count);

            // ── TURN 3: RGW reports evolution → LLM reflects ──
            println!("\n  ╔══ TURN 3: RGW reports its evolution ══╗\n");
            let wm2 = sm2.format_working_memory();
            let user2 = format!(
                "EARLIER, YOU SAID: \"{}\"\n\n\
                 AFTER YOUR INSIGHT, I RE-EXPLORED. NEW STATE:\n\
                 Beliefs: {} (was {}) | Noticings: {} (was {}) | Surprises: {} (was {})\n\
                 Plasticity: {:.3}\n\
                 Working memory:\n{}\n\n\
                 What changed? What does this tell us about the system?",
                core::safe_truncate(&response1, 150),
                sm2.beliefs.len(), sm1.beliefs.len(),
                sm2.noticings.len(), sm1.noticings.len(),
                sm2.surprise_count, sm1.surprise_count,
                sm2.plasticity_gate,
                if wm2.is_empty() { "  (empty)" } else { &wm2 },
            );
            let system2 = "You are RGW's expression layer. Earlier you provided an insight. \
                           Now you see how the system changed after your insight was incorporated. \
                           Reflect on what this means about the RGW+LLM cognitive loop.";
            match llm.chat(system2, &user2, Some(512), 0.8).await {
                Ok(response2) => {
                    println!("  DeepSeek reflects:\n");
                    for line in response2.lines() {
                        println!("  {}", line);
                    }
                    println!();
                    let mut sm = self_model.lock().await;
                    let reflect_signal = Signal::new("llm_reflection", &response2)
                        .with_domain("integration").with_intensity(0.5);
                    core::process(reflect_signal, &mut sm);
                }
                Err(e) => println!("  Turn 3 failed: {}\n", e),
            }
        }
    } else {
        println!("  (set DEEPSEEK_API_KEY to enable LLM expression)\n");
    }

    Ok(())
}

// ── Main ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rgw=debug")))
        .try_init().ok();

    println!("Loading embedding model...");
    rgw::embed::init()?;

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql:///rgw_test?host=/tmp".into());
    let pool = rgw::db::connect(&db_url).await?;
    ensure_schema(&pool).await?;

    // Re-seed: drop old data
    let stats = rgw::db::graph_stats(&pool).await?;
    if stats.nodes > 0 {
        println!("Clearing existing {} nodes...", stats.nodes);
        sqlx::query("DELETE FROM memory_edges").execute(&pool).await?;
        sqlx::query("DELETE FROM memory_vectors").execute(&pool).await?;
    }

    let node_ids = seed_graph(&pool).await?;
    run_cascade(&pool, &node_ids).await?;

    Ok(())
}
