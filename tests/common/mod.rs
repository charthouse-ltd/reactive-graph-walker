//! Shared test fixtures: the small philosophy graph used by the integration
//! tests. Each test binary that needs a seeded graph does `mod common;`.
#![allow(dead_code)]

pub mod fixture;

use sqlx::PgPool;

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

pub async fn seed_database(pool: &PgPool) -> anyhow::Result<Vec<i32>> {
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

