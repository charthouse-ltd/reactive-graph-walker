//! Graph types and emotional modulation logic.

use std::collections::{HashMap, HashSet};

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Emotional state — modulates how walkers traverse edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f32,  // -1 to 1
    pub arousal: f32,  // 0 to 1
    pub energy: f32,   // 0 to 1
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.3,
            energy: 0.7,
        }
    }
}

// ── Walker Collective ───────────────────────────────────────────
// Stigmergic shared context for inter-walker coordination.
// Walkers leave trails; other walkers read them.
// Like ants with pheromones — no explicit message passing.
// Uses try_lock() — non-blocking, advisory.

/// Shared context for parallel walker coordination during traversal.
/// Walkers write discoveries, read each other's trails, and
/// modulate edge scoring based on what the collective knows.
#[derive(Debug, Clone)]
pub struct WalkerCollective {
    /// Nodes visited by any walker (node_id → visit_count)
    pub visited_nodes: HashMap<i32, u32>,

    /// Dead ends discovered by any walker
    pub dead_end_nodes: HashSet<i32>,

    /// Domains where surprises were found (cross-domain transitions)
    /// Walkers converge toward these
    pub surprise_domains: HashSet<String>,

    /// Domains actively being explored (domain → walker_count)
    pub active_domains: HashMap<String, u32>,

    /// Collective emotional drift — running average across walkers
    pub drift_valence: f32,
    pub drift_arousal: f32,
}

impl WalkerCollective {
    pub fn new() -> Self {
        Self {
            visited_nodes: HashMap::new(),
            dead_end_nodes: HashSet::new(),
            surprise_domains: HashSet::new(),
            active_domains: HashMap::new(),
            drift_valence: 0.0,
            drift_arousal: 0.3,
        }
    }
}

impl Default for WalkerCollective {
    fn default() -> Self {
        Self::new()
    }
}

/// Walker bias — each bias changes which edges a walker prefers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WalkerBias {
    Fear,
    Curiosity,
    Experience,
    Random,
    Analytical,
    Contrarian,
}

impl WalkerBias {
    /// All available biases
    pub fn all() -> &'static [WalkerBias] {
        &[
            WalkerBias::Fear,
            WalkerBias::Curiosity,
            WalkerBias::Experience,
            WalkerBias::Random,
        ]
    }

    /// Score an edge based on this bias + emotional state
    pub fn score_edge(
        &self,
        edge_type: &str,
        edge_weight: f32,
        emotional_charge: f32,
        traversal_count: i32,
        emotion: &EmotionalState,
    ) -> f32 {
        let mut w = edge_weight;

        match self {
            WalkerBias::Fear => {
                if edge_type == "caused" || edge_type == "contradicts" {
                    w *= 2.5;
                }
            }
            WalkerBias::Curiosity => {
                if edge_type == "reminds_of" {
                    w *= 1.5;
                }
                // Follow weak/unexplored edges
                if edge_weight < 0.3 {
                    w += 3.0;
                }
            }
            WalkerBias::Experience => {
                if edge_type == "reinforces" || edge_type == "similar" {
                    w *= 2.0;
                }
            }
            WalkerBias::Analytical => {
                if edge_type == "caused" || edge_type == "reinforces" {
                    w *= 2.0;
                }
            }
            WalkerBias::Contrarian => {
                if edge_type == "contradicts" {
                    w *= 3.0;
                }
            }
            WalkerBias::Random => {
                // Pure exploration — weight is random
                return rand::rng().random::<f32>();
            }
        }

        // Emotional modulation (applies to all biases except random)
        w *= 1.0 + emotion.arousal; // High arousal = wider reach

        // Valence alignment
        if emotion.valence.abs() > 0.3 {
            let alignment = 1.0 - (emotional_charge - emotion.valence).abs();
            w *= 0.5 + alignment;
        }

        // Freshness penalty — recently traversed edges slightly deprioritized
        if traversal_count > 5 {
            w *= 0.8;
        }

        w.max(0.001)
    }

    /// Score an edge in Compliant mode — no emotional modulation.
    /// Pure weight-based scoring. The edge's weight IS the score,
    /// plus bias-specific type preferences. No arousal, no valence
    /// alignment, no emotional charge influence.
    pub fn score_edge_compliant(
        &self,
        edge_type: &str,
        edge_weight: f32,
        traversal_count: i32,
    ) -> f32 {
        let mut w = edge_weight;

        match self {
            WalkerBias::Experience => {
                if edge_type == "reinforces" || edge_type == "similar" {
                    w *= 2.0;
                }
            }
            WalkerBias::Analytical => {
                if edge_type == "caused" || edge_type == "reinforces" {
                    w *= 2.0;
                }
            }
            // Other biases shouldn't reach here in compliant mode,
            // but handle gracefully — pure weight, no modification
            _ => {}
        }

        // Freshness penalty still applies
        if traversal_count > 5 {
            w *= 0.8;
        }

        w.max(0.001)
    }

    /// Score an edge with full context: collective trails + self-model state.
    /// This is where inter-walker stigmergy and accumulated knowledge
    /// (beliefs, wounds, competencies, working memory) influence traversal.
    ///
    /// Parameters beyond the base scoring:
    ///   collective     — what other walkers have discovered (stigmergy)
    ///   next_node_id   — the target node being considered
    ///   next_domain    — the domain of the target node
    ///   beliefs        — accumulated beliefs from pattern detection
    ///   wounds         — accumulated pain by domain
    ///   competencies   — accumulated skill by domain
    ///   wm_domains     — domains currently held in working memory
    pub fn score_edge_with_context(
        &self,
        edge_type: &str,
        edge_weight: f32,
        emotional_charge: f32,
        traversal_count: i32,
        emotion: &EmotionalState,
        collective: Option<&WalkerCollective>,
        next_node_id: i32,
        next_domain: &str,
        beliefs: &[crate::core::Belief],
        wounds: &HashMap<String, f32>,
        competencies: &HashMap<String, f32>,
        wm_domains: &HashSet<String>,
    ) -> f32 {
        // Start with base emotional scoring
        let mut w = self.score_edge(edge_type, edge_weight, emotional_charge, traversal_count, emotion);

        // ── Stigmergic modulation (inter-walker collective) ──

        if let Some(c) = collective {
            // Dead end avoidance: another walker already hit a wall here
            if c.dead_end_nodes.contains(&next_node_id) {
                return 0.0;
            }

            // Visited node penalty: more walkers = less interesting
            if let Some(&visit_count) = c.visited_nodes.get(&next_node_id) {
                w *= 1.0 / (1.0 + visit_count as f32);
            }

            // Surprise convergence: another walker found a cross-domain leap
            if c.surprise_domains.contains(next_domain) && edge_type != "contradicts" {
                w *= 2.0;
            }

            // Domain diversification: too many walkers in same domain → explore elsewhere
            if let Some(&active_count) = c.active_domains.get(next_domain) {
                if active_count >= 2 {
                    w *= 0.5;
                }
            }
        }

        // ── Self-model modulation (accumulated knowledge) ──

        // Wound avoidance: this domain has caused pain → avoid
        if let Some(&wound) = wounds.get(next_domain) {
            if wound > 0.3 {
                w *= 1.0 - wound * 0.5; // High wound = strong avoidance
            }
        }

        // Competence preference: I'm good at this domain → prefer
        if let Some(&comp) = competencies.get(next_domain) {
            if comp > 0.3 {
                w *= 1.0 + comp * 0.5; // High competence = mild preference
            }
        }

        // Belief alignment: does this edge align or contradict a known belief?
        for belief in beliefs {
            if belief.domain == next_domain && belief.confidence > 0.5 {
                if edge_type == "contradicts" {
                    // Walking toward something that contradicts a strong belief → interesting
                    w *= 1.5;
                } else if edge_type == "reinforces" || edge_type == "similar" {
                    // Walking toward something that confirms a belief → safe
                    w *= 1.2;
                }
            }
        }

        // Working memory resonance: domain currently in conscious awareness → boost
        if wm_domains.contains(next_domain) {
            w *= 1.3;
        }

        w.max(0.001)
    }
}

/// Result of a single walker's traversal
#[derive(Debug, Clone, Serialize)]
pub struct WalkerResult {
    pub bias: WalkerBias,
    pub path: Vec<i32>,
    pub domains_visited: Vec<String>,
    pub edge_types_used: Vec<String>,
    pub total_weight: f32,
    pub surprises: usize,   // cross-domain transitions
    pub dead_ends: usize,
    pub edges_traversed: Vec<i32>,  // edge IDs for batch strengthening
}

/// Aggregated output of all parallel walkers
#[derive(Debug, Clone, Serialize)]
pub struct WalkOutput {
    pub recommended_action: String,
    pub primary_domain: String,
    pub domain_distribution: std::collections::HashMap<String, usize>,
    pub agreement_score: f32,
    pub novelty_score: f32,
    pub emotional_resonance: f32,
    pub search_query: Option<String>,
    pub expression_seeds: Vec<serde_json::Value>,
    pub novel_connections: usize,
    pub consensus_nodes: Vec<i32>,
    pub divergent_nodes: Vec<i32>,
    pub blind_spots: Vec<i32>,
    pub walker_count: usize,
    pub total_hops: usize,
    pub walk_ms: f64,
    pub total_ms: f64,
    pub hops_per_sec: f64,
}
