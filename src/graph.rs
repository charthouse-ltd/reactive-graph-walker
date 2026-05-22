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

// ── Learned Bias ─────────────────────────────────────────────────
// Emergent, self-modifying biases that adapt from walk outcomes.
// Each walker starts from a neutral profile; outcomes adjust weights.
// Over time, biases specialize — some become more curious, others
// more analytical, based on what works. Pure algorithmic reinforcement.

/// A learned bias profile with adjustable weights.
/// Replaces the fixed WalkerBias enum with emergent, self-tuned behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedBias {
    pub novelty_seeking: f32,
    pub contradiction_seeking: f32,
    pub experience_reliance: f32,
    pub emotional_alignment: f32,
    pub cross_domain_curiosity: f32,
    pub sessions_learned: u32,
}

impl Default for LearnedBias {
    fn default() -> Self {
        Self {
            novelty_seeking: 0.3,
            contradiction_seeking: 0.3,
            experience_reliance: 0.3,
            emotional_alignment: 0.2,
            cross_domain_curiosity: 0.3,
            sessions_learned: 0,
        }
    }
}

impl LearnedBias {
    /// Update weights from a walk session's outcomes.
    /// Learning rate decays — early experiences shape more than later ones.
    pub fn update_from_session(&mut self, surprises: u32, dead_ends: u32, novelty: f32, domain_count: usize) {
        let lr = 0.08 / (1.0 + self.sessions_learned as f32 * 0.05);
        self.sessions_learned += 1;

        if surprises > 0 {
            let boost = (surprises as f32 * 0.12).min(0.3);
            self.contradiction_seeking = (self.contradiction_seeking + boost * lr).min(1.0);
            self.cross_domain_curiosity = (self.cross_domain_curiosity + boost * lr * 0.5).min(1.0);
            self.experience_reliance = (self.experience_reliance - 0.02 * lr).max(0.05);
        }
        if dead_ends > 0 {
            self.novelty_seeking = (self.novelty_seeking + 0.05 * lr * dead_ends as f32).min(1.0);
            self.experience_reliance = (self.experience_reliance - 0.03 * lr).max(0.05);
        }
        if novelty > 0.5 {
            self.novelty_seeking = (self.novelty_seeking + 0.04 * lr).min(1.0);
            self.cross_domain_curiosity = (self.cross_domain_curiosity + 0.03 * lr).min(1.0);
        }
        if domain_count > 2 {
            self.cross_domain_curiosity = (self.cross_domain_curiosity + 0.04 * lr * domain_count as f32).min(1.0);
        }

        // Gentle regression toward neutral
        for w in [&mut self.novelty_seeking, &mut self.contradiction_seeking,
                   &mut self.experience_reliance, &mut self.emotional_alignment,
                   &mut self.cross_domain_curiosity] {
            *w = (*w * 0.995 + 0.3 * 0.005).clamp(0.05, 0.95);
        }
    }

    /// Score an edge using learned weights.
    pub fn score_edge(&self, edge_type: &str, edge_weight: f32, emotional_charge: f32,
                       traversal_count: i32, emotion: &EmotionalState,
                       next_domain: &str, current_domain: &str) -> f32 {
        let mut w = edge_weight;

        if (edge_type == "contradicts" || edge_type == "opposes") && self.contradiction_seeking > 0.3 {
            w *= 1.0 + self.contradiction_seeking * 2.0;
        }
        if traversal_count < 2 && self.novelty_seeking > 0.3 {
            w += self.novelty_seeking * 2.0;
        }
        if !next_domain.is_empty() && next_domain != current_domain && self.cross_domain_curiosity > 0.3 {
            w *= 1.0 + self.cross_domain_curiosity;
        }
        if traversal_count > 10 && self.experience_reliance > 0.4 {
            w *= 1.0 + self.experience_reliance * 0.5;
        }
        if emotional_charge * emotion.valence > 0.0 && self.emotional_alignment > 0.3 {
            w *= 1.0 + self.emotional_alignment * emotional_charge.abs() * emotion.arousal;
        }
        if emotion.arousal > 0.5 {
            w *= 1.0 + (emotion.arousal - 0.5);
        }
        w.max(0.01)
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
