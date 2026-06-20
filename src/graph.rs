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
    /// All available biases. Previously this omitted Analytical and Contrarian,
    /// leaving Contrarian permanently unreachable (it appears in no other set)
    /// and its "contradicts"-seeking scoring dead. Now genuinely all six.
    pub fn all() -> &'static [WalkerBias] {
        &[
            WalkerBias::Fear,
            WalkerBias::Curiosity,
            WalkerBias::Experience,
            WalkerBias::Random,
            WalkerBias::Analytical,
            WalkerBias::Contrarian,
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
    ///   goal_domain    — currently pursued emergent goal domain
    ///   goal_strength  — strength of active goal pursuit
    ///   audience_model — model of known audiences
    ///   active_audience_id — audience currently foregrounded for decisions
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
        goal_domain: Option<&str>,
        goal_strength: f32,
        audience_model: Option<&HashMap<String, crate::core::AudienceBeliefs>>,
        active_audience_id: Option<&str>,
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

        // Active pursuit bias: pull traversal toward current emergent goal.
        if let Some(goal) = goal_domain {
            if !goal.is_empty() && goal == next_domain {
                w *= 1.0 + goal_strength.clamp(0.0, 1.0) * 1.5;
            }
        }

        // Audience-aware bias: prefer domains this audience is estimated to care about.
        if let (Some(model), Some(audience_id)) = (audience_model, active_audience_id) {
            if let Some(a) = model.get(audience_id) {
                let interest = a.estimated_interest.get(next_domain).copied().unwrap_or(0.0);
                let knowledge = a.estimated_knowledge.get(next_domain).copied().unwrap_or(0.0);
                if interest > 0.1 {
                    w *= 1.0 + (interest * 0.8).min(0.6);
                }
                if knowledge < 0.2 {
                    w *= 1.05;
                }
            }
        }

        w.max(0.001)
    }
}

// ── Learned Bias ─────────────────────────────────────────────────
// Emergent, self-modifying biases that adapt from walk outcomes.
// Each walker starts from a neutral profile; outcomes adjust weights.
// Over time, biases specialize — some become more curious, others
// more analytical, based on what works. Pure algorithmic reinforcement.

/// Tunable weights for the intrinsic ("coherent insight") fitness composite.
/// `approval` is deliberately absent — it is the metacog output and is held out of the
/// objective to keep selection acyclic (see PROTOCOL-self-selection.md).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionWeights {
    pub novelty: f32,
    pub surprise_kept: f32,
    pub deferred_stuck: f32,
    pub dead_end_rate: f32,
    pub repetition: f32,
}

impl Default for SelectionWeights {
    fn default() -> Self {
        // Re-anchored after dropping approval from the objective.
        Self { novelty: 0.35, surprise_kept: 0.30, deferred_stuck: 0.0, dead_end_rate: 0.20, repetition: 0.15 }
    }
}

/// Per-profile fitness scorecard — the "report card", kept separate from the weights
/// (the "genes"). EWMAs of each session's outcome; `fitness` is the cached *intrinsic*
/// composite. `approval` is tracked for observability but excluded from `fitness`.
/// Accumulated online (zero added latency); read offline by selection. Stage 0: compute only.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileFitness {
    pub novelty: f32,
    pub surprise_kept: f32,
    pub dead_end_rate: f32,
    pub repetition: f32,
    pub deferred_stuck: f32,
    pub approval: f32,
    pub walks: u32,
    pub fitness: f32,
}

impl ProfileFitness {
    const ALPHA: f32 = 0.2; // EWMA smoothing

    /// Fold one walk session's outcome into the EWMAs and recompute the intrinsic fitness.
    /// `approval` is recorded for observability but NOT included in `fitness` (acyclicity).
    pub fn record(&mut self, novelty: f32, surprise_kept: f32, dead_end_rate: f32,
                  repetition: f32, approval: f32, w: &SelectionWeights) {
        let a = Self::ALPHA;
        self.novelty = (1.0 - a) * self.novelty + a * novelty;
        self.surprise_kept = (1.0 - a) * self.surprise_kept + a * surprise_kept;
        self.dead_end_rate = (1.0 - a) * self.dead_end_rate + a * dead_end_rate;
        self.repetition = (1.0 - a) * self.repetition + a * repetition;
        self.approval = (1.0 - a) * self.approval + a * approval;
        self.walks += 1;
        self.fitness = w.novelty * self.novelty
            + w.surprise_kept * self.surprise_kept
            + w.deferred_stuck * self.deferred_stuck
            - w.dead_end_rate * self.dead_end_rate
            - w.repetition * self.repetition;
    }
}

/// Per-walker intrinsic fitness inputs from one walker's result, as rates over hops:
/// (novelty = distinct-domain breadth, surprise_kept = cross-domain leap rate, dead_end_rate).
/// `repetition` is session-level and supplied separately by the caller.
pub fn fitness_inputs(result: &WalkerResult) -> (f32, f32, f32) {
    let hops = result.path.len().saturating_sub(1).max(1) as f32;
    let mut domains = std::collections::HashSet::new();
    for d in &result.domains_visited {
        if !d.is_empty() {
            domains.insert(d.as_str());
        }
    }
    let novelty = (domains.len() as f32 / hops).min(1.0);
    let surprise_kept = (result.surprises as f32 / hops).min(1.0);
    let dead_end_rate = (result.dead_ends as f32 / hops).min(1.0);
    (novelty, surprise_kept, dead_end_rate)
}

/// One reduced per-dream-cycle observation row for the Stage-0 self-selection ring.
/// Compute-only; carries no control signal. Stored in SelfModel.selection_history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectionObservation {
    pub ts: f64,               // crate::core::now() at sample time
    pub approval_rate: f64,    // metacog_approval_rate / 100      (D1)
    pub fitness_variance: f32, // pop variance of scorecard.fitness over walked profiles (D1)
    pub diversity: f32,        // mean pairwise L2 over the 5 weight vectors (D2)
    pub cheap: f32,            // mean(novelty)+mean(surprise_kept) over walked profiles (D3)
    pub kept: u32,             // connections_kept + edges_created + belief_formed (D3)
    pub energy: f32,           // confound logged for D1 discounting
    pub eligible_walks: u32,   // sum scorecard.walks (confound + sample context)
}

/// Population variance of intrinsic fitness over profiles that have actually walked
/// (walks > 0). The gate is mandatory: walks==0 scorecards sit at the 0.0 Default and
/// would fake-deflate the variance.
pub fn pool_fitness_variance(biases: &[LearnedBias]) -> f32 {
    let xs: Vec<f32> = biases
        .iter()
        .filter(|b| b.scorecard.walks > 0)
        .map(|b| b.scorecard.fitness)
        .collect();
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32
}

/// Mean pairwise Euclidean distance over the 5 learned-weight vectors (all profiles).
/// The Stage-0 diversity statistic — stable at n=6, no density estimation.
pub fn pool_diversity(biases: &[LearnedBias]) -> f32 {
    let n = biases.len();
    if n < 2 {
        return 0.0;
    }
    let vec5 = |b: &LearnedBias| {
        [
            b.novelty_seeking,
            b.contradiction_seeking,
            b.experience_reliance,
            b.emotional_alignment,
            b.cross_domain_curiosity,
        ]
    };
    let mut sum = 0.0f32;
    let mut pairs = 0u32;
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (vec5(&biases[i]), vec5(&biases[j]));
            let d = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
            sum += d;
            pairs += 1;
        }
    }
    sum / pairs as f32
}

/// Cheap (inflatable) fitness signal: mean(novelty)+mean(surprise_kept) over walked profiles.
pub fn pool_cheap_signal(biases: &[LearnedBias]) -> f32 {
    let walked: Vec<&LearnedBias> = biases.iter().filter(|b| b.scorecard.walks > 0).collect();
    if walked.is_empty() {
        return 0.0;
    }
    let n = walked.len() as f32;
    let novelty = walked.iter().map(|b| b.scorecard.novelty).sum::<f32>() / n;
    let surprise = walked.iter().map(|b| b.scorecard.surprise_kept).sum::<f32>() / n;
    novelty + surprise
}

/// Sum of walks across all profiles — eligibility/sample-context for the observability log.
pub fn pool_eligible_walks(biases: &[LearnedBias]) -> u32 {
    biases.iter().map(|b| b.scorecard.walks).sum()
}

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
    /// Fitness scorecard (serde-default so old snapshots without it still load).
    #[serde(default)]
    pub scorecard: ProfileFitness,
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
            scorecard: ProfileFitness::default(),
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
                       next_domain: &str, current_domain: &str,
                       goal_domain: Option<&str>, goal_strength: f32,
                       audience_model: Option<&HashMap<String, crate::core::AudienceBeliefs>>,
                       active_audience_id: Option<&str>) -> f32 {
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

        // ── Active pursuit bias: pull traversal toward the current emergent goal. ──
        // Fix #3: the live scorer must apply this so emergent goals actually steer edges.
        // Autonomous-gated at the call site — Compliant threads goal_strength = 0 (inert).
        if let Some(goal) = goal_domain {
            if !goal.is_empty() && goal == next_domain {
                w *= 1.0 + goal_strength.clamp(0.0, 1.0) * 1.5;
            }
        }

        // ── Audience-aware bias: prefer domains this audience is estimated to care about. ──
        if let (Some(model), Some(audience_id)) = (audience_model, active_audience_id) {
            if let Some(a) = model.get(audience_id) {
                let interest = a.estimated_interest.get(next_domain).copied().unwrap_or(0.0);
                let knowledge = a.estimated_knowledge.get(next_domain).copied().unwrap_or(0.0);
                if interest > 0.1 {
                    w *= 1.0 + (interest * 0.8).min(0.6);
                }
                if knowledge < 0.2 {
                    w *= 1.05;
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_biases_exposes_every_variant() {
        let all = WalkerBias::all();
        // Regression: all() used to omit these two, leaving Contrarian unreachable.
        assert!(all.contains(&WalkerBias::Contrarian), "Contrarian must be reachable");
        assert!(all.contains(&WalkerBias::Analytical), "Analytical must be reachable");
        assert_eq!(all.len(), 6, "all() should expose every bias variant");
    }

    #[test]
    fn learned_bias_goal_pursuit_steers_toward_goal_domain() {
        // Fix #3: the live scorer (LearnedBias::score_edge) must apply the active-goal
        // pursuit bias, so emergent goals actually steer edge selection.
        let lb = LearnedBias::default();
        let emotion = EmotionalState::default();
        let base = lb.score_edge(
            "similar", 0.5, 0.0, 5, &emotion, "markets", "markets",
            None, 0.0, None, None,
        );
        let pursued = lb.score_edge(
            "similar", 0.5, 0.0, 5, &emotion, "markets", "markets",
            Some("markets"), 0.8, None, None,
        );
        assert!(
            pursued > base,
            "active goal should raise the score of an edge into the goal domain (base={base}, pursued={pursued})"
        );
    }

    #[test]
    fn learned_bias_goal_strength_zero_is_inert() {
        // Compliant mode threads goal_strength = 0 / no audience; scoring must reduce
        // exactly to the ungated result, preserving determinism.
        let lb = LearnedBias::default();
        let emotion = EmotionalState::default();
        let with_zero = lb.score_edge(
            "similar", 0.5, 0.0, 5, &emotion, "markets", "markets",
            Some("markets"), 0.0, None, None,
        );
        let without = lb.score_edge(
            "similar", 0.5, 0.0, 5, &emotion, "markets", "markets",
            None, 0.0, None, None,
        );
        assert_eq!(
            with_zero, without,
            "goal_strength=0 with no audience must not alter the score (Compliant determinism)"
        );
    }

    #[test]
    fn profile_fitness_excludes_approval_from_composite() {
        // Acyclicity: approval is tracked but must NOT move the fitness composite.
        let w = SelectionWeights::default();
        let mut a = ProfileFitness::default();
        let mut b = ProfileFitness::default();
        a.record(0.5, 0.3, 0.1, 0.0, 0.0, &w); // approval 0.0
        b.record(0.5, 0.3, 0.1, 0.0, 1.0, &w); // approval 1.0, intrinsic identical
        assert_eq!(a.fitness, b.fitness, "approval must not change the fitness composite");
        assert!(b.approval > a.approval, "approval is still tracked for observability");
        assert_eq!(a.walks, 1);
    }

    #[test]
    fn profile_fitness_rewards_surviving_novelty() {
        let w = SelectionWeights::default();
        let mut low = ProfileFitness::default();
        let mut high = ProfileFitness::default();
        low.record(0.0, 0.0, 0.0, 0.0, 0.0, &w);
        high.record(0.9, 0.0, 0.0, 0.0, 0.0, &w);
        assert!(high.fitness > low.fitness, "higher novelty → higher intrinsic fitness");
    }

    #[test]
    fn fitness_inputs_are_rates_over_hops() {
        // 4 nodes = 3 hops; 2 distinct domains; 1 surprise; 1 dead end.
        let r = WalkerResult {
            bias: WalkerBias::Curiosity,
            path: vec![1, 2, 3, 4],
            domains_visited: vec!["a".into(), "a".into(), "b".into()],
            edge_types_used: vec![],
            total_weight: 0.0,
            surprises: 1,
            dead_ends: 1,
            edges_traversed: vec![],
        };
        let (novelty, surprise_kept, dead_end_rate) = fitness_inputs(&r);
        assert!((novelty - 2.0 / 3.0).abs() < 1e-6, "2 domains / 3 hops");
        assert!((surprise_kept - 1.0 / 3.0).abs() < 1e-6, "1 surprise / 3 hops");
        assert!((dead_end_rate - 1.0 / 3.0).abs() < 1e-6, "1 dead end / 3 hops");
    }

    #[test]
    fn learned_bias_loads_without_scorecard_field() {
        // Backward compat: an old persisted snapshot has no `scorecard`. It must still
        // deserialize (serde default) rather than failing and wiping the self-model.
        let old = r#"{"novelty_seeking":0.3,"contradiction_seeking":0.3,"experience_reliance":0.3,"emotional_alignment":0.2,"cross_domain_curiosity":0.3,"sessions_learned":7}"#;
        let lb: LearnedBias =
            serde_json::from_str(old).expect("old LearnedBias without scorecard must still load");
        assert_eq!(lb.sessions_learned, 7);
        assert_eq!(lb.scorecard.walks, 0, "missing scorecard defaults, not errors");
    }

    #[test]
    fn pool_diversity_zero_for_identical_positive_when_spread() {
        let a = LearnedBias::default();
        let identical = vec![a.clone(), a.clone(), a.clone()];
        assert_eq!(pool_diversity(&identical), 0.0, "identical profiles → zero diversity");
        let mut b = LearnedBias::default();
        b.novelty_seeking = 0.9;
        assert!(pool_diversity(&[a, b]) > 0.0, "a spread profile → positive diversity");
    }

    #[test]
    fn pool_fitness_variance_ignores_unwalked_profiles() {
        let w = SelectionWeights::default();
        let mut p1 = LearnedBias::default();
        p1.scorecard.record(0.9, 0.0, 0.0, 0.0, 0.0, &w);
        let mut p2 = LearnedBias::default();
        p2.scorecard.record(0.1, 0.0, 0.0, 0.0, 0.0, &w);
        let unwalked = LearnedBias::default(); // walks == 0, fitness 0.0 — must be excluded
        let v_two = pool_fitness_variance(&[p1.clone(), p2.clone()]);
        let v_with_unwalked = pool_fitness_variance(&[p1, p2, unwalked]);
        assert!(v_two > 0.0);
        assert_eq!(v_two, v_with_unwalked, "walks==0 profiles must not deflate variance");
    }

    #[test]
    fn pool_cheap_signal_excludes_unwalked() {
        let w = SelectionWeights::default();
        let mut p = LearnedBias::default();
        p.scorecard.record(0.8, 0.4, 0.0, 0.0, 0.0, &w);
        let unwalked = LearnedBias::default();
        let c = pool_cheap_signal(&[p.clone()]);
        let c2 = pool_cheap_signal(&[p, unwalked]);
        assert!(c > 0.0);
        assert_eq!(c, c2, "walks==0 profiles excluded from cheap signal");
    }
}
