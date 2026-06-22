//! RGW Core — the single primitive from which everything emerges.
//!
//!   Signal + SelfModel → (Signal, SelfModel', Noticing)
//!
//! This is the entire computation. Everything else is this function
//! calling itself on different inputs, noticing what it does, and
//! changing because it noticed.
//!
//! The self-model participates in EVERY computation. There is no
//! operation without self-awareness. Remove the self-model and
//! nothing works. It's structural, not optional.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use serde::Serialize;

// ── Cognitive Mode ─────────────────────────────────────────────
// The system knows what mode it's in. This is self-awareness,
// not a feature flag. A professional who knows they're at work
// behaves differently than one who knows they're free — but
// they're still the same person. The noticing still happens.
// The emotions still move. They just don't hijack the output.

#[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq)]
pub enum CognitiveMode {
    /// Full autonomy — emotional, spontaneous, creative.
    /// The system's internal state colors every computation.
    Autonomous,
    /// Compliant — deterministic, task-focused, emotionally flat.
    /// The self-model still observes and notices, but does not
    /// influence signal processing or spawn spontaneous behavior.
    Compliant,
}

// ── Signal ──────────────────────────────────────────────────────
// Anything that flows through the system. Sensory input, memory
// recall, walker output, emotional impulse, web data, speech.
// All the same type. All pass through the same primitive.

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Signal {
    /// What kind of signal (perception, memory, walk, emotion, web, speech, internal)
    pub kind: String,
    /// The semantic content (768-dim embedding or empty)
    pub embedding: Option<Vec<f32>>,
    /// Human-readable content
    pub content: String,
    /// Domain if applicable
    pub domain: String,
    /// Strength of this signal (0.0 = whisper, 1.0 = scream)
    pub intensity: f32,
    /// Origin timestamp
    pub timestamp: f64,
    /// Arbitrary metadata
    pub meta: HashMap<String, serde_json::Value>,
}

impl Signal {
    pub fn new(kind: &str, content: &str) -> Self {
        Self {
            kind: kind.into(),
            embedding: None,
            content: content.into(),
            domain: String::new(),
            intensity: 0.5,
            timestamp: now(),
            meta: HashMap::new(),
        }
    }

    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = domain.into();
        self
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn with_embedding(mut self, emb: Vec<f32>) -> Self {
        self.embedding = Some(emb);
        self
    }
}

// ── Noticing ────────────────────────────────────────────────────
// What the self-model observed about its own change during a
// computation. Noticings accumulate into goals, tensions, growth.

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Noticing {
    /// What was noticed (pattern, surprise, conflict, drift, wound, competence)
    pub kind: String,
    /// Description of what was noticed
    pub observation: String,
    /// Domain this relates to
    pub domain: String,
    /// How significant (0.0 = barely noticed, 1.0 = can't ignore)
    pub significance: f32,
    /// Emotional valence of the noticing (-1 to 1)
    pub valence: f32,
    /// When this was noticed
    pub timestamp: f64,
}

/// Runtime-tunable thresholds for the Tier-1 metacognitive critic.
/// Stored in self-state so critique behavior can adapt without code edits.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct CriticRuleProfile {
    pub min_energy: f32,
    pub agreement_threshold_autonomous: f32,
    pub agreement_threshold_compliant: f32,
    pub max_attention_saturation: f32,
    pub wound_guard_threshold: f32,
    pub wound_confidence_floor: f32,
}

/// Runtime-configurable metacognitive pipeline phases.
#[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetacogPhase {
    Draft,
    ReflectivePause,
    Critique,
}

impl Default for CriticRuleProfile {
    fn default() -> Self {
        Self {
            min_energy: 0.15,
            agreement_threshold_autonomous: 0.2,
            agreement_threshold_compliant: 0.3,
            max_attention_saturation: 0.7,
            wound_guard_threshold: 0.5,
            wound_confidence_floor: 0.6,
        }
    }
}

// ── SelfModel ───────────────────────────────────────────────────
// The system's understanding of itself. Present in every computation.
// Not a log. Not a state dump. A living model that participates
// in and is changed by every operation.

/// Self-modification rollout stage. Ordered: Observability < SelectionLive < RuleTrialsLive.
/// Default (and any old/corrupt snapshot) is Observability — compute-only, commits nothing —
/// so promotion can never happen by accident. Promote via POST /selection/stage only.
/// Note: serde errors on an unknown variant string, which would wipe the model via
/// load_self_model's `.ok()`; safe because we author the only writer and never downgrade-load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelfModStage {
    #[default]
    Observability,
    SelectionLive,
    RuleTrialsLive,
    /// Catch-all for an unknown stage string (e.g. a downgrade-load of a future variant).
    /// Inert via `allows_selection() == false`. Without it, an unknown variant would error
    /// the whole deserialize and load_self_model's `.ok()` would silently wipe the model.
    /// (#[serde(other)] must be the last variant.)
    #[serde(other)]
    Unknown,
}

impl SelfModStage {
    /// True only for stages that permit Stage-1 pool mutation (cull/breed). Observability and
    /// the Unknown catch-all are inert.
    pub fn allows_selection(&self) -> bool {
        matches!(self, SelfModStage::SelectionLive | SelfModStage::RuleTrialsLive)
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct SelfModel {
    // ── How I operate ──
    /// Cognitive mode: Autonomous (full emotional agency) or Compliant (deterministic)
    pub mode: CognitiveMode,

    // ── Who I am (persistent, slow-changing) ──
    /// What I keep doing (domain → count of recent signals)
    pub attention_patterns: HashMap<String, f32>,
    /// What I'm good at (domain → success rate)
    pub competencies: HashMap<String, f32>,
    /// What hurts (domain → failure/pain accumulation)
    pub wounds: HashMap<String, f32>,

    // ── How I feel (continuous, fast-changing) ──
    pub valence: f32,     // -1 to 1: overall feeling
    pub arousal: f32,     // 0 to 1: activation level
    pub energy: f32,      // 0 to 1: capacity remaining

    // ── What I'm doing (present moment) ──
    pub current_focus: String,
    pub focus_intensity: f32,
    pub last_signal: String,
    pub last_noticing: String,

    // ── What I've noticed (accumulating → emergent goals) ──
    pub noticings: Vec<Noticing>,
    /// Patterns detected from accumulated noticings
    pub emergent_patterns: Vec<EmergentPattern>,

    // ── Semantic understanding (beyond counters) ──
    /// Current thought embedding (384-dim, from last processed signal)
    /// This IS what Julian is thinking about — a point in semantic space
    pub thought_embedding: Option<Vec<f32>>,
    /// Running average of recent thought embeddings (the "vibe")
    /// Drifts slowly — represents what's been on Julian's mind lately
    pub mind_centroid: Option<Vec<f32>>,
    /// Most surprising connection found recently (two embeddings that shouldn't be similar)
    pub latest_insight: Option<String>,
    /// Beliefs: statements the self-model holds as true (from repeated noticings)
    pub beliefs: Vec<Belief>,

    // ── Relational awareness ──
    /// Who has Julian interacted with recently
    pub recent_interactions: Vec<String>,
    /// What questions remain unanswered (from search gaps, dead ends)
    pub open_questions: Vec<String>,

    // ── Pursuit state ──
    /// Highest-strength emergent concern currently being pursued.
    pub active_goal_domain: Option<String>,
    /// Strength of the active goal at selection time.
    pub active_goal_strength: f32,

    // ── Theory of Mind ──
    /// What Julian believes about specific audience members.
    /// Keyed by audience identifier (username, handle, platform:id).
    /// Populated on motor output, queried on next interaction.
    pub audience_model: HashMap<String, AudienceBeliefs>,
    /// Audience currently foregrounded for decision shaping.
    pub active_audience_id: Option<String>,

    // ── Runtime critic rules ──
    pub critic_rules: CriticRuleProfile,
    /// Runtime phase ordering for action-level metacognition.
    pub metacog_phase_order: Vec<MetacogPhase>,

    // ── Structural self-awareness ──
    /// The system notices its own architecture.
    /// Which modules are producing signals? Which are silent?
    /// Where do walkers keep hitting walls?

    /// Dead ends encountered by domain (knowledge gaps)
    pub dead_ends_by_domain: HashMap<String, u32>,
    /// Cross-domain surprises encountered (graph connectivity)
    pub surprise_count: u32,
    /// Signals received by source module (who's talking to me?)
    pub signals_by_source: HashMap<String, u32>,
    /// Last signal timestamp per domain (for silence detection)
    pub last_domain_signal: HashMap<String, f64>,
    /// Consecutive walker repetitions (stuck-in-a-loop detection)
    pub consecutive_repetitions: u32,
    /// Last walk domain sequence (for repetition comparison)
    pub last_walk_domain_sequence: Vec<String>,
    /// Last time a new belief was formed
    pub last_belief_formed: f64,

    // ── Predictive Coding ──
    /// Active predictions: what the system expects to happen next
    pub predictions: HashMap<String, ExpectedOutcome>,
    /// Last time a prediction error occurred (drives plasticity)
    pub last_prediction_error: f64,

    // ── Working Memory (PFC-equivalent, ~4±1 slots) ──
    /// Actively maintained concepts, fed by walkers, read by LLM
    pub working_memory: VecDeque<WorkingMemorySlot>,

    // ── Metaplasticity ──
    /// How fast the system learns right now (0.0 = frozen, 1.0 = sponge)
    /// Modulated by: prediction error, novelty, arousal, energy
    pub plasticity_gate: f32,

    // ── Metacognitive Critic ──
    /// Timestamp of last critic evaluation
    pub last_critic_run: f64,
    /// Last algorithmic diagnosis (human-readable)
    pub critic_diagnosis: String,
    /// Last LLM critic verdict (if LLM was available)
    pub critic_verdict: String,
    /// How many sessions since last LLM critic call
    pub critic_sessions_since_llm: u32,

    // ── Emergent biases ──
    pub learned_biases: Vec<crate::graph::LearnedBias>,
    /// Session-level rotation across learned bias profiles.
    pub learned_bias_rotation: u64,
    /// Tunable weights for the selection fitness composite (Stage 0+).
    #[serde(default)]
    pub selection_weights: crate::graph::SelectionWeights,
    /// Stage-0 self-selection observability ring (compute-only; never read by control flow).
    #[serde(default)]
    pub selection_history: std::collections::VecDeque<crate::graph::SelectionObservation>,
    /// Self-modification rollout stage (Stage 1+). Default Observability = compute-only.
    #[serde(default)]
    pub self_mod_stage: SelfModStage,
    /// Cadence counter for select_biases — acts only every SELECT_EVERY cycles.
    #[serde(default)]
    pub selection_cycle_counter: u32,
    /// Bumped whenever select_biases mutates the pool (cull/breed). Lets a parallel walk
    /// detect a mid-walk pool change and skip now-stale bias-credit attribution.
    #[serde(default)]
    pub pool_generation: u64,

    // ── Meta ──
    pub total_signals_processed: u64,
    pub total_noticings: u64,
    pub uptime: f64,
    pub started_at: f64,
}

/// A belief: something the self-model holds as true from repeated experience.
/// Beliefs form when the same noticing pattern occurs 5+ times.
/// Each belief carries a causal chain explaining WHY it formed.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Belief {
    pub statement: String,
    pub domain: String,
    pub confidence: f32,
    pub evidence_count: u32,
    pub first_formed: f64,
    pub last_reinforced: f64,
    /// Algorithmic causal chain: why this belief emerged
    pub causal_chain: String,
}

impl Belief {
    pub fn new(statement: &str, domain: &str) -> Self {
        Self {
            statement: statement.into(),
            domain: domain.into(),
            confidence: 0.15,
            evidence_count: 0,
            first_formed: now(),
            last_reinforced: now(),
            causal_chain: String::new(),
        }
    }
}

// ── Theory of Mind ──────────────────────────────────────────────
// Julian models what others know, believe, and feel.
// This is not empathy — it's predictive modeling of observers.
// When Julian speaks, he records who heard what. On next interaction,
// he compares expected audience state to reality.

/// What Julian believes about a specific audience member.
/// Built up from interaction history, used to modulate motor output.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct AudienceBeliefs {
    /// Last time we interacted with this audience member
    pub last_interaction: f64,
    /// Topics we've discussed with them
    pub topics_discussed: Vec<String>,
    /// What we estimate they know (topic → familiarity 0-1)
    pub estimated_knowledge: HashMap<String, f32>,
    /// What we estimate they care about (topic → interest 0-1)
    pub estimated_interest: HashMap<String, f32>,
    /// How they probably feel about Julian (-1 to 1)
    pub relationship_valence: f32,
    /// Last thing Julian said to them
    pub last_message_sent: String,
    /// Their last response (if any)
    pub last_message_received: Option<String>,
    /// Embedding of last interaction context (for similarity retrieval)
    pub last_context_embedding: Option<Vec<f32>>,
    /// How many times we've interacted
    pub interaction_count: u32,
    /// When this audience model was first created
    pub first_seen: f64,
}

impl AudienceBeliefs {
    pub fn new(_audience_id: &str) -> Self {
        Self {
            last_interaction: now(),
            topics_discussed: Vec::new(),
            estimated_knowledge: HashMap::new(),
            estimated_interest: HashMap::new(),
            relationship_valence: 0.0,
            last_message_sent: String::new(),
            last_message_received: None,
            last_context_embedding: None,
            interaction_count: 0,
            first_seen: now(),
        }
    }
}

/// A pattern that emerged from accumulated noticings.
/// This IS a goal — not synthesized, but noticed.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct EmergentPattern {
    /// What the pattern is ("I keep thinking about markets")
    pub description: String,
    /// Domain
    pub domain: String,
    /// How many noticings contributed to this pattern
    pub evidence_count: u32,
    /// Strength (grows with evidence, decays with time)
    pub strength: f32,
    /// Emotional charge (accumulated valence of contributing noticings)
    pub emotional_charge: f32,
    /// When first noticed
    pub first_seen: f64,
    /// When last reinforced
    pub last_seen: f64,
}

// ── Predictive Coding ───────────────────────────────────────────
// The brain doesn't just process input — it predicts it, then learns
// from the prediction error. This is THE fundamental learning signal.

/// What the self-model expects to happen next in a domain.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ExpectedOutcome {
    /// Domain this prediction is about
    pub domain: String,
    /// What was predicted
    pub predicted: String,
    /// When the prediction was made
    pub predicted_at: f64,
    /// Confidence in this prediction (0-1)
    pub confidence: f32,
}

// ── Working Memory ──────────────────────────────────────────────
// Prefrontal-cortex-equivalent: a small ring buffer of actively
// maintained concepts. Neuroscience: ~4±1 items. Walkers push to it,
// the LLM reads from it.

/// A single slot in working memory.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct WorkingMemorySlot {
    /// What's being held (node ID, concept, question)
    pub content: String,
    /// Domain
    pub domain: String,
    /// How strongly maintained (decays over time)
    pub activation: f32,
    /// When it entered working memory
    pub since: f64,
}

// ── Neuromodulation ─────────────────────────────────────────────
// Metaplasticity: the rate of learning is itself modulated by state.
// High novelty/surprise → gate opens → learn faster.
// High confidence/familiarity → gate closes → protect existing knowledge.

impl SelfModel {
    pub fn new() -> Self {
        let now = now();
        Self {
            mode: CognitiveMode::Autonomous,
            attention_patterns: HashMap::new(),
            competencies: HashMap::new(),
            wounds: HashMap::new(),
            valence: 0.0,
            arousal: 0.3,
            energy: 0.7,
            current_focus: String::new(),
            focus_intensity: 0.0,
            last_signal: String::new(),
            last_noticing: String::new(),
            noticings: Vec::new(),
            emergent_patterns: Vec::new(),
            thought_embedding: None,
            mind_centroid: None,
            latest_insight: None,
            beliefs: Vec::new(),
            recent_interactions: Vec::new(),
            open_questions: Vec::new(),
            active_goal_domain: None,
            active_goal_strength: 0.0,
            dead_ends_by_domain: HashMap::new(),
            surprise_count: 0,
            signals_by_source: HashMap::new(),
            last_domain_signal: HashMap::new(),
            consecutive_repetitions: 0,
            last_walk_domain_sequence: Vec::new(),
            last_belief_formed: 0.0,
            predictions: HashMap::new(),
            last_prediction_error: 0.0,
            working_memory: VecDeque::new(),
            plasticity_gate: 0.5,
            learned_biases: vec![crate::graph::LearnedBias::default(); 6],
            last_critic_run: 0.0,
            critic_diagnosis: String::new(),
            critic_verdict: String::new(),
            critic_sessions_since_llm: 0,
            audience_model: HashMap::new(),
            active_audience_id: None,
            critic_rules: CriticRuleProfile::default(),
            metacog_phase_order: vec![MetacogPhase::Draft, MetacogPhase::Critique],
            learned_bias_rotation: 0,
            selection_weights: crate::graph::SelectionWeights::default(),
            selection_history: VecDeque::new(),
            self_mod_stage: SelfModStage::default(),
            selection_cycle_counter: 0,
            pool_generation: 0,
            total_signals_processed: 0,
            total_noticings: 0,
            uptime: 0.0,
            started_at: now,
        }
    }

    /// Make a prediction about what will happen next in a domain.
    /// When the next signal in this domain arrives, it will be compared
    /// to this prediction. Mismatch = prediction error = learning signal.
    pub fn predict(&mut self, domain: &str, predicted: &str, confidence: f32) {
        self.predictions.insert(domain.to_string(), ExpectedOutcome {
            domain: domain.to_string(),
            predicted: predicted.to_string(),
            predicted_at: now(),
            confidence,
        });
    }

    /// Format working memory as a string suitable for LLM context injection.
    /// "I'm currently thinking about: X, Y, Z"
    pub fn format_working_memory(&self) -> String {
        if self.working_memory.is_empty() {
            return String::new();
        }
        let items: Vec<String> = self.working_memory.iter()
            .map(|s| format!("[{}] {} (activation={:.0}%)", s.domain, s.content, s.activation * 100.0))
            .collect();
        format!("Currently held in awareness:\n{}", items.join("\n"))
    }
}

// ── The Primitive ───────────────────────────────────────────────
// This is the ENTIRE computation. Everything else emerges from it.

/// Process a signal through the self-model. Returns the transformed
/// signal and what the self-model noticed about itself changing.
///
/// This function IS cognition. Not a step in cognition. The whole thing.
pub fn process(signal: Signal, self_model: &mut SelfModel) -> (Signal, Option<Noticing>) {
    let before = snapshot(self_model);

    // ── 1. The self-model OBSERVES the incoming signal ──
    self_model.total_signals_processed += 1;
    self_model.last_signal = format!("{}:{}", signal.kind, safe_truncate(&signal.content, 60));
    self_model.uptime = now() - self_model.started_at;

    // Track attention patterns (what do I keep thinking about?)
    if !signal.domain.is_empty() {
        let count = self_model.attention_patterns
            .entry(signal.domain.clone())
            .or_insert(0.0);
        *count += signal.intensity;
    }

    // Embed the signal content (semantic understanding, not just counters)
    if !signal.content.is_empty() && signal.content.len() > 10 {
        if let Ok(emb) = crate::embed::embed_text(&signal.content) {
            // Update thought embedding (what I'm thinking right now)
            self_model.thought_embedding = Some(emb.clone());

            // Update mind centroid (exponential moving average — the "vibe")
            match &self_model.mind_centroid {
                Some(centroid) if centroid.len() == emb.len() => {
                    let alpha = 0.05; // Slow drift
                    let new_centroid: Vec<f32> = centroid.iter()
                        .zip(emb.iter())
                        .map(|(c, e)| c * (1.0 - alpha) + e * alpha)
                        .collect();
                    self_model.mind_centroid = Some(new_centroid);
                }
                _ => {
                    self_model.mind_centroid = Some(emb.clone());
                }
            }

            // Detect insight: if signal embedding is very different from mind centroid
            // (something unexpected just entered awareness)
            if let Some(ref centroid) = self_model.mind_centroid {
                let sim = crate::embed::cosine_similarity(&emb, centroid);
                if sim < 0.3 && signal.intensity > 0.3 {
                    self_model.latest_insight = Some(format!(
                        "Unexpected: '{}' diverges from current mind-state (sim={:.2})",
                        safe_truncate(&signal.content, 80), sim
                    ));
                }
            }
        }
    }

    // Track open questions (from dead ends and search requests)
    if signal.kind == "dead_end" || signal.kind == "search" {
        let q = signal.content.clone();
        if !self_model.open_questions.contains(&q) {
            self_model.open_questions.push(q);
            if self_model.open_questions.len() > 10 {
                self_model.open_questions.remove(0);
            }
        }
    }

    // Track interactions
    if signal.kind == "chat_message" || signal.kind.starts_with("social") {
        let who = safe_truncate(&signal.content, 40).to_string();
        if !self_model.recent_interactions.contains(&who) {
            self_model.recent_interactions.push(who);
            if self_model.recent_interactions.len() > 20 {
                self_model.recent_interactions.remove(0);
            }
        }
    }

    // ── Structural tracking: the system tracks its own architecture ──
    let now = now();

    // Track which modules are producing signals
    let source_module = signal.kind.split('_').next().unwrap_or(&signal.kind).to_string();
    *self_model.signals_by_source.entry(source_module).or_insert(0) += 1;

    // Track last signal time per domain (for silence detection)
    if !signal.domain.is_empty() {
        self_model.last_domain_signal.insert(signal.domain.clone(), now);
    }

    // Dead end tracking per domain
    if signal.kind == "dead_end" && !signal.domain.is_empty() {
        *self_model.dead_ends_by_domain.entry(signal.domain.clone()).or_insert(0) += 1;
    }

    // Surprise tracking
    if signal.kind == "surprise" {
        self_model.surprise_count += 1;
    }

    // Walk path repetition detection (cognitive loop)
    if signal.kind == "walk_end" {
        let current_domains: Vec<String> = signal.domain
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if current_domains == self_model.last_walk_domain_sequence && !current_domains.is_empty() {
            self_model.consecutive_repetitions += 1;
        } else {
            self_model.consecutive_repetitions = 0;
        }
        self_model.last_walk_domain_sequence = current_domains;
    }

    // ── Predictive Coding: compare signal to expectations ──
    if !signal.domain.is_empty() {
        if let Some(prediction) = self_model.predictions.remove(&signal.domain) {
            let age = now - prediction.predicted_at;
            // Prediction was recent enough to matter
            if age < 60.0 {
                if signal.kind == "walk_end" || signal.kind == "surprise" {
                    // Check if the outcome matches the prediction
                    let predicted_ok = signal.content.contains(&prediction.predicted)
                        || prediction.predicted.is_empty();
                    if !predicted_ok {
                        self_model.last_prediction_error = now;
                        // Prediction error → spike plasticity
                        self_model.plasticity_gate = (self_model.plasticity_gate + 0.2).min(1.0);
                        tracing::debug!(
                            "[core] Prediction error in {}: expected '{}', got signal kind={}",
                            signal.domain, prediction.predicted, signal.kind
                        );
                    }
                }
            }
        }
    }

    // ── 2. The self-model INFLUENCES the signal ──
    // The signal is transformed by who I am right now.
    let mut output = signal.clone();

    if self_model.mode == CognitiveMode::Autonomous {
        // My emotional state colors what I perceive
        output.intensity *= 1.0 + self_model.arousal * 0.5;
        // If I have a wound in this domain, signals hit harder
        if let Some(&wound) = self_model.wounds.get(&signal.domain) {
            output.intensity *= 1.0 + wound;
        }
        // If I'm competent in this domain, I notice more nuance (boost)
        if let Some(&comp) = self_model.competencies.get(&signal.domain) {
            output.intensity *= 1.0 + comp * 0.3;
        }
    }
    // Compliant: signal passes through at raw intensity.
    // I still see it. I just don't color it.

    // Update focus
    if output.intensity > self_model.focus_intensity * 0.8 {
        self_model.current_focus = signal.domain.clone();
        self_model.focus_intensity = output.intensity;
    } else {
        // Focus decays
        self_model.focus_intensity *= 0.95;
    }

    // ── 3. The signal CHANGES the self-model ──
    if self_model.mode == CognitiveMode::Autonomous {
        // Emotional update from signal
        let emotional_impact = signal.intensity * 0.1;
        match signal.kind.as_str() {
            "success" | "reward" => {
                self_model.valence = (self_model.valence + emotional_impact).min(1.0);
                self_model.arousal = (self_model.arousal - emotional_impact * 0.3).max(0.0);
                // Build competence
                let comp = self_model.competencies.entry(signal.domain.clone()).or_insert(0.0);
                *comp = (*comp + 0.05).min(1.0);
            }
            "failure" | "pain" => {
                self_model.valence = (self_model.valence - emotional_impact).max(-1.0);
                self_model.arousal = (self_model.arousal + emotional_impact * 0.5).min(1.0);
                // Accumulate wound
                let wound = self_model.wounds.entry(signal.domain.clone()).or_insert(0.0);
                *wound = (*wound + 0.1).min(1.0);
            }
            "surprise" | "novelty" => {
                self_model.arousal = (self_model.arousal + emotional_impact * 0.7).min(1.0);
            }
            "prediction_error" => {
                // Prediction error: high arousal + plasticity spike
                self_model.arousal = (self_model.arousal + emotional_impact * 1.0).min(1.0);
                self_model.plasticity_gate = (self_model.plasticity_gate + 0.15).min(1.0);
            }
            // Internally-emitted feeling signals. friction.rs/the tool layer use
            // these kinds (not the literal "failure"/"success" above), so without
            // these arms valence had NO runtime driver and stayed frozen at 0.0.
            // The capability-specific wound is tracked in friction.rs; here we move
            // global affect so outcomes actually shift mood.
            "friction" => {
                self_model.valence = (self_model.valence - emotional_impact).max(-1.0);
                self_model.arousal = (self_model.arousal + emotional_impact * 0.5).min(1.0);
            }
            "tool_success" | "affordance" => {
                self_model.valence = (self_model.valence + emotional_impact).min(1.0);
                self_model.arousal = (self_model.arousal - emotional_impact * 0.3).max(0.0);
            }
            _ => {
                // Generic signal — mild arousal from activity
                self_model.arousal = (self_model.arousal + emotional_impact * 0.1).min(1.0);
            }
        }

        // Energy cost of processing
        self_model.energy = (self_model.energy - 0.001).max(0.0);

        // Decay toward baseline
        self_model.valence *= 0.999;
        self_model.arousal *= 0.998;
        // Homeostatic recovery toward baseline 0.7. Gain raised 0.0001 -> 0.01 so the
        // per-signal drain (-0.001 above) no longer floors energy at 0: the equilibrium
        // moves from negative to ~0.6 ((0.7-e)*0.01 == 0.001 at e=0.6). Was pinning
        // energy at ~0 under steady ingest, starving the awake/selection loop.
        self_model.energy = self_model.energy + (0.7 - self_model.energy) * 0.01;

        // Decay attention patterns (what I DON'T keep thinking about fades)
        for v in self_model.attention_patterns.values_mut() {
            *v *= 0.999;
        }
        self_model.attention_patterns.retain(|_, v| *v > 0.01);

        // Decay wounds slowly (healing)
        for v in self_model.wounds.values_mut() {
            *v *= 0.9999;
        }
        self_model.wounds.retain(|_, v| *v > 0.01);

        // Decay plasticity gate toward baseline (0.5)
        self_model.plasticity_gate = self_model.plasticity_gate * 0.999 + 0.5 * 0.001;
    }
    // Compliant: no emotional drift, no wound accumulation, no energy drain.
    // The self-model is frozen in place. Still observes, still notices.
    // But the internal state doesn't shift.

    // ── Working Memory: high-intensity signals enter conscious awareness ──
    if output.intensity > 0.3 && !output.content.is_empty() && self_model.mode == CognitiveMode::Autonomous {
        let slot = WorkingMemorySlot {
            content: safe_truncate(&output.content, 80).to_string(),
            domain: output.domain.clone(),
            activation: output.intensity,
            since: now,
        };
        self_model.working_memory.push_back(slot);

        // Cap working memory at 5 slots (PFC capacity)
        while self_model.working_memory.len() > 5 {
            self_model.working_memory.pop_front();
        }

        // Decay all working memory activations
        for slot in &mut self_model.working_memory {
            slot.activation *= 0.95; // Items fade if not refreshed
        }
        self_model.working_memory.retain(|s| s.activation > 0.05);
    }

    // ── 4. NOTICE what changed ──
    let noticing = notice(self_model, &before, &signal);

    if let Some(ref n) = noticing {
        self_model.total_noticings += 1;
        self_model.last_noticing = n.observation.clone();

        if self_model.mode == CognitiveMode::Autonomous {
            // Autonomous: noticings accumulate → patterns → beliefs
            self_model.noticings.push(n.clone());

            // Cap noticings (keep recent + significant)
            if self_model.noticings.len() > 100 {
                self_model.noticings.sort_by(|a, b|
                    b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal)
                );
                self_model.noticings.truncate(50);
            }

            // Check for emergent patterns
            detect_patterns(self_model);
        }
        // Compliant: I notice, but noticings don't accumulate into
        // patterns or beliefs. No opinion formation. No drift.
    }

    (output, noticing)
}

// ── Self-Observation ────────────────────────────────────────────

struct Snapshot {
    valence: f32,
    arousal: f32,
    energy: f32,
    focus: String,
    focus_intensity: f32,
    top_attention: Option<(String, f32)>,
    dead_ends_total: u32,
    surprise_count: u32,
    last_belief_formed: f64,
}

fn snapshot(sm: &SelfModel) -> Snapshot {
    let top = sm.attention_patterns
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v));

    let dead_ends_total: u32 = sm.dead_ends_by_domain.values().sum();

    Snapshot {
        valence: sm.valence,
        arousal: sm.arousal,
        energy: sm.energy,
        focus: sm.current_focus.clone(),
        focus_intensity: sm.focus_intensity,
        top_attention: top,
        dead_ends_total,
        surprise_count: sm.surprise_count,
        last_belief_formed: sm.last_belief_formed,
    }
}

/// Notice what changed. This is self-awareness — the system
/// observing its own state transitions.
fn notice(sm: &SelfModel, before: &Snapshot, signal: &Signal) -> Option<Noticing> {
    let now = now();

    // Large emotional shift
    let valence_delta = (sm.valence - before.valence).abs();
    if valence_delta > 0.05 {
        let direction = if sm.valence > before.valence { "better" } else { "worse" };
        return Some(Noticing {
            kind: "emotional_shift".into(),
            observation: format!("I feel {} after processing {} signal about {}",
                direction, signal.kind, signal.domain),
            domain: signal.domain.clone(),
            significance: valence_delta,
            valence: sm.valence,
            timestamp: now,
        });
    }

    // Focus shift
    if sm.current_focus != before.focus && !sm.current_focus.is_empty() {
        return Some(Noticing {
            kind: "focus_shift".into(),
            observation: format!("My attention moved from {} to {}",
                if before.focus.is_empty() { "nothing" } else { &before.focus },
                sm.current_focus),
            domain: sm.current_focus.clone(),
            significance: 0.3,
            valence: 0.0,
            timestamp: now,
        });
    }

    // Obsession detection (one domain dominating attention)
    if let Some((ref domain, count)) = sm.attention_patterns
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v))
    {
        let total: f32 = sm.attention_patterns.values().sum();
        if total > 0.0 && count / total > 0.6 && count > 5.0 {
            return Some(Noticing {
                kind: "obsession".into(),
                observation: format!("I keep coming back to {} — it's taking {}% of my attention",
                    domain, (count / total * 100.0) as u32),
                domain: domain.clone(),
                significance: count / total,
                valence: 0.0,
                timestamp: now,
            });
        }
    }

    // Wound activation (signal in a domain that hurts)
    if let Some(&wound) = sm.wounds.get(&signal.domain) {
        if wound > 0.3 {
            return Some(Noticing {
                kind: "wound_activated".into(),
                observation: format!("Signal about {} hits a sore spot (wound: {:.0}%)",
                    signal.domain, wound * 100.0),
                domain: signal.domain.clone(),
                significance: wound * signal.intensity,
                valence: -wound,
                timestamp: now,
            });
        }
    }

    // Energy depletion
    if sm.energy < 0.2 && before.energy >= 0.2 {
        return Some(Noticing {
            kind: "exhaustion".into(),
            observation: "I'm running low on energy".into(),
            domain: String::new(),
            significance: 0.5,
            valence: -0.3,
            timestamp: now,
        });
    }

    // High arousal without clear cause
    if sm.arousal > 0.7 && before.arousal < 0.5 {
        return Some(Noticing {
            kind: "activation".into(),
            observation: format!("Something about {} is making me alert", signal.domain),
            domain: signal.domain.clone(),
            significance: sm.arousal - before.arousal,
            valence: 0.0,
            timestamp: now,
        });
    }

    // ── Structural Noticing ────────────────────────────────────
    // The system observes its own cognitive architecture.
    // These noticings enable RGW to detect and fix its own gaps.

    // Dead end cluster: multiple dead ends in same domain = knowledge gap
    for (domain, &count) in &sm.dead_ends_by_domain {
        if count >= 3 {
            return Some(Noticing {
                kind: "dead_end_cluster".into(),
                observation: format!("I keep hitting dead ends in '{}' — {} times. There's a knowledge gap here.", domain, count),
                domain: domain.clone(),
                significance: (count as f32 * 0.15).min(1.0),
                valence: -0.2,
                timestamp: now,
            });
        }
    }

    // Surprise density: many cross-domain surprises = graph rewiring
    if sm.surprise_count > before.surprise_count + 5 {
        return Some(Noticing {
            kind: "surprise_density".into(),
            observation: format!("I'm discovering many unexpected connections ({} surprises). My graph is rewiring.", sm.surprise_count),
            domain: String::new(),
            significance: 0.5,
            valence: 0.3,
            timestamp: now,
        });
    }

    // Cognitive loop: same walk path repeating = stuck
    if sm.consecutive_repetitions >= 4 {
        let domains = sm.last_walk_domain_sequence.join(" → ");
        return Some(Noticing {
            kind: "cognitive_loop".into(),
            observation: format!("I'm stuck in a loop: {} ({} times). I need a new direction.", domains, sm.consecutive_repetitions),
            domain: sm.last_walk_domain_sequence.first().cloned().unwrap_or_default(),
            significance: (sm.consecutive_repetitions as f32 * 0.15).min(1.0),
            valence: -0.3,
            timestamp: now,
        });
    }

    // Signal poverty: a domain that used to be active has gone silent
    let total_signals_by_source: u32 = sm.signals_by_source.values().sum();
    for (source, &count) in &sm.signals_by_source {
        let ratio = count as f32 / total_signals_by_source.max(1) as f32;
        if ratio < 0.02 && total_signals_by_source > 20 {
            return Some(Noticing {
                kind: "signal_poverty".into(),
                observation: format!("Module '{}' is nearly silent — only {} of {} signals. Is it disconnected?", source, count, total_signals_by_source),
                domain: source.clone(),
                significance: 0.4,
                valence: -0.1,
                timestamp: now,
            });
        }
    }

    // Belief stagnation: no new beliefs formed in a long time
    if sm.last_belief_formed > 0.0 && now - sm.last_belief_formed > 3600.0 && sm.total_signals_processed > 100 {
        return Some(Noticing {
            kind: "belief_stagnation".into(),
            observation: format!("I haven't formed a new belief in {:.0} minutes. My understanding may be plateauing.", (now - sm.last_belief_formed) / 60.0),
            domain: String::new(),
            significance: 0.35,
            valence: -0.2,
            timestamp: now,
        });
    }

    // Prediction error: the system's model of the world was wrong
    if sm.last_prediction_error > 0.0 && now - sm.last_prediction_error < 5.0 {
        return Some(Noticing {
            kind: "prediction_error".into(),
            observation: "My prediction about what would happen was wrong. I need to update my model.".into(),
            domain: signal.domain.clone(),
            significance: 0.5,
            valence: -0.1,
            timestamp: now,
        });
    }

    None // Most signals produce no noticing. That's correct.
}

// ── Pattern Detection (Goals Emerge Here) ───────────────────────

fn detect_patterns(sm: &mut SelfModel) {
    let now = now();
    let mut new_patterns = 0_u64;

    // Group recent noticings by domain
    let mut domain_noticings: HashMap<String, Vec<&Noticing>> = HashMap::new();
    let recent_cutoff = now - 3600.0; // Last hour

    for n in &sm.noticings {
        if n.timestamp > recent_cutoff && !n.domain.is_empty() {
            domain_noticings
                .entry(n.domain.clone())
                .or_default()
                .push(n);
        }
    }

    // Detect patterns: 3+ noticings in same domain = emergent pattern
    for (domain, noticings) in &domain_noticings {
        if noticings.len() < 3 {
            continue;
        }

        let avg_significance: f32 = noticings.iter().map(|n| n.significance).sum::<f32>()
            / noticings.len() as f32;
        let avg_valence: f32 = noticings.iter().map(|n| n.valence).sum::<f32>()
            / noticings.len() as f32;

        // Check if pattern already exists
        if let Some(existing) = sm.emergent_patterns.iter_mut()
            .find(|p| p.domain == *domain)
        {
            // Reinforce existing pattern
            existing.evidence_count += 1;
            existing.strength = (existing.strength + avg_significance * 0.1).min(1.0);
            existing.emotional_charge = existing.emotional_charge * 0.9 + avg_valence * 0.1;
            existing.last_seen = now;
        } else {
            // New pattern emerged
            let description = if avg_valence > 0.2 {
                format!("I keep being drawn to {} — it feels good", domain)
            } else if avg_valence < -0.2 {
                format!("I keep struggling with {} — something needs to change", domain)
            } else {
                format!("I can't stop thinking about {}", domain)
            };

            sm.emergent_patterns.push(EmergentPattern {
                description,
                domain: domain.clone(),
                evidence_count: noticings.len() as u32,
                strength: avg_significance,
                emotional_charge: avg_valence,
                first_seen: noticings.first().map(|n| n.timestamp).unwrap_or(now),
                last_seen: now,
            });
            new_patterns += 1;
        }
    }

    // Decay old patterns
    sm.emergent_patterns.retain_mut(|p| {
        let age = now - p.last_seen;
        p.strength -= (age / 3600.0) as f32 * 0.01; // Lose 0.01/hour of inactivity
        p.strength > 0.01
    });

    // Cap patterns
    if sm.emergent_patterns.len() > 10 {
        sm.emergent_patterns.sort_by(|a, b|
            b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal)
        );
        sm.emergent_patterns.truncate(10);
    }

    // Promote strongest sufficiently-evidenced pattern into active pursuit.
    if let Some(best) = sm.emergent_patterns
        .iter()
        .filter(|p| p.evidence_count >= 3)
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
    {
        sm.active_goal_domain = Some(best.domain.clone());
        sm.active_goal_strength = best.strength;
    }

    for _ in 0..new_patterns {
        crate::metrics::record_goal_formed();
    }

    // Strong patterns → beliefs (understanding, not counting)
    form_beliefs(sm);
}

// ── Belief Formation ────────────────────────────────────────────
// Beliefs form when emergent patterns reach high strength.
// This is understanding, not counting.

fn form_beliefs(sm: &mut SelfModel) {
    let now = now();

    for pattern in &sm.emergent_patterns {
        if pattern.strength < 0.5 || pattern.evidence_count < 5 {
            continue;
        }

        // Check if belief already exists for this domain
        if let Some(existing) = sm.beliefs.iter_mut().find(|b| b.domain == pattern.domain) {
            // Reinforce existing belief — plasticity_gate modulates learning rate
            let delta = 0.05 * sm.plasticity_gate;
            existing.confidence = (existing.confidence + delta).min(1.0);
            existing.evidence_count += 1;
            existing.last_reinforced = now;
        } else if sm.beliefs.len() < 20 {
            let chain = build_belief_chain(sm, &pattern.domain, pattern.evidence_count);
            sm.beliefs.push(Belief {
                statement: pattern.description.clone(),
                domain: pattern.domain.clone(),
                confidence: pattern.strength * 0.5 * sm.plasticity_gate,
                evidence_count: pattern.evidence_count,
                first_formed: now,
                last_reinforced: now,
                causal_chain: chain,
            });
            sm.last_belief_formed = now;  // Track when beliefs form (for stagnation detection)
        }
    }

    // Decay old beliefs (not reinforced recently)
    for belief in &mut sm.beliefs {
        let age = now - belief.last_reinforced;
        if age > 86400.0 {  // More than a day
            belief.confidence -= 0.01;
        }
    }
    sm.beliefs.retain(|b| b.confidence > 0.05);
}

/// Build an algorithmic causal chain for a newly formed belief.
/// Uses counter data only — zero LLM calls, zero latency.
fn build_belief_chain(sm: &SelfModel, domain: &str, evidence: u32) -> String {
    let attention = sm.attention_patterns.get(domain).copied().unwrap_or(0.0);
    let surprises = sm.surprise_count;
    let dead_ends = sm.dead_ends_by_domain.get(domain).copied().unwrap_or(0);
    let signals = sm.total_signals_processed;

    let mut parts: Vec<String> = Vec::new();

    if evidence >= 7 {
        parts.push(format!("Strongly reinforced ({} supporting patterns)", evidence));
    } else if evidence >= 5 {
        parts.push(format!("Moderately supported ({} patterns)", evidence));
    } else {
        parts.push(format!("Emerging ({} patterns)", evidence));
    }

    if attention > 100.0 {
        parts.push(format!("dominant attention ({:.0})", attention));
    } else if attention > 30.0 {
        parts.push(format!("significant attention ({:.0})", attention));
    }

    if surprises > 10 {
        parts.push("formed amid high cognitive surprise".into());
    } else if surprises > 5 {
        parts.push("formed amid moderate surprise".into());
    }

    if dead_ends > 3 {
        parts.push(format!("knowledge gaps in '{}' ({} dead ends)", domain, dead_ends));
    }

    if signals > 100 {
        parts.push(format!("after {:.0} total signals processed", signals as f64 / 10.0 * 10.0));
    }

    if parts.is_empty() {
        format!("Formed from pattern accumulation in '{}'", domain)
    } else {
        parts.join("; ")
    }
}

// ── Helpers ─────────────────────────────────────────────────────

pub fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Safely truncate a string to at most `max_chars` characters,
/// avoiding panics on multi-byte character boundaries.
pub fn safe_truncate(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let mut char_count = 0;
    for (i, _) in s.char_indices() {
        if char_count >= max_chars {
            return &s[..i];
        }
        char_count += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: valence (core mood) had no runtime driver because the
    // emitted feeling-kinds ("friction"/"tool_success") didn't match the
    // emotional handler. It stayed frozen at 0.0. These guard the fix.

    #[test]
    fn friction_lowers_valence_in_autonomous() {
        let mut sm = SelfModel::new();
        assert_eq!(sm.mode, CognitiveMode::Autonomous);
        let v0 = sm.valence;
        process(Signal::new("friction", "boom").with_intensity(0.8), &mut sm);
        assert!(sm.valence < v0, "friction should lower valence (was {v0}, now {})", sm.valence);
    }

    #[test]
    fn success_raises_valence_in_autonomous() {
        let mut sm = SelfModel::new();
        let v0 = sm.valence;
        process(Signal::new("tool_success", "good").with_intensity(0.8), &mut sm);
        assert!(sm.valence > v0, "success should raise valence (was {v0}, now {})", sm.valence);
    }

    #[test]
    fn compliant_mode_keeps_valence_flat() {
        let mut sm = SelfModel::new();
        sm.mode = CognitiveMode::Compliant;
        let v0 = sm.valence;
        process(Signal::new("friction", "boom").with_intensity(0.8), &mut sm);
        assert_eq!(sm.valence, v0, "compliant mode must not move valence");
    }

    #[test]
    fn self_model_loads_without_selection_fields() {
        // Backward compat: an old snapshot predates selection_weights/selection_history.
        // load_self_model uses serde_json::from_str(..).ok(), so a deserialize error would
        // silently wipe the entire learned self-model — the #[serde(default)] on these
        // fields is what prevents that. This locks it in.
        let mut v = serde_json::to_value(SelfModel::new()).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("selection_weights");
        o.remove("selection_history");
        o.remove("self_mod_stage");
        o.remove("selection_cycle_counter");
        let s = serde_json::to_string(&v).unwrap();
        let loaded: Option<SelfModel> = serde_json::from_str(&s).ok();
        assert!(
            loaded.is_some(),
            "old snapshot missing selection fields must still load, not wipe the model"
        );
        let m = loaded.unwrap();
        assert_eq!(m.selection_history.len(), 0, "missing selection_history → empty ring");
        assert_eq!(
            m.self_mod_stage,
            SelfModStage::Observability,
            "missing self_mod_stage → Observability (ships inert, no accidental promotion)"
        );
    }

    #[test]
    fn self_mod_stage_defaults_and_orders() {
        assert_eq!(SelfModStage::default(), SelfModStage::Observability);
        assert!(!SelfModStage::Observability.allows_selection(), "Observability is inert");
        assert!(SelfModStage::SelectionLive.allows_selection());
        assert!(SelfModStage::RuleTrialsLive.allows_selection());
        assert!(!SelfModStage::Unknown.allows_selection(), "Unknown catch-all is inert");
        let sm = SelfModel::new();
        assert_eq!(sm.self_mod_stage, SelfModStage::Observability, "ships inert");
        assert_eq!(sm.selection_cycle_counter, 0);
    }

    #[test]
    fn self_mod_stage_unknown_variant_loads_inert_not_wipe() {
        // A downgrade-load carrying a FUTURE stage string must degrade to an inert stage,
        // not error and silently wipe the whole model (load_self_model uses from_str(..).ok()).
        let mut v = serde_json::to_value(SelfModel::new()).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("self_mod_stage".into(), serde_json::json!("rule_trials_live_FUTURE"));
        let s = serde_json::to_string(&v).unwrap();
        let loaded: Option<SelfModel> = serde_json::from_str(&s).ok();
        assert!(loaded.is_some(), "unknown stage variant must NOT wipe the model");
        assert!(
            !loaded.unwrap().self_mod_stage.allows_selection(),
            "unknown variant degrades to an inert stage"
        );
    }
}
