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

// ── SelfModel ───────────────────────────────────────────────────
// The system's understanding of itself. Present in every computation.
// Not a log. Not a state dump. A living model that participates
// in and is changed by every operation.

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

    // ── Meta: awareness of own state ──
    pub total_signals_processed: u64,
    pub total_noticings: u64,
    pub uptime: f64,
    pub started_at: f64,
}

/// A belief: something the self-model holds as true from repeated experience.
/// Beliefs form when the same noticing pattern occurs 5+ times.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Belief {
    pub statement: String,
    pub domain: String,
    pub confidence: f32,      // 0-1: how sure (grows with evidence)
    pub evidence_count: u32,  // how many noticings support this
    pub first_formed: f64,
    pub last_reinforced: f64,
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
    self_model.last_signal = format!("{}:{}", signal.kind, &signal.content[..signal.content.len().min(60)]);
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
                        &signal.content[..signal.content.len().min(80)], sim
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
        let who = signal.content[..signal.content.len().min(40)].to_string();
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
        self_model.energy = self_model.energy + (0.7 - self_model.energy) * 0.0001;

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
            content: output.content[..output.content.len().min(80)].to_string(),
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
            // Form new belief
            sm.beliefs.push(Belief {
                statement: pattern.description.clone(),
                domain: pattern.domain.clone(),
                confidence: pattern.strength * 0.5 * sm.plasticity_gate,
                evidence_count: pattern.evidence_count,
                first_formed: now,
                last_reinforced: now,
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

// ── Helpers ─────────────────────────────────────────────────────

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
