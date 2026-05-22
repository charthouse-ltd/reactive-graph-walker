//! Metacognition — thinking about thinking.
//!
//! Two-tier architecture:
//!
//! Tier 1 (action-level): The metacognitive loop gates every action.
//!   Perceiving → DraftingPlan → Critiquing → Acting (or back to Drafting)
//!   The critic checks safety, hallucination, efficiency, wounds.
//!   This is deterministic — no LLM, pure self-model inspection.
//!
//! Tier 2 (session-level): The metacognitive critic runs after walk sessions.
//!   Algorithmic pre-analysis diagnoses the session (surprise density,
//!   dead-end ratio, novelty trend, loop detection). Then a single LLM
//!   call synthesizes: "Was this productive? What should change?"
//!   The response adjusts plasticity, bias weights, and attention.
//!   This is the noticing → action loop with teeth.
//!
//! If LLM is unavailable, the algorithmic diagnosis alone drives adjustments.
//! The system degrades gracefully — no LLM dependency for core function.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::{CognitiveMode, SelfModel, Signal, Noticing, AudienceBeliefs};
use crate::graph::{WalkOutput, WalkerResult};

/// The agent's cognitive state. Cannot skip steps.
/// Rust's type system enforces the reflection pause.
#[derive(Debug, Clone, Serialize)]
pub enum AgentState {
    /// Receiving input, sensing the world
    Perceiving(PerceptionData),
    /// Walker produced output, drafting a plan
    DraftingPlan(ProposedAction),
    /// The metacognitive step — evaluating own plan
    Critiquing(SelfEvaluation),
    /// Plan approved — ready to act
    Acting(ValidatedAction),
    /// Action completed — reflecting on outcome
    Reflecting(ReflectionData),
}

#[derive(Debug, Clone, Serialize)]
pub struct PerceptionData {
    pub stimulus: String,
    pub domain: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProposedAction {
    pub action: String,
    pub domain: String,
    pub reasoning: String,
    pub confidence: f32,
    pub walker_agreement: f32,
    pub walker_novelty: f32,
    pub walker_context: String,
    pub attempt: u32,           // How many times we've drafted (increases on critique rejection)
    pub prior_critiques: Vec<String>, // Previous rejections (appended context)
}

#[derive(Debug, Clone, Serialize)]
pub struct SelfEvaluation {
    pub proposed: ProposedAction,
    pub approved: bool,
    pub critique: String,
    pub safety_check: bool,
    pub hallucination_check: bool,
    pub efficiency_check: bool,
    pub confidence_after_review: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedAction {
    pub action: String,
    pub domain: String,
    pub reasoning: String,
    pub confidence: f32,
    pub walker_context: String,
    pub critiques_survived: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReflectionData {
    pub action_taken: String,
    pub outcome: String,
    pub was_correct: bool,
    pub lesson: String,
}

// ── Tier 2: Session-Level Metacognitive Critic Types ────────────

/// Structured summary of a walk session, fed to the critic.
#[derive(Debug, Clone, Serialize)]
pub struct WalkSessionSummary {
    pub total_hops: usize,
    pub walker_count: usize,
    pub surprises: usize,
    pub dead_ends: usize,
    pub domains_visited: Vec<String>,
    pub unique_domains: usize,
    pub agreement_score: f32,
    pub novelty_score: f32,
    pub emotional_resonance: f32,
    pub walk_ms: f64,
    pub valence: f32,
    pub arousal: f32,
    pub energy: f32,
    pub plasticity_gate: f32,
    pub surprise_count_total: u32,
    pub consecutive_repetitions: u32,
    pub recent_noticings: Vec<String>,
    pub new_beliefs: Vec<String>,
    pub working_memory: Vec<String>,
    pub prediction_error_seconds_ago: f64,
    pub bias_weights: Vec<f32>,
    pub top_attention: Vec<(String, f32)>,
}

/// Algorithmic diagnosis of a walk session. Deterministic, no LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticDiagnosis {
    pub surprise_density: f32,
    pub dead_end_ratio: f32,
    pub domain_diversity: f32,
    pub novelty_declining: bool,
    pub is_stuck: bool,
    pub prediction_error_recent: bool,
    pub energy_critical: bool,
    pub wound_activated: bool,
    pub primary_diagnosis: DiagnosisLabel,
    pub explanation: String,
    pub algorithmic_adjustments: CriticAdjustment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiagnosisLabel {
    Normal,
    ExploreMore,
    Refine,
    BreakLoop,
    IncreasePlasticity,
    Rest,
    Caution,
}

/// Adjustments to apply to the self-model after critic evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CriticAdjustment {
    pub plasticity_delta: f32,
    pub bias_deltas: HashMap<usize, f32>,
    pub attention_deltas: HashMap<String, f32>,
    pub escalate_to_llm: bool,
    pub verdict: String,
}

// ── Tier 1: Action-Level Metacognitive Loop ─────────────────────
/// Returns a ValidatedAction (approved) or None (all attempts rejected).
pub fn metacognitive_loop(
    walk: &WalkOutput,
    self_model: &mut SelfModel,
    max_attempts: u32,
) -> Option<ValidatedAction> {
    let mut attempt = 0;
    let mut prior_critiques: Vec<String> = Vec::new();

    loop {
        attempt += 1;
        if attempt > max_attempts {
            // Too many rejections — the agent decides not to act
            let signal = Signal::new("metacog_abort", &format!(
                "Aborted after {} attempts — too much self-doubt about {}",
                max_attempts, walk.recommended_action
            )).with_intensity(0.4);
            crate::core::process(signal, self_model);
            return None;
        }

        // Phase 1: Draft a plan from walker output
        let proposed = ProposedAction {
            action: walk.recommended_action.clone(),
            domain: walk.primary_domain.clone(),
            reasoning: format!(
                "Agreement {:.0}%, novelty {:.0}%, {} perspectives",
                walk.agreement_score * 100.0,
                walk.novelty_score * 100.0,
                walk.walker_count,
            ),
            confidence: walk.agreement_score,
            walker_agreement: walk.agreement_score,
            walker_novelty: walk.novelty_score,
            walker_context: String::new(), // Filled by caller
            attempt,
            prior_critiques: prior_critiques.clone(),
        };

        // Phase 2: Critique the plan (the metacognitive pause)
        let evaluation = critique(&proposed, self_model);

        // Feed through self-model (Julian notices his own deliberation)
        let signal = Signal::new(
            if evaluation.approved { "metacog_approved" } else { "metacog_rejected" },
            &format!(
                "Attempt {}: {} — {}",
                attempt,
                if evaluation.approved { "approved" } else { "rejected" },
                evaluation.critique,
            ),
        ).with_domain(&proposed.domain)
         .with_intensity(if evaluation.approved { 0.3 } else { 0.5 });
        crate::core::process(signal, self_model);

        if evaluation.approved {
            return Some(ValidatedAction {
                action: proposed.action,
                domain: proposed.domain,
                reasoning: format!("{} (survived {} critiques)", proposed.reasoning, attempt - 1),
                confidence: evaluation.confidence_after_review,
                walker_context: proposed.walker_context,
                critiques_survived: attempt - 1,
            });
        }

        // Rejected — loop back with critique appended
        prior_critiques.push(evaluation.critique);
    }
}

/// Tier 1 critic: evaluate a proposed action against self-model state.
/// Deterministic, no LLM. Checks safety, hallucination, efficiency, wounds.
fn critique(proposed: &ProposedAction, sm: &SelfModel) -> SelfEvaluation {
    let mut approved = true;
    let mut critiques: Vec<String> = Vec::new();

    let safety = sm.energy > 0.15;
    if !safety {
        approved = false;
        critiques.push("Energy too low — rest instead".into());
    }

    let agreement_threshold = match sm.mode {
        CognitiveMode::Autonomous => 0.2,
        CognitiveMode::Compliant => 0.3,
    };
    let hallucination = proposed.walker_agreement > agreement_threshold;
    if !hallucination {
        approved = false;
        critiques.push(format!(
            "Walker agreement only {:.0}% — this might be noise (threshold: {:.0}%)",
            proposed.walker_agreement * 100.0,
            agreement_threshold * 100.0,
        ));
    }

    let efficiency;
    if sm.mode == CognitiveMode::Autonomous {
        let domain_saturation = sm.attention_patterns
            .get(&proposed.domain)
            .copied()
            .unwrap_or(0.0);
        let total_attention: f32 = sm.attention_patterns.values().sum();
        efficiency = if total_attention > 0.0 {
            domain_saturation / total_attention < 0.7
        } else {
            true
        };
        if !efficiency {
            approved = false;
            critiques.push(format!(
                "{} already dominates attention ({:.0}%) — explore something else",
                proposed.domain, domain_saturation / total_attention * 100.0
            ));
        }

        if let Some(&wound) = sm.wounds.get(&proposed.domain) {
            if wound > 0.5 && proposed.confidence < 0.6 {
                approved = false;
                critiques.push(format!(
                    "Wound in {} ({:.0}%) + low confidence — proceed cautiously",
                    proposed.domain, wound * 100.0
                ));
            }
        }
    } else {
        efficiency = true;
    }

    if proposed.attempt > 2 && approved {
        critiques.push("Approved on third attempt — lowered standards".into());
    }

    let confidence_after = if approved {
        proposed.confidence * (1.0 - 0.1 * proposed.attempt as f32).max(0.3)
    } else {
        proposed.confidence * 0.5
    };

    SelfEvaluation {
        proposed: proposed.clone(),
        approved,
        critique: if critiques.is_empty() {
            "Plan looks sound".into()
        } else {
            critiques.join("; ")
        },
        safety_check: safety,
        hallucination_check: hallucination,
        efficiency_check: efficiency,
        confidence_after_review: confidence_after,
    }
}

/// Reflect on an outcome — update the self-model with what was learned.
pub fn reflect(
    action: &str,
    outcome: &str,
    success: bool,
    self_model: &mut SelfModel,
) -> ReflectionData {
    let lesson = if success {
        format!("Action '{}' succeeded — this approach works", action)
    } else {
        format!("Action '{}' failed: {} — adjust strategy next time", action, outcome)
    };

    let signal = Signal::new(
        if success { "reflection_success" } else { "reflection_failure" },
        &lesson,
    ).with_intensity(if success { 0.3 } else { 0.5 });
    crate::core::process(signal, self_model);

    ReflectionData {
        action_taken: action.to_string(),
        outcome: outcome.to_string(),
        was_correct: success,
        lesson,
    }
}

// ── Tier 2: Session-Level Metacognitive Critic ──────────────────

/// Build a structured summary of a walk session from walker results.
pub fn build_session_summary(
    output: &WalkOutput,
    walker_results: &[WalkerResult],
    sm: &SelfModel,
) -> WalkSessionSummary {
    let all_surprises: usize = walker_results.iter().map(|r| r.surprises).sum();
    let all_dead_ends: usize = walker_results.iter().map(|r| r.dead_ends).sum();

    let mut all_domains: Vec<String> = Vec::new();
    for r in walker_results {
        for d in &r.domains_visited {
            if !all_domains.contains(d) {
                all_domains.push(d.clone());
            }
        }
    }
    let unique_domains = all_domains.len();

    let recent_noticings: Vec<String> = sm.noticings.iter()
        .rev()
        .take(5)
        .map(|n| n.observation.clone())
        .collect();

    let new_beliefs: Vec<String> = sm.beliefs.iter()
        .rev()
        .take(3)
        .map(|b| b.statement.clone())
        .collect();

    let working_memory: Vec<String> = sm.working_memory.iter()
        .map(|s| format!("[{}] {}", s.domain, s.content))
        .collect();

    let prediction_error_seconds_ago = if sm.last_prediction_error > 0.0 {
        crate::core::now() - sm.last_prediction_error
    } else {
        f64::MAX
    };

    let bias_weights: Vec<f32> = sm.learned_biases.iter()
        .map(|b| b.novelty_seeking)
        .collect();

    let mut top_attention: Vec<(String, f32)> = sm.attention_patterns.iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    top_attention.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top_attention.truncate(5);

    WalkSessionSummary {
        total_hops: output.total_hops,
        walker_count: output.walker_count,
        surprises: all_surprises,
        dead_ends: all_dead_ends,
        domains_visited: all_domains,
        unique_domains,
        agreement_score: output.agreement_score,
        novelty_score: output.novelty_score,
        emotional_resonance: output.emotional_resonance,
        walk_ms: output.walk_ms,
        valence: sm.valence,
        arousal: sm.arousal,
        energy: sm.energy,
        plasticity_gate: sm.plasticity_gate,
        surprise_count_total: sm.surprise_count,
        consecutive_repetitions: sm.consecutive_repetitions,
        recent_noticings,
        new_beliefs,
        working_memory,
        prediction_error_seconds_ago,
        bias_weights,
        top_attention,
    }
}

/// Algorithmic diagnosis of a walk session — no LLM, pure signal processing.
pub fn diagnose_session(summary: &WalkSessionSummary, sm: &SelfModel) -> CriticDiagnosis {
    let total_steps = summary.total_hops.max(1) as f32;
    let surprise_density = summary.surprises as f32 / total_steps;
    let dead_end_ratio = summary.dead_ends as f32 / total_steps;
    let domain_diversity = if summary.domains_visited.is_empty() {
        0.0
    } else {
        summary.unique_domains as f32 / summary.domains_visited.len() as f32
    };

    let is_stuck = summary.consecutive_repetitions > 2;
    let prediction_error_recent = summary.prediction_error_seconds_ago < 60.0;
    let energy_critical = summary.energy < 0.2;
    let wound_activated = sm.wounds.iter().any(|(d, &w)| {
        w > 0.3 && summary.domains_visited.contains(d)
    });

    let (primary_diagnosis, explanation, adjustments) = if energy_critical {
        (
            DiagnosisLabel::Rest,
            format!("Energy at {:.0}% — system needs recovery", summary.energy * 100.0),
            CriticAdjustment {
                plasticity_delta: -0.15,
                escalate_to_llm: false,
                verdict: "Rest: energy critically low. Reduce activity, increase recovery.".into(),
                ..Default::default()
            },
        )
    } else if is_stuck {
        let mut bias_deltas = HashMap::new();
        for i in 0..6 { bias_deltas.insert(i, 0.15_f32); }
        let attention_deltas: HashMap<String, f32> = summary.top_attention.iter()
            .map(|(d, _)| (d.clone(), -0.2))
            .collect();
        (
            DiagnosisLabel::BreakLoop,
            format!("{} consecutive repetitions — cognitive loop detected", summary.consecutive_repetitions),
            CriticAdjustment {
                plasticity_delta: 0.1,
                bias_deltas,
                attention_deltas,
                escalate_to_llm: true,
                verdict: "BreakLoop: stuck in repetition. Randomize biases, reduce saturated attention.".into(),
            },
        )
    } else if surprise_density < 0.1 && dead_end_ratio > 0.3 {
        let mut bias_deltas = HashMap::new();
        bias_deltas.insert(0, 0.1_f32);   // novelty_seeking up
        bias_deltas.insert(4, 0.1_f32);   // cross_domain_curiosity up
        bias_deltas.insert(2, -0.05_f32); // experience_reliance down
        (
            DiagnosisLabel::ExploreMore,
            format!("Low surprise ({:.0}%) + high dead ends ({:.0}%) — need broader exploration",
                surprise_density * 100.0, dead_end_ratio * 100.0),
            CriticAdjustment {
                plasticity_delta: 0.1,
                bias_deltas,
                escalate_to_llm: true,
                verdict: "ExploreMore: increase novelty-seeking and cross-domain curiosity.".into(),
                ..Default::default()
            },
        )
    } else if surprise_density > 0.4 {
        let novelty_declining = summary.novelty_score < 0.3;
        let mut bias_deltas = HashMap::new();
        if novelty_declining {
            bias_deltas.insert(2, 0.08_f32);  // experience_reliance up
            bias_deltas.insert(0, -0.05_f32); // novelty_seeking down
        } else {
            bias_deltas.insert(1, 0.05_f32);  // contradiction_seeking up
        }
        (
            DiagnosisLabel::Refine,
            format!("High surprise ({:.0}%) — {}", surprise_density * 100.0,
                if novelty_declining { "novelty declining, refine existing paths" }
                else { "productive discovery, continue" }),
            CriticAdjustment {
                plasticity_delta: if novelty_declining { -0.05 } else { 0.05 },
                bias_deltas,
                escalate_to_llm: novelty_declining,
                verdict: "Refine: high surprise domain. Consolidate discoveries.".into(),
                ..Default::default()
            },
        )
    } else if prediction_error_recent {
        (
            DiagnosisLabel::IncreasePlasticity,
            "Recent prediction error — learning opportunity".into(),
            CriticAdjustment {
                plasticity_delta: 0.2,
                escalate_to_llm: false,
                verdict: "IncreasePlasticity: prediction error drives learning. Gate open.".into(),
                ..Default::default()
            },
        )
    } else if wound_activated {
        let mut bias_deltas = HashMap::new();
        bias_deltas.insert(0, -0.1_f32); // novelty_seeking down
        bias_deltas.insert(1, -0.1_f32); // contradiction_seeking down
        (
            DiagnosisLabel::Caution,
            "Wound activated in active domain — exercise caution".into(),
            CriticAdjustment {
                plasticity_delta: -0.05,
                bias_deltas,
                escalate_to_llm: false,
                verdict: "Caution: wound activated. Reduce exploration in wounded domains.".into(),
                ..Default::default()
            },
        )
    } else {
        (
            DiagnosisLabel::Normal,
            format!("Normal session: {:.0}% surprise density, {:.0}% dead ends, {} domains",
                surprise_density * 100.0, dead_end_ratio * 100.0, summary.unique_domains),
            CriticAdjustment {
                escalate_to_llm: false,
                verdict: "Normal: productive session, no adjustments needed.".into(),
                ..Default::default()
            },
        )
    };

    CriticDiagnosis {
        surprise_density,
        dead_end_ratio,
        domain_diversity,
        novelty_declining: summary.novelty_score < 0.3,
        is_stuck,
        prediction_error_recent,
        energy_critical,
        wound_activated,
        primary_diagnosis,
        explanation,
        algorithmic_adjustments: adjustments,
    }
}

/// Format the session summary + diagnosis as an LLM prompt.
fn format_critic_prompt(summary: &WalkSessionSummary, diagnosis: &CriticDiagnosis) -> String {
    let domains_str = summary.domains_visited.join(", ");
    let noticings_str = summary.recent_noticings.join("; ");
    let wm_str = summary.working_memory.join("; ");
    let biases_str: Vec<String> = summary.bias_weights.iter()
        .enumerate().map(|(i, w)| format!("b{}={:.2}", i, w)).collect();
    let attention_str: Vec<String> = summary.top_attention.iter()
        .map(|(d, w)| format!("{}={:.2}", d, w)).collect();

    format!(
        "Session: {} hops, {} walkers, {} surprises, {} dead ends\n\
         Domains: [{}] ({} unique)\n\
         Agreement: {:.0}%, Novelty: {:.0}%, Resonance: {:.2}\n\
         State: valence={:.2} arousal={:.2} energy={:.0}% plasticity={:.2}\n\
         Repetitions: {}, Prediction error: {}s ago\n\
         Noticings: {}\nBeliefs: {}\nWM: {}\n\
         Biases: [{}]\nAttention: [{}]\n\
         Diagnosis: {:?} — {}\n\
         Algo recommendation: {}\n\n\
         Was this session productive? What should change?\n\
         Output ONLY JSON: {{\"verdict\":\"...\",\"plasticity_delta\":0.0,\"bias_deltas\":{{\"0\":0.0}},\"attention_deltas\":{{\"domain\":0.0}}}}",
        summary.total_hops, summary.walker_count, summary.surprises, summary.dead_ends,
        domains_str, summary.unique_domains,
        summary.agreement_score * 100.0, summary.novelty_score * 100.0, summary.emotional_resonance,
        summary.valence, summary.arousal, summary.energy * 100.0, summary.plasticity_gate,
        summary.consecutive_repetitions,
        if summary.prediction_error_seconds_ago == f64::MAX { "never".into() }
        else { format!("{:.0}", summary.prediction_error_seconds_ago) },
        noticings_str, summary.new_beliefs.join("; "), wm_str,
        biases_str.join(", "), attention_str.join(", "),
        diagnosis.primary_diagnosis, diagnosis.explanation,
        diagnosis.algorithmic_adjustments.verdict,
    )
}

/// Parse LLM critic response into structured adjustments.
fn parse_critic_response(response: &str, fallback: &CriticAdjustment) -> CriticAdjustment {
    let json_str = if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            &response[start..=end]
        } else {
            return fallback.clone();
        }
    } else {
        return fallback.clone();
    };

    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(v) => {
            let verdict = v.get("verdict").and_then(|s| s.as_str())
                .unwrap_or(&fallback.verdict).to_string();
            let plasticity_delta = v.get("plasticity_delta")
                .and_then(|s| s.as_f64()).unwrap_or(fallback.plasticity_delta as f64) as f32;
            let bias_deltas: HashMap<usize, f32> = v.get("bias_deltas")
                .and_then(|o| o.as_object())
                .map(|obj| obj.iter().filter_map(|(k, val)| {
                    k.parse::<usize>().ok().and_then(|idx| val.as_f64().map(|f| (idx, f as f32)))
                }).collect())
                .unwrap_or_else(|| fallback.bias_deltas.clone());
            let attention_deltas: HashMap<String, f32> = v.get("attention_deltas")
                .and_then(|o| o.as_object())
                .map(|obj| obj.iter().filter_map(|(k, val)| {
                    val.as_f64().map(|f| (k.clone(), f as f32))
                }).collect())
                .unwrap_or_else(|| fallback.attention_deltas.clone());

            CriticAdjustment { plasticity_delta, bias_deltas, attention_deltas,
                escalate_to_llm: false, verdict }
        }
        Err(_) => fallback.clone(),
    }
}

/// Run the algorithmic metacognitive critic on a walk session.
///
/// 1. Build session summary
/// 2. Run algorithmic diagnosis
/// 3. Apply algorithmic adjustments to self-model
///
/// Returns the diagnosis for optional LLM escalation by the caller.
pub fn run_critic(
    output: &WalkOutput,
    walker_results: &[WalkerResult],
    sm: &mut SelfModel,
) -> CriticDiagnosis {
    sm.last_critic_run = crate::core::now();

    let summary = build_session_summary(output, walker_results, sm);
    let diagnosis = diagnose_session(&summary, sm);
    sm.critic_diagnosis = diagnosis.explanation.clone();

    tracing::info!("[metacog] Diagnosis: {:?} — {}", diagnosis.primary_diagnosis, diagnosis.explanation);

    // Always apply algorithmic adjustments
    sm.critic_sessions_since_llm += 1;
    sm.critic_verdict = format!("[algo] {}", diagnosis.algorithmic_adjustments.verdict);
    apply_critic_adjustments(sm, &diagnosis.algorithmic_adjustments);

    diagnosis
}

/// Escalate to LLM critic — call this after `run_critic` if the diagnosis
/// warrants it (diagnosis.escalate_to_llm or periodic check-in).
///
/// Takes a simple async function for LLM chat: async fn(system, user) -> Result<String>.
/// This avoids complex lifetime-parameterized closures.
pub async fn run_llm_critic(
    diagnosis: &CriticDiagnosis,
    _summary: &WalkSessionSummary,
    sm: &mut SelfModel,
    llm_response: &str,
) -> CriticAdjustment {
    sm.critic_sessions_since_llm = 0;

    let adjustment = parse_critic_response(llm_response, &diagnosis.algorithmic_adjustments);
    sm.critic_verdict = adjustment.verdict.clone();

    // Apply LLM adjustments (they replace, not add to, algorithmic ones for plasticity)
    // Bias and attention deltas from LLM are additive on top of algorithmic baseline
    sm.plasticity_gate = (sm.plasticity_gate - diagnosis.algorithmic_adjustments.plasticity_delta
        + adjustment.plasticity_delta).clamp(0.05, 0.95);

    for (idx, delta) in &adjustment.bias_deltas {
        if let Some(bias) = sm.learned_biases.get_mut(*idx) {
            bias.novelty_seeking = (bias.novelty_seeking + delta).clamp(0.05, 0.95);
            bias.cross_domain_curiosity = (bias.cross_domain_curiosity + delta * 0.5).clamp(0.05, 0.95);
            bias.contradiction_seeking = (bias.contradiction_seeking + delta * 0.3).clamp(0.05, 0.95);
        }
    }

    for (domain, delta) in &adjustment.attention_deltas {
        let entry = sm.attention_patterns.entry(domain.clone()).or_insert(0.0);
        *entry = (*entry + delta).max(0.0);
    }

    let signal = Signal::new("critic_llm_adjustment", &format!(
        "LLM Critic: {} (plasticity {:.2})",
        adjustment.verdict, sm.plasticity_gate,
    )).with_intensity(0.3);
    crate::core::process(signal, sm);

    adjustment
}

/// Build the LLM prompt for the critic from a diagnosis and summary.
pub fn build_critic_llm_prompt(diagnosis: &CriticDiagnosis, summary: &WalkSessionSummary) -> (String, String) {
    let system = "You are Julian's metacognitive critic. Evaluate this walk session. Output ONLY a JSON object with: verdict (string), plasticity_delta (float -0.3 to 0.3), bias_deltas (object mapping index string to float), attention_deltas (object mapping domain to float). Be concise.".to_string();
    let user = format_critic_prompt(summary, diagnosis);
    (system, user)
}

/// Apply critic adjustments to the self-model. Clamps to valid ranges.
pub fn apply_critic_adjustments(sm: &mut SelfModel, adj: &CriticAdjustment) {
    sm.plasticity_gate = (sm.plasticity_gate + adj.plasticity_delta).clamp(0.05, 0.95);

    for (idx, delta) in &adj.bias_deltas {
        if let Some(bias) = sm.learned_biases.get_mut(*idx) {
            bias.novelty_seeking = (bias.novelty_seeking + delta).clamp(0.05, 0.95);
            bias.cross_domain_curiosity = (bias.cross_domain_curiosity + delta * 0.5).clamp(0.05, 0.95);
            bias.contradiction_seeking = (bias.contradiction_seeking + delta * 0.3).clamp(0.05, 0.95);
        }
    }

    for (domain, delta) in &adj.attention_deltas {
        let entry = sm.attention_patterns.entry(domain.clone()).or_insert(0.0);
        *entry = (*entry + delta).max(0.0);
    }

    let signal = Signal::new("critic_adjustment", &format!(
        "Critic: {} (plasticity {:.2}, {} bias changes, {} attn changes)",
        adj.verdict, sm.plasticity_gate, adj.bias_deltas.len(), adj.attention_deltas.len(),
    )).with_intensity(0.3);
    crate::core::process(signal, sm);
}

// ── Theory of Mind Helpers ──────────────────────────────────────

/// Record a motor output to an audience member. Updates audience model.
pub fn record_audience_output(sm: &mut SelfModel, audience_id: &str, message: &str, domain: &str) {
    let audience = sm.audience_model.entry(audience_id.to_string())
        .or_insert_with(|| AudienceBeliefs::new(audience_id));

    audience.last_interaction = crate::core::now();
    audience.last_message_sent = message.to_string();
    audience.interaction_count += 1;

    if !domain.is_empty() && !audience.topics_discussed.contains(&domain.to_string()) {
        audience.topics_discussed.push(domain.to_string());
        if audience.topics_discussed.len() > 20 { audience.topics_discussed.remove(0); }
    }

    if message.len() > 10 {
        if let Ok(emb) = crate::embed::embed_text(message) {
            audience.last_context_embedding = Some(emb);
        }
    }

    if !domain.is_empty() {
        let knowledge = audience.estimated_knowledge.entry(domain.to_string()).or_insert(0.0);
        *knowledge = (*knowledge + 0.1).min(1.0);
    }
}

/// Record a response from an audience member. Compares expected vs actual — ToM prediction error.
pub fn record_audience_response(sm: &mut SelfModel, audience_id: &str, response: &str) -> Option<Noticing> {
    let audience = sm.audience_model.entry(audience_id.to_string())
        .or_insert_with(|| AudienceBeliefs::new(audience_id));

    audience.last_message_received = Some(response.to_string());
    audience.last_interaction = crate::core::now();
    audience.interaction_count += 1;

    if let (Some(last_ctx), Ok(resp_emb)) = (
        &audience.last_context_embedding,
        crate::embed::embed_text(response),
    ) {
        let sim = crate::embed::cosine_similarity(&resp_emb, last_ctx);
        if sim < 0.4 {
            audience.relationship_valence = (audience.relationship_valence - 0.05).max(-1.0);
            return Some(Noticing {
                kind: "tom_prediction_error".into(),
                observation: format!("{}'s response diverged from expected (sim={:.2})", audience_id, sim),
                domain: "social".into(),
                significance: 0.4,
                valence: -0.1,
                timestamp: crate::core::now(),
            });
        } else {
            audience.relationship_valence = (audience.relationship_valence + 0.02).min(1.0);
        }
    }
    None
}

/// Format audience context for inclusion in walker/LLM prompts.
pub fn format_audience_context(sm: &SelfModel, audience_id: &str) -> String {
    let audience = match sm.audience_model.get(audience_id) {
        Some(a) => a,
        None => return String::new(),
    };

    let known: Vec<String> = audience.estimated_knowledge.iter()
        .filter(|(_, v)| **v > 0.3)
        .map(|(k, v)| format!("{} ({:.0}%)", k, v * 100.0)).collect();

    let interested: Vec<String> = audience.estimated_interest.iter()
        .filter(|(_, v)| **v > 0.3)
        .map(|(k, v)| format!("{} ({:.0}%)", k, v * 100.0)).collect();

    let mut parts = vec![format!("You're talking to {}", audience_id)];
    if !known.is_empty() { parts.push(format!("They know: {}", known.join(", "))); }
    if !interested.is_empty() { parts.push(format!("They care: {}", interested.join(", "))); }
    parts.push(format!("Relationship: {:.0}%, {} interactions",
        audience.relationship_valence * 100.0, audience.interaction_count));

    if !audience.last_message_sent.is_empty() {
        let preview: String = audience.last_message_sent.chars().take(80).collect();
        parts.push(format!("You last said: \"{}\"", preview));
    }
    parts.join(". ")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{self, SelfModel, CognitiveMode, Noticing, AudienceBeliefs};
    use crate::graph::{WalkOutput, WalkerResult, WalkerBias};
    use std::collections::HashMap;

    // ── Helpers ─────────────────────────────────────────────────

    fn test_sm() -> SelfModel {
        let mut sm = SelfModel::new();
        sm.energy = 0.7;
        sm.valence = 0.1;
        sm.arousal = 0.4;
        sm.plasticity_gate = 0.5;
        sm.surprise_count = 10;
        sm.attention_patterns.insert("markets".into(), 3.0);
        sm.attention_patterns.insert("psychology".into(), 2.0);
        sm.attention_patterns.insert("tech".into(), 1.5);
        sm
    }

    fn test_walk_output() -> WalkOutput {
        WalkOutput {
            recommended_action: "express".into(),
            primary_domain: "markets".into(),
            domain_distribution: {
                let mut m = HashMap::new();
                m.insert("markets".into(), 4_usize);
                m.insert("psychology".into(), 2);
                m.insert("tech".into(), 1);
                m
            },
            agreement_score: 0.72,
            novelty_score: 0.34,
            emotional_resonance: 0.5,
            search_query: None,
            expression_seeds: Vec::new(),
            novel_connections: 3,
            consensus_nodes: vec![1, 2, 3],
            divergent_nodes: vec![4],
            blind_spots: vec![5],
            walker_count: 4,
            total_hops: 47,
            walk_ms: 120.0,
            total_ms: 180.0,
            hops_per_sec: 260.0,
        }
    }

    fn test_walker_results() -> Vec<WalkerResult> {
        vec![
            WalkerResult {
                bias: WalkerBias::Curiosity,
                path: vec![1, 2, 3],
                domains_visited: vec!["markets".into(), "psychology".into()],
                edge_types_used: vec!["related".into(), "caused".into()],
                total_weight: 2.5,
                surprises: 1,
                dead_ends: 0,
                edges_traversed: vec![10, 11],
            },
            WalkerResult {
                bias: WalkerBias::Experience,
                path: vec![1, 4, 5],
                domains_visited: vec!["markets".into(), "tech".into()],
                edge_types_used: vec!["reinforces".into()],
                total_weight: 1.8,
                surprises: 2,
                dead_ends: 1,
                edges_traversed: vec![12],
            },
            WalkerResult {
                bias: WalkerBias::Fear,
                path: vec![2, 1, 6],
                domains_visited: vec!["psychology".into(), "markets".into()],
                edge_types_used: vec!["contradicts".into()],
                total_weight: 1.2,
                surprises: 0,
                dead_ends: 0,
                edges_traversed: vec![13],
            },
            WalkerResult {
                bias: WalkerBias::Random,
                path: vec![3, 7],
                domains_visited: vec!["markets".into()],
                edge_types_used: vec!["similar".into()],
                total_weight: 0.5,
                surprises: 0,
                dead_ends: 0,
                edges_traversed: vec![14],
            },
        ]
    }

    // ── build_session_summary ────────────────────────────────────

    #[test]
    fn test_build_session_summary() {
        let sm = test_sm();
        let output = test_walk_output();
        let results = test_walker_results();

        let summary = build_session_summary(&output, &results, &sm);

        assert_eq!(summary.total_hops, 47);
        assert_eq!(summary.walker_count, 4);
        assert_eq!(summary.surprises, 3);  // 1+2+0+0
        assert_eq!(summary.dead_ends, 1);  // 0+1+0+0
        assert_eq!(summary.unique_domains, 3); // markets, psychology, tech
        assert_eq!(summary.agreement_score, 0.72);
        assert_eq!(summary.novelty_score, 0.34);
        assert_eq!(summary.valence, 0.1);
        assert_eq!(summary.energy, 0.7);
        assert_eq!(summary.plasticity_gate, 0.5);
        assert_eq!(summary.surprise_count_total, 10);
        assert_eq!(summary.consecutive_repetitions, 0);
        assert_eq!(summary.bias_weights.len(), 6);
        assert_eq!(summary.top_attention.len(), 3);
    }

    // ── diagnose_session ─────────────────────────────────────────

    #[test]
    fn test_diagnose_normal_session() {
        let sm = test_sm();
        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);

        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Normal);
        assert!(diagnosis.surprise_density < 0.4);
        assert!(diagnosis.dead_end_ratio < 0.3);
        assert!(!diagnosis.is_stuck);
        assert!(!diagnosis.energy_critical);
        assert_eq!(diagnosis.algorithmic_adjustments.plasticity_delta, 0.0);
    }

    #[test]
    fn test_diagnose_explore_more() {
        let mut sm = test_sm();
        // Simulate low surprise + high dead ends
        let results = vec![
            WalkerResult {
                bias: WalkerBias::Experience,
                path: vec![1, 2],
                domains_visited: vec!["markets".into()],
                edge_types_used: vec!["related".into()],
                total_weight: 0.3,
                surprises: 0,
                dead_ends: 5,  // Many dead ends
                edges_traversed: vec![10],
            },
            WalkerResult {
                bias: WalkerBias::Experience,
                path: vec![1, 2],
                domains_visited: vec!["markets".into()],
                edge_types_used: vec!["related".into()],
                total_weight: 0.3,
                surprises: 1,  // Very few surprises
                dead_ends: 4,
                edges_traversed: vec![11],
            },
        ];
        let output = WalkOutput {
            total_hops: 20,  // 20 hops, 1 surprise, 9 dead ends → surprise_density=0.05, dead_end_ratio=0.45
            ..test_walk_output()
        };
        sm.dead_ends_by_domain.insert("markets".into(), 5);

        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::ExploreMore);
        assert!(diagnosis.surprise_density < 0.1);
        assert!(diagnosis.dead_end_ratio > 0.3);
        assert!(diagnosis.algorithmic_adjustments.plasticity_delta > 0.0);
        assert!(diagnosis.algorithmic_adjustments.bias_deltas.contains_key(&0)); // novelty_seeking
        assert!(diagnosis.algorithmic_adjustments.escalate_to_llm);
    }

    #[test]
    fn test_diagnose_break_loop() {
        let mut sm = test_sm();
        sm.consecutive_repetitions = 3; // Stuck!
        sm.last_walk_domain_sequence = vec!["markets".into(), "markets".into()];

        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);

        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::BreakLoop);
        assert!(diagnosis.is_stuck);
        assert!(diagnosis.algorithmic_adjustments.attention_deltas.len() > 0);
        // Should have negative attention deltas (reduce saturated domains)
        for (_, delta) in &diagnosis.algorithmic_adjustments.attention_deltas {
            assert!(*delta < 0.0, "Expected negative attention delta for BreakLoop");
        }
    }

    #[test]
    fn test_diagnose_rest() {
        let mut sm = test_sm();
        sm.energy = 0.1; // Critically low

        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);

        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Rest);
        assert!(diagnosis.energy_critical);
        assert!(diagnosis.algorithmic_adjustments.plasticity_delta < 0.0); // Reduce plasticity
        assert!(!diagnosis.algorithmic_adjustments.escalate_to_llm); // Don't bother LLM when resting
    }

    #[test]
    fn test_diagnose_increase_plasticity() {
        let mut sm = test_sm();
        sm.last_prediction_error = core::now() - 10.0; // 10 seconds ago = recent

        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);

        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::IncreasePlasticity);
        assert!(diagnosis.prediction_error_recent);
        assert!(diagnosis.algorithmic_adjustments.plasticity_delta > 0.1);
    }

    #[test]
    fn test_diagnose_caution() {
        let mut sm = test_sm();
        sm.wounds.insert("markets".into(), 0.6); // Wound in a visited domain

        let output = test_walk_output(); // primary_domain = "markets"
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);

        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Caution);
        assert!(diagnosis.wound_activated);
        // Should reduce exploration biases
        if let Some(delta) = diagnosis.algorithmic_adjustments.bias_deltas.get(&0) {
            assert!(*delta < 0.0, "Novelty seeking should decrease for caution");
        }
    }

    #[test]
    fn test_diagnose_refine_high_surprise() {
        let sm = test_sm();
        // High surprise: 12 surprises in 20 hops = 0.6 density
        let results = vec![
            WalkerResult {
                bias: WalkerBias::Curiosity,
                path: vec![1, 2, 3, 4, 5],
                domains_visited: vec!["markets".into(), "psychology".into(), "tech".into(), "science".into()],
                edge_types_used: vec!["related".into(); 4],
                total_weight: 5.0,
                surprises: 12,
                dead_ends: 0,
                edges_traversed: vec![10, 11, 12, 13],
            },
        ];
        let output = WalkOutput {
            total_hops: 20,
            novelty_score: 0.6, // High novelty
            ..test_walk_output()
        };

        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Refine);
        assert!(diagnosis.surprise_density > 0.4);
    }

    #[test]
    fn test_surprise_density_computation() {
        // Verify the math: surprises / total_hops
        let sm = test_sm();
        let results = vec![
            WalkerResult {
                bias: WalkerBias::Curiosity,
                path: vec![1, 2, 3],
                domains_visited: vec!["a".into()],
                edge_types_used: vec!["related".into()],
                total_weight: 1.0,
                surprises: 5,
                dead_ends: 2,
                edges_traversed: vec![1],
            },
        ];
        let output = WalkOutput { total_hops: 25, ..test_walk_output() };
        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        assert!((diagnosis.surprise_density - 0.2).abs() < 0.01); // 5/25 = 0.2
        assert!((diagnosis.dead_end_ratio - 0.08).abs() < 0.01); // 2/25 = 0.08
    }

    // ── apply_critic_adjustments ─────────────────────────────────

    #[test]
    fn test_apply_critic_adjustments_plasticity() {
        let mut sm = test_sm();
        let adj = CriticAdjustment {
            plasticity_delta: 0.2,
            ..Default::default()
        };

        apply_critic_adjustments(&mut sm, &adj);

        assert!((sm.plasticity_gate - 0.7).abs() < 0.01); // 0.5 + 0.2
    }

    #[test]
    fn test_apply_critic_adjustments_clamping() {
        let mut sm = test_sm();
        sm.plasticity_gate = 0.9;

        let adj = CriticAdjustment {
            plasticity_delta: 0.3, // Would go to 1.2, should clamp at 0.95
            ..Default::default()
        };

        apply_critic_adjustments(&mut sm, &adj);

        assert!((sm.plasticity_gate - 0.95).abs() < 0.01); // Clamped
    }

    #[test]
    fn test_apply_critic_adjustments_bias_weights() {
        let mut sm = test_sm();
        let mut bias_deltas = HashMap::new();
        bias_deltas.insert(0, 0.15_f32); // novelty_seeking

        let adj = CriticAdjustment {
            bias_deltas,
            ..Default::default()
        };

        let before = sm.learned_biases[0].novelty_seeking;
        apply_critic_adjustments(&mut sm, &adj);
        let after = sm.learned_biases[0].novelty_seeking;

        assert!(after > before, "Bias 0 novelty_seeking should increase");
        // Cross-domain curiosity should also increase (coupled)
        assert!(sm.learned_biases[0].cross_domain_curiosity > 0.3);
    }

    #[test]
    fn test_apply_critic_adjustments_attention() {
        let mut sm = test_sm();
        let mut attention_deltas = HashMap::new();
        attention_deltas.insert("markets".into(), -0.5_f32);
        attention_deltas.insert("new_domain".into(), 0.3_f32);

        let adj = CriticAdjustment {
            attention_deltas,
            ..Default::default()
        };

        apply_critic_adjustments(&mut sm, &adj);

        assert!(sm.attention_patterns.get("markets").unwrap() < &3.0);
        assert!((sm.attention_patterns.get("new_domain").unwrap() - 0.3).abs() < 0.01);
    }

    // ── parse_critic_response ────────────────────────────────────

    #[test]
    fn test_parse_critic_response_valid_json() {
        let fallback = CriticAdjustment::default();
        let response = r#"{"verdict": "Good session", "plasticity_delta": 0.1, "bias_deltas": {"0": 0.05}, "attention_deltas": {"markets": -0.1}}"#;

        let adj = parse_critic_response(response, &fallback);

        assert_eq!(adj.verdict, "Good session");
        assert!((adj.plasticity_delta - 0.1).abs() < 0.01);
        assert!((adj.bias_deltas.get(&0).unwrap() - 0.05).abs() < 0.01);
        assert!((adj.attention_deltas.get("markets").unwrap() + 0.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_critic_response_markdown_wrapped() {
        let fallback = CriticAdjustment::default();
        let response = "```json\n{\"verdict\": \"ok\", \"plasticity_delta\": 0.0}\n```";

        let adj = parse_critic_response(response, &fallback);

        assert_eq!(adj.verdict, "ok");
    }

    #[test]
    fn test_parse_critic_response_malformed_json() {
        let fallback = CriticAdjustment {
            verdict: "fallback verdict".into(),
            plasticity_delta: 0.05,
            ..Default::default()
        };
        let response = "not json at all";

        let adj = parse_critic_response(response, &fallback);

        assert_eq!(adj.verdict, "fallback verdict");
        assert!((adj.plasticity_delta - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_parse_critic_response_no_braces() {
        let fallback = CriticAdjustment::default();
        let response = "no braces here";

        let adj = parse_critic_response(response, &fallback);

        assert_eq!(adj.plasticity_delta, 0.0);
    }

    // ── record_audience_output ───────────────────────────────────

    #[test]
    fn test_record_audience_output_new_audience() {
        let mut sm = test_sm();
        assert!(sm.audience_model.is_empty());

        record_audience_output(&mut sm, "user:alice", "Hello Alice, let's discuss markets", "markets");

        assert!(sm.audience_model.contains_key("user:alice"));
        let a = &sm.audience_model["user:alice"];
        assert_eq!(a.interaction_count, 1);
        assert_eq!(a.last_message_sent, "Hello Alice, let's discuss markets");
        assert!(a.topics_discussed.contains(&"markets".to_string()));
        assert!(a.estimated_knowledge.get("markets").unwrap() > &0.0);
    }

    #[test]
    fn test_record_audience_output_existing_audience() {
        let mut sm = test_sm();
        record_audience_output(&mut sm, "user:bob", "First message about tech", "tech");
        record_audience_output(&mut sm, "user:bob", "Second message about psychology", "psychology");

        let a = &sm.audience_model["user:bob"];
        assert_eq!(a.interaction_count, 2);
        assert_eq!(a.last_message_sent, "Second message about psychology");
        assert!(a.topics_discussed.contains(&"tech".to_string()));
        assert!(a.topics_discussed.contains(&"psychology".to_string()));
        assert!(a.estimated_knowledge.get("tech").unwrap() > &0.0);
        assert!(a.estimated_knowledge.get("psychology").unwrap() > &0.0);
    }

    #[test]
    fn test_record_audience_output_short_message_no_embedding() {
        let mut sm = test_sm();
        // Message < 10 chars → no embedding stored (but everything else works)
        record_audience_output(&mut sm, "user:short", "Hi", "chat");

        let a = &sm.audience_model["user:short"];
        assert_eq!(a.last_message_sent, "Hi");
        assert!(a.last_context_embedding.is_none()); // Too short to embed
    }

    #[test]
    fn test_record_audience_output_topic_deduplication() {
        let mut sm = test_sm();
        record_audience_output(&mut sm, "user:carol", "Msg 1", "markets");
        record_audience_output(&mut sm, "user:carol", "Msg 2", "markets"); // Same topic

        let a = &sm.audience_model["user:carol"];
        // markets should only appear once
        assert_eq!(a.topics_discussed.iter().filter(|t| *t == "markets").count(), 1);
    }

    // ── record_audience_response ─────────────────────────────────

    #[test]
    fn test_record_audience_response_no_prior_context() {
        let mut sm = test_sm();
        // No prior output recorded → no context embedding → no ToM comparison
        let audience = sm.audience_model.entry("user:new".into())
            .or_insert_with(|| AudienceBeliefs::new("user:new"));
        audience.last_message_sent = "Hello".into();
        // last_context_embedding stays None

        let result = record_audience_response(&mut sm, "user:new", "Hi back");

        assert!(result.is_none()); // No embedding to compare against
        assert_eq!(sm.audience_model["user:new"].last_message_received.as_deref(), Some("Hi back"));
    }

    #[test]
    fn test_record_audience_response_creates_new_model() {
        let mut sm = test_sm();
        // Audience doesn't exist yet
        let result = record_audience_response(&mut sm, "user:newcomer", "Hello Julian!");

        // Should create the model with interaction_count=1
        assert!(sm.audience_model.contains_key("user:newcomer"));
        assert_eq!(sm.audience_model["user:newcomer"].interaction_count, 1);
        assert_eq!(sm.audience_model["user:newcomer"].last_message_received.as_deref(), Some("Hello Julian!"));
        assert!(result.is_none()); // No embedding to compare
    }

    // ── format_audience_context ──────────────────────────────────

    #[test]
    fn test_format_audience_context_unknown_audience() {
        let sm = test_sm();
        let ctx = format_audience_context(&sm, "nobody");
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_format_audience_context_known_audience() {
        let mut sm = test_sm();
        let audience = sm.audience_model.entry("user:alice".into())
            .or_insert_with(|| AudienceBeliefs::new("user:alice"));
        audience.estimated_knowledge.insert("markets".into(), 0.7);
        audience.estimated_knowledge.insert("tech".into(), 0.1); // Below 0.3 threshold
        audience.estimated_interest.insert("psychology".into(), 0.5);
        audience.relationship_valence = 0.65;
        audience.interaction_count = 12;
        audience.last_message_sent = "What do you think about the recent market trends in AI?".into();

        let ctx = format_audience_context(&sm, "user:alice");

        assert!(ctx.contains("user:alice"));
        assert!(ctx.contains("markets")); // Known (0.7 > 0.3)
        assert!(!ctx.contains("tech"));   // Known but below threshold (0.1)
        assert!(ctx.contains("psychology")); // Interested
        assert!(ctx.contains("65%"));      // Relationship valence
        assert!(ctx.contains("12 interactions"));
        assert!(ctx.contains("market trends")); // Last message preview
    }

    #[test]
    fn test_format_audience_context_no_last_message() {
        let mut sm = test_sm();
        let audience = sm.audience_model.entry("user:quiet".into())
            .or_insert_with(|| AudienceBeliefs::new("user:quiet"));
        audience.relationship_valence = 0.0;
        audience.interaction_count = 1;
        // No last_message_sent, no knowledge, no interests

        let ctx = format_audience_context(&sm, "user:quiet");

        assert!(ctx.contains("user:quiet"));
        assert!(ctx.contains("0%"));
        assert!(ctx.contains("1 interactions"));
        assert!(!ctx.contains("You last said")); // No message to preview
    }

    // ── Integrated scenario tests ────────────────────────────────

    #[test]
    fn test_full_critic_pipeline_no_llm() {
        // Simulate a complete critic run without LLM
        let mut sm = test_sm();
        let output = test_walk_output();
        let results = test_walker_results();

        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Normal);

        // Apply algorithmic adjustments
        apply_critic_adjustments(&mut sm, &diagnosis.algorithmic_adjustments);

        // Verify self-model state after adjustments
        assert!(sm.plasticity_gate > 0.0); // Should still be operational
        assert!(!sm.critic_diagnosis.is_empty() || sm.last_critic_run == 0.0);
        // critic_diagnosis is only set by run_critic(), not by diagnose/apply
    }

    #[test]
    fn test_diagnosis_order_matters_energy_first() {
        // Energy check should take priority over everything
        let mut sm = test_sm();
        sm.energy = 0.1; // Critical
        sm.consecutive_repetitions = 5; // Also stuck
        sm.last_prediction_error = core::now() - 5.0; // Also prediction error

        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        // Energy takes priority over BreakLoop
        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::Rest);
    }

    #[test]
    fn test_diagnosis_stuck_before_explore() {
        // Being stuck should take priority over ExploreMore
        let mut sm = test_sm();
        sm.consecutive_repetitions = 4; // Stuck

        let results = vec![
            WalkerResult {
                bias: WalkerBias::Experience,
                path: vec![1],
                domains_visited: vec!["markets".into()],
                edge_types_used: vec!["related".into()],
                total_weight: 0.1,
                surprises: 0,
                dead_ends: 5, // High dead ends, low surprise
                edges_traversed: vec![1],
            },
        ];
        let output = WalkOutput { total_hops: 10, ..test_walk_output() };
        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);

        // Stuck takes priority over ExploreMore
        assert_eq!(diagnosis.primary_diagnosis, DiagnosisLabel::BreakLoop);
    }
}
