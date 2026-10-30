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

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::core::{
    AgreementMetric, AudienceBeliefs, CognitiveMode, CriticRuleProfile, MetacogPhase, Noticing,
    SelfModStage, SelfModel, Signal,
};
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
    pub critic_rule_delta: CriticRuleDelta,
    pub spawn_bias_variants: u8,
    pub escalate_to_llm: bool,
    pub verdict: String,
}

/// Runtime updates for Tier-1 critique thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CriticRuleDelta {
    pub min_energy: Option<f32>,
    pub agreement_threshold_autonomous: Option<f32>,
    pub agreement_threshold_compliant: Option<f32>,
    pub max_attention_saturation: Option<f32>,
    pub wound_guard_threshold: Option<f32>,
    pub wound_confidence_floor: Option<f32>,
    pub phase_order: Option<Vec<MetacogPhase>>,
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
        let mut proposed = draft_proposal(
            &walk.recommended_action,
            &walk.primary_domain,
            walk.agreement_score,
            walk.novelty_score,
            walk.walker_count,
            // Hallucination check reads the operator-selected agreement signal; confidence
            // (and the wound floor that reads it) stays on the legacy score.
            select_walker_agreement(self_model.agreement_metric, walk.agreement_score, walk.domain_agreement_score),
            attempt,
            &prior_critiques,
        );

        // Phases in configured order (shared with the Stage-2 pure replay).
        let phases = self_model.metacog_phase_order.clone();
        let evaluation = run_phases(&mut proposed, &phases, |p| critique(p, self_model));

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
        // Unit tests run the real loop without mutating process-global METRICS (read by
        // walker/dream under parallel tests). Production records exactly as before.
        #[cfg(not(test))]
        crate::metrics::record_metacog_decision(evaluation.approved);

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

/// Phase-1 draft (shared by metacognitive_loop and the Stage-2 replay).
#[allow(clippy::too_many_arguments)]
fn draft_proposal(
    action: &str,
    domain: &str,
    legacy_agreement: f32,
    novelty: f32,
    walker_count: usize,
    walker_agreement: f32,
    attempt: u32,
    prior_critiques: &[String],
) -> ProposedAction {
    ProposedAction {
        action: action.to_string(),
        domain: domain.to_string(),
        reasoning: format!(
            "Agreement {:.0}%, novelty {:.0}%, {} perspectives",
            legacy_agreement * 100.0,
            novelty * 100.0,
            walker_count,
        ),
        confidence: legacy_agreement,
        walker_agreement,
        walker_novelty: novelty,
        walker_context: String::new(), // Filled by caller
        attempt,
        prior_critiques: prior_critiques.to_vec(),
    }
}

/// The agreement value the hallucination check reads (Domain only when the operator chose it).
fn select_walker_agreement(metric: AgreementMetric, legacy: f32, domain: f32) -> f32 {
    if metric.is_domain() { domain } else { legacy }
}

/// Run the configured phases over a draft and return the evaluation.
/// Safety fallback: critique can never be skipped, even if phase order is misconfigured.
fn run_phases(
    proposed: &mut ProposedAction,
    phases: &[MetacogPhase],
    mut critique_fn: impl FnMut(&ProposedAction) -> SelfEvaluation,
) -> SelfEvaluation {
    let mut evaluation: Option<SelfEvaluation> = None;
    for phase in phases {
        match phase {
            MetacogPhase::Draft => {}
            MetacogPhase::ReflectivePause => {
                proposed.confidence = (proposed.confidence * 0.95).clamp(0.0, 1.0);
                proposed.reasoning = format!("{}; pause: rechecked assumptions", proposed.reasoning);
            }
            MetacogPhase::Critique => {
                evaluation = Some(critique_fn(proposed));
            }
        }
    }
    evaluation.unwrap_or_else(|| critique_fn(proposed))
}

/// The slice of self-model state critique() reads, captured for one domain.
#[derive(Debug, Clone)]
struct CritiqueContext {
    mode: CognitiveMode,
    energy: f32,
    /// attention[domain] / total attention; None when total attention is 0.
    attention_share: Option<f32>,
    wound: Option<f32>,
}

impl CritiqueContext {
    fn of(sm: &SelfModel, domain: &str) -> Self {
        let domain_saturation = sm.attention_patterns.get(domain).copied().unwrap_or(0.0);
        let total_attention: f32 = sm.attention_patterns.values().sum();
        Self {
            mode: sm.mode.clone(),
            energy: sm.energy,
            attention_share: (total_attention > 0.0).then(|| domain_saturation / total_attention),
            wound: sm.wounds.get(domain).copied(),
        }
    }
}

/// Tier 1 critic: evaluate a proposed action against self-model state.
/// Deterministic, no LLM. Checks safety, hallucination, efficiency, wounds.
fn critique(proposed: &ProposedAction, sm: &SelfModel) -> SelfEvaluation {
    critique_with(proposed, &CritiqueContext::of(sm, &proposed.domain), &sm.critic_rules)
}

/// Pure core of [`critique`]: reads only the captured context and the given rules.
fn critique_with(proposed: &ProposedAction, ctx: &CritiqueContext, rules: &CriticRuleProfile) -> SelfEvaluation {
    let mut approved = true;
    let mut critiques: Vec<String> = Vec::new();

    let safety = ctx.energy > rules.min_energy;
    if !safety {
        approved = false;
        critiques.push(format!(
            "Energy too low ({:.0}%) — rest instead (min {:.0}%)",
            ctx.energy * 100.0,
            rules.min_energy * 100.0
        ));
    }

    let agreement_threshold = match ctx.mode {
        CognitiveMode::Autonomous => rules.agreement_threshold_autonomous,
        CognitiveMode::Compliant => rules.agreement_threshold_compliant,
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
    if ctx.mode == CognitiveMode::Autonomous {
        efficiency = match ctx.attention_share {
            Some(share) => share < rules.max_attention_saturation,
            None => true,
        };
        if !efficiency {
            approved = false;
            critiques.push(format!(
                "{} already dominates attention ({:.0}%) — explore something else",
                proposed.domain, ctx.attention_share.unwrap_or(0.0) * 100.0
            ));
        }

        if let Some(wound) = ctx.wound
            && wound > rules.wound_guard_threshold
                && proposed.confidence < rules.wound_confidence_floor
            {
                approved = false;
                critiques.push(format!(
                    "Wound in {} ({:.0}%) + low confidence — proceed cautiously",
                    proposed.domain, wound * 100.0
                ));
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
                critic_rule_delta: CriticRuleDelta {
                    min_energy: Some((sm.critic_rules.min_energy + 0.02).min(0.4)),
                    ..Default::default()
                },
                spawn_bias_variants: 0,
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
                critic_rule_delta: CriticRuleDelta {
                    agreement_threshold_autonomous: Some((sm.critic_rules.agreement_threshold_autonomous - 0.02).max(0.05)),
                    max_attention_saturation: Some((sm.critic_rules.max_attention_saturation - 0.05).max(0.45)),
                    phase_order: Some(vec![MetacogPhase::Draft, MetacogPhase::ReflectivePause, MetacogPhase::Critique]),
                    ..Default::default()
                },
                spawn_bias_variants: 1,
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
                critic_rule_delta: CriticRuleDelta {
                    agreement_threshold_autonomous: Some((sm.critic_rules.agreement_threshold_autonomous - 0.01).max(0.05)),
                    phase_order: Some(vec![MetacogPhase::Draft, MetacogPhase::ReflectivePause, MetacogPhase::Critique]),
                    ..Default::default()
                },
                spawn_bias_variants: 1,
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
                critic_rule_delta: CriticRuleDelta {
                    wound_guard_threshold: Some((sm.critic_rules.wound_guard_threshold - 0.05).max(0.2)),
                    phase_order: Some(vec![MetacogPhase::Draft, MetacogPhase::Critique]),
                    ..Default::default()
                },
                spawn_bias_variants: 0,
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
    // Top-k the domain list: it grows with the graph and would otherwise
    // dump an unbounded subgraph into the prompt. The unique count is kept
    // separately, so the signal survives the cap.
    const MAX_DOMAINS_IN_PROMPT: usize = 12;
    let domains_str = if summary.domains_visited.len() > MAX_DOMAINS_IN_PROMPT {
        format!(
            "{}, …(+{} more)",
            summary.domains_visited[..MAX_DOMAINS_IN_PROMPT].join(", "),
            summary.domains_visited.len() - MAX_DOMAINS_IN_PROMPT
        )
    } else {
        summary.domains_visited.join(", ")
    };
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

            let critic_rule_delta = v.get("critic_rule_delta")
                .and_then(|x| serde_json::from_value::<CriticRuleDelta>(x.clone()).ok())
                .unwrap_or_else(|| fallback.critic_rule_delta.clone());

            let spawn_bias_variants = v.get("spawn_bias_variants")
                .and_then(|n| n.as_u64())
                .map(|n| n.min(u8::MAX as u64) as u8)
                .unwrap_or(fallback.spawn_bias_variants);

            CriticAdjustment { plasticity_delta, bias_deltas, attention_deltas,
                critic_rule_delta, spawn_bias_variants, escalate_to_llm: false, verdict }
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

    route_rule_delta(sm, &adjustment.critic_rule_delta, "llm_critic");
    spawn_bias_variants(sm, adjustment.spawn_bias_variants);

    let signal = Signal::new("critic_llm_adjustment", &format!(
        "LLM Critic: {} (plasticity {:.2})",
        adjustment.verdict, sm.plasticity_gate,
    )).with_intensity(0.3);
    crate::core::process(signal, sm);

    adjustment
}

/// Build the LLM prompt for the critic from a diagnosis and summary.
pub fn build_critic_llm_prompt(diagnosis: &CriticDiagnosis, summary: &WalkSessionSummary) -> (String, String) {
    let system = "You are Julian's metacognitive critic. Evaluate this walk session. Output ONLY a JSON object with: verdict (string), plasticity_delta (float -0.3 to 0.3), bias_deltas (object mapping index string to float), attention_deltas (object mapping domain to float), spawn_bias_variants (int 0..2), critic_rule_delta (object with optional keys: min_energy, agreement_threshold_autonomous, agreement_threshold_compliant, max_attention_saturation, wound_guard_threshold, wound_confidence_floor, phase_order where phase_order is an array of [draft, reflective_pause, critique]). Be concise.".to_string();
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

    route_rule_delta(sm, &adj.critic_rule_delta, "algorithmic_critic");
    spawn_bias_variants(sm, adj.spawn_bias_variants);

    let signal = Signal::new("critic_adjustment", &format!(
        "Critic: {} (plasticity {:.2}, {} bias changes, {} attn changes)",
        adj.verdict, sm.plasticity_gate, adj.bias_deltas.len(), adj.attention_deltas.len(),
    )).with_intensity(0.3);
    crate::core::process(signal, sm);
}

fn apply_rule_delta(sm: &mut SelfModel, delta: &CriticRuleDelta) {
    apply_rule_delta_parts(&mut sm.critic_rules, &mut sm.metacog_phase_order, delta);
}

/// The clamps of [`apply_rule_delta`], on bare parts (so trials can apply to a snapshot).
fn apply_rule_delta_parts(rules: &mut CriticRuleProfile, phase_order: &mut Vec<MetacogPhase>, delta: &CriticRuleDelta) {
    if let Some(v) = delta.min_energy {
        rules.min_energy = v.clamp(0.05, 0.6);
    }
    if let Some(v) = delta.agreement_threshold_autonomous {
        rules.agreement_threshold_autonomous = v.clamp(0.05, 0.9);
    }
    if let Some(v) = delta.agreement_threshold_compliant {
        rules.agreement_threshold_compliant = v.clamp(0.05, 0.95);
    }
    if let Some(v) = delta.max_attention_saturation {
        rules.max_attention_saturation = v.clamp(0.3, 0.95);
    }
    if let Some(v) = delta.wound_guard_threshold {
        rules.wound_guard_threshold = v.clamp(0.1, 0.95);
    }
    if let Some(v) = delta.wound_confidence_floor {
        rules.wound_confidence_floor = v.clamp(0.1, 0.99);
    }
    if let Some(phases) = &delta.phase_order {
        // Keep a safe floor: Draft + Critique must be present.
        let has_draft = phases.iter().any(|p| p == &MetacogPhase::Draft);
        let has_critique = phases.iter().any(|p| p == &MetacogPhase::Critique);
        if has_draft && has_critique {
            *phase_order = phases.clone();
        }
    }
}

// ── Stage 1: quality-diversity selection over the learned-bias pool ──────────────
// PROTOCOL-self-selection.md. Ships gated behind SelfModStage (default Observability =
// compute-only). The plan is computed by a pure, RNG-free `plan_selection`, so the
// Observability would-cull/would-breed logs faithfully predict SelectionLive behaviour.
const MAX_LEARNED_BIASES: usize = 16; // pool cap (shared with spawn_bias_variants)
const SELECTION_FLOOR: usize = 6; // pool never shrinks below (matches the default-6 pool)
const SELECTION_MIN_WALKS: u32 = 8; // cull eligibility — under-sampled profiles are immune
const SELECTION_K: f32 = 1.0; // CI half-width multiplier (heuristic safety margin, retune)
const ELITE_PER_NICHE: usize = 1; // a niche keeps its top-1 by fitness
const SELECT_EVERY: u32 = 8; // act only every Nth cycle (let profiles accrue walks)

/// The one perturbation, shared by spawn_bias_variants (critic path) and select_biases
/// (breeding). Resets the child's scorecard → born under-sampled → cull-immune until it
/// earns MIN_WALKS; sessions_learned=0 so it re-learns from scratch.
fn perturb_child(parent: &crate::graph::LearnedBias) -> crate::graph::LearnedBias {
    let mut child = parent.clone();
    child.sessions_learned = 0;
    child.scorecard = crate::graph::ProfileFitness::default();
    child.novelty_seeking = (child.novelty_seeking + 0.08).clamp(0.05, 0.95);
    child.cross_domain_curiosity = (child.cross_domain_curiosity + 0.06).clamp(0.05, 0.95);
    child.experience_reliance = (child.experience_reliance - 0.05).clamp(0.05, 0.95);
    child
}

fn spawn_bias_variants(sm: &mut SelfModel, count: u8) {
    for _ in 0..count {
        if sm.learned_biases.len() >= MAX_LEARNED_BIASES {
            break;
        }
        if sm.learned_biases.is_empty() {
            sm.learned_biases.push(crate::graph::LearnedBias::default());
        }
        let parent_idx = (sm.learned_bias_rotation as usize) % sm.learned_biases.len();
        let child = perturb_child(&sm.learned_biases[parent_idx]);
        sm.learned_biases.push(child);
    }
}

/// What a selection cycle would do — computed purely from the pool (no &mut, no RNG) so
/// Observability and SelectionLive produce IDENTICAL plans (the keystone safety property).
pub struct SelectionPlan {
    pub acted: bool, // guards passed (pool above FLOOR)
    pub cull: Option<CullPlan>, // at most one (CULL_PER_CYCLE = 1)
    pub breed_parent: Option<crate::graph::LearnedBias>, // elite clone to perturb (replaces a cull)
}

pub struct CullPlan {
    pub idx: usize,
    pub niche: (u8, u8),
    pub cand_fitness: f32,
    pub cand_ucb: f32,
    pub incumbent_idx: usize,
    pub incumbent_fitness: f32,
    pub incumbent_lcb: f32,
}

fn plan_selection(sm: &SelfModel) -> SelectionPlan {
    let biases = &sm.learned_biases;
    if biases.len() <= SELECTION_FLOOR {
        return SelectionPlan { acted: false, cull: None, breed_parent: None };
    }

    // Niche-assign walked profiles; BTreeMap → deterministic iteration order (invariant #8).
    let mut niches: std::collections::BTreeMap<(u8, u8), Vec<usize>> = std::collections::BTreeMap::new();
    for (i, b) in biases.iter().enumerate() {
        if let Some(key) = crate::graph::behavioural_niche(&b.scorecard) {
            niches.entry(key).or_default().push(i); // members ascending by index
        }
    }

    let mut best: Option<CullPlan> = None;
    let mut best_incumbent: Option<usize> = None; // global elite for breeding
    for (&niche, members) in &niches {
        // niche incumbent: max fitness, tie → lowest index (strict-greater keeps the first).
        let mut inc = members[0];
        for &m in &members[1..] {
            if biases[m].scorecard.fitness > biases[inc].scorecard.fitness {
                inc = m;
            }
        }
        match best_incumbent {
            Some(bi) if biases[inc].scorecard.fitness <= biases[bi].scorecard.fitness => {}
            _ => best_incumbent = Some(inc),
        }
        if members.len() <= ELITE_PER_NICHE {
            continue; // not over-full
        }
        let inc_walks = biases[inc].scorecard.walks;
        let inc_lcb = biases[inc].scorecard.lcb(SELECTION_K);
        for &m in members {
            if m == inc {
                continue;
            }
            let cand = &biases[m].scorecard;
            if cand.walks < SELECTION_MIN_WALKS || inc_walks < SELECTION_MIN_WALKS {
                continue; // under-sampled either side → never cull
            }
            let cand_ucb = cand.ucb(SELECTION_K);
            if cand_ucb < inc_lcb {
                // confidently dominated → eligible; keep the lowest-ucb candidate overall.
                if best.as_ref().is_none_or(|b| cand_ucb < b.cand_ucb) {
                    best = Some(CullPlan {
                        idx: m,
                        niche,
                        cand_fitness: cand.fitness,
                        cand_ucb,
                        incumbent_idx: inc,
                        incumbent_fitness: biases[inc].scorecard.fitness,
                        incumbent_lcb: inc_lcb,
                    });
                }
            }
        }
    }

    // A single cull from len>FLOOR leaves len-1>=FLOOR (coverage safe). Breed (replacement)
    // only when a cull is planned, from the global best elite — keeps pool size constant.
    // DELIBERATE deviation from PROTOCOL §#1 step 4 (proactive empty-/under-represented-niche
    // breeding): replacement-only is the safe Stage-1 interim (no FLOOR drift, no aimless
    // growth); empty-niche breeding from a niche-local parent is deferred to Follow-up.
    let breed_parent = best.as_ref().and_then(|_| best_incumbent.map(|i| biases[i].clone()));
    SelectionPlan { acted: true, cull: best, breed_parent }
}

/// Stage-1 selection: cadence- and stage-gated quality-diversity over the bias pool.
/// Observability logs the plan and commits nothing (only the cadence counter advances);
/// SelectionLive+ applies cull-then-breed. Called once per dream cycle, Autonomous-only.
pub fn select_biases(sm: &mut SelfModel) {
    if sm.mode != CognitiveMode::Autonomous {
        return; // frozen in Compliant (the dream loop already excludes it; defensive)
    }
    // Cadence: the ONLY mutation permitted at Observability.
    sm.selection_cycle_counter = sm.selection_cycle_counter.wrapping_add(1);
    if !sm.selection_cycle_counter.is_multiple_of(SELECT_EVERY) {
        return;
    }

    let plan = plan_selection(sm);
    if !plan.acted {
        return;
    }

    let live = sm.self_mod_stage.allows_selection();
    match &plan.cull {
        Some(c) => tracing::info!(
            "[select-biases] WOULD-CULL profile#{} niche={:?} fit={:.3} ucb={:.3} < incumbent#{} (fit={:.3}) lcb={:.3} committed={}",
            c.idx, c.niche, c.cand_fitness, c.cand_ucb, c.incumbent_idx, c.incumbent_fitness, c.incumbent_lcb, live
        ),
        None => tracing::info!("[select-biases] no confident cull this cycle (pool len={})", sm.learned_biases.len()),
    }

    if !live {
        return; // Observability: logged, commit nothing
    }

    // SelectionLive+: apply cull-then-breed (the bred parent is a clone, immune to index shift).
    if let Some(c) = &plan.cull {
        let idx = c.idx;
        sm.learned_biases.remove(idx);
        if let Some(parent) = &plan.breed_parent
            && sm.learned_biases.len() < MAX_LEARNED_BIASES {
                let child = perturb_child(parent);
                tracing::info!(
                    "[select-biases] CULLED #{}; bred replacement from incumbent (novelty_seeking={:.3})",
                    idx, child.novelty_seeking
                );
                sm.learned_biases.push(child);
            }
        // Re-normalise rotation so walker credit-attribution stays in-bounds after the shift.
        if !sm.learned_biases.is_empty() {
            sm.learned_bias_rotation %= sm.learned_biases.len() as u64;
        }
        // Signal the pool mutated → a walk that snapshotted the old pool skips stale credit.
        sm.pool_generation = sm.pool_generation.wrapping_add(1);
    }
}

// ── Stage 2: evaluate-then-keep over rule changes (PROTOCOL-self-selection §#2) ──────────
// A CriticRuleDelta is trialled offline in the dream loop by replaying recent walks under the
// current and proposed rule configs, scored by calibration against the acyclic intrinsic
// fitness (approval held out). It reaches live config ONLY via an accepted trial, and only when
// trials_live(): Autonomous + RuleTrialsLive + agreement_metric == Domain. Below that, deltas
// keep applying directly (flag-off parity) and are trialled in shadow (log only).
//
// Every constant below is a placeholder for a parameter the protocol leaves open (see the PR's
// "Unresolved design parameters"); retune from Observability logs before promoting.

/// Pending-proposal FIFO cap; oldest dropped when full. [GAP: coalescing unspecified]
pub const PENDING_CAP: usize = 8;
/// Rollback ring depth ("last N"). [GAP]
pub const ROLLBACK_N: usize = 5;
/// Replay buffer size M (doc §Open questions: "Start at M=50").
pub const REPLAY_M: usize = 50;
/// Rule-trial history cap. [GAP]
pub const TRIAL_HISTORY_CAP: usize = 50;

/// Tunables of the trial score and guards. All placeholders. [GAP]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RuleTrialConfig {
    /// Accept iff calib(proposed) > calib(current) + margin.
    pub margin: f32,
    /// Over-abort penalty weight in replay_calibration.
    pub lambda: f32,
    /// Overall abort rate above which the over-abort penalty applies.
    pub abort_cap: f32,
    /// Attempts per replayed walk. With a frozen context approval does not depend on the
    /// attempt number, so any value >= 1 gives the same decision.
    pub replay_max_attempts: u32,
    /// Regression window T in dream cycles (placeholder = SELECT_EVERY).
    pub regression_cycles: u32,
}

pub const RULE_TRIAL_CONFIG: RuleTrialConfig = RuleTrialConfig {
    margin: 0.05,
    lambda: 1.0,
    abort_cap: 0.5,
    replay_max_attempts: 3,
    regression_cycles: SELECT_EVERY,
};

/// The rule config a trial compares and a rollback restores (critic_rules + phase order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleConfigSnapshot {
    pub critic_rules: CriticRuleProfile,
    pub phase_order: Vec<MetacogPhase>,
}

impl RuleConfigSnapshot {
    pub fn of(sm: &SelfModel) -> Self {
        Self { critic_rules: sm.critic_rules.clone(), phase_order: sm.metacog_phase_order.clone() }
    }

    /// Exact, field-for-field restore.
    pub fn restore_into(&self, sm: &mut SelfModel) {
        sm.critic_rules = self.critic_rules.clone();
        sm.metacog_phase_order = self.phase_order.clone();
    }

    /// This config with a delta applied using the same clamps as the live path.
    pub fn with_delta(&self, delta: &CriticRuleDelta) -> Self {
        let mut out = self.clone();
        apply_rule_delta_parts(&mut out.critic_rules, &mut out.phase_order, delta);
        out
    }
}

/// One of the seven rule-config fields (the six CriticRuleProfile thresholds + phase order).
/// Field-level rollback works per field so operator edits made after an accept survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleField {
    MinEnergy,
    AgreementThresholdAutonomous,
    AgreementThresholdCompliant,
    MaxAttentionSaturation,
    WoundGuardThreshold,
    WoundConfidenceFloor,
    PhaseOrder,
}

impl RuleField {
    const ALL: [RuleField; 7] = [
        RuleField::MinEnergy,
        RuleField::AgreementThresholdAutonomous,
        RuleField::AgreementThresholdCompliant,
        RuleField::MaxAttentionSaturation,
        RuleField::WoundGuardThreshold,
        RuleField::WoundConfidenceFloor,
        RuleField::PhaseOrder,
    ];

    fn name(self) -> &'static str {
        match self {
            RuleField::MinEnergy => "min_energy",
            RuleField::AgreementThresholdAutonomous => "agreement_threshold_autonomous",
            RuleField::AgreementThresholdCompliant => "agreement_threshold_compliant",
            RuleField::MaxAttentionSaturation => "max_attention_saturation",
            RuleField::WoundGuardThreshold => "wound_guard_threshold",
            RuleField::WoundConfidenceFloor => "wound_confidence_floor",
            RuleField::PhaseOrder => "phase_order",
        }
    }

    /// The f32 threshold this field names; None for PhaseOrder.
    fn threshold(self, r: &CriticRuleProfile) -> Option<f32> {
        match self {
            RuleField::MinEnergy => Some(r.min_energy),
            RuleField::AgreementThresholdAutonomous => Some(r.agreement_threshold_autonomous),
            RuleField::AgreementThresholdCompliant => Some(r.agreement_threshold_compliant),
            RuleField::MaxAttentionSaturation => Some(r.max_attention_saturation),
            RuleField::WoundGuardThreshold => Some(r.wound_guard_threshold),
            RuleField::WoundConfidenceFloor => Some(r.wound_confidence_floor),
            RuleField::PhaseOrder => None,
        }
    }

    /// Is this field equal in the two configs?
    fn same(self, a_rules: &CriticRuleProfile, a_phases: &[MetacogPhase], b_rules: &CriticRuleProfile, b_phases: &[MetacogPhase]) -> bool {
        match self {
            RuleField::PhaseOrder => a_phases == b_phases,
            _ => self.threshold(a_rules) == self.threshold(b_rules),
        }
    }

    /// Copy this field from a snapshot into the live config.
    fn copy_into(self, from: &RuleConfigSnapshot, sm: &mut SelfModel) {
        let (src, dst) = (&from.critic_rules, &mut sm.critic_rules);
        match self {
            RuleField::MinEnergy => dst.min_energy = src.min_energy,
            RuleField::AgreementThresholdAutonomous => dst.agreement_threshold_autonomous = src.agreement_threshold_autonomous,
            RuleField::AgreementThresholdCompliant => dst.agreement_threshold_compliant = src.agreement_threshold_compliant,
            RuleField::MaxAttentionSaturation => dst.max_attention_saturation = src.max_attention_saturation,
            RuleField::WoundGuardThreshold => dst.wound_guard_threshold = src.wound_guard_threshold,
            RuleField::WoundConfidenceFloor => dst.wound_confidence_floor = src.wound_confidence_floor,
            RuleField::PhaseOrder => sm.metacog_phase_order = from.phase_order.clone(),
        }
    }
}

/// One accepted trial, as a rollback record: the config just before and just after the accept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackEntry {
    pub before: RuleConfigSnapshot,
    pub after: RuleConfigSnapshot,
    pub ts: f64,
    pub reason: String,
}

impl RollbackEntry {
    /// Fields that differ between before and after — computed by comparing the snapshots, NOT
    /// from which delta fields were Some (a Some field can be a no-op: an invalid phase_order
    /// is ignored, a clamp can land on the same value).
    fn changed(&self) -> Vec<RuleField> {
        RuleField::ALL
            .into_iter()
            .filter(|f| !f.same(&self.before.critic_rules, &self.before.phase_order, &self.after.critic_rules, &self.after.phase_order))
            .collect()
    }

    pub fn changed_field_names(&self) -> Vec<&'static str> {
        self.changed().into_iter().map(RuleField::name).collect()
    }
}

/// Result of a field-level rollback.
#[derive(Debug, Clone, Serialize)]
pub struct RollbackOutcome {
    pub entry: RollbackEntry,
    /// Changed fields whose live value still equalled the accept's value → restored to before.
    pub restored: Vec<&'static str>,
    /// Changed fields edited since the accept (operator or otherwise) → left alone.
    pub skipped: Vec<&'static str>,
}

impl CriticRuleDelta {
    /// True when no field is set (nothing to apply or trial).
    pub fn is_empty(&self) -> bool {
        self.min_energy.is_none()
            && self.agreement_threshold_autonomous.is_none()
            && self.agreement_threshold_compliant.is_none()
            && self.max_attention_saturation.is_none()
            && self.wound_guard_threshold.is_none()
            && self.wound_confidence_floor.is_none()
            && self.phase_order.is_none()
    }
}

/// A rule delta awaiting an offline trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRuleProposal {
    pub delta: CriticRuleDelta,
    pub reason: String,
    pub created_at: f64,
    /// The rule config when the proposal was made.
    pub base: RuleConfigSnapshot,
    /// True when the delta was already applied by the legacy direct path (below Stage 2), so
    /// it is only ever trialled in shadow, even if the stage is promoted before its trial.
    #[serde(default)]
    pub applied_directly: bool,
}

/// One replayable walk: exactly what critique() reads, captured at walk time, plus the walk's
/// intrinsic fitness. The doc's "WalkOutputSummary". Deliberately has NO approval input
/// (acyclicity: the trial objective must never see metacog's own output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEntry {
    pub recommended_action: String,
    pub primary_domain: String,
    pub agreement_score: f32,
    pub domain_agreement_score: f32,
    pub novelty_score: f32,
    pub walker_count: usize,
    pub mode: CognitiveMode,
    pub energy: f32,
    /// attention[primary_domain] / total attention at walk time; None when total is 0.
    pub attention_share: Option<f32>,
    pub wound: Option<f32>,
    /// Session-mean per-walker intrinsic fitness (graph::walk_intrinsic).
    pub fitness: f32,
}

impl ReplayEntry {
    pub fn from_walk(walk: &WalkOutput, sm: &SelfModel, fitness: f32) -> Self {
        let ctx = CritiqueContext::of(sm, &walk.primary_domain);
        Self {
            recommended_action: walk.recommended_action.clone(),
            primary_domain: walk.primary_domain.clone(),
            agreement_score: walk.agreement_score,
            domain_agreement_score: walk.domain_agreement_score,
            novelty_score: walk.novelty_score,
            walker_count: walk.walker_count,
            mode: ctx.mode,
            energy: ctx.energy,
            attention_share: ctx.attention_share,
            wound: ctx.wound,
            fitness,
        }
    }

    fn context(&self) -> CritiqueContext {
        CritiqueContext {
            mode: self.mode.clone(),
            energy: self.energy,
            attention_share: self.attention_share,
            wound: self.wound,
        }
    }
}

/// Outcome of one rule trial (the doc's "EvaluationResult").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleTrialRecord {
    pub ts: f64,
    pub delta: CriticRuleDelta,
    pub calib_current: f32,
    pub calib_proposed: f32,
    pub margin: f32,
    /// Live: applied. Shadow: would-accept (nothing applied).
    pub accepted: bool,
    pub shadow: bool,
    /// Metric the decision pair (calib_current/calib_proposed) was scored under. Shadow trials
    /// always score under Domain (Stage 2.1); live trials under the live metric (Domain).
    /// Default Legacy: records written before Stage 2.1 were scored under the live metric,
    /// which was Legacy everywhere.
    #[serde(default)]
    pub scored_with: AgreementMetric,
    /// Shadow trials only, when the live metric is not Domain: the same pair scored under the
    /// live metric (informational; not used for the decision).
    #[serde(default)]
    pub calib_current_live: Option<f32>,
    #[serde(default)]
    pub calib_proposed_live: Option<f32>,
}

/// Live-regression guard armed by an accepted trial (doc §"Rollback has two triggers").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionWatch {
    pub baseline_fitness: f32,
    pub cycles_below: u32,
    pub cycles_observed: u32,
}

/// Rule trials may APPLY only when all three hold (doc §Compliance: agreement-soundness gate).
pub fn trials_live(sm: &SelfModel) -> bool {
    sm.mode == CognitiveMode::Autonomous
        && sm.self_mod_stage == SelfModStage::RuleTrialsLive
        && sm.agreement_metric.is_domain()
}

/// Push a replay entry (Autonomous only), keeping the buffer at REPLAY_M.
pub fn record_replay_entry(sm: &mut SelfModel, entry: ReplayEntry) {
    if sm.mode != CognitiveMode::Autonomous {
        return;
    }
    let was = sm.recent_walk_fitness.len();
    sm.recent_walk_fitness.push_back(entry);
    while sm.recent_walk_fitness.len() > REPLAY_M {
        sm.recent_walk_fitness.pop_front();
    }
    // One INFO line on the transition to a full buffer (per-cycle skips are DEBUG).
    if was + 1 == REPLAY_M {
        tracing::info!("[selection] replay buffer full ({} walks); rule trials can run", REPLAY_M);
    }
}

fn enqueue_rule_proposal(sm: &mut SelfModel, proposal: PendingRuleProposal) {
    sm.pending_rule_proposals.push_back(proposal);
    while sm.pending_rule_proposals.len() > PENDING_CAP {
        if let Some(dropped) = sm.pending_rule_proposals.pop_front() {
            tracing::info!(
                "[selection] pending rule queue full ({}); dropped oldest ({})",
                PENDING_CAP, dropped.reason
            );
        }
    }
}

/// Route a critic rule delta. At Stage 2 (trials_live) it is ENQUEUED ONLY — never applied
/// directly. Below that it applies exactly as before (flag-off parity) and, in Autonomous mode,
/// a shadow proposal is enqueued so the dream loop can log would-accept. Compliant: apply as
/// before, enqueue nothing. Empty deltas are a no-op either way.
pub fn route_rule_delta(sm: &mut SelfModel, delta: &CriticRuleDelta, reason: &str) {
    if delta.is_empty() {
        return;
    }
    let base = RuleConfigSnapshot::of(sm);
    if trials_live(sm) {
        tracing::info!("[selection] rule delta queued for trial ({}): {:?}", reason, delta);
        enqueue_rule_proposal(sm, PendingRuleProposal {
            delta: delta.clone(),
            reason: reason.to_string(),
            created_at: crate::core::now(),
            base,
            applied_directly: false,
        });
        return;
    }
    apply_rule_delta(sm, delta);
    if sm.mode == CognitiveMode::Autonomous {
        enqueue_rule_proposal(sm, PendingRuleProposal {
            delta: delta.clone(),
            reason: reason.to_string(),
            created_at: crate::core::now(),
            base,
            applied_directly: true,
        });
    }
}

/// Pure replay of one walk's metacognitive decision under a rule config: the same draft,
/// phase loop and critique logic as metacognitive_loop, with the self-model context frozen at
/// walk time. No self-model mutation, no metrics, no RNG.
pub fn replay_decide(entry: &ReplayEntry, cfg: &RuleConfigSnapshot, metric: AgreementMetric, max_attempts: u32) -> bool {
    let ctx = entry.context();
    let walker_agreement = select_walker_agreement(metric, entry.agreement_score, entry.domain_agreement_score);
    let mut prior_critiques: Vec<String> = Vec::new();
    for attempt in 1..=max_attempts {
        let mut proposed = draft_proposal(
            &entry.recommended_action,
            &entry.primary_domain,
            entry.agreement_score,
            entry.novelty_score,
            entry.walker_count,
            walker_agreement,
            attempt,
            &prior_critiques,
        );
        let evaluation = run_phases(&mut proposed, &cfg.phase_order, |p| critique_with(p, &ctx, &cfg.critic_rules));
        if evaluation.approved {
            return true;
        }
        prior_critiques.push(evaluation.critique);
    }
    false
}

/// Calibration of a rule config over the replay buffer. Placeholder formula [GAP]: split the
/// buffer at the median fitness (stable sort, low half = first n/2); score = approve rate on
/// the high half + abort rate on the low half − lambda·max(0, overall abort rate − abort_cap).
/// Uniform weights (recency weighting unspecified [GAP]). Deterministic: same buffer, no RNG.
pub fn replay_calibration(
    cfg: &RuleConfigSnapshot,
    buffer: &VecDeque<ReplayEntry>,
    metric: AgreementMetric,
    tc: &RuleTrialConfig,
) -> f32 {
    let n = buffer.len();
    if n == 0 {
        return 0.0;
    }
    let approved: Vec<bool> = buffer
        .iter()
        .map(|e| replay_decide(e, cfg, metric, tc.replay_max_attempts))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| buffer[a].fitness.total_cmp(&buffer[b].fitness).then(a.cmp(&b)));
    let (low, high) = order.split_at(n / 2);
    let rate = |set: &[usize], want: bool| -> f32 {
        if set.is_empty() {
            0.0
        } else {
            set.iter().filter(|&&i| approved[i] == want).count() as f32 / set.len() as f32
        }
    };
    let abort_all = approved.iter().filter(|a| !**a).count() as f32 / n as f32;
    rate(high, true) + rate(low, false) - tc.lambda * (abort_all - tc.abort_cap).max(0.0)
}

fn push_trial_record(sm: &mut SelfModel, record: RuleTrialRecord) {
    sm.rule_trial_history.push_back(record);
    while sm.rule_trial_history.len() > TRIAL_HISTORY_CAP {
        sm.rule_trial_history.pop_front();
    }
}

/// Mean intrinsic fitness over profiles that have walked (the Stage 0/1 pool). This is the
/// placeholder for the doc's "live fitness EWMA" [GAP]. None when no profile has walked.
pub fn live_pool_fitness(sm: &SelfModel) -> Option<f32> {
    let walked: Vec<f32> = sm
        .learned_biases
        .iter()
        .filter(|b| b.scorecard.walks >= 1)
        .map(|b| b.scorecard.fitness)
        .collect();
    (!walked.is_empty()).then(|| walked.iter().sum::<f32>() / walked.len() as f32)
}

/// Offline trial of the oldest pending proposal — at most one per dream cycle.
/// Shadow (anything below trials_live, or a delta already applied directly): compare
/// calibration(base) vs calibration(base + delta), log would-accept/reject, commit nothing
/// else. Live: accept iff calib(current + delta) > calib(current) + margin; on accept push
/// current to the rollback ring, apply, and arm the regression watch.
pub fn trial_pending_rules(sm: &mut SelfModel) {
    let tc = RULE_TRIAL_CONFIG;
    if sm.mode != CognitiveMode::Autonomous {
        tracing::debug!("[selection] rule trial skipped: not Autonomous");
        return;
    }
    if sm.pending_rule_proposals.is_empty() {
        tracing::debug!("[selection] rule trial skipped: no pending proposals");
        return;
    }
    if sm.recent_walk_fitness.len() < REPLAY_M {
        tracing::debug!(
            "[selection] rule trial skipped: replay buffer {}/{} ({} pending)",
            sm.recent_walk_fitness.len(), REPLAY_M, sm.pending_rule_proposals.len()
        );
        return;
    }
    // A LIVE trial may only run when a regression baseline exists: never accept a delta that
    // the live-regression guard could not watch. The proposal stays at the front; no record
    // is written (so the duplicate-rejection check is not poisoned). Shadow trials need none.
    if let Some(front) = sm.pending_rule_proposals.front()
        && !front.applied_directly
        && trials_live(sm)
        && live_pool_fitness(sm).is_none()
    {
        tracing::debug!(
            "[selection] rule trial skipped: no live fitness baseline ({} pending)",
            sm.pending_rule_proposals.len()
        );
        return;
    }
    let Some(proposal) = sm.pending_rule_proposals.pop_front() else { return };
    let shadow = proposal.applied_directly || !trials_live(sm);

    // Shadow trials score under Domain whatever the live metric is, so would-accept is
    // informative at the defaults (the replay is pure — no live effect). Live trials require
    // Domain via trials_live, so they score under the live metric, which is Domain.
    let live_metric = sm.agreement_metric;
    let scored_with = if shadow { AgreementMetric::Domain } else { live_metric };

    // Never re-trial an identical delta that was already rejected under the same scoring
    // (exact equality only). A Legacy-scored rejection does not block a Domain-scored trial.
    if sm
        .rule_trial_history
        .iter()
        .any(|r| !r.accepted && r.shadow == shadow && r.scored_with == scored_with && r.delta == proposal.delta)
    {
        tracing::info!(
            "[selection] rule trial skipped: identical delta already rejected ({})",
            proposal.reason
        );
        return;
    }

    let current = if shadow { proposal.base.clone() } else { RuleConfigSnapshot::of(sm) };
    let proposed = current.with_delta(&proposal.delta);
    let calib_current = replay_calibration(&current, &sm.recent_walk_fitness, scored_with, &tc);
    let calib_proposed = replay_calibration(&proposed, &sm.recent_walk_fitness, scored_with, &tc);
    let accepted = calib_proposed > calib_current + tc.margin;
    // Shadow + non-Domain live metric: also record the pair under the live metric.
    let (calib_current_live, calib_proposed_live) = if shadow && live_metric != scored_with {
        (
            Some(replay_calibration(&current, &sm.recent_walk_fitness, live_metric, &tc)),
            Some(replay_calibration(&proposed, &sm.recent_walk_fitness, live_metric, &tc)),
        )
    } else {
        (None, None)
    };

    // Live accept requires a baseline. The guard above makes None unreachable here; if it ever
    // happens, apply nothing, record nothing, and put the proposal back at the front.
    let baseline = live_pool_fitness(sm);
    if !shadow && accepted && baseline.is_none() {
        tracing::warn!(
            "[selection] accept withheld: no live fitness baseline; delta not applied ({})",
            proposal.reason
        );
        sm.pending_rule_proposals.push_front(proposal);
        return;
    }

    push_trial_record(sm, RuleTrialRecord {
        ts: crate::core::now(),
        delta: proposal.delta.clone(),
        calib_current,
        calib_proposed,
        margin: tc.margin,
        accepted,
        shadow,
        scored_with,
        calib_current_live,
        calib_proposed_live,
    });

    if shadow {
        let live_pair = match (calib_current_live, calib_proposed_live) {
            (Some(c), Some(p)) => format!("; live({:?}) calib {:.3} -> {:.3}", live_metric, c, p),
            _ => String::new(),
        };
        tracing::info!(
            "[selection] SHADOW would-{} rule delta ({}): decision(Domain) calib {:.3} -> {:.3} (margin {:.2}){}; nothing applied",
            if accepted { "accept" } else { "reject" },
            proposal.reason, calib_current, calib_proposed, tc.margin, live_pair
        );
        return;
    }

    if accepted {
        push_rollback(sm, RollbackEntry {
            before: current,
            after: proposed.clone(),
            ts: crate::core::now(),
            reason: proposal.reason.clone(),
        });
        proposed.restore_into(sm);
        // Always armed: `baseline` is Some here (a live accept without one returned above).
        sm.rule_regression_watch = baseline.map(|baseline_fitness| RegressionWatch {
            baseline_fitness,
            cycles_below: 0,
            cycles_observed: 0,
        });
        tracing::warn!(
            "[selection] ACCEPTED rule delta ({}): calib {:.3} -> {:.3} (margin {:.2}); rollback depth {}; delta {:?}",
            proposal.reason, calib_current, calib_proposed, tc.margin,
            sm.rule_rollback_entries.len(), proposal.delta
        );
    } else {
        tracing::info!(
            "[selection] rejected rule delta ({}): calib {:.3} -> {:.3} (margin {:.2})",
            proposal.reason, calib_current, calib_proposed, tc.margin
        );
    }
}

fn push_rollback(sm: &mut SelfModel, entry: RollbackEntry) {
    sm.rule_rollback_entries.push_back(entry);
    while sm.rule_rollback_entries.len() > ROLLBACK_N {
        sm.rule_rollback_entries.pop_front();
    }
}

/// Field-level rollback of the newest accepted trial. For each field the accept changed:
/// restore the before value only if the live value still equals the accept's value; otherwise
/// (an operator or anything else changed it since) leave it and report it skipped. Fields the
/// accept did not change are never touched. Clears any regression watch. None when empty.
pub fn rollback_last_rule(sm: &mut SelfModel) -> Option<RollbackOutcome> {
    let entry = sm.rule_rollback_entries.pop_back()?;
    let mut restored = Vec::new();
    let mut skipped = Vec::new();
    for field in entry.changed() {
        if field.same(&sm.critic_rules, &sm.metacog_phase_order, &entry.after.critic_rules, &entry.after.phase_order) {
            field.copy_into(&entry.before, sm);
            restored.push(field.name());
        } else {
            skipped.push(field.name());
        }
    }
    sm.rule_regression_watch = None;
    if !skipped.is_empty() {
        tracing::warn!(
            "[selection] rollback left {:?} alone: edited since the accept ({})",
            skipped, entry.reason
        );
    }
    Some(RollbackOutcome { entry, restored, skipped })
}

/// Clear an armed regression watch, WARN-logging the cause. No-op when none is armed.
pub fn clear_regression_watch(sm: &mut SelfModel, cause: &str) {
    if let Some(w) = sm.rule_regression_watch.take() {
        tracing::warn!(
            "[selection] regression watch cleared ({}): baseline {:.3}, {} cycles observed",
            cause, w.baseline_fitness, w.cycles_observed
        );
    }
}

/// Stage change (POST /selection/stage): leaving RuleTrialsLive freezes the guard, so a later
/// re-promotion would resume it with a stale baseline; clear it on demotion. Entering
/// RuleTrialsLive from another stage starts fresh, so clear any watch that somehow exists.
/// Re-posting the same stage is a no-op. Demotion never reverts accepted rules (use
/// /selection/rollback).
pub fn on_stage_change(sm: &mut SelfModel, prev: SelfModStage) {
    let now = sm.self_mod_stage;
    if prev == now {
        return;
    }
    if prev == SelfModStage::RuleTrialsLive {
        clear_regression_watch(sm, "demoted from rule_trials_live");
    } else if now == SelfModStage::RuleTrialsLive {
        clear_regression_watch(sm, "promoted to rule_trials_live (fresh start)");
    }
}

/// Metric change (POST /metacog/config): switching away from Domain closes the trials gate and
/// freezes the guard; clear the watch so re-enabling does not resume a stale baseline.
pub fn on_metric_change(sm: &mut SelfModel, prev: AgreementMetric) {
    if prev.is_domain() && !sm.agreement_metric.is_domain() {
        clear_regression_watch(sm, "agreement_metric switched away from domain");
    }
}

/// Operator rule edit (POST /metacog/config): if the rule config differs from `before`, WARN-log
/// the change and clear the regression watch — its baseline is no longer attributable to the
/// accepted delta. Rollback entries are kept; field-level rollback protects the operator's
/// values. Returns whether the config changed.
pub fn note_operator_rule_edit(sm: &mut SelfModel, before: &RuleConfigSnapshot) -> bool {
    let after = RuleConfigSnapshot::of(sm);
    if &after == before {
        return false;
    }
    tracing::warn!(
        "[selection] operator changed rule config via /metacog/config: {:?} -> {:?}",
        before, after
    );
    clear_regression_watch(sm, "operator rule edit");
    true
}

/// Live-regression guard (trigger 2): once per dream cycle, only when trials_live. If live
/// pool fitness stays below the accept-time baseline for T consecutive cycles, revert the last
/// accepted delta; if it holds for T cycles, disarm. Until metacognitive_loop is wired into a
/// live path, rules cannot move live fitness, so this guard measures noise.
pub fn check_rule_regression(sm: &mut SelfModel) {
    if !trials_live(sm) {
        return;
    }
    let Some(mut watch) = sm.rule_regression_watch.clone() else { return };
    let Some(live) = live_pool_fitness(sm) else { return };
    let t = RULE_TRIAL_CONFIG.regression_cycles;
    watch.cycles_observed += 1;
    if live < watch.baseline_fitness {
        watch.cycles_below += 1;
    } else {
        watch.cycles_below = 0;
    }
    if watch.cycles_below >= t {
        match rollback_last_rule(sm) {
            Some(out) => tracing::warn!(
                "[selection] ROLLBACK: live fitness {:.3} below baseline {:.3} for {} cycles; restored {:?}, skipped {:?} (depth now {})",
                live, watch.baseline_fitness, t, out.restored, out.skipped, sm.rule_rollback_entries.len()
            ),
            None => {
                sm.rule_regression_watch = None;
                tracing::warn!("[selection] regression detected but rollback ring is empty; watch cleared");
            }
        }
    } else if watch.cycles_observed >= t {
        sm.rule_regression_watch = None;
        tracing::info!(
            "[selection] accepted rule held for {} cycles (live {:.3} vs baseline {:.3}); watch cleared",
            t, live, watch.baseline_fitness
        );
    } else {
        sm.rule_regression_watch = Some(watch);
    }
}

// ── Theory of Mind Helpers ──────────────────────────────────────

/// Record a motor output to an audience member. Updates audience model.
pub fn record_audience_output(sm: &mut SelfModel, audience_id: &str, message: &str, domain: &str) {
    sm.active_audience_id = Some(audience_id.to_string());
    let audience = sm.audience_model.entry(audience_id.to_string())
        .or_insert_with(|| AudienceBeliefs::new(audience_id));

    audience.last_interaction = crate::core::now();
    audience.last_message_sent = message.to_string();
    audience.interaction_count += 1;

    if !domain.is_empty() && !audience.topics_discussed.contains(&domain.to_string()) {
        audience.topics_discussed.push(domain.to_string());
        if audience.topics_discussed.len() > 20 { audience.topics_discussed.remove(0); }
    }

    if message.len() > 10
        && let Ok(emb) = crate::embed::embed_text(message) {
            audience.last_context_embedding = Some(emb);
        }

    if !domain.is_empty() {
        let knowledge = audience.estimated_knowledge.entry(domain.to_string()).or_insert(0.0);
        *knowledge = (*knowledge + 0.1).min(1.0);
    }

    // Bound the audience model so an always-on social process can't leak memory:
    // beyond the cap, evict the least-recently-seen audience. The entry we just
    // touched has last_interaction = now, so it is never the one evicted.
    const MAX_AUDIENCES: usize = 500;
    if sm.audience_model.len() > MAX_AUDIENCES
        && let Some(oldest) = sm
            .audience_model
            .iter()
            .min_by(|a, b| {
                a.1.last_interaction
                    .partial_cmp(&b.1.last_interaction)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
        {
            sm.audience_model.remove(&oldest);
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
        crate::metrics::record_tom_prediction(sim >= 0.4);
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

    #[test]
    fn audience_model_stays_bounded() {
        // Now that ToM is wired into the chat path, an unbounded audience map
        // would leak in an always-on social process. Verify the cap holds.
        let mut sm = SelfModel::new();
        // Short message (<=10 chars) skips embedding so the test needs no model.
        for i in 0..620 {
            record_audience_output(&mut sm, &format!("user:{i}"), "hi", "chat");
        }
        assert!(
            sm.audience_model.len() <= 500,
            "audience model must stay bounded, got {}",
            sm.audience_model.len()
        );
    }

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
            domain_agreement_score: 0.5,
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

    #[test]
    fn test_apply_critic_adjustments_phase_order() {
        let mut sm = test_sm();
        let adj = CriticAdjustment {
            critic_rule_delta: CriticRuleDelta {
                phase_order: Some(vec![
                    MetacogPhase::Draft,
                    MetacogPhase::ReflectivePause,
                    MetacogPhase::Critique,
                ]),
                ..Default::default()
            },
            ..Default::default()
        };

        apply_critic_adjustments(&mut sm, &adj);
        assert_eq!(
            sm.metacog_phase_order,
            vec![
                MetacogPhase::Draft,
                MetacogPhase::ReflectivePause,
                MetacogPhase::Critique,
            ]
        );
    }

    #[test]
    fn test_apply_critic_adjustments_spawns_bias_variants() {
        let mut sm = test_sm();
        let before = sm.learned_biases.len();

        let adj = CriticAdjustment {
            spawn_bias_variants: 2,
            ..Default::default()
        };

        apply_critic_adjustments(&mut sm, &adj);
        assert!(sm.learned_biases.len() >= before + 2);
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

    // ── Efficiency cognition-quality guards ──────────────────────
    // These assert that the cost-bounding levers (top-k context, token
    // budget, difficulty routing) do NOT strip the essential cognitive
    // signal from the critic prompt. Run on a fixed scenario set.

    #[test]
    fn efficiency_budget_preserves_critic_signal() {
        let mut sm = test_sm();
        // Pathological growth: a huge domain list that would otherwise dump
        // an unbounded subgraph into the prompt.
        let output = test_walk_output();
        let mut results = test_walker_results();
        results[0].domains_visited =
            (0..100).map(|i| format!("domain_{}", i)).collect();
        sm.consecutive_repetitions = 4; // force BreakLoop (a difficult session)

        let summary = build_session_summary(&output, &results, &sm);
        let diagnosis = diagnose_session(&summary, &sm);
        let (system, user) = build_critic_llm_prompt(&diagnosis, &summary);

        // Top-k cap engaged, but the unique-domain count still conveys breadth.
        assert!(user.contains("more)"), "domain list should be capped, not dumped");
        // Essential instructions survive both the cap and the wire-level budget.
        let budgeted = crate::budget::budget_prompt(&user);
        assert!(system.contains("metacognitive critic"));
        assert!(budgeted.contains("Output ONLY JSON"), "JSON contract must survive budgeting");
        assert!(budgeted.contains("Diagnosis:"), "diagnosis must survive budgeting");
        // And the prompt is actually bounded.
        assert!(crate::budget::approx_tokens(&budgeted) <= crate::budget::max_prompt_tokens());

        // Even under FORCED truncation, the trailing JSON contract survives
        // (head+tail policy) — head-only truncation would have dropped it.
        let forced = crate::budget::budget_text(&user, 60);
        assert!(forced.len() < user.len(), "should have truncated");
        assert!(forced.contains("Session:"), "head must survive forced truncation");
        assert!(
            forced.contains("attention_deltas"),
            "trailing JSON contract must survive forced truncation"
        );
    }

    #[test]
    fn efficiency_routing_escalates_only_hard_sessions() {
        // Easy/normal session → algorithmic only → cheap chat tier.
        let sm = test_sm();
        let output = test_walk_output();
        let results = test_walker_results();
        let summary = build_session_summary(&output, &results, &sm);
        let normal = diagnose_session(&summary, &sm);
        assert!(
            !crate::budget::wants_reasoner(0.5, normal.algorithmic_adjustments.escalate_to_llm),
            "a normal session must NOT pay for the reasoner"
        );

        // Stuck session → genuine difficulty → reasoner tier.
        let mut stuck_sm = test_sm();
        stuck_sm.consecutive_repetitions = 4;
        let stuck_summary = build_session_summary(&output, &results, &stuck_sm);
        let stuck = diagnose_session(&stuck_summary, &stuck_sm);
        assert!(
            crate::budget::wants_reasoner(0.5, stuck.algorithmic_adjustments.escalate_to_llm),
            "a genuinely stuck session SHOULD escalate to the reasoner"
        );
    }

    // ── Stage 1: select_biases / plan_selection ──────────────────────────────

    fn walked_bias(novelty: f32, surprise: f32, fitness: f32, walks: u32, m2: f32) -> crate::graph::LearnedBias {
        let mut b = crate::graph::LearnedBias::default();
        b.scorecard.novelty = novelty;
        b.scorecard.surprise_kept = surprise;
        b.scorecard.fitness = fitness;
        b.scorecard.walks = walks;
        b.scorecard.fitness_m2 = m2;
        b
    }

    /// Pool with two same-niche, well-sampled profiles where B (idx 1) is confidently
    /// dominated by A (idx 0), padded with unwalked defaults so len (7) > FLOOR (6).
    fn confident_cull_pool() -> SelfModel {
        let mut sm = SelfModel::new();
        sm.learned_biases.clear();
        sm.learned_biases.push(walked_bias(0.5, 0.5, 0.9, 10, 0.01)); // A incumbent, niche (2,2)
        sm.learned_biases.push(walked_bias(0.5, 0.5, 0.1, 10, 0.01)); // B dominated, same niche
        for _ in 0..5 {
            sm.learned_biases.push(crate::graph::LearnedBias::default()); // unwalked → no niche, immune
        }
        sm
    }

    fn arm_cadence(sm: &mut SelfModel) {
        sm.selection_cycle_counter = SELECT_EVERY - 1; // next select_biases call is an active cycle
    }

    #[test]
    fn select_biases_observability_commits_nothing() {
        let mut sm = confident_cull_pool();
        arm_cadence(&mut sm);
        assert_eq!(sm.self_mod_stage, crate::core::SelfModStage::Observability);
        let before = serde_json::to_string(&sm.learned_biases).unwrap();
        select_biases(&mut sm);
        assert_eq!(
            serde_json::to_string(&sm.learned_biases).unwrap(),
            before,
            "Observability must not mutate the pool even with a confident cull available"
        );
    }

    #[test]
    fn select_biases_culls_dominated_and_breeds_at_selection_live() {
        let mut sm = confident_cull_pool();
        sm.self_mod_stage = crate::core::SelfModStage::SelectionLive;
        arm_cadence(&mut sm);
        let len_before = sm.learned_biases.len();
        select_biases(&mut sm);
        assert_eq!(sm.learned_biases.len(), len_before, "cull-then-replacement-breed → size constant");
        assert!(
            !sm.learned_biases.iter().any(|b| (b.scorecard.fitness - 0.1).abs() < 1e-6 && b.scorecard.walks == 10),
            "the confidently-dominated candidate (fitness 0.1, walks 10) was culled"
        );
        assert!(
            sm.learned_biases.iter().any(|b| (b.novelty_seeking - 0.38).abs() < 1e-6 && b.scorecard.walks == 0),
            "a fresh child bred from incumbent A (novelty_seeking 0.3+0.08, reset scorecard) was added"
        );
    }

    #[test]
    fn select_biases_never_culls_under_sampled() {
        let mut sm = confident_cull_pool();
        sm.self_mod_stage = crate::core::SelfModStage::SelectionLive;
        sm.learned_biases[1].scorecard.walks = 3; // < MIN_WALKS
        arm_cadence(&mut sm);
        let before = serde_json::to_string(&sm.learned_biases).unwrap();
        select_biases(&mut sm);
        assert_eq!(
            serde_json::to_string(&sm.learned_biases).unwrap(),
            before,
            "under-sampled (walks<MIN_WALKS) profile is cull-immune"
        );
    }

    #[test]
    fn select_biases_respects_floor() {
        let mut sm = SelfModel::new();
        sm.self_mod_stage = crate::core::SelfModStage::SelectionLive;
        sm.learned_biases.clear();
        sm.learned_biases.push(walked_bias(0.5, 0.5, 0.9, 10, 0.01));
        sm.learned_biases.push(walked_bias(0.5, 0.5, 0.1, 10, 0.01));
        for _ in 0..4 {
            sm.learned_biases.push(crate::graph::LearnedBias::default());
        }
        assert_eq!(sm.learned_biases.len(), 6, "== FLOOR");
        arm_cadence(&mut sm);
        let before = serde_json::to_string(&sm.learned_biases).unwrap();
        select_biases(&mut sm);
        assert_eq!(serde_json::to_string(&sm.learned_biases).unwrap(), before, "pool at FLOOR never shrinks");
    }

    #[test]
    fn select_biases_frozen_in_compliant() {
        let mut sm = confident_cull_pool();
        sm.self_mod_stage = crate::core::SelfModStage::SelectionLive;
        sm.mode = CognitiveMode::Compliant;
        arm_cadence(&mut sm);
        let before = serde_json::to_string(&sm.learned_biases).unwrap();
        let counter_before = sm.selection_cycle_counter;
        select_biases(&mut sm);
        assert_eq!(serde_json::to_string(&sm.learned_biases).unwrap(), before, "Compliant freezes the pool");
        assert_eq!(sm.selection_cycle_counter, counter_before, "Compliant: not even the counter advances");
    }

    #[test]
    fn plan_selection_is_deterministic() {
        let sm = confident_cull_pool();
        let p1 = plan_selection(&sm);
        let p2 = plan_selection(&sm);
        assert_eq!(
            p1.cull.as_ref().map(|c| c.idx),
            p2.cull.as_ref().map(|c| c.idx),
            "plan_selection must be deterministic (no RNG / HashMap iteration order)"
        );
        assert_eq!(p1.cull.as_ref().map(|c| c.idx), Some(1), "the dominated candidate (idx 1) is the cull");
    }

    #[test]
    fn perturb_child_resets_scorecard_and_perturbs() {
        let mut parent = crate::graph::LearnedBias::default();
        parent.scorecard.walks = 50;
        parent.scorecard.fitness = 0.7;
        parent.sessions_learned = 9;
        let child = perturb_child(&parent);
        assert_eq!(child.scorecard.walks, 0, "child born under-sampled → cull-immune");
        assert_eq!(child.sessions_learned, 0);
        assert!((child.novelty_seeking - (parent.novelty_seeking + 0.08)).abs() < 1e-6);
        assert!((child.cross_domain_curiosity - (parent.cross_domain_curiosity + 0.06)).abs() < 1e-6);
    }

    #[test]
    fn select_biases_bumps_pool_generation_on_cull() {
        let mut sm = confident_cull_pool();
        sm.self_mod_stage = crate::core::SelfModStage::SelectionLive;
        arm_cadence(&mut sm);
        let gen_before = sm.pool_generation;
        select_biases(&mut sm);
        assert_eq!(sm.pool_generation, gen_before + 1, "a committed cull bumps pool_generation");
    }

    #[test]
    fn select_biases_observability_does_not_bump_generation() {
        let mut sm = confident_cull_pool(); // default Observability
        arm_cadence(&mut sm);
        let gen_before = sm.pool_generation;
        select_biases(&mut sm);
        assert_eq!(sm.pool_generation, gen_before, "Observability commits nothing → no generation bump");
    }

    // ── Stage 2 Step 1: flag-gated domain agreement in the hallucination check ──

    #[test]
    fn metacognitive_loop_reads_domain_agreement_only_under_domain_metric() {
        // Legacy agreement 0 (fails the 0.2 threshold), domain agreement 0.9 (passes).
        let walk = WalkOutput { agreement_score: 0.0, domain_agreement_score: 0.9, ..test_walk_output() };
        let mut base = SelfModel::new();
        base.mode = CognitiveMode::Autonomous;
        base.energy = 0.9;
        base.attention_patterns.clear();
        base.wounds.clear();

        let mut legacy = base.clone();
        legacy.agreement_metric = crate::core::AgreementMetric::Legacy;
        assert!(metacognitive_loop(&walk, &mut legacy, 2).is_none(), "Legacy reads 0.0 → rejected");

        let mut domain = base.clone();
        domain.agreement_metric = crate::core::AgreementMetric::Domain;
        assert!(metacognitive_loop(&walk, &mut domain, 2).is_some(), "Domain reads 0.9 → approved");

        let mut unknown = base.clone();
        unknown.agreement_metric = crate::core::AgreementMetric::Unknown;
        assert!(metacognitive_loop(&walk, &mut unknown, 2).is_none(), "Unknown behaves as Legacy");
    }

    // ── Stage 2: rule trials, replay, rollback (PROTOCOL-self-selection §#2, §Compliance) ──

    use crate::core::{AgreementMetric, SelfModStage};

    fn replay_entry(fitness: f32, domain_agree: f32) -> ReplayEntry {
        ReplayEntry {
            recommended_action: "explore".into(),
            primary_domain: "markets".into(),
            agreement_score: 0.0,
            domain_agreement_score: domain_agree,
            novelty_score: 0.3,
            walker_count: 6,
            mode: CognitiveMode::Autonomous,
            energy: 0.9,
            attention_share: None,
            wound: None,
            fitness,
        }
    }

    /// REPLAY_M walks: high-fitness walks have high domain agreement, low-fitness walks low.
    /// Under Domain + default threshold 0.2 every walk is approved (calibration 1.0); raising
    /// the threshold to 0.5 approves the good half and aborts the bad half (calibration 2.0).
    fn separable_buffer() -> VecDeque<ReplayEntry> {
        (0..REPLAY_M)
            .map(|i| if i % 2 == 0 { replay_entry(0.8, 0.9) } else { replay_entry(0.1, 0.25) })
            .collect()
    }

    fn delta_threshold(v: f32) -> CriticRuleDelta {
        CriticRuleDelta { agreement_threshold_autonomous: Some(v), ..Default::default() }
    }

    fn stage2_sm(stage: SelfModStage, metric: AgreementMetric) -> SelfModel {
        let mut sm = SelfModel::new();
        sm.mode = CognitiveMode::Autonomous;
        sm.self_mod_stage = stage;
        sm.agreement_metric = metric;
        sm.recent_walk_fitness = separable_buffer();
        if let Some(b) = sm.learned_biases.get_mut(0) {
            b.scorecard.walks = 1;
            b.scorecard.fitness = 0.3;
        }
        sm
    }

    /// Whole model minus the Stage-2 rings, for commit-nothing comparisons.
    fn model_minus_stage2_rings(m: &SelfModel) -> String {
        let mut c = m.clone();
        c.pending_rule_proposals.clear();
        c.rule_trial_history.clear();
        serde_json::to_string(&c).unwrap()
    }

    #[test]
    fn stage2_flag_off_route_is_exactly_legacy_apply_plus_one_shadow_proposal() {
        let delta = CriticRuleDelta {
            min_energy: Some(0.3),
            phase_order: Some(vec![MetacogPhase::Draft, MetacogPhase::ReflectivePause, MetacogPhase::Critique]),
            ..Default::default()
        };
        let base = stage2_sm(SelfModStage::Observability, AgreementMetric::Legacy);

        let mut routed = base.clone();
        route_rule_delta(&mut routed, &delta, "test");
        let mut legacy = base.clone();
        apply_rule_delta(&mut legacy, &delta);

        assert_eq!(model_minus_stage2_rings(&routed), model_minus_stage2_rings(&legacy),
            "flag-off: identical to the pre-Stage-2 direct apply");
        assert_eq!(routed.pending_rule_proposals.len(), 1, "the only new state: one shadow proposal");
        assert!(routed.pending_rule_proposals[0].applied_directly);
        assert_eq!(routed.pending_rule_proposals[0].base, RuleConfigSnapshot::of(&base));

        // Through the real entry point: critic_rules change exactly as on base.
        let mut via_adj = base.clone();
        let adj = CriticAdjustment { critic_rule_delta: delta.clone(), ..Default::default() };
        apply_critic_adjustments(&mut via_adj, &adj);
        assert_eq!(RuleConfigSnapshot::of(&via_adj), RuleConfigSnapshot::of(&base).with_delta(&delta));

        // An empty delta is a pure no-op (no proposal).
        let mut empty = base.clone();
        route_rule_delta(&mut empty, &CriticRuleDelta::default(), "test");
        assert!(empty.pending_rule_proposals.is_empty());
    }

    #[test]
    fn stage2_shadow_trial_commits_nothing_at_stage0_and_stage1() {
        for stage in [SelfModStage::Observability, SelfModStage::SelectionLive] {
            let mut sm = stage2_sm(stage, AgreementMetric::Domain);
            route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
            // route applied directly (legacy path); the shadow trial must change nothing else.
            let before = model_minus_stage2_rings(&sm);
            trial_pending_rules(&mut sm);
            assert_eq!(model_minus_stage2_rings(&sm), before, "{stage:?}: shadow trial must commit nothing");
            assert_eq!(sm.rule_trial_history.len(), 1);
            assert!(sm.rule_trial_history[0].shadow);
            assert!(sm.rule_rollback_entries.is_empty());
            assert!(sm.rule_regression_watch.is_none());
        }
    }

    #[test]
    fn stage2_no_silent_apply_at_rule_trials_live() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        assert!(trials_live(&sm));
        let before = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        assert_eq!(RuleConfigSnapshot::of(&sm), before, "Stage 2: enqueue only, never apply directly");
        assert_eq!(sm.pending_rule_proposals.len(), 1);
        assert!(!sm.pending_rule_proposals[0].applied_directly);

        // Same through apply_critic_adjustments.
        let mut sm2 = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let adj = CriticAdjustment { critic_rule_delta: delta_threshold(0.5), ..Default::default() };
        apply_critic_adjustments(&mut sm2, &adj);
        assert_eq!(RuleConfigSnapshot::of(&sm2), before);
    }

    #[test]
    fn stage2_agreement_gate_refuses_to_apply_at_rule_trials_live_with_legacy_metric() {
        for metric in [AgreementMetric::Legacy, AgreementMetric::Unknown] {
            let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, metric);
            assert!(!trials_live(&sm), "{metric:?}: gate closed");
            route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
            assert!(sm.pending_rule_proposals[0].applied_directly, "legacy direct path, not a live trial");
            trial_pending_rules(&mut sm);
            assert!(sm.rule_trial_history.back().unwrap().shadow, "{metric:?}: trial is shadow only");
            assert!(sm.rule_rollback_entries.is_empty(), "{metric:?}: nothing accepted");
            assert!(sm.rule_regression_watch.is_none());
        }
    }

    #[test]
    fn stage2_frozen_in_compliant() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        sm.mode = CognitiveMode::Compliant;
        assert!(!trials_live(&sm));

        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        assert!(sm.pending_rule_proposals.is_empty(), "Compliant: no enqueue");

        let n = sm.recent_walk_fitness.len();
        record_replay_entry(&mut sm, replay_entry(0.5, 0.5));
        assert_eq!(sm.recent_walk_fitness.len(), n, "Compliant: no replay recording");

        // Even with a proposal, a watch and a snapshot present: no trial, accept or rollback.
        sm.pending_rule_proposals.push_back(PendingRuleProposal {
            delta: delta_threshold(0.5),
            reason: "test".into(),
            created_at: 0.0,
            base: RuleConfigSnapshot::of(&sm),
            applied_directly: false,
        });
        sm.rule_rollback_entries.push_back(RollbackEntry {
            before: RuleConfigSnapshot::of(&SelfModel::new()),
            after: RuleConfigSnapshot::of(&sm),
            ts: 0.0,
            reason: "test".into(),
        });
        sm.rule_regression_watch = Some(RegressionWatch { baseline_fitness: 10.0, cycles_below: 0, cycles_observed: 0 });
        let before = serde_json::to_string(&sm).unwrap();
        for _ in 0..(RULE_TRIAL_CONFIG.regression_cycles * 2) {
            trial_pending_rules(&mut sm);
            check_rule_regression(&mut sm);
        }
        assert_eq!(serde_json::to_string(&sm).unwrap(), before, "Compliant: nothing moves");
    }

    #[test]
    fn stage2_accept_path_applies_and_grows_rollback_ring() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let original = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);

        let rec = sm.rule_trial_history.back().unwrap();
        assert!(rec.accepted && !rec.shadow);
        assert!(rec.calib_proposed > rec.calib_current + RULE_TRIAL_CONFIG.margin);
        assert_eq!(sm.critic_rules.agreement_threshold_autonomous, 0.5, "accepted delta is live");
        assert_eq!(sm.rule_rollback_entries.len(), 1);
        assert_eq!(sm.rule_rollback_entries[0].before, original);
        assert_eq!(sm.rule_rollback_entries[0].after, RuleConfigSnapshot::of(&sm));
        let w = sm.rule_regression_watch.as_ref().expect("watch armed");
        assert_eq!(w.baseline_fitness, 0.3);
    }

    #[test]
    fn stage2_reject_path_not_applied_recorded_and_not_retrialled() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let original = RuleConfigSnapshot::of(&sm);
        let worse = delta_threshold(0.95); // clamps to 0.9 → aborts everything
        route_rule_delta(&mut sm, &worse, "test");
        trial_pending_rules(&mut sm);

        assert_eq!(RuleConfigSnapshot::of(&sm), original, "rejected delta never goes live");
        assert_eq!(sm.rule_trial_history.len(), 1);
        assert!(!sm.rule_trial_history[0].accepted);
        assert!(sm.rule_rollback_entries.is_empty());

        route_rule_delta(&mut sm, &worse, "test-again");
        trial_pending_rules(&mut sm);
        assert_eq!(sm.rule_trial_history.len(), 1, "identical rejected delta is not re-trialled");
        assert!(sm.pending_rule_proposals.is_empty(), "the duplicate is consumed, not re-queued");
        assert_eq!(RuleConfigSnapshot::of(&sm), original);
    }

    #[test]
    fn stage2_rollback_is_exact() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        sm.metacog_phase_order = vec![MetacogPhase::Draft, MetacogPhase::ReflectivePause, MetacogPhase::Critique];
        let original = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        assert_ne!(RuleConfigSnapshot::of(&sm), original);

        let out = rollback_last_rule(&mut sm).expect("ring has one entry");
        assert_eq!(out.entry.before, original);
        assert_eq!(RuleConfigSnapshot::of(&sm), original, "field-for-field restore");
        assert_eq!(out.restored, out.entry.changed_field_names(), "every changed field restored");
        assert_eq!(out.restored, vec!["agreement_threshold_autonomous"]);
        assert!(out.skipped.is_empty());
        assert!(sm.rule_regression_watch.is_none());
        assert!(rollback_last_rule(&mut sm).is_none(), "empty ring → None (API returns 409)");
    }

    #[test]
    fn stage2_regression_guard_reverts_after_t_cycles_not_before() {
        let t = RULE_TRIAL_CONFIG.regression_cycles;
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let original = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        let accepted = RuleConfigSnapshot::of(&sm);

        sm.learned_biases[0].scorecard.fitness = 0.1; // live fitness drops below baseline 0.3
        for i in 1..t {
            check_rule_regression(&mut sm);
            assert_eq!(RuleConfigSnapshot::of(&sm), accepted, "not reverted after {i} < T cycles");
        }
        check_rule_regression(&mut sm);
        assert_eq!(RuleConfigSnapshot::of(&sm), original, "reverted at T cycles");
        assert!(sm.rule_regression_watch.is_none());

        // Holding at/above baseline for T cycles disarms the watch without reverting.
        let mut held = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        route_rule_delta(&mut held, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut held);
        let kept = RuleConfigSnapshot::of(&held);
        for _ in 0..t {
            check_rule_regression(&mut held);
        }
        assert!(held.rule_regression_watch.is_none(), "watch cleared after holding T cycles");
        assert_eq!(RuleConfigSnapshot::of(&held), kept, "no revert when fitness holds");
    }

    #[test]
    fn stage2_calibration_is_acyclic_approval_has_no_input() {
        // Compile-level: exhaustive destructure (no `..`) — adding any field, e.g. an approval
        // input, breaks this test and forces a review of acyclicity.
        let ReplayEntry {
            recommended_action: _, primary_domain: _, agreement_score: _, domain_agreement_score: _,
            novelty_score: _, walker_count: _, mode: _, energy: _, attention_share: _, wound: _,
            fitness: _,
        } = replay_entry(0.5, 0.5);

        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let cfg = RuleConfigSnapshot::of(&sm);
        let tc = RULE_TRIAL_CONFIG;
        let before = replay_calibration(&cfg, &sm.recent_walk_fitness, sm.agreement_metric, &tc);
        // Move the per-profile approval signal. Global METRICS is deliberately not mutated
        // (process-global, read by walker/dream under parallel tests); it cannot reach the
        // score structurally: replay_calibration's inputs are the RuleConfigSnapshot, the
        // ReplayEntry buffer (exhaustively destructured above), the metric and RuleTrialConfig,
        // none of which carries an approval value.
        for b in sm.learned_biases.iter_mut() {
            b.scorecard.approval = 1.0;
        }
        let after = replay_calibration(&cfg, &sm.recent_walk_fitness, sm.agreement_metric, &tc);
        assert_eq!(before, after, "approval must not influence the trial score");
    }

    #[test]
    fn stage2_every_ring_is_bounded() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        for i in 0..(PENDING_CAP * 10) {
            route_rule_delta(&mut sm, &delta_threshold(0.3 + (i % 5) as f32 * 0.01), "bound");
        }
        assert!(sm.pending_rule_proposals.len() <= PENDING_CAP);

        for _ in 0..(REPLAY_M * 10) {
            record_replay_entry(&mut sm, replay_entry(0.5, 0.5));
        }
        assert!(sm.recent_walk_fitness.len() <= REPLAY_M);

        for _ in 0..(TRIAL_HISTORY_CAP * 10) {
            push_trial_record(&mut sm, RuleTrialRecord {
                ts: 0.0, delta: CriticRuleDelta::default(), calib_current: 0.0, calib_proposed: 0.0,
                margin: 0.0, accepted: false, shadow: true,
                scored_with: AgreementMetric::Domain, calib_current_live: None, calib_proposed_live: None,
            });
        }
        assert!(sm.rule_trial_history.len() <= TRIAL_HISTORY_CAP);

        for _ in 0..(ROLLBACK_N * 10) {
            let snap = RuleConfigSnapshot::of(&sm);
            push_rollback(&mut sm, RollbackEntry { before: snap.clone(), after: snap, ts: 0.0, reason: "bound".into() });
        }
        assert!(sm.rule_rollback_entries.len() <= ROLLBACK_N);

        // And at most one trial per dream cycle.
        let mut one = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        route_rule_delta(&mut one, &delta_threshold(0.5), "a");
        route_rule_delta(&mut one, &delta_threshold(0.6), "b");
        trial_pending_rules(&mut one);
        assert_eq!(one.rule_trial_history.len(), 1);
        assert_eq!(one.pending_rule_proposals.len(), 1);
    }

    #[test]
    fn stage2_replay_decide_matches_metacognitive_loop() {
        // Deterministic LCG over contexts, rules, phase orders, metric, mode and attempts.
        let mut seed: u64 = 0x5eed_2026;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / ((1u64 << 31) as f32)
        };
        let phase_orders = [
            vec![MetacogPhase::Draft, MetacogPhase::Critique],
            vec![MetacogPhase::Draft, MetacogPhase::ReflectivePause, MetacogPhase::Critique],
            vec![MetacogPhase::Critique, MetacogPhase::ReflectivePause],
            vec![MetacogPhase::Draft], // no critique → safety fallback
        ];
        let (mut approved_n, mut rejected_n) = (0, 0);
        for case in 0..500 {
            let mut sm = SelfModel::new();
            sm.mode = if next() < 0.8 { CognitiveMode::Autonomous } else { CognitiveMode::Compliant };
            sm.energy = next();
            sm.agreement_metric = if next() < 0.5 { AgreementMetric::Domain } else { AgreementMetric::Legacy };
            sm.attention_patterns.clear();
            if next() < 0.7 {
                sm.attention_patterns.insert("markets".into(), next() * 10.0);
                sm.attention_patterns.insert("other".into(), next() * 10.0);
            }
            sm.wounds.clear();
            if next() < 0.5 {
                sm.wounds.insert("markets".into(), next());
            }
            sm.critic_rules = CriticRuleProfile {
                min_energy: 0.05 + next() * 0.5,
                agreement_threshold_autonomous: 0.05 + next() * 0.8,
                agreement_threshold_compliant: 0.05 + next() * 0.8,
                max_attention_saturation: 0.3 + next() * 0.6,
                wound_guard_threshold: 0.1 + next() * 0.8,
                wound_confidence_floor: 0.1 + next() * 0.8,
            };
            sm.metacog_phase_order = phase_orders[(next() * 4.0) as usize % 4].clone();
            let walk = WalkOutput {
                primary_domain: "markets".into(),
                agreement_score: next(),
                domain_agreement_score: next(),
                ..test_walk_output()
            };
            let max_attempts = 1 + (next() * 3.0) as u32 % 3;

            let entry = ReplayEntry::from_walk(&walk, &sm, 0.0);
            let cfg = RuleConfigSnapshot::of(&sm);
            let replay = replay_decide(&entry, &cfg, sm.agreement_metric, max_attempts);
            // metacognitive_loop does not touch global METRICS under cfg(test) (see the loop).
            let live = metacognitive_loop(&walk, &mut sm.clone(), max_attempts).is_some();
            assert_eq!(replay, live, "case {case}: replay_decide diverged from metacognitive_loop");
            if live { approved_n += 1 } else { rejected_n += 1 }
        }
        assert!(approved_n >= 50 && rejected_n >= 50,
            "both outcomes must be exercised (approved {approved_n}, rejected {rejected_n})");
    }

    // ── Stage 2.1 Fix 1: shadow trials scored under Domain ──

    #[test]
    fn stage2_shadow_trial_scores_under_domain_at_observability_legacy() {
        let mut sm = stage2_sm(SelfModStage::Observability, AgreementMetric::Legacy);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        let before = model_minus_stage2_rings(&sm);
        trial_pending_rules(&mut sm);

        let r = sm.rule_trial_history.back().expect("shadow trial recorded");
        assert!(r.shadow);
        assert_eq!(r.scored_with, AgreementMetric::Domain, "decision pair scored under Domain");
        assert!(r.calib_proposed > r.calib_current + RULE_TRIAL_CONFIG.margin, "would-accept is informative");
        let (c, p) = (r.calib_current_live.expect("live pair recorded"), r.calib_proposed_live.expect("live pair recorded"));
        assert_eq!(c, p, "the degenerate legacy pair is still recorded");
        assert_eq!(model_minus_stage2_rings(&sm), before, "shadow trial commits nothing else");
        assert!(sm.rule_rollback_entries.is_empty() && sm.rule_regression_watch.is_none());
    }

    #[test]
    fn stage2_shadow_trial_under_domain_live_metric_records_no_live_pair() {
        let mut sm = stage2_sm(SelfModStage::Observability, AgreementMetric::Domain);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        let r = sm.rule_trial_history.back().unwrap();
        assert_eq!(r.scored_with, AgreementMetric::Domain);
        assert!(r.calib_current_live.is_none() && r.calib_proposed_live.is_none());
    }

    #[test]
    fn stage2_old_trial_record_without_scored_with_loads_as_legacy() {
        let rec = RuleTrialRecord {
            ts: 1.0, delta: delta_threshold(0.5), calib_current: 0.5, calib_proposed: 0.5,
            margin: 0.05, accepted: false, shadow: true,
            scored_with: AgreementMetric::Domain, calib_current_live: Some(0.5), calib_proposed_live: Some(0.5),
        };
        let mut v = serde_json::to_value(&rec).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("scored_with");
        o.remove("calib_current_live");
        o.remove("calib_proposed_live");
        let loaded: RuleTrialRecord = serde_json::from_value(v).expect("old record still loads");
        assert_eq!(loaded.scored_with, AgreementMetric::Legacy);
        assert!(loaded.calib_current_live.is_none() && loaded.calib_proposed_live.is_none());
    }

    #[test]
    fn stage2_dedup_respects_scored_with() {
        let rejected = |delta: CriticRuleDelta, scored_with: AgreementMetric| RuleTrialRecord {
            ts: 0.0, delta, calib_current: 0.5, calib_proposed: 0.5, margin: 0.05,
            accepted: false, shadow: true, scored_with, calib_current_live: None, calib_proposed_live: None,
        };
        // A Legacy-scored rejection does NOT block a Domain-scored shadow trial of the same delta.
        let mut sm = stage2_sm(SelfModStage::Observability, AgreementMetric::Legacy);
        sm.rule_trial_history.push_back(rejected(delta_threshold(0.5), AgreementMetric::Legacy));
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        assert_eq!(sm.rule_trial_history.len(), 2, "re-trialled under Domain scoring");

        // A Domain-scored rejection of the same delta does block it.
        let mut sm = stage2_sm(SelfModStage::Observability, AgreementMetric::Legacy);
        sm.rule_trial_history.push_back(rejected(delta_threshold(0.5), AgreementMetric::Domain));
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        assert_eq!(sm.rule_trial_history.len(), 1, "same delta, same scoring → not re-trialled");
        assert!(sm.pending_rule_proposals.is_empty());
    }

    // ── Stage 2.1 Fix 2: field-level rollback preserves operator edits ──

    /// Live sm with delta_threshold(0.5) accepted (threshold 0.2 → 0.5) and the watch armed.
    fn accepted_sm() -> (SelfModel, RuleConfigSnapshot) {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let original = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        assert_eq!(sm.critic_rules.agreement_threshold_autonomous, 0.5, "precondition: accepted");
        assert!(sm.rule_regression_watch.is_some(), "precondition: watch armed");
        (sm, original)
    }

    #[test]
    fn stage2_rollback_preserves_operator_edit_on_changed_field() {
        let (mut sm, original) = accepted_sm();
        sm.critic_rules.agreement_threshold_autonomous = 0.7; // operator edit after the accept
        let out = rollback_last_rule(&mut sm).unwrap();
        assert_eq!(sm.critic_rules.agreement_threshold_autonomous, 0.7, "operator value survives");
        assert_eq!(out.skipped, vec!["agreement_threshold_autonomous"]);
        assert!(out.restored.is_empty());
        assert_eq!(sm.critic_rules.min_energy, original.critic_rules.min_energy);
    }

    #[test]
    fn stage2_rollback_preserves_operator_edit_on_untouched_field() {
        let (mut sm, original) = accepted_sm();
        sm.critic_rules.min_energy = 0.4; // operator edit of a field the delta did not touch
        let out = rollback_last_rule(&mut sm).unwrap();
        assert_eq!(sm.critic_rules.agreement_threshold_autonomous,
            original.critic_rules.agreement_threshold_autonomous, "accepted field restored");
        assert_eq!(sm.critic_rules.min_energy, 0.4, "untouched field keeps the operator value");
        assert_eq!(out.restored, vec!["agreement_threshold_autonomous"]);
        assert!(out.skipped.is_empty(), "min_energy was never a changed field, so not 'skipped'");
    }

    #[test]
    fn stage2_regression_rollback_preserves_operator_edit() {
        let (mut sm, original) = accepted_sm();
        sm.critic_rules.min_energy = 0.4; // simulated operator edit (direct, watch stays armed)
        sm.learned_biases[0].scorecard.fitness = 0.1; // below baseline 0.3
        for _ in 0..RULE_TRIAL_CONFIG.regression_cycles {
            check_rule_regression(&mut sm);
        }
        assert_eq!(sm.critic_rules.agreement_threshold_autonomous,
            original.critic_rules.agreement_threshold_autonomous, "regression reverted the accept");
        assert_eq!(sm.critic_rules.min_energy, 0.4, "operator edit survives the automatic rollback");
        assert!(sm.rule_regression_watch.is_none());
        assert!(sm.rule_rollback_entries.is_empty());
    }

    #[test]
    fn stage2_noop_delta_field_not_counted_as_changed() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        let delta = CriticRuleDelta {
            agreement_threshold_autonomous: Some(0.5),
            phase_order: Some(vec![MetacogPhase::Draft]), // no Critique → ignored by the clamps
            ..Default::default()
        };
        route_rule_delta(&mut sm, &delta, "test");
        trial_pending_rules(&mut sm);
        let entry = sm.rule_rollback_entries.back().expect("accepted");
        assert_eq!(entry.changed_field_names(), vec!["agreement_threshold_autonomous"]);
    }

    #[test]
    fn stage2_operator_edit_clears_regression_watch() {
        let (mut sm, _) = accepted_sm();
        let entries = sm.rule_rollback_entries.clone();
        let before = RuleConfigSnapshot::of(&sm);
        assert!(!note_operator_rule_edit(&mut sm, &before), "no change → nothing happens");
        assert!(sm.rule_regression_watch.is_some());

        sm.critic_rules.min_energy = 0.4;
        assert!(note_operator_rule_edit(&mut sm, &before));
        assert!(sm.rule_regression_watch.is_none(), "baseline no longer attributable → cleared");
        assert_eq!(sm.rule_rollback_entries, entries, "rollback entries are kept");
    }

    // ── Stage 2.1 Fixes 3+4: baseline guard, stale watch ──

    #[test]
    fn stage2_live_trial_skipped_without_baseline_keeps_proposal() {
        let mut sm = stage2_sm(SelfModStage::RuleTrialsLive, AgreementMetric::Domain);
        for b in sm.learned_biases.iter_mut() {
            b.scorecard.walks = 0;
        }
        let rules = RuleConfigSnapshot::of(&sm);
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        let queued = sm.pending_rule_proposals.len();
        trial_pending_rules(&mut sm);
        assert_eq!(RuleConfigSnapshot::of(&sm), rules, "nothing applied");
        assert_eq!(sm.pending_rule_proposals.len(), queued, "proposal kept at the front");
        assert!(sm.rule_trial_history.is_empty(), "no record written");
        assert!(sm.rule_rollback_entries.is_empty());
        assert!(sm.rule_regression_watch.is_none());
    }

    #[test]
    fn stage2_shadow_trial_runs_without_baseline() {
        let mut sm = stage2_sm(SelfModStage::Observability, AgreementMetric::Legacy);
        for b in sm.learned_biases.iter_mut() {
            b.scorecard.walks = 0;
        }
        route_rule_delta(&mut sm, &delta_threshold(0.5), "test");
        trial_pending_rules(&mut sm);
        assert_eq!(sm.rule_trial_history.len(), 1, "shadow trials need no baseline");
        assert!(sm.rule_trial_history[0].shadow);
    }

    #[test]
    fn stage2_demotion_clears_watch() {
        let (mut sm, _) = accepted_sm();
        let rules = RuleConfigSnapshot::of(&sm);
        sm.self_mod_stage = SelfModStage::SelectionLive;
        on_stage_change(&mut sm, SelfModStage::RuleTrialsLive);
        assert!(sm.rule_regression_watch.is_none(), "demotion clears the watch");
        assert_eq!(RuleConfigSnapshot::of(&sm), rules, "demotion does not revert accepted rules");

        // Re-posting the same stage leaves a legitimately armed watch alone.
        let (mut same, _) = accepted_sm();
        on_stage_change(&mut same, SelfModStage::RuleTrialsLive);
        assert!(same.rule_regression_watch.is_some());

        // Promotion into RuleTrialsLive clears a stale watch (fresh start).
        let (mut stale, _) = accepted_sm();
        stale.self_mod_stage = SelfModStage::RuleTrialsLive;
        on_stage_change(&mut stale, SelfModStage::Observability);
        assert!(stale.rule_regression_watch.is_none());
    }

    #[test]
    fn stage2_metric_switch_from_domain_clears_watch() {
        let (mut sm, _) = accepted_sm();
        let rules = RuleConfigSnapshot::of(&sm);
        sm.agreement_metric = AgreementMetric::Legacy;
        on_metric_change(&mut sm, AgreementMetric::Domain);
        assert!(sm.rule_regression_watch.is_none());
        assert_eq!(RuleConfigSnapshot::of(&sm), rules);

        // Domain → Domain (no switch) keeps it.
        let (mut keep, _) = accepted_sm();
        on_metric_change(&mut keep, AgreementMetric::Domain);
        assert!(keep.rule_regression_watch.is_some());
    }
}
