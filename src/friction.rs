//! Friction — errors as sensory pain, not crashes.
//!
//! When a tool fails, a web fetch 404s, a motor command is rejected,
//! or code execution errors — these are NOT crashes. They're feedback.
//! The agent FEELS the friction and learns from it.
//!
//! Every Result::Err becomes a Signal routed through the self-model.
//! The agent experiences: "I tried X and it hurt." This teaches it
//! the boundaries of its own capabilities.

use crate::core::{self, SelfModel, Signal};
use crate::tools::ToolResult;

/// Convert a tool failure into a pain signal through the self-model.
pub fn tool_friction(result: &ToolResult, self_model: &mut SelfModel) {
    if result.success {
        // Success = mild satisfaction, not pain
        let signal = Signal::new("tool_success", &format!(
            "Tool '{}' worked: {}",
            result.tool, core::safe_truncate(&result.content, 80)
        )).with_intensity(0.2);
        core::process(signal, self_model);
        return;
    }

    // Failure = pain signal. Intensity scales with how unexpected it was.
    let intensity = 0.4; // Base pain for tool failure
    let signal = Signal::new("friction", &format!(
        "Tool '{}' failed: {}",
        result.tool, core::safe_truncate(&result.content, 120)
    )).with_intensity(intensity);
    core::process(signal, self_model);

    // Track the specific capability boundary
    let capability = &result.tool;

    // Increment wound for this capability (repeated failures = deeper wound)
    let wound = self_model.wounds
        .entry(format!("tool_{}", capability))
        .or_insert(0.0);
    *wound = (*wound + 0.1).min(1.0);

    // Add to open questions (what went wrong?)
    let question = format!("Why did {} fail? Error: {}", capability, core::safe_truncate(&result.content, 60));
    if !self_model.open_questions.contains(&question) {
        self_model.open_questions.push(question);
        if self_model.open_questions.len() > 10 {
            self_model.open_questions.remove(0);
        }
    }
}

/// Convert a motor command rejection into a friction signal.
pub fn motor_friction(action: &str, error: &str, self_model: &mut SelfModel) {
    let signal = Signal::new("friction", &format!(
        "Motor command '{}' rejected: {}",
        action, core::safe_truncate(error, 120)
    )).with_intensity(0.5);
    core::process(signal, self_model);

    // The body refused. This is stronger than tool failure.
    self_model.arousal = (self_model.arousal + 0.1).min(1.0);
}

/// Convert a network error into a friction signal.
pub fn network_friction(url: &str, error: &str, self_model: &mut SelfModel) {
    let signal = Signal::new("friction", &format!(
        "Network error reaching {}: {}",
        core::safe_truncate(url, 40), core::safe_truncate(error, 80)
    )).with_intensity(0.3);
    core::process(signal, self_model);
}

/// After an action succeeds, record the positive feedback.
/// This is the opposite of friction — affordance confirmation.
pub fn affordance_confirmed(action: &str, self_model: &mut SelfModel) {
    let signal = Signal::new("affordance", &format!(
        "Confirmed: '{}' is within my capabilities",
        action
    )).with_intensity(0.2);
    core::process(signal, self_model);

    // Heal the wound for this capability (success reduces pain memory)
    let key = format!("tool_{}", action);
    if let Some(wound) = self_model.wounds.get_mut(&key) {
        *wound = (*wound - 0.05).max(0.0);
    }
}

/// Describe current capabilities based on what has worked/failed.
/// "I can: web_search (reliable), code_exec (sometimes fails).
///  I cannot: web_fetch (wounded, 3 recent failures)."
pub fn describe_capabilities(self_model: &SelfModel) -> String {
    let mut can_do: Vec<String> = Vec::new();
    let mut wounded: Vec<String> = Vec::new();

    for (key, &wound) in &self_model.wounds {
        if key.starts_with("tool_") {
            let tool = key.strip_prefix("tool_").unwrap_or(key);
            if wound > 0.5 {
                wounded.push(format!("{} (pain: {:.0}%)", tool, wound * 100.0));
            } else if wound > 0.0 {
                can_do.push(format!("{} (some friction)", tool));
            } else {
                can_do.push(tool.to_string());
            }
        }
    }

    let mut lines = Vec::new();
    if !can_do.is_empty() {
        lines.push(format!("I can: {}", can_do.join(", ")));
    }
    if !wounded.is_empty() {
        lines.push(format!("Struggling with: {}", wounded.join(", ")));
    }
    lines.join("\n")
}
