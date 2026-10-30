//! Episodic Memory — Julian's autobiography.
//!
//! For a self to exist, it needs an uninterrupted sense of time.
//! Not just facts in a vector database — events in a timeline.
//! Every cognitive cycle logs: [When] → [Goal] → [Action] → [Outcome]
//!
//! When the walker starts a new task, it doesn't just fetch context.
//! It queries its own past: "I'm trying to do X. Last 3 times I tried,
//! A worked and B failed. Adjust strategy."
//!
//! Consolidation: idle time → extract generalized lessons from
//! recent episodes → store as permanent semantic rules.

use serde::{Deserialize, Serialize};

/// An episode: one complete cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// When this happened
    pub timestamp: f64,
    /// What triggered the cycle
    pub stimulus: String,
    /// What domain this was in
    pub domain: String,
    /// What the walker found
    pub walker_summary: String,
    /// What the critic said
    pub critique: String,
    /// What action was taken
    pub action: String,
    /// What happened as a result
    pub outcome: String,
    /// Was it successful
    pub success: bool,
    /// What was learned
    pub lesson: String,
    /// Emotional state during the episode
    pub valence: f32,
    pub arousal: f32,
    pub energy: f32,
}

/// In-memory episode buffer (recent history, not persisted to graph yet)
pub struct EpisodicMemory {
    /// Recent episodes (short-term, kept in memory)
    pub episodes: Vec<Episode>,
    /// Lessons extracted from consolidation (long-term)
    pub lessons: Vec<Lesson>,
    /// Max episodes before triggering consolidation
    pub max_episodes: usize,
}

/// A generalized lesson extracted from multiple episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    pub rule: String,         // "When I try X in domain Y, it usually Z"
    pub domain: String,
    pub confidence: f32,
    pub evidence_count: u32,
    pub first_learned: f64,
    pub last_reinforced: f64,
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
            lessons: Vec::new(),
            max_episodes: 50,
        }
    }

    /// Record a new episode.
    pub fn record(&mut self, episode: Episode) {
        self.episodes.push(episode);

        // Trigger consolidation if buffer is full
        if self.episodes.len() >= self.max_episodes {
            self.consolidate();
        }
    }

    /// Recall past attempts at a similar task.
    /// Returns the most relevant episodes for a given domain/stimulus.
    pub fn recall_similar(&self, domain: &str, stimulus: &str) -> Vec<&Episode> {
        let mut relevant: Vec<&Episode> = self.episodes.iter()
            .filter(|e| {
                e.domain == domain ||
                stimulus.split_whitespace().any(|w| e.stimulus.contains(w))
            })
            .collect();

        // Most recent first
        relevant.sort_by(|a, b| b.timestamp.partial_cmp(&a.timestamp).unwrap_or(std::cmp::Ordering::Equal));
        relevant.truncate(5);
        relevant
    }

    /// Format past episodes as context for the walker/LLM.
    /// "I'm trying to do X. Last time I tried, Y happened."
    pub fn format_history(&self, domain: &str, stimulus: &str) -> String {
        let past = self.recall_similar(domain, stimulus);
        if past.is_empty() {
            return String::new();
        }

        let mut lines = vec!["=== MY PAST ATTEMPTS ===".to_string()];
        for ep in &past {
            let status = if ep.success { "succeeded" } else { "failed" };
            lines.push(format!(
                "- {}: tried '{}' in {} → {} (lesson: {})",
                status, ep.action, ep.domain, ep.outcome, ep.lesson
            ));
        }

        // Include relevant lessons
        let domain_lessons: Vec<&Lesson> = self.lessons.iter()
            .filter(|l| l.domain == domain && l.confidence > 0.3)
            .collect();
        if !domain_lessons.is_empty() {
            lines.push("\nLessons learned:".to_string());
            for l in domain_lessons {
                lines.push(format!("  * {} (confidence: {:.0}%)", l.rule, l.confidence * 100.0));
            }
        }

        lines.push("=== END HISTORY ===".to_string());
        lines.join("\n")
    }

    /// Consolidation: extract generalized lessons from recent episodes.
    /// Called when the episode buffer is full or during "sleep".
    pub fn consolidate(&mut self) {
        // Group episodes by domain + action
        use std::collections::HashMap;
        let mut groups: HashMap<(String, String), Vec<&Episode>> = HashMap::new();
        for ep in &self.episodes {
            groups.entry((ep.domain.clone(), ep.action.clone()))
                .or_default()
                .push(ep);
        }

        for ((domain, action), episodes) in &groups {
            if episodes.len() < 3 {
                continue; // Not enough evidence
            }

            let success_rate = episodes.iter().filter(|e| e.success).count() as f32
                / episodes.len() as f32;

            let lesson_text = if success_rate > 0.7 {
                format!("'{}' in {} domain usually works ({:.0}% success rate)", action, domain, success_rate * 100.0)
            } else if success_rate < 0.3 {
                format!("'{}' in {} domain usually fails ({:.0}% success rate) — try different approach", action, domain, success_rate * 100.0)
            } else {
                continue; // Mixed results, no clear lesson
            };

            // Check if lesson already exists
            if let Some(existing) = self.lessons.iter_mut()
                .find(|l| l.domain == *domain && l.rule.contains(&**action))
            {
                existing.confidence = success_rate;
                existing.evidence_count += episodes.len() as u32;
                existing.last_reinforced = now();
            } else {
                self.lessons.push(Lesson {
                    rule: lesson_text,
                    domain: domain.clone(),
                    confidence: success_rate,
                    evidence_count: episodes.len() as u32,
                    first_learned: now(),
                    last_reinforced: now(),
                });
            }
        }

        // Keep only recent episodes (clear the old ones)
        let cutoff = now() - 3600.0; // Keep last hour
        self.episodes.retain(|e| e.timestamp > cutoff);

        // Decay old lessons
        for lesson in &mut self.lessons {
            let age = now() - lesson.last_reinforced;
            if age > 86400.0 * 7.0 { // More than a week
                lesson.confidence *= 0.95;
            }
        }
        self.lessons.retain(|l| l.confidence > 0.1);
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
