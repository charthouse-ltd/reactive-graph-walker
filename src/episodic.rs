//! Bounded internal context from verified execution records, never autobiography.
//! Consolidation summarizes the current evidence window; generated dreams are not inputs.

use serde::{Deserialize, Serialize};

/// An episode: one complete cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Stable execution reference; corrections replace evidence.
    pub source_ref: String,
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
    /// Reported score; None is unscored and is never counted as failure
    pub success: Option<bool>,
    /// What was learned
    pub lesson: String,
    /// Emotional state during the episode
    pub valence: f32,
    pub arousal: f32,
    pub energy: f32,
}

/// Bounded in-memory view of execution history; the ledger remains authoritative.
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// Recent episodes (short-term, kept in memory)
    pub episodes: Vec<Episode>,
    /// Lessons supported by the current evidence window
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
        self.episodes.retain(|e| e.source_ref != episode.source_ref);
        self.episodes.push(episode);
        self.episodes.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
        let cap = self.max_episodes.max(1);
        if self.episodes.len() > cap {
            self.episodes.drain(..self.episodes.len() - cap);
        }
        // A correction may invalidate an existing lesson even below the cap.
        if !self.lessons.is_empty() {
            self.consolidate();
        }

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
            let status = match ep.success { Some(true) => "succeeded", Some(false) => "failed", None => "unscored" };
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
        // Recompute from unique current evidence: repeat sleep must not inflate counts,
        // and corrections must be able to withdraw a previously inferred rule.
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<(String, String), Vec<&Episode>> = BTreeMap::new();
        for ep in &self.episodes {
            if ep.success.is_some() {
                groups.entry((ep.domain.clone(), ep.action.clone())).or_default().push(ep);
            }
        }
        self.lessons.clear();
        for ((domain, action), episodes) in groups {
            if episodes.len() < 3 { continue; }
            let success_rate = episodes.iter().filter(|e| e.success == Some(true)).count() as f32
                / episodes.len() as f32;
            let (rule, confidence) = if success_rate > 0.7 {
                (format!("'{}' in {} domain usually works ({:.0}% success rate)", action, domain, success_rate * 100.0), success_rate)
            } else if success_rate < 0.3 {
                (format!("'{}' in {} domain usually fails ({:.0}% success rate) — try different approach", action, domain, success_rate * 100.0), 1.0 - success_rate)
            } else { continue; };
            self.lessons.push(Lesson {
                rule, domain, confidence, evidence_count: episodes.len() as u32,
                first_learned: episodes.iter().map(|e| e.timestamp).fold(f64::INFINITY, f64::min),
                last_reinforced: episodes.iter().map(|e| e.timestamp).fold(0.0, f64::max),
            });
        }
    }
}

#[cfg(test)]
fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episode(id: &str, success: Option<bool>) -> Episode {
        Episode {
            source_ref: id.into(), timestamp: now(), stimulus: "motor callback".into(),
            domain: "science".into(), walker_summary: String::new(), critique: String::new(),
            action: "search".into(), outcome: "observed result".into(), success,
            lesson: String::new(), valence: 0.0, arousal: 0.0, energy: 0.0,
        }
    }

    #[test]
    fn consolidation_counts_unique_scored_evidence_and_corrections() {
        let mut memory = EpisodicMemory::new();
        memory.max_episodes = 4;
        for id in ["a", "b", "c"] {
            memory.record(episode(id, Some(false)));
        }
        memory.record(episode("skip", None));
        assert_eq!(memory.lessons.len(), 1, "buffer full consolidates");
        assert_eq!(memory.lessons[0].evidence_count, 3);
        assert_eq!(memory.lessons[0].confidence, 1.0);
        memory.consolidate();
        memory.record(episode("a", Some(false)));
        assert_eq!(memory.lessons[0].evidence_count, 3);
        memory.record(episode("a", Some(true)));
        memory.consolidate();
        assert!(memory.lessons.is_empty(), "mixed corrected results withdraw the rule");
        assert!(memory.format_history("science", "").contains("unscored"));
    }

    #[test]
    fn recent_memory_is_bounded_even_when_all_records_are_fresh() {
        let mut memory = EpisodicMemory::new();
        memory.max_episodes = 3;
        for i in 0..20 {
            memory.record(episode(&i.to_string(), Some(true)));
        }
        assert_eq!(memory.episodes.len(), 3);
        assert_eq!(memory.episodes[0].source_ref, "17");
        assert_eq!(memory.lessons[0].evidence_count, 3);
    }
}
