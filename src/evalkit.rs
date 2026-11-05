//! Gold-node recall metrics and the G1/G1b verdict (spec §5.1, §14).
//! Pure functions; `examples/recall_eval.rs` drives them against a running server.

use std::collections::BTreeMap;

/// One gold query. `category` is `lookup`, `multi_hop`, `temporal` or `unanswerable`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GoldQuery {
    pub query: String,
    pub gold_node_ids: Vec<i32>,
    pub category: String,
}

/// One query answered by one arm.
#[derive(Debug, Clone)]
pub struct EvalRow {
    pub arm: String,
    pub category: String,
    /// Ranked node ids as the server returned them, best first.
    pub ranked: Vec<i32>,
    pub gold: Vec<i32>,
    pub walked: bool,
    pub context_chars: usize,
    pub ms: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CategorySummary {
    pub n: usize,
    pub recall_at_10: f32,
    pub median_first_gold_rank: Option<f32>,
    /// Share of rows where no walker ran.
    pub fallback_share: f32,
    /// Context characters / 4.
    pub mean_tokens: f32,
    pub p95_ms: f64,
}

pub const RECALL_K: usize = 10;
/// Summary category that pools every category except `unanswerable`.
pub const OVERALL: &str = "overall";
const UNANSWERABLE: &str = "unanswerable";

// Spec §14.
const G1_MULTI_HOP_GAIN: f32 = 0.10;
const G1_LOOKUP_SLACK: f32 = 0.02;
const G1_FALLBACK_SHARE: f32 = 0.70;
const G1_ADDED_P95_MS: f64 = 50.0;
const G1B_PPR_SLACK: f32 = 0.02;
/// Recall is a ratio of small integers; an exact 10-point gain must not fail on f32 rounding.
const GATE_EPS: f32 = 1e-6;

/// Share of the gold nodes found in the first `k` ranked nodes; 0.0 when there is no gold.
pub fn recall_at_k(ranked: &[i32], gold: &[i32], k: usize) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }
    let top = &ranked[..ranked.len().min(k)];
    gold.iter().filter(|g| top.contains(g)).count() as f32 / gold.len() as f32
}

/// 1-based rank of the first gold node, if any is ranked.
pub fn first_gold_rank(ranked: &[i32], gold: &[i32]) -> Option<usize> {
    ranked.iter().position(|id| gold.contains(id)).map(|i| i + 1)
}

fn summarise_group(rows: &[&EvalRow]) -> CategorySummary {
    let n = rows.len();
    let nf = n.max(1) as f32;
    let mut ranks: Vec<f32> = rows.iter().filter_map(|r| first_gold_rank(&r.ranked, &r.gold)).map(|r| r as f32).collect();
    ranks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut ms: Vec<f64> = rows.iter().map(|r| r.ms).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_index = ((0.95 * n as f64).ceil() as usize).clamp(1, n.max(1)) - 1;
    CategorySummary {
        n,
        recall_at_10: rows.iter().map(|r| recall_at_k(&r.ranked, &r.gold, RECALL_K)).sum::<f32>() / nf,
        median_first_gold_rank: (!ranks.is_empty()).then(|| ranks[ranks.len() / 2]),
        fallback_share: rows.iter().filter(|r| !r.walked).count() as f32 / nf,
        mean_tokens: rows.iter().map(|r| r.context_chars as f32 / 4.0).sum::<f32>() / nf,
        p95_ms: ms.get(p95_index).copied().unwrap_or(0.0),
    }
}

/// `(arm, category) -> summary`, plus an `overall` category per arm that leaves out `unanswerable`.
pub fn summarise(rows: &[EvalRow]) -> BTreeMap<(String, String), CategorySummary> {
    let mut groups: BTreeMap<(String, String), Vec<&EvalRow>> = BTreeMap::new();
    for row in rows {
        groups.entry((row.arm.clone(), row.category.clone())).or_default().push(row);
        if row.category != UNANSWERABLE {
            groups.entry((row.arm.clone(), OVERALL.to_string())).or_default().push(row);
        }
    }
    groups.into_iter().map(|(key, group)| (key, summarise_group(&group))).collect()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GateVerdict {
    pub g1: bool,
    pub g1b: bool,
    pub reasons: Vec<String>,
}

/// Spec §14. G1: `query` beats `fixed` on multi_hop recall by at least 0.10; `query` is within
/// 0.02 of `vector` on lookup; `query` falls back on at least 0.70 of unanswerable; the p95 of
/// `query` exceeds that of `fixed` by at most 50 ms (overall). G1b: `ppr` beats `query` overall
/// by at most 0.02. A missing arm or category makes the corresponding gate false with a reason.
/// Recall thresholds are compared with a 1e-6 tolerance, so a gain of exactly the threshold passes.
pub fn judge_gates(summary: &BTreeMap<(String, String), CategorySummary>) -> GateVerdict {
    let get = |arm: &str, category: &str| summary.get(&(arm.to_string(), category.to_string()));
    let mut reasons = Vec::new();
    let mut g1 = true;
    let mut g1b = true;

    match (get("query", "multi_hop"), get("fixed", "multi_hop")) {
        (Some(q), Some(f)) => {
            if q.recall_at_10 - f.recall_at_10 < G1_MULTI_HOP_GAIN - GATE_EPS {
                g1 = false;
                reasons.push(format!(
                    "G1: query multi_hop recall {:.3} is not {:.2} above fixed {:.3}",
                    q.recall_at_10, G1_MULTI_HOP_GAIN, f.recall_at_10
                ));
            }
        }
        _ => {
            g1 = false;
            reasons.push("G1: missing multi_hop results for query or fixed".to_string());
        }
    }
    match (get("query", "lookup"), get("vector", "lookup")) {
        (Some(q), Some(v)) => {
            if v.recall_at_10 - q.recall_at_10 > G1_LOOKUP_SLACK + GATE_EPS {
                g1 = false;
                reasons.push(format!(
                    "G1: query lookup recall {:.3} is more than {:.2} below vector {:.3}",
                    q.recall_at_10, G1_LOOKUP_SLACK, v.recall_at_10
                ));
            }
        }
        _ => {
            g1 = false;
            reasons.push("G1: missing lookup results for query or vector".to_string());
        }
    }
    match get("query", UNANSWERABLE) {
        Some(q) => {
            if q.fallback_share < G1_FALLBACK_SHARE - GATE_EPS {
                g1 = false;
                reasons.push(format!(
                    "G1: query fell back on {:.2} of unanswerable queries, below {:.2}",
                    q.fallback_share, G1_FALLBACK_SHARE
                ));
            }
        }
        None => {
            g1 = false;
            reasons.push("G1: missing unanswerable results for query".to_string());
        }
    }
    match (get("query", OVERALL), get("fixed", OVERALL)) {
        (Some(q), Some(f)) => {
            if q.p95_ms - f.p95_ms > G1_ADDED_P95_MS {
                g1 = false;
                reasons.push(format!(
                    "G1: query p95 {:.0} ms exceeds fixed p95 {:.0} ms by more than {:.0} ms",
                    q.p95_ms, f.p95_ms, G1_ADDED_P95_MS
                ));
            }
        }
        _ => {
            g1 = false;
            reasons.push("G1: missing overall results for query or fixed".to_string());
        }
    }
    match (get("ppr", OVERALL), get("query", OVERALL)) {
        (Some(p), Some(q)) => {
            if p.recall_at_10 - q.recall_at_10 > G1B_PPR_SLACK + GATE_EPS {
                g1b = false;
                reasons.push(format!(
                    "G1b: ppr overall recall {:.3} beats query {:.3} by more than {:.2}",
                    p.recall_at_10, q.recall_at_10, G1B_PPR_SLACK
                ));
            }
        }
        _ => {
            g1b = false;
            reasons.push("G1b: missing overall results for ppr or query".to_string());
        }
    }
    GateVerdict { g1, g1b, reasons }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_and_first_rank() {
        assert_eq!(recall_at_k(&[9, 5, 7], &[5, 6], 10), 0.5);
        assert_eq!(recall_at_k(&[9, 5, 7], &[5, 6], 1), 0.0);
        assert_eq!(recall_at_k(&[1], &[], 10), 0.0);
        assert_eq!(first_gold_rank(&[9, 5, 7], &[7, 5]), Some(2));
        assert_eq!(first_gold_rank(&[9], &[5]), None);
    }

    fn row(arm: &str, cat: &str, ranked: Vec<i32>, gold: Vec<i32>, walked: bool, ms: f64) -> EvalRow {
        EvalRow { arm: arm.into(), category: cat.into(), ranked, gold, walked, context_chars: 400, ms }
    }

    #[test]
    fn summary_is_per_arm_and_category_with_an_overall_that_skips_unanswerable() {
        let rows = vec![
            row("query", "multi_hop", vec![1, 2], vec![1, 2], true, 10.0),
            row("query", "multi_hop", vec![3], vec![1, 2], true, 30.0),
            row("query", "unanswerable", vec![], vec![], false, 5.0),
        ];
        let s = summarise(&rows);
        let mh = &s[&("query".to_string(), "multi_hop".to_string())];
        assert_eq!((mh.n, mh.recall_at_10), (2, 0.5));
        assert_eq!(mh.mean_tokens, 100.0);
        assert_eq!(s[&("query".to_string(), "unanswerable".to_string())].fallback_share, 1.0);
        assert_eq!(s[&("query".to_string(), "overall".to_string())].n, 2);
    }

    fn cs(recall: f32, fallback: f32, p95: f64) -> CategorySummary {
        CategorySummary { n: 20, recall_at_10: recall, median_first_gold_rank: Some(1.0), fallback_share: fallback, mean_tokens: 100.0, p95_ms: p95 }
    }
    fn table(entries: &[(&str, &str, CategorySummary)]) -> std::collections::BTreeMap<(String, String), CategorySummary> {
        entries.iter().map(|(a, c, s)| ((a.to_string(), c.to_string()), s.clone())).collect()
    }

    #[test]
    fn gates_pass_on_the_spec_thresholds_and_fail_just_inside_them() {
        let pass = table(&[
            ("fixed", "multi_hop", cs(0.30, 0.0, 40.0)), ("query", "multi_hop", cs(0.41, 0.0, 60.0)),
            ("vector", "lookup", cs(0.80, 1.0, 5.0)),    ("query", "lookup", cs(0.79, 0.0, 60.0)),
            ("query", "unanswerable", cs(0.0, 0.70, 5.0)),
            ("fixed", "overall", cs(0.30, 0.0, 40.0)),   ("query", "overall", cs(0.60, 0.0, 90.0)),
            ("ppr", "overall", cs(0.61, 0.0, 20.0)),
        ]);
        let v = judge_gates(&pass);
        assert!(v.g1 && v.g1b, "{:?}", v.reasons);

        let mut fail = pass.clone();
        fail.insert(("query".into(), "multi_hop".into()), cs(0.39, 0.0, 60.0));
        fail.insert(("ppr".into(), "overall".into()), cs(0.63, 0.0, 20.0));
        let v = judge_gates(&fail);
        assert!(!v.g1 && !v.g1b);
        assert_eq!(v.reasons.len(), 2);
    }

    #[test]
    fn a_missing_arm_fails_its_gate_with_a_reason() {
        let v = judge_gates(&table(&[("query", "overall", cs(0.5, 0.0, 10.0))]));
        assert!(!v.g1 && !v.g1b && !v.reasons.is_empty());
    }

    #[test]
    fn gates_pass_exactly_on_the_boundary() {
        // 0.40 - 0.30 is 0.099999994 in f32; an exact 10-point gain must still pass.
        let exact = table(&[
            ("fixed", "multi_hop", cs(0.30, 0.0, 40.0)), ("query", "multi_hop", cs(0.40, 0.0, 60.0)),
            ("vector", "lookup", cs(0.80, 1.0, 5.0)),    ("query", "lookup", cs(0.78, 0.0, 60.0)),
            ("query", "unanswerable", cs(0.0, 0.70, 5.0)),
            ("fixed", "overall", cs(0.30, 0.0, 40.0)),   ("query", "overall", cs(0.60, 0.0, 90.0)),
            ("ppr", "overall", cs(0.62, 0.0, 20.0)),
        ]);
        let v = judge_gates(&exact);
        assert!(v.g1 && v.g1b, "{:?}", v.reasons);
    }
}
