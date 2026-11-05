//! Gold-node recall harness: does seeding walks from the query find the right nodes?
//!
//! For every query in a gold set and every arm it posts one `/walk` request to a running
//! server and records the ranked nodes, whether a walker ran, the size of the context and
//! the wall-clock time. It then prints recall per arm and category and the verdict for
//! gates G1 and G1b of `docs/superpowers/specs/2026-11-05-learning-walkers-design.md` §14.
//!
//! The server must run with `RGW_EVAL=1`, otherwise it rejects the `arm` field.
//!
//! Run:
//!   export RGW_BASE_URL=http://localhost:11435
//!   export RGW_API_KEY=...            # sent as X-RGW-Key
//!   cargo run --release --example recall_eval -- <gold.json> \
//!       [--arms fixed,query,vector,ppr] [--walkers 4] [--steps 5] [--seed 1] \
//!       [--out recall_results.json]
//!
//! Further arms: `query:linear`, `drop:<bias>`.
//!
//! A gold set is a JSON array of `{ "query", "gold_node_ids", "category" }` with category
//! `lookup`, `multi_hop`, `temporal` or `unanswerable`. Write it from knowledge of the
//! graph, never from what the current system retrieves. **Gold sets and results describe a
//! deployment's memory: keep them outside the repository and never commit them.**
//! `tests/fixtures/recall_fixture.json` only shows the format; its ids belong to the
//! synthetic test graph, and its query strings do not align with that graph's vectors.

use std::time::{Duration, Instant};

use rgw::evalkit::{judge_gates, summarise, EvalRow, GoldQuery};
use serde_json::Value;

struct Args {
    gold: String,
    arms: Vec<String>,
    walkers: usize,
    steps: usize,
    seed: u64,
    out: String,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        gold: String::new(),
        arms: ["fixed", "query", "vector", "ppr"].iter().map(|s| s.to_string()).collect(),
        walkers: 4,
        steps: 5,
        seed: 1,
        out: "recall_results.json".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| anyhow::anyhow!("{name} needs a value"));
        match arg.as_str() {
            "--arms" => args.arms = value("--arms")?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            "--walkers" => args.walkers = value("--walkers")?.parse()?,
            "--steps" => args.steps = value("--steps")?.parse()?,
            "--seed" => args.seed = value("--seed")?.parse()?,
            "--out" => args.out = value("--out")?,
            other if other.starts_with("--") => anyhow::bail!("unknown option {other}"),
            other => args.gold = other.to_string(),
        }
    }
    if args.gold.is_empty() {
        anyhow::bail!("usage: recall_eval <gold.json> [--arms a,b] [--walkers N] [--steps N] [--seed N] [--out file]");
    }
    Ok(args)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let gold: Vec<GoldQuery> = serde_json::from_str(&std::fs::read_to_string(&args.gold)?)?;
    if gold.is_empty() {
        anyhow::bail!("{} holds no queries", args.gold);
    }
    let base_url = std::env::var("RGW_BASE_URL").unwrap_or_else(|_| "http://localhost:11435".into());
    let token = std::env::var("RGW_API_KEY").unwrap_or_default();
    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;

    let mut rows: Vec<EvalRow> = Vec::new();
    for (i, q) in gold.iter().enumerate() {
        for arm in &args.arms {
            let body = serde_json::json!({
                "stimulus": q.query,
                "n_walkers": args.walkers,
                "steps": args.steps,
                "seed": args.seed,
                "arm": arm,
            });
            let mut request = client.post(format!("{base_url}/walk")).json(&body);
            if !token.is_empty() {
                request = request.header("X-RGW-Key", &token);
            }
            let started = Instant::now();
            let response = request.send().await?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let status = response.status();
            let text = response.text().await?;
            if status == reqwest::StatusCode::BAD_REQUEST && text.contains("RGW_EVAL") {
                anyhow::bail!("the server rejected the arm: start the server with RGW_EVAL=1");
            }
            if !status.is_success() {
                anyhow::bail!("query {} arm {arm}: HTTP {status}: {text}", i + 1);
            }
            let out: Value = serde_json::from_str(&text)?;
            let ranked: Vec<i32> = out["ranked_nodes"]
                .as_array()
                .map(|nodes| nodes.iter().filter_map(|n| n.get(0).and_then(Value::as_i64)).map(|id| id as i32).collect())
                .unwrap_or_default();
            rows.push(EvalRow {
                arm: arm.clone(),
                category: q.category.clone(),
                ranked,
                gold: q.gold_node_ids.clone(),
                walked: out["walked"].as_bool().unwrap_or(true),
                context_chars: out["walker_context"].as_str().map(str::len).unwrap_or(0),
                ms,
            });
        }
        eprintln!("  [{}/{}] {}", i + 1, gold.len(), q.category);
    }

    let summary = summarise(&rows);
    println!("\n{:<14} {:<14} {:>4} {:>10} {:>10} {:>10} {:>8} {:>8}", "arm", "category", "n", "recall@10", "1st gold", "fallback", "tokens", "p95 ms");
    for ((arm, category), s) in &summary {
        println!(
            "{:<14} {:<14} {:>4} {:>10.3} {:>10} {:>10.2} {:>8.0} {:>8.0}",
            arm,
            category,
            s.n,
            s.recall_at_10,
            s.median_first_gold_rank.map(|r| format!("{r:.0}")).unwrap_or_else(|| "-".into()),
            s.fallback_share,
            s.mean_tokens,
            s.p95_ms
        );
    }

    let verdict = judge_gates(&summary);
    println!("\nG1  (seed shadow -> query): {}", if verdict.g1 { "PASS" } else { "FAIL" });
    println!("G1b (keep persona walkers): {}", if verdict.g1b { "PASS" } else { "FAIL" });
    for reason in &verdict.reasons {
        println!("  - {reason}");
    }

    let summary_json: serde_json::Map<String, Value> = summary
        .iter()
        .map(|((arm, category), s)| (format!("{arm}/{category}"), serde_json::to_value(s).unwrap_or(Value::Null)))
        .collect();
    let rows_json: Vec<Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "arm": r.arm, "category": r.category, "ranked": r.ranked, "gold": r.gold,
                "walked": r.walked, "context_chars": r.context_chars, "ms": r.ms,
            })
        })
        .collect();
    let report = serde_json::json!({
        "gold_file": args.gold, "arms": args.arms, "walkers": args.walkers, "steps": args.steps, "seed": args.seed,
        "summary": summary_json, "verdict": verdict, "rows": rows_json,
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nwrote {}", args.out);
    Ok(())
}
