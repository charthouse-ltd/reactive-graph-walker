//! A/B evaluation harness — does the RGW substrate beat a plain LLM + retrieval?
//!
//! For each scenario it asks the running server twice on identical input:
//!   arm A = full RGW path        (rgw_bypass: false)
//!   arm B = retrieval-only base  (rgw_bypass: true)
//! then has DeepSeek blind-judge the pair (randomized order) on three axes —
//! on-brand, continuity, specificity — and tallies win-rates plus latency and
//! the run's LLM cost delta from /metrics.
//!
//! This turns "I think the substrate helps" into a number. Pre-register your
//! decision rule BEFORE running (see PR description / the audit thread).
//!
//! Run:
//!   export RGW_BASE_URL=http://localhost:11435      # or your prod URL
//!   export RGW_API_KEY=...                            # X-RGW-Key for protected routes
//!   export DEEPSEEK_API_KEY=sk-...                    # the blind judge
//!   export RGW_JUDGE_MODEL=deepseek-chat              # optional (default shown)
//!   cargo run --release --example ab_eval -- ab_scenarios.json
//!
//! Writes ab_results.json and prints a summary.

use std::time::Duration;

use rand::Rng;
use serde_json::Value;

#[derive(Clone)]
struct ArmResult {
    text: String,
    total_ms: f64,
    action: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "ab_scenarios.json".into());
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;

    let base_url = std::env::var("RGW_BASE_URL")
        .ok()
        .or_else(|| spec["base_url"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "http://localhost:11435".into());
    let token = std::env::var("RGW_API_KEY").unwrap_or_default();
    let judge_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    let judge_model = std::env::var("RGW_JUDGE_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
    let rubric = spec["brand_rubric"]
        .as_str()
        .unwrap_or("A distinctive, consistent persona voice — specific and opinionated, never generic AI-speak.")
        .to_string();

    let scenarios = spec["scenarios"].as_array().cloned().unwrap_or_default();
    if scenarios.is_empty() {
        anyhow::bail!("no scenarios in {path} (expected a top-level \"scenarios\" array)");
    }
    if judge_key.is_empty() {
        anyhow::bail!("DEEPSEEK_API_KEY not set — needed for the blind judge");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let cost_before = fetch_cost(&client, &base_url).await;

    // axis -> [full_wins, bypass_wins, ties]
    let mut tally: std::collections::HashMap<&str, [u32; 3]> = [
        ("on_brand", [0, 0, 0]),
        ("continuity", [0, 0, 0]),
        ("specificity", [0, 0, 0]),
        ("overall", [0, 0, 0]),
    ]
    .into_iter()
    .collect();

    let mut full_ms_sum = 0.0;
    let mut bypass_ms_sum = 0.0;
    let mut rows: Vec<Value> = Vec::new();

    for sc in &scenarios {
        let id = sc["id"].as_str().unwrap_or("?").to_string();
        let stimulus = sc["stimulus"].as_str().unwrap_or("").to_string();
        if stimulus.is_empty() {
            continue;
        }

        let full = complete(&client, &base_url, &token, &stimulus, false).await?;
        let bypass = complete(&client, &base_url, &token, &stimulus, true).await?;
        full_ms_sum += full.total_ms;
        bypass_ms_sum += bypass.total_ms;

        // Randomize which arm is shown as "Response 1" so the judge stays blind.
        let full_is_first = rand::rng().random::<bool>();
        let (r1, r2) = if full_is_first {
            (&full.text, &bypass.text)
        } else {
            (&bypass.text, &full.text)
        };

        let verdict = judge(&client, &judge_key, &judge_model, &rubric, &stimulus, r1, r2).await?;

        let mut row = serde_json::json!({
            "id": id,
            "full_action": full.action,
            "bypass_action": bypass.action,
            "full_is_first": full_is_first,
        });
        for axis in ["on_brand", "continuity", "specificity", "overall"] {
            let pick = verdict.get(axis).and_then(|v| v.as_str()).unwrap_or("tie");
            // Map the judge's "1"/"2"/"tie" back to full/bypass/tie.
            let slot = match pick {
                "1" | "2" => {
                    let picked_first = pick == "1";
                    if picked_first == full_is_first { 0 } else { 1 } // 0=full, 1=bypass
                }
                _ => 2, // tie
            };
            tally.get_mut(axis).unwrap()[slot] += 1;
            row[axis] = serde_json::json!(["full", "bypass", "tie"][slot]);
        }
        row["reason"] = verdict.get("reason").cloned().unwrap_or(Value::Null);
        println!("[{}] overall={}", row["id"], row["overall"]);
        rows.push(row);
    }

    let cost_after = fetch_cost(&client, &base_url).await;
    let n = rows.len().max(1) as f64;

    let summary = serde_json::json!({
        "scenarios": rows.len(),
        "win_rates": tally.iter().map(|(axis, [f, b, t])| {
            let total = (f + b + t).max(1) as f64;
            (axis.to_string(), serde_json::json!({
                "full_pct": (*f as f64 / total * 100.0).round(),
                "bypass_pct": (*b as f64 / total * 100.0).round(),
                "tie_pct": (*t as f64 / total * 100.0).round(),
            }))
        }).collect::<serde_json::Map<_, _>>(),
        "avg_latency_ms": {"full": (full_ms_sum / n).round(), "bypass": (bypass_ms_sum / n).round()},
        "run_cost_usd": (cost_after - cost_before).max(0.0),
    });

    std::fs::write(
        "ab_results.json",
        serde_json::to_string_pretty(&serde_json::json!({"summary": summary, "rows": rows}))?,
    )?;

    println!("\n=== A/B SUMMARY (full RGW vs retrieval-only baseline) ===");
    println!("{}", serde_json::to_string_pretty(&summary)?);
    println!("\nWrote ab_results.json. Compare against your pre-registered decision rule.");
    Ok(())
}

/// Call /v1/chat/completions for one arm.
async fn complete(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    stimulus: &str,
    bypass: bool,
) -> anyhow::Result<ArmResult> {
    let body = serde_json::json!({
        "model": "rgw",
        "messages": [{"role": "user", "content": stimulus}],
        "rgw_bypass": bypass,
    });
    let mut req = client.post(format!("{base_url}/v1/chat/completions")).json(&body);
    if !token.is_empty() {
        req = req.header("X-RGW-Key", token);
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned {} for bypass={bypass}", resp.status());
    }
    let data: Value = resp.json().await?;
    Ok(ArmResult {
        text: data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
        total_ms: data["rgw_metadata"]["total_ms"].as_f64().unwrap_or(0.0),
        action: data["rgw_metadata"]["walker_action"].as_str().unwrap_or("?").to_string(),
    })
}

/// Blind pairwise judge via DeepSeek. Returns the parsed verdict object.
async fn judge(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    rubric: &str,
    stimulus: &str,
    r1: &str,
    r2: &str,
) -> anyhow::Result<Value> {
    let system = format!(
        "You are a blind, impartial judge of a persona's writing. Brand rubric:\n{rubric}\n\n\
         You are given a prompt and two candidate responses. Decide which response \
         is better on each axis. Reply with ONLY a JSON object, no prose:\n\
         {{\"on_brand\":\"1|2|tie\",\"continuity\":\"1|2|tie\",\"specificity\":\"1|2|tie\",\"overall\":\"1|2|tie\",\"reason\":\"...\"}}\n\
         on_brand = fits the rubric's voice; continuity = reads as a consistent ongoing \
         persona, not a stateless reply; specificity = concrete and particular vs generic."
    );
    let user = format!(
        "PROMPT:\n{stimulus}\n\n--- RESPONSE 1 ---\n{r1}\n\n--- RESPONSE 2 ---\n{r2}"
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    });
    let resp = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;
    let data: Value = resp.json().await?;
    let content = data["choices"][0]["message"]["content"].as_str().unwrap_or("{}");
    // Extract the first {...} block in case the model wraps it in prose/markdown.
    let json_slice = match (content.find('{'), content.rfind('}')) {
        (Some(s), Some(e)) if e > s => &content[s..=e],
        _ => "{}",
    };
    Ok(serde_json::from_str(json_slice).unwrap_or_else(|_| serde_json::json!({})))
}

/// Read lifetime est_cost_usd from /metrics (0.0 if unavailable).
async fn fetch_cost(client: &reqwest::Client, base_url: &str) -> f64 {
    match client.get(format!("{base_url}/metrics")).send().await {
        Ok(r) => r.json::<Value>().await.ok().and_then(|v| v["est_cost_usd"].as_f64()).unwrap_or(0.0),
        Err(_) => 0.0,
    }
}
