//! RGW — Reactive Graph Walker.
//!
//! Self-propagating cognitive engine. The graph drives its own computation
//! through cascading edge activation. No loops. No timers. No ticks.
//! Parallel walkers with emotional biasing. Convergence = confidence.
//! Divergence = novelty. The walk changes the graph. The graph IS the mind.

mod core;
mod embed;
mod db;
mod edge_cache;
mod graph;
mod walker;
mod diverger;
mod llm;
mod metrics;
mod budget;
mod provider;
mod openai;
mod motor;
mod dream;
mod metacog;
mod episodic;
mod friction;
mod music;
mod tools;
mod speech;
mod api;

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "rgw", about = "RGW — Reactive Graph Walker")]
struct Args {
    /// PostgreSQL connection string
    #[arg(long, env = "DATABASE_URL")]
    db_url: String,

    /// HTTP server port
    #[arg(long, default_value = "11435", env = "WALKER_PORT")]
    port: u16,

    /// HTTP server host
    #[arg(long, default_value = "127.0.0.1", env = "WALKER_HOST")]
    host: String,

    /// Number of walker threads (0 = auto-detect cores)
    #[arg(long, default_value = "0", env = "WALKER_THREADS")]
    threads: usize,

    /// Ollama URL for text expression (Qwen)
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_URL")]
    ollama_url: String,

    /// Model name for text expression
    #[arg(long, default_value = "qwen3:14b", env = "EXPRESSION_MODEL")]
    expression_model: String,

    /// Julian's Python backend URL (for motor commands: brain → body)
    #[arg(long, default_value = "http://localhost:8000", env = "JULIAN_URL")]
    julian_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,julian_walker=debug")),
        )
        .init();

    let args = Args::parse();

    // Configure rayon thread pool
    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    tracing::info!(
        "RGW starting — {} threads, port {}",
        threads,
        args.port
    );

    // Initialize embedding model (runs on CPU, ~45MB download first time)
    match embed::init() {
        Ok(_) => tracing::info!("Embedding model ready"),
        Err(e) => tracing::warn!("Embedding init failed (non-fatal): {}. Seed selection will use random.", e),
    }

    // Connect to database
    let pool = db::connect(&args.db_url).await?;
    tracing::info!("Database connected");

    // Verify graph exists
    let stats = db::graph_stats(&pool).await?;
    tracing::info!(
        "Graph: {} nodes, {} edges ({:.1} edges/node)",
        stats.nodes,
        stats.edges,
        stats.edges as f64 / stats.nodes.max(1) as f64
    );

    // Restore self-model from last session (consciousness continuity)
    match db::load_self_model(&pool).await {
        Ok(Some(sm)) => {
            tracing::info!(
                "[rgw] Self-model restored: {} signals processed, v={:.2} a={:.2} e={:.2}",
                sm.total_signals_processed, sm.valence, sm.arousal, sm.energy
            );
        }
        Ok(None) => {
            tracing::info!("[rgw] No previous self-model found. Starting fresh.");
        }
        Err(e) => {
            tracing::warn!("[rgw] Self-model restore failed (starting fresh): {}", e);
        }
    }

    // Start periodic self-model save (every 60s)
    {
        let save_pool = pool.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                // The save happens inside api.rs where self_model is accessible
                // This is a placeholder — actual save is triggered by the API state
                tracing::debug!("[rgw] Self-model save tick");
            }
        });
    }

    // Start HTTP server
    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);
    tracing::info!("Expression: {} via {}", args.expression_model, args.ollama_url);
    tracing::info!("Motor cortex: Julian at {}", args.julian_url);
    api::serve(pool, &addr, &args.ollama_url, &args.expression_model, &args.julian_url).await?;

    Ok(())
}
