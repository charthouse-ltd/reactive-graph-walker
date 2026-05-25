//! Rust-native text embeddings via fastembed.
//!
//! RGW can now understand text independently — no Python needed.
//! Embeddings power:
//!   - Seed selection (find nodes closest to input stimulus)
//!   - New node creation (embed web search results before storing)
//!   - Similarity computation (edge weight from embedding distance)
//!   - Dream chimera creation (vector math on embeddings)
//!
//! Model: nomic-ai/nomic-embed-text-v1.5 (768-dim) — matches
//! Julian's pgvector embeddings (nomic-embed-text via Ollama).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use dashmap::DashMap;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

lazy_static::lazy_static! {
    static ref EMBEDDER: Mutex<Option<TextEmbedding>> = Mutex::new(None);
    /// Cache identical text → embedding so repeated ingest content (RSS
    /// re-polls, retried webhooks) isn't re-embedded. ~3KB per entry; capped.
    static ref EMBED_CACHE: DashMap<u64, Vec<f32>> = DashMap::new();
}

/// Initialize the embedding model (call once on startup)
pub fn init() -> anyhow::Result<()> {
    // Use NomicEmbedTextV15 (768-dim) to match Julian's pgvector embeddings.
    // This is the same model Julian's Python uses via Ollama (nomic-embed-text).
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::NomicEmbedTextV15).with_show_download_progress(true),
    )?;

    *EMBEDDER.lock().unwrap() = Some(model);
    tracing::info!("[embed] Model loaded: nomic-embed-text-v1.5 (768-dim, matches pgvector)");
    Ok(())
}

/// Embed a single text string. Returns a 768-dim vector.
/// Prepends "search_document:" prefix for pgvector compatibility
/// (matches Julian's Python-embedded memory_vectors format).
pub fn embed_text(text: &str) -> anyhow::Result<Vec<f32>> {
    // Reuse a cached embedding for identical text instead of recomputing.
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let key = hasher.finish();
    if let Some(cached) = EMBED_CACHE.get(&key) {
        return Ok(cached.clone());
    }

    let mut lock = EMBEDDER.lock().unwrap();
    let model = lock.as_mut().ok_or_else(|| anyhow::anyhow!("Embedder not initialized"))?;

    // Do NOT prepend a prefix manually: fastembed auto-applies the nomic
    // "search_document:" passage prefix inside `.embed()`, like Julian's
    // Python `model.embed()`. Prepending here double-prefixes and collapses
    // cosine similarity against Julian's stored vectors → 0 matches.
    let embeddings = model.embed(vec![text], None)?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding produced"))?;

    if EMBED_CACHE.len() > 1024 {
        EMBED_CACHE.clear(); // simple bounded reset; embeddings are cheap to rebuild
    }
    EMBED_CACHE.insert(key, embedding.clone());
    Ok(embedding)
}

/// Embed multiple texts at once (batch processing)
pub fn embed_batch(texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut lock = EMBEDDER.lock().unwrap();
    let model = lock.as_mut().ok_or_else(|| anyhow::anyhow!("Embedder not initialized"))?;

    let texts_owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let embeddings = model.embed(texts_owned, None)?;
    Ok(embeddings)
}

/// Cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Find the most similar text from a list (returns index + similarity)
pub fn most_similar(query: &str, candidates: &[&str]) -> anyhow::Result<Option<(usize, f32)>> {
    let query_emb = embed_text(query)?;
    let candidate_embs = embed_batch(candidates)?;

    let mut best = None;
    for (i, emb) in candidate_embs.iter().enumerate() {
        let sim = cosine_similarity(&query_emb, emb);
        match best {
            None => best = Some((i, sim)),
            Some((_, best_sim)) if sim > best_sim => best = Some((i, sim)),
            _ => {}
        }
    }

    Ok(best)
}
