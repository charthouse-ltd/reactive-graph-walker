//! Database connection and queries for the unified graph model.

use sqlx::{postgres::PgPoolOptions, PgPool, Row};

/// Graph statistics
pub struct GraphStats {
    pub nodes: i64,
    pub edges: i64,
}

/// A memory node in the graph
#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: i32,
    pub domain: String,
    pub valence: f32,
    pub arousal: f32,
    pub importance: f32,
    pub access_count: i32,
    pub embedding: Option<Vec<f32>>,
}

/// An edge between two memory nodes
#[derive(Debug, Clone)]
pub struct MemoryEdge {
    pub id: i32,
    pub source_id: i32,
    pub target_id: i32,
    pub edge_type: String,
    pub weight: f32,
    pub emotional_charge: f32,
    pub traversal_count: i32,
}

/// A knowledge vector — immutable reference material from the compendium.
/// Walkers cross-reference these to ground cognition in verified truth.
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: i32,
    pub skill_name: String,
    pub layer: String,
    pub title: String,
}

/// Connect to PostgreSQL
pub async fn connect(url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(16)
        .connect(url)
        .await
}

/// Get graph statistics
pub async fn graph_stats(pool: &PgPool) -> Result<GraphStats, sqlx::Error> {
    let nodes: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM memory_vectors")
        .fetch_one(pool)
        .await?;

    let edges: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM memory_edges")
        .fetch_optional(pool)
        .await?
        .unwrap_or(Some(0));

    Ok(GraphStats {
        nodes: nodes.unwrap_or(0),
        edges: edges.unwrap_or(0),
    })
}

/// Get edges from a node (outgoing + incoming treated as bidirectional)
pub async fn edges_from(pool: &PgPool, node_id: i32) -> Result<Vec<MemoryEdge>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT id, source_id, target_id, edge_type, \
         COALESCE(weight, 0.5)::real as weight, COALESCE(emotional_charge, 0.0)::real as emotional_charge, \
         COALESCE(traversal_count, 0) as traversal_count \
         FROM memory_edges \
         WHERE source_id = $1 OR target_id = $1 \
         ORDER BY weight DESC \
         LIMIT 20"
    )
    .bind(node_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(|r| MemoryEdge {
        id: r.get("id"),
        source_id: r.get("source_id"),
        target_id: r.get("target_id"),
        edge_type: r.get("edge_type"),
        weight: r.get("weight"),
        emotional_charge: r.get("emotional_charge"),
        traversal_count: r.get("traversal_count"),
    }).collect())
}

/// Get a node by ID (lightweight — no embedding)
pub async fn get_node(pool: &PgPool, id: i32) -> Result<Option<MemoryNode>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT id, COALESCE(domain, '') as domain, \
         COALESCE(valence, 0.0)::real as valence, COALESCE(arousal, 0.5)::real as arousal, \
         COALESCE(importance, 5.0)::real as importance, COALESCE(access_count, 0) as access_count \
         FROM memory_vectors WHERE id = $1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| MemoryNode {
        id: r.get("id"),
        domain: r.get("domain"),
        valence: r.get("valence"),
        arousal: r.get("arousal"),
        importance: r.get("importance"),
        access_count: r.get("access_count"),
        embedding: None,
    }))
}

/// Get seed nodes — high importance, recently accessed
pub async fn seed_nodes(pool: &PgPool, limit: i32) -> Result<Vec<i32>, sqlx::Error> {
    let rows = sqlx::query_scalar::<_, i32>(
        "SELECT id FROM memory_vectors \
         WHERE importance >= 4.0 \
         ORDER BY access_count DESC \
         LIMIT $1"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Strengthen an edge after traversal (the walk changes the graph)
pub async fn strengthen_edge(pool: &PgPool, edge_id: i32, delta: f32) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE memory_edges SET \
         weight = LEAST(1.0, weight + $1), \
         traversal_count = traversal_count + 1, \
         last_traversed = NOW() \
         WHERE id = $2"
    )
    .bind(delta)
    .bind(edge_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Batch strengthen multiple edges
pub async fn strengthen_edges(pool: &PgPool, edge_ids: &[i32], delta: f32) -> Result<(), sqlx::Error> {
    if edge_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "UPDATE memory_edges SET \
         weight = LEAST(1.0, weight + $1), \
         traversal_count = traversal_count + 1, \
         last_traversed = NOW() \
         WHERE id = ANY($2)"
    )
    .bind(delta)
    .bind(edge_ids)
    .execute(pool)
    .await?;
    Ok(())
}

/// Create a new edge between two nodes. Returns the edge ID.
/// This is how the graph GROWS — new connections form from:
/// - Walker discovering cross-domain similarity
/// - Memory compression reinforcing patterns
/// - Web content linking to existing knowledge
pub async fn create_edge(
    pool: &PgPool,
    source_id: i32,
    target_id: i32,
    edge_type: &str,
    weight: f32,
    emotional_charge: f32,
) -> Result<i32, sqlx::Error> {
    let row = sqlx::query(
        "INSERT INTO memory_edges (source_id, target_id, edge_type, weight, emotional_charge, \
         traversal_count, last_traversed, created_at) \
         VALUES ($1, $2, $3, $4, $5, 0, NOW(), NOW()) \
         ON CONFLICT (source_id, target_id, edge_type) \
         DO UPDATE SET weight = LEAST(1.0, memory_edges.weight + $4 * 0.5), \
                       last_traversed = NOW() \
         RETURNING id"
    )
    .bind(source_id)
    .bind(target_id)
    .bind(edge_type)
    .bind(weight)
    .bind(emotional_charge)
    .fetch_one(pool)
    .await?;

    Ok(row.get("id"))
}

/// Synaptic pruning: decay edges that haven't been traversed recently.
/// Edges lose weight over time. Dead edges (weight ≤ 0.01) are deleted.
/// This prevents the graph from becoming a dense hairball.
pub async fn prune_edges(pool: &PgPool, decay_per_day: f32, min_weight: f32) -> Result<(i64, i64), sqlx::Error> {
    // 1. Decay: reduce weight of edges not traversed in the last day
    let decayed = sqlx::query(
        "UPDATE memory_edges SET weight = GREATEST($1, weight - $2) \
         WHERE last_traversed < NOW() - INTERVAL '1 day' \
         AND weight > $1"
    )
    .bind(min_weight)
    .bind(decay_per_day)
    .execute(pool)
    .await?
    .rows_affected() as i64;

    // 2. Delete: remove edges below minimum weight (forgotten)
    let deleted = sqlx::query(
        "DELETE FROM memory_edges WHERE weight <= $1"
    )
    .bind(min_weight)
    .execute(pool)
    .await?
    .rows_affected() as i64;

    Ok((decayed, deleted))
}

// ── Node Creation & Embedding Writes ────────────────────────────
// RGW must be able to write back to the graph. The walker discovers
// patterns → embeds them → creates nodes → creates edges.
// This closes the autonomous growth loop.

/// Create a new memory node with an embedding vector.
/// Returns the new node ID. This is how the graph GROWS autonomously.
pub async fn create_memory_node(
    pool: &PgPool,
    content: &str,
    domain: &str,
    embedding: &[f32],
    importance: f32,
    valence: f32,
    arousal: f32,
) -> Result<i32, sqlx::Error> {
    // pgvector expects the embedding as a formatted string: '[0.1,0.2,...]'
    let emb_str = format!(
        "[{}]",
        embedding.iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let row = sqlx::query(
        "INSERT INTO memory_vectors (document, domain, embedding, importance, valence, arousal, access_count, created_at) \
         VALUES ($1, $2, $3::vector, $4, $5, $6, 0, NOW()) \
         RETURNING id"
    )
    .bind(content)
    .bind(domain)
    .bind(&emb_str)
    .bind(importance)
    .bind(valence)
    .bind(arousal)
    .fetch_one(pool)
    .await?;

    let id: i32 = row.get("id");
    let preview: String = content.chars().take(60).collect();
    tracing::debug!("[db] Created/updated memory node {}: {} (domain={})", id, preview, domain);
    Ok(id)
}

/// Update an existing node's embedding (memory reconsolidation).
/// When the self-model's understanding of a concept changes,
/// the embedding should drift to reflect the new understanding.
pub async fn update_node_embedding(
    pool: &PgPool,
    node_id: i32,
    embedding: &[f32],
) -> Result<(), sqlx::Error> {
    let emb_str = format!(
        "[{}]",
        embedding.iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    sqlx::query(
        "UPDATE memory_vectors SET embedding = $1::vector, updated_at = NOW() WHERE id = $2"
    )
    .bind(&emb_str)
    .bind(node_id)
    .execute(pool)
    .await?;

    tracing::debug!("[db] Updated embedding for node {}", node_id);
    Ok(())
}

/// Find nodes with embeddings similar to the given vector.
/// Uses pgvector's cosine distance operator (<=>).
/// Returns (node_id, similarity_score) ordered by most similar.
pub async fn find_similar_nodes(
    pool: &PgPool,
    embedding: &[f32],
    limit: i32,
    threshold: f32,  // max cosine distance (0 = identical, 2 = opposite)
) -> Result<Vec<(i32, f32, String)>, sqlx::Error> {
    let emb_str = format!(
        "[{}]",
        embedding.iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let rows = sqlx::query(
        "SELECT id, (1.0 - (embedding <=> $1::vector))::real AS similarity, \
                COALESCE(document, '') AS content \
         FROM memory_vectors \
         WHERE embedding IS NOT NULL \
           AND embedding <=> $1::vector < $2 \
         ORDER BY embedding <=> $1::vector \
         LIMIT $3"
    )
    .bind(&emb_str)
    .bind(threshold)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(|r| {
        (r.get("id"), r.get("similarity"), r.get("content"))
    }).collect())
}

/// Update node metadata (importance, access_count, valence, arousal).
pub async fn bump_node(
    pool: &PgPool,
    node_id: i32,
    importance_delta: f32,
    valence_delta: f32,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE memory_vectors SET \
         importance = LEAST(10.0, GREATEST(0.0, importance + $1)), \
         valence = LEAST(1.0, GREATEST(-1.0, valence + $2)), \
         access_count = access_count + 1, \
         updated_at = NOW() \
         WHERE id = $3"
    )
    .bind(importance_delta)
    .bind(valence_delta)
    .bind(node_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Synaptic node pruning: remove nodes with importance below threshold,
/// or nodes that haven't been accessed in a long time.
pub async fn prune_nodes(pool: &PgPool, min_importance: f32, days_untouched: i32) -> Result<i64, sqlx::Error> {
    let deleted = sqlx::query(
        "DELETE FROM memory_vectors \
         WHERE (importance < $1 AND access_count < 3) \
            OR (updated_at < NOW() - ($2 || ' days')::INTERVAL AND importance < 2.0)"
    )
    .bind(min_importance)
    .bind(days_untouched)
    .execute(pool)
    .await?
    .rows_affected() as i64;

    if deleted > 0 {
        tracing::info!("[db] Pruned {} low-importance/stale nodes", deleted);
    }
    Ok(deleted)
}

/// Save self-model state to database (persistence across restarts)
pub async fn save_self_model(pool: &PgPool, model: &crate::core::SelfModel) -> Result<(), sqlx::Error> {
    let json_str = serde_json::to_string(model).unwrap_or_default();
    sqlx::query(
        "INSERT INTO runtime_settings (key, value, description, category, updated_at) \
         VALUES ('rgw_self_model', $1, 'RGW self-model snapshot', 'rgw', NOW()) \
         ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = NOW()"
    )
    .bind(&json_str)
    .execute(pool)
    .await?;
    Ok(())
}

/// Load self-model state from database (restore on startup)
pub async fn load_self_model(pool: &PgPool) -> Result<Option<crate::core::SelfModel>, sqlx::Error> {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT value FROM runtime_settings WHERE key = 'rgw_self_model'"
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|(s,)| serde_json::from_str(&s).ok()))
}

/// Get detailed graph stats for API
pub async fn detailed_stats(pool: &PgPool) -> Result<serde_json::Value, sqlx::Error> {
    let stats = graph_stats(pool).await?;

    let edge_types: Vec<(String, i64, f64)> = sqlx::query_as(
        "SELECT edge_type, COUNT(*)::bigint, AVG(weight)::float8 \
         FROM memory_edges GROUP BY edge_type ORDER BY COUNT(*) DESC"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let domain_dist: Vec<(String, i64)> = sqlx::query_as(
        "SELECT COALESCE(domain, 'unknown'), COUNT(*)::bigint \
         FROM memory_vectors GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 15"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    Ok(serde_json::json!({
        "nodes": stats.nodes,
        "edges": stats.edges,
        "avg_edges_per_node": stats.edges as f64 / stats.nodes.max(1) as f64,
        "edge_types": edge_types.iter().map(|(t, c, w)| serde_json::json!({
            "type": t, "count": c, "avg_weight": w
        })).collect::<Vec<_>>(),
        "domain_distribution": domain_dist.iter().map(|(d, c)| (d.clone(), *c)).collect::<std::collections::HashMap<_, _>>(),
    }))
}

/// Find knowledge vectors matching a domain (skill_name).
/// Used by walkers to cross-reference memory nodes against immutable
/// reference material — grounding cognition in verified truth.
pub async fn find_knowledge_by_domain(
    pool: &PgPool,
    domain: &str,
    limit: i32,
) -> Result<Vec<KnowledgeNode>, sqlx::Error> {
    if domain.is_empty() {
        return Ok(Vec::new());
    }
    let rows = sqlx::query(
        "SELECT id, skill_name, layer, COALESCE(title, '') as title \
         FROM knowledge_vectors \
         WHERE skill_name = $1 \
         LIMIT $2"
    )
    .bind(domain)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(|r| KnowledgeNode {
        id: r.get("id"),
        skill_name: r.get("skill_name"),
        layer: r.get("layer"),
        title: r.get("title"),
    }).collect())
}

/// Memory domains that have a semantically-matching RAG skill in
/// knowledge_vectors. Walkers/dream use this to anchor memory traversal to
/// verified compendium knowledge.
///
/// NOTE: this returns *memory* domains (e.g. "market", "travel"), NOT skill
/// names — the previous version returned skill_names ("market-brief", …),
/// which never matched node.domain, so the interlink was inert. The bridge is
/// semantic: embed each distinct memory domain and keep it if any knowledge
/// vector is within cosine 0.75. Computed once and cached (the corpus is stable
/// at runtime; restart picks up newly-ingested knowledge).
static KNOWLEDGE_DOMAINS: tokio::sync::OnceCell<std::collections::HashSet<String>> =
    tokio::sync::OnceCell::const_new();

pub async fn all_knowledge_domains(pool: &PgPool) -> Result<std::collections::HashSet<String>, sqlx::Error> {
    let set = KNOWLEDGE_DOMAINS
        .get_or_init(|| async { compute_knowledge_domains(pool).await.unwrap_or_default() })
        .await;
    Ok(set.clone())
}

async fn compute_knowledge_domains(pool: &PgPool) -> Result<std::collections::HashSet<String>, sqlx::Error> {
    let domains: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT domain FROM memory_vectors WHERE domain IS NOT NULL AND domain != ''"
    )
    .fetch_all(pool)
    .await?;

    let mut grounded: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (domain,) in domains {
        let emb = match crate::embed::embed_text(&domain) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let emb_str = format!("[{}]", emb.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","));
        let hit: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM knowledge_vectors \
             WHERE embedding IS NOT NULL AND embedding <=> $1::vector < $2 LIMIT 1"
        )
        .bind(&emb_str)
        .bind(0.75_f32)
        .fetch_optional(pool)
        .await?;
        if hit.is_some() {
            grounded.insert(domain);
        }
    }
    tracing::info!("[knowledge] {} memory domains anchored to RAG: {:?}", grounded.len(), grounded);
    Ok(grounded)
}
