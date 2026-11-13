//! A small graph in its own Postgres schema, with synthetic embeddings.
//!
//! Each test uses an isolated schema and synthetic embeddings, without downloads.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

pub const DIM: usize = 768;

pub const HUB: i32 = 1;
pub const A1: i32 = 2;
pub const A2: i32 = 3;
pub const A3: i32 = 4;
pub const B1: i32 = 5;
pub const B2: i32 = 6;
pub const B3: i32 = 7;
pub const B4: i32 = 8;
pub const LONE: i32 = 9;

/// Unit vector along `axis`.
pub fn axis(axis: usize) -> Vec<f32> {
    mix(&[(axis, 1.0)])
}

/// Normalised weighted sum of axes.
pub fn mix(parts: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for &(a, w) in parts {
        v[a] += w;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn vector_literal(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
}

fn base_url() -> String {
    std::env::var("DATABASE_URL").expect("DATABASE_URL must explicitly select a throwaway test database")
}

/// A pool whose connections use `search_path = <schema>, public`. Drops and
/// recreates the schema, runs tests/schema.sql inside it, and inserts the graph.
/// `schema` must be a plain identifier; each test passes its own.
pub async fn fixture_pool(schema: &str) -> anyhow::Result<PgPool> {
    anyhow::ensure!(
        schema.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "schema name must be a plain identifier"
    );
    let url = base_url();
    let admin = PgPoolOptions::new().max_connections(1).connect(&url).await?;
    sqlx::raw_sql(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};"))
        .execute(&admin)
        .await?;
    admin.close().await;

    let schema_owned = schema.to_string();
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .after_connect(move |conn, _meta| {
            let s = schema_owned.clone();
            Box::pin(async move {
                sqlx::query(&format!("SET search_path TO {s}, public")).execute(&mut *conn).await?;
                Ok(())
            })
        })
        .connect(&url)
        .await?;
    sqlx::raw_sql(include_str!("../schema.sql")).execute(&pool).await?;

    // (id, domain, importance, access_count, embedding)
    let nodes: Vec<(i32, &str, f32, i32, Vec<f32>)> = vec![
        (HUB, "hub", 9.0, 100, mix(&[(0, 0.5), (1, 0.5)])),
        (A1, "alpha", 9.0, 90, axis(0)),
        (A2, "alpha", 3.0, 0, mix(&[(0, 0.9), (3, 0.1)])),
        (A3, "alpha", 3.0, 0, mix(&[(0, 0.8), (3, 0.2)])),
        (B1, "beta", 3.0, 0, axis(1)),
        (B2, "beta", 3.0, 0, mix(&[(1, 0.9), (4, 0.1)])),
        (B3, "beta", 3.0, 0, mix(&[(1, 0.8), (4, 0.2)])),
        (B4, "beta", 3.0, 0, mix(&[(1, 0.7), (4, 0.3)])),
        (LONE, "gamma", 3.0, 0, axis(2)),
    ];
    for (id, domain, importance, access_count, emb) in &nodes {
        sqlx::query(
            "INSERT INTO memory_vectors (id, document, domain, embedding, importance, access_count, doc_id, post_id, title, mood, persona, location, source_type, published_at, month, word_count, stored_at, last_accessed, valence, arousal, created_at, updated_at) \
             VALUES ($1, $2, $3, $4::vector, $5, $6, $1::text, 0, '', '', '', '', 'fixture', '', '', 0, 0, 0, 0, 0.5, NOW(), NOW())",
        )
        .bind(id)
        .bind(format!("fixture node {id} in {domain}"))
        .bind(domain)
        .bind(vector_literal(emb))
        .bind(importance)
        .bind(access_count)
        .execute(&pool)
        .await?;
    }
    sqlx::query("SELECT setval(pg_get_serial_sequence('memory_vectors', 'id'), 100)")
        .execute(&pool)
        .await?;

    // (source, target, type, weight)
    let edges: Vec<(i32, i32, &str, f32)> = vec![
        (HUB, A1, "related", 0.9),
        (HUB, A2, "related", 0.5),
        (HUB, A3, "related", 0.5),
        (HUB, B1, "related", 0.3),
        (A1, A2, "related", 0.8),
        (A2, A3, "related", 0.7),
        (B1, B2, "related", 0.8),
        (B2, B3, "related", 0.8),
        (B3, B4, "related", 0.8),
        (B1, B3, "reinforces", 0.4),
        (LONE, A3, "related", 0.2),
    ];
    for (s, t, ty, w) in &edges {
        sqlx::query("INSERT INTO memory_edges (source_id, target_id, edge_type, weight, emotional_charge, traversal_count, last_traversed, created_at) VALUES ($1, $2, $3, $4, 0, 0, NOW(), NOW())")
            .bind(s)
            .bind(t)
            .bind(ty)
            .bind(w)
            .execute(&pool)
            .await?;
    }
    Ok(pool)
}
