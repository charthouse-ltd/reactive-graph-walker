-- Julian Alembic contract: 0001_baseline + 0006_add_rgw_episodes.
-- Python/ORM defaults deliberately are NOT database defaults.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memory_vectors (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(200) NOT NULL UNIQUE,
    document TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    embedding_tq BYTEA,
    post_id BIGINT NOT NULL,
    title VARCHAR(300) NOT NULL,
    mood VARCHAR(50) NOT NULL,
    persona VARCHAR(100) NOT NULL,
    location VARCHAR(300) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    published_at VARCHAR(30) NOT NULL,
    month VARCHAR(7) NOT NULL,
    word_count INTEGER NOT NULL,
    stored_at DOUBLE PRECISION NOT NULL,
    access_count INTEGER NOT NULL,
    last_accessed DOUBLE PRECISION NOT NULL,
    importance DOUBLE PRECISION NOT NULL,
    valence DOUBLE PRECISION NOT NULL,
    arousal DOUBLE PRECISION NOT NULL,
    residue TEXT,
    historical_anchors JSONB,
    extra JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
    edge_type VARCHAR(30) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    emotional_charge DOUBLE PRECISION NOT NULL,
    traversal_count INTEGER NOT NULL,
    last_traversed TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL,
    CONSTRAINT uq_edge_pair_type UNIQUE(source_id, target_id, edge_type)
);
CREATE TABLE IF NOT EXISTS runtime_settings (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS rgw_episodes (
    id SERIAL PRIMARY KEY,
    occurred_at TIMESTAMP NOT NULL,
    kind VARCHAR(20) NOT NULL,
    location VARCHAR(160),
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    headline TEXT NOT NULL,
    detail TEXT,
    valence DOUBLE PRECISION,
    arousal DOUBLE PRECISION,
    energy DOUBLE PRECISION,
    source_type VARCHAR(40) NOT NULL,
    source_ref VARCHAR(80) NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    CONSTRAINT uq_rgw_episodes_source UNIQUE(source_type, source_ref)
);
CREATE INDEX IF NOT EXISTS ix_edges_source_weight ON memory_edges(source_id, weight);
CREATE INDEX IF NOT EXISTS ix_edges_type_weight ON memory_edges(edge_type, weight);
CREATE INDEX IF NOT EXISTS ix_rgw_episodes_occurred_at ON rgw_episodes(occurred_at);
CREATE INDEX IF NOT EXISTS ix_rgw_episodes_location ON rgw_episodes(location);
