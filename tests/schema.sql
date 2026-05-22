-- RGW Test Schema — PostgreSQL + pgvector
-- Memory graph: nodes with embeddings, edges with emotional charge.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory nodes: concepts, facts, experiences with semantic embeddings
CREATE TABLE IF NOT EXISTS memory_vectors (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    domain VARCHAR(100) DEFAULT '',
    embedding vector(768),
    importance REAL DEFAULT 5.0,
    valence REAL DEFAULT 0.0,
    arousal REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Memory edges: connections between nodes with emotional charge
CREATE TABLE IF NOT EXISTS memory_edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memory_vectors(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL DEFAULT 'related',
    weight REAL DEFAULT 0.5,
    emotional_charge REAL DEFAULT 0.0,
    traversal_count INTEGER DEFAULT 0,
    last_traversed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, edge_type)
);

-- Runtime settings (self-model persistence)
CREATE TABLE IF NOT EXISTS runtime_settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memory_vectors_domain ON memory_vectors(domain);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_importance ON memory_vectors(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_memory_edges_target ON memory_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_memory_edges_type ON memory_edges(edge_type);
