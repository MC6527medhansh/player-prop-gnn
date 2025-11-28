-- Phase 6: Graph Interaction Table
-- Stores the "edges" for the GNN (Pass Network)

CREATE TABLE IF NOT EXISTS match_interactions (
    interaction_id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL,
    sender_id INTEGER NOT NULL,
    receiver_id INTEGER NOT NULL,
    interaction_type VARCHAR(20) NOT NULL DEFAULT 'pass',
    success BOOLEAN DEFAULT TRUE,
    
    -- Metadata for potential edge features
    timestamp_second INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_match_inter FOREIGN KEY (match_id) REFERENCES matches(match_id),
    CONSTRAINT fk_sender FOREIGN KEY (sender_id) REFERENCES players(player_id),
    CONSTRAINT fk_receiver FOREIGN KEY (receiver_id) REFERENCES players(player_id)
);

-- Compound index for fast graph building: "Get all edges for match X"
CREATE INDEX idx_interactions_match ON match_interactions(match_id);