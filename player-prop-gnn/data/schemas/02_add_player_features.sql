-- Add player_features table matching actual CSV structure
-- Run: psql football_props -f data/schemas/add_player_features_table.sql

CREATE TABLE IF NOT EXISTS player_features (
    feature_id SERIAL PRIMARY KEY,
    
    -- From CSV (exact column names)
    player_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL,
    was_home BOOLEAN NOT NULL,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    minutes_played INTEGER NOT NULL,
    match_date DATE NOT NULL,
    team_id INTEGER,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    days_since_last_match NUMERIC(5,1),
    opponent_id INTEGER,
    opponent_strength NUMERIC(5,3),
    
    -- Per-90 stats
    goals_per_90 NUMERIC(6,3),
    assists_per_90 NUMERIC(6,3),
    shots_on_target_per_90 NUMERIC(6,3),
    yellow_cards_per_90 NUMERIC(6,3),
    red_cards_per_90 NUMERIC(6,3),
    
    -- Rolling features (5-game)
    goals_rolling_5 NUMERIC(6,3),
    assists_rolling_5 NUMERIC(6,3),
    shots_on_target_rolling_5 NUMERIC(6,3),
    yellow_cards_rolling_5 NUMERIC(6,3),
    red_cards_rolling_5 NUMERIC(6,3),
    
    -- Rolling features (10-game)
    goals_rolling_10 NUMERIC(6,3),
    assists_rolling_10 NUMERIC(6,3),
    shots_on_target_rolling_10 NUMERIC(6,3),
    yellow_cards_rolling_10 NUMERIC(6,3),
    red_cards_rolling_10 NUMERIC(6,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id),
    CONSTRAINT fk_match FOREIGN KEY (match_id) REFERENCES matches(match_id),
    UNIQUE (player_id, match_id)
);

-- Indexes
CREATE INDEX idx_pf_player ON player_features(player_id);
CREATE INDEX idx_pf_match ON player_features(match_id);
CREATE INDEX idx_pf_date ON player_features(match_date);