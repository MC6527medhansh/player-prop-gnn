-- Teams table FIRST (no dependencies)
CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    league VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table SECOND (depends on teams)
CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(20) NOT NULL CHECK (position IN ('Forward', 'Midfielder', 'Defender', 'Goalkeeper')),
    team_id INTEGER NOT NULL,
    date_of_birth DATE,
    height INTEGER,  -- in cm
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Index for fast lookups by name
CREATE INDEX idx_players_name ON players(name);
CREATE INDEX idx_players_team ON players(team_id);

-- Matches table THIRD (depends on teams)
CREATE TABLE IF NOT EXISTS matches (
    match_id SERIAL PRIMARY KEY,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date DATE NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    league VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    CONSTRAINT fk_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- Index for date-based queries
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);

-- Player match statistics table FOURTH (depends on players and matches)
CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL,
    goals INTEGER DEFAULT 0 CHECK (goals >= 0 AND goals <= 10),
    assists INTEGER DEFAULT 0 CHECK (assists >= 0 AND assists <= 10),
    shots_on_target INTEGER DEFAULT 0 CHECK (shots_on_target >= 0),
    total_shots INTEGER DEFAULT 0 CHECK (total_shots >= shots_on_target),
    yellow_cards INTEGER DEFAULT 0 CHECK (yellow_cards >= 0 AND yellow_cards <= 2),
    red_cards INTEGER DEFAULT 0 CHECK (red_cards >= 0 AND red_cards <= 1),
    minutes_played INTEGER NOT NULL CHECK (minutes_played >= 0 AND minutes_played <= 120),
    was_home BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id),
    CONSTRAINT fk_match FOREIGN KEY (match_id) REFERENCES matches(match_id),
    UNIQUE (player_id, match_id)  -- One record per player per match
);

-- Compound index for time-series queries
CREATE INDEX idx_player_match_stats_player_date 
ON player_match_stats(player_id, match_id);

-- Rolling features table FIFTH (depends on players and matches)
CREATE TABLE IF NOT EXISTS rolling_features (
    feature_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL,  -- Features calculated UP TO this match (not including)
    window_size INTEGER NOT NULL,  -- 5 or 10 games
    goals_per_90 NUMERIC(5,2),
    assists_per_90 NUMERIC(5,2),
    shots_per_90 NUMERIC(5,2),
    cards_per_90 NUMERIC(5,2),
    games_in_window INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_player_rolling FOREIGN KEY (player_id) REFERENCES players(player_id),
    CONSTRAINT fk_match_rolling FOREIGN KEY (match_id) REFERENCES matches(match_id),
    UNIQUE (player_id, match_id, window_size)
);

CREATE INDEX idx_rolling_features_lookup 
ON rolling_features(player_id, match_id, window_size);

-- Bookmaker odds table SIXTH (depends on matches)
CREATE TABLE IF NOT EXISTS bookmaker_odds (
    odds_id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL,  -- 'match_winner', 'over_under', etc.
    odds_value NUMERIC(6,2) NOT NULL CHECK (odds_value >= 1.01),
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_match_odds FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE INDEX idx_bookmaker_odds_match ON bookmaker_odds(match_id, market_type);