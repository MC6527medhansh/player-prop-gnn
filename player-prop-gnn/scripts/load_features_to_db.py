#!/usr/bin/env python3
"""
Load player_features.csv into database.
Usage: python scripts/load_features_to_db.py
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# DB config
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='football_props',
    user='medhanshchoubey',
    password=''
)

# Load CSV
print("Loading CSV...")
df = pd.read_csv('player_features.csv')
print(f"Loaded {len(df)} rows")

# Convert date
df['match_date'] = pd.to_datetime(df['match_date']).dt.date

# Columns to insert (matching CSV, skip match_date_match duplicate)
cols = [
    'player_id', 'match_id', 'was_home', 'goals', 'assists',
    'shots_on_target', 'yellow_cards', 'red_cards', 'minutes_played',
    'match_date', 'team_id', 'home_team_id', 'away_team_id',
    'home_score', 'away_score', 'days_since_last_match',
    'opponent_id', 'opponent_strength', 'goals_per_90',
    'assists_per_90', 'shots_on_target_per_90', 'yellow_cards_per_90',
    'red_cards_per_90', 'goals_rolling_5', 'assists_rolling_5',
    'shots_on_target_rolling_5', 'yellow_cards_rolling_5',
    'red_cards_rolling_5', 'goals_rolling_10', 'assists_rolling_10',
    'shots_on_target_rolling_10', 'yellow_cards_rolling_10',
    'red_cards_rolling_10'
]

# Prepare data
data = [tuple(row) for row in df[cols].values]

# Insert
print("Inserting...")
cursor = conn.cursor()
query = f"""
    INSERT INTO player_features ({', '.join(cols)})
    VALUES %s
    ON CONFLICT (player_id, match_id) DO NOTHING
"""
execute_values(cursor, query, data)
conn.commit()

# Verify
cursor.execute("SELECT COUNT(*) FROM player_features")
count = cursor.fetchone()[0]
print(f"Done! {count} rows in player_features")

cursor.close()
conn.close()