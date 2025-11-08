#!/usr/bin/env python3
"""
Calculate features for 1720 records using sqlalchemy.
"""
import sys
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, '.')

from src.data.features import calculate_all_features

# Connect to database
print("Connecting to database...")
engine = create_engine("postgresql://medhanshchoubey@localhost:5432/football_props")

# Load data
print("Loading player_match_stats...")
stats_df = pd.read_sql("""
    SELECT 
        pms.player_id, pms.match_id, pms.was_home,
        pms.goals, pms.assists, pms.shots_on_target,
        pms.yellow_cards, pms.red_cards, pms.minutes_played,
        m.match_date,
        p.team_id
    FROM player_match_stats pms
    JOIN matches m ON pms.match_id = m.match_id
    JOIN players p ON pms.player_id = p.player_id
    ORDER BY m.match_date, pms.player_id
""", engine)
print(f"✓ Loaded {len(stats_df)} records")

print("Loading matches...")
matches_df = pd.read_sql("""
    SELECT match_id, home_team_id, away_team_id, match_date, home_score, away_score
    FROM matches ORDER BY match_date
""", engine)
print(f"✓ Loaded {len(matches_df)} matches")

engine.dispose()

# Calculate features
print("\nCalculating features...")
features_df = calculate_all_features(stats_df, matches_df, window_sizes=[5, 10])
print(f"✓ Calculated {len(features_df)} feature records")

# Check for issues
rolling_cols = [c for c in features_df.columns if 'rolling' in c]
print(f"\nFeatures: {len(rolling_cols)} rolling columns")

nan_counts = features_df[rolling_cols].isnull().sum()
if nan_counts.any():
    print(f"⚠️  NaN found: {nan_counts[nan_counts > 0].to_dict()}")
else:
    print("✓ No NaN values")

# Save
features_df.to_csv('player_features.csv', index=False)
print(f"\n✓ Saved to player_features.csv")
print(f"  Shape: {features_df.shape}")

print("\nSample:")
print(features_df[['player_id', 'match_date', 'goals', 'goals_rolling_5', 'opponent_strength']].head(3).to_string())

print("\n✅ DONE")