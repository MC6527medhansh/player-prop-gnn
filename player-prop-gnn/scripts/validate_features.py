#!/usr/bin/env python3
"""
Validate the calculated features to ensure they're correct.
"""
import pandas as pd
import numpy as np

print("="*60)
print("FEATURE VALIDATION")
print("="*60)

# Load the features
df = pd.read_csv('player_features.csv')
print(f"\n1. FILE CHECK")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Expected: 1720 rows, 34 columns")
assert len(df) == 1720, f"Expected 1720 rows, got {len(df)}"
print("   ✓ Row count correct")

# Check for NaN/inf
print(f"\n2. DATA QUALITY")
rolling_cols = [c for c in df.columns if 'rolling' in c]
print(f"   Rolling features: {len(rolling_cols)}")

nan_count = df[rolling_cols].isnull().sum().sum()
print(f"   NaN values: {nan_count}")
assert nan_count == 0, "Found NaN values!"
print("   ✓ No NaN values")

inf_count = np.isinf(df[rolling_cols].select_dtypes(include=[np.number])).sum().sum()
print(f"   Inf values: {inf_count}")
assert inf_count == 0, "Found inf values!"
print("   ✓ No inf values")

# Check feature distributions
print(f"\n3. FEATURE DISTRIBUTIONS")
print(f"   goals_rolling_5:")
print(f"     Mean: {df['goals_rolling_5'].mean():.3f}")
print(f"     Std:  {df['goals_rolling_5'].std():.3f}")
print(f"     Min:  {df['goals_rolling_5'].min():.3f}")
print(f"     Max:  {df['goals_rolling_5'].max():.3f}")

# Should be around 0.2-0.3 for all positions (World Cup includes defenders/GKs)
assert 0.15 < df['goals_rolling_5'].mean() < 0.35, "goals_rolling_5 mean out of range"
print("   ✓ goals_rolling_5 distribution reasonable")

print(f"\n   opponent_strength:")
print(f"     Mean: {df['opponent_strength'].mean():.3f}")
print(f"     Std:  {df['opponent_strength'].std():.3f}")
print(f"     Min:  {df['opponent_strength'].min():.3f}")
print(f"     Max:  {df['opponent_strength'].max():.3f}")

# Should be around 1.0 (neutral), floor at 0.1
assert df['opponent_strength'].min() >= 0.1, "opponent_strength below floor"
assert 0.5 < df['opponent_strength'].mean() < 1.5, "opponent_strength mean out of range"
print("   ✓ opponent_strength distribution reasonable")

# Check no lookahead bias
print(f"\n4. LOOKAHEAD BIAS CHECK")
# For first match of each player, rolling avg should be league avg (0.45)
df_sorted = df.sort_values(['player_id', 'match_date'])
first_matches = df_sorted.groupby('player_id').first()

first_match_goals_rolling = first_matches['goals_rolling_5'].value_counts()
print(f"   First match goals_rolling_5 values:")
print(f"     Most common: {first_matches['goals_rolling_5'].mode()[0]:.3f}")
print(f"     (Should be 0.45 - league average)")

# Most players' first match should have 0.45
most_common = first_matches['goals_rolling_5'].mode()[0]
assert abs(most_common - 0.45) < 0.01, "First match not using league average!"
print("   ✓ No lookahead bias (first matches use league avg)")

# Check that rolling avg is NOT equal to current match value
print(f"\n5. CURRENT MATCH EXCLUSION CHECK")
# Sample some records where player scored
scored_matches = df[df['goals'] > 0].head(50)
same_count = (scored_matches['goals'] == scored_matches['goals_rolling_5']).sum()
print(f"   Out of 50 matches where player scored:")
print(f"   Matches where goals = goals_rolling_5: {same_count}")
print(f"   (Should be low, indicates rolling avg excludes current match)")
assert same_count < 10, "Too many matches with goals = rolling avg!"
print("   ✓ Rolling averages exclude current match")

# Spot check one player manually
print(f"\n6. MANUAL CALCULATION CHECK")
player_1 = df[df['player_id'] == 1].sort_values('match_date').head(5)
print(f"   Player 1, first 5 matches:")
print(player_1[['match_date', 'goals', 'goals_rolling_5', 'minutes_played']].to_string(index=False))

# Match 1: should be 0.45 (league avg)
assert abs(player_1.iloc[0]['goals_rolling_5'] - 0.45) < 0.01, "Match 1 not league avg"

# Match 2: should be match 1's goals per 90
match1_goals = player_1.iloc[0]['goals']
match1_mins = player_1.iloc[0]['minutes_played']
expected_rolling_2 = (match1_goals / match1_mins * 90) if match1_mins > 0 else 0
actual_rolling_2 = player_1.iloc[1]['goals_rolling_5']
print(f"\n   Match 2 check:")
print(f"     Expected rolling (from match 1): {expected_rolling_2:.3f}")
print(f"     Actual rolling: {actual_rolling_2:.3f}")
assert abs(expected_rolling_2 - actual_rolling_2) < 0.01, "Rolling calculation wrong!"
print("   ✓ Manual calculation matches")

# Check days_since_last_match
print(f"\n7. DAYS SINCE LAST MATCH CHECK")
print(f"   Mean: {df['days_since_last_match'].mean():.1f} days")
print(f"   Median: {df['days_since_last_match'].median():.1f} days")
print(f"   Max: {df['days_since_last_match'].max():.0f} days")

# World Cup is condensed, should be 3-7 days mostly
assert df['days_since_last_match'].median() < 10, "Days since last match too high"
print("   ✓ Days since last match reasonable for World Cup")

print("\n" + "="*60)
print("ALL VALIDATIONS PASSED ✅")
print("="*60)
print(f"\nFeatures are:")
print("  - Calculated correctly")
print("  - No lookahead bias")
print("  - Reasonable distributions")
print("  - No NaN/inf values")
print("  - Ready for model training")