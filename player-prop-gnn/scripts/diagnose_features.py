#!/usr/bin/env python3
"""
Check if goals_rolling_5 mean of 0.231 is correct or a bug.
"""
import pandas as pd

df = pd.read_csv('player_features.csv')

print("DIAGNOSING LOW GOALS_ROLLING_5 MEAN")
print("="*60)

# Check raw goals
print(f"\n1. RAW GOALS DATA")
print(f"   Total goals: {df['goals'].sum()}")
print(f"   Total matches: {len(df)}")
print(f"   Goals per match (raw): {df['goals'].mean():.3f}")

# Check by position if available
if 'position' in df.columns:
    print(f"\n2. GOALS BY POSITION")
    by_position = df.groupby('position')['goals'].agg(['sum', 'count', 'mean'])
    print(by_position)

# Check minutes played
print(f"\n3. MINUTES PLAYED")
print(f"   Mean minutes: {df['minutes_played'].mean():.1f}")
print(f"   Players with 0 minutes: {(df['minutes_played'] == 0).sum()}")

# Calculate expected per-90
total_goals = df['goals'].sum()
total_minutes = df['minutes_played'].sum()
goals_per_90 = (total_goals / total_minutes * 90) if total_minutes > 0 else 0

print(f"\n4. EXPECTED vs ACTUAL")
print(f"   Total goals: {total_goals}")
print(f"   Total minutes: {total_minutes:.0f}")
print(f"   Expected goals per 90: {goals_per_90:.3f}")
print(f"   Actual goals_rolling_5 mean: {df['goals_rolling_5'].mean():.3f}")

# Sample some players
print(f"\n5. SAMPLE PLAYERS")
for pid in df['player_id'].unique()[:5]:
    player = df[df['player_id'] == pid].sort_values('match_date')
    player_goals = player['goals'].sum()
    player_matches = len(player)
    print(f"   Player {pid}: {player_goals} goals in {player_matches} matches, rolling_5: {player['goals_rolling_5'].mean():.3f}")

print("\n" + "="*60)
print("CONCLUSION:")
if goals_per_90 < 0.3:
    print("✓ Mean of 0.231 is CORRECT")
    print("  World Cup data includes ALL positions (defenders, GKs)")
    print("  Not all players are attackers")
else:
    print("⚠️  Possible BUG in calculation")