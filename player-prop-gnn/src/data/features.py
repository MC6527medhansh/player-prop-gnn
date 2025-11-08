"""
Feature engineering functions for player props prediction.
Calculate rolling averages, opponent strength, and match context features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_rolling_averages(
    stats_df: pd.DataFrame,
    player_id: int,
    n_games: int = 5,
    stats_to_roll: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate rolling averages for a player.
    
    Features calculated:
    - Goals per 90 minutes (rolling average)
    - Assists per 90 minutes (rolling average)
    - Shots per 90 minutes (rolling average)
    - Cards per 90 minutes (rolling average)
    
    Args:
        stats_df: DataFrame with columns: player_id, match_date, goals, shots, etc.
        player_id: Player to calculate for
        n_games: Window size (default 5)
        stats_to_roll: List of stat columns to average
        
    Returns:
        DataFrame with rolling average columns added
        
    Notes:
        - Uses shift(1) to exclude current match from rolling average (no lookahead)
        - Handles insufficient data by filling with league averages
        - Converts to per-90 stats before averaging
    """
    if stats_to_roll is None:
        stats_to_roll = ['goals', 'assists', 'shots_on_target', 'yellow_cards', 'red_cards']
    
    # Filter to this player and sort by date
    player_df = stats_df[stats_df['player_id'] == player_id].copy()
    player_df = player_df.sort_values('match_date')
    
    if len(player_df) == 0:
        logger.warning(f"No data for player {player_id}")
        return player_df
    
    # Calculate per-90 stats first
    for stat in stats_to_roll:
        if stat in player_df.columns and 'minutes_played' in player_df.columns:
            # Handle zero minutes (didn't play)
            player_df[f'{stat}_per_90'] = np.where(
                player_df['minutes_played'] > 0,
                (player_df[stat] / player_df['minutes_played'] * 90),
                0
            )
            # Replace inf values (shouldn't happen after above fix, but defensive)
            player_df[f'{stat}_per_90'] = player_df[f'{stat}_per_90'].replace(
                [np.inf, -np.inf], 0
            )
    
    # Calculate rolling averages (excluding current match)
    for stat in stats_to_roll:
        per_90_col = f'{stat}_per_90'
        if per_90_col in player_df.columns:
            # Rolling mean excluding current row (shift(1))
            player_df[f'{stat}_rolling_{n_games}'] = (
                player_df[per_90_col]
                .shift(1)  # Exclude current match to prevent lookahead
                .rolling(window=n_games, min_periods=1)
                .mean()
            )
    
    # League averages for cold start (from World Cup 2018 data analysis)
    LEAGUE_AVERAGES = {
        'goals': 0.45,  # ~0.45 goals per 90 for field players
        'assists': 0.30,  # ~0.30 assists per 90
        'shots_on_target': 1.2,  # ~1.2 shots on target per 90
        'yellow_cards': 0.25,  # ~0.25 yellow cards per 90
        'red_cards': 0.02,  # ~0.02 red cards per 90
    }
    
    # Fill NaN (first match for player) with league averages
    for stat in stats_to_roll:
        roll_col = f'{stat}_rolling_{n_games}'
        if roll_col in player_df.columns:
            player_df[roll_col] = player_df[roll_col].fillna(
                LEAGUE_AVERAGES.get(stat, 0)
            )
    
    return player_df


def calculate_opponent_strength(
    matches_df: pd.DataFrame,
    team_id: int,
    n_games: int = 10
) -> float:
    """
    Calculate defensive strength of opponent team.
    
    Metric: Average goals conceded per game (inverted so higher = stronger defense)
    
    Args:
        matches_df: DataFrame with match results (home_team_id, away_team_id, scores)
        team_id: Team to calculate strength for
        n_games: Number of recent games to consider (default 10)
        
    Returns:
        Defensive strength metric: 
        - 1.0 = league average defense
        - >1.0 = strong defense (concedes fewer goals)
        - <1.0 = weak defense (concedes more goals)
        - Floor at 0.1 to prevent extreme values
        
    Notes:
        - Uses recent form (last n_games) to capture current strength
        - Returns 1.0 (neutral) if insufficient data
    """
    # Get recent matches for this team
    team_matches = matches_df[
        (matches_df['home_team_id'] == team_id) |
        (matches_df['away_team_id'] == team_id)
    ].copy()
    
    # Sort by date and take most recent
    team_matches = team_matches.sort_values('match_date', ascending=False).head(n_games)
    
    if len(team_matches) == 0:
        logger.debug(f"No match history for team {team_id}, returning neutral strength")
        return 1.0  # Neutral strength
    
    # Calculate goals conceded per game
    goals_conceded = []
    for _, match in team_matches.iterrows():
        if match['home_team_id'] == team_id:
            # This team played at home, conceded away_score goals
            goals_conceded.append(match.get('away_score', 0) or 0)
        else:
            # This team played away, conceded home_score goals
            goals_conceded.append(match.get('home_score', 0) or 0)
    
    avg_goals_conceded = np.mean(goals_conceded)
    
    # Invert: Lower goals conceded = stronger defense = higher value
    # Formula: 2.0 - avg_goals_conceded
    # If team concedes 0.5 goals/game: strength = 1.5 (strong)
    # If team concedes 1.5 goals/game: strength = 0.5 (weak)
    # If team concedes 1.0 goals/game: strength = 1.0 (average)
    defensive_strength = 2.0 - avg_goals_conceded
    
    # Floor at 0.1 to prevent negative or zero values
    return max(0.1, defensive_strength)


def add_match_context_features(
    stats_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add match context features to player stats.
    
    Features added:
    - days_since_last_match: Rest days between games
    - opponent_strength: Defensive rating of opponent
    - is_home: Boolean, already exists as 'was_home' in stats_df
    
    Args:
        stats_df: Player match stats DataFrame
        matches_df: Match information DataFrame
        teams_df: Teams DataFrame (optional, for additional context)
        
    Returns:
        stats_df with added context features
        
    Notes:
        - Requires match_date to be datetime type
        - Handles first match (no previous match) by defaulting to 7 days rest
    """
    enriched = stats_df.copy()
    
    # Merge with match info to get dates and teams
    match_info = matches_df[[
        'match_id', 'match_date', 'home_team_id', 'away_team_id',
        'home_score', 'away_score'
    ]].copy()
    
    enriched = enriched.merge(
        match_info,
        on='match_id',
        how='left',
        suffixes=('', '_match')
    )
    
    # Ensure match_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(enriched['match_date']):
        enriched['match_date'] = pd.to_datetime(enriched['match_date'])
    
    # Calculate days since last match for each player
    enriched = enriched.sort_values(['player_id', 'match_date'])
    enriched['days_since_last_match'] = (
        enriched.groupby('player_id')['match_date']
        .diff()
        .dt.days
        .fillna(7)  # Default 7 days rest if first match
    )
    
    # Calculate opponent ID (the team this player is NOT on)
    enriched['opponent_id'] = np.where(
        enriched['was_home'],
        enriched['away_team_id'],
        enriched['home_team_id']
    )
    
    # Calculate opponent strength
    logger.info("Calculating opponent strength ratings...")
    opponent_strengths = {}
    for team_id in enriched['opponent_id'].unique():
        if pd.notna(team_id):
            strength = calculate_opponent_strength(matches_df, int(team_id))
            opponent_strengths[int(team_id)] = strength
    
    enriched['opponent_strength'] = enriched['opponent_id'].map(opponent_strengths)
    enriched['opponent_strength'] = enriched['opponent_strength'].fillna(1.0)
    
    logger.info(f"Added match context features to {len(enriched)} records")
    
    return enriched


def calculate_all_features(
    stats_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
    window_sizes: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Calculate all features for modeling.
    
    This is the main feature engineering pipeline that combines:
    1. Rolling averages (multiple window sizes)
    2. Match context features
    3. Opponent strength
    
    Args:
        stats_df: Player match stats
        matches_df: Match information
        teams_df: Team information (optional)
        window_sizes: List of rolling window sizes (default: [5, 10])
        
    Returns:
        DataFrame with all features calculated
        
    Example:
        >>> features = calculate_all_features(stats_df, matches_df)
        >>> features.columns
        ['player_id', 'match_id', 'goals_rolling_5', 'goals_rolling_10', 
         'opponent_strength', 'days_since_last_match', ...]
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Step 1: Add match context
    logger.info("Step 1: Adding match context features...")
    enriched = add_match_context_features(stats_df, matches_df, teams_df)
    
    # Step 2: Calculate rolling averages for each player
    logger.info("Step 2: Calculating rolling averages...")
    all_player_features = []
    
    unique_players = enriched['player_id'].unique()
    logger.info(f"Calculating features for {len(unique_players)} players")
    
    for idx, player_id in enumerate(unique_players):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(unique_players)} players")
        
        player_data = enriched[enriched['player_id'] == player_id].copy()
        
        # Calculate rolling averages for each window size
        for window in window_sizes:
            player_data = calculate_rolling_averages(
                player_data,
                player_id,
                n_games=window
            )
        
        all_player_features.append(player_data)
    
    # Combine all player features
    features_df = pd.concat(all_player_features, ignore_index=True)
    
    logger.info(f"Feature engineering complete! Generated {len(features_df)} records")
    logger.info(f"Feature columns: {[col for col in features_df.columns if 'rolling' in col]}")
    
    return features_df


def get_feature_summary(features_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of calculated features.
    
    Useful for validation and understanding feature distributions.
    
    Args:
        features_df: DataFrame with calculated features
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_records': len(features_df),
        'n_players': features_df['player_id'].nunique(),
        'date_range': {
            'start': features_df['match_date'].min(),
            'end': features_df['match_date'].max()
        },
        'features': {}
    }
    
    # Get stats for rolling features
    rolling_cols = [col for col in features_df.columns if 'rolling' in col]
    for col in rolling_cols:
        summary['features'][col] = {
            'mean': float(features_df[col].mean()),
            'std': float(features_df[col].std()),
            'min': float(features_df[col].min()),
            'max': float(features_df[col].max()),
            'null_count': int(features_df[col].isnull().sum())
        }
    
    # Opponent strength
    if 'opponent_strength' in features_df.columns:
        summary['features']['opponent_strength'] = {
            'mean': float(features_df['opponent_strength'].mean()),
            'std': float(features_df['opponent_strength'].std()),
            'min': float(features_df['opponent_strength'].min()),
            'max': float(features_df['opponent_strength'].max())
        }
    
    # Days since last match
    if 'days_since_last_match' in features_df.columns:
        summary['features']['days_since_last_match'] = {
            'mean': float(features_df['days_since_last_match'].mean()),
            'median': float(features_df['days_since_last_match'].median()),
            'max': float(features_df['days_since_last_match'].max())
        }
    
    return summary