"""
Data validation functions.
Validate data BEFORE inserting into database to catch errors early.
"""
from datetime import datetime, date
from typing import Dict, List, Optional
import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails"""
    pass


def validate_player_data(player: Dict) -> None:
    """
    Validate player data before insertion.
    
    Args:
        player: Dict with keys: name, position, team_id
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['name', 'position', 'team_id']
    for field in required_fields:
        if field not in player:
            raise ValidationError(f"Missing required field: {field}")
    
    # Name should not be empty
    if not player['name'] or player['name'].strip() == '':
        raise ValidationError("Player name cannot be empty")
    
    # Position must be valid
    valid_positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    if player['position'] not in valid_positions:
        raise ValidationError(f"Invalid position: {player['position']}")
    
    # team_id must be positive integer
    if not isinstance(player['team_id'], int) or player['team_id'] <= 0:
        raise ValidationError(f"Invalid team_id: {player['team_id']}")


def validate_match_data(match: Dict) -> None:
    """
    Validate match data before insertion.
    
    Args:
        match: Dict with keys: home_team_id, away_team_id, match_date
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['home_team_id', 'away_team_id', 'match_date']
    for field in required_fields:
        if field not in match:
            raise ValidationError(f"Missing required field: {field}")
    
    # Teams must be different
    if match['home_team_id'] == match['away_team_id']:
        raise ValidationError("Home and away teams must be different")
    
    # Match date must be valid date
    match_date = match['match_date']
    if isinstance(match_date, str):
        try:
            match_date = datetime.strptime(match_date, '%Y-%m-%d').date()
        except ValueError:
            raise ValidationError(f"Invalid date format: {match_date}")
    
    # Match date should not be in far future
    if match_date > date.today() + pd.Timedelta(days=365):
        raise ValidationError(f"Match date too far in future: {match_date}")


def validate_player_match_stats(stats: Dict) -> None:
    """
    Validate player match statistics before insertion.
    
    Args:
        stats: Dict with player performance data
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['player_id', 'match_id', 'minutes_played']
    for field in required_fields:
        if field not in stats:
            raise ValidationError(f"Missing required field: {field}")
    
    # Goals should be non-negative and reasonable
    if 'goals' in stats:
        if not 0 <= stats['goals'] <= 10:
            raise ValidationError(f"Invalid goals: {stats['goals']}")
    
    # Shots on target cannot exceed total shots
    if 'shots_on_target' in stats and 'total_shots' in stats:
        if stats['shots_on_target'] > stats['total_shots']:
            raise ValidationError(
                f"Shots on target ({stats['shots_on_target']}) > "
                f"total shots ({stats['total_shots']})"
            )
    
    # Minutes played must be valid
    if not 0 <= stats['minutes_played'] <= 120:
        raise ValidationError(f"Invalid minutes_played: {stats['minutes_played']}")
    
    # Cards validation
    if 'yellow_cards' in stats:
        if not 0 <= stats['yellow_cards'] <= 2:
            raise ValidationError(f"Invalid yellow_cards: {stats['yellow_cards']}")
    
    if 'red_cards' in stats:
        if not 0 <= stats['red_cards'] <= 1:
            raise ValidationError(f"Invalid red_cards: {stats['red_cards']}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate a DataFrame has required columns and no nulls.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        
    Raises:
        ValidationError: If validation fails
    """
    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing columns: {missing_cols}")
    
    # Check for nulls in required columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValidationError(f"Null values found: {null_cols}")