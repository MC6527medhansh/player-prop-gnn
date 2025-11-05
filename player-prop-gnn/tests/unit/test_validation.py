"""
Unit tests for data validation functions.
Run with: pytest tests/unit/test_validation.py -v
"""
import pytest
from datetime import date, timedelta
from src.data.validation import (
    validate_player_data,
    validate_match_data,
    validate_player_match_stats,
    validate_dataframe,
    ValidationError
)
import pandas as pd


class TestPlayerValidation:
    """Test player data validation."""
    
    def test_valid_player(self):
        """Test validation of valid player data."""
        player = {
            'name': 'Mohamed Salah',
            'position': 'Forward',
            'team_id': 1
        }
        # Should not raise
        validate_player_data(player)
    
    def test_missing_name(self):
        """Test validation fails when name is missing."""
        player = {
            'position': 'Forward',
            'team_id': 1
        }
        with pytest.raises(ValidationError, match="Missing required field: name"):
            validate_player_data(player)
    
    def test_empty_name(self):
        """Test validation fails when name is empty."""
        player = {
            'name': '   ',
            'position': 'Forward',
            'team_id': 1
        }
        with pytest.raises(ValidationError, match="Player name cannot be empty"):
            validate_player_data(player)
    
    def test_invalid_position(self):
        """Test validation fails for invalid position."""
        player = {
            'name': 'Test Player',
            'position': 'INVALID',
            'team_id': 1
        }
        with pytest.raises(ValidationError, match="Invalid position"):
            validate_player_data(player)
    
    def test_valid_positions(self):
        """Test all valid positions pass validation."""
        valid_positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
        for position in valid_positions:
            player = {
                'name': 'Test Player',
                'position': position,
                'team_id': 1
            }
            validate_player_data(player)
    
    def test_invalid_team_id(self):
        """Test validation fails for invalid team_id."""
        player = {
            'name': 'Test Player',
            'position': 'Forward',
            'team_id': 0  # Must be positive
        }
        with pytest.raises(ValidationError, match="Invalid team_id"):
            validate_player_data(player)


class TestMatchValidation:
    """Test match data validation."""
    
    def test_valid_match(self):
        """Test validation of valid match data."""
        match = {
            'home_team_id': 1,
            'away_team_id': 2,
            'match_date': '2025-11-01'
        }
        # Should not raise
        validate_match_data(match)
    
    def test_same_teams(self):
        """Test validation fails when home and away teams are same."""
        match = {
            'home_team_id': 1,
            'away_team_id': 1,  # Same as home
            'match_date': '2025-11-01'
        }
        with pytest.raises(ValidationError, match="teams must be different"):
            validate_match_data(match)
    
    def test_invalid_date_format(self):
        """Test validation fails for invalid date format."""
        match = {
            'home_team_id': 1,
            'away_team_id': 2,
            'match_date': 'not-a-date'
        }
        with pytest.raises(ValidationError, match="Invalid date format"):
            validate_match_data(match)
    
    def test_far_future_date(self):
        """Test validation fails for date too far in future."""
        future_date = (date.today() + timedelta(days=400)).strftime('%Y-%m-%d')
        match = {
            'home_team_id': 1,
            'away_team_id': 2,
            'match_date': future_date
        }
        with pytest.raises(ValidationError, match="too far in future"):
            validate_match_data(match)
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        match = {
            'home_team_id': 1,
            # Missing away_team_id
            'match_date': '2025-11-01'
        }
        with pytest.raises(ValidationError, match="Missing required field"):
            validate_match_data(match)


class TestPlayerStatsValidation:
    """Test player match statistics validation."""
    
    def test_valid_stats(self):
        """Test validation of valid player stats."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 90,
            'goals': 2,
            'shots_on_target': 5,
            'total_shots': 8,
            'yellow_cards': 1,
            'red_cards': 0
        }
        # Should not raise
        validate_player_match_stats(stats)
    
    def test_invalid_goals(self):
        """Test validation fails for invalid goals."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 90,
            'goals': 11  # Too many
        }
        with pytest.raises(ValidationError, match="Invalid goals"):
            validate_player_match_stats(stats)
    
    def test_shots_on_target_exceeds_total(self):
        """Test validation fails when shots on target > total shots."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 90,
            'shots_on_target': 10,
            'total_shots': 5  # Less than shots on target
        }
        with pytest.raises(ValidationError, match="Shots on target"):
            validate_player_match_stats(stats)
    
    def test_invalid_minutes(self):
        """Test validation fails for invalid minutes."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 150  # Too many
        }
        with pytest.raises(ValidationError, match="Invalid minutes_played"):
            validate_player_match_stats(stats)
    
    def test_invalid_yellow_cards(self):
        """Test validation fails for too many yellow cards."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 90,
            'yellow_cards': 3  # Max is 2
        }
        with pytest.raises(ValidationError, match="Invalid yellow_cards"):
            validate_player_match_stats(stats)
    
    def test_invalid_red_cards(self):
        """Test validation fails for too many red cards."""
        stats = {
            'player_id': 1,
            'match_id': 1,
            'minutes_played': 90,
            'red_cards': 2  # Max is 1
        }
        with pytest.raises(ValidationError, match="Invalid red_cards"):
            validate_player_match_stats(stats)


class TestDataFrameValidation:
    """Test DataFrame validation."""
    
    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'match_id': [1, 1, 1],
            'goals': [0, 1, 2]
        })
        required_columns = ['player_id', 'match_id', 'goals']
        # Should not raise
        validate_dataframe(df, required_columns)
    
    def test_missing_columns(self):
        """Test validation fails when columns are missing."""
        df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'match_id': [1, 1, 1]
            # Missing 'goals' column
        })
        required_columns = ['player_id', 'match_id', 'goals']
        with pytest.raises(ValidationError, match="Missing columns"):
            validate_dataframe(df, required_columns)
    
    def test_null_values(self):
        """Test validation fails when nulls present."""
        df = pd.DataFrame({
            'player_id': [1, 2, None],  # Null value
            'match_id': [1, 1, 1],
            'goals': [0, 1, 2]
        })
        required_columns = ['player_id', 'match_id', 'goals']
        with pytest.raises(ValidationError, match="Null values found"):
            validate_dataframe(df, required_columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])