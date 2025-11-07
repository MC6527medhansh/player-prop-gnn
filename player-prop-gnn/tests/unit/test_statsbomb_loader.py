"""
Unit tests for StatsBomb loader.
Run with: pytest test_statsbomb_loader.py -v
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from statsbomb_loader import StatsBombLoader


class TestStatsBombLoader:
    """Test StatsBomb data loading functionality"""
    
    @pytest.fixture
    def loader(self):
        """Create a StatsBomb loader instance"""
        return StatsBombLoader()
    
    def test_get_available_competitions(self, loader):
        """Test fetching available competitions"""
        comps = loader.get_available_competitions()
        
        # Should return a DataFrame
        assert isinstance(comps, pd.DataFrame)
        
        # Should have required columns
        assert 'competition_id' in comps.columns
        assert 'competition_name' in comps.columns
        assert 'season_id' in comps.columns
        
        # Should have multiple competitions
        assert len(comps) > 0
        
        # World Cup should be available
        assert 'FIFA World Cup' in comps['competition_name'].values
    
    def test_get_world_cup_matches(self, loader):
        """Test fetching World Cup 2018 matches"""
        # Get World Cup competition
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        assert len(wc_2018) > 0, "World Cup 2018 not found"
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        # Get matches
        matches = loader.get_matches(comp_id, season_id)
        
        # Should return DataFrame
        assert isinstance(matches, pd.DataFrame)
        
        # Should have required columns
        assert 'match_id' in matches.columns
        assert 'match_date' in matches.columns
        assert 'home_team' in matches.columns
        assert 'away_team' in matches.columns
        
        # World Cup 2018 had 64 matches
        assert len(matches) == 64
    
    def test_load_match_events(self, loader):
        """Test loading events for a single match"""
        # Get a World Cup 2018 match
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        matches = loader.get_matches(comp_id, season_id)
        match_id = int(matches.iloc[0]['match_id'])
        
        # Load events
        events = loader.load_match_events(match_id)
        
        # Should return DataFrame
        assert isinstance(events, pd.DataFrame)
        
        # Should have events
        assert len(events) > 0
        
        # Should have key columns
        assert 'player_id' in events.columns
        assert 'player_name' in events.columns
        assert 'type' in events.columns
        assert 'team_name' in events.columns
        
        # Should have different event types
        assert 'Pass' in events['type'].values
        assert 'Shot' in events['type'].values or 'Carry' in events['type'].values
    
    def test_extract_player_match_stats(self, loader):
        """Test extracting stats for a specific player"""
        # Get a World Cup 2018 match
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        matches = loader.get_matches(comp_id, season_id)
        match_id = int(matches.iloc[0]['match_id'])
        
        events = loader.load_match_events(match_id)
        
        # Get first player with events
        player_id = int(events['player_id'].dropna().iloc[0])
        
        # Extract stats
        stats = loader.extract_player_match_stats(events, player_id)
        
        # Should return dict
        assert isinstance(stats, dict)
        
        # Should have required fields
        assert 'player_id' in stats
        assert 'player_name' in stats
        assert 'team_name' in stats
        assert 'goals' in stats
        assert 'assists' in stats
        assert 'total_shots' in stats
        assert 'shots_on_target' in stats
        assert 'yellow_cards' in stats
        assert 'red_cards' in stats
        assert 'minutes_played' in stats
        
        # Values should be valid
        assert stats['goals'] >= 0
        assert stats['total_shots'] >= 0
        assert stats['shots_on_target'] >= 0
        assert stats['shots_on_target'] <= stats['total_shots']
        assert stats['yellow_cards'] >= 0
        assert stats['yellow_cards'] <= 2
        assert stats['red_cards'] >= 0
        assert stats['red_cards'] <= 1
        assert 0 <= stats['minutes_played'] <= 120
    
    def test_extract_player_match_stats_no_events(self, loader):
        """Test extracting stats for player with no events"""
        # Create empty events DataFrame
        events = pd.DataFrame(columns=['player_id', 'type'])
        
        # Try to extract stats for non-existent player
        stats = loader.extract_player_match_stats(events, 99999)
        
        # Should return None
        assert stats is None
    
    def test_extract_pass_network(self, loader):
        """Test extracting pass network for a team"""
        # Get a World Cup 2018 match
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        matches = loader.get_matches(comp_id, season_id)
        match = matches.iloc[0]
        match_id = int(match['match_id'])
        
        events = loader.load_match_events(match_id)
        team_name = str(match['home_team'])
        
        # Extract pass network
        pass_net = loader.extract_pass_network(events, team_name)
        
        # Should return DataFrame
        assert isinstance(pass_net, pd.DataFrame)
        
        if len(pass_net) > 0:
            # Should have required columns
            assert 'passer_id' in pass_net.columns
            assert 'receiver_id' in pass_net.columns
            assert 'pass_count' in pass_net.columns
            
            # Pass counts should be positive
            assert (pass_net['pass_count'] > 0).all()
    
    def test_load_world_cup_2018_full(self, loader):
        """Test loading complete World Cup 2018 dataset"""
        # This is a slow test - only run if explicitly requested
        pytest.skip("Slow test - only run manually")
        
        data = loader.load_world_cup_2018()
        
        # Should return dict with expected keys
        assert 'matches' in data
        assert 'player_stats' in data
        assert 'pass_networks' in data
        
        # Matches
        matches = data['matches']
        assert isinstance(matches, pd.DataFrame)
        assert len(matches) == 64
        
        # Player stats
        player_stats = data['player_stats']
        assert isinstance(player_stats, pd.DataFrame)
        assert len(player_stats) > 0
        
        # Should have ~22 players per match = ~1400 records
        # (Some players may be substituted)
        assert len(player_stats) > 1000
        
        # Check data quality
        assert player_stats['goals'].sum() > 100  # World Cup 2018 had 169 goals
        assert (player_stats['goals'] >= 0).all()
        assert (player_stats['total_shots'] >= 0).all()
        assert (player_stats['shots_on_target'] <= player_stats['total_shots']).all()
        
        print(f"\nLoaded {len(player_stats)} player-match records")
        print(f"Total goals: {player_stats['goals'].sum()}")
        print(f"Total assists: {player_stats['assists'].sum()}")
        print(f"Total shots: {player_stats['total_shots'].sum()}")
    
    def test_position_mapping(self, loader):
        """Test position mapping from StatsBomb to our schema"""
        # Test various position strings
        assert loader._map_position('Left Wing') == 'Forward'
        assert loader._map_position('Center Forward') == 'Forward'
        assert loader._map_position('Left Center Midfield') == 'Midfielder'
        assert loader._map_position('Right Center Back') == 'Defender'
        assert loader._map_position('Goalkeeper') == 'Goalkeeper'
        
        # Test fallback
        assert loader._map_position('Unknown Position') == 'Midfielder'
    
    def test_card_extraction(self, loader):
        """Test extracting cards from events"""
        # Create test events with cards
        events = pd.DataFrame({
            'foul_committed_card': ['Yellow Card', 'Second Yellow', 'Red Card', None, 'Yellow Card']
        })
        
        cards = loader._extract_cards(events)
        
        assert cards['yellow'] >= 2  # At least 2 yellow cards
        assert cards['red'] >= 1  # At least 1 red card
    
    def test_assists_counting(self, loader):
        """Test assist counting logic"""
        # Create mock events with a pass leading to a goal
        events = pd.DataFrame({
            'player_id': [1, 2, 3],
            'type': ['Pass', 'Shot', 'Pass'],
            'pass_recipient_id': [2, None, 3],
            'shot_outcome': [None, 'Goal', None],
            'second': [10, 12, 20]
        })
        
        # Player 1 passed to player 2, who scored
        assists = loader._count_assists(events, 1)
        assert assists == 1
        
        # Player 3 didn't assist
        assists = loader._count_assists(events, 3)
        assert assists == 0


class TestDataQuality:
    """Test data quality and validation"""
    
    def test_no_null_player_names(self):
        """Ensure all player stats have names"""
        loader = StatsBombLoader()
        
        # Load a small sample
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        matches = loader.get_matches(comp_id, season_id)
        match_id = int(matches.iloc[0]['match_id'])
        
        events = loader.load_match_events(match_id)
        
        # Extract stats for all players
        for player_id in events['player_id'].dropna().unique():
            stats = loader.extract_player_match_stats(events, player_id)
            if stats:
                assert stats['player_name'] is not None
                assert stats['player_name'] != ''
    
    def test_shot_logic_consistency(self):
        """Ensure shots on target <= total shots"""
        loader = StatsBombLoader()
        
        # Load a small sample
        comps = loader.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        matches = loader.get_matches(comp_id, season_id)
        match_id = int(matches.iloc[0]['match_id'])
        
        events = loader.load_match_events(match_id)
        
        # Check all players
        for player_id in events['player_id'].dropna().unique():
            stats = loader.extract_player_match_stats(events, player_id)
            if stats:
                assert stats['shots_on_target'] <= stats['total_shots'], \
                    f"Player {stats['player_name']}: shots_on_target > total_shots"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])