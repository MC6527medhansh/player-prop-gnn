from statsbombpy import sb
import pandas as pd
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatsBombLoader:
    
    @staticmethod
    def get_available_competitions():
        return sb.competitions()
    
    @staticmethod
    def get_matches(competition_id, season_id):
        return sb.matches(competition_id=competition_id, season_id=season_id)
    
    @staticmethod
    def load_match_events(match_id):
        events = sb.events(match_id=match_id)
        return events
    
    @staticmethod
    def extract_player_match_stats(events, player_id):
        player_events = events[events['player_id'] == player_id].copy()
        
        if player_events.empty:
            return None
        
        # Get player name - handle different column names
        player_name = None
        if 'player' in player_events.columns:
            player_name = str(player_events['player'].iloc[0])
        elif 'player_name' in player_events.columns:
            player_name = str(player_events['player_name'].iloc[0])
        else:
            player_name = f"Player_{player_id}"
        
        # Get team name
        team_name = None
        if 'team' in player_events.columns:
            team_name = str(player_events['team'].iloc[0])
        elif 'team_name' in player_events.columns:
            team_name = str(player_events['team_name'].iloc[0])
        else:
            team_name = "Unknown"
        
        stats = {
            'player_id': int(player_id),
            'player_name': player_name,
            'team_name': team_name,
        }
        
        # Extract position
        if 'position' in player_events.columns:
            position = player_events['position'].iloc[0]
            if pd.notna(position):
                stats['position'] = str(position)
        
        # Goals
        shots = player_events[player_events['type'] == 'Shot']
        if 'shot_outcome' in shots.columns:
            stats['goals'] = int(len(shots[shots['shot_outcome'] == 'Goal']))
            stats['total_shots'] = int(len(shots))
            stats['shots_on_target'] = int(len(shots[shots['shot_outcome'].isin(['Goal', 'Saved', 'Saved To Post'])]))
        else:
            stats['goals'] = 0
            stats['total_shots'] = 0
            stats['shots_on_target'] = 0
        
        # Assists
        stats['assists'] = StatsBombLoader._count_assists(events, player_id)
        
        # Cards
        cards = StatsBombLoader._extract_cards(player_events)
        stats['yellow_cards'] = cards['yellow']
        stats['red_cards'] = cards['red']
        
        # Minutes
        if 'minute' in player_events.columns:
            max_minute = player_events['minute'].max()
            stats['minutes_played'] = int(max_minute) if pd.notna(max_minute) else 0
        else:
            stats['minutes_played'] = 0
        
        return stats
    
    @staticmethod
    def _count_assists(events, player_id):
        assists = 0
        player_passes = events[(events['player_id'] == player_id) & (events['type'] == 'Pass')].copy()
        
        if player_passes.empty:
            return 0
        
        pass_recipient_col = 'pass_recipient_id' if 'pass_recipient_id' in player_passes.columns else 'pass_recipient'
        if pass_recipient_col not in player_passes.columns:
            return 0
        
        for idx, pass_event in player_passes.iterrows():
            recipient_id = pass_event.get(pass_recipient_col)
            if pd.isna(recipient_id):
                continue
            
            pass_time = pass_event.get('second', 0)
            next_events = events[
                (events['second'] > pass_time) &
                (events['second'] <= pass_time + 10) &
                (events['player_id'] == recipient_id) &
                (events['type'] == 'Shot')
            ]
            
            if not next_events.empty and 'shot_outcome' in next_events.columns:
                if (next_events['shot_outcome'] == 'Goal').any():
                    assists += 1
        
        return assists
    
    @staticmethod
    def _extract_cards(player_events):
        yellow_count = 0
        red_count = 0
        
        if 'foul_committed_card' in player_events.columns:
            card_events = player_events[player_events['foul_committed_card'].notna()]
            for card in card_events['foul_committed_card']:
                card_str = str(card).lower()
                if 'yellow' in card_str:
                    yellow_count += 1
                if 'red' in card_str or 'second yellow' in card_str:
                    red_count += 1
        
        return {'yellow': yellow_count, 'red': red_count}
    
    def load_world_cup_2018(self):
        comps = self.get_available_competitions()
        wc = comps[comps['competition_name'] == 'FIFA World Cup']
        wc_2018 = wc[wc['season_name'] == '2018']
        
        if wc_2018.empty:
            raise ValueError("World Cup 2018 not found")
        
        comp_id = int(wc_2018.iloc[0]['competition_id'])
        season_id = int(wc_2018.iloc[0]['season_id'])
        
        logger.info(f"Found World Cup 2018: competition_id={comp_id}, season_id={season_id}")
        
        matches = self.get_matches(comp_id, season_id)
        logger.info(f"Loading {len(matches)} matches...")
        
        all_player_stats = []
        
        for idx, match_row in matches.iterrows():
            match_id = match_row['match_id']
            logger.info(f"Processing match {idx+1}/{len(matches)}: {match_row['home_team']} vs {match_row['away_team']}")
            
            try:
                events = self.load_match_events(match_id)
                player_ids = events['player_id'].dropna().unique()
                
                for player_id in player_ids:
                    stats = self.extract_player_match_stats(events, player_id)
                    
                    if stats:
                        stats['match_id'] = int(match_id)
                        stats['match_date'] = str(match_row['match_date'])
                        stats['home_team'] = str(match_row['home_team'])
                        stats['away_team'] = str(match_row['away_team'])
                        stats['was_home'] = stats['team_name'] == match_row['home_team']
                        all_player_stats.append(stats)
            
            except Exception as e:
                logger.error(f"Error processing match {match_id}: {e}")
                continue
        
        player_stats_df = pd.DataFrame(all_player_stats)
        logger.info(f"Loaded {len(player_stats_df)} player-match records from {len(matches)} matches")
        
        return {
            'matches': matches,
            'player_stats': player_stats_df,
        }
