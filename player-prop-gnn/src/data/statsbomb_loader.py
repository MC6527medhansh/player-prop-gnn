from statsbombpy import sb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatsBombLoader:
    
    @staticmethod
    def get_matches(competition_id, season_id):
        return sb.matches(competition_id=competition_id, season_id=season_id)
    
    @staticmethod
    def load_match_events(match_id):
        return sb.events(match_id=match_id)
    
    @staticmethod
    def _map_position(raw_pos: str) -> str:
        """
        Map detailed StatsBomb positions to the 4 DB-allowed categories.
        Logic aligns with Tier 1 training data.
        """
        pos = str(raw_pos).lower()
        
        # Priority mapping (Goalkeeper first)
        if 'goalkeeper' in pos or 'keeper' in pos:
            return 'Goalkeeper'
            
        # Defenders (Center Backs, Full Backs, Wing Backs)
        if any(x in pos for x in ['back', 'defender', 'defense']):
            return 'Defender'
            
        # Forwards (Strikers, Wings, Attacking Midfielders often act as FWDs)
        # Note: 'Right Wing' -> Forward, 'Attacking Midfield' -> Forward (aggressive mapping)
        if any(x in pos for x in ['forward', 'striker', 'wing', 'attacking']):
            return 'Forward'
            
        # Midfielders (Defensive Mids, Center Mids)
        if any(x in pos for x in ['midfield', 'central']):
            return 'Midfielder'
            
        # Default Fallback
        return 'Midfielder'

    @staticmethod
    def extract_player_match_stats(events: pd.DataFrame, player_id: int) -> Optional[Dict]:
        """
        Extract robust stats for a single player in a match.
        """
        player_events = events[events['player_id'] == player_id].copy()
        
        if player_events.empty:
            return None
        
        # Robust Name Extraction
        player_name = f"Player_{player_id}"
        if 'player' in player_events.columns:
            player_name = str(player_events['player'].iloc[0])
        elif 'player_name' in player_events.columns:
            player_name = str(player_events['player_name'].iloc[0])
            
        # Robust Team Name
        team_name = "Unknown"
        if 'team' in player_events.columns:
            team_name = str(player_events['team'].iloc[0])
        elif 'team_name' in player_events.columns:
            team_name = str(player_events['team_name'].iloc[0])
            
        stats = {
            'player_id': int(player_id),
            'player_name': player_name,
            'team_name': team_name,
        }
        
        # Position Mapping (THE FIX)
        raw_pos = 'Midfielder'
        if 'position' in player_events.columns:
            found_pos = player_events['position'].dropna()
            if not found_pos.empty:
                raw_pos = found_pos.iloc[0]
        
        stats['position'] = StatsBombLoader._map_position(raw_pos)
            
        # --- ROBUST METRICS ---
        
        # 1. Goals
        stats['goals'] = 0
        stats['total_shots'] = 0
        stats['shots_on_target'] = 0
        
        if 'shot_outcome' in player_events.columns:
            shots = player_events[player_events['type'] == 'Shot']
            stats['total_shots'] = len(shots)
            stats['goals'] = len(shots[shots['shot_outcome'] == 'Goal'])
            stats['shots_on_target'] = len(shots[shots['shot_outcome'].isin(['Goal', 'Saved', 'Saved to Post', 'Saved Off Target'])])
            
        # 2. Assists (Fixes deprecation warning too)
        stats['assists'] = 0
        if 'pass_goal_assist' in player_events.columns:
            # fillna(False) avoids the FutureWarning about downcasting
            stats['assists'] = int(player_events['pass_goal_assist'].fillna(False).astype(bool).sum())
        
        # 3. Cards
        yellow = 0
        red = 0
        if 'foul_committed_card' in player_events.columns:
            cards = player_events['foul_committed_card'].dropna()
            for card in cards:
                c = str(card).lower()
                if 'yellow' in c:
                    yellow += 1
                if 'red' in c:
                    red += 1
        stats['yellow_cards'] = yellow
        stats['red_cards'] = red
        
        # 4. Minutes
        stats['minutes_played'] = 0
        if 'minute' in player_events.columns:
            stats['minutes_played'] = int(player_events['minute'].max())
            
        return stats

    @staticmethod
    def extract_pass_network(events: pd.DataFrame) -> List[Tuple]:
        """
        Extract directed edges (Passes) for Graph Construction.
        Returns list of (sender_id, receiver_id, success_bool, timestamp)
        """
        if 'type' not in events.columns or 'player_id' not in events.columns:
            return []
            
        passes = events[events['type'] == 'Pass'].copy()
        
        recipient_col = 'pass_recipient_id' if 'pass_recipient_id' in passes.columns else None
        if recipient_col is None:
            return []

        outcome_col = 'pass_outcome' if 'pass_outcome' in passes.columns else None
        
        edges = []
        for _, row in passes.iterrows():
            sender = row['player_id']
            receiver = row[recipient_col]
            
            if pd.isna(receiver):
                continue
                
            success = True
            if outcome_col and pd.notna(row[outcome_col]):
                success = False
                
            timestamp = int(row.get('minute', 0) * 60 + row.get('second', 0))
            
            edges.append((int(sender), int(receiver), success, timestamp))
            
        return edges