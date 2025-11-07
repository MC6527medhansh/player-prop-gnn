"""
Database loader for World Cup 2018 data.
"""
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import logging
from statsbomb_loader import StatsBombLoader
from validation import validate_player_data, validate_match_data, validate_player_match_stats, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseLoader:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect(self):
        logger.info(f"Connecting to database: {self.db_config['database']}")
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        logger.info("Database connection established")
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def load_world_cup_2018(self):
        try:
            self.connect()
            loader = StatsBombLoader()
            data = loader.load_world_cup_2018()
            
            matches_df = data['matches']
            player_stats_df = data['player_stats']
            
            logger.info("Loading teams...")
            teams_loaded = self._load_teams(matches_df)
            logger.info(f"Loaded {teams_loaded} teams")
            
            logger.info("Loading players...")
            players_loaded = self._load_players(player_stats_df)
            logger.info(f"Loaded {players_loaded} players")
            
            logger.info("Loading matches...")
            matches_loaded = self._load_matches(matches_df)
            logger.info(f"Loaded {matches_loaded} matches")
            
            logger.info("Loading player match stats...")
            stats_loaded = self._load_player_stats(player_stats_df)
            logger.info(f"Loaded {stats_loaded} player-match records")
            
            self.conn.commit()
            logger.info("All data committed!")
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self.close()
    
    def _load_teams(self, matches_df):
        teams = set()
        for _, match in matches_df.iterrows():
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        teams_data = [(team, 'FIFA World Cup') for team in sorted(teams)]
        query = "INSERT INTO teams (name, league) VALUES %s ON CONFLICT (name) DO NOTHING"
        execute_values(self.cursor, query, teams_data, template="(%s, %s)")
        return len(teams)
    
    def _load_players(self, player_stats_df):
        players = player_stats_df[['player_id', 'player_name', 'team_name', 'position']].drop_duplicates('player_id')
        loaded_count = 0
        
        for _, player in players.iterrows():
            self.cursor.execute("SELECT team_id FROM teams WHERE name = %s", (player['team_name'],))
            result = self.cursor.fetchone()
            if not result:
                continue
            
            team_id = result[0]
            position = self._map_position(player.get('position', 'Midfielder'))
            
            try:
                validate_player_data({'name': player['player_name'], 'position': position, 'team_id': team_id})
                self.cursor.execute(
                    "INSERT INTO players (name, position, team_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (player['player_name'], position, team_id)
                )
                loaded_count += 1
            except ValidationError:
                continue
        
        return loaded_count
    
    def _load_matches(self, matches_df):
        loaded_count = 0
        for _, match in matches_df.iterrows():
            self.cursor.execute("SELECT team_id FROM teams WHERE name = %s", (match['home_team'],))
            home_result = self.cursor.fetchone()
            self.cursor.execute("SELECT team_id FROM teams WHERE name = %s", (match['away_team'],))
            away_result = self.cursor.fetchone()
            
            if not home_result or not away_result:
                continue
            
            try:
                validate_match_data({
                    'home_team_id': home_result[0],
                    'away_team_id': away_result[0],
                    'match_date': str(match['match_date'])
                })
                self.cursor.execute(
                    "INSERT INTO matches (home_team_id, away_team_id, match_date, home_score, away_score, league) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (home_result[0], away_result[0], match['match_date'], match.get('home_score'), match.get('away_score'), 'FIFA World Cup')
                )
                loaded_count += 1
            except ValidationError:
                continue
        
        return loaded_count
    
    def _load_player_stats(self, player_stats_df):
        loaded_count = 0
        for _, stats in player_stats_df.iterrows():
            try:
                self.cursor.execute("SELECT player_id FROM players WHERE name = %s", (stats['player_name'],))
                player_result = self.cursor.fetchone()
                if not player_result:
                    continue
                
                self.cursor.execute("""
                    SELECT m.match_id FROM matches m
                    JOIN teams ht ON m.home_team_id = ht.team_id
                    JOIN teams at ON m.away_team_id = at.team_id
                    WHERE ht.name = %s AND at.name = %s AND m.match_date = %s
                """, (stats['home_team'], stats['away_team'], stats['match_date']))
                
                match_result = self.cursor.fetchone()
                if not match_result:
                    continue
                
                stats_data = {
                    'player_id': player_result[0],
                    'match_id': match_result[0],
                    'goals': int(stats.get('goals', 0)),
                    'assists': int(stats.get('assists', 0)),
                    'shots_on_target': int(stats.get('shots_on_target', 0)),
                    'total_shots': int(stats.get('total_shots', 0)),
                    'yellow_cards': int(stats.get('yellow_cards', 0)),
                    'red_cards': int(stats.get('red_cards', 0)),
                    'minutes_played': int(stats.get('minutes_played', 0))
                }
                
                validate_player_match_stats(stats_data)
                
                self.cursor.execute("""
                    INSERT INTO player_match_stats (player_id, match_id, goals, assists, shots_on_target, total_shots, yellow_cards, red_cards, minutes_played, was_home)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, match_id) DO UPDATE SET
                        goals = EXCLUDED.goals,
                        assists = EXCLUDED.assists,
                        shots_on_target = EXCLUDED.shots_on_target,
                        total_shots = EXCLUDED.total_shots,
                        yellow_cards = EXCLUDED.yellow_cards,
                        red_cards = EXCLUDED.red_cards,
                        minutes_played = EXCLUDED.minutes_played,
                        was_home = EXCLUDED.was_home
                """, (
                    stats_data['player_id'], stats_data['match_id'], stats_data['goals'],
                    stats_data['assists'], stats_data['shots_on_target'], stats_data['total_shots'],
                    stats_data['yellow_cards'], stats_data['red_cards'], stats_data['minutes_played'],
                    bool(stats.get('was_home', False))
                ))
                loaded_count += 1
            except:
                continue
        
        return loaded_count
    
    def _map_position(self, statsbomb_position):
        position = str(statsbomb_position).lower()
        if any(x in position for x in ['forward', 'striker', 'wing', 'attacking']):
            return 'Forward'
        elif any(x in position for x in ['midfield', 'central']):
            return 'Midfielder'
        elif any(x in position for x in ['back', 'defender', 'defense']):
            return 'Defender'
        elif 'goalkeeper' in position or 'keeper' in position:
            return 'Goalkeeper'
        else:
            return 'Midfielder'
    
    def _print_summary(self):
        print("\n" + "="*60)
        print("DATABASE LOAD SUMMARY")
        print("="*60)
        
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        print(f"Teams: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM players")
        print(f"Players: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM matches")
        print(f"Matches: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM player_match_stats")
        print(f"Player-Match Records: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT SUM(goals), SUM(assists), SUM(total_shots) FROM player_match_stats")
        stats = self.cursor.fetchone()
        print(f"\nTotal Goals: {stats[0]}")
        print(f"Total Assists: {stats[1]}")
        print(f"Total Shots: {stats[2]}")
        
        print("="*60 + "\n")

def main():
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'football_props',
        'user': 'medhanshchoubey',
        'password': ''
    }
    
    print("Loading World Cup 2018 data...")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    loader = DatabaseLoader(db_config)
    loader.load_world_cup_2018()
    print("\n✅ Done!")

if __name__ == '__main__':
    main()
