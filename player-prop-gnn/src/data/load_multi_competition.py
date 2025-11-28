"""
Multi-competition loader for GNN training.
Loads recent data from multiple leagues (2018-2024).
"""
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import logging
from src.data.statsbomb_loader import StatsBombLoader
from src.data.validation import validate_player_data, validate_match_data, validate_player_match_stats, ValidationError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCompetitionLoader:
    """
    Load data from multiple StatsBomb competitions.
    
    Competitions loaded (2018-2024):
    - Euro 2024
    - Copa America 2024  
    - J1 League 2024
    - Champions League 2018/2019
    - World Cup 2018 (if not already loaded)
    
    Total: ~600-800 matches
    """
    
    COMPETITIONS = [
        # === NEW: Add these ===
        {'comp_id': 2, 'season_id': 27, 'name': 'Premier League 2015/2016', 'year': 2016},
        {'comp_id': 11, 'season_id': 27, 'name': 'La Liga 2015/2016', 'year': 2016},
        {'comp_id': 37, 'season_id': 4, 'name': 'FA WSL 2018/2019', 'year': 2019},
        {'comp_id': 37, 'season_id': 42, 'name': 'FA WSL 2019/2020', 'year': 2020},
        {'comp_id': 37, 'season_id': 90, 'name': 'FA WSL 2020/2021', 'year': 2021},
        
        # === Existing (will skip if already loaded) ===
        {'comp_id': 43, 'season_id': 106, 'name': 'FIFA World Cup 2022', 'year': 2022},
        {'comp_id': 55, 'season_id': 282, 'name': 'Euro 2024', 'year': 2024},
        {'comp_id': 223, 'season_id': 282, 'name': 'Copa America 2024', 'year': 2024},
    ]
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.loader = StatsBombLoader()
        
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
    
    def load_all_competitions(self, skip_existing=True):
        """
        Load all competitions into database.
        
        Args:
            skip_existing: If True, skip competitions that already have matches loaded
        """
        try:
            self.connect()
            
            total_stats = {
                'teams': 0,
                'players': 0,
                'matches': 0,
                'player_stats': 0
            }
            
            for comp_config in self.COMPETITIONS:
                logger.info(f"\n{'='*60}")
                logger.info(f"Loading: {comp_config['name']}")
                logger.info(f"{'='*60}")
                
                # Check if already loaded
                if skip_existing and self._competition_exists(comp_config):
                    logger.info(f"✓ {comp_config['name']} already loaded, skipping...")
                    continue
                
                try:
                    # Load competition data
                    data = self._load_competition(comp_config)
                    
                    if data is None:
                        logger.warning(f"No data available for {comp_config['name']}")
                        continue
                    
                    matches_df = data['matches']
                    player_stats_df = data['player_stats']
                    
                    # Load into database
                    teams = self._load_teams(matches_df)
                    players = self._load_players(player_stats_df)
                    matches = self._load_matches(matches_df, comp_config['name'])
                    stats = self._load_player_stats(player_stats_df)
                    
                    total_stats['teams'] += teams
                    total_stats['players'] += players
                    total_stats['matches'] += matches
                    total_stats['player_stats'] += stats
                    
                    logger.info(f"✓ {comp_config['name']}: {matches} matches, {stats} player-match records")
                    
                    self.conn.commit()
                    
                except Exception as e:
                    logger.error(f"Error loading {comp_config['name']}: {e}")
                    self.conn.rollback()
                    continue
            
            logger.info(f"\n{'='*60}")
            logger.info("FINAL SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total Teams: {total_stats['teams']}")
            logger.info(f"Total Players: {total_stats['players']}")
            logger.info(f"Total Matches: {total_stats['matches']}")
            logger.info(f"Total Player-Match Records: {total_stats['player_stats']}")
            logger.info(f"{'='*60}\n")
            
            self._print_database_summary()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self.close()
    
    def _competition_exists(self, comp_config):
        """Check if competition already has matches in database."""
        self.cursor.execute(
            "SELECT COUNT(*) FROM matches WHERE league = %s",
            (comp_config['name'],)
        )
        count = self.cursor.fetchone()[0]
        return count > 0
    
    def _load_competition(self, comp_config):
        """Load competition data from StatsBomb."""
        try:
            matches = self.loader.get_matches(
                comp_config['comp_id'],
                comp_config['season_id']
            )
            
            if matches.empty:
                return None
            
            logger.info(f"Found {len(matches)} matches")
            
            all_player_stats = []
            
            for idx, match_row in matches.iterrows():
                match_id = match_row['match_id']
                
                if (idx + 1) % 20 == 0:
                    logger.info(f"  Progress: {idx+1}/{len(matches)} matches")
                
                try:
                    events = self.loader.load_match_events(match_id)
                    player_ids = events['player_id'].dropna().unique()
                    
                    for player_id in player_ids:
                        stats = self.loader.extract_player_match_stats(events, player_id)
                        
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
            
            return {
                'matches': matches,
                'player_stats': player_stats_df,
            }
            
        except Exception as e:
            logger.error(f"Error loading competition data: {e}")
            return None
    
    def _load_teams(self, matches_df):
        """Load teams from matches."""
        teams = set()
        for _, match in matches_df.iterrows():
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        teams_data = [(team, 'Multi-Competition') for team in sorted(teams)]
        query = "INSERT INTO teams (name, league) VALUES %s ON CONFLICT (name) DO NOTHING"
        execute_values(self.cursor, query, teams_data, template="(%s, %s)")
        return len(teams)
    
    def _load_players(self, player_stats_df):
        """Load players from player stats."""
        if player_stats_df.empty:
            return 0
            
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
    
    def _load_matches(self, matches_df, league_name):
        """Load matches."""
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
                    """INSERT INTO matches (home_team_id, away_team_id, match_date, home_score, away_score, league) 
                       VALUES (%s, %s, %s, %s, %s, %s) 
                       ON CONFLICT DO NOTHING""",
                    (home_result[0], away_result[0], match['match_date'], 
                     match.get('home_score'), match.get('away_score'), league_name)
                )
                loaded_count += 1
            except ValidationError:
                continue
        
        return loaded_count
    
    def _load_player_stats(self, player_stats_df):
        """Load player match stats."""
        if player_stats_df.empty:
            return 0
            
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
                    INSERT INTO player_match_stats 
                    (player_id, match_id, goals, assists, shots_on_target, total_shots, 
                     yellow_cards, red_cards, minutes_played, was_home)
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
        """Map StatsBomb position to our schema."""
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
    
    def _print_database_summary(self):
        """Print database summary."""
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        print(f"Teams: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM players")
        print(f"Players: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM matches")
        print(f"Matches: {self.cursor.fetchone()[0]}")
        
        self.cursor.execute("SELECT COUNT(*) FROM player_match_stats")
        print(f"Player-Match Records: {self.cursor.fetchone()[0]}")
        
        print("\nBreakdown by League:")
        self.cursor.execute("""
            SELECT league, COUNT(*) 
            FROM matches 
            GROUP BY league 
            ORDER BY COUNT(*) DESC
        """)
        for league, count in self.cursor.fetchall():
            print(f"  {league}: {count} matches")
        
        print("="*60 + "\n")


def main():
    """Main execution."""
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'football_props',
        'user': 'postgres',
        'password': 'rsTKEn2JcfsJujMghw589SMCkvT/3lT1cqF1xf3Y6Y8='
    }
    
    print("\n" + "="*60)
    print("MULTI-COMPETITION DATA LOADER FOR GNN")
    print("="*60)
    print("\nThis will load:")
    for comp in MultiCompetitionLoader.COMPETITIONS:
        print(f"  - {comp['name']} ({comp['year']})")
    print(f"\nEstimated: 600-800 matches, ~15,000-20,000 player-match records")
    print(f"Time: ~2-3 hours (first run with API calls)")
    print("="*60 + "\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    loader = MultiCompetitionLoader(db_config)
    loader.load_all_competitions(skip_existing=True)
    
    print("\n✅ Data loading complete!")
    print("Next step: Update player_features table with new data")
    print("Run: python -m src.data.update_player_features")


if __name__ == '__main__':
    main()