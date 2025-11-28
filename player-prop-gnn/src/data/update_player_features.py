"""
Update player_features table with newly loaded competition data.
SAFE VERSION: Loads credentials from environment.
"""
import psycopg2
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_config():
    """Load DB config securely from .env file."""
    # Resolve paths to find .env (Project Root or Docker dir)
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent.parent
    
    env_paths = [
        project_root / '.env',
        project_root / 'deployment' / 'docker' / '.env'
    ]
    
    env_loaded = False
    for path in env_paths:
        if path.exists():
            load_dotenv(path)
            env_loaded = True
            break
    
    if not env_loaded:
        logger.warning("⚠️ .env file not found. Relying on system environment variables.")

    # Get credentials from environment
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'football_props'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD') # No default here for security
    }

def update_player_features(db_config):
    """
    Populate/update player_features table with all matches.
    """
    # Fail fast if password is missing
    if not db_config.get('password'):
        logger.error("❌ CRITICAL: No database password found in environment variables.")
        logger.error("   Please ensure DATABASE_PASSWORD is set in your .env file.")
        return

    try:
        conn = psycopg2.connect(**db_config)
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return

    cursor = conn.cursor()
    
    try:
        logger.info("Truncating player_features table...")
        cursor.execute("TRUNCATE TABLE player_features CASCADE")
        
        logger.info("Populating player_features with all competition data...")
        cursor.execute("""
            INSERT INTO player_features (
                player_id, match_id, was_home, goals, assists, shots_on_target,
                yellow_cards, red_cards, minutes_played, match_date, team_id,
                home_team_id, away_team_id, home_score, away_score,
                days_since_last_match, opponent_id, opponent_strength,
                goals_rolling_5, assists_rolling_5, shots_on_target_rolling_5,
                yellow_cards_rolling_5, red_cards_rolling_5
            )
            SELECT 
                pms.player_id, pms.match_id, pms.was_home, pms.goals, pms.assists,
                pms.shots_on_target, pms.yellow_cards, pms.red_cards, pms.minutes_played,
                m.match_date, p.team_id, m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                
                COALESCE((m.match_date - LAG(m.match_date) OVER (PARTITION BY pms.player_id ORDER BY m.match_date))::numeric, 7.0),
                CASE WHEN pms.was_home THEN m.away_team_id ELSE m.home_team_id END,
                0.5,
                
                COALESCE(AVG(pms.goals) OVER (PARTITION BY pms.player_id ORDER BY m.match_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0.0),
                COALESCE(AVG(pms.assists) OVER (PARTITION BY pms.player_id ORDER BY m.match_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0.0),
                COALESCE(AVG(pms.shots_on_target) OVER (PARTITION BY pms.player_id ORDER BY m.match_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0.0),
                COALESCE(AVG(pms.yellow_cards) OVER (PARTITION BY pms.player_id ORDER BY m.match_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0.0),
                COALESCE(AVG(pms.red_cards) OVER (PARTITION BY pms.player_id ORDER BY m.match_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0.0)
            FROM player_match_stats pms
            JOIN matches m ON pms.match_id = m.match_id
            JOIN players p ON pms.player_id = p.player_id
        """)
        
        cursor.execute("SELECT COUNT(*) FROM player_features")
        count = cursor.fetchone()[0]
        
        conn.commit()
        logger.info(f"✓ player_features updated: {count} records")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    # Load config securely
    db_config = get_db_config()
    update_player_features(db_config)
    print("\n✅ Done! Ready for GNN training.")