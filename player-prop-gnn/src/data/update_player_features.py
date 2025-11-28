"""
Update player_features table with newly loaded competition data.
"""
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_player_features(db_config):
    """
    Populate/update player_features table with all matches.
    Uses same SQL logic as before but for ALL competitions.
    """
    conn = psycopg2.connect(**db_config)
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
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'football_props',
        'user': 'postgres',
        'password': 'rsTKEn2JcfsJujMghw589SMCkvT/3lT1cqF1xf3Y6Y8='
    }
    
    update_player_features(db_config)
    print("\n✅ Done! Ready for GNN training.")