"""
Phase 6.0: Bulk Data Loader (Final Robust Version)
Loads matches AND pass interaction networks for GNN training.

Fixes:
- Aligns INSERT statement with DB Schema
- Handles StatsBomb API warnings
- Auto-creates Teams to avoid Foreign Key errors
- CORRECTLY assigns team_id to players (Fixes FK violation)
"""
import sys
import os
import logging
from pathlib import Path
from tqdm import tqdm
import psycopg2
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.statsbomb_loader import StatsBombLoader

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bulk_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
def load_db_config():
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent.parent
    
    env_paths = [project_root / '.env', project_root / 'deployment' / 'docker' / '.env']
    for path in env_paths:
        if path.exists():
            load_dotenv(path)
            break

    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'football_props'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD', '')
    }

DB_CONFIG = load_db_config()

TARGET_COMPETITIONS = [
    (2, 27, "Premier League 2015/2016"),
    (11, 27, "La Liga 2015/2016"),
    (37, 4, "FA WSL 2018/2019"),
    (37, 42, "FA WSL 2019/2020"),
    (37, 90, "FA WSL 2020/2021"),
]

def ensure_schema(conn):
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS match_interactions (
                interaction_id SERIAL PRIMARY KEY,
                match_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                receiver_id INTEGER NOT NULL,
                interaction_type VARCHAR(20) NOT NULL DEFAULT 'pass',
                success BOOLEAN DEFAULT TRUE,
                timestamp_second INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_match_inter FOREIGN KEY (match_id) REFERENCES matches(match_id)
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_match ON match_interactions(match_id);
        """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()

def get_or_create_team(cur, team_name, league_name):
    """Get team_id by name, creating if missing."""
    if not team_name: return None

    # Try find
    cur.execute("SELECT team_id FROM teams WHERE name = %s", (team_name,))
    res = cur.fetchone()
    if res: return res[0]

    # Create
    cur.execute("""
        INSERT INTO teams (name, league) VALUES (%s, %s) 
        ON CONFLICT (name) DO UPDATE SET league = EXCLUDED.league
        RETURNING team_id
    """, (team_name, league_name))
    return cur.fetchone()[0]

def process_match(loader, conn, match_row, competition_name):
    match_id = int(match_row['match_id'])
    cur = conn.cursor()
    
    try:
        # 1. Check exists
        cur.execute("SELECT 1 FROM matches WHERE match_id = %s", (match_id,))
        if cur.fetchone(): return True 

        # 2. Fetch Data
        events = loader.load_match_events(match_id)
        
        # 3. Teams
        home_name = match_row.get('home_team')
        away_name = match_row.get('away_team')
        
        home_id = get_or_create_team(cur, home_name, competition_name)
        away_id = get_or_create_team(cur, away_name, competition_name)
        
        # 4. Insert Match
        cur.execute("""
            INSERT INTO matches (match_id, match_date, league, home_team_id, away_team_id, home_score, away_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (match_id, str(match_row['match_date']), competition_name, home_id, away_id, 
              int(match_row.get('home_score', 0)), int(match_row.get('away_score', 0))))

        # 5. Insert Players
        unique_pids = events['player_id'].dropna().unique()
        
        for pid in unique_pids:
            stats = loader.extract_player_match_stats(events, pid)
            if not stats: continue
            
            # DETERMINISTIC TEAM RESOLUTION (Fixes FK Violation)
            p_team_name = stats.get('team_name')
            p_team_id = home_id if p_team_name == home_name else (away_id if p_team_name == away_name else None)
            
            # Fallback: If player's team name doesn't match home/away (rare data error), create it
            if not p_team_id:
                p_team_id = get_or_create_team(cur, p_team_name, competition_name)

            # Insert Player
            cur.execute("""
                INSERT INTO players (player_id, name, position, team_id)
                VALUES (%s, %s, %s, %s) ON CONFLICT (player_id) DO NOTHING
            """, (int(pid), stats['player_name'], stats['position'], p_team_id))
            
            # Insert Stats
            cur.execute("""
                INSERT INTO player_match_stats 
                (player_id, match_id, goals, assists, total_shots, shots_on_target,
                 yellow_cards, red_cards, minutes_played, was_home)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (int(pid), match_id, stats['goals'], stats['assists'], 
                  stats['total_shots'], stats['shots_on_target'], 
                  stats['yellow_cards'], stats['red_cards'], 
                  stats['minutes_played'], (p_team_id == home_id)))

        # 6. Interactions
        edges = loader.extract_pass_network(events)
        if edges:
            args_str = ','.join(cur.mogrify("(%s, %s, %s, %s, %s)", (match_id, s, r, suc, ts)).decode('utf-8') for s, r, suc, ts in edges)
            cur.execute(f"INSERT INTO match_interactions (match_id, sender_id, receiver_id, success, timestamp_second) VALUES {args_str}")

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Match {match_id} failed: {e}")
        return False
    finally:
        cur.close()

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        ensure_schema(conn)
        
        loader = StatsBombLoader()
        total_new, total_err = 0, 0
        
        for comp_id, season_id, name in TARGET_COMPETITIONS:
            logger.info(f"Processing {name}...")
            try:
                matches = loader.get_matches(comp_id, season_id)
                pbar = tqdm(matches.iterrows(), total=len(matches), desc=name)
                for _, row in pbar:
                    if process_match(loader, conn, row, name):
                        total_new += 1
                    else:
                        total_err += 1
                    pbar.set_postfix(new=total_new, err=total_err)
            except Exception as e:
                logger.error(f"Competition failed: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if 'conn' in locals() and conn: conn.close()

if __name__ == "__main__":
    main()