"""
Verification Script for Phase 6.0 Data Load.
Checks counts, integrity, and graph readiness.
"""
import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load Environment
env_path = Path('.env')
load_dotenv(env_path)

DB_CONFIG = {
    'host': os.getenv('DATABASE_HOST', 'localhost'),
    'port': int(os.getenv('DATABASE_PORT', 5432)),
    'database': os.getenv('DATABASE_NAME', 'football_props'),
    'user': os.getenv('DATABASE_USER', 'postgres'),
    'password': os.getenv('DATABASE_PASSWORD')
}

def verify_counts(cur):
    print("\n--- 1. ROW COUNTS ---")
    tables = ['matches', 'players', 'teams', 'player_match_stats', 'match_interactions']
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"✅ {table.ljust(20)}: {count:,}")
        
        if table == 'match_interactions' and count == 0:
            print("   ❌ CRITICAL FAIL: No graph edges found!")

def verify_distributions(cur):
    print("\n--- 2. MATCH DISTRIBUTION ---")
    cur.execute("""
        SELECT league, COUNT(*) 
        FROM matches 
        GROUP BY league 
        ORDER BY COUNT(*) DESC
    """)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['League', 'Count'])
    print(df.to_string(index=False))

def verify_graph_integrity(cur):
    print("\n--- 3. GRAPH INTEGRITY (The GNN Requirement) ---")
    
    # Check avg edges per match
    cur.execute("""
        SELECT AVG(edge_count) 
        FROM (
            SELECT match_id, COUNT(*) as edge_count 
            FROM match_interactions 
            GROUP BY match_id
        ) sub
    """)
    avg_edges = cur.fetchone()[0]
    print(f"Average Interactions/Match: {float(avg_edges or 0):.1f} (Target: >50)")

    # Check for "Zombie Matches" (Matches with NO interactions)
    cur.execute("""
        SELECT COUNT(*) 
        FROM matches m 
        LEFT JOIN match_interactions mi ON m.match_id = mi.match_id 
        WHERE mi.interaction_id IS NULL
    """)
    zombies = cur.fetchone()[0]
    if zombies > 0:
        print(f"⚠️  WARNING: {zombies} matches have 0 interactions (Tier 1 ok, GNN will skip)")
    else:
        print("✅ All matches have interaction data.")

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print(f"🔌 Connected to {DB_CONFIG['database']}")
        
        verify_counts(cur)
        verify_distributions(cur)
        verify_graph_integrity(cur)
        
        cur.close()
        conn.close()
        print("\n✅ VERIFICATION COMPLETE")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    main()