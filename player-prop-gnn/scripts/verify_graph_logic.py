"""
Forensic Verification of Graph Builder.
Compares Raw Database Rows vs. PyTorch Graph Objects.
"""
import sys
import os
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.graph_builder import MatchGraphBuilder

def get_db_config():
    # Load your specific .env file
    env_path = Path('.env')
    load_dotenv(env_path)
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'football_props',
        'user': 'postgres',
        'password': os.getenv('DATABASE_PASSWORD')
    }

def audit_match(match_id):
    print(f"\n🔍 AUDITING MATCH: {match_id}")
    print("=" * 50)
    
    conn = psycopg2.connect(**get_db_config())
    
    # 1. RAW DATABASE AUDIT
    print("\n[DATABASE TRUTH]")
    
    # Count Players (Nodes)
    players = pd.read_sql(
        "SELECT player_id, name, position, was_home FROM player_features "
        "JOIN players USING(player_id) WHERE match_id = %s", 
        conn, params=(match_id,)
    )
    print(f"  - Database Rows (Players): {len(players)}")
    print(f"  - Sample Player: {players.iloc[0]['name']} ({players.iloc[0]['position']})")
    
    # Count Interactions (Edges)
    edges = pd.read_sql(
        "SELECT count(*) FROM match_interactions WHERE match_id = %s",
        conn, params=(match_id,)
    )
    raw_interactions = edges.iloc[0,0]
    print(f"  - Database Rows (Passes):  {raw_interactions}")
    
    conn.close()

    # 2. GRAPH BUILDER OUTPUT
    print("\n[GRAPH BUILDER OUTPUT]")
    builder = MatchGraphBuilder(get_db_config())
    graph = builder.build_graph(match_id)
    
    print(f"  - Graph Nodes (x):         {graph.num_nodes}")
    print(f"  - Graph Edges (edge_index):{graph.edge_index.shape[1]}")
    
    # 3. THE VERIFICATION (The "Lie Detector")
    print("\n[VERIFICATION]")
    
    # Check Nodes
    if len(players) == graph.num_nodes:
        print("  ✅ NODES MATCH: Database rows equals Graph Nodes.")
    else:
        print(f"  ❌ NODES FAIL: DB={len(players)} vs Graph={graph.num_nodes}")
        
    # Check Edges (Interactions + Self Loops)
    # The builder adds 1 self-loop per player. 
    # So Graph Edges should = DB Interactions + DB Players
    expected_edges = raw_interactions + len(players)
    
    if graph.edge_index.shape[1] == expected_edges:
        print(f"  ✅ EDGES MATCH: {raw_interactions} (DB Passes) + {len(players)} (Self Loops) = {graph.edge_index.shape[1]}")
    else:
        print(f"  ❌ EDGES FAIL: Expected {expected_edges}, got {graph.edge_index.shape[1]}")

    # Check Features
    print("\n[DATA INTEGRITY]")
    # Check if features are not all zeros (which would imply fake data)
    if graph.x.sum() > 0:
        print("  ✅ Feature Matrix is populated (Non-zero).")
    else:
        print("  ❌ Feature Matrix is empty (All zeros).")

if __name__ == "__main__":
    # Test on the same match ID from your logs
    audit_match(3754058)