"""
Phase 6.1: Match Graph Builder (Database Driven)
Converts SQL match data into PyTorch Geometric graphs.

Key Features:
- Fetches nodes from `player_features` (pre-computed rolling stats)
- Fetches edges from `match_interactions` (pass networks)
- Zero API calls (fast graph construction)
"""
import torch
import pandas as pd
import numpy as np
import psycopg2
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MatchGraphBuilder:
    """
    Constructs PyG Data objects from the database.
    
    Structure:
    - Nodes: Players in a specific match
    - Edges: Passes between players
    - Node Features: 
        - Position (One-hot, 4 dims)
        - Context (Home/Away, Opponent Strength, Rest Days)
        - Form (Rolling Goals/Shots/Assists/Cards - 5 game window)
    - Targets (y): Actual match outcomes (Goals, Assists, Shots, Cards)
    """

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        
        # Position mapping for one-hot encoding
        self.pos_map = {
            'Goalkeeper': 0,
            'Defender': 1,
            'Midfielder': 2,
            'Forward': 3
        }

    def build_graph(self, match_id: int) -> Optional[Data]:
        """
        Build a single graph for a given match_id.
        Returns None if match has insufficient data (e.g., < 22 players or 0 edges).
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # 1. Fetch Nodes (Players & Features)
            # We join with 'players' table to get the static position
            nodes_df = pd.read_sql("""
                SELECT 
                    pf.player_id,
                    p.position,
                    pf.was_home,
                    pf.opponent_strength,
                    pf.days_since_last_match,
                    pf.goals_rolling_5,
                    pf.assists_rolling_5,
                    pf.shots_on_target_rolling_5,
                    pf.yellow_cards_rolling_5,
                    pf.red_cards_rolling_5,
                    -- TARGETS (Ground Truth)
                    pf.goals,
                    pf.assists,
                    pf.shots_on_target,
                    pf.yellow_cards,
                    pf.red_cards
                FROM player_features pf
                JOIN players p ON pf.player_id = p.player_id
                WHERE pf.match_id = %s
                ORDER BY pf.was_home DESC, p.position  -- Deterministic order
            """, conn, params=(match_id,))

            if len(nodes_df) < 11:  # Need at least one full team
                logger.warning(f"Match {match_id}: Insufficient players ({len(nodes_df)}). Skipping.")
                return None

            # 2. Fetch Edges (Interactions)
            edges_df = pd.read_sql("""
                SELECT sender_id, receiver_id, success, timestamp_second
                FROM match_interactions
                WHERE match_id = %s
            """, conn, params=(match_id,))

            # 3. Construct Graph Components
            x = self._build_node_features(nodes_df)
            edge_index, edge_attr = self._build_edges(nodes_df, edges_df)
            y = self._build_targets(nodes_df)

            if edge_index.shape[1] == 0:
                logger.warning(f"Match {match_id}: No interaction edges found. Skipping.")
                return None

            # 4. Create PyG Object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                match_id=match_id,
                num_nodes=len(nodes_df)
            )
            
            return data

        except Exception as e:
            logger.error(f"Failed to build graph for {match_id}: {e}")
            return None
        finally:
            if conn: conn.close()

    def _build_node_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Construct feature matrix [num_nodes, num_features].
        Current Dim: 4 (Pos) + 3 (Context) + 5 (Form) = 12 Features
        """
        features = []
        
        for _, row in df.iterrows():
            # 1. Position One-Hot (4 dims)
            pos_vec = [0.0] * 4
            idx = self.pos_map.get(row['position'], 2) # Default Midfielder
            pos_vec[idx] = 1.0
            
            # 2. Context (3 dims)
            context = [
                1.0 if row['was_home'] else 0.0,
                float(row['opponent_strength'] or 0.5),
                float(min(row['days_since_last_match'] or 7, 30) / 30.0) # Normalized
            ]
            
            # 3. Form (5 dims) - Rolling averages
            form = [
                float(row['goals_rolling_5'] or 0),
                float(row['assists_rolling_5'] or 0),
                float(row['shots_on_target_rolling_5'] or 0),
                float(row['yellow_cards_rolling_5'] or 0),
                float(row['red_cards_rolling_5'] or 0)
            ]
            
            features.append(pos_vec + context + form)
            
        return torch.tensor(features, dtype=torch.float)

    def _build_edges(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Construct edge indices and attributes.
        Maps database player_ids to graph node indices (0..N).
        """
        # Create map: player_id -> node_index
        pid_to_idx = {pid: i for i, pid in enumerate(nodes_df['player_id'])}
        
        src_nodes = []
        dst_nodes = []
        attrs = []
        
        # 1. Pass Network Edges
        for _, row in edges_df.iterrows():
            sender = row['sender_id']
            receiver = row['receiver_id']
            
            # Skip if either player is missing from node list (e.g., subbed out early/late or data glitch)
            if sender not in pid_to_idx or receiver not in pid_to_idx:
                continue
                
            src_nodes.append(pid_to_idx[sender])
            dst_nodes.append(pid_to_idx[receiver])
            
            # Edge Features: [1=Pass, Success(0/1), Normalized Time]
            time_norm = row['timestamp_second'] / (90 * 60) if row['timestamp_second'] else 0.5
            attrs.append([1.0, 1.0 if row['success'] else 0.0, time_norm])

        # 2. Self Loops (Important for GNN message passing)
        for i in range(len(nodes_df)):
            src_nodes.append(i)
            dst_nodes.append(i)
            attrs.append([0.0, 1.0, 0.0]) # [0=Self, Success, Time]

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
        
        return edge_index, edge_attr

    def _build_targets(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Construct target matrix [num_nodes, 4].
        Columns: Goals, Assists, Shots, Cards (Binary)
        """
        targets = []
        for _, row in df.iterrows():
            targets.append([
                float(row['goals']),
                float(row['assists']),
                float(row['shots_on_target']),
                1.0 if (row['yellow_cards'] > 0 or row['red_cards'] > 0) else 0.0
            ])
            
        return torch.tensor(targets, dtype=torch.float)

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load env
    load_dotenv()
    db_config = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'football_props'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD')
    }
    
    builder = MatchGraphBuilder(db_config)
    
    # Test on a known match ID from your data
    # (e.g., from the Premier League batch we just loaded)
    test_match_id = 3754058 # From your logs
    
    print(f"Building graph for match {test_match_id}...")
    graph = builder.build_graph(test_match_id)
    
    if graph:
        print("\n✅ GRAPH BUILT SUCCESSFULLY:")
        print(graph)
        print(f"Nodes: {graph.num_nodes}")
        print(f"Features (x): {graph.x.shape}")
        print(f"Edges: {graph.edge_index.shape}")
        print(f"Targets (y): {graph.y.shape}")
    else:
        print("\n❌ Graph build failed (check match_id or data).")