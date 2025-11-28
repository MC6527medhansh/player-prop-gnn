"""
Phase 6.2: Graph Dataset Wrapper
Handles lazy loading, caching, and batching of match graphs.

Features:
- Caches processed graphs to disk (data/processed/gnn_v1/)
- Filters out invalid/empty matches automatically
- Compatible with PyTorch Geometric DataLoader
"""
import sys
import os
from pathlib import Path

# --- FIX IMPORT PATHS ---
# Allow importing from 'src' regardless of where script is run
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import shutil
from typing import List, Dict, Optional
from tqdm import tqdm
from torch_geometric.data import Dataset
import logging

# Now this import will work
from src.models.graph_builder import MatchGraphBuilder

logger = logging.getLogger(__name__)

class MatchGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Match Graphs.
    """
    def __init__(
        self, 
        root: str, 
        match_ids: List[int], 
        db_config: Dict, 
        force_reprocess: bool = False
    ):
        """
        Args:
            root: Root directory for dataset storage
            match_ids: List of match IDs to include
            db_config: Database credentials
            force_reprocess: If True, delete cache and rebuild
        """
        self.match_ids = match_ids
        self.db_config = db_config
        self.builder = MatchGraphBuilder(db_config)
        
        # Clean specific cache if forced
        if force_reprocess and os.path.exists(os.path.join(root, 'processed')):
            shutil.rmtree(os.path.join(root, 'processed'))
            
        super().__init__(root)
        
        # Filter out IDs that failed processing (based on file existence)
        # We check the processed path for each match ID
        self.valid_match_ids = []
        for mid in self.match_ids:
            # PyG creates processed file names based on the list we return in processed_file_names
            # But since we are filtering, we need to check existence manually or handle indices carefully
            # Simpler approach: Check if file exists in processed_dir
            if os.path.exists(os.path.join(self.processed_dir, f'match_{mid}.pt')):
                self.valid_match_ids.append(mid)
        
        if len(self.valid_match_ids) < len(self.match_ids):
            logger.warning(f"Filtered {len(self.match_ids) - len(self.valid_match_ids)} failed matches.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # We expect a file for every match ID in the input list
        return [f'match_{mid}.pt' for mid in self.match_ids]

    def download(self):
        # No downloading needed, data is in DB
        pass

    def process(self):
        """
        Builds graphs for all matches and saves them to disk.
        This runs ONLY if files are missing.
        """
        logger.info(f"Processing {len(self.match_ids)} matches...")
        
        processed_count = 0
        failed_count = 0
        
        # Ensure directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        for match_id in tqdm(self.match_ids, desc="Building Graphs"):
            out_path = os.path.join(self.processed_dir, f'match_{match_id}.pt')
            
            # Skip if already exists
            if os.path.exists(out_path):
                continue
                
            # Build Graph using the verified Builder
            graph = self.builder.build_graph(match_id)
            
            if graph is not None:
                torch.save(graph, out_path)
                processed_count += 1
            else:
                failed_count += 1
                
        logger.info(f"Processing complete. Success: {processed_count}, Failed: {failed_count}")

    def len(self):
        return len(self.valid_match_ids)

    def get(self, idx):
        """Load a single graph from disk."""
        match_id = self.valid_match_ids[idx]
        return torch.load(os.path.join(self.processed_dir, f'match_{match_id}.pt'))

# --- VERIFICATION SCRIPT ---
if __name__ == "__main__":
    import psycopg2
    from dotenv import load_dotenv
    from torch_geometric.loader import DataLoader
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Config (Securely)
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent.parent
    
    # Try finding .env
    env_path = project_root / '.env'
    if not env_path.exists():
        env_path = project_root / 'deployment' / 'docker' / '.env'
        
    load_dotenv(env_path)
    
    db_config = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'football_props'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD')
    }
    
    # 2. Fetch all match IDs
    print("Fetching match IDs from DB...")
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT match_id FROM matches ORDER BY match_date DESC")
        all_match_ids = [row[0] for row in cur.fetchall()]
        conn.close()
        
        print(f"Found {len(all_match_ids)} matches.")
        
        # 3. Create Dataset (This triggers processing loop)
        print("Initializing Dataset (this may take 2-3 mins on first run)...")
        dataset = MatchGraphDataset(
            root='data/processed/gnn_v1', 
            match_ids=all_match_ids, 
            db_config=db_config
        )
        
        print(f"\n✅ Dataset Ready: {len(dataset)} valid graphs")
        
        if len(dataset) > 0:
            # 4. Test Batching (Simulate Training Loop)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            batch = next(iter(loader))
            
            print("\n--- BATCH VERIFICATION ---")
            print(f"Batch Size: {batch.num_graphs}")
            print(f"Total Nodes: {batch.num_nodes}")
            print(f"Total Edges: {batch.edge_index.shape[1]}")
            print(f"Batch Object: {batch}")
        else:
            print("⚠️ No valid graphs created. Check database data.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")