"""
Forensic Audit of Correlation Logic.
Proves that predictions are dynamic and correlations are calculated, not hardcoded.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graph_dataset import MatchGraphDataset
from src.models.gnn_gat import PlayerPropGAT
from src.models.train_gnn import get_db_config

def audit_correlation_engine():
    print("\n🔍 AUDITING CORRELATION ENGINE")
    print("=" * 50)

    # 1. SETUP
    device = torch.device("cpu")
    db_config = get_db_config()
    
    # 2. LOAD REAL DATA (Not dummy)
    print("\n[1. DATA SOURCE]")
    import psycopg2
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SELECT match_id FROM matches ORDER BY match_date DESC LIMIT 50")
    recent_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    
    dataset = MatchGraphDataset('data/processed/gnn_v1', recent_ids, db_config)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print(f"  - Loaded Batch: {batch.num_graphs} matches, {batch.num_nodes} players")

    # 3. RUN MODEL INFERENCE
    print("\n[2. MODEL OUTPUT AUDIT]")
    model = PlayerPropGAT(in_channels=12, hidden_channels=64, heads=4)
    model.load_state_dict(torch.load("models/gnn_best.pt", map_location=device))
    model.eval()
    
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr)
    
    # Print sample predictions to prove variance
    print("  - Sampling 5 random players from batch:")
    print("    (If these numbers are identical, the model is fake/broken)")
    print("    Player | Goals | Shots | Cards")
    print("    -------|-------|-------|------")
    
    indices = np.random.choice(batch.num_nodes, 5, replace=False)
    for idx in indices:
        g = out['goals'][idx].item()
        s = out['shots'][idx].item()
        c = out['cards'][idx].item()
        print(f"    ID {idx:03d} | {g:.3f} | {s:.3f} | {c:.3f}")

    # 4. CHECK VARIANCE
    goals_std = out['goals'].std().item()
    if goals_std < 0.01:
        print(f"\n  ❌ FAIL: Model predicts same value for everyone (Std={goals_std:.4f})")
        sys.exit(1)
    else:
        print(f"\n  ✅ PASS: Predictions vary by player (Std={goals_std:.4f})")

    # 5. LIVE CORRELATION CALCULATION
    print("\n[3. LIVE MATH CHECK]")
    # Stack columns: Goals, Assists, Shots, Cards
    matrix = torch.cat([out['goals'], out['assists'], out['shots'], out['cards']], dim=1).numpy()
    live_corr = np.corrcoef(matrix, rowvar=False)
    
    goals_shots_corr = live_corr[0, 2] # Goals vs Shots
    print(f"  - Live Batch Correlation (Goals vs Shots): {goals_shots_corr:.4f}")
    
    if 0.4 < goals_shots_corr < 0.99:
        print("  ✅ PASS: Correlation is physically realistic (Positive but not 1.0)")
    else:
        print(f"  ❌ FAIL: Correlation suspicious ({goals_shots_corr:.4f})")

    # 6. COMPARE TO SAVED MATRIX
    print("\n[4. FILE INTEGRITY]")
    if os.path.exists('models/gnn_correlation_matrix.npy'):
        saved_corr = np.load('models/gnn_correlation_matrix.npy')
        saved_val = saved_corr[0, 2]
        print(f"  - Saved Matrix Value (Goals vs Shots):     {saved_val:.4f}")
        
        # The saved matrix uses calibration (shrinkage), so it should be DIFFERENT from raw
        if abs(saved_val - goals_shots_corr) > 0.01:
            print("  ✅ PASS: Saved matrix is Calibrated (Different from raw batch)")
        else:
            print("  ⚠️ WARNING: Saved matrix matches raw batch exactly (Did calibration run?)")
    else:
        print("  ❌ FAIL: Correlation matrix file missing!")

if __name__ == "__main__":
    audit_correlation_engine()