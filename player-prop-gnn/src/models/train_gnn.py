"""
Phase 6.4: GNN Training Pipeline (Robust CPU Version)
Optimized for Apple Silicon stability and speed.

Changes:
- REMOVED: All MPS/Metal code (source of instability)
- ADDED: Pure CPU execution (fastest for small graphs)
- ADDED: Explicit error handling for data integrity
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import psycopg2
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv

# --- ROBUST IMPORT SETUP ---
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.graph_dataset import MatchGraphDataset
from src.models.gnn_gat import PlayerPropGAT

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_db_config():
    """Load DB config securely."""
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent.parent
    
    env_path = project_root / '.env'
    if not env_path.exists():
        env_path = project_root / 'deployment' / 'docker' / '.env'
        
    load_dotenv(env_path)
    
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'football_props'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD')
    }

class GNNTrainer:
    def __init__(
        self, 
        model, 
        device, 
        learning_rate=0.001, 
        patience=10,
        checkpoint_dir='models'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss Functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Loss Weights (Can be tuned)
        self.weights = {
            'goals': 2.0,
            'assists': 1.5,
            'shots': 1.0,
            'cards': 1.5
        }

    def compute_loss(self, preds, batch):
        """Multi-task weighted loss calculation."""
        y_goals = batch.y[:, 0].unsqueeze(1)
        y_assists = batch.y[:, 1].unsqueeze(1)
        y_shots = batch.y[:, 2].unsqueeze(1)
        y_cards = batch.y[:, 3].unsqueeze(1)

        l_goals = self.mse_loss(preds['goals'], y_goals)
        l_assists = self.mse_loss(preds['assists'], y_assists)
        l_shots = self.mse_loss(preds['shots'], y_shots)
        l_cards = self.bce_loss(preds['cards'], y_cards)
        
        total_loss = (
            self.weights['goals'] * l_goals +
            self.weights['assists'] * l_assists +
            self.weights['shots'] * l_shots +
            self.weights['cards'] * l_cards
        )
        
        return total_loss, {
            'goals': l_goals.item(),
            'shots': l_shots.item(),
            'cards': l_cards.item()
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        metrics = {'goals': 0, 'shots': 0, 'cards': 0}
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            preds = self.model(batch.x, batch.edge_index, batch.edge_attr)
            loss, batch_metrics = self.compute_loss(preds, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] += v
                
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in metrics.items()}
        return avg_loss, avg_metrics

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        metrics = {'goals': 0, 'shots': 0, 'cards': 0}
        
        for batch in loader:
            batch = batch.to(self.device)
            preds = self.model(batch.x, batch.edge_index, batch.edge_attr)
            loss, batch_metrics = self.compute_loss(preds, batch)
            
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] += v
                
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in metrics.items()}
        return avg_loss, avg_metrics

    def fit(self, train_loader, val_loader, epochs=50):
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            train_loss, _ = self.train_epoch(train_loader)
            val_loss, v_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch:02d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} [G:{v_metrics['goals']:.3f} S:{v_metrics['shots']:.3f} C:{v_metrics['cards']:.3f}]"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'gnn_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
        return best_val_loss

def main():
    # --- ROBUST DEVICE SELECTION ---
    # We explicitly force CPU to avoid the Mac Metal/MPS instability.
    # For graphs of this size (22 nodes), CPU is effectively instant.
    device = torch.device("cpu")
    logger.info("✅ Selected Device: CPU (Optimized for Stability)")

    db_config = get_db_config()
    
    logger.info("Loading Dataset...")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SELECT match_id FROM matches ORDER BY match_date ASC")
    all_match_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    
    # Enable reprocessing to fix any cached corrupted graphs
    dataset = MatchGraphDataset(
        root='data/processed/gnn_v1',
        match_ids=all_match_ids,
        db_config=db_config
    )
    
    if len(dataset) == 0:
        logger.error("Dataset empty!")
        return

    # 80/20 Split
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    model = PlayerPropGAT(in_channels=12, hidden_channels=64, heads=4, dropout=0.2)
    
    trainer = GNNTrainer(model, device)
    trainer.fit(train_loader, val_loader, epochs=30)

if __name__ == "__main__":
    main()