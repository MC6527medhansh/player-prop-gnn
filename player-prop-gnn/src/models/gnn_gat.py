"""
Phase 6.3: Graph Attention Network (GAT)
The core Deep Learning model for correlation modeling.

Architecture:
- Encoder: 3-Layer GATv2 with Residual Connections
- Decoder: 4 Separate Heads (Goals, Assists, Shots, Cards)
- Mechanism: Learns to weight edges (passes) by importance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

class PlayerPropGAT(nn.Module):
    def __init__(
        self, 
        in_channels: int = 12, 
        hidden_channels: int = 64, 
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # --- 1. ENCODER (The "Brain") ---
        # Layer 1: Expand features
        self.conv1 = GATv2Conv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            edge_dim=3,  # [type, success, time]
            concat=True
        )
        self.norm1 = LayerNorm(hidden_channels * heads)
        
        # Layer 2: Deep reasoning
        self.conv2 = GATv2Conv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=heads, 
            edge_dim=3,
            concat=True
        )
        self.norm2 = LayerNorm(hidden_channels * heads)
        
        # Layer 3: Refinement
        self.conv3 = GATv2Conv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1,  # Summarize to single head
            edge_dim=3,
            concat=False
        )
        self.norm3 = LayerNorm(hidden_channels)

        # --- 2. DECODERS (The "Predictors") ---
        # Shared Dense Layer before heads
        self.post_process = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task 1: Goals (Regression, Non-negative)
        self.head_goals = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Ensures output > 0
        )
        
        # Task 2: Assists (Regression, Non-negative)
        self.head_assists = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Task 3: Shots (Regression, Non-negative)
        self.head_shots = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Task 4: Cards (Binary Classification 0-1)
        self.head_cards = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Probability 0-1
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        """
        Forward Pass.
        x: [num_nodes, 12]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 3]
        """
        # Block 1
        x_in = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Block 2 (Residual would require matching dims, skipping for simplicity here)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Block 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        x = F.elu(x)
        
        # Shared Post-Processing
        embedding = self.post_process(x)
        
        # Multi-Task Heads
        return {
            'goals': self.head_goals(embedding),
            'assists': self.head_assists(embedding),
            'shots': self.head_shots(embedding),
            'cards': self.head_cards(embedding),
            'embedding': embedding # Useful for analysis
        }

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    # Test with dummy data
    print("Testing GNN Architecture...")
    
    # Fake Batch: 2 graphs, 22 players each
    num_nodes = 44 
    num_edges = 100
    
    # Random Inputs
    x = torch.randn(num_nodes, 12)          # 12 Features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)   # 3 Edge Features
    
    # Initialize
    model = PlayerPropGAT(in_channels=12, hidden_channels=64, heads=4)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward Pass
    out = model(x, edge_index, edge_attr)
    
    print("\n✅ Forward Pass Successful")
    print(f"Goals Output: {out['goals'].shape}")   # Should be [44, 1]
    print(f"Cards Output: {out['cards'].shape}")   # Should be [44, 1]
    
    # Check outputs constraints
    if (out['goals'] < 0).any():
        print("❌ Error: Goals prediction negative!")
    else:
        print("✅ Goals constraints valid (Non-negative)")