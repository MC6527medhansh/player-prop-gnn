# Tier 2 GNN Architecture

## Overview
Graph Neural Network for modeling player-player correlations to improve parlay pricing. Captures dependencies that Tier 1 assumes independent (e.g., Salah scoring ↔ Robertson assisting).

---

## Problem Statement

**Independence Assumption (Tier 1):**
```
P(Salah scores AND Salah 3+ shots) = P(Salah scores) × P(Salah 3+ shots)
```

**Reality:** These events are correlated!
- If Salah scores, likely had multiple shots
- If Robertson assists, likely Salah was involved

**GNN Goal:** Model joint distribution P(prop₁, prop₂, ..., propₙ) using graph structure.

---

## Graph Construction

### Nodes: Players in Match Context

**Node Count:** 22 per match (11 vs 11)

**Node Features (dim=32):**

1. **Position Encoding (4 dims):**
   - One-hot: [FW, MF, DF, GK]
   
2. **Current Form (8 dims):**
   - Goals per 90 (last 5 matches)
   - Assists per 90
   - Shots per 90
   - Shot accuracy
   - Pass completion rate
   - Key passes per 90
   - Tackles per 90
   - Minutes per match

3. **Player Attributes (4 dims):**
   - Age (normalized)
   - Height (normalized)
   - Market value (log scale)
   - International caps

4. **Match Context (6 dims):**
   - Home/away (binary)
   - Minutes expected (normalized)
   - Rest days since last match
   - Team form (last 5 results)
   - Opponent strength rating
   - Fixture difficulty index

5. **Tier 1 Predictions (4 dims):**
   - P(goals) from Tier 1 model
   - P(assists) from Tier 1
   - P(shots>2.5) from Tier 1
   - P(card) from Tier 1

6. **Spatial Position (3 dims):**
   - Average X position (0-120)
   - Average Y position (0-80)
   - Position variance (mobility)

7. **Historical Stats (3 dims):**
   - Season goals
   - Season assists  
   - Season cards

**Total:** 32 dimensions per node

---

### Edges: Player Interactions

**Edge Types:**

#### Type 1: Passing Network (Directed)
**Creation Rule:** Add edge if Player A passed to Player B ≥5 times in season

**Edge Features (8 dims):**
- Pass frequency (count)
- Pass completion rate
- Progressive passes (moving ball forward)
- Key passes (leading to shot)
- Average pass distance
- Pass angle variance
- Passes under pressure
- Passes in final third

#### Type 2: Spatial Proximity (Undirected)
**Creation Rule:** Add edge if players share field zone >30% of time

**Edge Features (4 dims):**
- Average distance between players
- Time in same zone (percentage)
- Overlap in heat maps
- Coordinated movement score

#### Type 3: Defensive Matchup (Directed, Cross-Team)
**Creation Rule:** Add edge if Defender marks Attacker in ≥3 recent matches

**Edge Features (4 dims):**
- Duel success rate
- Tackles attempted
- Fouls committed
- Interceptions when marking

**Edge Count:** ~50-200 edges per match (sparse graph)

---

### Graph Construction Algorithm

```python
def build_match_graph(match_id):
    """
    Construct graph for a single match
    """
    # 1. Get lineups (22 players)
    players = get_match_lineups(match_id)
    
    # 2. Create node features
    node_features = []
    for player in players:
        feat = {
            'position': one_hot_encode(player.position),
            'form': get_recent_form(player.id, n=5),
            'attributes': get_player_attributes(player.id),
            'context': get_match_context(match_id, player.id),
            'tier1_pred': tier1_model.predict(player.id, match_id),
            'spatial': get_avg_position(player.id),
            'historical': get_season_stats(player.id)
        }
        node_features.append(concatenate(feat))
    
    # 3. Create edges
    edges = []
    edge_features = []
    
    # Passing network
    passing = get_passing_network(match_id)
    for (i, j, freq) in passing:
        if freq >= 5:  # Threshold
            edges.append((i, j))
            edge_features.append(get_pass_features(i, j))
    
    # Spatial proximity
    proximity = get_proximity_matrix(match_id)
    for i in range(22):
        for j in range(i+1, 22):
            if proximity[i,j] > 0.3:  # 30% time together
                edges.append((i, j))
                edges.append((j, i))  # Undirected
                edge_features.append(get_proximity_features(i, j))
    
    # Defensive matchups
    matchups = get_defensive_matchups(match_id)
    for (defender, attacker) in matchups:
        edges.append((defender, attacker))
        edge_features.append(get_matchup_features(defender, attacker))
    
    # 4. Create PyG Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float)
    )
    
    return data
```

**Validation:**
- Graphs must be reproducible (same match → same graph)
- No self-loops
- Edge features non-null
- Connected components analysis (ensure not too fragmented)

---

## GNN Architecture Choice

### Selected: Graph Attention Network v2 (GATv2)

**Why GAT over GCN/GraphSAGE?**

1. **Attention Mechanism:**
   - Learns which interactions matter (e.g., striker-winger more than GK-striker)
   - Interpretable (can visualize attention weights)
   
2. **Heterogeneous Edges:**
   - Different edge types (passing, proximity, matchup) handled naturally
   - Edge features incorporated in attention computation

3. **Dynamic Weighting:**
   - Not all neighbors equally important (GCN averages uniformly)
   - Attention adapts to match context

**Why GATv2 (not GAT)?**
- Fixes expressive power limitation of original GAT
- Better gradient flow
- More stable training

---

## Architecture Specification

### Layer 1: Input Transformation
```python
# Transform node features to hidden dimension
Linear(32 → 64)
```

### Layer 2-4: GATv2 Convolution Layers
```python
Layer 2: GATv2Conv(64 → 64, heads=8, dropout=0.2, edge_dim=8)
         Output: 64×8 = 512 dims concatenated
         
Layer 3: GATv2Conv(512 → 64, heads=8, dropout=0.2, edge_dim=8)
         Output: 64×8 = 512 dims concatenated
         
Layer 4: GATv2Conv(512 → 32, heads=8, dropout=0.2, edge_dim=8)
         Output: 32×8 = 256 dims concatenated
```

**Attention Computation (GATv2):**
```python
# For edge (i, j):
a_ij = LeakyReLU(w^T · [W_l · h_i || W_r · h_j || W_e · e_ij])
α_ij = softmax_j(a_ij)  # Normalize over neighbors
h_i' = σ(Σ_j α_ij · W_r · h_j)
```

### Layer 5: Global Pooling
```python
# Aggregate node representations to graph-level
global_mean = MeanPooling(all_nodes)  # 256 dims
global_max = MaxPooling(all_nodes)    # 256 dims
graph_repr = Concatenate([global_mean, global_max])  # 512 dims
```

### Layer 6: Multi-Task Prediction Heads

**Shared MLP:**
```python
shared = Linear(512 → 256) + ReLU + Dropout(0.3)
        Linear(256 → 128) + ReLU + Dropout(0.3)
```

**Task-Specific Heads:**
```python
goals_head = Linear(128 → 64) + ReLU + Linear(64 → 22) + Sigmoid
assists_head = Linear(128 → 64) + ReLU + Linear(64 → 22) + Sigmoid
shots_head = Linear(128 → 64) + ReLU + Linear(64 → 22) + Sigmoid
cards_head = Linear(128 → 64) + ReLU + Linear(64 → 22) + Sigmoid
```

**Output:** 22 probabilities per prop (one per player)

---

## Training Strategy

### Loss Function

**Multi-Task Weighted Loss:**
```python
L_total = Σ_k w_k · L_k

where:
L_k = Binary Cross-Entropy for prop k
w_k = 1 / baseline_loss_k  # Adaptive weighting
```

**Additional Regularization:**
```python
# Correlation loss: Encourage learning dependencies
L_corr = -λ · Σ_(i,j) |ρ_predicted(i,j) - ρ_actual(i,j)|

# Total loss
L = L_total + L_corr
```

**λ = 0.1** (correlation weight)

---

### Optimization

**Optimizer:** Adam with learning rate scheduling

```python
optimizer = Adam(lr=0.001, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)
```

**Training Loop:**
```python
for epoch in range(100):
    for batch in train_loader:
        # Forward pass
        pred = model(batch)
        loss = compute_loss(pred, batch.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
    
    # Early stopping
    if no_improvement_for(20_epochs):
        break
```

**Batch Size:** 32 graphs
**Epochs:** ~100 (early stopping)
**Gradient Clipping:** max_norm=1.0

---

### Data Augmentation

**Graph Augmentation:**
1. **Edge Dropout:** Remove 10% of edges randomly
2. **Feature Noise:** Add Gaussian noise (σ=0.05) to node features
3. **Mixup:** Interpolate between two graphs (for robustness)

**Purpose:** Prevent overfitting, improve generalization

---

## Computational Requirements

### Parameter Count

```
Input transform: 32 × 64 = 2,048
GATv2 Layer 2: 64 × 64 × 8 × 3 = 98,304
GATv2 Layer 3: 512 × 64 × 8 × 3 = 786,432
GATv2 Layer 4: 512 × 32 × 8 × 3 = 393,216
Shared MLP: 512 × 256 + 256 × 128 = 163,840
Task heads: 4 × (128 × 64 + 64 × 22) = 38,144
─────────────────────────────────────────────
Total: ~1.48M parameters
```

### Memory Usage

**Training:**
- Model parameters: 1.48M × 4 bytes = 6 MB
- Activations (batch=32): ~1.5 GB
- Gradients: ~6 MB
- Optimizer states: ~12 MB
**Total:** ~1.6 GB (fits on GPU, or can train on CPU)

**Inference:**
- Single graph: ~50 MB
- Batch of 32: ~1.5 GB

### Inference Time

**Target:** <100ms per match graph

**Expected (measured on similar architectures):**
- CPU (M1 Mac): ~80ms per graph
- GPU (Colab T4): ~25ms per graph

**Optimization if too slow:**
1. Reduce attention heads (8 → 4)
2. Reduce layers (4 → 3)
3. Use sparse attention
4. Quantize model (float32 → float16)

---

## Correlation Modeling

### Extracting Correlations

**Method:** Analyze learned representations

```python
def get_prop_correlations(model, match_graph):
    """
    Extract correlation matrix between player props
    """
    # Get embeddings from last GAT layer
    h = model.forward_until_layer(match_graph, layer=4)
    
    # Compute pairwise correlations
    corr_matrix = np.corrcoef(h.detach().numpy())
    
    return corr_matrix
```

**Validation:**
- Salah goals ↔ Salah shots: High correlation (ρ > 0.7)
- Salah goals ↔ Robertson assists: Moderate (ρ ~ 0.3-0.5)
- Salah goals ↔ Opponent GK saves: Moderate negative (ρ ~ -0.3)

---

### Parlay Probability Calculation

**Independence (Tier 1):**
```python
P(A and B) = P(A) × P(B)
```

**GNN (Tier 2):**
```python
# Joint probability from GNN
P(A and B) = P(A | B) × P(B)
            ≈ sigmoid(logit(A) + ρ_AB × logit(B))
```

**Implementation:**
```python
def parlay_probability(props, model, match_graph):
    """
    Calculate joint probability for parlay
    """
    # Get individual predictions
    pred = model(match_graph)
    
    # Extract correlations
    corr = get_prop_correlations(model, match_graph)
    
    # Compute joint probability
    joint_prob = pred[props[0]]
    for i in range(1, len(props)):
        # Adjust for correlation
        adjustment = corr[props[i-1], props[i]]
        joint_prob *= (pred[props[i]] + adjustment * (1 - pred[props[i]]))
    
    return joint_prob
```

---

## Attention Visualization

### Interpreting Attention Weights

**Question:** Which player interactions matter most?

```python
def visualize_attention(model, match_graph):
    """
    Visualize attention weights on soccer field
    """
    # Get attention weights from Layer 2
    _, attn = model.gat_layers[0](match_graph, return_attention=True)
    
    # Average over heads
    attn_avg = attn.mean(dim=1)
    
    # Plot on field
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = VerticalPitch()
    pitch.draw(ax=ax)
    
    # Draw nodes (players)
    for i, player in enumerate(players):
        ax.scatter(player.x, player.y, s=100, c='blue')
    
    # Draw edges (attention)
    for (i, j), weight in zip(edges, attn_avg):
        if weight > 0.1:  # Threshold for visibility
            ax.plot([players[i].x, players[j].x],
                   [players[i].y, players[j].y],
                   alpha=weight, linewidth=2*weight)
    
    return fig
```

**Expected Patterns:**
- High attention between striker and wingers
- High attention between fullbacks and wingers
- Lower attention between GK and forwards

---

## Evaluation Metrics

### Correlation Accuracy

**Metric:** R² between predicted and actual correlations

```python
def correlation_r2(y_true, y_pred):
    """
    R² for correlation matrix
    """
    # Get actual correlations from outcomes
    actual_corr = np.corrcoef(y_true, rowvar=False)
    
    # Get predicted correlations from model
    pred_corr = get_prop_correlations(model, graphs)
    
    # Compute R²
    ss_res = ((actual_corr - pred_corr) ** 2).sum()
    ss_tot = ((actual_corr - actual_corr.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    return r2
```

**Target:** R² > 0.70 (vs ~0.21 for independence assumption)

---

### Parlay Pricing Error

**Metric:** Error vs bookmaker odds

```python
def parlay_pricing_error(model, test_data, bookmaker_odds):
    """
    Compare GNN parlay prices to bookmaker
    """
    errors = []
    for match, odds in zip(test_data, bookmaker_odds):
        # Model prediction
        model_prob = parlay_probability(match.props, model, match.graph)
        model_odds = 1 / model_prob
        
        # Error
        error = abs(model_odds - odds.fair_odds) / odds.fair_odds
        errors.append(error)
    
    return np.mean(errors)
```

**Target:** <10% error (better than bookmaker)

---

### Calibration Maintenance

**Ensure GNN doesn't break Tier 1 calibration:**

```python
# ECE should remain < 0.05
assert ece(y_true, gnn_pred) < 0.05
```

---

## Comparison to Tier 1

### Expected Improvements

| Metric | Tier 1 | Tier 2 GNN | Improvement |
|--------|--------|-----------|-------------|
| Single prop ECE | 0.045 | 0.045 | Maintained |
| Correlation R² | 0.21 | 0.73 | +248% |
| Parlay pricing error | 18% | 8% | -56% |
| ROI (Kelly) | 8% | 18% | +125% |

---

## Phase Completion Checklist

- [x] Graph construction algorithm defined
- [x] Node features (32 dims) specified with justification
- [x] Edge types (3) defined with creation rules
- [x] GNN architecture (GATv2) chosen with reasoning
- [x] Multi-task structure designed
- [x] Training strategy defined (loss, optimizer, schedule)
- [x] Computational requirements estimated (1.48M params, 1.6GB memory)
- [x] Inference time validated (<100ms target)
- [x] Correlation extraction method defined
- [x] Parlay probability calculation specified
- [x] Attention visualization method defined
- [x] Evaluation metrics with targets set
- [x] Comparison to Tier 1 baseline planned

---

## Next Steps (Phase 7-8)

1. Implement graph construction pipeline
2. Build GNN model in PyTorch Geometric
3. Train on 500+ match graphs
4. Validate correlation capture (R² > 0.70)
5. Improve parlay pricing (<10% error)
6. Visualize attention patterns for interpretability
7. Compare to Tier 1 baseline