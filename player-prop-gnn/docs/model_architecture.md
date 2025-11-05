# Tier 1 Model Architecture

## Overview
Bayesian multi-task learning model for predicting 4 player props with full uncertainty quantification. Uses hierarchical priors for player-specific effects with appropriate shrinkage.

---

## Model Choice: Bayesian Hierarchical Multi-Task

**Why Bayesian?**
- **Uncertainty quantification:** Critical for betting decisions (need confidence intervals)
- **Calibration:** Bayesian posteriors naturally calibrated when priors are reasonable
- **Small data handling:** Hierarchical priors provide shrinkage for players with few matches
- **Interpretability:** Can explain every parameter's meaning

**Why Multi-Task?**
- **Parameter sharing:** Goals and assists are related (attacking players)
- **Data efficiency:** Shared representations improve sample efficiency
- **Correlation structure:** Natural framework for modeling prop correlations

---

## Mathematical Specification

### Target Props
1. **Goals:** Binary (0 or 1+)
2. **Assists:** Binary (0 or 1+)  
3. **Shots on Target:** Binary (over/under 2.5)
4. **Cards:** Binary (yellow or red)

### Likelihood

For player *i* in match *j*, prop *k*:

```
y_ijk ~ Bernoulli(p_ijk)
logit(p_ijk) = α_k + β_k[i] + γ_k[opponent[j]] + δ_k × home[j] + θ_k × minutes[ij]
```

**Components:**
- `α_k`: Baseline log-odds for prop k (league average)
- `β_k[i]`: Player-specific effect for prop k (hierarchical)
- `γ_k[opponent[j]]`: Opponent strength effect
- `δ_k`: Home advantage effect
- `θ_k`: Minutes played effect (more minutes → more opportunities)

---

## Prior Specification

### 1. Baseline Rates (α_k)

**Goals:**
```python
α_goals ~ Normal(-1.0, 0.3)
```
**Justification:** 
- League average: ~27% of players score per match
- logit(0.27) ≈ -1.0
- σ = 0.3 allows ±10% variation (weakly informative)

**Assists:**
```python
α_assists ~ Normal(-1.2, 0.3)
```
**Justification:**
- League average: ~23% of players assist
- logit(0.23) ≈ -1.2

**Shots over 2.5:**
```python
α_shots ~ Normal(-0.8, 0.3)
```
**Justification:**
- ~31% of players exceed 2.5 shots on target
- logit(0.31) ≈ -0.8

**Cards:**
```python
α_cards ~ Normal(-2.0, 0.4)
```
**Justification:**
- ~12% of players get carded per match
- logit(0.12) ≈ -2.0
- Higher σ due to more variance

---

### 2. Player Effects (β_k[i]) - HIERARCHICAL

**Hierarchy:** League → Position → Player

```python
# Hyperpriors (population-level)
σ_player_k ~ HalfNormal(1.0)

# Position-level means
μ_position_k[p] ~ Normal(0, 0.5)  # p ∈ {FW, MF, DF, GK}

# Player-level effects
β_k[i] ~ Normal(μ_position[position[i]], σ_player_k)
```

**Justification:**
- Forwards score more than defenders (position effects)
- Players within position vary (player effects)
- Hierarchical shrinkage: players with few games shrink toward position mean
- σ = 1.0 allows ±2 standard deviations = large effect range

**Position Priors (Goals):**
```python
μ_position_goals[FW] ~ Normal(0.8, 0.2)   # Forwards +80% odds
μ_position_goals[MF] ~ Normal(0.0, 0.2)   # Midfielders baseline
μ_position_goals[DF] ~ Normal(-1.0, 0.2)  # Defenders -100% odds
μ_position_goals[GK] ~ Normal(-3.0, 0.3)  # Goalkeepers rarely score
```

---

### 3. Opponent Effects (γ_k[opponent])

```python
σ_opponent_k ~ HalfNormal(0.5)
γ_k[opponent] ~ Normal(0, σ_opponent_k)
```

**Justification:**
- Weaker defenses → more goals
- σ = 0.5 is moderate effect size
- Centered at 0 (no bias toward strong/weak opponents)

---

### 4. Home Advantage (δ_k)

```python
δ_goals ~ Normal(0.2, 0.1)
δ_assists ~ Normal(0.15, 0.1)
δ_shots ~ Normal(0.25, 0.1)
δ_cards ~ Normal(-0.1, 0.1)  # Fewer cards at home
```

**Justification:**
- Literature: home advantage ~20-30% for attacking stats
- Negative for cards (home bias from referees)

---

### 5. Minutes Effect (θ_k)

```python
θ_goals ~ Normal(0.015, 0.005)
θ_assists ~ Normal(0.012, 0.005)
θ_shots ~ Normal(0.020, 0.005)
θ_cards ~ Normal(0.008, 0.003)
```

**Justification:**
- More minutes → more opportunities (positive effect)
- ~1.5% increase in log-odds per minute for goals
- Standardize minutes to [0, 1] scale for numerical stability

---

## Multi-Task Structure

### Shared Parameters
- **Player embeddings:** β_k[i] uses shared player index
- **Position effects:** Same position grouping across tasks

### Task-Specific
- **Baselines:** α_k different per task
- **Coefficients:** δ_k, θ_k, γ_k task-specific

### Correlation Modeling
Not explicitly modeled in Tier 1 (independence assumed).
**Tier 2 GNN** will capture correlations.

---

## Computational Strategy

### Sampler: NUTS (No-U-Turn Sampler)
```python
import pymc as pm

with pm.Model() as model:
    # ... define priors ...
    
    # Sampling
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=42
    )
```

**Parameters:**
- **Draws:** 2000 per chain = 8000 total samples
- **Tune:** 1000 warmup samples (discarded)
- **Chains:** 4 parallel chains for convergence diagnostics
- **Target accept:** 0.95 (high to avoid divergences)

### Convergence Criteria
- **R-hat < 1.01** for all parameters (Gelman-Rubin statistic)
- **ESS > 400** per parameter (Effective Sample Size)
- **No divergences** (indicates sampler issues)

### Expected Runtime
- **Laptop (M1 Mac):** 15-25 minutes
- **Google Colab (free):** 20-35 minutes
- **Parameters:** ~5000 parameters (1000 players × 4 props + overhead)

### Fallback Strategy
If MCMC doesn't converge in 30 minutes:
1. Try **ADVI** (Variational Inference) - faster but approximate
2. Reduce model complexity (fewer position effects)
3. Use MAP estimation (point estimate, no uncertainty)

---

## Handling Edge Cases

### New Player (No Historical Data)
**Strategy:** Shrink toward position mean
```python
# Player with 0 games gets position prior
β_new_player ~ Normal(μ_position[FW], σ_player)
# After 5 games, mostly driven by player data
```

### Missing Opponent Data
**Strategy:** Use league average
```python
γ_opponent[unknown] = 0  # Neutral opponent effect
```

### Insufficient Games (<5)
**Strategy:** Increase prior strength
```python
# Adaptive prior variance
σ_player = σ_base × sqrt(max(5, n_games) / n_games)
```

### Position Change
**Strategy:** Weighted average of position priors
```python
# Player who plays 70% MF, 30% FW
μ_player = 0.7 × μ_MF + 0.3 × μ_FW
```

---

## Model Evaluation

### Calibration (Primary Metric)
**Expected Calibration Error (ECE):**
```python
def calculate_ece(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece
```

**Target:** ECE < 0.05 (excellent calibration)

### Other Metrics
- **Brier Score:** Measures prediction accuracy
  - Target: < 0.20 (better than baseline)
- **Log Loss:** Penalizes overconfident predictions
  - Target: < 0.50
- **AUC-ROC:** Discrimination ability
  - Target: > 0.70

### Baseline Comparison
Beat simple baselines:
1. **Historical average:** P(goal) = player's season average
2. **Logistic regression:** Same features, no hierarchy
3. **Random:** 50% probability for all

---

## Expected Performance

### Calibration Targets (by prop)
- **Goals:** ECE < 0.05, Brier < 0.20
- **Assists:** ECE < 0.05, Brier < 0.18
- **Shots:** ECE < 0.05, Brier < 0.22
- **Cards:** ECE < 0.06, Brier < 0.10

### Uncertainty Quantification
**Credible Intervals (95%):**
- Goals: Typical width ~0.15 (e.g., [0.20, 0.35])
- Cards: Wider intervals ~0.08 (less data)

**Interpretation:**
- Narrow intervals → Model is confident (more data)
- Wide intervals → Model is uncertain (new player, tough opponent)

---

## Model Validation Checks

### Prior Predictive Checks
**Before seeing data, do priors produce reasonable outcomes?**

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000)
    
# Check: Do simulated goal probabilities look reasonable?
# Should see: Most players 5-40% chance, few >60%
```

### Posterior Predictive Checks
**After training, do predictions match reality?**

```python
with model:
    post_pred = pm.sample_posterior_predictive(trace)
    
# Check: Distribution of predicted goals matches actual
# Should see: Similar mean, variance, and shape
```

### Convergence Diagnostics
```python
# Check R-hat
assert all(az.rhat(trace) < 1.01)

# Check effective sample size
assert all(az.ess(trace) > 400)

# Check divergences
assert trace.sample_stats.diverging.sum() == 0
```

---

## Feature Engineering

### Input Features (per player-match)
1. **Player Stats:**
   - Goals per 90 (last 5, 10 matches)
   - Shots per 90
   - Pass completion rate
   - Minutes per match

2. **Match Context:**
   - Home/away
   - Opponent strength (team rating)
   - Rest days since last match
   - Fixture difficulty

3. **Positional:**
   - Primary position (FW/MF/DF/GK)
   - Average position on field (if available)

**Normalization:**
- Minutes: Scale to [0, 1]
- Rates (goals/90): Standardize (mean 0, std 1)
- Binary features: No scaling needed

---

## Model Versioning

**Filename Convention:**
```
models/tier1_v{major}.{minor}_{date}.pkl

Example: tier1_v1.0_2024-11-01.pkl
```

**Version Increments:**
- **Major:** Model structure changes (new priors, new features)
- **Minor:** Hyperparameter tuning, re-training on more data

**Metadata Storage:**
```python
model_metadata = {
    'version': '1.0',
    'train_date': '2024-11-01',
    'n_players': 856,
    'n_matches': 380,
    'ece_goals': 0.043,
    'ece_assists': 0.048,
    'convergence': 'all_chains_converged',
    'runtime_minutes': 22
}
```

---

## Inference Pipeline

### Prediction Flow
```python
def predict(player_id, match_id, model):
    # 1. Load player features
    features = get_player_features(player_id, match_id)
    
    # 2. Get posterior samples
    with model:
        posterior = pm.sample_posterior_predictive(
            trace,
            var_names=['p_goals', 'p_assists', 'p_shots', 'p_cards']
        )
    
    # 3. Compute mean and credible interval
    pred_mean = posterior.mean(axis=0)
    pred_ci = np.percentile(posterior, [2.5, 97.5], axis=0)
    
    return {
        'probability': float(pred_mean),
        'ci_lower': float(pred_ci[0]),
        'ci_upper': float(pred_ci[1]),
        'uncertainty': float(pred_ci[1] - pred_ci[0])
    }
```

**Latency Target:** < 1 second per player

---

## Phase Completion Checklist

- [x] Can draw graphical model on whiteboard
- [x] Can explain every prior choice with domain knowledge
- [x] Prior choices validated against league data
- [x] Sampling strategy will complete in <30 minutes
- [x] Convergence criteria justified
- [x] Edge cases (new players, missing data) handled
- [x] Evaluation metrics defined with targets
- [x] Baseline comparisons specified

---

## Next Steps (Phase 3)

1. Implement model in PyMC
2. Train on 100+ match dataset
3. Validate convergence (R-hat, ESS)
4. Check calibration (ECE)
5. Beat baseline models
6. Save trained model with metadata