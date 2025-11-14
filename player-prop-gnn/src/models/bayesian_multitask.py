"""
Multi-Task Hierarchical Bayesian Model for Goals, Shots, and Cards
Phase 3.3: Joint prediction with shared position hierarchy

Key Features:
1. SHARED σ_α across all props (simpler, fewer parameters)
2. TIGHT priors (β ~ N(0, 0.25)) learned from Step 3.2
3. HARD BOUNDS on log(λ) for each prop
4. Independent opponent effects per prop
5. Prop-specific feature coefficients

Model Specification:
    For k ∈ {goals, shots, cards}:
    y_k,i ~ Poisson(λ_k,i)
    log(λ_k,i) = CLIP(α_k,pos[i] + γ_k,opp[i] + β_k @ x_i, L_k, U_k)
    
    Shared hierarchy:
    σ_α ~ TruncatedNormal(0, 0.12, upper=0.25)  [SHARED]
    μ_α_k ~ Normal(log(mean_k), 0.15)           [per-prop]
    α_k,pos ~ Normal(μ_α_k, σ_α)                [uses shared σ]
    
    Independent effects:
    γ_k,opp ~ Normal(0, 0.10)                   [per-prop]
    β_k ~ Normal(0, 0.25)                       [per-prop, TIGHT]
    
    Hard bounds:
    Goals: log(λ) ∈ [-6, 1] → λ ∈ [0.0025, 2.7]
    Shots: log(λ) ∈ [-6, 2] → λ ∈ [0.0025, 7.4]
    Cards: log(λ) ∈ [-6, 0] → λ ∈ [0.0025, 1.0]
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sqlalchemy import create_engine
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data(db_url='postgresql://medhanshchoubey@localhost:5432/football_props'):
    """
    Load player features with position information.
    
    Returns DataFrame with columns:
    - player_id, match_id, match_date
    - goals, shots_on_target, yellow_cards, red_cards
    - position, opponent_id, was_home
    - rolling features, opponent_strength, days_since_last_match
    """
    engine = create_engine(db_url)
    
    query = """
    SELECT 
        pf.player_id,
        pf.match_id,
        pf.match_date,
        pf.goals,
        pf.shots_on_target,
        pf.yellow_cards,
        pf.red_cards,
        pf.opponent_id,
        pf.was_home,
        pf.goals_rolling_5,
        pf.shots_on_target_rolling_5,
        pf.opponent_strength,
        pf.days_since_last_match,
        p.position
    FROM player_features pf
    JOIN players p ON pf.player_id = p.player_id
    ORDER BY pf.match_date, pf.player_id
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    # Verify required columns
    required_cols = [
        'player_id', 'match_id', 'match_date', 
        'goals', 'shots_on_target', 'yellow_cards', 'red_cards',
        'opponent_id', 'was_home', 
        'goals_rolling_5', 'shots_on_target_rolling_5',
        'opponent_strength', 'days_since_last_match', 'position'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # CRITICAL: Convert match_date to pd.Timestamp for consistent comparisons
    # Database might return datetime.date, string, or datetime
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Enforce integer types for count data (critical for Poisson)
    int_cols = ['player_id', 'match_id', 'goals', 'shots_on_target', 
                'yellow_cards', 'red_cards', 'opponent_id']
    for col in int_cols:
        df[col] = df[col].astype(np.int64)
    
    # Enforce boolean for was_home
    df['was_home'] = df['was_home'].astype(bool)
    
    # Enforce float for rolling features
    float_cols = ['goals_rolling_5', 'shots_on_target_rolling_5', 
                  'opponent_strength', 'days_since_last_match']
    for col in float_cols:
        df[col] = df[col].astype(np.float64)
    
    # Clean data (drop rows with ANY missing values in required columns)
    df_clean = df.dropna(subset=required_cols).copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after cleaning! Check for missing values.")
    
    # Create combined cards target (integer)
    df_clean['cards_total'] = (df_clean['yellow_cards'] + df_clean['red_cards']).astype(np.int64)
    
    print(f"Loaded {len(df_clean)} records (dropped {len(df) - len(df_clean)} with missing values)")
    print(f"Date range: {df_clean['match_date'].min().date()} to {df_clean['match_date'].max().date()}")
    
    return df_clean


def _prep_multitask_inputs(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, np.ndarray]:
    """
    Prepare inputs for multi-task model with robust error handling.
    
    Returns:
        targets: Dict with 'goals', 'shots', 'cards' arrays
        indices: Dict with 'position', 'opponent' arrays
        coords: Dict with coordinate arrays
        X: Feature matrix (standardized)
    """
    if len(df) == 0:
        raise ValueError("Empty dataframe provided")
    
    # Targets (all int64, non-negative)
    targets = {
        'goals': df['goals'].astype(np.int64).values,
        'shots': df['shots_on_target'].astype(np.int64).values,
        'cards': df['cards_total'].astype(np.int64).values
    }
    
    # Validate targets
    for name, arr in targets.items():
        if not np.all(arr >= 0):
            raise ValueError(f"{name} contains negative values")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    
    # Position encoding (deterministic sorting for reproducibility)
    unique_positions = sorted(df['position'].unique())
    if len(unique_positions) == 0:
        raise ValueError("No positions found in data")
    
    position_map = {pos: idx for idx, pos in enumerate(unique_positions)}
    pos_idx = df['position'].map(position_map).values
    
    # Check for unmapped positions
    if np.any(pd.isna(pos_idx)):
        raise ValueError("Some positions could not be mapped")
    
    pos_idx = pos_idx.astype(np.int64)
    
    # Opponent encoding (deterministic sorting)
    unique_opponents = sorted(df['opponent_id'].unique())
    if len(unique_opponents) == 0:
        raise ValueError("No opponents found in data")
    
    opponent_map = {opp: idx for idx, opp in enumerate(unique_opponents)}
    opp_idx = df['opponent_id'].map(opponent_map).values
    
    # Check for unmapped opponents
    if np.any(pd.isna(opp_idx)):
        raise ValueError("Some opponents could not be mapped")
    
    opp_idx = opp_idx.astype(np.int64)
    
    indices = {
        'position': pos_idx,
        'opponent': opp_idx
    }
    
    # Features (standardized)
    feature_cols = [
        'goals_rolling_5',
        'shots_on_target_rolling_5',
        'opponent_strength',
        'days_since_last_match',
        'was_home'
    ]
    
    # Verify all feature columns exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[feature_cols].values.astype(np.float64)
    
    # Check for non-finite values
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains non-finite values (NaN or inf)")
    
    # Standardize (critical for tight priors to work correctly)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    # Handle zero variance (constant features)
    zero_var_mask = X_std < 1e-10
    if np.any(zero_var_mask):
        print(f"  Warning: Zero-variance features: {[feature_cols[i] for i, m in enumerate(zero_var_mask) if m]}")
        X_std[zero_var_mask] = 1.0
    
    X = (X - X_mean) / X_std
    
    # Final check
    if not np.all(np.abs(X) < 100):
        raise ValueError("Standardized features have extreme values")
    
    # Coordinates for PyMC
    coords = {
        'position': unique_positions,
        'opponent': unique_opponents,
        'feature': feature_cols,
        'obs': np.arange(len(df))
    }
    
    return targets, indices, coords, X


def build_multitask_model(df: pd.DataFrame) -> Tuple[pm.Model, Dict]:
    """
    Build multi-task Bayesian model for goals, shots, and cards.
    
    Returns:
        model: PyMC model
        coords: Coordinate dict for dimensions
    """
    print("="*60)
    print("BUILDING MULTI-TASK BAYESIAN MODEL")
    print("="*60)
    
    # Prepare inputs
    targets, indices, coords, X = _prep_multitask_inputs(df)
    
    # Extract for model
    y_goals = targets['goals']
    y_shots = targets['shots']
    y_cards = targets['cards']
    pos_idx = indices['position']
    opp_idx = indices['opponent']
    
    # Observed means for hyperpriors
    mean_goals = y_goals.mean() + 0.01  # Avoid log(0)
    mean_shots = y_shots.mean() + 0.01
    mean_cards = y_cards.mean() + 0.01
    
    print(f"\nObserved means:")
    print(f"  Goals: {mean_goals:.3f}")
    print(f"  Shots: {mean_shots:.3f}")
    print(f"  Cards: {mean_cards:.3f}")
    
    with pm.Model(coords=coords) as model:
        # Data containers
        y_goals_data = pm.Data('y_goals', y_goals, dims='obs')
        y_shots_data = pm.Data('y_shots', y_shots, dims='obs')
        y_cards_data = pm.Data('y_cards', y_cards, dims='obs')
        
        pos_data = pm.Data('position_idx', pos_idx, dims='obs')
        opp_data = pm.Data('opponent_idx', opp_idx, dims='obs')
        X_data = pm.Data('X', X, dims=('obs', 'feature'))
        
        # ========================================
        # SHARED POSITION VARIANCE (KEY DESIGN)
        # ========================================
        print(f"\n1. Shared Position Variance (CRITICAL):")
        sigma_alpha = pm.TruncatedNormal(
            'sigma_alpha', 
            mu=0, 
            sigma=0.12, 
            lower=0, 
            upper=0.25
        )
        print(f"   σ_α ~ TruncatedNormal(0, 0.12, upper=0.25)")
        print(f"   → SHARED across all props")
        print(f"   → Max position effect: exp(0.25) = 1.28x")
        
        # ========================================
        # PER-PROP HYPERPRIORS
        # ========================================
        print(f"\n2. Per-Prop Hyperpriors:")
        
        mu_alpha_goals = pm.Normal(
            'mu_alpha_goals',
            mu=np.log(mean_goals),
            sigma=0.15
        )
        print(f"   μ_α_goals ~ Normal({np.log(mean_goals):.3f}, 0.15)")
        
        mu_alpha_shots = pm.Normal(
            'mu_alpha_shots',
            mu=np.log(mean_shots),
            sigma=0.15
        )
        print(f"   μ_α_shots ~ Normal({np.log(mean_shots):.3f}, 0.15)")
        
        mu_alpha_cards = pm.Normal(
            'mu_alpha_cards',
            mu=np.log(mean_cards),
            sigma=0.15
        )
        print(f"   μ_α_cards ~ Normal({np.log(mean_cards):.3f}, 0.15)")
        
        # ========================================
        # POSITION EFFECTS (SHARED σ_α)
        # ========================================
        print(f"\n3. Position Effects (using shared σ_α):")
        
        alpha_goals_pos = pm.Normal(
            'alpha_goals_position',
            mu=mu_alpha_goals,
            sigma=sigma_alpha,
            dims='position'
        )
        
        alpha_shots_pos = pm.Normal(
            'alpha_shots_position',
            mu=mu_alpha_shots,
            sigma=sigma_alpha,
            dims='position'
        )
        
        alpha_cards_pos = pm.Normal(
            'alpha_cards_position',
            mu=mu_alpha_cards,
            sigma=sigma_alpha,
            dims='position'
        )
        print(f"   All position effects use same σ_α")
        
        # ========================================
        # OPPONENT EFFECTS (INDEPENDENT, TIGHT)
        # ========================================
        print(f"\n4. Opponent Effects (independent per prop):")
        
        gamma_goals_opp = pm.Normal(
            'gamma_goals_opponent',
            mu=0,
            sigma=0.10,
            dims='opponent'
        )
        
        gamma_shots_opp = pm.Normal(
            'gamma_shots_opponent',
            mu=0,
            sigma=0.10,
            dims='opponent'
        )
        
        gamma_cards_opp = pm.Normal(
            'gamma_cards_opponent',
            mu=0,
            sigma=0.10,
            dims='opponent'
        )
        print(f"   γ ~ Normal(0, 0.10) for each prop")
        
        # ========================================
        # FEATURE COEFFICIENTS (TIGHT!)
        # ========================================
        print(f"\n5. Feature Coefficients (MUCH TIGHTER):")
        
        beta_goals = pm.Normal(
            'beta_goals',
            mu=0,
            sigma=0.25,
            dims='feature'
        )
        
        beta_shots = pm.Normal(
            'beta_shots',
            mu=0,
            sigma=0.25,
            dims='feature'
        )
        
        beta_cards = pm.Normal(
            'beta_cards',
            mu=0,
            sigma=0.25,
            dims='feature'
        )
        print(f"   β ~ Normal(0, 0.25) for each prop")
        print(f"   → 1-SD feature → 1.28x effect")
        print(f"   → Prevents extreme cumulative effects")
        
        # ========================================
        # LINEAR PREDICTORS WITH HARD BOUNDS
        # ========================================
        print(f"\n6. Linear Predictors with HARD BOUNDS:")
        
        # Goals: λ ∈ [0.0025, 2.7]
        log_lambda_goals_raw = (
            alpha_goals_pos[pos_data] +
            gamma_goals_opp[opp_data] +
            pm.math.dot(X_data, beta_goals)
        )
        log_lambda_goals = pm.Deterministic(
            'log_lambda_goals',
            pm.math.clip(log_lambda_goals_raw, -6.0, 1.0),
            dims='obs'
        )
        print(f"   Goals: log(λ) ∈ [-6, 1] → λ ∈ [0.0025, 2.7]")
        
        # Shots: λ ∈ [0.0025, 7.4]
        log_lambda_shots_raw = (
            alpha_shots_pos[pos_data] +
            gamma_shots_opp[opp_data] +
            pm.math.dot(X_data, beta_shots)
        )
        log_lambda_shots = pm.Deterministic(
            'log_lambda_shots',
            pm.math.clip(log_lambda_shots_raw, -6.0, 2.0),
            dims='obs'
        )
        print(f"   Shots: log(λ) ∈ [-6, 2] → λ ∈ [0.0025, 7.4]")
        
        # Cards: λ ∈ [0.0025, 1.0]
        log_lambda_cards_raw = (
            alpha_cards_pos[pos_data] +
            gamma_cards_opp[opp_data] +
            pm.math.dot(X_data, beta_cards)
        )
        log_lambda_cards = pm.Deterministic(
            'log_lambda_cards',
            pm.math.clip(log_lambda_cards_raw, -6.0, 0.0),
            dims='obs'
        )
        print(f"   Cards: log(λ) ∈ [-6, 0] → λ ∈ [0.0025, 1.0]")
        
        # ========================================
        # LIKELIHOODS
        # ========================================
        print(f"\n7. Likelihoods (Poisson for all props):")
        
        lambda_goals = pm.Deterministic(
            'lambda_goals',
            pm.math.exp(log_lambda_goals),
            dims='obs'
        )
        pm.Poisson('goals_obs', mu=lambda_goals, observed=y_goals_data, dims='obs')
        
        lambda_shots = pm.Deterministic(
            'lambda_shots',
            pm.math.exp(log_lambda_shots),
            dims='obs'
        )
        pm.Poisson('shots_obs', mu=lambda_shots, observed=y_shots_data, dims='obs')
        
        lambda_cards = pm.Deterministic(
            'lambda_cards',
            pm.math.exp(log_lambda_cards),
            dims='obs'
        )
        pm.Poisson('cards_obs', mu=lambda_cards, observed=y_cards_data, dims='obs')
        
        print(f"   All props use Poisson likelihood")
        
    print(f"\n{'='*60}")
    print(f"MODEL BUILT SUCCESSFULLY")
    print(f"{'='*60}\n")
    
    return model, coords


def prior_predictive_check(model: pm.Model, n_samples: int = 2000) -> Dict:
    """
    Sample from prior predictive and compute statistics for all props.
    
    Returns dict with statistics for goals, shots, and cards.
    """
    print("  Sampling from prior predictive (n={})...".format(n_samples))
    
    with model:
        pp = pm.sample_prior_predictive(samples=n_samples, random_seed=42)
    
    # Helper to flatten samples
    def _flatten(arr):
        a = np.asarray(arr)
        if a.ndim >= 3:
            return a.reshape(-1, a.shape[-1])
        return a
    
    # Extract samples for each prop
    goals_obs = _flatten(pp.prior_predictive['goals_obs'].values).flatten()
    shots_obs = _flatten(pp.prior_predictive['shots_obs'].values).flatten()
    cards_obs = _flatten(pp.prior_predictive['cards_obs'].values).flatten()
    
    # Compute statistics
    result = {
        # Goals
        'goals_obs': goals_obs,
        'goals_mean': goals_obs.mean(),
        'goals_median': np.median(goals_obs),
        'goals_ci': (np.percentile(goals_obs, 2.5), np.percentile(goals_obs, 97.5)),
        'goals_pct_extreme': 100 * (goals_obs > 10).sum() / len(goals_obs),
        
        # Shots
        'shots_obs': shots_obs,
        'shots_mean': shots_obs.mean(),
        'shots_median': np.median(shots_obs),
        'shots_ci': (np.percentile(shots_obs, 2.5), np.percentile(shots_obs, 97.5)),
        'shots_pct_extreme': 100 * (shots_obs > 10).sum() / len(shots_obs),
        
        # Cards
        'cards_obs': cards_obs,
        'cards_mean': cards_obs.mean(),
        'cards_median': np.median(cards_obs),
        'cards_ci': (np.percentile(cards_obs, 2.5), np.percentile(cards_obs, 97.5)),
        'cards_pct_extreme': 100 * (cards_obs > 5).sum() / len(cards_obs),
    }
    
    print(f"  ✓ Prior predictive samples collected")
    
    return result


def fit_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42
) -> az.InferenceData:
    """
    Fit model using NUTS sampler with fallback to ADVI.
    """
    print(f"\nSampling with NUTS:")
    print(f"  Draws: {draws}")
    print(f"  Tune: {tune}")
    print(f"  Chains: {chains}")
    print(f"  Target accept: {target_accept}")
    
    try:
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True
            )
        
        # Check for divergences
        n_divergences = int(idata.sample_stats['diverging'].sum().values)
        if n_divergences > 0:
            print(f"\n⚠️  WARNING: {n_divergences} divergences detected")
            print("   Consider increasing target_accept to 0.99")
        
        return idata
        
    except Exception as e:
        print(f"\n⚠️  NUTS failed: {e}")
        print("   Falling back to ADVI...")
        
        with model:
            approx = pm.fit(n=50000, method='advi')
            idata = approx.sample(draws=draws * chains, return_inferencedata=True)
        
        print("✓ ADVI completed (approximate inference)")
        return idata


def check_convergence(idata: az.InferenceData, rhat_threshold: float = 1.01, ess_threshold: int = 400) -> bool:
    """
    Check convergence using R-hat and ESS.
    
    Returns True if all parameters converged.
    """
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    summary = az.summary(idata, kind='diagnostics')
    
    # R-hat check
    max_rhat = summary['r_hat'].max()
    rhat_ok = max_rhat < rhat_threshold
    
    print(f"\nR-hat:")
    print(f"  Max: {max_rhat:.4f}")
    print(f"  Threshold: {rhat_threshold}")
    print(f"  Status: {'✓ PASS' if rhat_ok else '✗ FAIL'}")
    
    # ESS check
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()
    ess_ok = (min_ess_bulk > ess_threshold) and (min_ess_tail > ess_threshold)
    
    print(f"\nESS:")
    print(f"  Min bulk: {min_ess_bulk:.0f}")
    print(f"  Min tail: {min_ess_tail:.0f}")
    print(f"  Threshold: {ess_threshold}")
    print(f"  Status: {'✓ PASS' if ess_ok else '✗ FAIL'}")
    
    # Divergences
    if 'diverging' in idata.sample_stats:
        n_divergences = int(idata.sample_stats['diverging'].sum().values)
        div_ok = n_divergences < 10
        
        print(f"\nDivergences:")
        print(f"  Count: {n_divergences}")
        print(f"  Status: {'✓ PASS' if div_ok else '✗ FAIL'}")
    else:
        div_ok = True
    
    # Overall
    converged = rhat_ok and ess_ok and div_ok
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")
    print(f"{'='*60}")
    
    return converged


def posterior_predictive_check(idata: az.InferenceData, df: pd.DataFrame) -> Dict:
    """
    Generate posterior predictive samples and compare to observed.
    Uses the trace data to generate predictions without rebuilding model.
    
    Returns statistics for all props.
    """
    print("\n  Generating posterior predictive samples...")
    
    # Get predictions using the existing predict function
    # Extract coords from trace
    post = idata.posterior
    
    # Build coords from trace dims
    coords = {
        'position': list(post.coords['position'].values),
        'opponent': list(post.coords['opponent'].values),
        'feature': list(post.coords['feature'].values),
        'obs': np.arange(len(df))
    }
    
    # Generate predictions
    preds = predict_all_props(idata, df, coords, n_samples=1000)
    
    # Extract posterior predictive samples by sampling from Poisson
    goals_pp_samples = []
    shots_pp_samples = []
    cards_pp_samples = []
    
    for i in range(1000):
        goals_pp_samples.append(np.random.poisson(preds['goals']['lambda_samples'][i]))
        shots_pp_samples.append(np.random.poisson(preds['shots']['lambda_samples'][i]))
        cards_pp_samples.append(np.random.poisson(preds['cards']['lambda_samples'][i]))
    
    goals_pp = np.array(goals_pp_samples).flatten()
    shots_pp = np.array(shots_pp_samples).flatten()
    cards_pp = np.array(cards_pp_samples).flatten()
    
    # Observed means
    goals_obs = df['goals'].mean()
    shots_obs = df['shots_on_target'].mean()
    cards_obs = df['cards_total'].mean()
    
    return {
        # Goals
        'goals_obs_mean': goals_obs,
        'goals_pp_mean': goals_pp.mean(),
        'goals_pp_ci': (np.percentile(goals_pp, 2.5), np.percentile(goals_pp, 97.5)),
        'goals_pp_samples': goals_pp,
        
        # Shots
        'shots_obs_mean': shots_obs,
        'shots_pp_mean': shots_pp.mean(),
        'shots_pp_ci': (np.percentile(shots_pp, 2.5), np.percentile(shots_pp, 97.5)),
        'shots_pp_samples': shots_pp,
        
        # Cards
        'cards_obs_mean': cards_obs,
        'cards_pp_mean': cards_pp.mean(),
        'cards_pp_ci': (np.percentile(cards_pp, 2.5), np.percentile(cards_pp, 97.5)),
        'cards_pp_samples': cards_pp,
    }


def predict_all_props(
    idata: az.InferenceData,
    df: pd.DataFrame,
    coords: Dict,
    n_samples: int = 1000
) -> Dict:
    """
    Generate predictions for all 3 props on new data.
    
    Returns dict with:
    - lambda_samples: (n_samples, n_obs) array
    - lambda_mean: (n_obs,) array
    - prob_atleast_1: (n_obs,) array - P(y >= 1)
    """
    if len(df) == 0:
        raise ValueError("Cannot predict on empty dataframe")
    
    # Prepare inputs
    targets, indices, new_coords, X = _prep_multitask_inputs(df)
    
    pos_idx = indices['position']
    opp_idx = indices['opponent']
    n_obs = len(df)
    
    # Validate indices against coords
    if np.max(pos_idx) >= len(coords['position']):
        raise ValueError(f"Position index {np.max(pos_idx)} out of range for coords with {len(coords['position'])} positions")
    if np.max(opp_idx) >= len(coords['opponent']):
        raise ValueError(f"Opponent index {np.max(opp_idx)} out of range for coords with {len(coords['opponent'])} opponents")
    
    # Extract posterior samples
    post = idata.posterior
    
    # Calculate total samples available
    n_chains = post.dims['chain']
    n_draws = post.dims['draw']
    n_total = n_chains * n_draws
    
    if n_samples > n_total:
        print(f"  Warning: Requested {n_samples} samples but only {n_total} available. Using {n_total}.")
        n_samples = n_total
    
    # Sample indices (reproducible)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, size=n_samples, replace=False)
    
    # Helper to extract and flatten samples
    def _get_samples(var_name):
        arr = post[var_name].values
        # Flatten chain and draw dimensions
        flat = arr.reshape(-1, *arr.shape[2:])
        return flat[sample_idx]
    
    predictions = {}
    
    # ========================================
    # GOALS PREDICTIONS
    # ========================================
    try:
        alpha_goals = _get_samples('alpha_goals_position')  # (n_samples, n_positions)
        gamma_goals = _get_samples('gamma_goals_opponent')  # (n_samples, n_opponents)
        beta_goals = _get_samples('beta_goals')             # (n_samples, n_features)
        
        # Validate shapes
        assert alpha_goals.shape == (n_samples, len(coords['position'])), f"alpha_goals shape mismatch: {alpha_goals.shape}"
        assert beta_goals.shape == (n_samples, len(coords['feature'])), f"beta_goals shape mismatch: {beta_goals.shape}"
        
        lambda_goals = np.zeros((n_samples, n_obs))
        for i in range(n_samples):
            log_lam = (
                alpha_goals[i, pos_idx] +
                gamma_goals[i, opp_idx] +
                X @ beta_goals[i]
            )
            log_lam = np.clip(log_lam, -6.0, 1.0)
            lambda_goals[i] = np.exp(log_lam)
        
        predictions['goals'] = {
            'lambda_samples': lambda_goals,
            'lambda_mean': lambda_goals.mean(axis=0),
            'prob_atleast_1': 1 - np.exp(-lambda_goals.mean(axis=0))
        }
    except Exception as e:
        raise RuntimeError(f"Failed to predict goals: {e}")
    
    # ========================================
    # SHOTS PREDICTIONS
    # ========================================
    try:
        alpha_shots = _get_samples('alpha_shots_position')
        gamma_shots = _get_samples('gamma_shots_opponent')
        beta_shots = _get_samples('beta_shots')
        
        lambda_shots = np.zeros((n_samples, n_obs))
        for i in range(n_samples):
            log_lam = (
                alpha_shots[i, pos_idx] +
                gamma_shots[i, opp_idx] +
                X @ beta_shots[i]
            )
            log_lam = np.clip(log_lam, -6.0, 2.0)
            lambda_shots[i] = np.exp(log_lam)
        
        predictions['shots'] = {
            'lambda_samples': lambda_shots,
            'lambda_mean': lambda_shots.mean(axis=0),
            'prob_atleast_1': 1 - np.exp(-lambda_shots.mean(axis=0))
        }
    except Exception as e:
        raise RuntimeError(f"Failed to predict shots: {e}")
    
    # ========================================
    # CARDS PREDICTIONS
    # ========================================
    try:
        alpha_cards = _get_samples('alpha_cards_position')
        gamma_cards = _get_samples('gamma_cards_opponent')
        beta_cards = _get_samples('beta_cards')
        
        lambda_cards = np.zeros((n_samples, n_obs))
        for i in range(n_samples):
            log_lam = (
                alpha_cards[i, pos_idx] +
                gamma_cards[i, opp_idx] +
                X @ beta_cards[i]
            )
            log_lam = np.clip(log_lam, -6.0, 0.0)
            lambda_cards[i] = np.exp(log_lam)
        
        predictions['cards'] = {
            'lambda_samples': lambda_cards,
            'lambda_mean': lambda_cards.mean(axis=0),
            'prob_atleast_1': 1 - np.exp(-lambda_cards.mean(axis=0))
        }
    except Exception as e:
        raise RuntimeError(f"Failed to predict cards: {e}")
    
    return predictions


def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute calibration metrics (ECE, Brier, MAE).
    
    Args:
        y_true: Binary labels (0 or 1)
        y_pred: Predicted probabilities
        n_bins: Number of bins for ECE
    
    Returns:
        Dict with 'ece', 'brier', 'mae', 'bins'
    """
    # Input validation
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays provided")
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # Convert to numpy arrays and validate
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for valid binary labels
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0 or 1)")
    
    # Check for valid probabilities
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("y_pred must be probabilities in [0, 1]")
    
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains non-finite values")
    
    from sklearn.metrics import brier_score_loss
    
    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bins_info = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Handle edge case: last bin should be inclusive on upper bound
        if bin_upper == 1.0:
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        else:
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bins_info.append({
                'bin_center': (bin_lower + bin_upper) / 2,
                'pred_prob': avg_confidence_in_bin,
                'true_freq': accuracy_in_bin,
                'count': in_bin.sum()
            })
    
    # Brier score
    brier = brier_score_loss(y_true, y_pred)
    
    # MAE
    mae = np.abs(y_true - y_pred).mean()
    
    return {
        'ece': ece,
        'brier': brier,
        'mae': mae,
        'bins': bins_info
    }


def plot_calibration_curves(
    test_df: pd.DataFrame,
    test_preds: Dict,
    calib_results: Dict,
    save_path: str = None
):
    """
    Plot calibration curves for all 3 props.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    props = ['goals', 'shots', 'cards']
    titles = ['Goals', 'Shots on Target', 'Cards']
    
    for idx, (prop, title) in enumerate(zip(props, titles)):
        ax = axes[idx]
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
        
        # Model calibration
        bins = calib_results[prop]['bins']
        
        if len(bins) > 0:
            pred_probs = [b['pred_prob'] for b in bins]
            true_freqs = [b['true_freq'] for b in bins]
            
            ax.plot(pred_probs, true_freqs, 'o-', markersize=8, linewidth=2, 
                   label='Model', color='tab:blue')
        else:
            # No bins - likely all predictions in one range
            ax.text(0.5, 0.5, 'Insufficient data\nfor calibration plot',
                   ha='center', va='center', fontsize=12)
        
        # Labels
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'{title}\nECE = {calib_results[prop]["ece"]:.4f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle('Calibration Curves - Multi-Task Model', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved calibration curves to {save_path}")
    else:
        plt.show()


def plot_posterior_distributions(idata: az.InferenceData, coords: Dict, save_path: str = None):
    """Plot posterior distributions - SIMPLE 4x3 grid."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Row 0: sigma_alpha and two hyperpriors
    ax = axes[0, 0]
    sigma_alpha = idata.posterior['sigma_alpha'].values.flatten()
    ax.hist(sigma_alpha, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(sigma_alpha.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'sigma_alpha: {sigma_alpha.mean():.3f}')
    
    ax = axes[0, 1]
    mu = idata.posterior['mu_alpha_goals'].values.flatten()
    ax.hist(mu, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(mu.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'mu_alpha_goals: {mu.mean():.3f}')
    
    ax = axes[0, 2]
    mu = idata.posterior['mu_alpha_shots'].values.flatten()
    ax.hist(mu, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(mu.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'mu_alpha_shots: {mu.mean():.3f}')
    
    # Rows 1-3: Feature coefficients for each prop
    feature_names = coords['feature'][:3]
    props = ['goals', 'shots', 'cards']
    
    for row, prop in enumerate(props, start=1):
        beta = idata.posterior[f'beta_{prop}'].values
        beta_flat = beta.reshape(-1, beta.shape[-1])
        
        for col, feat in enumerate(feature_names):
            ax = axes[row, col]
            beta_feat = beta_flat[:, col]
            
            ax.hist(beta_feat, bins=40, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.axvline(beta_feat.mean(), color='blue', linestyle='-', linewidth=2)
            
            mean_val = beta_feat.mean()
            ax.set_title(f'{prop}: {feat[:15]}\n{mean_val:.3f}')
    
    plt.suptitle('Posterior Distributions', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved to {save_path if save_path else 'display'}")


def save_model(
    idata: az.InferenceData,
    metadata: dict,
    model_path: str,
    trace_path: str
):
    """Save model artifacts with robust error handling."""
    model_path = Path(model_path).expanduser().resolve()
    trace_path = Path(trace_path).expanduser().resolve()
    
    # Create directories
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate metadata
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    
    # Save trace with fallback engines
    tmp_trace = trace_path.with_suffix(trace_path.suffix + ".tmp")
    wrote = False
    last_error = None
    
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            if eng is None:
                idata.to_netcdf(str(tmp_trace))
            else:
                idata.to_netcdf(str(tmp_trace), engine=eng)
            wrote = True
            break
        except Exception as e:
            last_error = e
            continue
    
    if not wrote:
        raise RuntimeError(f"Failed to write NetCDF at {trace_path}. Last error: {last_error}")
    
    # Atomic move
    tmp_trace.replace(trace_path)
    
    # Save metadata with backup
    tmp_pkl = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        with open(tmp_pkl, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_pkl.replace(model_path)
    except Exception as e:
        if tmp_pkl.exists():
            tmp_pkl.unlink()
        raise RuntimeError(f"Failed to save metadata: {e}")
    
    # Verify files were written
    if not trace_path.exists():
        raise RuntimeError(f"Trace file was not created: {trace_path}")
    if not model_path.exists():
        raise RuntimeError(f"Metadata file was not created: {model_path}")
    
    print(f"✓ Saved trace to {trace_path} ({trace_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"✓ Saved metadata to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")


def load_model(model_path: str, trace_path: str) -> Tuple[Dict, az.InferenceData]:
    """Load saved model with validation."""
    model_path = Path(model_path)
    trace_path = Path(trace_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {model_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    try:
        with open(model_path, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")
    
    try:
        idata = az.from_netcdf(str(trace_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load trace: {e}")
    
    return metadata, idata


if __name__ == '__main__':
    print("="*60)
    print("MULTI-TASK BAYESIAN MODEL")
    print("="*60)
    print("\nImport this module in Jupyter notebook")
    print("See notebooks/exploration/03_train_multitask.ipynb")
    print("\nKey functions:")
    print("  - load_data()")
    print("  - build_multitask_model(df)")
    print("  - prior_predictive_check(model)")
    print("  - fit_model(model)")
    print("  - check_convergence(idata)")
    print("  - predict_all_props(idata, df, coords)")
    print("  - evaluate_calibration(y_true, y_pred)")
    print("  - save_model(idata, metadata, ...)")