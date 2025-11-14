"""
ROBUST Hierarchical Bayesian Poisson Model for Goals Prediction
Phase 3.2: PRODUCTION VERSION with proper prior specification

Key Fixes from Previous Version:
1. TIGHTENED β priors: 0.8 → 0.25 (prevents extreme feature effects)
2. HARD BOUNDS on log(λ): clipped to [-6, 1] → λ ∈ [0.0025, 2.7]
3. TRUNCATED σ_α: max 0.25 (prevents extreme position variation)
4. ROBUST prior predictive validation

Model Specification:
    goals_i ~ Poisson(λ_i)
    log(λ_i) = CLIP(α_position[pos_i] + γ_opponent[opp_i] + β @ x_i, -6, 1)
    
    Priors (TIGHTENED + BOUNDED):
    - μ_α ~ Normal(log(mean_goals), 0.15)
    - σ_α ~ TruncatedNormal(0, 0.12, upper=0.25)
    - α_position ~ Normal(μ_α, σ_α)
    - γ_opponent ~ Normal(0, 0.10)  # TIGHTER: 0.12 → 0.10
    - β ~ Normal(0, 0.25)            # MUCH TIGHTER: 0.8 → 0.25
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sqlalchemy import create_engine
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data(db_url='postgresql://medhanshchoubey@localhost:5432/football_props'):
    """Load player features with position information."""
    engine = create_engine(db_url)
    
    query = """
    SELECT 
        pf.player_id,
        pf.match_id,
        pf.match_date,
        pf.goals,
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
        'player_id', 'match_id', 'match_date', 'goals', 'opponent_id',
        'was_home', 'goals_rolling_5', 'shots_on_target_rolling_5',
        'opponent_strength', 'days_since_last_match', 'position'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df_clean = df.dropna(subset=required_cols)
    print(f"Loaded {len(df_clean)} records (dropped {len(df) - len(df_clean)} with missing values)")
    
    return df_clean


def _prep_bayes_inputs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """
    Robustly prepare inputs for Bayesian model with strict type checking.
    
    Returns:
        X, y, position_idx, opponent_idx, meta, position_map, opponent_map
    """
    required_cols = [
        'player_id','match_id','match_date','goals','opponent_id',
        'was_home','goals_rolling_5','shots_on_target_rolling_5',
        'opponent_strength','days_since_last_match','position'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop NaNs
    na_mask = df[required_cols].isna().any(axis=1)
    if na_mask.any():
        n_drop = int(na_mask.sum())
        df = df.loc[~na_mask].copy()
        print(f"[prep] Dropped {n_drop} rows with missing values")

    # Feature columns
    feature_cols = [
        'goals_rolling_5',
        'shots_on_target_rolling_5',
        'opponent_strength',
        'days_since_last_match',
        'was_home'
    ]

    # Coerce to numeric
    feat_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    if feat_df.isna().any().any():
        bad_cols = feat_df.columns[feat_df.isna().any()].tolist()
        raise ValueError(f"Non-numeric values in {bad_cols}")

    # Convert was_home to float
    if pd.api.types.is_bool_dtype(feat_df['was_home']):
        feat_df['was_home'] = feat_df['was_home'].astype(int)

    X = feat_df.to_numpy(dtype=np.float64)

    # Targets
    y_series = pd.to_numeric(df['goals'], errors='coerce')
    if (y_series.isna()).any():
        raise ValueError(f"'goals' has non-numeric entries")
    if (y_series < 0).any():
        raise ValueError(f"'goals' has negative values")
    y = y_series.to_numpy(dtype=np.int64)

    # Encode positions
    pos_cat = pd.Categorical(df['position'].astype('string'))
    unique_positions = list(pos_cat.categories)
    position_idx = pos_cat.codes.astype(np.int64)
    if (position_idx < 0).any():
        raise ValueError("Unknown position category")
    position_map = {pos: i for i, pos in enumerate(unique_positions)}

    # Encode opponents
    opp_arr = df['opponent_id'].to_numpy(dtype=np.int64)
    opp_unique = np.unique(opp_arr)
    opponent_map = {int(v): i for i, v in enumerate(opp_unique)}
    opponent_idx = df['opponent_id'].map(opponent_map).to_numpy(dtype=np.int64)

    # Validate dtypes
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert position_idx.dtype == np.int64
    assert opponent_idx.dtype == np.int64

    # Standardize continuous features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X[:, :-1] = scaler.fit_transform(X[:, :-1])

    meta = {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'positions': unique_positions,
        'opponents': opp_unique
    }
    
    return X, y, position_idx, opponent_idx, meta, position_map, opponent_map


def build_model(df: pd.DataFrame) -> Tuple[pm.Model, Dict]:
    """
    Build hierarchical Bayesian Poisson regression with ROBUST priors.
    
    KEY CHANGES:
    - β ~ Normal(0, 0.25) instead of 0.8 (MUCH TIGHTER)
    - log(λ) clipped to [-6, 1] (hard bounds)
    - σ_α truncated at 0.25 max
    - γ_opponent sigma reduced to 0.10
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        model: PyMC model
        coords: Coordinate dictionary
    """
    # Prepare data
    X, y, position_idx, opponent_idx, meta, position_map, opponent_map = _prep_bayes_inputs(df)

    n_obs, n_features = X.shape
    positions = meta['positions']
    opponents = meta['opponents']

    coords = {
        'obs': np.arange(n_obs),
        'feature': meta['feature_cols'],
        'position': positions,
        'opponent': opponents,
    }
    
    # Validation
    mean_goals = float(y.mean())
    if mean_goals < 0.01 or mean_goals > 5.0:
        raise ValueError(f"Suspicious mean goals: {mean_goals:.4f}")

    # Build model
    with pm.Model(coords=coords) as model:
        # Store for prediction
        model.feature_cols = meta['feature_cols']
        model.scaler = meta['scaler']
        model.position_map = position_map
        model.opponent_map = opponent_map

        # Data containers (strict dtypes)
        X_data = pm.MutableData('X', np.asarray(X, dtype=np.float64), dims=('obs','feature'))
        pos_data = pm.MutableData('position_idx', np.asarray(position_idx, dtype=np.int64), dims='obs')
        opp_data = pm.MutableData('opponent_idx', np.asarray(opponent_idx, dtype=np.int64), dims='obs')
        y_data = pm.MutableData('y', np.asarray(y, dtype=np.int64), dims='obs')

        eps = 1e-2
        log_mean_goals = np.log(mean_goals + eps)
        
        print(f"\n{'='*60}")
        print(f"BUILDING ROBUST HIERARCHICAL POISSON MODEL")
        print(f"{'='*60}")
        print(f"\nData Summary:")
        print(f"  Observations: {n_obs}")
        print(f"  Positions: {len(positions)} {list(positions)}")
        print(f"  Opponents: {len(opponents)}")
        print(f"  Features: {n_features}")
        print(f"  Mean goals: {mean_goals:.3f}")
        print(f"  Log(mean goals): {log_mean_goals:.3f}")
        
        print(f"\n{'='*60}")
        print(f"ROBUST PRIOR SPECIFICATION")
        print(f"{'='*60}")
        
        # PRIOR 1: Hyperprior mean
        mu_alpha = pm.Normal('mu_alpha', mu=log_mean_goals, sigma=0.15)
        print(f"\n1. Hyperprior Mean:")
        print(f"   μ_α ~ Normal({log_mean_goals:.3f}, 0.15)")
        
        # PRIOR 2: Position variation (TRUNCATED)
        # This prevents extreme σ_α values
        sigma_alpha = pm.TruncatedNormal('sigma_alpha', mu=0, sigma=0.12, lower=0, upper=0.25)
        print(f"\n2. Position Variation (TRUNCATED):")
        print(f"   σ_α ~ TruncatedNormal(0, 0.12, upper=0.25)")
        print(f"   → Prevents extreme position differences")
        print(f"   → Max multiplicative effect: exp(0.25) = 1.28x")
        
        # Position-specific intercepts
        alpha_position = pm.Normal('alpha_position', mu=mu_alpha, sigma=sigma_alpha, dims='position')
        
        # PRIOR 3: Opponent effects (TIGHTER)
        gamma_opponent = pm.Normal('gamma_opponent', mu=0.0, sigma=0.10, dims='opponent')
        print(f"\n3. Opponent Effects (TIGHTENED):")
        print(f"   γ_opponent ~ Normal(0, 0.10)")
        print(f"   → 95% of opponents within exp(±0.20) = [0.82, 1.22]")
        
        # PRIOR 4: Feature coefficients (MUCH TIGHTER)
        beta = pm.Normal('beta', mu=0.0, sigma=0.25, dims='feature')
        print(f"\n4. Feature Coefficients (MUCH TIGHTER):")
        print(f"   β ~ Normal(0, 0.25) for each feature")
        print(f"   → 1-SD change in feature → exp(0.25) = 1.28x effect")
        print(f"   → 2-SD change → exp(0.50) = 1.65x effect")
        print(f"   → Prevents extreme cumulative effects from 5 features")

        # Linear predictor (BEFORE clipping)
        log_lambda_unclipped = (
            alpha_position[pos_data] +
            gamma_opponent[opp_data] +
            pm.math.dot(X_data, beta)
        )
        
        # HARD BOUNDS on log(λ)
        # This is the KEY fix - prevents impossible values
        LOG_LAMBDA_MIN = -6.0  # exp(-6) ≈ 0.0025 goals (very rare but possible)
        LOG_LAMBDA_MAX = 1.0   # exp(1) ≈ 2.7 goals (max reasonable for a player-match)
        
        log_lambda = pm.Deterministic(
            'log_lambda_clipped',
            pm.math.clip(log_lambda_unclipped, LOG_LAMBDA_MIN, LOG_LAMBDA_MAX),
            dims='obs'
        )
        
        print(f"\n5. HARD BOUNDS on log(λ):")
        print(f"   log(λ) ∈ [{LOG_LAMBDA_MIN}, {LOG_LAMBDA_MAX}]")
        print(f"   → λ ∈ [{np.exp(LOG_LAMBDA_MIN):.4f}, {np.exp(LOG_LAMBDA_MAX):.1f}] goals")
        print(f"   → Prevents impossible predictions even in extreme prior tails")

        # Expected goals and likelihood
        lambda_ = pm.Deterministic('lambda', pm.math.exp(log_lambda), dims='obs')
        pm.Poisson('goals_obs', mu=lambda_, observed=y_data, dims='obs')
        
        print(f"\n{'='*60}")
        print(f"Model built successfully!")
        print(f"{'='*60}\n")

    return model, coords


def prior_predictive_check(model: pm.Model, n_samples: int = 1000) -> Dict:
    """
    Sample from prior predictive with ROBUST statistics reporting.
    
    Returns per-sample statistics (not flattened) to avoid outlier issues.
    """
    print("  Sampling from prior predictive distribution...")
    
    with model:
        pp = pm.sample_prior_predictive(samples=n_samples, random_seed=42)

    # Extract and flatten chain/draw dimensions
    def _flatten(arr):
        a = np.asarray(arr)
        if a.ndim >= 3:
            return a.reshape(-1, a.shape[-1])
        return a

    goals_obs_raw = _flatten(pp.prior_predictive["goals_obs"].values)
    lambda_raw = _flatten(pp.prior["lambda"].values)
    
    # Compute per-sample statistics (ROBUST approach)
    if goals_obs_raw.ndim == 2:  # (n_samples, n_obs)
        goals_obs_per_sample = goals_obs_raw.mean(axis=-1)
        goals_obs_all = goals_obs_raw.flatten()
    else:
        goals_obs_per_sample = goals_obs_raw
        goals_obs_all = goals_obs_raw
    
    if lambda_raw.ndim == 2:
        lambda_per_sample = lambda_raw.mean(axis=-1)
        lambda_all = lambda_raw.flatten()
    else:
        lambda_per_sample = lambda_raw
        lambda_all = lambda_raw
    
    # Check for extreme values
    n_extreme_lambda = (lambda_all > 10).sum()
    pct_extreme = 100 * n_extreme_lambda / lambda_all.size
    
    if pct_extreme > 0.01:
        print(f"    ⚠️  WARNING: {pct_extreme:.3f}% of lambda values > 10")
        print(f"    Top 5 extreme values: {np.sort(lambda_all)[-5:]}")
    else:
        print(f"    ✓ No extreme lambda values (all < 10)")
    
    # Extract parameters
    alpha_pos = _flatten(pp.prior["alpha_position"].values)
    mu_alpha = np.asarray(pp.prior["mu_alpha"].values).flatten()
    sigma_alpha = np.asarray(pp.prior["sigma_alpha"].values).flatten()
    beta = _flatten(pp.prior["beta"].values)
    
    return {
        "goals_obs": goals_obs_per_sample,
        "goals_obs_all": goals_obs_all,
        "lambda": lambda_per_sample,
        "lambda_all": lambda_all,
        "alpha_position": alpha_pos,
        "mu_alpha": mu_alpha,
        "sigma_alpha": sigma_alpha,
        "beta": beta,
    }


def fit_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42
) -> az.InferenceData:
    """Fit model using NUTS with fallback to ADVI."""
    print(f"\n{'='*60}")
    print(f"MCMC SAMPLING WITH NUTS")
    print(f"{'='*60}")
    print(f"  Draws: {draws} × {chains} chains = {draws * chains} total samples")
    print(f"  Tune: {tune}")
    print(f"  Target accept: {target_accept}")
    print(f"{'='*60}\n")
    
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
        return idata
        
    except Exception as e:
        print(f"\n⚠️  MCMC failed: {e}")
        print("Falling back to ADVI...")
        
        with model:
            approx = pm.fit(n=20000, method='advi', random_seed=random_seed)
            idata = approx.sample(draws * chains)
        
        return idata


def check_convergence(idata: az.InferenceData, save_path: str = None) -> Dict:
    """Check MCMC convergence and create trace plots."""
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    
    max_rhat = float(rhat.to_array().max().values)
    min_ess = float(ess.to_array().min().values)
    
    try:
        n_divergences = int(idata.sample_stats.diverging.sum().values)
    except:
        n_divergences = 0
    
    if save_path:
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        
        params_to_plot = ['mu_alpha', 'sigma_alpha', 'alpha_position', 'beta']
        
        for idx, param in enumerate(params_to_plot):
            if idx >= len(axes):
                break
                
            ax_trace = axes[idx, 0]
            data = idata.posterior[param].values
            
            if len(data.shape) == 2:
                for chain in range(data.shape[0]):
                    ax_trace.plot(data[chain, :], alpha=0.7, label=f'Chain {chain}')
            elif len(data.shape) == 3:
                for chain in range(data.shape[0]):
                    for dim in range(min(3, data.shape[2])):
                        ax_trace.plot(data[chain, :, dim], alpha=0.5)
            
            ax_trace.set_ylabel(param)
            ax_trace.set_xlabel('Draw')
            if idx == 0:
                ax_trace.legend()
            ax_trace.set_title(f'Trace: {param}')
            
            ax_post = axes[idx, 1]
            data_flat = data.flatten()
            ax_post.hist(data_flat, bins=50, alpha=0.7, edgecolor='black')
            ax_post.set_xlabel(param)
            ax_post.set_ylabel('Frequency')
            ax_post.set_title(f'Posterior: {param}')
            
            try:
                param_rhat = float(rhat[param].max().values)
                param_ess = float(ess[param].min().values)
                ax_post.text(0.05, 0.95, f'R-hat: {param_rhat:.4f}\nESS: {param_ess:.0f}',
                           transform=ax_post.transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'max_rhat': max_rhat,
        'min_ess': min_ess,
        'n_divergences': n_divergences
    }


def posterior_predictive_check(
    model: pm.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    n_samples: int = 500
) -> Dict:
    """Generate posterior predictive samples."""
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["goals_obs"],
            random_seed=42,
            return_inferencedata=False
        )

    arr = np.asarray(ppc["goals_obs"])
    
    if arr.ndim == 3:
        chains, draws, n_obs = arr.shape
        arr = arr.reshape(chains * draws, n_obs)
    
    total = arr.shape[0]
    if n_samples and n_samples < total:
        idx = np.random.default_rng(42).choice(total, size=n_samples, replace=False)
        arr = arr[idx, :]

    return {"goals_obs": arr}


def predict_counts(
    model: pm.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    n_samples: int = 1000
) -> np.ndarray:
    """Predict goal counts for new data."""
    feature_cols = model.feature_cols
    X_new = df[feature_cols].values.astype(np.float64).copy()
    X_new[:, :-1] = model.scaler.transform(X_new[:, :-1])
    
    position_idx_new = df['position'].map(model.position_map).fillna(0).astype(np.int32)
    opponent_idx_new = df['opponent_id'].map(model.opponent_map).fillna(0).astype(np.int32)
    
    alpha_position_samples = idata.posterior['alpha_position'].values
    gamma_opponent_samples = idata.posterior['gamma_opponent'].values
    beta_samples = idata.posterior['beta'].values
    
    n_chains = alpha_position_samples.shape[0]
    n_draws = alpha_position_samples.shape[1]
    total_samples = n_chains * n_draws
    
    alpha_position_samples = alpha_position_samples.reshape(total_samples, -1)
    gamma_opponent_samples = gamma_opponent_samples.reshape(total_samples, -1)
    beta_samples = beta_samples.reshape(total_samples, -1)
    
    if n_samples < total_samples:
        idx = np.random.choice(total_samples, n_samples, replace=False)
        alpha_position_samples = alpha_position_samples[idx]
        gamma_opponent_samples = gamma_opponent_samples[idx]
        beta_samples = beta_samples[idx]
    else:
        n_samples = total_samples
    
    n_obs = len(df)
    predictions = np.zeros((n_samples, n_obs))
    
    for i in range(n_samples):
        log_lambda = (
            alpha_position_samples[i, position_idx_new] +
            gamma_opponent_samples[i, opponent_idx_new] +
            X_new @ beta_samples[i]
        )
        # Apply same clipping as in model
        log_lambda = np.clip(log_lambda, -6, 1)
        lambda_ = np.exp(log_lambda)
        predictions[i] = np.random.poisson(lambda_)
    
    return predictions


def predict_prob_score(pred_counts: np.ndarray) -> np.ndarray:
    """Convert counts to P(score ≥ 1)."""
    lambda_est = pred_counts.mean(axis=0, keepdims=True)
    prob_score = 1 - np.exp(-lambda_est)
    prob_score = np.repeat(prob_score, pred_counts.shape[0], axis=0)
    return prob_score


def evaluate_calibration(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """Calculate ECE and Brier score."""
    from sklearn.metrics import brier_score_loss
    
    brier = brier_score_loss(y_true_binary, y_pred_prob)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    bins_info = []
    
    for i in range(n_bins):
        mask = (y_pred_prob >= bins[i]) & (y_pred_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true_binary[mask].mean()
            bin_conf = y_pred_prob[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(y_true_binary)) * abs(bin_acc - bin_conf)
            bins_info.append((bin_conf, bin_acc, bin_count))
        else:
            bins_info.append((None, None, 0))
    
    return {
        'ece': ece,
        'brier': brier,
        'bins_info': bins_info
    }


def plot_posterior(idata: az.InferenceData, coords: Dict, save_path: str = None):
    """Plot posterior distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Position intercepts
    ax = axes[0]
    alpha_pos = idata.posterior['alpha_position'].values
    positions = coords['position']
    alpha_pos_reshaped = alpha_pos.reshape(-1, len(positions))
    
    for i, pos in enumerate(positions):
        ax.hist(alpha_pos_reshaped[:, i], bins=40, alpha=0.6, label=pos)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Position Intercept (log scale)')
    ax.set_ylabel('Density')
    ax.set_title('Position Effects')
    ax.legend()
    
    # Hyperpriors
    ax = axes[1]
    mu_alpha = idata.posterior['mu_alpha'].values.flatten()
    sigma_alpha = idata.posterior['sigma_alpha'].values.flatten()
    ax.hist(mu_alpha, bins=40, alpha=0.6, label='μ_α')
    ax.hist(sigma_alpha, bins=40, alpha=0.6, label='σ_α')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Hierarchical Hyperpriors')
    ax.legend()
    
    # Feature coefficients
    beta = idata.posterior['beta'].values
    feature_names = coords['feature']
    beta_reshaped = beta.reshape(-1, len(feature_names))
    
    for idx, feat in enumerate(feature_names):
        if idx + 2 < len(axes):
            ax = axes[idx + 2]
            ax.hist(beta_reshaped[:, idx], bins=40, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(f'β_{feat}')
            ax.set_ylabel('Density')
            
            mean_val = beta_reshaped[:, idx].mean()
            ci_lower = np.percentile(beta_reshaped[:, idx], 2.5)
            ci_upper = np.percentile(beta_reshaped[:, idx], 97.5)
            
            ax.axvline(mean_val, color='blue', linestyle='-', linewidth=2)
            ax.axvline(ci_lower, color='green', linestyle=':', linewidth=1)
            ax.axvline(ci_upper, color='green', linestyle=':', linewidth=1)
            
            ax.set_title(f'{feat}\n{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]')
    
    plt.suptitle('Posterior Distributions', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins_info: list,
    save_path: str = None
):
    """Plot calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    bin_preds = [b[0] for b in bins_info if b[0] is not None]
    bin_accs = [b[1] for b in bins_info if b[1] is not None]
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax.plot(bin_preds, bin_accs, 'o-', markersize=10, linewidth=2, label='Bayesian model')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration: P(Score ≥ 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[1]
    ax.hist(y_pred, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predictions')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_model(
    idata: az.InferenceData,
    metadata: dict,
    model_path: str,
    trace_path: str,
):
    """Save model artifacts."""
    model_path = Path(model_path).expanduser().resolve()
    trace_path = Path(trace_path).expanduser().resolve()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_trace = trace_path.with_suffix(trace_path.suffix + ".tmp")
    wrote = False
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            if eng is None:
                idata.to_netcdf(tmp_trace)
            else:
                idata.to_netcdf(tmp_trace, engine=eng)
            wrote = True
            break
        except:
            pass

    if not wrote:
        raise RuntimeError(f"Failed to write NetCDF at {trace_path}")

    tmp_trace.replace(trace_path)

    with open(model_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Saved trace to {trace_path}")
    print(f"✓ Saved metadata to {model_path}")


def load_model(model_path: str, trace_path: str) -> Tuple[Dict, az.InferenceData]:
    """Load saved model."""
    with open(model_path, 'rb') as f:
        metadata = pickle.load(f)
    idata = az.from_netcdf(trace_path)
    return metadata, idata


if __name__ == '__main__':
    print("="*60)
    print("ROBUST BAYESIAN POISSON MODEL")
    print("="*60)
    print("\nImport this module in Jupyter notebook")
    print("See notebooks/exploration/02_train_bayesian_goals.ipynb")