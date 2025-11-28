"""
Fast Bayesian Inference for Player Props
Phase 3.5: Production-ready prediction class

Key Features:
1. Loads model and caches posterior samples once (avoid repeated IO)
2. Vectorized predictions (<100ms target)
3. Handles unknown players gracefully (return league averages)
4. Comprehensive input validation
5. Detailed error messages with solutions

Design Decisions:
- Cache samples as numpy arrays (faster than xarray)
- Vectorize all computations (no Python loops over samples)
- Pre-compute log transforms
- Return detailed uncertainty (CI, probabilities per outcome)

Usage:
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    result = predictor.predict_player(
        player_id=123,
        opponent_id=45,
        was_home=True,
        features={'goals_rolling_5': 0.2, ...}
    )
"""

import json
import hashlib
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd  # kept if needed elsewhere; harmless to keep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when model operations attempted before loading."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class BayesianPredictor:
    """
    Fast inference with cached posterior samples.

    Loads model once, caches posterior samples in memory,
    then provides fast predictions (<100ms) for any player.

    Attributes:
        model_path: Path to pickled metadata
        trace_path: Path to NetCDF trace
        n_samples: Number of posterior samples to cache
        metadata: Model metadata (coords, version, etc.)
        cached_samples: Dict of numpy arrays (posterior samples)
        feature_names: List of required feature names
    """

    # Optional version tag for deterministic seeds
    _SEED_VERSION_TAG = "v1.0"

    def __init__(
        self,
        model_path: str,
        trace_path: str,
        n_samples: int = 1000
    ):
        """
        Initialize predictor and load model.

        Args:
            model_path: Path to .pkl metadata file
            trace_path: Path to .nc trace file
            n_samples: Number of posterior samples to cache (default 1000)

        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If model loading fails
            ValueError: If n_samples invalid
        """
        # Validate inputs
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be positive integer, got {n_samples}")

        model_path = Path(model_path).expanduser().resolve()
        trace_path = Path(trace_path).expanduser().resolve()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {model_path}\n"
                f"Solution: Ensure model trained with save_model()"
            )
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Trace file not found: {trace_path}\n"
                f"Solution: Ensure model trained with save_model()"
            )

        self.model_path = model_path
        self.trace_path = trace_path
        self.n_samples = n_samples

        # Load model
        logger.info(f"Loading model from {model_path}")
        self._load_model()
        logger.info("✓ Model loaded successfully")

    def _load_model(self):
        """Load metadata and cache posterior samples."""
        # Load metadata
        try:
            with open(self.model_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

        # Load trace
        try:
            idata = az.from_netcdf(str(self.trace_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load trace: {e}")

        # Extract coordinates
        self.coords = self.metadata['coords']
        self.feature_names = self.coords['feature']

        # Build index mappings
        self.position_to_idx = {pos: i for i, pos in enumerate(self.coords['position'])}
        self.opponent_to_idx = {opp: i for i, opp in enumerate(self.coords['opponent'])}

        # Cache posterior samples
        logger.info(f"Caching {self.n_samples} posterior samples...")
        self._cache_posterior_samples(idata)
        logger.info("✓ Cached samples ready")

    def _cache_posterior_samples(self, idata: az.InferenceData):
        """Extract and cache posterior samples as numpy arrays."""
        post = idata.posterior

        # Calculate total samples
        # Note: xarray's dims mapping may warn; access works in current versions.
        n_chains = post.dims['chain']
        n_draws = post.dims['draw']
        n_total = n_chains * n_draws

        if self.n_samples > n_total:
            logger.warning(
                f"Requested {self.n_samples} samples but only {n_total} available. Using {n_total}."
            )
            self.n_samples = n_total

        # Sample indices (reproducible selection of draws)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, size=self.n_samples, replace=False)

        # Helper to extract and flatten samples
        def _get_samples(var_name: str) -> np.ndarray:
            arr = post[var_name].values  # shape: (chain, draw, ...)
            flat = arr.reshape(-1, *arr.shape[2:])  # (chain*draw, ...)
            return flat[sample_idx]  # (n_samples, ...)

        # Cache all parameters
        self.cached_samples = {
            # Goals
            'alpha_goals': _get_samples('alpha_goals_position'),
            'gamma_goals': _get_samples('gamma_goals_opponent'),
            'beta_goals': _get_samples('beta_goals'),

            # Shots
            'alpha_shots': _get_samples('alpha_shots_position'),
            'gamma_shots': _get_samples('gamma_shots_opponent'),
            'beta_shots': _get_samples('beta_shots'),

            # Cards
            'alpha_cards': _get_samples('alpha_cards_position'),
            'gamma_cards': _get_samples('gamma_cards_opponent'),
            'beta_cards': _get_samples('beta_cards'),
        }

        # Validate shapes
        n_pos = len(self.coords['position'])
        n_opp = len(self.coords['opponent'])
        n_feat = len(self.coords['feature'])

        assert self.cached_samples['alpha_goals'].shape == (self.n_samples, n_pos)
        assert self.cached_samples['beta_goals'].shape == (self.n_samples, n_feat)

        logger.info(f"  Positions: {n_pos}, Opponents: {n_opp}, Features: {n_feat}")

    # --------------------------
    # Deterministic RNG helpers
    # --------------------------
    def _stable_seed(self, *parts) -> int:
        """
        Create a stable 32-bit seed from arbitrary parts.
        All parts are stringified and joined; MD5 digest -> first 8 hex chars.
        """
        s = "|".join(map(str, parts))
        h = hashlib.md5(s.encode()).hexdigest()
        return int(h[:8], 16)  # 32-bit int

    def _seeded_rng(self, *parts) -> np.random.RandomState:
        """RandomState seeded deterministically from parts."""
        return np.random.RandomState(self._stable_seed(*parts))

    # --------------------------
    # Public prediction methods
    # --------------------------
    def predict_player(
        self,
        player_id: int,
        opponent_id: int,
        position: str,
        was_home: bool,
        features: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        Predict all props for a single player.

        Args:
            player_id: Player database ID (for logging)
            opponent_id: Opponent database ID
            position: Player position (Forward/Midfielder/Defender/Goalkeeper)
            was_home: True if home game
            features: Dict of feature values
                Required keys: those listed in self.feature_names

        Returns:
            Dict with structure:
            {
                'goals': {...stats...},
                'shots': {...stats...},
                'cards': {...stats...},
                'metadata': {...}
            }

        Raises:
            ValueError: If inputs invalid
            PredictionError: If prediction fails
        """
        start_time = time.time()

        # ========================================
        # INPUT VALIDATION
        # ========================================

        # Validate player_id
        if not isinstance(player_id, (int, np.integer)):
            raise ValueError(f"player_id must be int, got {type(player_id)}")
        if player_id <= 0:
            raise ValueError(f"player_id must be positive, got {player_id}")

        # Validate opponent_id
        if not isinstance(opponent_id, (int, np.integer)):
            raise ValueError(f"opponent_id must be int, got {type(opponent_id)}")
        if opponent_id <= 0:
            raise ValueError(f"opponent_id must be positive, got {opponent_id}")

        # Validate position
        if not isinstance(position, str):
            raise ValueError(f"position must be string, got {type(position)}")
        if position not in self.position_to_idx:
            raise ValueError(
                f"Unknown position: {position}\n"
                f"Valid positions: {list(self.position_to_idx.keys())}\n"
                f"Solution: Ensure position is one of the training positions"
            )

        # Validate was_home
        if not isinstance(was_home, bool):
            raise ValueError(f"was_home must be bool, got {type(was_home)}")

        # Validate features
        if not isinstance(features, dict):
            raise ValueError(f"features must be dict, got {type(features)}")

        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}\n"
                f"Required features: {self.feature_names}\n"
                f"Provided features: {list(features.keys())}"
            )

        # Check for non-finite values
        for feat_name, feat_val in features.items():
            if not isinstance(feat_val, (int, float, np.integer, np.floating)):
                raise ValueError(f"Feature '{feat_name}' must be numeric, got {type(feat_val)}")
            if not np.isfinite(feat_val):
                raise ValueError(f"Feature '{feat_name}' is not finite: {feat_val}")

        # ========================================
        # PREPARE INDICES / FEATURES
        # ========================================

        pos_idx = self.position_to_idx[position]

        # Handle unknown opponent (use average opponent effect)
        if opponent_id not in self.opponent_to_idx:
            logger.warning(f"Unknown opponent {opponent_id}, using average opponent effect")
            opp_idx = None  # Will use mean of opponent effects
        else:
            opp_idx = self.opponent_to_idx[opponent_id]

        # Prepare feature vector (order matches training coords)
        X = np.array([features[feat] for feat in self.feature_names], dtype=np.float64)

        # ========================================
        # DETERMINISTIC RNG (per unique input state)
        # ========================================
        # IMPORTANT: player_id intentionally excluded so identical inputs
        # across different player_ids yield identical predictions.
        features_json = json.dumps(
            {k: float(features[k]) for k in self.feature_names},
            sort_keys=True
        )
        base_seed_parts = (opponent_id, position, was_home, features_json, self._SEED_VERSION_TAG)

        rng_goals = self._seeded_rng(*base_seed_parts, "goals")
        rng_shots = self._seeded_rng(*base_seed_parts, "shots")
        rng_cards = self._seeded_rng(*base_seed_parts, "cards")

        # ========================================
        # PREDICT GOALS
        # ========================================
        try:
            lambda_goals = self._predict_poisson(
                alpha=self.cached_samples['alpha_goals'][:, pos_idx],
                gamma=self.cached_samples['gamma_goals'][:, opp_idx]
                if opp_idx is not None
                else self.cached_samples['gamma_goals'].mean(axis=1),
                beta=self.cached_samples['beta_goals'],
                X=X,
                clip_min=-6,
                clip_max=1
            )
            goals_samples = rng_goals.poisson(lambda_goals)
            goals_stats = self._compute_stats(goals_samples)
        except Exception as e:
            raise PredictionError(f"Failed to predict goals: {e}")

        # ========================================
        # PREDICT SHOTS
        # ========================================
        try:
            lambda_shots = self._predict_poisson(
                alpha=self.cached_samples['alpha_shots'][:, pos_idx],
                gamma=self.cached_samples['gamma_shots'][:, opp_idx]
                if opp_idx is not None
                else self.cached_samples['gamma_shots'].mean(axis=1),
                beta=self.cached_samples['beta_shots'],
                X=X,
                clip_min=-6,
                clip_max=2
            )
            shots_samples = rng_shots.poisson(lambda_shots)
            shots_stats = self._compute_stats(shots_samples)
        except Exception as e:
            raise PredictionError(f"Failed to predict shots: {e}")

        # ========================================
        # PREDICT CARDS
        # ========================================
        try:
            lambda_cards = self._predict_poisson(
                alpha=self.cached_samples['alpha_cards'][:, pos_idx],
                gamma=self.cached_samples['gamma_cards'][:, opp_idx]
                if opp_idx is not None
                else self.cached_samples['gamma_cards'].mean(axis=1),
                beta=self.cached_samples['beta_cards'],
                X=X,
                clip_min=-6,
                clip_max=0
            )
            cards_samples = rng_cards.poisson(lambda_cards)
            cards_stats = self._compute_stats(cards_samples)
        except Exception as e:
            raise PredictionError(f"Failed to predict cards: {e}")

        # ========================================
        # ASSEMBLE RESULT
        # ========================================

        inference_time_ms = (time.time() - start_time) * 1000.0

        result = {
            'goals': goals_stats,
            'shots': shots_stats,
            'cards': cards_stats,
            'metadata': {
                'player_id': int(player_id),
                'opponent_id': int(opponent_id),
                'position': position,
                'was_home': was_home,
                'inference_time_ms': float(inference_time_ms),
                'n_samples': self.n_samples,
                'unknown_opponent': opp_idx is None
            }
        }

        return result

    def _predict_poisson(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        X: np.ndarray,
        clip_min: float,
        clip_max: float
    ) -> np.ndarray:
        """
        Vectorized Poisson rate prediction.

        Args:
            alpha: Position effects (n_samples,)
            gamma: Opponent effects (n_samples,)
            beta: Feature coefficients (n_samples, n_features)
            X: Feature vector (n_features,)
            clip_min: Min value for log(lambda)
            clip_max: Max value for log(lambda)

        Returns:
            lambda values (n_samples,)
        """
        # Compute log(lambda) = alpha + gamma + beta @ X
        log_lambda = alpha + gamma + (beta @ X)

        # Clip to prevent extreme values
        log_lambda = np.clip(log_lambda, clip_min, clip_max)

        # Transform to rate
        lambda_val = np.exp(log_lambda)

        return lambda_val

    def _compute_stats(self, samples: np.ndarray) -> Dict:
        """
        Compute summary statistics from posterior predictive samples.

        Args:
            samples: Posterior predictive samples (n_samples,)

        Returns:
            Dict with mean, median, std, CI, and probabilities
        """
        return {
            'mean': float(samples.mean()),
            'median': float(np.median(samples)),
            'std': float(samples.std()),
            'ci_low': float(np.percentile(samples, 2.5)),
            'ci_high': float(np.percentile(samples, 97.5)),
            'probability': {
                '0': float((samples == 0).mean()),
                '1': float((samples == 1).mean()),
                '2+': float((samples >= 2).mean()),
            }
        }

    def predict_batch(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """
        Batch prediction for multiple players.

        Args:
            requests: List of dicts with keys:
                - player_id: int
                - opponent_id: int
                - position: str
                - was_home: bool
                - features: dict

        Returns:
            List of prediction dicts

        Raises:
            ValueError: If requests invalid
        """
        if not isinstance(requests, list):
            raise ValueError(f"requests must be list, got {type(requests)}")

        if len(requests) == 0:
            raise ValueError("requests list cannot be empty")

        results = []
        for i, req in enumerate(requests):
            try:
                result = self.predict_player(**req)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for request {i}: {e}")
                # Include error in results
                results.append({
                    'error': str(e),
                    'request_index': i,
                    'request': req
                })

        return results


if __name__ == '__main__':
    print("=" * 60)
    print("BAYESIAN PREDICTOR")
    print("=" * 60)
    print("\nUsage:")
    print("  predictor = BayesianPredictor('model.pkl', 'trace.nc')")
    print("  result = predictor.predict_player(...)")
    print("\nSee tests/unit/test_inference.py for examples")
