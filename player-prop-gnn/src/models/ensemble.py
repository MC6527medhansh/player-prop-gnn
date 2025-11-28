"""
Ensemble Predictor for Player Props.
Phase 8.1: Integration & Routing Logic.

Design:
- Acts as a Facade over Tier 1 (Bayesian) and Tier 2 (GNN).
- Implements 'Cold Start' logic for new players.
- Implements 'Circuit Breaker' pattern for GNN failures.
- Fuses predictions using weighted averages.
"""

import logging
import time
from typing import Dict, Optional, Any
import numpy as np

# Type hints for dependencies (to be injected)
# We avoid direct imports at module level to prevent circular dependencies
# if the project structure shifts.
from src.models.inference import BayesianPredictor

# Placeholder for the GNN Predictor type hint (will be implemented in next step)
# from src.models.gnn_inference import GNNPredictor 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Intelligent routing and ensemble logic for combining:
    - Tier 1: Bayesian Multi-Task Model (Calibration Anchor)
    - Tier 2: GNN (Correlation/Context Expert)
    """

    def __init__(
        self,
        bayesian_predictor: BayesianPredictor,
        gnn_predictor: Any, # Typed as Any until GNNPredictor is formally defined
        tier1_weight: float = 0.6,
        tier2_weight: float = 0.4
    ):
        """
        Initialize the Ensemble.

        Args:
            bayesian_predictor: Initialized Tier 1 predictor.
            gnn_predictor: Initialized Tier 2 predictor.
            tier1_weight: Weight for Bayesian model (default 0.6 for safety).
            tier2_weight: Weight for GNN model (default 0.4).
        """
        self.tier1 = bayesian_predictor
        self.tier2 = gnn_predictor
        self.w1 = tier1_weight
        self.w2 = tier2_weight
        
        # Validate weights sum to 1.0 (approx)
        if abs((self.w1 + self.w2) - 1.0) > 0.001:
            logger.warning(f"Ensemble weights {self.w1}/{self.w2} do not sum to 1.0")

        logger.info(f"Ensemble initialized with Weights(Tier1={self.w1}, Tier2={self.w2})")

    def predict_player(
        self,
        player_id: int,
        opponent_id: int,
        position: str,
        was_home: bool,
        features: Dict[str, float],
        games_played: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Route and combine predictions.

        Args:
            player_id: Database ID of player.
            opponent_id: Database ID of opponent.
            position: Player position.
            was_home: Home/Away boolean.
            features: Feature dict for Tier 1.
            games_played: (Optional) Count of previous games for 'Cold Start' logic.
                          If None, defaults to established player logic.

        Returns:
            Unified prediction dictionary.
        """
        start_time = time.time()
        
        # 1. Get Tier 1 Prediction (The Anchor)
        # This MUST succeed. If Tier 1 fails, we propagate the error as the system is broken.
        try:
            t1_pred = self.tier1.predict_player(
                player_id=player_id,
                opponent_id=opponent_id,
                position=position,
                was_home=was_home,
                features=features
            )
        except Exception as e:
            logger.error(f"Tier 1 Critical Failure for player {player_id}: {e}")
            raise e

        # 2. Decision Logic: Should we use GNN?
        use_gnn = True
        skip_reason = ""

        # Check Cold Start
        if games_played is not None and games_played < 5:
            use_gnn = False
            skip_reason = "insufficient_history"
        
        # Check GNN Availability
        if self.tier2 is None:
            use_gnn = False
            skip_reason = "gnn_not_loaded"

        t2_pred = None
        
        # 3. Get Tier 2 Prediction (The Expert)
        if use_gnn:
            try:
                # Assuming GNN interface matches Tier 1 for simplicity, 
                # or takes specific graph args. We pass standard identifiers.
                t2_pred = self.tier2.predict_player(
                    player_id=player_id,
                    opponent_id=opponent_id,
                    position=position,
                    was_home=was_home
                )
            except Exception as e:
                # Circuit Breaker: Log error but fall back to Tier 1
                logger.error(f"Tier 2 (GNN) Runtime Failure for player {player_id}: {e}")
                use_gnn = False
                skip_reason = f"gnn_error: {str(e)}"

        # 4. Construct Response
        inference_time = (time.time() - start_time) * 1000.0

        if not use_gnn or t2_pred is None:
            # Return Tier 1 only
            t1_pred['metadata']['model_mode'] = 'tier1_only'
            t1_pred['metadata']['ensemble_reason'] = skip_reason
            t1_pred['metadata']['total_inference_time_ms'] = inference_time
            return t1_pred

        # 5. Fuse Predictions (Ensemble)
        ensemble_result = self._fuse_predictions(t1_pred, t2_pred)
        
        ensemble_result['metadata'] = {
            'player_id': player_id,
            'model_mode': 'ensemble_weighted',
            'weights': {'tier1': self.w1, 'tier2': self.w2},
            'tier1_time_ms': t1_pred['metadata'].get('inference_time_ms'),
            'total_inference_time_ms': inference_time,
            # Pass through Tier 1 metadata
            'position': position,
            'was_home': was_home
        }
        
        return ensemble_result

    def _fuse_predictions(self, t1: Dict, t2: Dict) -> Dict:
        """
        Combine Tier 1 and Tier 2 outputs using weighted averaging.
        
        Expects both dicts to have keys: 'goals', 'shots', 'cards'
        And subkeys: 'mean', 'probability'
        """
        fused = {}
        props = ['goals', 'shots', 'cards']

        for prop in props:
            if prop not in t1 or prop not in t2:
                continue

            p1 = t1[prop]
            p2 = t2[prop]

            # Fuse Means
            fused_mean = (p1['mean'] * self.w1) + (p2['mean'] * self.w2)
            
            # Fuse Probabilities (Dicts)
            fused_probs = {}
            # Union of keys (e.g. '0', '1', '2+')
            all_outcomes = set(p1['probability'].keys()) | set(p2['probability'].keys())
            
            for outcome in all_outcomes:
                val1 = p1['probability'].get(outcome, 0.0)
                val2 = p2['probability'].get(outcome, 0.0)
                fused_probs[outcome] = (val1 * self.w1) + (val2 * self.w2)

            # Fuse Standard Deviation (Approximation for Mixture)
            # Var(mix) ≈ w1*Var1 + w2*Var2 + w1*w2*(mean1 - mean2)^2
            var1 = p1['std']**2
            var2 = p2['std']**2
            mean_diff_sq = (p1['mean'] - p2['mean'])**2
            
            fused_var = (self.w1 * var1) + (self.w2 * var2) + \
                        (self.w1 * self.w2 * mean_diff_sq)
            fused_std = np.sqrt(fused_var)

            fused[prop] = {
                'mean': float(fused_mean),
                'std': float(fused_std),
                'probability': {k: float(v) for k, v in fused_probs.items()},
                # Note: Exact CI fusion is complex. We approximate or 
                # fallback to Tier 1's bounds shifted by the mean difference? 
                # Ideally, we'd resample, but that's slow.
                # For now, we omit explicit CI or re-calculate from fused mean/std 
                # assuming normality (imperfect but fast).
                'median': (p1['median'] * self.w1) + (p2['median'] * self.w2)
            }

        return fused