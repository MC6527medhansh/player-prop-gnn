"""
Phase 7.2: Parlay Pricing Engine
Calculates 'True Odds' for correlated parlays using Gaussian Copula.

Inputs:
- Marginal Probabilities (from Tier 1 Bayesian Model)
- Correlation Matrix (from Tier 2 GNN, Calibrated)

Output:
- Joint Probability of multiple events occurring
"""
import numpy as np
import logging
import os
from scipy.stats import norm
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParlayPricer:
    """
    Prices parlays by combining marginal probabilities with correlation structure.
    Uses Gaussian Copula to model dependence while preserving marginals.
    """
    
    def __init__(self, correlation_matrix_path: str = 'models/gnn_correlation_matrix.npy'):
        # Map props to matrix indices (Must match training order: Goals, Assists, Shots, Cards)
        self.prop_map = {'goals': 0, 'assists': 1, 'shots': 2, 'cards': 3}
        
        # Load the safe, calibrated matrix
        if os.path.exists(correlation_matrix_path):
            self.corr_matrix = np.load(correlation_matrix_path)
            logger.info(f"Loaded correlation matrix shape: {self.corr_matrix.shape}")
        else:
            logger.warning(f"❌ Correlation matrix not found at {correlation_matrix_path}")
            logger.warning("Defaulting to Identity Matrix (Independence assumption).")
            self.corr_matrix = np.eye(4)

    def price_parlay(self, legs: List[Dict]) -> Dict:
        """
        Calculate the true price of a parlay.
        
        Args:
            legs: List of dicts, e.g.:
            [
                {'prop': 'goals', 'prob': 0.30}, 
                {'prop': 'shots', 'prob': 0.60}
            ]
            
        Returns:
            Dictionary containing:
            - independent_prob: P(A)*P(B)
            - correlated_prob: Copula result
            - correlation_factor: correlated / independent
            - fair_odds: Decimal odds (1/prob)
        """
        if not legs:
            return {'error': 'Empty parlay'}
            
        # 1. Extract Marginals and Indices
        marginals = []
        indices = []
        
        for leg in legs:
            prop = leg.get('prop')
            prob = leg.get('prob')
            
            if prop not in self.prop_map:
                logger.warning(f"Unknown prop '{prop}', skipping.")
                continue
                
            marginals.append(prob)
            indices.append(self.prop_map[prop])

        # If single leg or invalid props, return simple probability
        if len(marginals) < 2:
            prob = marginals[0] if marginals else 0
            return {
                'correlated_prob': prob, 
                'fair_odds': 1/prob if prob > 0 else 0,
                'note': 'Single leg or independence fallback'
            }

        # 2. Gaussian Copula Calculation
        try:
            # A. Convert Marginals (Uniform) -> Standard Normal (Z-scores)
            # We want P(X > threshold), so we look at the upper tail.
            # norm.ppf(1 - p) gives the z-score threshold.
            z_thresholds = [norm.ppf(1 - p) for p in marginals]
            
            # B. Extract Sub-Correlation Matrix for these specific props
            sub_corr = self.corr_matrix[np.ix_(indices, indices)]
            
            # C. Compute Joint Probability (Monte Carlo)
            # Analytical integration of MVN is complex/slow for >2 dims.
            # Monte Carlo is robust and fast enough for 4x4 matrices.
            n_samples = 50000 
            
            # Generate correlated samples (Mean = 0)
            mean = np.zeros(len(indices))
            samples = np.random.multivariate_normal(mean, sub_corr, size=n_samples)
            
            # Convert back to Uniform [0,1]
            uniform_samples = norm.cdf(samples)
            
            # Count successes
            # A "success" is when the simulated outcome > threshold probability
            success_flags = np.ones(n_samples, dtype=bool)
            for i, p in enumerate(marginals):
                threshold = 1.0 - p
                success_flags &= (uniform_samples[:, i] > threshold)
            
            joint_prob = np.mean(success_flags)
            
            # D. Baseline (Independence)
            indep_prob = np.prod(marginals)
            
            # Handle edge case where joint_prob is 0 (extremely unlikely events)
            joint_prob = max(joint_prob, 1e-6)
            
            return {
                'inputs': legs,
                'independent_prob': float(indep_prob),
                'correlated_prob': float(joint_prob),
                'correlation_factor': float(joint_prob / indep_prob) if indep_prob > 0 else 0,
                'fair_odds': 1.0 / joint_prob
            }

        except Exception as e:
            logger.error(f"Pricing failed: {e}")
            return {'error': str(e)}

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    print("Testing Parlay Pricing...")
    pricer = ParlayPricer()
    
    # Test 1: Positive Correlation (Goals + Shots)
    # Uses your calibrated matrix (~0.54 correlation)
    print("\n--- TEST CASE 1: Goals + Shots (Correlated) ---")
    legs_corr = [
        {'prop': 'goals', 'prob': 0.30},
        {'prop': 'shots', 'prob': 0.50}
    ]
    res1 = pricer.price_parlay(legs_corr)
    print(f"Independent Prob: {res1['independent_prob']:.4f}")
    print(f"Correlated Prob:  {res1['correlated_prob']:.4f}")
    print(f"Lift Factor:      {res1['correlation_factor']:.2f}x")
    
    if res1['correlation_factor'] > 1.05:
        print("✅ SUCCESS: Engine correctly prices positive correlation.")
    else:
        print("❌ FAILURE: Engine failed to detect positive correlation.")

    # Test 2: Negative/Weak Correlation (Goals + Cards)
    # Uses your calibrated matrix (~ -0.09 or similar weak correlation)
    print("\n--- TEST CASE 2: Goals + Cards (Weak/Negative) ---")
    legs_neg = [
        {'prop': 'goals', 'prob': 0.30},
        {'prop': 'cards', 'prob': 0.20}
    ]
    res2 = pricer.price_parlay(legs_neg)
    print(f"Independent Prob: {res2['independent_prob']:.4f}")
    print(f"Correlated Prob:  {res2['correlated_prob']:.4f}")
    print(f"Lift Factor:      {res2['correlation_factor']:.2f}x")
    
    if res2['correlation_factor'] <= 1.05:
        print("✅ SUCCESS: Engine respects negative/neutral correlation.")
    else:
        print("❌ FAILURE: Engine incorrectly boosted negative correlation.")