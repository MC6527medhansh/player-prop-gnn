"""
Validation script for bayesian_multitask.py
Tests all critical functions with synthetic data to catch errors before use.

Run this BEFORE running the full notebook to verify everything works.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.bayesian_multitask import (
    _prep_multitask_inputs,
    evaluate_calibration,
)

def test_timestamp_comparison():
    """Test that pd.Timestamp comparisons work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Timestamp Comparison")
    print("="*60)
    
    # Create test dataframe with different date types
    dates = pd.date_range('2018-06-01', periods=10, freq='D')
    df = pd.DataFrame({
        'match_date': dates,
        'value': range(10)
    })
    
    # Test conversion
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Test comparison
    split_date = pd.Timestamp('2018-06-05')
    train = df[df['match_date'] < split_date]
    test = df[df['match_date'] >= split_date]
    
    assert len(train) == 4, f"Expected 4 train records, got {len(train)}"
    assert len(test) == 6, f"Expected 6 test records, got {len(test)}"
    
    print("✓ Timestamp comparisons work correctly")


def test_prep_inputs_edge_cases():
    """Test _prep_multitask_inputs with edge cases."""
    print("\n" + "="*60)
    print("TEST 2: Input Preparation Edge Cases")
    print("="*60)
    
    # Create valid test dataframe
    n = 100
    df = pd.DataFrame({
        'goals': np.random.poisson(0.1, n),
        'shots_on_target': np.random.poisson(0.4, n),
        'yellow_cards': np.random.binomial(1, 0.08, n),
        'red_cards': np.random.binomial(1, 0.006, n),
        'position': np.random.choice(['Forward', 'Midfielder', 'Defender'], n),
        'opponent_id': np.random.choice([1, 2, 3, 4], n),
        'goals_rolling_5': np.random.uniform(0, 0.3, n),
        'shots_on_target_rolling_5': np.random.uniform(0, 1.0, n),
        'opponent_strength': np.random.uniform(0, 1, n),
        'days_since_last_match': np.random.uniform(3, 10, n),
        'was_home': np.random.choice([True, False], n),
    })
    
    df['cards_total'] = df['yellow_cards'] + df['red_cards']
    
    # Test normal case
    targets, indices, coords, X = _prep_multitask_inputs(df)
    
    assert targets['goals'].dtype == np.int64, "Goals should be int64"
    assert targets['shots'].dtype == np.int64, "Shots should be int64"
    assert targets['cards'].dtype == np.int64, "Cards should be int64"
    assert X.dtype == np.float64, "Features should be float64"
    assert not np.any(np.isnan(X)), "Features should not contain NaN"
    assert not np.any(np.isinf(X)), "Features should not contain inf"
    
    print("✓ Input preparation works with valid data")
    
    # Test empty dataframe
    try:
        empty_df = pd.DataFrame()
        _prep_multitask_inputs(empty_df)
        assert False, "Should raise error on empty df"
    except ValueError as e:
        assert "Empty dataframe" in str(e)
        print("✓ Correctly rejects empty dataframe")
    
    # Test with negative values
    try:
        bad_df = df.copy()
        bad_df.loc[0, 'goals'] = -1
        _prep_multitask_inputs(bad_df)
        assert False, "Should raise error on negative values"
    except ValueError as e:
        assert "negative values" in str(e)
        print("✓ Correctly rejects negative values")
    
    # Test with missing columns
    try:
        bad_df = df.drop(columns=['goals'])
        _prep_multitask_inputs(bad_df)
        assert False, "Should raise error on missing columns"
    except (ValueError, KeyError):
        print("✓ Correctly rejects missing columns")


def test_calibration_edge_cases():
    """Test evaluate_calibration with edge cases."""
    print("\n" + "="*60)
    print("TEST 3: Calibration Evaluation Edge Cases")
    print("="*60)
    
    # Valid case
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.05, 0.95])
    
    result = evaluate_calibration(y_true, y_pred, n_bins=5)
    
    assert 0 <= result['ece'] <= 1, "ECE should be in [0, 1]"
    assert 0 <= result['brier'] <= 1, "Brier should be in [0, 1]"
    assert 0 <= result['mae'] <= 1, "MAE should be in [0, 1]"
    assert len(result['bins']) <= 5, "Should have at most n_bins bins"
    
    print("✓ Calibration works with valid data")
    
    # Test empty arrays
    try:
        evaluate_calibration(np.array([]), np.array([]))
        assert False, "Should reject empty arrays"
    except ValueError:
        print("✓ Correctly rejects empty arrays")
    
    # Test length mismatch
    try:
        evaluate_calibration(np.array([0, 1]), np.array([0.5]))
        assert False, "Should reject length mismatch"
    except ValueError:
        print("✓ Correctly rejects length mismatch")
    
    # Test invalid probabilities
    try:
        evaluate_calibration(np.array([0, 1]), np.array([0.5, 1.5]))
        assert False, "Should reject probabilities > 1"
    except ValueError:
        print("✓ Correctly rejects invalid probabilities")
    
    # Test non-binary labels
    try:
        evaluate_calibration(np.array([0, 2]), np.array([0.5, 0.5]))
        assert False, "Should reject non-binary labels"
    except ValueError:
        print("✓ Correctly rejects non-binary labels")


def test_standardization_with_zero_variance():
    """Test that zero-variance features are handled correctly."""
    print("\n" + "="*60)
    print("TEST 4: Zero-Variance Feature Handling")
    print("="*60)
    
    n = 50
    df = pd.DataFrame({
        'goals': np.random.poisson(0.1, n),
        'shots_on_target': np.random.poisson(0.4, n),
        'yellow_cards': np.zeros(n, dtype=int),  # All zeros
        'red_cards': np.zeros(n, dtype=int),
        'position': np.random.choice(['Forward', 'Midfielder'], n),
        'opponent_id': np.random.choice([1, 2], n),
        'goals_rolling_5': np.random.uniform(0, 0.3, n),
        'shots_on_target_rolling_5': np.random.uniform(0, 1.0, n),
        'opponent_strength': np.random.uniform(0, 1, n),
        'days_since_last_match': np.random.uniform(3, 10, n),
        'was_home': np.ones(n, dtype=bool),  # All True - zero variance
    })
    
    df['cards_total'] = df['yellow_cards'] + df['red_cards']
    
    # This should NOT crash despite zero variance in was_home
    targets, indices, coords, X = _prep_multitask_inputs(df)
    
    # Check that standardization didn't produce NaN
    assert not np.any(np.isnan(X)), "Standardization produced NaN"
    assert not np.any(np.isinf(X)), "Standardization produced inf"
    
    # Check was_home column (index 4) is handled
    was_home_col = X[:, 4]
    assert np.all(np.isfinite(was_home_col)), "was_home column has non-finite values"
    
    print("✓ Zero-variance features handled correctly")


def test_dtypes_preservation():
    """Test that data types are preserved correctly throughout."""
    print("\n" + "="*60)
    print("TEST 5: Data Type Preservation")
    print("="*60)
    
    n = 30
    df = pd.DataFrame({
        'goals': np.array([0, 1, 0, 2] * (n // 4 + 1))[:n],
        'shots_on_target': np.array([0, 1, 2, 3] * (n // 4 + 1))[:n],
        'yellow_cards': np.array([0, 1, 0, 0] * (n // 4 + 1))[:n],
        'red_cards': np.zeros(n, dtype=int),
        'position': ['Forward'] * n,
        'opponent_id': [1] * n,
        'goals_rolling_5': np.random.uniform(0, 0.3, n),
        'shots_on_target_rolling_5': np.random.uniform(0, 1.0, n),
        'opponent_strength': np.random.uniform(0, 1, n),
        'days_since_last_match': np.random.uniform(3, 10, n),
        'was_home': ([True, False] * (n // 2 + 1))[:n],  # FIX: Added parentheses
    })
    
    df['cards_total'] = df['yellow_cards'] + df['red_cards']
    
    targets, indices, coords, X = _prep_multitask_inputs(df)
    
    # Check all dtypes
    assert targets['goals'].dtype == np.int64, f"Goals dtype is {targets['goals'].dtype}"
    assert targets['shots'].dtype == np.int64, f"Shots dtype is {targets['shots'].dtype}"
    assert targets['cards'].dtype == np.int64, f"Cards dtype is {targets['cards'].dtype}"
    assert indices['position'].dtype == np.int64, f"Position idx dtype is {indices['position'].dtype}"
    assert indices['opponent'].dtype == np.int64, f"Opponent idx dtype is {indices['opponent'].dtype}"
    assert X.dtype == np.float64, f"Features dtype is {X.dtype}"
    
    # Check shapes
    assert len(targets['goals']) == n, "Goals length mismatch"
    assert X.shape == (n, 5), f"Feature matrix shape is {X.shape}, expected ({n}, 5)"
    
    print("✓ All data types preserved correctly")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("BAYESIAN MULTITASK MODEL - VALIDATION SUITE")
    print("="*60)
    print("\nTesting critical functions before use...")
    
    try:
        test_timestamp_comparison()
        test_prep_inputs_edge_cases()
        test_calibration_edge_cases()
        test_standardization_with_zero_variance()
        test_dtypes_preservation()
        
        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*60)
        print("\nThe code is ready to use. You can now run the full notebook.")
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nDO NOT proceed with the notebook until this is fixed.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)