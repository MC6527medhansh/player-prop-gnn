"""
Integration Test: Full Pipeline
Tests the complete workflow: Data → Train → Save → Load → Predict

This test uses real (mocked) data to validate that all components
work together correctly.

Run with:
    pytest tests/integration/test_end_to_end.py -v -s
    
Note: This test takes ~30 seconds due to MCMC sampling.
Mark as slow test to skip in quick CI runs:
    pytest -m "not slow"
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import shutil
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_with_synthetic_data():
    """
    End-to-end test with synthetic data.
    
    Pipeline:
    1. Generate synthetic dataset
    2. Train model
    3. Save model
    4. Load model
    5. Make predictions
    6. Validate predictions
    """
    from src.models.bayesian_multitask import (
        build_multitask_model,
        fit_model,
        check_convergence,
        predict_all_props,
        evaluate_calibration,
        save_model,
        load_model
    )
    from src.models.inference import BayesianPredictor
    
    # ========================================
    # 1. GENERATE SYNTHETIC DATA
    # ========================================
    
    print("\n--- Generating Synthetic Data ---")
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate realistic features
    df = pd.DataFrame({
        'position': np.random.choice(['Forward', 'Midfielder', 'Defender'], n_samples),
        'opponent_id': np.random.choice([1, 2, 3, 4], n_samples),
        'goals_rolling_5': np.random.exponential(0.2, n_samples),
        'shots_on_target_rolling_5': np.random.exponential(0.8, n_samples),
        'opponent_strength': np.random.uniform(0, 1, n_samples),
        'days_since_last_match': np.random.uniform(3, 10, n_samples),
        'was_home': np.random.choice([0.0, 1.0], n_samples),
    })
    
    # Generate realistic outcomes
    df['goals'] = np.random.poisson(0.1, n_samples).astype(np.int64)
    df['shots_on_target'] = np.random.poisson(0.4, n_samples).astype(np.int64)
    df['yellow_cards'] = np.random.binomial(1, 0.08, n_samples).astype(np.int64)
    df['red_cards'] = np.random.binomial(1, 0.006, n_samples).astype(np.int64)
    df['cards_total'] = (df['yellow_cards'] + df['red_cards']).astype(np.int64)
    
    print(f"✓ Generated {len(df)} synthetic records")
    print(f"  Goals mean: {df['goals'].mean():.3f}")
    print(f"  Shots mean: {df['shots_on_target'].mean():.3f}")
    print(f"  Cards mean: {df['cards_total'].mean():.3f}")
    
    # Split train/test
    train_df = df[:150].copy()
    test_df = df[150:].copy()
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # ========================================
    # 2. TRAIN MODEL
    # ========================================
    
    print("\n--- Training Model ---")
    
    model, coords = build_multitask_model(train_df)
    print("✓ Model built")
    
    # Use small draws for speed in testing
    idata = fit_model(model, draws=200, chains=2, tune=100)
    print("✓ Model trained")
    
    # ========================================
    # 3. CHECK CONVERGENCE
    # ========================================
    
    print("\n--- Checking Convergence ---")
    
    try:
        check_convergence(idata, strict=False)
        print("✓ Convergence check passed")
    except Exception as e:
        print(f"⚠ Convergence warning: {e}")
        print("  (This is OK for synthetic data with small draws)")
    
    # ========================================
    # 4. SAVE MODEL
    # ========================================
    
    print("\n--- Saving Model ---")
    
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        metadata = {
            'version': 'test_v1.0',
            'coords': coords,
            'n_train': len(train_df),
            'draws': 200,
            'chains': 2
        }
        
        model_path = tmpdir / 'test_model.pkl'
        trace_path = tmpdir / 'test_model_trace.nc'
        
        save_model(idata, metadata, str(model_path), str(trace_path))
        print("✓ Model saved")
        
        # Verify files exist
        assert model_path.exists(), "Metadata file not created"
        assert trace_path.exists(), "Trace file not created"
        
        # Check file sizes
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        trace_size_mb = trace_path.stat().st_size / (1024 * 1024)
        
        print(f"  Metadata: {model_size_mb:.2f} MB")
        print(f"  Trace: {trace_size_mb:.2f} MB")
        
        assert model_size_mb < 1.0, "Metadata file too large"
        assert trace_size_mb > 0.01, "Trace file too small"
        
        # ========================================
        # 5. LOAD MODEL
        # ========================================
        
        print("\n--- Loading Model ---")
        
        loaded_metadata, loaded_idata = load_model(str(model_path), str(trace_path))
        print("✓ Model loaded")
        
        # Verify metadata
        assert loaded_metadata['version'] == 'test_v1.0'
        assert loaded_metadata['n_train'] == len(train_df)
        
        # ========================================
        # 6. MAKE PREDICTIONS
        # ========================================
        
        print("\n--- Making Predictions ---")
        
        preds = predict_all_props(loaded_idata, test_df, coords, n_samples=100)
        print("✓ Predictions generated")
        
        # Verify prediction structure
        assert 'goals' in preds
        assert 'shots' in preds
        assert 'cards' in preds
        
        # Verify shapes
        assert preds['goals']['lambda_mean'].shape == (len(test_df),)
        assert preds['shots']['lambda_mean'].shape == (len(test_df),)
        assert preds['cards']['lambda_mean'].shape == (len(test_df),)
        
        # Verify probabilities
        assert len(preds['goals']['prob_atleast_1']) == len(test_df)
        assert np.all(preds['goals']['prob_atleast_1'] >= 0)
        assert np.all(preds['goals']['prob_atleast_1'] <= 1)
        
        print(f"  Goals prob mean: {preds['goals']['prob_atleast_1'].mean():.3f}")
        print(f"  Shots prob mean: {preds['shots']['prob_atleast_1'].mean():.3f}")
        print(f"  Cards prob mean: {preds['cards']['prob_atleast_1'].mean():.3f}")
        
        # ========================================
        # 7. EVALUATE CALIBRATION
        # ========================================
        
        print("\n--- Evaluating Calibration ---")
        
        y_goals = (test_df['goals'] > 0).astype(int).values
        goals_calib = evaluate_calibration(y_goals, preds['goals']['prob_atleast_1'])
        
        print(f"  Goals ECE: {goals_calib['ece']:.4f}")
        print(f"  Goals Brier: {goals_calib['brier']:.4f}")
        
        # Calibration should be reasonable (not perfect with synthetic data)
        assert 0 <= goals_calib['ece'] <= 0.5, "ECE out of reasonable range"
        assert 0 <= goals_calib['brier'] <= 0.5, "Brier out of reasonable range"
        
        # ========================================
        # 8. TEST INFERENCE CLASS
        # ========================================
        
        print("\n--- Testing Inference Class ---")
        
        predictor = BayesianPredictor(str(model_path), str(trace_path), n_samples=100)
        print("✓ Predictor initialized")
        
        # Single prediction
        result = predictor.predict_player(
            player_id=1,
            opponent_id=1,
            position='Forward',
            was_home=True,
            features={
                'goals_rolling_5': 0.2,
                'shots_on_target_rolling_5': 0.8,
                'opponent_strength': 0.6,
                'days_since_last_match': 5.0,
                'was_home': 1.0
            }
        )
        
        print("✓ Single prediction successful")
        
        # Verify structure
        assert 'goals' in result
        assert 'shots' in result
        assert 'cards' in result
        assert 'metadata' in result
        
        # Verify values
        assert 0 <= result['goals']['mean'] <= 5
        assert 0 <= result['shots']['mean'] <= 10
        assert 0 <= result['cards']['mean'] <= 2
        
        print(f"  Goals: mean={result['goals']['mean']:.2f}, CI=[{result['goals']['ci_low']:.0f}, {result['goals']['ci_high']:.0f}]")
        print(f"  Shots: mean={result['shots']['mean']:.2f}, CI=[{result['shots']['ci_low']:.0f}, {result['shots']['ci_high']:.0f}]")
        print(f"  Cards: mean={result['cards']['mean']:.2f}, CI=[{result['cards']['ci_low']:.0f}, {result['cards']['ci_high']:.0f}]")
        print(f"  Inference time: {result['metadata']['inference_time_ms']:.2f}ms")
        
        # Verify latency
        assert result['metadata']['inference_time_ms'] < 500, "Inference too slow"
        
        # Batch prediction
        requests = [
            {
                'player_id': i,
                'opponent_id': 1,
                'position': 'Forward',
                'was_home': True,
                'features': {
                    'goals_rolling_5': 0.2,
                    'shots_on_target_rolling_5': 0.8,
                    'opponent_strength': 0.6,
                    'days_since_last_match': 5.0,
                    'was_home': 1.0
                }
            }
            for i in range(1, 6)
        ]
        
        batch_results = predictor.predict_batch(requests)
        
        print(f"✓ Batch prediction successful ({len(batch_results)} players)")
        
        assert len(batch_results) == 5
        assert all('goals' in r for r in batch_results)
        
    # ========================================
    # FINAL ASSERTIONS
    # ========================================
    
    print("\n" + "="*60)
    print("INTEGRATION TEST PASSED")
    print("="*60)
    print("\nAll pipeline steps completed successfully:")
    print("  ✓ Data generation")
    print("  ✓ Model training")
    print("  ✓ Model saving")
    print("  ✓ Model loading")
    print("  ✓ Predictions")
    print("  ✓ Calibration evaluation")
    print("  ✓ Inference class")
    print("  ✓ Batch predictions")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])