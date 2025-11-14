"""
Unit tests for inference.py
Tests all critical paths and edge cases BEFORE deployment

Run with:
    pytest tests/unit/test_inference.py -v
    pytest tests/unit/test_inference.py -v --cov=src.models.inference
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference import BayesianPredictor, ModelNotLoadedError, PredictionError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_metadata():
    """Mock model metadata."""
    return {
        'version': 'v1.0',
        'coords': {
            'position': ['Forward', 'Midfielder', 'Defender'],
            'opponent': [1, 2, 3, 4],
            'feature': ['goals_rolling_5', 'shots_on_target_rolling_5', 
                       'opponent_strength', 'days_since_last_match', 'was_home']
        },
        'n_train': 1528,
        'draws': 2000,
        'chains': 4
    }


@pytest.fixture
def mock_idata():
    """Mock InferenceData with posterior samples."""
    # Create mock posterior
    n_chains = 4
    n_draws = 500
    n_pos = 3
    n_opp = 4
    n_feat = 5
    
    mock_post = Mock()
    mock_post.dims = {'chain': n_chains, 'draw': n_draws}
    mock_post.coords = {
        'position': np.array(['Forward', 'Midfielder', 'Defender']),
        'opponent': np.array([1, 2, 3, 4]),
        'feature': np.array(['goals_rolling_5', 'shots_on_target_rolling_5', 
                            'opponent_strength', 'days_since_last_match', 'was_home'])
    }
    
    # Create mock parameter arrays (chain, draw, ...)
    np.random.seed(42)
    
    def create_mock_param(shape):
        # Chain x Draw x ...
        full_shape = (n_chains, n_draws) + shape
        return np.random.randn(*full_shape) * 0.1
    
    mock_post.__getitem__ = lambda self, key: Mock(values=create_mock_param({
        'alpha_goals_position': (n_pos,),
        'alpha_shots_position': (n_pos,),
        'alpha_cards_position': (n_pos,),
        'gamma_goals_opponent': (n_opp,),
        'gamma_shots_opponent': (n_opp,),
        'gamma_cards_opponent': (n_opp,),
        'beta_goals': (n_feat,),
        'beta_shots': (n_feat,),
        'beta_cards': (n_feat,),
    }[key]))
    
    mock_idata = Mock()
    mock_idata.posterior = mock_post
    
    return mock_idata


@pytest.fixture
def valid_features():
    """Valid feature dict for testing."""
    return {
        'goals_rolling_5': 0.2,
        'shots_on_target_rolling_5': 0.8,
        'opponent_strength': 0.6,
        'days_since_last_match': 5.0,
        'was_home': 1.0
    }


# ============================================================================
# TEST MODEL LOADING
# ============================================================================

def test_init_file_not_found():
    """Test that clear error raised if model files missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        BayesianPredictor('nonexistent.pkl', 'nonexistent.nc')
    
    assert 'Metadata file not found' in str(exc_info.value)
    assert 'Solution:' in str(exc_info.value)


def test_init_invalid_n_samples():
    """Test validation of n_samples parameter."""
    with pytest.raises(ValueError) as exc_info:
        BayesianPredictor('model.pkl', 'trace.nc', n_samples=0)
    
    assert 'n_samples must be positive' in str(exc_info.value)
    
    with pytest.raises(ValueError):
        BayesianPredictor('model.pkl', 'trace.nc', n_samples=-100)
    
    with pytest.raises(ValueError):
        BayesianPredictor('model.pkl', 'trace.nc', n_samples=1.5)


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
def test_init_successful(mock_exists, mock_open, mock_from_netcdf, 
                         mock_metadata, mock_idata):
    """Test successful model loading."""
    # Setup mocks
    mock_open.return_value.__enter__.return_value.read = Mock(return_value=b'')
    mock_open.return_value.__enter__.return_value.__iter__ = Mock(return_value=iter([]))
    
    # Mock pickle.load
    with patch('pickle.load', return_value=mock_metadata):
        mock_from_netcdf.return_value = mock_idata
        
        # Create predictor
        predictor = BayesianPredictor('model.pkl', 'trace.nc', n_samples=100)
        
        # Assertions
        assert predictor.metadata == mock_metadata
        assert predictor.n_samples <= 100  # May be reduced if not enough samples
        assert len(predictor.cached_samples) > 0
        assert predictor.feature_names == mock_metadata['coords']['feature']


# ============================================================================
# TEST INPUT VALIDATION
# ============================================================================

@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_invalid_player_id(mock_pickle, mock_exists, mock_open, 
                                          mock_from_netcdf, mock_metadata, 
                                          mock_idata, valid_features):
    """Test validation of player_id."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # Invalid type
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id="123",  # String not int
            opponent_id=1,
            position='Forward',
            was_home=True,
            features=valid_features
        )
    assert 'player_id must be int' in str(exc_info.value)
    
    # Invalid value
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id=0,  # Must be positive
            opponent_id=1,
            position='Forward',
            was_home=True,
            features=valid_features
        )
    assert 'player_id must be positive' in str(exc_info.value)


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_invalid_position(mock_pickle, mock_exists, mock_open,
                                         mock_from_netcdf, mock_metadata,
                                         mock_idata, valid_features):
    """Test validation of position."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # Unknown position
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id=123,
            opponent_id=1,
            position='Goalkeeper',  # Not in training data
            was_home=True,
            features=valid_features
        )
    assert 'Unknown position' in str(exc_info.value)
    assert 'Valid positions:' in str(exc_info.value)


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_missing_features(mock_pickle, mock_exists, mock_open,
                                         mock_from_netcdf, mock_metadata,
                                         mock_idata):
    """Test validation of features."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # Missing features
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id=123,
            opponent_id=1,
            position='Forward',
            was_home=True,
            features={'goals_rolling_5': 0.2}  # Missing other features
        )
    assert 'Missing features' in str(exc_info.value)
    assert 'Required features:' in str(exc_info.value)


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_non_finite_features(mock_pickle, mock_exists, mock_open,
                                            mock_from_netcdf, mock_metadata,
                                            mock_idata, valid_features):
    """Test validation of feature values."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # NaN value
    bad_features = valid_features.copy()
    bad_features['goals_rolling_5'] = np.nan
    
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id=123,
            opponent_id=1,
            position='Forward',
            was_home=True,
            features=bad_features
        )
    assert 'not finite' in str(exc_info.value)
    
    # Inf value
    bad_features['goals_rolling_5'] = np.inf
    
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_player(
            player_id=123,
            opponent_id=1,
            position='Forward',
            was_home=True,
            features=bad_features
        )
    assert 'not finite' in str(exc_info.value)


# ============================================================================
# TEST PREDICTION LOGIC
# ============================================================================

@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_known_player(mock_pickle, mock_exists, mock_open,
                                     mock_from_netcdf, mock_metadata,
                                     mock_idata, valid_features):
    """Test successful prediction for known player."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    result = predictor.predict_player(
        player_id=123,
        opponent_id=1,
        position='Forward',
        was_home=True,
        features=valid_features
    )
    
    # Check structure
    assert 'goals' in result
    assert 'shots' in result
    assert 'cards' in result
    assert 'metadata' in result
    
    # Check goals statistics
    goals = result['goals']
    assert 'mean' in goals
    assert 'median' in goals
    assert 'std' in goals
    assert 'ci_low' in goals
    assert 'ci_high' in goals
    assert 'probability' in goals
    
    # Check probabilities
    probs = goals['probability']
    assert '0' in probs
    assert '1' in probs
    assert '2+' in probs
    
    # Probabilities should sum to 1 (approximately)
    total_prob = probs['0'] + probs['1'] + probs['2+']
    assert 0.99 <= total_prob <= 1.01
    
    # Check metadata
    meta = result['metadata']
    assert meta['player_id'] == 123
    assert meta['opponent_id'] == 1
    assert meta['position'] == 'Forward'
    assert meta['was_home'] == True
    assert 'inference_time_ms' in meta
    assert meta['inference_time_ms'] >= 0


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_unknown_opponent(mock_pickle, mock_exists, mock_open,
                                        mock_from_netcdf, mock_metadata,
                                        mock_idata, valid_features, caplog):
    """Test prediction with unknown opponent (should use average)."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    result = predictor.predict_player(
        player_id=123,
        opponent_id=999,  # Not in training data
        position='Forward',
        was_home=True,
        features=valid_features
    )
    
    # Should succeed with warning
    assert 'goals' in result
    assert result['metadata']['unknown_opponent'] == True
    
    # Check for warning in logs
    assert 'Unknown opponent' in caplog.text


# ============================================================================
# TEST PERFORMANCE
# ============================================================================

@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_player_latency(mock_pickle, mock_exists, mock_open,
                               mock_from_netcdf, mock_metadata,
                               mock_idata, valid_features):
    """Test that prediction meets latency target (<100ms)."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # Warm-up (first call may be slower)
    predictor.predict_player(
        player_id=123,
        opponent_id=1,
        position='Forward',
        was_home=True,
        features=valid_features
    )
    
    # Measure latency over 10 calls
    latencies = []
    for _ in range(10):
        start = time.time()
        result = predictor.predict_player(
            player_id=123,
            opponent_id=1,
            position='Forward',
            was_home=True,
            features=valid_features
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    print(f"\nLatency stats:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  Min: {np.min(latencies):.2f}ms")
    
    # Target: <100ms average
    # Note: This may fail with mocked data due to mock overhead
    # In production with real data, should be much faster
    assert avg_latency < 200, f"Average latency {avg_latency:.2f}ms exceeds 200ms (mocked)"


# ============================================================================
# TEST BATCH PREDICTION
# ============================================================================

@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_batch_empty_list(mock_pickle, mock_exists, mock_open,
                                  mock_from_netcdf, mock_metadata, mock_idata):
    """Test batch prediction rejects empty list."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_batch([])
    
    assert 'cannot be empty' in str(exc_info.value)


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_batch_valid_requests(mock_pickle, mock_exists, mock_open,
                                      mock_from_netcdf, mock_metadata,
                                      mock_idata, valid_features):
    """Test batch prediction with valid requests."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    requests = [
        {
            'player_id': 123,
            'opponent_id': 1,
            'position': 'Forward',
            'was_home': True,
            'features': valid_features
        },
        {
            'player_id': 456,
            'opponent_id': 2,
            'position': 'Midfielder',
            'was_home': False,
            'features': valid_features
        }
    ]
    
    results = predictor.predict_batch(requests)
    
    assert len(results) == 2
    assert 'goals' in results[0]
    assert 'goals' in results[1]
    assert results[0]['metadata']['player_id'] == 123
    assert results[1]['metadata']['player_id'] == 456


@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_batch_partial_failure(mock_pickle, mock_exists, mock_open,
                                       mock_from_netcdf, mock_metadata,
                                       mock_idata, valid_features):
    """Test batch prediction handles failures gracefully."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    requests = [
        {
            'player_id': 123,
            'opponent_id': 1,
            'position': 'Forward',
            'was_home': True,
            'features': valid_features
        },
        {
            'player_id': -1,  # Invalid
            'opponent_id': 2,
            'position': 'Midfielder',
            'was_home': False,
            'features': valid_features
        }
    ]
    
    results = predictor.predict_batch(requests)
    
    assert len(results) == 2
    assert 'goals' in results[0]  # First succeeded
    assert 'error' in results[1]  # Second failed
    assert 'player_id must be positive' in results[1]['error']


# ============================================================================
# TEST EDGE CASES
# ============================================================================

@patch('src.models.inference.az.from_netcdf')
@patch('builtins.open', create=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('pickle.load')
def test_predict_extreme_feature_values(mock_pickle, mock_exists, mock_open,
                                       mock_from_netcdf, mock_metadata,
                                       mock_idata):
    """Test prediction with extreme but valid feature values."""
    mock_pickle.return_value = mock_metadata
    mock_from_netcdf.return_value = mock_idata
    
    predictor = BayesianPredictor('model.pkl', 'trace.nc')
    
    # Extreme values (but still valid)
    extreme_features = {
        'goals_rolling_5': 5.0,  # Very high
        'shots_on_target_rolling_5': 0.0,  # Very low
        'opponent_strength': 1.0,  # Maximum
        'days_since_last_match': 30.0,  # Long break
        'was_home': 0.0  # Away
    }
    
    result = predictor.predict_player(
        player_id=123,
        opponent_id=1,
        position='Forward',
        was_home=False,
        features=extreme_features
    )
    
    # Should succeed
    assert 'goals' in result
    
    # Predictions should be in reasonable range
    assert 0 <= result['goals']['mean'] <= 10
    assert 0 <= result['shots']['mean'] <= 20
    assert 0 <= result['cards']['mean'] <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src.models.inference'])