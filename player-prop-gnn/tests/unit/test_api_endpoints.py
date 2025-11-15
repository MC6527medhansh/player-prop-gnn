"""
Unit Tests for API Endpoints
Phase 4: FastAPI Development

Tests FastAPI endpoints:
- GET /health
- POST /predict/player

Following Google AI guidelines:
- Mock external dependencies (model, Redis)
- Test all code paths (success, failure, edge cases)
- Test error handling
- Test caching logic
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from src.api.main import app

# Note: This would normally import from src.api.main
# For testing purposes, we'll mock the dependencies


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_model():
    """Mock BayesianPredictor for testing."""
    model = Mock()
    model.n_samples = 1000
    model.feature_names = [
        'goals_rolling_5',
        'shots_on_target_rolling_5',
        'opponent_strength',
        'days_since_last_match',
        'was_home'
    ]
    
    # Mock prediction result
    model.predict_player.return_value = {
        'goals': {
            'mean': 0.15,
            'median': 0.0,
            'std': 0.42,
            'ci_low': 0.0,
            'ci_high': 1.0,
            'probability': {'0': 0.87, '1': 0.11, '2+': 0.02}
        },
        'shots': {
            'mean': 0.32,
            'median': 0.0,
            'std': 0.65,
            'ci_low': 0.0,
            'ci_high': 2.0,
            'probability': {'0': 0.73, '1': 0.18, '2+': 0.09}
        },
        'cards': {
            'mean': 0.08,
            'median': 0.0,
            'std': 0.28,
            'ci_low': 0.0,
            'ci_high': 1.0,
            'probability': {'0': 0.93, '1': 0.06, '2+': 0.01}
        },
        'metadata': {
            'player_id': 123,
            'opponent_id': 45,
            'inference_time_ms': 85
        }
    }
    
    return model


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_client = Mock()
    redis_client.ping.return_value = True
    redis_client.get.return_value = None  # Cache miss by default
    redis_client.setex.return_value = True
    return redis_client


@pytest.fixture
def valid_request():
    """Valid prediction request for testing."""
    return {
        "player_id": 123,
        "opponent_id": 45,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.2,
            "shots_on_target_rolling_5": 0.8,
            "opponent_strength": 0.6,
            "days_since_last_match": 5.0,
            "was_home": 1.0
        }
    }
  
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

# ============================================================================
# TEST HEALTH ENDPOINT
# ============================================================================

@patch('src.api.main.MODEL', None)
@patch('src.api.main.REDIS_CLIENT', None)
def test_health_check_model_not_loaded(client):
    """Test health check returns 503 when model not loaded."""
    response = client.get("/health")
    
    assert response.status_code == 503
    data = response.json()
    assert data['status'] == 'unhealthy'
    assert data['model_loaded'] is False
    assert 'Model not loaded' in data.get('errors', [])


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_health_check_model_loaded_redis_down(mock_model, client):
    """Test health check returns 200 when model loaded but Redis down."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['redis_connected'] is False
    assert 'Redis not connected' in data.get('warnings', [])


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_health_check_all_healthy(mock_redis, mock_model, client):
    """Test health check returns 200 when everything healthy."""
    mock_redis.ping.return_value = True
    
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['redis_connected'] is True
    assert 'warnings' not in data
    assert 'errors' not in data


# ============================================================================
# TEST PREDICTION ENDPOINT - SUCCESS CASES
# ============================================================================

@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_predict_player_success_no_cache(mock_model, client, valid_request):
    """Test prediction succeeds without caching (Redis down)."""
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert data['player_id'] == 123
    assert data['opponent_id'] == 45
    assert data['position'] == 'Forward'
    assert data['was_home'] is True
    assert data['cached'] is False
    
    # Verify predictions present
    assert 'goals' in data['predictions']
    assert 'shots' in data['predictions']
    assert 'cards' in data['predictions']
    
    # Verify model was called
    mock_model.predict_player.assert_called_once()


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_predict_player_cache_miss(mock_redis, mock_model, client, valid_request):
    """Test prediction with cache miss - calls model and stores in cache."""
    mock_redis.get.return_value = None  # Cache miss
    
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data['cached'] is False
    
    # Verify cache was checked
    mock_redis.get.assert_called_once()
    
    # Verify model was called
    mock_model.predict_player.assert_called_once()
    
    # Verify result was stored in cache
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[1] == 300  # 5-minute TTL


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_predict_player_cache_hit(mock_redis, mock_model, client, valid_request):
    """Test prediction with cache hit - doesn't call model."""
    # Mock cached response
    cached_response = {
        "player_id": 123,
        "opponent_id": 45,
        "position": "Forward",
        "was_home": True,
        "predictions": {
            "goals": {
                "mean": 0.15,
                "median": 0.0,
                "std": 0.42,
                "ci_low": 0.0,
                "ci_high": 1.0,
                "probability": {"0": 0.87, "1": 0.11, "2+": 0.02}
            },
            "shots": {
                "mean": 0.32,
                "median": 0.0,
                "std": 0.65,
                "ci_low": 0.0,
                "ci_high": 2.0,
                "probability": {"0": 0.73, "1": 0.18, "2+": 0.09}
            },
            "cards": {
                "mean": 0.08,
                "median": 0.0,
                "std": 0.28,
                "ci_low": 0.0,
                "ci_high": 1.0,
                "probability": {"0": 0.93, "1": 0.06, "2+": 0.01}
            }
        },
        "cached": True,
        "inference_time_ms": 5,
        "model_version": "v1.0",
        "timestamp": "2024-11-14T10:30:00Z"
    }
    
    mock_redis.get.return_value = json.dumps(cached_response)
    
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 200
    
    # Verify cache was checked
    mock_redis.get.assert_called_once()
    
    # Verify model was NOT called
    mock_model.predict_player.assert_not_called()


# ============================================================================
# TEST PREDICTION ENDPOINT - ERROR CASES
# ============================================================================

@patch('src.api.main.MODEL', None)
@patch('src.api.main.REDIS_CLIENT', None)
def test_predict_player_model_not_loaded(client, valid_request):
    """Test prediction returns 503 when model not loaded."""
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 503
    assert 'Model not loaded' in response.json()['detail']


def test_predict_player_invalid_player_id(client):
    """Test prediction returns 400 for negative player_id."""
    invalid_request = {
        "player_id": -1,
        "opponent_id": 45,
        "position": "Forward",
        "was_home": True,
        "features": {}
    }
    
    response = client.post("/predict/player", json=invalid_request)
    
    assert response.status_code == 422  # Pydantic validation error


def test_predict_player_invalid_position(client):
    """Test prediction returns 400 for invalid position."""
    invalid_request = {
        "player_id": 123,
        "opponent_id": 45,
        "position": "InvalidPosition",
        "was_home": True,
        "features": {}
    }
    
    response = client.post("/predict/player", json=invalid_request)
    
    assert response.status_code == 422  # Pydantic validation error


def test_predict_player_missing_required_field(client):
    """Test prediction returns 400 when required field missing."""
    invalid_request = {
        "player_id": 123,
        # Missing opponent_id
        "position": "Forward",
        "was_home": True,
        "features": {}
    }
    
    response = client.post("/predict/player", json=invalid_request)
    
    assert response.status_code == 422


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_predict_player_model_prediction_fails(mock_model, client, valid_request):
    """Test prediction returns 500 when model prediction fails."""
    mock_model.predict_player.side_effect = Exception("Model error")
    
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 500
    assert 'Prediction failed' in response.json()['detail']


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_predict_player_model_validation_error(mock_model, client, valid_request):
    """Test prediction returns 400 when model rejects input."""
    mock_model.predict_player.side_effect = ValueError("Invalid input")
    
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 400
    assert 'Invalid input' in response.json()['detail']


# ============================================================================
# TEST CACHING LOGIC
# ============================================================================

@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_cache_key_format(mock_redis, mock_model, client, valid_request):
    """Test that cache key has correct format."""
    mock_redis.get.return_value = None
    
    response = client.post("/predict/player", json=valid_request)
    
    # Extract cache key from get() call
    cache_key = mock_redis.get.call_args[0][0]
    
    # Verify format: pred:tier1:{player_id}:{opponent_id}:{was_home}:v1.0
    expected_key = "pred:tier1:123:45:True:v1.0"
    assert cache_key == expected_key


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_cache_graceful_degradation_read_failure(mock_redis, mock_model, client, valid_request):
    """Test that cache read failure doesn't break prediction."""
    mock_redis.get.side_effect = Exception("Redis error")
    
    response = client.post("/predict/player", json=valid_request)
    
    # Should still succeed
    assert response.status_code == 200
    assert response.json()['cached'] is False
    
    # Model should still be called
    mock_model.predict_player.assert_called_once()


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT')
def test_cache_graceful_degradation_write_failure(mock_redis, mock_model, client, valid_request):
    """Test that cache write failure doesn't break prediction."""
    mock_redis.get.return_value = None
    mock_redis.setex.side_effect = Exception("Redis error")
    
    response = client.post("/predict/player", json=valid_request)
    
    # Should still succeed
    assert response.status_code == 200
    
    # Model should be called
    mock_model.predict_player.assert_called_once()


# ============================================================================
# TEST RESPONSE VALIDATION
# ============================================================================

@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_response_contains_all_props(mock_model, client, valid_request):
    """Test that response contains predictions for all props."""
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 200
    data = response.json()
    
    predictions = data['predictions']
    assert 'goals' in predictions
    assert 'shots' in predictions
    assert 'cards' in predictions


@patch('src.api.main.MODEL')
@patch('src.api.main.REDIS_CLIENT', None)
def test_response_contains_metadata(mock_model, client, valid_request):
    """Test that response contains performance metadata."""
    response = client.post("/predict/player", json=valid_request)
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'inference_time_ms' in data
    assert 'model_version' in data
    assert 'timestamp' in data
    assert data['inference_time_ms'] >= 0