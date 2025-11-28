"""
Integration Tests for API Endpoints
Phase 4.5 - Comprehensive Testing

PURPOSE: Catch bugs in API request/response handling

Tests:
1. Valid requests return 200
2. Invalid requests return appropriate errors (400, 503, 500)
3. Response schema validation
4. Error messages are clear
"""
import pytest
import requests
from typing import Dict

API_URL = "http://localhost:8000"


# ============================================================================
# TEST 1: VALID REQUESTS
# ============================================================================

def test_health_endpoint():
    """Test /health endpoint returns 200 and correct schema."""
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200, f"Health check failed: {response.text}"
    
    data = response.json()
    
    # Required fields
    assert "status" in data
    assert "model_loaded" in data
    assert "redis_connected" in data
    assert "timestamp" in data
    assert "version" in data
    
    # Status should be 'healthy' if model loaded
    if data["model_loaded"]:
        assert data["status"] == "healthy"


def test_metrics_endpoint():
    """Test /metrics endpoint returns Prometheus format."""
    response = requests.get(f"{API_URL}/metrics")
    
    assert response.status_code == 200
    assert "text/plain" in response.headers["Content-Type"]
    
    # Check for expected metrics
    text = response.text
    assert "prop_api_requests_total" in text
    assert "prop_api_cache_hit_rate" in text
    assert "prop_api_model_loaded" in text


def test_valid_prediction_request():
    """Test valid prediction request returns 200."""
    payload = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    assert response.status_code == 200, f"Prediction failed: {response.text}"
    
    data = response.json()
    
    # Required fields in response
    assert "player_id" in data
    assert "predictions" in data
    assert "cached" in data
    assert "inference_time_ms" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Predictions should have all 3 props
    assert "goals" in data["predictions"]
    assert "shots" in data["predictions"]
    assert "cards" in data["predictions"]


# ============================================================================
# TEST 2: INVALID REQUESTS (400 Errors)
# ============================================================================

def test_negative_player_id_rejected():
    """Test that negative player_id returns 400."""
    payload = {
        "player_id": -5,  # Invalid
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    assert response.status_code == 422, (  # FastAPI returns 422 for validation errors
        f"Should reject negative player_id. Got {response.status_code}: {response.text}"
    )


def test_invalid_position_rejected():
    """Test that invalid position returns 400."""
    payload = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "InvalidPosition",  # Invalid
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    assert response.status_code == 422


def test_missing_features_rejected():
    """Test that missing features returns 400."""
    payload = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4
            # Missing other features
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    # Should fail (either 400 or 500 depending on implementation)
    assert response.status_code in [400, 422, 500]


def test_nan_feature_rejected():
    """Test that NaN features are rejected."""
    payload = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": float('nan'),  # Invalid
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    assert response.status_code in [400, 422]


# ============================================================================
# TEST 3: RESPONSE SCHEMA VALIDATION
# ============================================================================

def test_prediction_response_schema():
    """Test that prediction response has correct schema."""
    payload = {
        "player_id": 70001,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Check top-level schema
    required_fields = [
        "player_id", "opponent_id", "position", "was_home",
        "predictions", "cached", "inference_time_ms",
        "model_version", "timestamp"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    # Check prediction schema
    for prop in ["goals", "shots", "cards"]:
        assert prop in data["predictions"]
        
        pred = data["predictions"][prop]
        
        # Required fields in each prediction
        required_pred_fields = [
            "mean", "median", "std", "ci_low", "ci_high", "probability"
        ]
        
        for field in required_pred_fields:
            assert field in pred, f"Missing field in {prop} prediction: {field}"
        
        # Check probability has expected keys
        prob = pred["probability"]
        expected_outcomes = ["0", "1", "2+"]
        
        for outcome in expected_outcomes:
            assert outcome in prob, f"Missing probability for {prop} outcome '{outcome}'"


# ============================================================================
# TEST 4: ERROR MESSAGES
# ============================================================================

def test_error_messages_are_clear():
    """Test that error responses have helpful messages."""
    # Invalid position
    payload = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "InvalidPosition",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict/player", json=payload)
    
    assert response.status_code in [400, 422]
    
    data = response.json()
    
    # Should have detail field with error message
    assert "detail" in data
    
    # Error message should mention the problem
    detail_str = str(data["detail"]).lower()
    assert "position" in detail_str or "invalid" in detail_str, (
        f"Error message should mention 'position' or 'invalid'. Got: {data['detail']}"
    )


# ============================================================================
# TEST 5: REQUEST ID TRACKING
# ============================================================================

def test_request_id_in_response_header():
    """Test that responses include X-Request-ID header."""
    response = requests.get(f"{API_URL}/health")
    
    assert "X-Request-ID" in response.headers, (
        "Response should include X-Request-ID header for tracing"
    )
    
    request_id = response.headers["X-Request-ID"]
    
    # Should be valid UUID format
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    assert re.match(uuid_pattern, request_id), (
        f"Request ID should be valid UUID. Got: {request_id}"
    )


# ============================================================================
# TEST 6: PERFORMANCE
# ============================================================================

def test_prediction_latency_acceptable():
    """Test that predictions complete in reasonable time."""
    import time
    
    payload = {
        "player_id": 70002,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/predict/player", json=payload, timeout=5)
    duration_ms = (time.time() - start) * 1000
    
    assert response.status_code == 200
    
    # Should complete in < 1000ms (generous for first request)
    assert duration_ms < 1000, (
        f"Prediction took too long: {duration_ms:.0f}ms. "
        f"Should be < 1000ms."
    )
    
    print(f"\n✓ Prediction latency: {duration_ms:.0f}ms")


def test_cached_prediction_is_faster():
    """Test that cached predictions are faster than fresh ones."""
    import time
    
    payload = {
        "player_id": 70003,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    # First request (cache miss)
    start1 = time.time()
    response1 = requests.post(f"{API_URL}/predict/player", json=payload)
    duration1_ms = (time.time() - start1) * 1000
    
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Second request (should be cached)
    start2 = time.time()
    response2 = requests.post(f"{API_URL}/predict/player", json=payload)
    duration2_ms = (time.time() - start2) * 1000
    
    assert response2.status_code == 200
    data2 = response2.json()
    
    # If caching works, second should be faster
    if data2.get("cached"):
        assert duration2_ms < duration1_ms, (
            f"Cached request should be faster. "
            f"First: {duration1_ms:.0f}ms, Second: {duration2_ms:.0f}ms"
        )
        
        print(f"\n✓ Cache speedup: {duration1_ms:.0f}ms → {duration2_ms:.0f}ms ({duration1_ms/duration2_ms:.1f}x faster)")
    else:
        print(f"\n⚠ Caching not working (both requests uncached)")


# ============================================================================
# SUMMARY
# ============================================================================

def test_api_validation_summary():
    """Generate API validation summary report."""
    print("\n" + "="*70)
    print("API ENDPOINT VALIDATION REPORT")
    print("="*70)
    
    # Test health
    health = requests.get(f"{API_URL}/health").json()
    print(f"\n1. HEALTH:")
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Redis connected: {health['redis_connected']}")
    
    # Test metrics
    metrics_resp = requests.get(f"{API_URL}/metrics")
    print(f"\n2. METRICS:")
    print(f"   Status: {metrics_resp.status_code}")
    print(f"   Format: {metrics_resp.headers.get('Content-Type')}")
    
    # Test prediction
    pred_payload = {
        "player_id": 80001,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.4,
            "shots_on_target_rolling_5": 1.2,
            "opponent_strength": 0.6,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    pred_resp = requests.post(f"{API_URL}/predict/player", json=pred_payload)
    pred_data = pred_resp.json()
    
    print(f"\n3. PREDICTION:")
    print(f"   Status: {pred_resp.status_code}")
    print(f"   Inference time: {pred_data.get('inference_time_ms')}ms")
    print(f"   Cached: {pred_data.get('cached')}")
    print(f"   Request ID: {pred_resp.headers.get('X-Request-ID')}")
    
    print("\n" + "="*70)