"""
Unit Tests for API Schemas
Phase 4: FastAPI Development

Tests Pydantic validation for:
- PlayerPredictionRequest
- PropPrediction
- PlayerPredictionResponse
- ErrorResponse

Following Google AI guidelines:
- Test edge cases (empty, null, extreme values)
- Test validation rules
- Test serialization/deserialization
"""
import pytest
from pydantic import ValidationError
import numpy as np

from src.api.schemas import (
    PlayerPredictionRequest,
    PropPrediction,
    PlayerPredictionResponse,
    ErrorResponse
)


# ============================================================================
# TEST PLAYER PREDICTION REQUEST
# ============================================================================

def test_valid_player_prediction_request():
    """Test that valid request passes validation."""
    request = PlayerPredictionRequest(
        player_id=123,
        opponent_id=45,
        position="Forward",
        was_home=True,
        features={
            "goals_rolling_5": 0.2,
            "shots_on_target_rolling_5": 0.8,
            "opponent_strength": 0.6,
            "days_since_last_match": 5.0,
            "was_home": 1.0
        }
    )
    
    assert request.player_id == 123
    assert request.opponent_id == 45
    assert request.position == "Forward"
    assert request.was_home is True
    assert len(request.features) == 5


def test_player_prediction_request_invalid_player_id():
    """Test that negative player_id is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=-1,
            opponent_id=45,
            position="Forward",
            was_home=True,
            features={}
        )
    
    assert 'player_id' in str(exc_info.value)


def test_player_prediction_request_zero_player_id():
    """Test that zero player_id is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=0,
            opponent_id=45,
            position="Forward",
            was_home=True,
            features={}
        )
    
    assert 'player_id' in str(exc_info.value)


def test_player_prediction_request_invalid_position():
    """Test that invalid position is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=123,
            opponent_id=45,
            position="InvalidPosition",
            was_home=True,
            features={}
        )
    
    assert 'Position must be one of' in str(exc_info.value)


def test_player_prediction_request_all_valid_positions():
    """Test that all valid positions are accepted."""
    valid_positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    
    for position in valid_positions:
        request = PlayerPredictionRequest(
            player_id=123,
            opponent_id=45,
            position=position,
            was_home=True,
            features={"test": 1.0}
        )
        assert request.position == position


def test_player_prediction_request_nan_feature():
    """Test that NaN feature values are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=123,
            opponent_id=45,
            position="Forward",
            was_home=True,
            features={"goals_rolling_5": float('nan')}
        )
    
    assert 'must be finite' in str(exc_info.value)


def test_player_prediction_request_inf_feature():
    """Test that infinite feature values are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=123,
            opponent_id=45,
            position="Forward",
            was_home=True,
            features={"goals_rolling_5": float('inf')}
        )
    
    assert 'must be finite' in str(exc_info.value)


def test_player_prediction_request_non_numeric_feature():
    """Test that non-numeric feature values are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionRequest(
            player_id=123,
            opponent_id=45,
            position="Forward",
            was_home=True,
            features={"goals_rolling_5": "not_a_number"}
        )
    
    assert 'valid number' in str(exc_info.value)


# ============================================================================
# TEST PROP PREDICTION
# ============================================================================

def test_valid_prop_prediction():
    """Test that valid PropPrediction passes validation."""
    prop = PropPrediction(
        mean=0.15,
        median=0.0,
        std=0.42,
        ci_low=0.0,
        ci_high=1.0,
        probability={"0": 0.87, "1": 0.11, "2+": 0.02}
    )
    
    assert prop.mean == 0.15
    assert prop.median == 0.0
    assert sum(prop.probability.values()) == pytest.approx(1.0, abs=0.01)


def test_prop_prediction_negative_mean():
    """Test that negative mean is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PropPrediction(
            mean=-0.15,
            median=0.0,
            std=0.42,
            ci_low=0.0,
            ci_high=1.0,
            probability={"0": 0.87, "1": 0.11, "2+": 0.02}
        )
    
    assert 'mean' in str(exc_info.value).lower()


def test_prop_prediction_probabilities_dont_sum_to_one():
    """Test that probabilities must sum to 1.0."""
    with pytest.raises(ValidationError) as exc_info:
        PropPrediction(
            mean=0.15,
            median=0.0,
            std=0.42,
            ci_low=0.0,
            ci_high=1.0,
            probability={"0": 0.5, "1": 0.3, "2+": 0.1}  # Sums to 0.9
        )
    
    assert 'Probabilities must sum to 1.0' in str(exc_info.value)


# ============================================================================
# TEST PLAYER PREDICTION RESPONSE
# ============================================================================

def test_valid_player_prediction_response():
    """Test that valid response passes validation."""
    response = PlayerPredictionResponse(
        player_id=123,
        opponent_id=45,
        position="Forward",
        was_home=True,
        predictions={
            "goals": PropPrediction(
                mean=0.15, median=0.0, std=0.42,
                ci_low=0.0, ci_high=1.0,
                probability={"0": 0.87, "1": 0.11, "2+": 0.02}
            )
        },
        cached=False,
        inference_time_ms=85,
        model_version="v1.0",
        timestamp="2024-11-14T10:30:00Z"
    )
    
    assert response.player_id == 123
    assert response.cached is False
    assert response.inference_time_ms == 85


def test_player_prediction_response_negative_inference_time():
    """Test that negative inference time is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PlayerPredictionResponse(
            player_id=123,
            opponent_id=45,
            position="Forward",
            was_home=True,
            predictions={},
            cached=False,
            inference_time_ms=-10,
            model_version="v1.0",
            timestamp="2024-11-14T10:30:00Z"
        )
    
    assert 'inference_time_ms' in str(exc_info.value)


# ============================================================================
# TEST ERROR RESPONSE
# ============================================================================

def test_valid_error_response():
    """Test that valid ErrorResponse passes validation."""
    error = ErrorResponse(
        error="ValidationError",
        message="Invalid player_id",
        detail="player_id must be positive",
        solution="Provide a valid player_id > 0",
        timestamp="2024-11-14T10:30:00Z"
    )
    
    assert error.error == "ValidationError"
    assert error.message == "Invalid player_id"


def test_error_response_optional_fields():
    """Test that detail and solution are optional."""
    error = ErrorResponse(
        error="InternalError",
        message="Something went wrong",
        timestamp="2024-11-14T10:30:00Z"
    )
    
    assert error.detail is None
    assert error.solution is None


# ============================================================================
# TEST SERIALIZATION
# ============================================================================

def test_request_serialization_roundtrip():
    """Test that request can be serialized and deserialized."""
    original = PlayerPredictionRequest(
        player_id=123,
        opponent_id=45,
        position="Forward",
        was_home=True,
        features={"test": 1.0}
    )
    
    # Serialize to JSON
    json_str = original.json()
    
    # Deserialize back
    restored = PlayerPredictionRequest.parse_raw(json_str)
    
    assert restored.player_id == original.player_id
    assert restored.opponent_id == original.opponent_id
    assert restored.position == original.position


def test_response_serialization_roundtrip():
    """Test that response can be serialized and deserialized."""
    original = PlayerPredictionResponse(
        player_id=123,
        opponent_id=45,
        position="Forward",
        was_home=True,
        predictions={
            "goals": PropPrediction(
                mean=0.15, median=0.0, std=0.42,
                ci_low=0.0, ci_high=1.0,
                probability={"0": 0.87, "1": 0.11, "2+": 0.02}
            )
        },
        cached=False,
        inference_time_ms=85,
        model_version="v1.0",
        timestamp="2024-11-14T10:30:00Z"
    )
    
    # Serialize to JSON
    json_str = original.json()
    
    # Deserialize back
    restored = PlayerPredictionResponse.parse_raw(json_str)
    
    assert restored.player_id == original.player_id
    assert restored.inference_time_ms == original.inference_time_ms
    assert restored.predictions["goals"].mean == original.predictions["goals"].mean