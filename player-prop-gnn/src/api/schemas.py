"""
Pydantic Schemas for Player Prop Prediction API
Phase 4.1 - Small Chunk 2: Request/Response Models

Design Decisions:
- Runtime validation: Don't trust type hints, validate at runtime
- Clear errors: Field descriptions and constraints help users fix issues
- Nested models: Structured response for easy parsing

Validation Rules:
- player_id > 0
- opponent_id > 0
- position in allowed set
- features must be finite numbers
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import numpy as np


class PlayerPredictionRequest(BaseModel):
    """
    Request schema for single player prediction.
    
    Example:
        {
            "player_id": 123,
            "opponent_id": 45,
            "position": "Forward",
            "was_home": true,
            "features": {
                "goals_rolling_5": 0.2,
                "shots_on_target_rolling_5": 0.8,
                "opponent_strength": 0.6,
                "days_since_last_match": 5.0,
                "was_home": 1.0
            }
        }
    """
    player_id: int = Field(..., gt=0, description="Database player ID (must be positive)")
    opponent_id: int = Field(..., gt=0, description="Database opponent team ID (must be positive)")
    position: str = Field(..., description="Player position: Forward, Midfielder, Defender, Goalkeeper")
    was_home: bool = Field(..., description="True if player's team is home, False if away")
    features: Dict[str, float] = Field(..., description="Feature dictionary with required keys")
    
    @validator('position')
    def validate_position(cls, v):
        """Ensure position is one of the allowed values."""
        allowed = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
        if v not in allowed:
            raise ValueError(f"Position must be one of {allowed}, got '{v}'")
        return v
    
    @validator('features')
    def validate_features(cls, v):
        """
        Ensure all feature values are finite numbers.
        
        Rejects: NaN, inf, -inf
        Accepts: Any finite float/int
        """
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric, got {type(value)}")
            if not np.isfinite(value):
                raise ValueError(f"Feature '{key}' must be finite (not NaN/inf), got {value}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PropPrediction(BaseModel):
    """
    Prediction for a single prop (goals, shots, or cards).
    
    Includes:
    - Point estimates (mean, median)
    - Uncertainty quantification (std, CI)
    - Probability distribution over outcomes
    """
    mean: float = Field(..., ge=0, description="Expected value (mean of posterior)")
    median: float = Field(..., ge=0, description="Median of posterior distribution")
    std: float = Field(..., ge=0, description="Standard deviation (uncertainty)")
    ci_low: float = Field(..., ge=0, description="Lower bound of 95% credible interval")
    ci_high: float = Field(..., ge=0, description="Upper bound of 95% credible interval")
    probability: Dict[str, float] = Field(..., description="P(outcome): 0, 1, 2+")
    
    @validator('probability')
    def validate_probability_sums_to_one(cls, v):
        """Ensure probabilities sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "mean": 0.15,
                "median": 0.0,
                "std": 0.42,
                "ci_low": 0.0,
                "ci_high": 1.0,
                "probability": {
                    "0": 0.87,
                    "1": 0.11,
                    "2+": 0.02
                }
            }
        }


class PlayerPredictionResponse(BaseModel):
    """
    Response schema for single player prediction.
    
    Contains:
    - Player/match metadata
    - Predictions for all props (goals, shots, cards)
    - Performance metadata (cached, inference time)
    """
    player_id: int = Field(..., description="Player ID from request")
    opponent_id: int = Field(..., description="Opponent team ID from request")
    position: str = Field(..., description="Player position")
    was_home: bool = Field(..., description="Home/away status")
    predictions: Dict[str, PropPrediction] = Field(..., description="Predictions by prop type")
    cached: bool = Field(..., description="True if served from cache, False if fresh prediction")
    inference_time_ms: int = Field(..., ge=0, description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    
    class Config:
        schema_extra = {
            "example": {
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
                "cached": False,
                "inference_time_ms": 85,
                "model_version": "v1.0",
                "timestamp": "2024-11-14T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    
    Provides:
    - Error type for programmatic handling
    - Human-readable message
    - Actionable solution
    - Timestamp for debugging
    """
    error: str = Field(..., description="Error type (e.g., ValidationError, ModelError)")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    solution: Optional[str] = Field(None, description="Suggested fix")
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid player_id",
                "detail": "player_id must be positive, got -5",
                "solution": "Provide a valid player_id > 0",
                "timestamp": "2024-11-14T10:30:00Z"
            }
        }