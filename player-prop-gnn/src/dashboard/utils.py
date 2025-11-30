"""
Dashboard Utilities
Phase 4.4 - API Client and Helpers

Design Decisions:
- Retry logic: 3 attempts with exponential backoff
- Timeout: 10 seconds (generous for demo, prevents hanging)
- Error handling: Specific exceptions for different failure modes
"""
import os
import requests
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

# API URL (environment variable for Docker vs local)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Request timeout (seconds)
REQUEST_TIMEOUT = 10

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


# ============================================================================
# EXCEPTIONS
# ============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    pass


class APIConnectionError(APIError):
    """Cannot connect to API."""
    pass


class APITimeoutError(APIError):
    """API request timed out."""
    pass


class APIValidationError(APIError):
    """API returned validation error (400)."""
    pass


class APIServerError(APIError):
    """API returned server error (500)."""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HealthStatus:
    """API health check response."""
    status: str
    model_loaded: bool
    redis_connected: bool
    version: str
    timestamp: str
    warnings: Optional[list] = None
    errors: Optional[list] = None


@dataclass
class PropPrediction:
    """Prediction for a single prop."""
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float
    probability: Dict[str, float]


@dataclass
class PredictionResponse:
    """Full prediction response."""
    player_id: int
    opponent_id: int
    position: str
    was_home: bool
    predictions: Dict[str, PropPrediction]
    cached: bool
    inference_time_ms: int
    model_version: str
    timestamp: str


# ============================================================================
# API CLIENT
# ============================================================================

def check_health(timeout: int = REQUEST_TIMEOUT) -> HealthStatus:
    """
    Check API health.
    
    Args:
        timeout: Request timeout in seconds
    
    Returns:
        HealthStatus object
    
    Raises:
        APIConnectionError: Cannot connect to API
        APITimeoutError: Request timed out
    """
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # FIX: Handle key name change from 'model_loaded' to 'ensemble_ready'
        # This ensures backward compatibility with older API versions too
        is_loaded = data.get('ensemble_ready', data.get('model_loaded', False))

        return HealthStatus(
            status=data['status'],
            model_loaded=is_loaded,
            redis_connected=data.get('redis_connected', False),
            version=data.get('version', 'unknown'),
            timestamp=data.get('timestamp', ''),
            warnings=data.get('warnings'),
            errors=data.get('errors')
        )
    
    except requests.exceptions.ConnectionError as e:
        raise APIConnectionError(
            f"Cannot connect to API at {API_URL}. "
            f"Is the API running? Error: {e}"
        )
    
    except requests.exceptions.Timeout as e:
        raise APITimeoutError(
            f"API health check timed out after {timeout}s. "
            f"API may be overloaded or starting up."
        )
    
    except Exception as e:
        raise APIError(f"Unexpected error checking health: {e}")


def get_metrics(timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Get Prometheus metrics from API.
    
    Args:
        timeout: Request timeout in seconds
    
    Returns:
        Metrics in Prometheus text format
    
    Raises:
        APIConnectionError: Cannot connect to API
    """
    try:
        response = requests.get(
            f"{API_URL}/metrics",
            timeout=timeout
        )
        response.raise_for_status()
        return response.text
    
    except requests.exceptions.ConnectionError:
        raise APIConnectionError(f"Cannot connect to API at {API_URL}")
    
    except Exception as e:
        raise APIError(f"Error fetching metrics: {e}")


def predict_player(
    player_id: int,
    opponent_id: int,
    position: str,
    was_home: bool,
    features: Dict[str, float],
    timeout: int = REQUEST_TIMEOUT,
    retries: int = MAX_RETRIES
) -> PredictionResponse:
    """
    Make prediction for a player.
    
    Args:
        player_id: Database player ID
        opponent_id: Opponent team ID
        position: Player position
        was_home: True if home game
        features: Feature dictionary
        timeout: Request timeout
        retries: Number of retry attempts
    
    Returns:
        PredictionResponse object
    
    Raises:
        APIConnectionError: Cannot connect to API
        APITimeoutError: Request timed out
        APIValidationError: Invalid input (400)
        APIServerError: Server error (500)
    """
    payload = {
        "player_id": player_id,
        "opponent_id": opponent_id,
        "position": position,
        "was_home": was_home,
        "features": features
    }
    
    last_error = None
    
    # Retry loop with exponential backoff
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{API_URL}/predict/player",
                json=payload,
                timeout=timeout
            )
            
            # Handle different status codes
            if response.status_code == 200:
                data = response.json()
                
                # Parse predictions
                predictions = {}
                for prop_type, prop_data in data['predictions'].items():
                    predictions[prop_type] = PropPrediction(**prop_data)
                
                return PredictionResponse(
                    player_id=data['player_id'],
                    opponent_id=data['opponent_id'],
                    position=data['position'],
                    was_home=data['was_home'],
                    predictions=predictions,
                    cached=data['cached'],
                    inference_time_ms=data['inference_time_ms'],
                    model_version=data['model_version'],
                    timestamp=data['timestamp']
                )
            
            elif response.status_code == 400:
                error_detail = response.json().get('detail', 'Invalid input')
                raise APIValidationError(f"Validation error: {error_detail}")
            
            elif response.status_code == 503:
                raise APIServerError("API is starting up or model not loaded")
            
            else:
                raise APIServerError(
                    f"API returned status {response.status_code}: {response.text}"
                )
        
        except requests.exceptions.ConnectionError as e:
            last_error = APIConnectionError(f"Cannot connect to API: {e}")
            
        except requests.exceptions.Timeout as e:
            last_error = APITimeoutError(f"Request timed out after {timeout}s")
        
        except (APIValidationError, APIServerError):
            # Don't retry on validation or server errors
            raise
        
        except Exception as e:
            last_error = APIError(f"Unexpected error: {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < retries - 1:
            wait_time = RETRY_DELAY * (2 ** attempt)
            time.sleep(wait_time)
    
    # All retries failed
    raise last_error


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_metric(metrics_text: str, metric_name: str) -> Optional[float]:
    """
    Parse a single metric value from Prometheus text format.
    
    Args:
        metrics_text: Full metrics response
        metric_name: Metric name to extract (e.g., 'prop_api_cache_hit_rate')
    
    Returns:
        Metric value or None if not found
    """
    for line in metrics_text.split('\n'):
        # Skip comments and empty lines
        if line.startswith('#') or not line.strip():
            continue
        
        # Parse metric line: "metric_name{labels} value"
        if line.startswith(metric_name):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
    
    return None


def format_probability(prob: float) -> str:
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"


def format_confidence_interval(ci_low: float, ci_high: float) -> str:
    """Format confidence interval."""
    return f"[{ci_low:.1f}, {ci_high:.1f}]"


def get_position_emoji(position: str) -> str:
    """Get emoji for player position."""
    emojis = {
        'Forward': '⚽',
        'Midfielder': '🎯',
        'Defender': '🛡️',
        'Goalkeeper': '🥅'
    }
    return emojis.get(position, '👤')


def get_status_color(status: str) -> str:
    """Get color for health status."""
    if status == 'healthy':
        return 'green'
    elif status == 'unhealthy':
        return 'red'
    else:
        return 'orange'