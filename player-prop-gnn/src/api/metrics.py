"""
Prometheus Metrics for Player Prop API
Phase 4.3 - Monitoring Infrastructure

Metrics Philosophy:
- Track what matters: requests, latency, errors, cache
- Use histograms for latency (not averages - they hide outliers)
- Counters never decrease (monotonic)
- Gauges for instantaneous values (cache hit rate)

Design Decisions:
- All metrics prefixed with 'prop_api_' to avoid collisions
- Labels for dimensions (endpoint, method, status)
- Sensible buckets for histograms (50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s)
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CollectorRegistry
from typing import Dict, Optional
import time

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Request count by endpoint and status
REQUEST_COUNT = Counter(
    'prop_api_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

# Request latency histogram (critical for SLA monitoring)
REQUEST_DURATION = Histogram(
    'prop_api_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Model inference time per prop type
INFERENCE_DURATION = Histogram(
    'prop_api_inference_duration_seconds',
    'Model inference duration in seconds',
    ['prop_type'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# Cache performance
CACHE_HITS = Counter(
    'prop_api_cache_hits_total',
    'Total number of cache hits'
)

CACHE_MISSES = Counter(
    'prop_api_cache_misses_total',
    'Total number of cache misses'
)

CACHE_ERRORS = Counter(
    'prop_api_cache_errors_total',
    'Total number of cache operation errors',
    ['operation']  # 'read' or 'write'
)

# Current cache hit rate (gauge for instant value)
CACHE_HIT_RATE = Gauge(
    'prop_api_cache_hit_rate',
    'Current cache hit rate (hits / total requests)'
)

# Prediction errors
PREDICTION_ERRORS = Counter(
    'prop_api_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']  # 'validation', 'model', 'unknown'
)

# Model health
MODEL_LOADED = Gauge(
    'prop_api_model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)'
)

REDIS_CONNECTED = Gauge(
    'prop_api_redis_connected',
    'Whether Redis is connected (1=connected, 0=disconnected)'
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_cache_hit_rate():
    """
    Calculate and update cache hit rate gauge.
    
    Formula: hits / (hits + misses)
    
    Thread-safe: Prometheus metrics are atomic
    Edge case: If no requests yet, return 0.0
    """
    hits = CACHE_HITS._value.get()
    misses = CACHE_MISSES._value.get()
    total = hits + misses
    
    if total == 0:
        rate = 0.0
    else:
        rate = hits / total
    
    CACHE_HIT_RATE.set(rate)


def record_cache_hit():
    """Record a cache hit and update hit rate."""
    CACHE_HITS.inc()
    update_cache_hit_rate()


def record_cache_miss():
    """Record a cache miss and update hit rate."""
    CACHE_MISSES.inc()
    update_cache_hit_rate()


def record_cache_error(operation: str):
    """
    Record a cache operation error.
    
    Args:
        operation: 'read' or 'write'
    """
    if operation not in ['read', 'write']:
        operation = 'unknown'
    CACHE_ERRORS.labels(operation=operation).inc()


def record_prediction_error(error_type: str):
    """
    Record a prediction error by type.
    
    Args:
        error_type: 'validation', 'model', or 'unknown'
    """
    if error_type not in ['validation', 'model', 'unknown']:
        error_type = 'unknown'
    PREDICTION_ERRORS.labels(error_type=error_type).inc()


# ============================================================================
# CONTEXT MANAGERS (for automatic timing)
# ============================================================================

class TimedContext:
    """
    Context manager for timing operations.
    
    Usage:
        with TimedContext(REQUEST_DURATION.labels(method='POST', endpoint='/predict')):
            # ... operation ...
            pass
    
    Automatically records duration to histogram on exit.
    Handles exceptions: Still records time even if operation fails.
    """
    def __init__(self, histogram_metric):
        """
        Args:
            histogram_metric: Prometheus histogram with labels already applied
        """
        self.histogram = histogram_metric
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.histogram.observe(duration)
        # Don't suppress exceptions
        return False


# ============================================================================
# METRICS EXPORT
# ============================================================================

def get_metrics() -> bytes:
    """
    Generate Prometheus metrics in text format.
    
    Returns:
        bytes: Metrics in Prometheus exposition format
    
    Example output:
        # HELP prop_api_requests_total Total number of HTTP requests
        # TYPE prop_api_requests_total counter
        prop_api_requests_total{method="POST",endpoint="/predict",status_code="200"} 42.0
        ...
    """
    return generate_latest(REGISTRY)


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_metrics():
    """
    Initialize metrics at startup.
    
    Sets:
    - Model loaded status (will be updated by startup_event)
    - Redis connected status (will be updated by startup_event)
    - Cache hit rate to 0.0
    """
    MODEL_LOADED.set(0)
    REDIS_CONNECTED.set(0)
    CACHE_HIT_RATE.set(0.0)