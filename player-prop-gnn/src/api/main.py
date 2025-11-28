"""
FastAPI Application for Player Prop Prediction
Phase 7.3 - INTEGRATED PARLAY PRICING & MONITORING

FEATURES:
1. Tier 1 (Bayesian) Prediction Endpoint
2. Tier 2 (GNN) Parlay Pricing Endpoint
3. Full Monitoring (Prometheus, JSON Logging)
4. Redis Caching
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional
import time
import hashlib
import json
import redis

# Import core modules
from src.models.inference import BayesianPredictor
from src.config.settings import settings
from src.utils.logging import setup_logging, RequestLogger

# Import API components
from .schemas import (
    PlayerPredictionRequest,
    PlayerPredictionResponse,
    PropPrediction
)
from .middleware import setup_middleware
from .metrics import (
    get_metrics,
    initialize_metrics,
    record_cache_hit,
    record_cache_miss,
    record_cache_error,
    record_prediction_error,
    MODEL_LOADED,
    REDIS_CONNECTED,
    INFERENCE_DURATION
)

# Import the new Parlay Router
from src.api import parlay_routes

# ============================================================================
# CONFIGURE LOGGING
# ============================================================================
format_json = settings.LOG_LEVEL.upper() != 'DEBUG'
logger = setup_logging(
    level=settings.LOG_LEVEL,
    format_json=format_json,
    log_file=None
)

# ============================================================================
# GLOBAL STATE
# ============================================================================
MODEL: Optional[BayesianPredictor] = None
REDIS_CLIENT: Optional[redis.Redis] = None

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Player Prop Prediction API",
    description="Bayesian + GNN system for football player prop predictions",
    version="2.0.0",  # Bumped version for GNN integration
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup Middleware
setup_middleware(app)

# REGISTER ROUTERS
# This hooks the Parlay Pricing Engine into the main API
app.include_router(parlay_routes.router, prefix="/parlay", tags=["Parlay Pricing"])

# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Load model and connect to Redis on startup."""
    global MODEL, REDIS_CLIENT
    
    initialize_metrics()
    
    logger.info("=" * 60)
    logger.info("STARTING PLAYER PROP API (TIER 1 + TIER 2)")
    logger.info("=" * 60)
    
    # 1. LOAD BAYESIAN MODEL (Tier 1)
    model_path = settings.model_path / "bayesian_multitask_v1.0.pkl"
    trace_path = settings.model_path / "bayesian_multitask_v1.0_trace.nc"
    
    logger.info(f"Loading Tier 1 model from {model_path}")
    
    try:
        start_time = time.time()
        # Ensure directory exists before loading
        if not model_path.exists():
             logger.warning(f"Model file not found at {model_path}. Running in limited mode (Parlay-only or Mock).")
        else:
            MODEL = BayesianPredictor(str(model_path), str(trace_path), n_samples=1000)
            MODEL_LOADED.set(1)
            logger.info("Tier 1 Model loaded successfully")
            
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Failed to load Tier 1 model: {e}")
    
    # 2. CONNECT TO REDIS
    logger.info(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    try:
        REDIS_CLIENT = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=5
        )
        REDIS_CLIENT.ping()
        REDIS_CONNECTED.set(1)
        logger.info("Redis connected successfully")
    except Exception as e:
        REDIS_CONNECTED.set(0)
        logger.warning(f"Redis connection failed: {e}")
        REDIS_CLIENT = None
    
    logger.info("API ready to serve requests")

@app.on_event("shutdown")
async def shutdown_event():
    global REDIS_CLIENT
    logger.info("Shutting down API")
    if REDIS_CLIENT:
        REDIS_CLIENT.close()

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tier1_loaded": MODEL is not None,
        "redis_connected": REDIS_CLIENT is not None,
        "version": "2.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain")

@app.post("/predict/player", response_model=PlayerPredictionResponse)
async def predict_player(request: PlayerPredictionRequest, http_request: Request):
    """
    Tier 1 Prediction: Single Player Props.
    """
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    req_logger = RequestLogger(logger, request_id)
    
    if MODEL is None:
        record_prediction_error('model_missing')
        raise HTTPException(status_code=503, detail="Tier 1 Model not loaded")
    
    # --- CACHE KEY GENERATION ---
    features_json = json.dumps(request.features, sort_keys=True)
    features_hash = hashlib.md5(features_json.encode()).hexdigest()[:8]
    
    cache_key = (
        f"pred:tier1:{request.player_id}:{request.opponent_id}:"
        f"{request.position}:{request.was_home}:{features_hash}:v1.0"
    )
    
    # --- CACHE READ ---
    if REDIS_CLIENT:
        try:
            cached = REDIS_CLIENT.get(cache_key)
            if cached:
                record_cache_hit()
                resp = json.loads(cached)
                resp['cached'] = True
                return PlayerPredictionResponse(**resp)
        except Exception:
            record_cache_error('read')
            
    # --- INFERENCE ---
    record_cache_miss()
    try:
        start_time = time.time()
        result = MODEL.predict_player(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            features=request.features
        )
        duration = int((time.time() - start_time) * 1000)
        
        # Record metrics
        for prop in ['goals', 'shots', 'cards']:
            INFERENCE_DURATION.labels(prop_type=prop).observe(duration / 1000.0 / 3)
            
        response = PlayerPredictionResponse(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            predictions={
                k: PropPrediction(**v) for k,v in result.items()
            },
            cached=False,
            inference_time_ms=duration,
            model_version="v1.0",
            timestamp=datetime.now().isoformat()
        )
        
        # --- CACHE WRITE ---
        if REDIS_CLIENT:
            try:
                REDIS_CLIENT.setex(cache_key, 300, response.json())
            except Exception:
                record_cache_error('write')
                
        return response

    except ValueError as e:
        record_prediction_error('validation')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_prediction_error('unknown')
        req_logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)