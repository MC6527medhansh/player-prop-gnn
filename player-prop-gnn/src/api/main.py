"""
FastAPI Application for Player Prop Prediction
Phase 8.3 - FULL ENSEMBLE INTEGRATION (TIER 1 + TIER 2)

FEATURES:
1. Ensemble Prediction Endpoint (Bayesian + GNN)
2. Intelligent Routing (Cold Start Logic)
3. Full Monitoring (Prometheus, JSON Logging)
4. Redis Caching
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import time
import hashlib
import json
import redis
import os

# Import core modules
from src.models.inference import BayesianPredictor
from src.models.gnn_inference import GNNPredictor
from src.models.ensemble import EnsemblePredictor
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

# Import the Parlay Router
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
ENSEMBLE: Optional[EnsemblePredictor] = None
REDIS_CLIENT: Optional[redis.Redis] = None

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Player Prop Ensemble API",
    description="Tier 1 (Bayesian) + Tier 2 (GNN) Ensemble System",
    version="3.0.0",  # Major version bump for Ensemble
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup Middleware
setup_middleware(app)

# REGISTER ROUTERS
app.include_router(parlay_routes.router, prefix="/parlay", tags=["Parlay Pricing"])

# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Load Tier 1, Tier 2, Initialize Ensemble, and connect to Redis."""
    global ENSEMBLE, REDIS_CLIENT
    
    initialize_metrics()
    
    logger.info("=" * 60)
    logger.info("STARTING PLAYER PROP ENSEMBLE API (PHASE 8)")
    logger.info("=" * 60)
    
    # --- 1. LOAD MODELS ---
    try:
        start_time = time.time()
        
        # Paths
        t1_model_path = settings.model_path / "bayesian_multitask_v1.0.pkl"
        t1_trace_path = settings.model_path / "bayesian_multitask_v1.0_trace.nc"
        t2_model_path = settings.model_path / "gnn_best.pt"

        # Tier 1: Bayesian
        if not t1_model_path.exists():
            raise FileNotFoundError(f"Tier 1 model missing at {t1_model_path}")
        
        logger.info(f"Loading Tier 1 (Bayesian) from {t1_model_path}...")
        tier1 = BayesianPredictor(str(t1_model_path), str(t1_trace_path), n_samples=1000)
        
        # Tier 2: GNN
        # Construct DB config from settings for GNN graph building
        db_config = {
            'host': settings.DATABASE_HOST,
            'port': settings.DATABASE_PORT,
            'database': settings.DATABASE_NAME,
            'user': settings.DATABASE_USER,
            'password': settings.DATABASE_PASSWORD
        }
        
        logger.info(f"Loading Tier 2 (GNN) from {t2_model_path}...")
        # Note: In production, we might mock GNN if file missing to allow partial startup
        if t2_model_path.exists():
            tier2 = GNNPredictor(
                model_path=str(t2_model_path),
                db_config=db_config,
                device=os.environ.get("GNN_DEVICE", "cpu")
            )
        else:
            logger.warning("⚠️ GNN model file not found. Ensemble will run in Tier 1 fallback mode.")
            tier2 = None

        # Ensemble
        logger.info("Initializing Ensemble Routing Logic...")
        ENSEMBLE = EnsemblePredictor(
            bayesian_predictor=tier1,
            gnn_predictor=tier2,
            tier1_weight=0.6,
            tier2_weight=0.4
        )
        
        MODEL_LOADED.set(1)
        logger.info(f"✅ Ensemble ready in {time.time() - start_time:.2f}s")
            
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.critical(f"❌ Failed to initialize models: {e}")
        # We don't raise here to allow the pod to start and report 'unhealthy' instead of crash loop
    
    # --- 2. CONNECT TO REDIS ---
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
        logger.info("✅ Redis connected successfully")
    except Exception as e:
        REDIS_CONNECTED.set(0)
        logger.warning(f"⚠️ Redis connection failed: {e}")
        REDIS_CLIENT = None
    
    logger.info("🚀 API ready to serve requests")

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
    is_ready = ENSEMBLE is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ensemble_ready": is_ready,
        "gnn_active": is_ready and ENSEMBLE.tier2 is not None,
        "redis_connected": REDIS_CLIENT is not None,
        "version": "3.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain")

@app.post("/predict/player", response_model=PlayerPredictionResponse)
async def predict_player(request: PlayerPredictionRequest, http_request: Request):
    """
    Ensemble Prediction: Intelligent routing between Tier 1 and Tier 2.
    """
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    req_logger = RequestLogger(logger, request_id)
    
    if ENSEMBLE is None:
        record_prediction_error('model_missing')
        raise HTTPException(status_code=503, detail="Ensemble model not initialized")
    
    # --- CACHE KEY GENERATION ---
    # We add 'ensemble_v1' to key to differentiate from old Tier 1 only keys
    features_json = json.dumps(request.features, sort_keys=True)
    features_hash = hashlib.md5(features_json.encode()).hexdigest()[:8]
    
    # Include games_played in cache key as it affects routing logic
    games_played = getattr(request, 'games_played', 0)
    
    cache_key = (
        f"pred:ensemble:{request.player_id}:{request.opponent_id}:"
        f"{request.position}:{request.was_home}:{features_hash}:gp{games_played}:v3.0"
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
        
        # Use Ensemble Predictor
        result = ENSEMBLE.predict_player(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            features=request.features,
            games_played=games_played
        )
        
        duration = int((time.time() - start_time) * 1000)
        
        # Record metrics
        for prop in ['goals', 'shots', 'cards']:
            INFERENCE_DURATION.labels(prop_type=prop).observe(duration / 1000.0 / 3)
            
        # Format Response
        # Extract metadata specifically for the response object
        meta = result.pop('metadata', {})
        
        response = PlayerPredictionResponse(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            predictions={
                k: PropPrediction(**v) for k,v in result.items() if k in ['goals', 'shots', 'cards']
            },
            cached=False,
            inference_time_ms=duration,
            model_version=f"Ensemble-v3.0 ({meta.get('model_mode', 'unknown')})",
            timestamp=datetime.now().isoformat()
        )
        
        # --- CACHE WRITE ---
        if REDIS_CLIENT:
            try:
                # Cache for 5 minutes
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
        # In production, we might fallback to a naive average here, but for now 500
        raise HTTPException(status_code=500, detail=f"Internal inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)