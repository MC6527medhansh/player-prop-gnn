"""
FastAPI Application for Player Prop Prediction
Phase 4.1 - Small Chunk 1: Basic Application Structure

Application Lifecycle:
- Startup: Load Bayesian model into memory (5-10s, done once)
- Running: Serve predictions from cached model
- Shutdown: Gracefully close connections

Design Decisions:
- Global MODEL variable: Model is stateless, safe to share across requests
- Fail-fast: If model fails to load, don't start API
- Async: Use async/await for concurrent request handling
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional

from src.models.inference import BayesianPredictor, ModelNotLoadedError
from src.config.settings import settings
from .schemas import (
    PlayerPredictionRequest,
    PlayerPredictionResponse,
    PropPrediction,
    ErrorResponse
)
import redis
import json

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (loaded once at startup)
MODEL: Optional[BayesianPredictor] = None

# Global Redis client (connected at startup)
REDIS_CLIENT: Optional[redis.Redis] = None

# Create FastAPI app
app = FastAPI(
    title="Player Prop Prediction API",
    description="Bayesian multi-task model for football player prop predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def startup_event():
    """
    Load model and connect to Redis on application startup.
    
    Failure Modes:
    - Model files missing: API won't start (fail-fast)
    - Redis down: API starts but runs without cache (graceful degradation)
    """
    global MODEL, REDIS_CLIENT
    
    logger.info("=" * 60)
    logger.info("STARTING PLAYER PROP API")
    logger.info("=" * 60)
    
    # ========================================
    # LOAD MODEL (REQUIRED)
    # ========================================
    
    model_path = settings.model_path / "bayesian_multitask_v1.0.pkl"
    trace_path = settings.model_path / "bayesian_multitask_v1.0_trace.nc"
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        MODEL = BayesianPredictor(
            str(model_path),
            str(trace_path),
            n_samples=1000
        )
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Posterior samples: {MODEL.n_samples}")
        logger.info(f"  Feature count: {len(MODEL.feature_names)}")
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error("Solution: Train model first with: python -m src.models.train")
        raise RuntimeError("Cannot start API without model files")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading error: {e}")
    
    # ========================================
    # CONNECT TO REDIS (OPTIONAL)
    # ========================================
    
    logger.info(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    try:
        REDIS_CLIENT = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test connection
        REDIS_CLIENT.ping()
        logger.info("✓ Redis connected successfully")
        
    except redis.ConnectionError as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        logger.warning("API will run without caching (slower but functional)")
        REDIS_CLIENT = None
        
    except Exception as e:
        logger.warning(f"Unexpected Redis error: {e}")
        logger.warning("API will run without caching")
        REDIS_CLIENT = None
    
    logger.info("✓ API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown.
    
    Closes:
    - Redis connection (if connected)
    
    Future:
    - Database connection pools
    - Flush pending caches
    """
    global REDIS_CLIENT
    
    logger.info("=" * 60)
    logger.info("SHUTTING DOWN PLAYER PROP API")
    logger.info("=" * 60)
    
    if REDIS_CLIENT is not None:
        try:
            REDIS_CLIENT.close()
            logger.info("✓ Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")
    
    logger.info("✓ Shutdown complete")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Checks:
    - Model loaded
    - Redis connected (optional, warning if down)
    
    Returns:
        - 200 OK if model loaded
        - 503 Service Unavailable if model not loaded
    
    Used by:
        - Docker health checks (HEALTHCHECK directive)
        - Kubernetes liveness probes
        - Load balancers
    
    Latency Target: <10ms
    """
    # Check Redis connection
    redis_connected = False
    if REDIS_CLIENT is not None:
        try:
            REDIS_CLIENT.ping()
            redis_connected = True
        except:
            redis_connected = False
    
    health_status = {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None,
        "redis_connected": redis_connected,
        "version": "1.0.0"
    }
    
    # Return 503 if model not loaded
    if MODEL is None:
        health_status["errors"] = ["Model not loaded"]
        return JSONResponse(
            status_code=503,
            content=health_status
        )
    
    # Warning if Redis down (not critical, API still works)
    if not redis_connected:
        health_status["warnings"] = ["Redis not connected - caching disabled"]
    
    return health_status


@app.post("/predict/player", response_model=PlayerPredictionResponse)
async def predict_player(request: PlayerPredictionRequest):
    """
    Predict props for a single player with Redis caching.
    
    Caching Strategy:
    - Key format: pred:tier1:{player_id}:{opponent_id}:{was_home}:v1.0
    - TTL: 5 minutes (300 seconds)
    - Fallback: If Redis down, skip cache (graceful degradation)
    
    Args:
        request: Player prediction request (validated by Pydantic)
    
    Returns:
        PlayerPredictionResponse with predictions for all props
    
    Raises:
        HTTPException 503: Model not loaded
        HTTPException 500: Prediction failed
        HTTPException 400: Invalid request (caught by Pydantic)
    
    Latency Target:
        - <100ms with cache hit
        - <500ms with cache miss (uncached prediction)
    
    Example:
        POST /predict/player
        {
            "player_id": 123,
            "opponent_id": 45,
            "position": "Forward",
            "was_home": true,
            "features": {...}
        }
    """
    # Check model loaded
    if MODEL is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API is starting up or failed to load model."
        )
    
    # ========================================
    # CHECK CACHE (if Redis available)
    # ========================================
    
    cache_key = f"pred:tier1:{request.player_id}:{request.opponent_id}:{request.was_home}:v1.0"
    cached_result = None
    
    if REDIS_CLIENT is not None:
        try:
            cached_result = REDIS_CLIENT.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT: {cache_key}")
                response_dict = json.loads(cached_result)
                return PlayerPredictionResponse(**response_dict)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            # Continue without cache (graceful degradation)
    
    # ========================================
    # CACHE MISS: Make fresh prediction
    # ========================================
    
    try:
        start_time = datetime.now()
        
        result = MODEL.predict_player(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            features=request.features
        )
        
        inference_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Convert to response format
        response = PlayerPredictionResponse(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            position=request.position,
            was_home=request.was_home,
            predictions={
                "goals": PropPrediction(**result['goals']),
                "shots": PropPrediction(**result['shots']),
                "cards": PropPrediction(**result['cards'])
            },
            cached=False,
            inference_time_ms=inference_time_ms,
            model_version="v1.0",
            timestamp=datetime.now().isoformat()
        )
        
        # ========================================
        # STORE IN CACHE (if Redis available)
        # ========================================
        
        if REDIS_CLIENT is not None:
            try:
                # Serialize response
                response_json = response.json()
                
                # Store with 5-minute TTL
                REDIS_CLIENT.setex(cache_key, 300, response_json)
                logger.info(f"Cache MISS: {cache_key} (stored for 300s)")
                
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
                # Continue (prediction succeeded, just cache failed)
        
        logger.info(
            f"Prediction complete: player={request.player_id}, "
            f"time={inference_time_ms}ms"
        )
        
        return response
        
    except ValueError as e:
        # Input validation error from BayesianPredictor
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
        
    except Exception as e:
        # Unexpected error during prediction
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )