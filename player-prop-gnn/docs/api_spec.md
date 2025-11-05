# API Specification

## Overview
RESTful API for serving player prop predictions with caching and monitoring. Built with FastAPI for automatic documentation and async support.

---

## Base URL

**Development:** `http://localhost:8000`
**Production:** `https://api.player-props.com` (if deployed)

---

## Authentication

**Phase 1-10:** No authentication (local development)
**Phase 11:** Optional API key for rate limiting

```http
Authorization: Bearer <api_key>
```

---

## Endpoints

### 1. GET /health

**Purpose:** Health check for monitoring

**Request:**
```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-11-01T10:30:00Z",
  "model_loaded": true,
  "database_connected": true,
  "redis_connected": true,
  "version": "1.0.0"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-11-01T10:30:00Z",
  "model_loaded": false,
  "database_connected": true,
  "redis_connected": false,
  "errors": ["Redis connection failed"]
}
```

**Latency Target:** <10ms

---

### 2. POST /predict/player

**Purpose:** Get predictions for a single player in a specific match

**Request:**
```json
{
  "player_id": 123,
  "match_id": 456,
  "tier": "tier1",
  "return_uncertainty": true
}
```

**Parameters:**
- `player_id` (required): Integer, database player ID
- `match_id` (required): Integer, database match ID
- `tier` (optional): String, one of ["tier1", "tier2", "ensemble"], default "tier1"
- `return_uncertainty` (optional): Boolean, include confidence intervals, default false

**Response (200 OK):**
```json
{
  "player_id": 123,
  "player_name": "Mohamed Salah",
  "team": "Liverpool",
  "match_id": 456,
  "opponent": "Manchester United",
  "match_date": "2024-11-15",
  "is_home": true,
  "predictions": {
    "goals": {
      "probability": 0.352,
      "confidence_interval": [0.285, 0.424],
      "uncertainty": 0.139,
      "tier": "tier1"
    },
    "assists": {
      "probability": 0.241,
      "confidence_interval": [0.189, 0.298],
      "uncertainty": 0.109,
      "tier": "tier1"
    },
    "shots_over_2.5": {
      "probability": 0.567,
      "confidence_interval": [0.492, 0.638],
      "uncertainty": 0.146,
      "tier": "tier1"
    },
    "cards": {
      "probability": 0.087,
      "confidence_interval": [0.052, 0.135],
      "uncertainty": 0.083,
      "tier": "tier1"
    }
  },
  "model_version": "tier1_v1.0_2024-11-01",
  "cached": false,
  "inference_time_ms": 85,
  "timestamp": "2024-11-01T10:30:00Z"
}
```

**Error Responses:**

**400 Bad Request:**
```json
{
  "error": "ValidationError",
  "message": "player_id must be a positive integer",
  "field": "player_id",
  "provided_value": -5
}
```

**404 Not Found:**
```json
{
  "error": "NotFound",
  "message": "Player with ID 123 not found in database",
  "resource": "player",
  "id": 123
}
```

**500 Internal Server Error:**
```json
{
  "error": "InferenceError",
  "message": "Model inference failed",
  "details": "MCMC sampling did not converge"
}
```

**Latency Target:** <100ms (p95) with cache, <500ms without

---

### 3. POST /predict/match

**Purpose:** Get predictions for all players in a match

**Request:**
```json
{
  "match_id": 456,
  "tier": "tier1",
  "min_minutes": 20
}
```

**Parameters:**
- `match_id` (required): Integer
- `tier` (optional): String, default "tier1"
- `min_minutes` (optional): Integer, only include players expected to play >= min_minutes, default 0

**Response (200 OK):**
```json
{
  "match_id": 456,
  "home_team": "Liverpool",
  "away_team": "Manchester United",
  "match_date": "2024-11-15",
  "predictions": [
    {
      "player_id": 123,
      "player_name": "Mohamed Salah",
      "team": "Liverpool",
      "is_home": true,
      "predictions": {
        "goals": {"probability": 0.352},
        "assists": {"probability": 0.241},
        "shots_over_2.5": {"probability": 0.567},
        "cards": {"probability": 0.087}
      }
    },
    {
      "player_id": 124,
      "player_name": "Trent Alexander-Arnold",
      "team": "Liverpool",
      "is_home": true,
      "predictions": {
        "goals": {"probability": 0.081},
        "assists": {"probability": 0.312},
        "shots_over_2.5": {"probability": 0.198},
        "cards": {"probability": 0.145}
      }
    }
    // ... 20 more players
  ],
  "model_version": "tier1_v1.0_2024-11-01",
  "cached": true,
  "inference_time_ms": 342,
  "timestamp": "2024-11-01T10:30:00Z"
}
```

**Latency Target:** <500ms (p95) for 22 players

---

### 4. POST /predict/parlay

**Purpose:** Calculate joint probability for a parlay bet

**Request:**
```json
{
  "match_id": 456,
  "legs": [
    {
      "player_id": 123,
      "prop": "goals",
      "line": "over_0.5"
    },
    {
      "player_id": 123,
      "prop": "shots_over_2.5",
      "line": "over"
    },
    {
      "player_id": 124,
      "prop": "assists",
      "line": "over_0.5"
    }
  ],
  "tier": "tier2",
  "include_correlation_matrix": false
}
```

**Parameters:**
- `match_id` (required): Integer
- `legs` (required): Array of leg objects
  - `player_id` (required): Integer
  - `prop` (required): String, one of ["goals", "assists", "shots_over_2.5", "cards"]
  - `line` (required): String, "over" or "under" (for binary props)
- `tier` (optional): "tier1" (independence), "tier2" (GNN correlations), default "tier2"
- `include_correlation_matrix` (optional): Boolean, default false

**Response (200 OK):**
```json
{
  "match_id": 456,
  "parlay": {
    "legs": [
      {
        "player_id": 123,
        "player_name": "Mohamed Salah",
        "prop": "goals",
        "line": "over_0.5",
        "individual_probability": 0.352
      },
      {
        "player_id": 123,
        "player_name": "Mohamed Salah",
        "prop": "shots_over_2.5",
        "line": "over",
        "individual_probability": 0.567
      },
      {
        "player_id": 124,
        "player_name": "Trent Alexander-Arnold",
        "prop": "assists",
        "line": "over_0.5",
        "individual_probability": 0.312
      }
    ],
    "joint_probability_tier1": 0.0623,
    "joint_probability_tier2": 0.0891,
    "correlation_adjustment": 0.0268,
    "implied_odds_tier1": 16.05,
    "implied_odds_tier2": 11.22,
    "correlation_matrix": null
  },
  "model_version": "tier2_v1.0_2024-11-01",
  "inference_time_ms": 127,
  "timestamp": "2024-11-01T10:30:00Z"
}
```

**With Correlation Matrix:**
```json
{
  // ... same as above ...
  "correlation_matrix": {
    "Salah_goals_Salah_shots": 0.74,
    "Salah_goals_TAA_assists": 0.32,
    "Salah_shots_TAA_assists": 0.28
  }
}
```

**Latency Target:** <200ms (p95)

---

### 5. POST /recommend/parlays

**Purpose:** Find +EV parlays based on bookmaker odds

**Request:**
```json
{
  "match_id": 456,
  "bookmaker_odds": [
    {
      "legs": [
        {"player_id": 123, "prop": "goals", "line": "over_0.5"},
        {"player_id": 123, "prop": "shots_over_2.5", "line": "over"}
      ],
      "odds": 7.5
    }
    // ... more parlays
  ],
  "min_edge": 0.05,
  "max_legs": 3,
  "max_results": 10,
  "kelly_fraction": 0.25
}
```

**Parameters:**
- `match_id` (required): Integer
- `bookmaker_odds` (required): Array of bookmaker parlay offers
- `min_edge` (optional): Float, minimum edge (%) to recommend, default 0.05 (5%)
- `max_legs` (optional): Integer, maximum parlay legs, default 3
- `max_results` (optional): Integer, max recommendations to return, default 10
- `kelly_fraction` (optional): Float, fraction of Kelly criterion for stake, default 0.25

**Response (200 OK):**
```json
{
  "match_id": 456,
  "recommendations": [
    {
      "rank": 1,
      "parlay": {
        "legs": [
          {"player_id": 123, "player_name": "Mohamed Salah", "prop": "goals", "line": "over_0.5"},
          {"player_id": 123, "player_name": "Mohamed Salah", "prop": "shots_over_2.5", "line": "over"}
        ],
        "model_probability": 0.089,
        "bookmaker_odds": 7.5,
        "implied_probability": 0.133,
        "edge": 0.044,
        "edge_percent": 4.4,
        "expected_value": 0.668,
        "kelly_stake_percent": 1.1,
        "recommended_stake_percent": 0.275,
        "confidence": "medium"
      }
    },
    {
      "rank": 2,
      "parlay": {
        "legs": [
          {"player_id": 124, "player_name": "Trent Alexander-Arnold", "prop": "assists", "line": "over_0.5"},
          {"player_id": 125, "player_name": "Andrew Robertson", "prop": "assists", "line": "over_0.5"}
        ],
        "model_probability": 0.097,
        "bookmaker_odds": 9.0,
        "implied_probability": 0.111,
        "edge": -0.014,
        "edge_percent": -1.4,
        "expected_value": 0.873,
        "kelly_stake_percent": 0.0,
        "recommended_stake_percent": 0.0,
        "confidence": "low"
      }
    }
    // ... up to 10 recommendations
  ],
  "total_parlays_analyzed": 247,
  "positive_ev_count": 18,
  "model_version": "tier2_v1.0_2024-11-01",
  "inference_time_ms": 843,
  "timestamp": "2024-11-01T10:30:00Z"
}
```

**Latency Target:** <1000ms (p95)

---

### 6. GET /models/info

**Purpose:** Get information about loaded models

**Request:**
```http
GET /models/info
```

**Response (200 OK):**
```json
{
  "models": {
    "tier1": {
      "version": "1.0",
      "filename": "tier1_v1.0_2024-11-01.pkl",
      "loaded_at": "2024-11-01T08:00:00Z",
      "training_date": "2024-11-01",
      "n_players": 856,
      "n_matches": 380,
      "performance": {
        "ece_goals": 0.043,
        "ece_assists": 0.048,
        "ece_shots": 0.051,
        "ece_cards": 0.039
      },
      "convergence": "all_chains_converged",
      "runtime_minutes": 22
    },
    "tier2": {
      "version": "1.0",
      "filename": "tier2_v1.0_2024-11-01.pt",
      "loaded_at": "2024-11-01T08:00:00Z",
      "training_date": "2024-11-01",
      "n_graphs": 500,
      "performance": {
        "correlation_r2": 0.73,
        "parlay_error": 0.08
      },
      "parameters": 1480000
    }
  }
}
```

---

### 7. GET /metrics

**Purpose:** Prometheus metrics endpoint for monitoring

**Request:**
```http
GET /metrics
```

**Response (200 OK):**
```
# HELP prediction_requests_total Total number of prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{endpoint="/predict/player",status="200"} 1523

# HELP prediction_latency_seconds Prediction latency in seconds
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{endpoint="/predict/player",le="0.1"} 1421
prediction_latency_seconds_bucket{endpoint="/predict/player",le="0.5"} 1523

# HELP cache_hit_rate Cache hit rate
# TYPE cache_hit_rate gauge
cache_hit_rate 0.67

# HELP model_inference_errors Total number of model inference errors
# TYPE model_inference_errors counter
model_inference_errors 3
```

---

## Error Handling

### Error Response Format

All errors follow this structure:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": "Optional detailed information",
  "field": "Optional field that caused error",
  "timestamp": "2024-11-01T10:30:00Z",
  "request_id": "abc-123-def-456"
}
```

### Error Codes

| Code | Error Type | Description | Action |
|------|------------|-------------|--------|
| 400 | ValidationError | Invalid request parameters | Fix request |
| 404 | NotFound | Resource not found | Check ID |
| 422 | UnprocessableEntity | Valid format but semantically invalid | Check logic |
| 429 | TooManyRequests | Rate limit exceeded | Wait and retry |
| 500 | InferenceError | Model prediction failed | Retry or report |
| 503 | ServiceUnavailable | Service dependency down | Wait and retry |

---

## Caching Strategy

### What to Cache (Redis)

1. **Player Predictions:**
   - Key: `pred:{tier}:{player_id}:{match_id}`
   - TTL: 5 minutes
   - Invalidate on: lineup change

2. **Match Predictions:**
   - Key: `pred_match:{tier}:{match_id}`
   - TTL: Until lineup confirmed
   - Invalidate on: lineup announcement

3. **Player Features:**
   - Key: `features:{player_id}:{date}`
   - TTL: 1 hour
   - Invalidate on: new match data

4. **Model Metadata:**
   - Key: `model:{tier}:metadata`
   - TTL: 1 day
   - Invalidate on: model update

### Cache Invalidation

**Manual Endpoint:**
```http
POST /cache/invalidate
{
  "keys": ["pred:tier1:123:456"],
  "pattern": "pred_match:*:456"
}
```

**Automatic Triggers:**
- Model reload
- Lineup changes
- Data updates

---

## Request/Response Models (Pydantic)

### PlayerPredictionRequest
```python
from pydantic import BaseModel, Field

class PlayerPredictionRequest(BaseModel):
    player_id: int = Field(..., gt=0, description="Database player ID")
    match_id: int = Field(..., gt=0, description="Database match ID")
    tier: str = Field("tier1", pattern="^(tier1|tier2|ensemble)$")
    return_uncertainty: bool = Field(False)
```

### PropPrediction
```python
class PropPrediction(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    confidence_interval: Optional[List[float]] = Field(None, min_items=2, max_items=2)
    uncertainty: Optional[float] = Field(None, ge=0)
    tier: str
```

### PlayerPredictionResponse
```python
class PlayerPredictionResponse(BaseModel):
    player_id: int
    player_name: str
    team: str
    match_id: int
    opponent: str
    match_date: str
    is_home: bool
    predictions: Dict[str, PropPrediction]
    model_version: str
    cached: bool
    inference_time_ms: int
    timestamp: str
```

---

## Performance Targets

| Endpoint | p50 Latency | p95 Latency | p99 Latency |
|----------|-------------|-------------|-------------|
| /health | 5ms | 10ms | 20ms |
| /predict/player (cached) | 15ms | 50ms | 100ms |
| /predict/player (uncached) | 200ms | 500ms | 1000ms |
| /predict/match | 250ms | 500ms | 1000ms |
| /predict/parlay | 100ms | 200ms | 500ms |
| /recommend/parlays | 500ms | 1000ms | 2000ms |

**SLA:** 99.9% availability (8.7 hours downtime/year)

---

## Rate Limiting (Phase 11)

**Per API Key:**
- 1000 requests/hour
- 100 requests/minute
- Burst: 20 requests/second

**Response Header:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1699012800
```

---

## API Versioning

**URL-based versioning:**
```
/v1/predict/player
/v2/predict/player
```

**Version Lifecycle:**
- v1: Supported for 6 months after v2 release
- v2: Current
- Deprecation notice: 90 days before shutdown

---

## Phase Completion Checklist

- [x] Endpoint design follows REST best practices
- [x] Request/response schemas defined with Pydantic
- [x] Error handling comprehensive and actionable
- [x] Caching strategy defined (what, TTL, invalidation)
- [x] Performance targets set (latency budgets)
- [x] Monitoring endpoint (/metrics) designed
- [x] All critical paths have <1s latency
- [x] Can implement all endpoints in FastAPI

---

## Implementation Notes

**FastAPI Setup:**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis

app = FastAPI(
    title="Player Prop Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

**Endpoint Example:**
```python
@app.post("/predict/player", response_model=PlayerPredictionResponse)
async def predict_player(request: PlayerPredictionRequest):
    # 1. Check cache
    cache_key = f"pred:{request.tier}:{request.player_id}:{request.match_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 2. Load player features
    features = get_player_features(request.player_id, request.match_id)
    
    # 3. Model inference
    predictions = model.predict(features)
    
    # 4. Cache result
    redis_client.setex(cache_key, 300, json.dumps(predictions))
    
    return predictions
```

---

## Next Steps (Phase 4)

1. Implement FastAPI application
2. Create Pydantic request/response models
3. Implement all endpoints
4. Add Redis caching
5. Add error handling
6. Test with curl/Postman
7. Load test with locust