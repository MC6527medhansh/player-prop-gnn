"""
API Router for Parlay Recommendations.
Handles requests to find value bets using the GNN Correlation Engine.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging

# Import our engines
from src.models.parlay_pricing import ParlayPricer

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependencies ---
# Initialize the Pricer once when the API starts
try:
    _PRICER = ParlayPricer()
    logger.info("✅ Parlay Pricer initialized in API")
except Exception as e:
    logger.error(f"❌ Failed to initialize Parlay Pricer: {e}")
    _PRICER = None

# --- Schemas ---
class ParlaySelection(BaseModel):
    prop: str  # 'goals', 'shots', 'assists', 'cards'
    # In a full production app, we would look up probabilities from Tier 1.
    # For this flexible endpoint, the user/frontend provides the Tier 1 marginals.
    prob: float 

class ParlayRequest(BaseModel):
    player_id: int
    bookmaker_odds: float  # Decimal odds (e.g. 5.50)
    selections: List[ParlaySelection]

class ParlayResponse(BaseModel):
    independent_prob: float
    correlated_prob: float
    fair_odds: float
    edge: float
    recommendation: str
    lift_factor: float

# --- Endpoints ---

@router.post("/price", response_model=ParlayResponse)
async def price_specific_parlay(request: ParlayRequest):
    """
    Price a specific parlay combination provided by the user.
    """
    if not _PRICER:
        raise HTTPException(status_code=503, detail="Pricing engine not available")

    # 1. Convert request to format expected by ParlayPricer
    legs = []
    for sel in request.selections:
        legs.append({'prop': sel.prop, 'prob': sel.prob})

    # 2. Run Pricing Engine (Tier 2)
    result = _PRICER.price_parlay(legs)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    # 3. Calculate Edge against Bookmaker
    my_prob = result['correlated_prob']
    bookie_prob = 1.0 / request.bookmaker_odds
    edge = my_prob - bookie_prob
    
    # Recommendation Logic
    # 5% edge is a standard threshold for "Bet"
    rec = "BET" if edge > 0.05 else "PASS"

    return {
        "independent_prob": result['independent_prob'],
        "correlated_prob": my_prob,
        "fair_odds": result['fair_odds'],
        "edge": edge,
        "recommendation": rec,
        "lift_factor": result['correlation_factor']
    }

@router.get("/health")
async def parlay_health():
    """Check if the correlation matrix is loaded."""
    return {
        "status": "active", 
        "matrix_loaded": _PRICER is not None
    }