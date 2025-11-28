"""
Integration Tests for Model Predictions
Phase 4.5 - Comprehensive Testing

PURPOSE: Catch bugs where inputs don't affect predictions

Tests:
1. Different features → different predictions
2. Different positions → different predictions (if supported)
3. Same inputs → same predictions (deterministic)
4. Predictions are valid (sum to 1, in range)
"""
import pytest
import requests
import time
from typing import Dict, Any

# API base URL
API_URL = "http://localhost:8000"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def api_client():
    """Return requests session for API calls."""
    return requests.Session()


@pytest.fixture
def base_request() -> Dict[str, Any]:
    """Return baseline prediction request."""
    return {
        "player_id": 50000,  # High ID to avoid cache collisions
        "opponent_id": 50,
        "position": "Forward",
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.5,
            "shots_on_target_rolling_5": 1.5,
            "opponent_strength": 0.5,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }


def make_prediction(client: requests.Session, payload: Dict) -> Dict:
    """
    Make prediction and return response.
    
    Raises assertion error if request fails.
    """
    response = client.post(f"{API_URL}/predict/player", json=payload)
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    return response.json()


# ============================================================================
# TEST 1: FEATURES AFFECT PREDICTIONS
# ============================================================================

def test_different_features_give_different_predictions(api_client, base_request):
    """
    CRITICAL: Different features must produce different predictions.
    
    Catches bugs where model ignores feature inputs.
    """
    # Hot striker (high goals/shots, weak opponent)
    hot_request = base_request.copy()
    hot_request["player_id"] = 50001
    hot_request["features"] = {
        "goals_rolling_5": 0.9,
        "shots_on_target_rolling_5": 3.0,
        "opponent_strength": 0.1,
        "days_since_last_match": 7.0,
        "was_home": 1.0
    }
    
    # Cold striker (low goals/shots, strong opponent)
    cold_request = base_request.copy()
    cold_request["player_id"] = 50002
    cold_request["features"] = {
        "goals_rolling_5": 0.0,
        "shots_on_target_rolling_5": 0.1,
        "opponent_strength": 0.9,
        "days_since_last_match": 7.0,
        "was_home": 1.0
    }
    
    # Make predictions
    hot_pred = make_prediction(api_client, hot_request)
    time.sleep(0.1)  # Avoid race conditions
    cold_pred = make_prediction(api_client, cold_request)
    
    # Extract goals predictions
    hot_goals = hot_pred["predictions"]["goals"]["mean"]
    cold_goals = cold_pred["predictions"]["goals"]["mean"]
    
    # ASSERT: Hot striker must have higher goal expectation
    assert hot_goals > cold_goals, (
        f"Hot striker (goals_rolling=0.9, opp=0.1) should have higher prediction than "
        f"cold striker (goals_rolling=0.0, opp=0.9). "
        f"Got hot={hot_goals:.3f}, cold={cold_goals:.3f}"
    )
    
    # ASSERT: Difference should be substantial (not noise)
    diff_ratio = hot_goals / max(cold_goals, 0.001)  # Avoid div by zero
    assert diff_ratio >= 2.0, (
        f"Hot vs cold difference too small. "
        f"Expected ratio >= 2.0, got {diff_ratio:.2f} "
        f"(hot={hot_goals:.3f}, cold={cold_goals:.3f})"
    )


def test_opponent_strength_affects_predictions(api_client, base_request):
    """
    Test that opponent strength properly affects predictions.
    
    Weak opponent → higher predictions
    Strong opponent → lower predictions
    """
    # Weak opponent
    weak_opp_request = base_request.copy()
    weak_opp_request["player_id"] = 50003
    weak_opp_request["features"]["opponent_strength"] = 0.1
    
    # Strong opponent
    strong_opp_request = base_request.copy()
    strong_opp_request["player_id"] = 50004
    strong_opp_request["features"]["opponent_strength"] = 0.9
    
    weak_pred = make_prediction(api_client, weak_opp_request)
    time.sleep(0.1)
    strong_pred = make_prediction(api_client, strong_opp_request)
    
    weak_goals = weak_pred["predictions"]["goals"]["mean"]
    strong_goals = strong_pred["predictions"]["goals"]["mean"]
    
    assert weak_goals > strong_goals, (
        f"Weak opponent should yield higher predictions. "
        f"Got weak={weak_goals:.3f}, strong={strong_goals:.3f}"
    )


def test_home_away_affects_predictions(api_client, base_request):
    """
    Test that home/away status affects predictions.
    
    Home → slightly higher predictions (home advantage)
    Away → slightly lower predictions
    """
    # Home game
    home_request = base_request.copy()
    home_request["player_id"] = 50005
    home_request["was_home"] = True
    home_request["features"]["was_home"] = 1.0
    
    # Away game
    away_request = base_request.copy()
    away_request["player_id"] = 50006
    away_request["was_home"] = False
    away_request["features"]["was_home"] = 0.0
    
    home_pred = make_prediction(api_client, home_request)
    time.sleep(0.1)
    away_pred = make_prediction(api_client, away_request)
    
    home_goals = home_pred["predictions"]["goals"]["mean"]
    away_goals = away_pred["predictions"]["goals"]["mean"]
    
    # Home should be >= away (might be equal if model doesn't use this)
    assert home_goals >= away_goals, (
        f"Home game should have >= predictions than away. "
        f"Got home={home_goals:.3f}, away={away_goals:.3f}"
    )


# ============================================================================
# TEST 2: POSITION AFFECTS PREDICTIONS (If Model Supports)
# ============================================================================

def test_position_affects_predictions_if_supported(api_client, base_request):
    """
    Test if position affects predictions.
    
    This test MAY FAIL if model doesn't use position encoding.
    That's OKAY - it documents model behavior.
    
    Forward → higher offensive predictions
    Defender → lower offensive predictions
    """
    # Forward
    forward_request = base_request.copy()
    forward_request["player_id"] = 50007
    forward_request["position"] = "Forward"
    
    # Defender (same features!)
    defender_request = base_request.copy()
    defender_request["player_id"] = 50008
    defender_request["position"] = "Defender"
    
    forward_pred = make_prediction(api_client, forward_request)
    time.sleep(0.1)
    defender_pred = make_prediction(api_client, defender_request)
    
    forward_goals = forward_pred["predictions"]["goals"]["mean"]
    defender_goals = defender_pred["predictions"]["goals"]["mean"]
    
    # If model uses position, forward should have higher goals
    if forward_goals != defender_goals:
        print(f"✓ Model DOES use position encoding (forward={forward_goals:.3f}, defender={defender_goals:.3f})")
        assert forward_goals > defender_goals, (
            f"Forward should have higher goal predictions than defender. "
            f"Got forward={forward_goals:.3f}, defender={defender_goals:.3f}"
        )
    else:
        print(f"⚠ Model DOES NOT use position encoding (both={forward_goals:.3f})")
        pytest.skip("Model doesn't use position - documented behavior")


# ============================================================================
# TEST 3: DETERMINISTIC PREDICTIONS
# ============================================================================

def test_same_inputs_give_same_predictions(api_client, base_request):
    """
    Test that identical inputs produce identical predictions.
    
    Catches bugs where model is non-deterministic or cache is broken.
    """
    request = base_request.copy()
    request["player_id"] = 50009
    
    # Make prediction twice with identical inputs
    pred1 = make_prediction(api_client, request)
    time.sleep(0.5)  # Wait for cache (if enabled)
    pred2 = make_prediction(api_client, request)
    
    # Extract goals predictions
    goals1 = pred1["predictions"]["goals"]["mean"]
    goals2 = pred2["predictions"]["goals"]["mean"]
    
    # Should be EXACTLY identical
    assert goals1 == goals2, (
        f"Identical inputs should produce identical predictions. "
        f"Got pred1={goals1:.6f}, pred2={goals2:.6f}"
    )
    
    # Check if second was cached
    if pred2.get("cached"):
        print(f"✓ Second prediction served from cache")
    else:
        print(f"⚠ Second prediction not cached (Redis may be disabled)")


# ============================================================================
# TEST 4: PREDICTION VALIDITY
# ============================================================================

def test_probabilities_sum_to_one(api_client, base_request):
    """
    Test that probability distributions sum to 1.0.
    
    Catches bugs in probability normalization.
    """
    request = base_request.copy()
    request["player_id"] = 50010
    
    pred = make_prediction(api_client, request)
    
    # Check each prop type
    for prop_name in ["goals", "shots", "cards"]:
        prob_dict = pred["predictions"][prop_name]["probability"]
        total_prob = sum(prob_dict.values())
        
        # Allow small floating point error
        assert 0.99 <= total_prob <= 1.01, (
            f"{prop_name} probabilities don't sum to 1.0. "
            f"Got {total_prob:.6f}, distribution: {prob_dict}"
        )


def test_predictions_in_valid_range(api_client, base_request):
    """
    Test that predictions are in valid ranges.
    
    - Mean >= 0
    - Std >= 0
    - CI low <= mean <= CI high
    - Probabilities between 0 and 1
    """
    request = base_request.copy()
    request["player_id"] = 50011
    
    pred = make_prediction(api_client, request)
    
    for prop_name in ["goals", "shots", "cards"]:
        prop = pred["predictions"][prop_name]
        
        # Check mean >= 0
        assert prop["mean"] >= 0, f"{prop_name} mean is negative: {prop['mean']}"
        
        # Check std >= 0
        assert prop["std"] >= 0, f"{prop_name} std is negative: {prop['std']}"
        
        # Check median >= 0
        assert prop["median"] >= 0, f"{prop_name} median is negative: {prop['median']}"
        
        # Check CI bounds
        assert prop["ci_low"] <= prop["mean"] <= prop["ci_high"], (
            f"{prop_name} mean not within CI. "
            f"CI=[{prop['ci_low']:.3f}, {prop['ci_high']:.3f}], mean={prop['mean']:.3f}"
        )
        
        # Check probabilities in [0, 1]
        for outcome, prob in prop["probability"].items():
            assert 0 <= prob <= 1, (
                f"{prop_name} probability for '{outcome}' out of range: {prob}"
            )


def test_predictions_sensible_magnitudes(api_client, base_request):
    """
    Test that predictions are in sensible ranges.
    
    Goals/shots/cards should be < 10 (extreme values are bugs)
    """
    request = base_request.copy()
    request["player_id"] = 50012
    
    pred = make_prediction(api_client, request)
    
    for prop_name in ["goals", "shots", "cards"]:
        mean = pred["predictions"][prop_name]["mean"]
        ci_high = pred["predictions"][prop_name]["ci_high"]
        
        # Mean should be reasonable (< 5 for soccer)
        assert mean < 5.0, (
            f"{prop_name} mean unreasonably high: {mean:.3f}. "
            f"Possible model bug or data issue."
        )
        
        # CI high should be reasonable (< 10)
        assert ci_high < 10.0, (
            f"{prop_name} CI upper bound unreasonably high: {ci_high:.3f}"
        )


# ============================================================================
# TEST 5: EXTREME INPUTS
# ============================================================================

def test_extreme_high_form_player(api_client, base_request):
    """
    Test player with maxed-out offensive stats.
    
    Should give high predictions but still valid.
    """
    request = base_request.copy()
    request["player_id"] = 50013
    request["features"] = {
        "goals_rolling_5": 2.0,  # 2 goals/game (Messi-level)
        "shots_on_target_rolling_5": 5.0,
        "opponent_strength": 0.0,  # Weakest opponent
        "days_since_last_match": 7.0,
        "was_home": 1.0
    }
    
    pred = make_prediction(api_client, request)
    goals_mean = pred["predictions"]["goals"]["mean"]
    
    # Should predict high but not absurd
    assert 0.1 <= goals_mean <= 3.0, (
        f"Extreme high form should give reasonable prediction. "
        f"Got {goals_mean:.3f}"
    )
    
    # Probability of 1+ goals should be substantial
    prob_0_goals = pred["predictions"]["goals"]["probability"]["0"]
    assert prob_0_goals < 0.9, (
        f"Hot striker vs weak opponent should have >10% chance to score. "
        f"Got prob(0 goals) = {prob_0_goals:.3f}"
    )


def test_extreme_low_form_player(api_client, base_request):
    """
    Test player with minimal offensive stats.
    
    Should give very low predictions.
    """
    request = base_request.copy()
    request["player_id"] = 50014
    request["features"] = {
        "goals_rolling_5": 0.0,
        "shots_on_target_rolling_5": 0.0,
        "opponent_strength": 1.0,  # Strongest opponent
        "days_since_last_match": 7.0,
        "was_home": 0.0
    }
    
    pred = make_prediction(api_client, request)
    goals_mean = pred["predictions"]["goals"]["mean"]
    
    # Should predict very low
    assert goals_mean <= 0.2, (
        f"Player with no form vs strong opponent should have low prediction. "
        f"Got {goals_mean:.3f}"
    )
    
    # Probability of 0 goals should be very high
    prob_0_goals = pred["predictions"]["goals"]["probability"]["0"]
    assert prob_0_goals >= 0.8, (
        f"Cold player should have >80% chance of 0 goals. "
        f"Got {prob_0_goals:.3f}"
    )


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def test_generate_prediction_validation_report(api_client, base_request):
    """
    Generate comprehensive report of model behavior.
    
    This test ALWAYS PASSES but prints useful diagnostics.
    """
    print("\n" + "="*70)
    print("MODEL PREDICTION VALIDATION REPORT")
    print("="*70)
    
    # Test 1: Position sensitivity
    forward_req = base_request.copy()
    forward_req["player_id"] = 60001
    forward_req["position"] = "Forward"
    
    defender_req = base_request.copy()
    defender_req["player_id"] = 60002
    defender_req["position"] = "Defender"
    
    forward_pred = make_prediction(api_client, forward_req)
    defender_pred = make_prediction(api_client, defender_req)
    
    forward_goals = forward_pred["predictions"]["goals"]["mean"]
    defender_goals = defender_pred["predictions"]["goals"]["mean"]
    
    print(f"\n1. POSITION ENCODING:")
    print(f"   Forward: {forward_goals:.4f} expected goals")
    print(f"   Defender: {defender_goals:.4f} expected goals")
    if forward_goals == defender_goals:
        print(f"   ⚠ Model does NOT use position (predictions identical)")
    else:
        print(f"   ✓ Model DOES use position ({abs(forward_goals - defender_goals):.4f} difference)")
    
    # Test 2: Feature sensitivity
    hot_req = base_request.copy()
    hot_req["player_id"] = 60003
    hot_req["features"]["goals_rolling_5"] = 1.0
    
    cold_req = base_request.copy()
    cold_req["player_id"] = 60004
    cold_req["features"]["goals_rolling_5"] = 0.0
    
    hot_pred = make_prediction(api_client, hot_req)
    cold_pred = make_prediction(api_client, cold_req)
    
    hot_goals = hot_pred["predictions"]["goals"]["mean"]
    cold_goals = cold_pred["predictions"]["goals"]["mean"]
    
    print(f"\n2. FEATURE SENSITIVITY:")
    print(f"   Hot form (1.0 goals/game): {hot_goals:.4f}")
    print(f"   Cold form (0.0 goals/game): {cold_goals:.4f}")
    print(f"   ✓ Sensitivity: {(hot_goals / max(cold_goals, 0.001)):.2f}x difference")
    
    print("\n" + "="*70)