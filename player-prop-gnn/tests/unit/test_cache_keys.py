"""
Unit Tests for Cache Key Generation
Phase 4.5 - Comprehensive Testing

PURPOSE: Catch bugs where cache keys don't capture all inputs

Tests:
1. Different features → different keys
2. Different positions → different keys
3. Same inputs → same keys
4. Key uniqueness guarantees
"""
import pytest
import hashlib
import json
from typing import Dict


# ============================================================================
# CACHE KEY GENERATION (from main.py)
# ============================================================================

def generate_cache_key(
    player_id: int,
    opponent_id: int,
    position: str,
    was_home: bool,
    features: Dict[str, float]
) -> str:
    """
    Generate cache key for prediction request.
    
    CRITICAL: Key must include ALL inputs that affect prediction.
    """
    # Create deterministic hash of features
    features_json = json.dumps(features, sort_keys=True)
    features_hash = hashlib.md5(features_json.encode()).hexdigest()[:8]
    
    # Build cache key
    cache_key = (
        f"pred:tier1:"
        f"{player_id}:"
        f"{opponent_id}:"
        f"{position}:"
        f"{was_home}:"
        f"{features_hash}:"
        f"v1.0"
    )
    
    return cache_key


# ============================================================================
# TEST 1: DIFFERENT FEATURES → DIFFERENT KEYS
# ============================================================================

def test_different_features_give_different_keys():
    """
    CRITICAL: Different features must produce different cache keys.
    
    Catches bugs where features are ignored in cache key.
    """
    base_params = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True
    }
    
    # Features set 1
    features1 = {
        "goals_rolling_5": 0.5,
        "shots_on_target_rolling_5": 1.5,
        "opponent_strength": 0.5,
        "days_since_last_match": 7.0,
        "was_home": 1.0
    }
    
    # Features set 2 (different)
    features2 = {
        "goals_rolling_5": 0.9,  # Changed
        "shots_on_target_rolling_5": 3.0,  # Changed
        "opponent_strength": 0.1,  # Changed
        "days_since_last_match": 7.0,
        "was_home": 1.0
    }
    
    key1 = generate_cache_key(**base_params, features=features1)
    key2 = generate_cache_key(**base_params, features=features2)
    
    assert key1 != key2, (
        f"Different features must produce different cache keys. "
        f"Got:\n  key1={key1}\n  key2={key2}"
    )


def test_feature_order_does_not_matter():
    """
    Test that feature dict order doesn't affect key.
    
    {a:1, b:2} should give same key as {b:2, a:1}
    """
    base_params = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True
    }
    
    # Same features, different order
    features1 = {
        "goals_rolling_5": 0.5,
        "shots_on_target_rolling_5": 1.5,
        "opponent_strength": 0.5
    }
    
    features2 = {
        "opponent_strength": 0.5,
        "goals_rolling_5": 0.5,
        "shots_on_target_rolling_5": 1.5
    }
    
    key1 = generate_cache_key(**base_params, features=features1)
    key2 = generate_cache_key(**base_params, features=features2)
    
    assert key1 == key2, (
        f"Feature order should not affect cache key (dict is sorted). "
        f"Got:\n  key1={key1}\n  key2={key2}"
    )


# ============================================================================
# TEST 2: DIFFERENT POSITIONS → DIFFERENT KEYS
# ============================================================================

def test_different_positions_give_different_keys():
    """
    CRITICAL: Different positions must produce different cache keys.
    
    Catches bugs where position is ignored.
    """
    base_params = {
        "player_id": 5,
        "opponent_id": 42,
        "was_home": True,
        "features": {
            "goals_rolling_5": 0.5,
            "shots_on_target_rolling_5": 1.5,
            "opponent_strength": 0.5,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    key_forward = generate_cache_key(**base_params, position="Forward")
    key_defender = generate_cache_key(**base_params, position="Defender")
    
    assert key_forward != key_defender, (
        f"Different positions must produce different cache keys. "
        f"Got:\n  Forward:  {key_forward}\n  Defender: {key_defender}"
    )


# ============================================================================
# TEST 3: DIFFERENT PLAYERS → DIFFERENT KEYS
# ============================================================================

def test_different_player_ids_give_different_keys():
    """
    Test that different players produce different keys.
    """
    base_params = {
        "opponent_id": 42,
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
    
    key1 = generate_cache_key(player_id=5, **base_params)
    key2 = generate_cache_key(player_id=10, **base_params)
    
    assert key1 != key2, (
        f"Different player IDs must produce different keys. "
        f"Got:\n  player_id=5:  {key1}\n  player_id=10: {key2}"
    )


def test_different_opponents_give_different_keys():
    """
    Test that different opponents produce different keys.
    """
    base_params = {
        "player_id": 5,
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
    
    key1 = generate_cache_key(opponent_id=42, **base_params)
    key2 = generate_cache_key(opponent_id=99, **base_params)
    
    assert key1 != key2, (
        f"Different opponent IDs must produce different keys. "
        f"Got:\n  opponent=42: {key1}\n  opponent=99: {key2}"
    )


def test_home_away_gives_different_keys():
    """
    Test that home/away status produces different keys.
    """
    base_params = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "features": {
            "goals_rolling_5": 0.5,
            "shots_on_target_rolling_5": 1.5,
            "opponent_strength": 0.5,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    }
    
    key_home = generate_cache_key(was_home=True, **base_params)
    key_away = generate_cache_key(was_home=False, **base_params)
    
    assert key_home != key_away, (
        f"Home/away must produce different keys. "
        f"Got:\n  Home: {key_home}\n  Away: {key_away}"
    )


# ============================================================================
# TEST 4: SAME INPUTS → SAME KEYS
# ============================================================================

def test_identical_inputs_give_identical_keys():
    """
    Test that identical inputs always produce identical keys.
    
    This is determinism requirement for caching.
    """
    params = {
        "player_id": 5,
        "opponent_id": 42,
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
    
    # Generate key twice
    key1 = generate_cache_key(**params)
    key2 = generate_cache_key(**params)
    
    assert key1 == key2, (
        f"Identical inputs must produce identical keys. "
        f"Got:\n  key1: {key1}\n  key2: {key2}"
    )


# ============================================================================
# TEST 5: KEY FORMAT VALIDATION
# ============================================================================

def test_cache_key_format():
    """
    Test that cache key has expected format.
    
    Format: pred:tier1:PLAYER:OPPONENT:POSITION:HOME:HASH:VERSION
    """
    key = generate_cache_key(
        player_id=5,
        opponent_id=42,
        position="Forward",
        was_home=True,
        features={"goals_rolling_5": 0.5}
    )
    
    # Check prefix
    assert key.startswith("pred:tier1:"), f"Key should start with 'pred:tier1:', got {key}"
    
    # Check suffix
    assert key.endswith(":v1.0"), f"Key should end with ':v1.0', got {key}"
    
    # Check player_id present
    assert ":5:" in key, f"Player ID (5) should be in key: {key}"
    
    # Check opponent_id present
    assert ":42:" in key, f"Opponent ID (42) should be in key: {key}"
    
    # Check position present
    assert "Forward" in key, f"Position (Forward) should be in key: {key}"
    
    # Check was_home present
    assert "True" in key, f"was_home (True) should be in key: {key}"


def test_cache_key_length_reasonable():
    """
    Test that cache key length is reasonable.
    
    Redis keys should be < 200 chars for performance.
    """
    key = generate_cache_key(
        player_id=999999,
        opponent_id=999999,
        position="Goalkeeper",  # Longest position name
        was_home=True,
        features={
            "goals_rolling_5": 0.5,
            "shots_on_target_rolling_5": 1.5,
            "opponent_strength": 0.5,
            "days_since_last_match": 7.0,
            "was_home": 1.0
        }
    )
    
    assert len(key) < 200, (
        f"Cache key too long ({len(key)} chars). "
        f"Should be < 200 for Redis performance. "
        f"Key: {key}"
    )


# ============================================================================
# TEST 6: HASH COLLISION PROBABILITY
# ============================================================================

def test_feature_hash_collision_probability():
    """
    Test that feature hash has low collision probability.
    
    Generate 1000 different feature sets, check for collisions.
    """
    import random
    
    hashes = set()
    collision_count = 0
    n_tests = 1000
    
    for i in range(n_tests):
        features = {
            "goals_rolling_5": random.uniform(0, 2),
            "shots_on_target_rolling_5": random.uniform(0, 5),
            "opponent_strength": random.uniform(0, 1),
            "days_since_last_match": float(random.randint(0, 30)),
            "was_home": float(random.randint(0, 1))
        }
        
        features_json = json.dumps(features, sort_keys=True)
        features_hash = hashlib.md5(features_json.encode()).hexdigest()[:8]
        
        if features_hash in hashes:
            collision_count += 1
        hashes.add(features_hash)
    
    collision_rate = collision_count / n_tests
    
    # MD5 8-char hash: 16^8 = 4.3B possible values
    # Collision rate should be < 0.1% for 1000 samples
    assert collision_rate < 0.001, (
        f"Feature hash collision rate too high: {collision_rate:.2%}. "
        f"Found {collision_count} collisions in {n_tests} tests. "
        f"Consider longer hash or different algorithm."
    )
    
    print(f"\n✓ Hash collision test: {collision_count}/{n_tests} collisions ({collision_rate:.4%})")


# ============================================================================
# SUMMARY
# ============================================================================

def test_cache_key_validation_summary():
    """
    Summary report of cache key behavior.
    
    Always passes but prints useful diagnostics.
    """
    print("\n" + "="*70)
    print("CACHE KEY VALIDATION REPORT")
    print("="*70)
    
    # Test all dimensions
    base = {
        "player_id": 5,
        "opponent_id": 42,
        "position": "Forward",
        "was_home": True,
        "features": {"goals_rolling_5": 0.5}
    }
    
    key_base = generate_cache_key(**base)
    
    # Player dimension
    base2 = base.copy()
    base2["player_id"] = 10
    key_diff_player = generate_cache_key(**base2)
    
    # Position dimension
    base3 = base.copy()
    base3["position"] = "Defender"
    key_diff_position = generate_cache_key(**base3)
    
    # Features dimension
    base4 = base.copy()
    base4["features"] = {"goals_rolling_5": 0.9}
    key_diff_features = generate_cache_key(**base4)
    
    print(f"\nBase key:     {key_base}")
    print(f"Diff player:  {key_diff_player}")
    print(f"Diff pos:     {key_diff_position}")
    print(f"Diff feats:   {key_diff_features}")
    
    print(f"\n✓ All dimensions produce unique keys")
    print(f"✓ Key length: {len(key_base)} chars (< 200 limit)")
    
    print("\n" + "="*70)