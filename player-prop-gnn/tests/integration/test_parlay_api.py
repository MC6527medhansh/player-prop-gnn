"""
Integration Tests for Parlay Pricing API.
Verifies the endpoint logic and math correctness.
"""
import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# --- FIX IMPORT PATH ---
# Get the absolute path of the current file
current_file = Path(__file__).resolve()
# Go up two levels: tests/integration -> tests -> project_root
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app

# Initialize test client
client = TestClient(app)

def test_parlay_pricing_endpoint():
    """
    Test the /parlay/price endpoint with a standard correlation scenario.
    Goals and Shots should be positively correlated.
    """
    print("\n🚀 Testing POST /parlay/price ...")
    
    payload = {
        "player_id": 123,
        "bookmaker_odds": 5.0, # +400 (Implied 20%)
        "selections": [
            {"prop": "goals", "prob": 0.30},
            {"prop": "shots", "prob": 0.50}
        ]
    }
    
    response = client.post("/parlay/price", json=payload)
    
    # 1. Status Check
    if response.status_code != 200:
        print(f"❌ API Failed with {response.status_code}: {response.text}")
        assert response.status_code == 200
    
    data = response.json()
    
    # 2. Logic Check (Lift Factor)
    # Independence: 0.30 * 0.50 = 0.15
    independent_prob = 0.30 * 0.50
    assert data['independent_prob'] == pytest.approx(independent_prob, abs=0.001)
    
    print(f"  - Independent Prob: {data['independent_prob']:.4f}")
    print(f"  - Correlated Prob:  {data['correlated_prob']:.4f}")
    print(f"  - Lift Factor:      {data['lift_factor']:.2f}x")
    
    # Note: Even if correlation is weak, it shouldn't be drastically lower for positive events
    assert data['correlated_prob'] >= data['independent_prob'] * 0.95, "Correlated prob suspiciously low"
    
    print(f"✅ Success: API returned valid pricing.")

def test_parlay_bad_input():
    """Test error handling for invalid input."""
    payload = {
        "player_id": 123,
        "bookmaker_odds": 5.0,
        "selections": [] # Empty list should fail or return error
    }
    response = client.post("/parlay/price", json=payload)
    
    # Expecting 400 (Bad Request) or 422 (Validation Error)
    assert response.status_code in [400, 422]
    print("✅ Success: API correctly rejected invalid input.")

if __name__ == "__main__":
    # Allow running directly for quick check
    try:
        test_parlay_pricing_endpoint()
        test_parlay_bad_input()
        print("\n🎉 ALL TESTS PASSED")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")