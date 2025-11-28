"""
Unit tests for EnsemblePredictor and GNNPredictor.
Verifies routing logic, mathematical fusion, and error handling.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from src.models.ensemble import EnsemblePredictor
from src.models.gnn_inference import GNNPredictor

# ==========================================
# 1. FIXTURES (Mocking the complex models)
# ==========================================

@pytest.fixture
def mock_tier1():
    """Mock Tier 1 (Bayesian) predictor."""
    predictor = MagicMock()
    # Setup a standard response
    predictor.predict_player.return_value = {
        'goals': {'mean': 0.3, 'std': 0.1, 'probability': {'0': 0.74, '1': 0.22, '2+': 0.04}, 'median': 0.0},
        'shots': {'mean': 2.0, 'std': 1.0, 'probability': {'0': 0.1, '1': 0.2, '2+': 0.7}, 'median': 2.0},
        'cards': {'mean': 0.1, 'std': 0.05, 'probability': {'0': 0.9, '1': 0.1}, 'median': 0.0},
        'metadata': {'model': 'Bayesian', 'inference_time_ms': 10}
    }
    return predictor

@pytest.fixture
def mock_tier2():
    """Mock Tier 2 (GNN) predictor."""
    predictor = MagicMock()
    # Setup a different response to verify averaging
    predictor.predict_player.return_value = {
        'goals': {'mean': 0.5, 'std': 0.2, 'probability': {'0': 0.6, '1': 0.3, '2+': 0.1}, 'median': 0.0},
        'shots': {'mean': 3.0, 'std': 1.5, 'probability': {'0': 0.05, '1': 0.15, '2+': 0.8}, 'median': 3.0},
        'cards': {'mean': 0.2, 'std': 0.1, 'probability': {'0': 0.8, '1': 0.2}, 'median': 0.0},
        'metadata': {'model': 'GNN', 'graph_nodes': 22}
    }
    return predictor

# ==========================================
# 2. ENSEMBLE LOGIC TESTS
# ==========================================

class TestEnsemblePredictor:
    
    def test_initialization(self, mock_tier1, mock_tier2):
        """Prove components load and weights are assigned."""
        ensemble = EnsemblePredictor(mock_tier1, mock_tier2, tier1_weight=0.7, tier2_weight=0.3)
        assert ensemble.w1 == 0.7
        assert ensemble.w2 == 0.3

    def test_cold_start_routing(self, mock_tier1, mock_tier2):
        """
        PROVE: New players (<5 games) bypass GNN.
        """
        ensemble = EnsemblePredictor(mock_tier1, mock_tier2)
        
        result = ensemble.predict_player(
            player_id=999, opponent_id=1, position="Forward", was_home=True, 
            features={}, games_played=2  # < 5 games
        )
        
        # Check routing
        mock_tier1.predict_player.assert_called_once()
        mock_tier2.predict_player.assert_not_called()
        
        # Check metadata
        assert result['metadata']['model_mode'] == 'tier1_only'
        assert result['metadata']['ensemble_reason'] == 'insufficient_history'

    def test_circuit_breaker(self, mock_tier1, mock_tier2):
        """
        PROVE: System survives GNN crash.
        """
        ensemble = EnsemblePredictor(mock_tier1, mock_tier2)
        
        # Simulate GNN crash
        mock_tier2.predict_player.side_effect = RuntimeError("Graph construction failed")
        
        result = ensemble.predict_player(
            player_id=10, opponent_id=1, position="Forward", was_home=True, 
            features={}, games_played=50
        )
        
        # Verify fallback
        assert result['metadata']['model_mode'] == 'tier1_only'
        assert "gnn_error" in result['metadata']['ensemble_reason']
        # The goals prediction should match Tier 1 exactly (0.3)
        assert result['goals']['mean'] == 0.3 

    def test_weighted_fusion(self, mock_tier1, mock_tier2):
        """
        PROVE: Predictions are mathematically combined.
        """
        ensemble = EnsemblePredictor(mock_tier1, mock_tier2, tier1_weight=0.6, tier2_weight=0.4)
        
        result = ensemble.predict_player(
            player_id=10, opponent_id=1, position="Forward", was_home=True, 
            features={}, games_played=50
        )
        
        # Check Goals Mean: (0.6 * 0.3) + (0.4 * 0.5) = 0.18 + 0.20 = 0.38
        expected_goals = (0.6 * 0.3) + (0.4 * 0.5)
        assert result['goals']['mean'] == pytest.approx(expected_goals)
        
        # Check Metadata
        assert result['metadata']['model_mode'] == 'ensemble_weighted'
        assert result['metadata']['weights']['tier1'] == 0.6

# ==========================================
# 3. GNN INFERENCE TESTS
# ==========================================

class TestGNNPredictor:
    
    @patch('src.models.gnn_inference.Path')
    @patch('src.models.gnn_inference.PlayerPropGAT')
    @patch('src.models.gnn_inference.MatchGraphBuilder')
    @patch('torch.load') 
    def test_gnn_formatting(self, mock_load, mock_builder_cls, mock_gat_cls, mock_path):
        """
        PROVE: GNN outputs are correctly converted to Probabilities.
        """
        # CRITICAL FIX 1: Make Path().exists() return True (Bypass FileNotFoundError)
        mock_path.return_value.exists.return_value = True
        
        # Setup Mocks
        mock_model = mock_gat_cls.return_value
        mock_builder = mock_builder_cls.return_value
        
        # Mock Graph Data
        mock_graph = MagicMock()
        mock_graph.x = "features"
        mock_graph.edge_index = "edges"
        mock_graph.edge_attr = "attr"
        mock_graph.num_nodes = 22
        mock_graph.num_edges = 100
        
        # CRITICAL FIX 2: Correctly mock the tensor comparison result
        comparison_tensor_mock = MagicMock()
        # Mock the .nonzero() result to return a tensor with index 0 (as expected by the code)
        comparison_tensor_mock.nonzero.return_value = (torch.tensor([0]),) 
        
        # Mock the '==' operation to return our comparison_tensor_mock
        mock_graph.player_ids = MagicMock()
        mock_graph.player_ids.__eq__.return_value = comparison_tensor_mock
        
        # Mock Graph Builder (it returns the mock graph object)
        mock_builder.build_graph.return_value = mock_graph 

        # Mock Model Output (Tensors)
        mock_model.return_value = {
            'goals': MagicMock(item=lambda: 0.5),   # Mean rate 0.5
            'shots': MagicMock(item=lambda: 2.5),
            'cards': MagicMock(item=lambda: 0.1)    # Probability 0.1
        }
        
        # Initialize (db_config passed to satisfy GNNPredictor constructor)
        predictor = GNNPredictor("dummy_path.pt", db_config={})
        predictor.model = mock_model # Force mock model
        
        # Run
        result = predictor.predict_player(10, 5, "Forward", True)
        
        # Verify Poisson Calculation for Goals (mean=0.5)
        # P(0) = e^-0.5 ≈ 0.6065
        assert result['goals']['mean'] == 0.5
        assert result['goals']['probability']['0'] == pytest.approx(0.6065, abs=0.001)
        
        # Verify GraphBuilder was called (testing the caching interface)
        mock_builder.build_graph.assert_called_once()