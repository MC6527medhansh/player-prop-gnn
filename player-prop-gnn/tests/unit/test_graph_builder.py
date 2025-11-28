"""
Unit tests for graph construction.
Run with: pytest tests/unit/test_graph_builder.py -v
"""

import pytest
import torch
import numpy as np
from src.models.graph_builder import MatchGraphBuilder


class TestGraphBuilder:
    """Test graph construction from StatsBomb data."""
    
    @pytest.fixture
    def builder(self):
        return MatchGraphBuilder()
    
    def test_builder_initialization(self, builder):
        """Test builder initializes correctly."""
        assert builder is not None
        assert hasattr(builder, 'position_encoding')
        assert len(builder.position_encoding) == 4  # 4 positions
    
    def test_build_graph_structure(self, builder):
        """Test graph has correct structure."""
        # Use World Cup 2018 match (known to exist)
        match_id = 7584  # Example match ID
        
        graph = builder.build_graph_from_match(match_id)
        
        # Check graph attributes
        assert hasattr(graph, 'x')  # Node features
        assert hasattr(graph, 'edge_index')  # Edges
        assert hasattr(graph, 'edge_attr')  # Edge features
        assert hasattr(graph, 'y')  # Labels
        
        # Check dimensions
        assert graph.x.shape[0] == 22  # 22 players
        assert graph.x.shape[1] == 32  # 32 features per player
        assert graph.edge_index.shape[0] == 2  # [source, target]
        
    def test_node_features_range(self, builder):
        """Test node features are normalized."""
        match_id = 7584
        graph = builder.build_graph_from_match(match_id)
        
        # Features should be roughly normalized
        assert torch.isfinite(graph.x).all()  # No NaN/inf
        assert graph.x.min() >= -5  # Roughly normalized
        assert graph.x.max() <= 5
    
    def test_edge_count_reasonable(self, builder):
        """Test edge count is in expected range."""
        match_id = 7584
        graph = builder.build_graph_from_match(match_id)
        
        num_edges = graph.edge_index.shape[1]
        
        # Expected: 40-100 edges (passes + self-loops)
        assert num_edges >= 22  # At least self-loops
        assert num_edges <= 200  # Not too many
    
    def test_labels_present(self, builder):
        """Test labels are extracted."""
        match_id = 7584
        graph = builder.build_graph_from_match(match_id)
        
        # Check label structure
        assert 'goals' in graph.y
        assert 'shots' in graph.y
        assert 'cards' in graph.y
        
        # Check label dimensions
        assert graph.y['goals'].shape == (22, 1)
        assert graph.y['shots'].shape == (22, 1)
        assert graph.y['cards'].shape == (22, 1)


class TestGraphDataset:
    """Test graph dataset creation."""
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        # TODO: Implement after dataset class is created
        pass
    
    def test_dataset_caching(self):
        """Test graphs are cached correctly."""
        # TODO: Implement
        pass
