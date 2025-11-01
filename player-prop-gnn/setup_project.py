"""
Setup script for Player Prop GNN project.
Creates the complete directory structure and placeholder files.
"""
import os
from pathlib import Path


def create_directory_structure():
    """Create the complete project directory structure."""
    
    # Define the structure
    directories = [
        # Documentation
        "docs",
        
        # Data directories
        "data/raw",
        "data/processed",
        "data/external",
        "data/schemas",
        
        # Source code
        "src/data",
        "src/models",
        "src/api",
        "src/dashboard",
        "src/utils",
        "src/config",
        
        # Tests
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/fixtures",
        
        # Notebooks
        "notebooks/exploration",
        "notebooks/analysis",
        
        # Deployment
        "deployment/docker",
        "deployment/scripts",
        
        # Monitoring
        "monitoring/dashboards",
        
        # Models and logs (created by settings.py but added here for completeness)
        "models",
        "logs",
    ]
    
    # Create all directories
    print("Creating directory structure...")
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
        
        # Add .gitkeep to empty directories that should be tracked
        if directory in ["data/raw", "data/processed", "data/external", "logs"]:
            gitkeep = path / ".gitkeep"
            gitkeep.touch(exist_ok=True)
    
    print("\n✅ Directory structure created successfully!")


def create_init_files():
    """Create __init__.py files for Python packages."""
    
    packages = [
        "src",
        "src/data",
        "src/models",
        "src/api",
        "src/dashboard",
        "src/utils",
        "src/config",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/fixtures",
    ]
    
    print("\nCreating __init__.py files...")
    for package in packages:
        init_file = Path(package) / "__init__.py"
        init_file.touch(exist_ok=True)
        print(f"  ✓ {init_file}")
    
    print("\n✅ Python package files created!")


def create_placeholder_docs():
    """Create placeholder documentation files."""
    
    docs = {
        "docs/data_schema.md": """# Data Schema Documentation

## Overview
Document your database schema here.

## Tables

### players
- player_id (PK)
- name
- position
- team_id (FK)

### matches
- match_id (PK)
- home_team_id (FK)
- away_team_id (FK)
- date
- competition

### player_stats
- stat_id (PK)
- player_id (FK)
- match_id (FK)
- goals
- assists
- shots_on_target
- cards

## Relationships
Document table relationships here.
""",
        
        "docs/api_spec.md": """# API Specification

## Endpoints

### GET /health
Health check endpoint.

### POST /predict/player
Predict props for a single player.

**Request:**
```json
{
  "player_id": "123",
  "match_id": "456",
  "props": ["goals", "assists"]
}
```

**Response:**
```json
{
  "predictions": {
    "goals": {"probability": 0.23, "credible_interval": [0.15, 0.32]},
    "assists": {"probability": 0.18, "credible_interval": [0.10, 0.28]}
  }
}
```

### POST /predict/match
Predict props for all players in a match.
""",
        
        "docs/model_architecture.md": """# Tier 1 Model Architecture

## Bayesian Multi-Task Model

### Priors
Document your prior choices here.

### Likelihood
Document your likelihood function.

### Inference
MCMC with NUTS sampler.
""",
        
        "docs/gnn_architecture.md": """# Tier 2 GNN Architecture

## Graph Construction
Describe how you build graphs from matches.

## Model Architecture
- Input: Node features (player stats + match context)
- Hidden: 3-layer GAT with 8 attention heads
- Output: Joint probability distributions
""",
        
        "docs/deployment_guide.md": """# Deployment Guide

## Prerequisites
- Docker
- Docker Compose

## Steps

1. Clone repository
2. Copy .env.example to .env
3. Run `docker-compose up -d`
4. Access API at http://localhost:8000
""",
    }
    
    print("\nCreating placeholder documentation files...")
    for filepath, content in docs.items():
        path = Path(filepath)
        if not path.exists():
            path.write_text(content)
            print(f"  ✓ {filepath}")
    
    print("\n✅ Documentation placeholders created!")


def create_pytest_config():
    """Create pytest configuration file."""
    
    pytest_ini = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""
    
    path = Path("pytest.ini")
    if not path.exists():
        path.write_text(pytest_ini)
        print("\n✅ pytest.ini created!")


def create_setup_py():
    """Create setup.py for package installation."""
    
    setup_py = """from setuptools import setup, find_packages

setup(
    name="player-prop-gnn",
    version="0.1.0",
    description="Player prop prediction with Bayesian models and GNN correlation modeling",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies from requirements.txt
        # Add key dependencies here or use requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.10.1",
            "flake8>=6.1.0",
            "mypy>=1.6.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "player-prop-train=src.models.train:main",
            "player-prop-api=src.api.main:main",
        ],
    },
)
"""
    
    path = Path("setup.py")
    if not path.exists():
        path.write_text(setup_py)
        print("✅ setup.py created!")


def main():
    """Run all setup functions."""
    print("=" * 60)
    print("Player Prop GNN - Project Setup")
    print("=" * 60)
    print()
    
    create_directory_structure()
    create_init_files()
    create_placeholder_docs()
    create_pytest_config()
    create_setup_py()
    
    print("\n" + "=" * 60)
    print("Setup Complete! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Copy .env.example to .env and configure")
    print("5. Initialize git: git init")
    print("6. Make first commit: git add . && git commit -m 'Initial project setup'")
    print("\nRefer to PHASE_00_Foundation.md for detailed next steps!")


if __name__ == "__main__":
    main()