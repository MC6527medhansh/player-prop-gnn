from setuptools import setup, find_packages

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
