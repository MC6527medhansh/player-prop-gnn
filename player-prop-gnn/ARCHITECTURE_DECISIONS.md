# Architecture Decision Records

## Overview
This document tracks all major technical decisions made during the project. Each decision includes context, alternatives considered, and validation criteria.

---

## ADR-001: PyMC3 for Bayesian Tier 1 Modeling
- **Date**: 2025-10-31
- **Status**: Proposed
- **Context**: Need a Bayesian framework for multi-task learning with uncertainty quantification. Must output full posterior distributions, not just point estimates. Need to model 4 correlated props (goals, assists, shots, cards).
- **Decision**: Use PyMC3 for Tier 1 Bayesian modeling
- **Alternatives Considered**: 
  - Stan: More flexible but steeper learning curve, harder to debug
  - TensorFlow Probability: Better for production but less interpretable priors
  - Edward2: Too experimental, less community support
- **Consequences**: 
  - Pro: Excellent documentation, easy prior specification, great diagnostics (R-hat, ESS)
  - Pro: Integrates well with ArviZ for posterior analysis
  - Con: Can be slow for large datasets (may need variational inference)
  - Con: Harder to deploy than pure PyTorch models
- **Validation Criteria**: 
  - Models converge with R-hat < 1.01
  - Can explain every prior choice
  - Inference time < 1s per player on CPU
  - Easy to extract calibrated probabilities

---

## ADR-002: PyTorch Geometric for GNN Implementation
- **Date**: 2025-10-31
- **Status**: Proposed
- **Context**: Need to model player-player correlations using graph neural networks. Match state is naturally a graph (players as nodes, interactions as edges). Need flexibility to experiment with different GNN architectures (GCN, GAT, GraphSAGE).
- **Decision**: Use PyTorch Geometric (PyG) for Tier 2 GNN modeling
- **Alternatives Considered**:
  - DGL (Deep Graph Library): Good but less intuitive API
  - Pure PyTorch: Too much boilerplate for graph operations
  - TensorFlow-GNN: Less mature ecosystem
- **Consequences**:
  - Pro: Rich set of pre-built GNN layers (GCN, GAT, etc.)
  - Pro: Efficient sparse tensor operations
  - Pro: Easy to customize message passing
  - Con: Installation can be tricky (CUDA dependencies)
  - Con: Documentation has gaps for advanced use cases
- **Validation Criteria**:
  - Can build and train graph models efficiently
  - Easy to visualize attention weights for interpretability
  - Inference time < 100ms per match graph
  - Memory usage reasonable (< 2GB per model)

---

## ADR-003: PostgreSQL + Redis for Data Layer
- **Date**: 2025-10-31
- **Status**: Proposed
- **Context**: Need to store historical player stats, match events, and model predictions. Need fast lookups for API (player stats for upcoming match). Some queries are expensive (aggregating season stats), need caching.
- **Decision**: PostgreSQL for persistence, Redis for caching
- **Alternatives Considered**:
  - SQLite: Too simple for production-like project
  - MongoDB: Overkill, SQL is more familiar to most teams
  - Pure Redis: No persistence guarantees
- **Consequences**:
  - Pro: PostgreSQL has excellent JSON support for flexible schema
  - Pro: Redis provides sub-millisecond lookups
  - Pro: Both have great Python libraries (psycopg2, redis-py)
  - Con: More infrastructure to manage (two databases)
  - Con: Cache invalidation logic can be complex
- **Validation Criteria**:
  - API latency < 100ms p95 with cache hits
  - Can recover from Redis failure gracefully
  - Database schema supports all queries without N+1 problems
  - Easy to backup and restore data

---

## ADR-004: FastAPI for REST API
- **Date**: 2025-10-31
- **Status**: Proposed
- **Context**: Need API to serve predictions to dashboard and potential external consumers. Must be async for good performance. Need automatic API documentation for interviews/demos.
- **Decision**: Use FastAPI with Pydantic validation
- **Alternatives Considered**:
  - Flask: Too basic, no async support, manual validation
  - Django REST: Overkill for this project, too opinionated
  - Raw ASGI: Too much boilerplate
- **Consequences**:
  - Pro: Automatic OpenAPI/Swagger docs
  - Pro: Pydantic validation prevents bad inputs
  - Pro: Async support for concurrent requests
  - Pro: Type hints improve IDE support
  - Con: Newer framework, less Stack Overflow answers
- **Validation Criteria**:
  - API docs auto-generate and look professional
  - Request validation catches all malformed inputs
  - Can handle 100+ concurrent requests
  - Easy to add new endpoints

---

## ADR-005: Docker for Deployment
- **Date**: 2025-10-31
- **Status**: Proposed
- **Context**: Need to demonstrate deployment readiness in interviews. Must work on any machine (Mac, Linux, Windows). Should be easy to spin up entire stack for demos.
- **Decision**: Use Docker Compose with separate containers for API, worker, DB, Redis
- **Alternatives Considered**:
  - Kubernetes: Overkill for single-person project
  - Virtual machines: Too heavyweight
  - Conda environments: Doesn't handle services (DB, Redis)
- **Consequences**:
  - Pro: Reproducible environment
  - Pro: Easy to demo ("just run docker-compose up")
  - Pro: Can show DevOps skills in interviews
  - Con: Docker learning curve
  - Con: Larger final artifact size
- **Validation Criteria**:
  - Fresh clone + docker-compose up works on new machine
  - All tests pass inside containers
  - Can access API and dashboard from host machine
  - Logs are accessible and useful

---

## Template for Future Decisions

## ADR-XXX: [Decision Title]
- **Date**: YYYY-MM-DD
- **Status**: [Proposed | Accepted | Deprecated | Superseded]
- **Context**: What problem does this solve? What constraints exist?
- **Decision**: What did you decide to do?
- **Alternatives Considered**: What other options did you evaluate?
- **Consequences**: What are the trade-offs? (Pros and Cons)
- **Validation Criteria**: How will you know this was the right choice?

---

## Notes
- Update status to "Accepted" after implementation validates the decision
- Mark as "Deprecated" if you find a better approach
- Link to "Superseded by ADR-XXX" if you reverse a decision
- Add validation results after implementation
