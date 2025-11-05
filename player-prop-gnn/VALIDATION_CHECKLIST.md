# Validation Checklist

## Overview
Quality gates for each development phase. Do not move to the next phase until all checkboxes for the current phase are complete.

---

## Phase 0: Foundation ✓
- [x] PROJECT_CONTEXT.md exists and is complete
- [x] ARCHITECTURE_DECISIONS.md exists with initial decisions documented
- [x] VALIDATION_CHECKLIST.md exists (this file)
- [x] Complete directory structure created
- [x] Virtual environment created and activated
- [x] Base dependencies installed from requirements.txt
- [x] Git repository initialized
- [x] .gitignore configured properly
- [x] Configuration management (settings.py) set up
- [x] README.md created with project overview
- [x] All Phase 0 files committed to git

**Time Investment**: 1-2 hours  
**Next Phase**: Phase 1 - Architecture Design & Decision Making

---

## Phase 1: Architecture Design & Decision Making ✓
- [x] `docs/data_schema.md` complete with schema, indexes, pipeline
- [x] `docs/model_architecture.md` complete with priors, justifications
- [x] `docs/gnn_architecture.md` complete with graph design, architecture
- [x] `docs/api_spec.md` complete with endpoints, caching, errors
- [x] `docs/deployment_guide.md` complete with containers, monitoring
- [x] Can draw system architecture on whiteboard
- [x] Can explain every design decision
- [x] Have considered 3+ alternatives for major choices
- [x] Documented all decisions in ARCHITECTURE_DECISIONS.md
- [x] Another engineer could implement from docs
- [x] Identified potential failure modes
- [x] Have mitigation strategies

**Time Investment**: 4-8 hours (design only, no code)  
**Critical**: DO NOT write code yet. This phase is pure design.  
**Next Phase**: Phase 2 - Data Infrastructure Implementation

---

## Phase 2: Data Infrastructure Implementation
### Step 2.1: Database Schema + Validation ✓
- [x] Database schema created (6 tables)
- [x] Validation functions implemented (simple functions, not class)
- [x] Unit tests created (20 tests, 98% coverage)
- [x] Database `football_props` exists
- [x] All tables created without errors

### Step 2.2: StatsBomb Loader (CURRENT)
- [ ] StatsBombLoader class implemented
- [ ] Can load World Cup 2018 data
- [ ] Extract player match stats from events
- [ ] Load 64 matches into database

### Step 2.3: FBref Scraper
- [ ] Not started

### Step 2.4: Features
- [ ] Not started

### Step 2.5: ETL Pipeline
- [ ] Not started
**Decision Gate 1**: 100+ matches collected, data quality validated, prop distributions understood  
**Next Phase**: Phase 3 - Tier 1 Baseline Model

---

## Phase 3: Tier 1 Baseline Model
- [ ] Simple baseline model implemented (logistic regression or similar)
- [ ] Baseline achieves ECE < 0.15 on holdout set
- [ ] Can generate predictions for all 4 props
- [ ] Evaluation metrics implemented (ECE, Brier, log loss)
- [ ] Calibration curves plotted and analyzed
- [ ] Feature importance analyzed and makes intuitive sense
- [ ] Model serialization/deserialization working
- [ ] Inference pipeline functional
- [ ] Unit tests for model training and inference pass
- [ ] Baseline performance documented for comparison

**Decision Gate 2**: Baseline model working, ECE < 0.15, ready for Bayesian upgrade  
**Next Phase**: Phase 4 - Tier 1 Bayesian Multi-Task Model

---

## Phase 4: Tier 1 Bayesian Multi-Task Model
### Model Development
- [ ] Bayesian multi-task model implemented in PyMC
- [ ] Prior choices documented and justified
- [ ] Model converges (R-hat < 1.01 for all parameters)
- [ ] Effective sample size (ESS) > 400 for key parameters
- [ ] Posterior predictive checks show good fit
- [ ] Calibration ECE < 0.05 on holdout set
- [ ] Uncertainty quantification working properly
- [ ] Model beats baseline on all metrics

### Understanding & Validation
- [ ] Can explain every prior distribution choice
- [ ] Posterior distributions make intuitive sense
- [ ] Hierarchical structure (if used) justified
- [ ] Trace plots look healthy (no divergences)
- [ ] Can identify when model is uncertain
- [ ] Feature effects align with soccer intuition

### Engineering
- [ ] Model training script parameterized and reproducible
- [ ] Model artifacts saved with versioning
- [ ] Inference time < 1s per player on CPU
- [ ] Unit tests for model components pass
- [ ] Integration test for full training pipeline passes

**Decision Gate 3**: Bayesian model ECE < 0.05, interpretable, beats baseline  
**Next Phase**: Phase 5 - API Development

---

## Phase 5: API Development
- [ ] FastAPI application structure set up
- [ ] Endpoint for single player prediction working
- [ ] Endpoint for match predictions (all players) working
- [ ] Request/response schemas defined with Pydantic
- [ ] Input validation working properly
- [ ] Error handling comprehensive
- [ ] API documentation auto-generated (Swagger/OpenAPI)
- [ ] Redis caching implemented for repeated queries
- [ ] API latency < 100ms p95 (with cache)
- [ ] API latency < 500ms p95 (without cache)
- [ ] Health check endpoint working
- [ ] Metrics endpoint for monitoring
- [ ] Unit tests for all endpoints pass
- [ ] Load testing shows acceptable performance

**Next Phase**: Phase 6 - Dashboard Development

---

## Phase 6: Dashboard Development
- [ ] Streamlit dashboard created
- [ ] Can select players and view predictions
- [ ] Displays uncertainty (credible intervals)
- [ ] Shows calibration metrics
- [ ] Visualizes feature importance
- [ ] Comparison view for multiple players
- [ ] Historical performance tracking
- [ ] Dashboard loads without crashes
- [ ] Responsive on different screen sizes
- [ ] Easy to navigate and understand

**Next Phase**: Phase 7 - Tier 2 Graph Construction

---

## Phase 7: Tier 2 - Graph Construction
### Graph Design
- [ ] Graph representation implemented
- [ ] Node features defined (player stats + match context)
- [ ] Edge construction logic implemented
- [ ] Edge features defined (interaction types)
- [ ] Graph builder tested on sample matches
- [ ] Graph statistics analyzed (degree distribution, etc.)

### Validation
- [ ] Graphs reproduce deterministically for same match
- [ ] Graph size reasonable (< 50 nodes typical)
- [ ] Edge types make soccer sense
- [ ] Node features normalized appropriately
- [ ] Can visualize graphs for debugging
- [ ] Unit tests for graph construction pass

**Decision Gate 4**: Graph construction solid, ready for GNN training  
**Next Phase**: Phase 8 - GNN Baseline

---

## Phase 8: Tier 2 - GNN Baseline
### Model Development
- [ ] Simple GCN baseline implemented
- [ ] Can train on 500+ match graphs
- [ ] Training converges without NaN losses
- [ ] Validation loss decreases over training
- [ ] Inference time < 100ms per match graph
- [ ] Memory usage < 2GB per model

### Validation
- [ ] GNN learns something (beats random baseline)
- [ ] Learned correlations match intuition
- [ ] Model generalizes to unseen matches
- [ ] Unit tests for GNN components pass

**Next Phase**: Phase 9 - Advanced GNN (GAT)

---

## Phase 9: Tier 2 - Advanced GNN (GAT)
### Model Development
- [ ] Graph Attention Network (GAT) implemented
- [ ] Attention mechanism working properly
- [ ] Multi-head attention implemented
- [ ] Correlation modeling functional
- [ ] Parlay probability calculation correct

### Performance
- [ ] GNN correlation R² > Tier 1 + 0.15
- [ ] Parlay pricing error < 10% vs bookmaker odds
- [ ] Captures known correlations (e.g., goals ↔ shots)
- [ ] Uncertainty calibration maintained (ECE < 0.05)
- [ ] Beats independence assumption significantly

### Understanding
- [ ] Can visualize attention weights
- [ ] Attention weights make soccer sense
- [ ] Can explain which player interactions matter
- [ ] Can identify when correlations are strong/weak

**Decision Gate 5**: GNN captures correlations, beats Tier 1, parlay pricing improved  
**Next Phase**: Phase 10 - Backtesting & Validation

---

## Phase 10: Backtesting & Validation
- [ ] Backtesting framework implemented
- [ ] Time-series cross-validation working
- [ ] ROI simulation using Kelly criterion
- [ ] ROI > 15% on historical data
- [ ] Calibration maintained over time
- [ ] No look-ahead bias in backtest
- [ ] Bankroll simulation shows positive growth
- [ ] Edge detection for positive EV bets working
- [ ] Results reproducible across runs

**Next Phase**: Phase 11 - Production Readiness

---

## Phase 11: Production Readiness
### Deployment
- [ ] Docker Compose configuration complete
- [ ] Dockerfile.api builds successfully
- [ ] docker-compose up works on fresh machine
- [ ] Environment variables properly configured
- [ ] Database initialization script working
- [ ] Can run full stack with one command

### Monitoring & Reliability
- [ ] Logging configured properly
- [ ] Metrics collection working
- [ ] Health checks implemented
- [ ] Can recover from database failure
- [ ] Can recover from Redis failure
- [ ] Performance monitoring dashboard created

### Testing
- [ ] All unit tests pass (coverage > 80%)
- [ ] All integration tests pass
- [ ] End-to-end test passes
- [ ] Performance tests pass (latency requirements met)

### Documentation
- [ ] README.md complete with setup instructions
- [ ] API documentation complete
- [ ] Model architecture documented
- [ ] Deployment guide complete
- [ ] Troubleshooting guide created
- [ ] Code comments sufficient for handoff

**Next Phase**: Phase 12 - Portfolio Readiness

---

## Phase 12: Portfolio Readiness
### Code Quality
- [ ] GitHub repository created and public
- [ ] Code organized and well-structured
- [ ] Naming conventions consistent
- [ ] No dead code or commented-out sections
- [ ] Type hints used throughout
- [ ] Docstrings for all public functions

### Presentation
- [ ] README.md is polished and impressive
- [ ] Demo script prepared
- [ ] Demo video recorded (or can do live demo)
- [ ] Blog post drafted explaining technical approach
- [ ] Can explain project in 2 minutes (elevator pitch)
- [ ] Can explain project in 10 minutes (technical deep dive)
- [ ] Screenshots/visualizations prepared

### Interview Prep
- [ ] Can explain every technical decision
- [ ] Can defend choice of Bayesian + GNN approach
- [ ] Can discuss what you'd do with more time/resources
- [ ] Can discuss failures and what you learned
- [ ] Can discuss trade-offs made
- [ ] Prepared questions about Swish Analytics role

**Project Complete!** 🎉

---

## Summary: Current Progress

**Completed Phases:**
- ✅ Phase 0: Foundation (1-2 hours)
- ✅ Phase 1: Architecture Design (4-8 hours)

**Next Phase:** Phase 2 - Data Infrastructure Implementation (8-12 hours)

**Total Progress:** 2/12 phases complete (17%)

---

## Notes
- Be honest with yourself on checklist completion
- "Working" means tested and validated, not just "runs once"
- If you skip items, document why in ARCHITECTURE_DECISIONS.md
- Update this checklist as you learn what's important