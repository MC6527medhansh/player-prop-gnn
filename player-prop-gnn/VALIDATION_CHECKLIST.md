# Validation Checklist

## Overview
Quality gates for each development phase. Do not move to the next phase until all checkboxes for the current phase are complete.

---

## Phase 0: Foundation ✓
- [ ] PROJECT_CONTEXT.md exists and is complete
- [ ] ARCHITECTURE_DECISIONS.md exists with initial decisions documented
- [ ] VALIDATION_CHECKLIST.md exists (this file)
- [ ] Complete directory structure created
- [ ] Virtual environment created and activated
- [ ] Base dependencies installed from requirements.txt
- [ ] Git repository initialized
- [ ] .gitignore configured properly
- [ ] Configuration management (settings.py) set up
- [ ] README.md created with project overview
- [ ] All Phase 0 files committed to git

**Time Investment**: 1-2 hours  
**Next Phase**: Phase 1 - Data Collection & Exploration

---

## Phase 1: Data Collection & Exploration
- [ ] FBref scraper working and tested on 10+ matches
- [ ] StatsBomb data loader working for open datasets
- [ ] Data validation functions catch common issues
- [ ] Database schema designed and documented
- [ ] PostgreSQL database created and accessible
- [ ] Can load 100+ matches into database
- [ ] Exploratory data analysis completed in notebook
- [ ] Understand prop distributions (goals, assists, shots, cards)
- [ ] Identified data quality issues and mitigation strategies
- [ ] Feature engineering plan documented
- [ ] Data pipeline can run end-to-end without errors
- [ ] Unit tests for data validation functions pass

**Decision Gate 1**: 100+ matches collected, data quality validated, prop distributions understood

---

## Phase 2: Tier 1 Baseline Model
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

**Decision Gate**: Baseline model working, ECE < 0.15, ready for Bayesian upgrade

---

## Phase 3: Tier 1 Bayesian Multi-Task Model
### Model Development
- [ ] Bayesian multi-task model implemented in PyMC3
- [ ] Prior choices documented and justified
- [ ] Model converges (R-hat < 1.01 for all parameters)
- [ ] Effective sample size (ESS) > 1000 for key parameters
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

**Decision Gate 2**: Bayesian model ECE < 0.05, interpretable, beats baseline

---

## Phase 4: API Development
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

---

## Phase 5: Dashboard Development
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

---

## Phase 6: Tier 2 - Graph Construction
### Graph Design
- [ ] Graph representation designed and documented
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

**Decision Gate**: Graph construction solid, ready for GNN training

---

## Phase 7: Tier 2 - GNN Baseline
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
- [ ] Attention patterns (if applicable) interpretable
- [ ] Model generalizes to unseen matches
- [ ] Unit tests for GNN components pass

---

## Phase 8: Tier 2 - Advanced GNN
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

**Decision Gate 3**: GNN captures correlations, beats Tier 1, parlay pricing improved

---

## Phase 9: Backtesting & Validation
- [ ] Backtesting framework implemented
- [ ] Time-series cross-validation working
- [ ] ROI simulation using Kelly criterion
- [ ] ROI > 15% on historical data
- [ ] Calibration maintained over time
- [ ] No look-ahead bias in backtest
- [ ] Bankroll simulation shows positive growth
- [ ] Edge detection for positive EV bets working
- [ ] Results reproducible across runs

---

## Phase 10: Production Readiness
### Deployment
- [ ] Docker Compose configuration complete
- [ ] Dockerfile.api builds successfully
- [ ] Dockerfile.worker (if needed) builds successfully
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
- [ ] Error alerts configured
- [ ] Performance monitoring dashboard created

### Testing
- [ ] All unit tests pass (coverage > 80%)
- [ ] All integration tests pass
- [ ] End-to-end test passes
- [ ] Performance tests pass (latency requirements met)
- [ ] Tests run in CI/CD pipeline (if configured)

### Documentation
- [ ] README.md complete with setup instructions
- [ ] API documentation complete
- [ ] Model architecture documented
- [ ] Deployment guide complete
- [ ] Troubleshooting guide created
- [ ] Code comments sufficient for handoff

---

## Phase 11: Portfolio Readiness
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

---

## Continuous Quality Checks
Run these checks regularly throughout development:

### Daily Checks
- [ ] All unit tests pass
- [ ] No linter warnings (flake8, black)
- [ ] Git commits have descriptive messages
- [ ] No sensitive data in git history

### Weekly Checks
- [ ] Integration tests pass
- [ ] Documentation is up-to-date
- [ ] ARCHITECTURE_DECISIONS.md reflects current state
- [ ] Can rebuild entire project from scratch

### Before Each Phase Transition
- [ ] All phase checklist items complete
- [ ] Decision gate criteria met
- [ ] Git repository clean (no uncommitted changes)
- [ ] Updated PROJECT_CONTEXT.md if needed

---

## Emergency Pivot Criteria

### When to Reassess
If you encounter any of these, pause and reassess:
- [ ] Stuck on same problem for > 1 week
- [ ] Decision gate fails after 2 attempts
- [ ] Model performance plateaus far from target
- [ ] Data quality issues unfixable with current sources
- [ ] Technical approach fundamentally flawed

### Pivot Options
1. **Simplify scope**: Drop Tier 2, perfect Tier 1
2. **Change data source**: Try different leagues/competitions
3. **Adjust metrics**: Maybe ECE < 0.08 is acceptable?
4. **Focus on subset**: Maybe just goals + assists props
5. **Change architecture**: Different GNN or different correlation approach

---

## Notes
- Be honest with yourself on checklist completion
- "Working" means tested and validated, not just "runs once"
- If you skip items, document why in ARCHITECTURE_DECISIONS.md
- Update this checklist as you learn what's important
- Some items may not apply to your specific implementation
