# Project Context: Player Prop Prediction with GNN Correlation Modeling

## Business Problem
Swish Analytics needs soccer player prop prediction with correlated parlay pricing. Current bookmakers assume independence when pricing parlays (e.g., P(Salah scores AND 3+ shots) = P(Salah scores) × P(3+ shots)), creating systematic mispricing. We exploit this by modeling actual correlations.

## Technical Approach
- **Tier 1**: Bayesian Multi-Task Learning for 4 props (goals, assists, shots on target, cards)
- **Tier 2**: Graph Neural Network capturing player interactions to model correlations

## Success Metrics
- **Calibration**: ECE < 0.05
- **Correlation capture**: R² > 0.7 vs actual outcomes
- **Parlay pricing error**: < 10% vs bookmaker odds
- **Inference latency**: < 100ms per prediction
- **ROI simulation**: > 15% using Kelly criterion

## Data Sources (All Free)
- **FBref**: Player statistics via worldfootballR
- **StatsBomb Open Data**: Event data with player IDs
- **Football-Data.co.uk**: Bookmaker odds for validation
- **Metrica Sports** (optional): Tracking data for spatial features

## Technical Constraints
- Must be deployable by individual without team
- Must use free compute (Colab Pro acceptable, no AWS credits required)
- Must be demonstrable in live interview setting
- All code must be production-grade, not research notebooks

## Target Props
1. **Goals** (binary: 0 or 1+)
2. **Assists** (binary: 0 or 1+)
3. **Shots on target** (over/under 2.5)
4. **Cards** (yellow/red, binary)

## Architecture Decisions (To Be Validated)
- **Bayesian framework**: PyMC3 for Tier 1
- **GNN framework**: PyTorch Geometric for Tier 2
- **API**: FastAPI
- **Database**: PostgreSQL + Redis cache
- **Frontend**: Streamlit
- **Deployment**: Docker containers

## Non-Negotiable Requirements
- Every model must output uncertainty quantification
- Every prediction must be calibrated (not just accurate)
- Every component must have unit tests
- Every API endpoint must have latency monitoring
- All data processing must be reproducible

## Decision Gates
- **Gate 1** (After Phase 2): 100+ matches collected, baseline ECE < 0.15
- **Gate 2** (After Phase 3): Bayesian model ECE < 0.05, ROI > 0 on backtest
- **Gate 3** (After Phase 6): GNN correlation R² > Tier 1 + 0.15
- **PIVOT**: If any gate fails after 2 attempts, reassess approach

## Data Requirements
- **Tier 1 minimum**: 100 matches, 500+ players
- **Tier 2 minimum**: 500 matches for GNN training
- **Validation**: 20% holdout, time-split not random split

## Learning Resources
- **PyMC3**: https://docs.pymc.io/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **StatsBomb**: https://github.com/statsbomb/open-data
- **Friends of Tracking**: https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w
- **TacticAI Paper**: https://www.nature.com/articles/s41467-024-45965-x

## Latest Decisions
(Update this section as you make major decisions during development)

### 2025-10-31 - Phase 0 Complete
- **Decision**: Foundation setup complete
- **Status**: All base files created, dependencies installed, git initialized
- **Next**: Phase 1 data collection

### 2025-11-02 - Phase 1 Architecture Complete
- **Decision**: All architecture documented in docs/
- **Status**: Design validated, ready for implementation
- **Docs Created**: data_schema.md, model_architecture.md, gnn_architecture.md, api_spec.md, deployment_guide.md
- **Next**: Phase 2 - Data Infrastructure Implementation

### 2025-11-02 - Phase 2.1 Complete: Database + Validation
- **Decision**: PostgreSQL schema implemented with 6 tables
- **Status**: Database working, 20 validation tests passing (98% coverage)
- **Files Created**:
  - data/schemas/create_tables.sql
  - src/data/validation.py
  - tests/unit/test_validation.py
- **Tables**: teams, players, matches, player_match_stats, rolling_features, bookmaker_odds
- **Positions**: 'Forward', 'Midfielder', 'Defender', 'Goalkeeper' (full words)
- **Next**: Phase 2.2 - StatsBomb Loader (World Cup 2018 data)

### 2025-11-07 - Phase 2.2 Complete: StatsBomb Loader
- **Decision**: World Cup 2018 data successfully loaded into PostgreSQL
- **Status**: 
  - 32 teams, 603 players, 64 matches, 1720 player-match records
  - 155 goals, 1606 shots recorded
  - Database: football_props on localhost
  - User: medhanshchoubey (no password)
- **Files Created**:
  - src/data/statsbomb_loader.py (working)
  - src/data/load_world_cup_2018.py (working)
  - tests/unit/test_statsbomb_loader.py (created)
- **Known Issues**: 
  - Assists = 0 (StatsBomb API column naming, non-critical for Phase 3)
  - Can be fixed later if needed
- **Data Quality**: All validation passing, sufficient for baseline model
- **Next**: Phase 2.4 - Feature Engineering OR Phase 3 - Baseline Model

### 2025-11-07 - Phase 2.4 Complete: Feature Engineering
- **Decision**: Features calculated for all 1720 player-match records
- **Status**: player_features.csv created (1720 rows × 34 columns)
- **Features**: 10 rolling features (goals, assists, shots, cards for 5 & 10 game windows)
- **Validation**: No NaN values, opponent_strength calculated, tests passing 16/16
- **Files**: src/data/features.py, tests/unit/test_features.py
- **Next**: Phase 2.5 - Load features into database player_features table

### 2025-11-09 - Phase 2 COMPLETE: Data Infrastructure
- **Status**: All 5 substeps complete
- **Database**: player_features table with 1720 records
- **Data Quality**: 
  - 0 NULL values in critical columns
  - Feature distributions validated
  - avg_goals_rolling_5 = 0.231 (correct for all positions)
- **Tables**: 7 total (teams, players, matches, player_match_stats, 
  player_features, rolling_features, bookmaker_odds)
- **Decision Gate 1**: ✅ PASSED
  - 64 matches collected (World Cup 2018)
  - 1720 feature records in database
  - Data quality validated
  - Ready for baseline model
- **Next**: Phase 3 - Baseline Model (Logistic Regression)
  - Goal: ECE < 0.15
  - Target: Beat historical average baseline

### 2025-11-11 - Phase 3.1 Complete: EDA + Baseline Calibration
- **Decision**: EDA finished; baseline logistic models trained for **goals**, **shots on target**, and **cards** (assists excluded).  
  Baseline achieves **average ECE = 0.0225 (< 0.05)** → calibration gate satisfied by baseline.
- **Status**: Generated distributions, position breakdowns, correlations, and calibration plots (docs/01–05).  
  Notable: **corr(goals, shots_on_target) ≈ 0.59** (supports multi-task).  
  **was_home** coefficient negative in baseline → treat with **neutral/cautious prior**.
- **Known Issues**: **Assists all zeros** → defer assists modeling until Phase 2 loader is fixed.
- **Next**: **Phase 3.2 – Bayesian Goals Model (PyMC v5)**  
  *Maintain* ECE < 0.05, add uncertainty (credible intervals), hierarchical priors by position, convergence checks (R-hat < 1.01, ESS > 400).


### [Date] - Decision Template
- **Decision**: What you decided
- **Reasoning**: Why you decided this
- **Alternatives considered**: What else you looked at
- **Validation**: How you'll know if this was right

---

## Quick Reference Commands

### Start New Conversation
```
"Read PROJECT_CONTEXT.md. I'm at Phase [X], Step [X.X]. 
Current task: [specific task]. Question: [question]"
```

### Update After Major Decision
```
Add to "Latest Decisions" section with date, decision, reasoning
```

### Before Moving to Next Phase
```
Check Decision Gates - have you met the criteria?
```
