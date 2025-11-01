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
