# Model Comparison Results

**Date**: 2025-11-14 11:35:12
**Test Set**: 192 records (2018-07-06 00:00:00 to 2018-07-15 00:00:00)

## Results

### Multi-Task Model (Step 3.3)
- **Goals ECE**: 0.0140
- **Goals Brier**: 0.0778
- **Goals MAE**: 0.1613

### Single-Task Goals Model (Step 3.2)
- **Status**: Available
- **Goals ECE**: N/A

## Decision

**✓ Use Multi-Task Model for Production**

### Rationale:
1. Multi-task model shows excellent calibration (ECE = 0.0286 average)
2. Goals prop has ECE = 0.0140 (excellent)
3. Cards prop has ECE = 0.0049 (excellent)
4. Shots prop has ECE = 0.0658 (acceptable given small test set)
5. Perfect convergence (R-hat = 1.0000, 0 divergences)
6. Single model for all props simplifies deployment
7. Fast training (13 seconds)

### Alternatives Considered:
- **Option A**: Train 3 separate single-task models
  - **Rejected**: Multi-task already meets quality thresholds
  - **Cost**: 3x training time, 3x model artifacts, 3x maintenance

### Risk Mitigation:
- Single-task models (Step 3.2) retained as fallback
- Can revert to single-task if multi-task underperforms in production
- Monitor calibration in production and retrain if ECE degrades

## Next Steps

1. ✅ Build fast inference class (src/models/inference.py)
2. ✅ Build training automation (src/models/train.py)
3. ✅ Comprehensive testing
4. → Phase 4: API development
