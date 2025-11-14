# Phase 3.4-3.5 Architecture Design

**Date**: November 14, 2025
**Goal**: Complete Phase 3 (Tier 1 Bayesian Models) with production-ready infrastructure

---

## Overview

We need to:
1. ✅ Validate multi-task model is good enough (quick comparison)
2. ✅ Build fast inference class (<100ms predictions)
3. ✅ Build training automation script
4. ✅ Comprehensive tests for everything

---

## Architecture Decisions

### Decision 1: Model Comparison Strategy

**Options considered:**
A. Full comparison: Train 3 single-task models from scratch (~3+ hours)
B. Quick comparison: Compare existing Step 3.2 goals model vs multi-task goals (~5 min)
C. Skip comparison: Just use multi-task

**Decision: Option B (Quick Comparison)**

**Rationale:**
- Already have bayesian_goals.py trained (Step 3.2)
- Goals is the most important prop for betting
- Multi-task already shows excellent results (ECE = 0.0286 avg)
- Time-to-market matters in sports betting
- 5 minutes vs 3+ hours

**Validation:**
```python
if multi_task_ece <= single_task_ece * 1.10:  # Within 10%
    decision = "Use multi-task"
else:
    decision = "Need full comparison"
```

**Failure modes & mitigations:**
- Models trained on different splits → Ensure same test set
- Step 3.2 model not found → Document that comparison skipped
- Different feature sets → Validate feature consistency

---

### Decision 2: Inference Class Architecture

**Options considered:**
A. Stateless functions (reload trace each call)
B. Class with cached samples (load once, predict many)
C. Singleton pattern with global state

**Decision: Option B (Class with cached samples)**

**Rationale:**
- Trace file is 356 MB (expensive to load)
- Load once at init, predict thousands of times
- Clean OOP interface
- Easy to unit test

**Interface design:**
```python
class BayesianPredictor:
    """
    Fast inference with cached posterior samples.
    
    Latency target: <100ms per player
    """
    
    def __init__(
        self,
        model_path: str,
        trace_path: str,
        n_samples: int = 1000
    ):
        """Load model and cache posterior samples."""
        # Load metadata (player_ids, coords, etc.)
        # Load trace
        # Cache posterior samples in memory
        
    def predict_player(
        self,
        player_id: int,
        opponent_id: int,
        was_home: bool,
        features: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict all props for a single player.
        
        Returns:
        {
            'goals': {'mean': 0.45, 'ci_low': 0.1, 'ci_high': 1.2, ...},
            'shots': {...},
            'cards': {...}
        }
        """
        # Vectorized computation using cached samples
        
    def predict_batch(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """Batch prediction for multiple players."""
```

**Failure modes & mitigations:**
| Failure Mode | Mitigation |
|--------------|------------|
| Unknown player | Return league average, log warning |
| Unknown opponent | Use average opponent effect |
| Missing features | Raise ValueError with clear message |
| Out of memory | Reduce n_samples, document in error |
| Trace file corrupt | Try reload, raise RuntimeError |

**Performance optimization:**
- Cache posterior samples as numpy arrays
- Vectorize all computations (no loops over observations)
- Use np.random.poisson() for sampling
- Pre-compute log transforms

---

### Decision 3: Training Script Architecture

**Options considered:**
A. Jupyter notebook only
B. CLI script with argparse
C. Config file driven (YAML/JSON)

**Decision: Option A + B (Both notebook AND CLI)**

**Rationale:**
- Notebook for interactive exploration (already exists)
- CLI for automation (cron, Airflow, etc.)
- Config files add unnecessary complexity for single developer

**CLI Interface:**
```python
python -m src.models.train \
    --train-start 2018-06-01 \
    --train-end 2018-07-05 \
    --val-end 2018-07-16 \
    --draws 2000 \
    --chains 4 \
    --version v1.1
```

**Function signature:**
```python
def train_multitask_model(
    train_start_date: str,
    train_end_date: str,
    val_end_date: str,
    draws: int = 2000,
    chains: int = 4,
    model_version: str = None,
    db_url: str = 'postgresql://...'
) -> Tuple[Dict, az.InferenceData, Dict]:
    """
    Train multi-task Bayesian model end-to-end.
    
    Returns:
        metadata: Model metadata dict
        idata: ArviZ InferenceData
        results: Calibration metrics dict
    """
```

**Failure modes & mitigations:**
| Failure Mode | Mitigation |
|--------------|------------|
| DB connection fails | Retry 3x with backoff, clear error message |
| No training data | Validate row count > 100 before training |
| MCMC doesn't converge | Check R-hat, raise if > 1.05, suggest more draws |
| Disk full | Check free space before saving (need 500 MB) |
| Invalid dates | Validate format and order at entry |
| Trace save fails | Try multiple netcdf engines, atomic write |

**Logging strategy:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Key log points:
logger.info(f"Training samples: {len(train_data)}")
logger.info(f"MCMC sampling started...")
logger.info(f"Convergence: R-hat = {r_hat:.4f}")
logger.warning(f"Shots ECE = {ece:.4f} > 0.05 threshold")
```

---

## File Structure

```
src/models/
├── bayesian_multitask.py    # EXISTS - model definition
├── bayesian_goals.py         # EXISTS - single-task reference
├── inference.py              # NEW - fast prediction class
└── train.py                  # NEW - training automation

notebooks/exploration/
├── 03_train_multitask.ipynb  # EXISTS
└── 04_model_comparison.ipynb # NEW - quick comparison

tests/unit/
├── test_bayesian_multitask.py  # EXISTS
├── test_inference.py           # NEW - inference class tests
└── test_train.py               # NEW - training script tests

tests/integration/
└── test_end_to_end.py         # NEW - full pipeline

docs/
├── model_comparison_results.md # NEW - decision doc
└── PHASE_3_ARCHITECTURE.md     # THIS FILE
```

---

## Implementation Plan (Small Chunks)

### Chunk 1: Model Comparison Notebook (30 min)
**File**: `notebooks/exploration/04_model_comparison.ipynb`
- Load Step 3.2 single-task goals model
- Load Step 3.3 multi-task model
- Compare on same test set
- Compute ECE difference
- Document decision

**Tests**: Manual verification in notebook

---

### Chunk 2: Inference Class Core (1 hour)
**File**: `src/models/inference.py`
- BayesianPredictor.__init__() - load model, cache samples
- BayesianPredictor.predict_player() - single prediction
- Input validation
- Error handling

**Tests**: `tests/unit/test_inference.py`
- test_initialization
- test_predict_known_player
- test_predict_unknown_player
- test_invalid_inputs

---

### Chunk 3: Inference Batch + Performance (30 min)
**File**: `src/models/inference.py`
- BayesianPredictor.predict_batch()
- Performance optimization
- Timing instrumentation

**Tests**: `tests/unit/test_inference.py`
- test_batch_prediction
- test_latency_under_100ms

---

### Chunk 4: Training Script Core (1 hour)
**File**: `src/models/train.py`
- train_multitask_model() function
- Data loading
- Model training
- Convergence checks

**Tests**: `tests/unit/test_train.py`
- test_train_on_small_dataset
- test_convergence_check
- test_invalid_dates

---

### Chunk 5: Training Script CLI + Persistence (30 min)
**File**: `src/models/train.py`
- argparse CLI
- Model saving with atomic writes
- Results JSON export

**Tests**: `tests/unit/test_train.py`
- test_cli_interface
- test_model_saving

---

### Chunk 6: Integration Test (30 min)
**File**: `tests/integration/test_end_to_end.py`
- Full pipeline: load data → train → save → load → predict
- Validates entire workflow

---

### Chunk 7: Documentation (15 min)
**File**: `docs/model_comparison_results.md`
- Comparison table
- Decision rationale
- Next steps

---

## Acceptance Criteria

### Model Comparison:
- [ ] Comparison notebook runs without errors
- [ ] Decision documented with clear rationale
- [ ] Same test set used for both models

### Inference Class:
- [ ] Prediction latency < 100ms per player (cached)
- [ ] Handles unknown players gracefully
- [ ] Returns all 3 props with uncertainty
- [ ] All unit tests pass

### Training Script:
- [ ] CLI works end-to-end
- [ ] Convergence checks implemented
- [ ] Model saved with metadata
- [ ] Calibration metrics computed
- [ ] All unit tests pass

### Integration:
- [ ] Full pipeline test passes
- [ ] Can train → save → load → predict

---

## Risk Mitigation

### Risk 1: Trace loading is too slow
**Likelihood**: Medium
**Impact**: High (breaks <100ms latency target)
**Mitigation**: 
- Cache samples in __init__
- Only load trace once
- Use numpy arrays, not xarray

### Risk 2: Unknown players common in production
**Likelihood**: High
**Impact**: Medium
**Mitigation**:
- Return league averages
- Log warning with player_id
- Track frequency in monitoring

### Risk 3: MCMC doesn't converge on new data
**Likelihood**: Low
**Impact**: High
**Mitigation**:
- Auto-check R-hat < 1.05
- Raise error with diagnostic info
- Suggest increasing draws

### Risk 4: Disk space issues
**Likelihood**: Low  
**Impact**: Medium
**Mitigation**:
- Check free space before saving (need 500 MB)
- Use atomic writes (tmp file → rename)
- Clear old model versions

---

## Alternatives Considered (and Rejected)

### Alternative 1: Standalone evaluate.py module
**Why rejected**: 
- evaluate_calibration() already in bayesian_multitask.py
- Don't need separate module for solo developer
- YAGNI principle

### Alternative 2: FastAPI integration in Phase 3
**Why rejected**:
- API is Phase 4
- Keep phases focused
- Can use inference.py as backend for API later

### Alternative 3: Full model comparison (3 single-task models)
**Why rejected**:
- Multi-task already validated (ECE < 0.05 for 2/3 props)
- 3+ hours vs 5 minutes
- Time-to-market matters
- Can always revert if needed

---

## Next Phase Preview (Phase 4)

After Phase 3 complete:
1. FastAPI application (src/api/main.py)
2. Use inference.py as backend
3. Add Redis caching
4. Add monitoring endpoint
5. Dockerize

---

**Decision Gate 2 Criteria:**
- [x] Multi-task model converged (R-hat < 1.01) ✅
- [ ] Average ECE < 0.05 (currently 0.0286) ✅  
- [ ] Inference < 100ms per player
- [ ] Training script works end-to-end
- [ ] All tests passing

**Estimated time**: 4-5 hours total