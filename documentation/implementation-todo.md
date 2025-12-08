# Implementation TODO: Algebra EBM Critical Fixes & Remaining Features

## =¨ CRITICAL BUG FIXES (Must Fix Before Any Other Work)

Based on comprehensive bug analysis in `documentation/implementation-plan.md`, the following critical issues must be resolved immediately as they prevent proper IRED functionality:

### Critical Priority - Fix Immediately (Dependencies: None)

#### - [ ] **BUG-1: Fix Loss Scale Imbalance** 
**File:** `src/diffusion/denoising_diffusion_pytorch_1d.py:1041-1084`
**Dependencies:** None
**Parallel Status:** Can run in parallel with BUG-2, BUG-3
**Issue:** Energy loss contributes only 0.3% vs 99.7% for MSE, preventing energy landscape formation
**Success Criteria:** Energy contributes 40-50% of total loss, energy gaps >8 units
**Implementation Details:**
```python
# Replace adaptive scaling with stronger energy supervision
adaptive_scale = torch.clamp(adaptive_scale, min=10.0, max=500.0)  # Raised from [0.1, 10.0]
target_energy_ratio = 0.5
```
**Validation:** Monitor energy gaps during training - target >8 units within current training window

#### - [ ] **BUG-2: Fix Negative Coefficient Format Bug**
**File:** `src/algebra/algebra_dataset.py:289, 294, 418, 423`
**Dependencies:** None
**Parallel Status:** Can run in parallel with BUG-1, BUG-3
**Issue:** Negative coefficients create malformed equations like "3*x+-15=42"
**Success Criteria:** Zero data format corruption, all equations syntactically valid
**Implementation Details:**
```python
def format_term(coeff, include_plus=True):
    if coeff >= 0:
        return f"+{coeff}" if include_plus else f"{coeff}"
    else:
        return f"{coeff}"  # Negative sign already included
```
**Validation:** Verify no equations contain "++" or "+-" sequences

#### - [ ] **BUG-3: Fix Energy Caching Bug in Inference**
**File:** `src/algebra/algebra_inference.py:454-530`
**Dependencies:** None
**Parallel Status:** Can run in parallel with BUG-1, BUG-2
**Issue:** IRED inference recomputes energy redundantly, degrading performance 30-50%
**Success Criteria:** Energy cached between iterations, 30-50% inference speedup
**Implementation Details:**
```python
# Add caching logic to avoid redundant energy computation
if have_cached_energy:
    energy_before_val = cached_energy_val
```
**Validation:** Verify energy is not recomputed when cache is valid

#### - [ ] **BUG-4: Integrate ContrastiveEnergyLoss**
**File:** `src/diffusion/denoising_diffusion_pytorch_1d.py:302, 990-1013`
**Dependencies:** BUG-1 (loss scaling) should be completed first
**Parallel Status:** Must run after BUG-1
**Issue:** Sophisticated ContrastiveEnergyLoss class exists but bypassed in training
**Success Criteria:** Explicit energy targets enforced (pos=1.0, neg=15.0, margin=10.0)
**Implementation Details:**
```python
self.use_contrastive_energy_loss = True  # Enable by default
self.contrastive_loss_fn = ContrastiveEnergyLoss(margin=10.0, pos_target=1.0, neg_target=15.0)
```
**Validation:** Monitor contrastive metrics during training

### High Priority - Fix Before Production (Dependencies: Critical fixes)

#### - [ ] **BUG-5: Fix Encoder Vocabulary Limitation**
**File:** `src/algebra/algebra_encoder.py:69`
**Dependencies:** BUG-2 (coefficient format)
**Parallel Status:** Must run after BUG-2
**Issue:** Vocabulary missing characters causes crashes on malformed equations
**Success Criteria:** No "Unknown character" crashes, robust handling of all equation formats
**Implementation Details:**
```python
self.vocab = '0123456789x.+-=*/()[]<> '  # Extended vocabulary
```

#### - [ ] **BUG-6: Fix Gradient Computation Bug**
**File:** `src/algebra/algebra_models.py:298-305`
**Dependencies:** None
**Parallel Status:** Can run in parallel with any other fixes
**Issue:** Missing requires_grad_(True) breaks gradient computation
**Success Criteria:** All gradient computations work correctly
**Implementation Details:**
```python
out = out.clone().requires_grad_(True)  # Add missing requires_grad_
```

#### - [ ] **BUG-7: Fix Silent Zero Coefficient Bug**
**File:** `src/algebra/algebra_dataset.py:455-456`
**Dependencies:** BUG-2 (coefficient format)
**Parallel Status:** Must run after BUG-2
**Issue:** Silent coefficient changes create incorrect training pairs
**Success Criteria:** Proper coefficient regeneration, no silent fallbacks
**Implementation Details:**
```python
# Replace silent fallback with regeneration
max_attempts = 10
for attempt in range(max_attempts):
    # ... generate coefficients ...
    if combined_coeff != 0:
        break
```

## =Ë REMAINING IMPLEMENTATION FEATURES

### Phase 6: Constraint Energies ó

#### - [ ] **Step 15: Implement Constraint Energy Functions**
**File:** `src/algebra/algebra_constraints.py` (create new)
**Dependencies:** All critical bugs (BUG-1 through BUG-7)
**Parallel Status:** Cannot start until critical bugs fixed
**Success Criteria:** Working constraint injection without retraining
**Implementation Details:**
- PositiveSolutionConstraint: energy += 100 if solution < 0
- IntegerConstraint: energy += distance to nearest integer
- RangeConstraint: energy += penalty outside [min, max]

#### - [ ] **Step 16: Test Constraint Injection**
**File:** `test_constraints.py` (create new)
**Dependencies:** Step 15
**Parallel Status:** Must run after Step 15
**Success Criteria:** Constraints provably affect solution selection

### Phase 7: Baseline Implementations ó

#### - [ ] **Step 17: Implement Monolithic IRED Baseline**
**File:** `train_monolithic.py` (create new)
**Dependencies:** All critical bugs (BUG-1 through BUG-7)
**Parallel Status:** Can run in parallel with Step 15-16
**Success Criteria:** Single model trained on multi-rule problems for comparison

#### - [ ] **Step 18: Implement NLM Baseline (Optional)**
**File:** `baselines/nlm_baseline.py` (create new)
**Dependencies:** None (independent implementation)
**Parallel Status:** Can run in parallel with any other work
**Success Criteria:** Neuro-symbolic baseline for comparison

### Phase 8: Ablation Studies ó

#### - [ ] **Step 19: Implement Encoder Ablations**
**File:** `experiments/encoder_ablation.py` (create new)
**Dependencies:** All critical bugs, Step 17
**Parallel Status:** Can run in parallel with Steps 20-22
**Success Criteria:** Character vs AST vs hybrid encoder comparison

#### - [ ] **Step 20: Implement Energy Granularity Ablations**
**File:** `experiments/granularity_ablation.py` (create new)
**Dependencies:** All critical bugs, Step 17
**Parallel Status:** Can run in parallel with Steps 19, 21-22
**Success Criteria:** Rule-level vs operation-level vs monolithic comparison

### Phase 9: Analysis and Visualization ó

#### - [ ] **Step 21: Implement Energy Landscape Visualization**
**File:** `visualization/landscape_viz.py` (create new)
**Dependencies:** All critical bugs
**Parallel Status:** Can run in parallel with any other work after critical fixes
**Success Criteria:** Interactive plots showing energy surfaces and minima

#### - [ ] **Step 22: Implement Inference Trajectory Visualization**
**File:** `visualization/trajectory_viz.py` (create new)
**Dependencies:** All critical bugs
**Parallel Status:** Can run in parallel with Step 21
**Success Criteria:** Animation of optimization paths through embedding space

### Phase 10: Integration and Testing ó

#### - [ ] **Step 23: Create End-to-End Pipeline**
**File:** `run_full_pipeline.py` (create new)
**Dependencies:** All previous steps
**Parallel Status:** Must run after everything else
**Success Criteria:** Single script for complete training ’ evaluation workflow

#### - [ ] **Step 24: Implement Unit Tests**
**File:** `tests/` directory (create new)
**Dependencies:** All critical bugs
**Parallel Status:** Can run in parallel with any implementation work
**Success Criteria:** 80%+ code coverage, all critical functions tested

#### - [ ] **Step 25: Create Results Analysis Notebook**
**File:** `analysis/results_analysis.ipynb` (create new)
**Dependencies:** Steps 15-24
**Parallel Status:** Must run after data collection
**Success Criteria:** Publication-ready figures and statistical analysis

## = DEPENDENCY TREE & PARALLEL EXECUTION PLAN

### Phase 1: Critical Bug Fixes (Week 1)
**MUST BE COMPLETED FIRST - BLOCKS ALL OTHER WORK**

```
Parallel Group A (can run simultaneously):
   BUG-1: Loss Scale Imbalance
   BUG-2: Coefficient Format Bug  
   BUG-3: Energy Caching Bug

Sequential Group B (must run after Group A):
   BUG-4: ContrastiveEnergyLoss (depends on BUG-1)
   BUG-5: Encoder Vocabulary (depends on BUG-2)
   BUG-7: Zero Coefficient Bug (depends on BUG-2)

Parallel Group C (can run anytime):
   BUG-6: Gradient Computation Bug
```

### Phase 2: Feature Implementation (Week 2-3)
**CAN ONLY START AFTER ALL CRITICAL BUGS FIXED**

```
Parallel Group D (can run simultaneously after critical fixes):
   Step 15: Constraint Energy Functions
   Step 17: Monolithic IRED Baseline
   Step 18: NLM Baseline (optional)
   Step 21: Energy Landscape Visualization
   Step 22: Inference Trajectory Visualization
   Step 24: Unit Tests

Sequential Group E (must run after Group D):
   Step 16: Test Constraints (depends on Step 15)
   Step 19: Encoder Ablations (depends on Step 17)
   Step 20: Granularity Ablations (depends on Step 17)

Final Group F (must run last):
   Step 23: End-to-End Pipeline (depends on most other steps)
   Step 25: Results Analysis (depends on data from all experiments)
```

##   CRITICAL SUCCESS CRITERIA

### Bug Fix Validation Requirements

**BUG-1 (Loss Scale):**
- [ ] Energy contribution >40% of total loss
- [ ] Energy gaps >8 units during training
- [ ] Training loss convergence stable

**BUG-2 (Data Format):**
- [ ] Zero equations with "++" or "+-" sequences
- [ ] 100% syntactically valid equations
- [ ] SymPy parsing success rate >99%

**BUG-3 (Energy Caching):**
- [ ] 30-50% inference speedup measured
- [ ] Numerical consistency in Metropolis decisions
- [ ] No redundant energy computations

**BUG-4 (ContrastiveEnergyLoss):**
- [ ] Positive energies ~1.0, negative ~15.0
- [ ] Energy gaps >10 units consistently
- [ ] Margin loss actively enforced

### Overall System Validation

**Training Quality:**
- [ ] Energy landscapes have >10 unit gaps
- [ ] Training loss converges stably
- [ ] No gradient explosions or NaN values

**Evaluation Accuracy:**
- [ ] Single-rule problems: >80% accuracy
- [ ] Multi-rule problems: >50% accuracy (2-3x improvement over current)
- [ ] Decoder distance metrics improve naturally

**Performance Requirements:**
- [ ] Training: <16GB GPU memory with batch_size=2048
- [ ] Inference: <5 seconds per equation on single GPU
- [ ] Evaluation: Complete test set in <30 minutes

## <¯ WORK PRIORITIZATION STRATEGY

### Week 1: Critical Infrastructure Repair
**DO NOT START ANY NEW FEATURES UNTIL ALL CRITICAL BUGS ARE FIXED**

1. **Day 1-2:** Parallel implementation of BUG-1, BUG-2, BUG-3
2. **Day 3:** Sequential implementation of BUG-4, BUG-5, BUG-7 
3. **Day 4:** Implementation of BUG-6 + comprehensive testing
4. **Day 5:** Integration testing, validation, documentation

### Week 2-3: Feature Implementation 
**Only after critical fixes are validated and working**

1. **Week 2:** Parallel implementation of constraints, baselines, visualizations
2. **Week 3:** Ablation studies, pipeline integration, final testing

## =Ê EXPECTED OUTCOMES

### Quantitative Improvements from Bug Fixes
- **Accuracy:** +50-100% relative improvement (e.g., 30% ’ 45-60%)
- **Energy Gaps:** 1-2 units ’ 10-14 units  
- **Training Stability:** Unstable ’ Stable convergence
- **Inference Speed:** +30-50% performance improvement
- **Data Quality:** 60-75% valid ’ 100% valid equations

### Research Deliverables
- Publication-quality results on compositional generalization
- Validated IRED implementation for symbolic reasoning
- Framework for constraint injection in energy-based models
- Comprehensive ablation studies on energy composition approaches

## =¨ CRITICAL NOTES

1. **NO NEW FEATURES UNTIL BUGS ARE FIXED**: The current implementation has fundamental issues that prevent proper IRED functionality. Any new feature work will build on a broken foundation.

2. **VALIDATION IS MANDATORY**: Each bug fix must be validated with specific success criteria before moving to the next step.

3. **PARALLEL WORK OPPORTUNITIES**: Multiple bug fixes can be implemented simultaneously by different developers, but dependencies must be respected.

4. **PERFORMANCE MONITORING**: Watch for energy gap formation, training stability, and evaluation accuracy throughout the process.

5. **CURRENT SETTINGS MAINTAINED**: Keep distance_threshold=6.0 and current training durations while focusing on algorithmic fixes first.

## =Ë COMPLETION TRACKING

**Critical Bugs Fixed:** 0/7  
**Features Implemented:** 0/11  
**Overall Progress:** 0/18 (0%)

**Next Action:** Start with parallel implementation of BUG-1, BUG-2, and BUG-3 immediately.