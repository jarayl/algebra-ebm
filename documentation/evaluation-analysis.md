# Comprehensive Evaluation Analysis - Post-DATAGEN Fixes

**Date**: 2026-02-16
**Git SHA**: 4dc7798
**Analysis Phase**: ANALYZE_RESULTS

## Executive Summary

After comprehensive dataset generation fixes (DATAGEN-001 through 005) and full model retraining, evaluation results reveal **critical failure across all compositional scenarios**:

- **Single-rule accuracy**: 6.0-6.7% (expected ~85%, actual 93-94% below target)
- **Multi-rule accuracy**: 0.0% across all configurations (2, 3, and 4 rules)
- **Constrained accuracy**: 0.0% across all constraint types
- **Compositional approach**: Complete failure to demonstrate any compositional advantage

**Key Finding**: The compositional energy-based model approach, as currently implemented, does not work. Despite correct training convergence (9-10 unit energy gaps), models fail to perform inference successfully on even the simplest single-rule problems.

---

## Evaluation Results Summary

### Experiment 001: Single-Rule Baseline
**Job**: 60613327 | **Runtime**: 12.5 min | **Status**: Completed

| Rule | Accuracy | Invalid Rate | Dist Improve |
|------|----------|--------------|--------------|
| distribute | 6.0% | 0.0% | -1.1% |
| combine | 6.3% | 0.0% | -4.1% |
| isolate | 6.1% | 0.0% | -2.6% |
| divide | 6.7% | 0.0% | +2.5% |
| **Average** | **6.3%** | **0.0%** | **-1.3%** |

**Analysis**:
- All four rules show ~6% accuracy (essentially random performance on multi-choice tasks)
- No invalid outputs (0.0% invalid rate indicates models produce parseable equations)
- Negative "distribution improvement" suggests inference moves solutions AWAY from correct answers
- Compare to previous evaluation (job 60189563 with flawed data): distribute=2.1%, combine=100%, isolate=7.0%, divide=5.7%
  - combine dropped from 100% to 6.3% (93.7 percentage point regression)
  - Other rules stayed in 2-7% range (no improvement from DATAGEN fixes)

**Expected**: 85% baseline accuracy on single-rule problems. **Actual**: 6.3% (78.7 points below).

---

### Experiment 002: Multi-Rule (2 rules)
**Job**: 60613338 | **Runtime**: 70 min | **Status**: Completed

| Metric | Value |
|--------|-------|
| Accuracy | 0.0% |
| Invalid Rate | 0.0% |
| Problems Generated | 1000 |

**Analysis**:
- Zero successful solutions across 1000 two-rule problems
- Dataset generation succeeded (1000 problems created)
- Models produce valid equation syntax but incorrect solutions
- 70-minute runtime suggests inference completed without crashes

---

### Experiment 003: Multi-Rule (3 rules)
**Job**: 60613232 | **Runtime**: 71 min | **Status**: Completed

| Metric | Value |
|--------|-------|
| Accuracy | 0.0% |
| Invalid Rate | 0.0% |
| Problems Generated | 1000 |

**Analysis**: Same pattern as 2-rule experiment - complete failure with valid outputs.

---

### Experiment 004: Multi-Rule (4 rules)
**Job**: 60613241 | **Runtime**: 70 min | **Status**: Completed

| Metric | Value |
|--------|-------|
| Accuracy | 0.0% |
| Invalid Rate | 0.0% |
| Problems Generated | 1000 |

**Analysis**: Same pattern as 2-rule and 3-rule experiments.

---

### Experiment 005: Constrained Evaluation
**Job**: 60613220 | **Runtime**: 205 min (3h 25m) | **Status**: Failed (0% accuracy)

| Constraint Type | Accuracy | Invalid Rate | Problems Generated |
|-----------------|----------|--------------|-------------------|
| Positive | 0.0% | 0.0% | 1000 |
| Integer | 0.0% | 0.0% | 1000 |
| Both | 0.0% | 0.0% | 1000 |

**Analysis**:
- Longest runtime (3.5 hours) suggests inference struggled but didn't crash
- All constraint types failed completely
- Dataset generation succeeded (3000 total problems)
- Zero accuracy despite valid outputs

---

### Experiment 007: Compositional vs Monolithic Comparison
**Job**: 60613252 | **Runtime**: 39 min | **Status**: Completed (no results output)

**Analysis**:
- Job completed successfully (exit code 0)
- SLURM logs show 150,000 training steps executed
- Final output shows: "ALGEBRA EBM EVALUATION REPORT" header with empty results section
- Log ends with: "Evaluation completed at Mon Feb 16 14:57:20 EST 2026"
- **Issue**: Evaluation code ran but produced no metrics output
- Previous run (job 60158011, old models) showed:
  - Monolithic single-rule: 29.1%
  - Monolithic multi-rule: 4.5%
  - Compositional multi-rule: 4.9%
  - Compositional advantage: +0.3 percentage points

**Recommendation**: Investigate exp_007 evaluation script to determine why results section is empty.

---

## Training Performance Review

All 5 models completed retraining successfully with DATAGEN-001 through 005 fixes applied.

### Training Convergence Metrics

| Model | Job ID | Runtime | Final Energy Gap | Pos Energy | Neg Energy | Status |
|-------|--------|---------|------------------|------------|------------|--------|
| distribute | 60535220 | 5.4 hours | ~10.0 units | 4.8 | 14.8 | CONVERGED |
| combine | 60535244 | 4.7 hours | ~9.9 units | 4.6 | 14.5 | CONVERGED |
| isolate | 60535245 | 4.4 hours | ~9.8 units | 4.7 | 14.5 | CONVERGED |
| divide | 60535246 | 4.1 hours | ~9.9 units | 4.8 | 14.7 | CONVERGED |
| monolithic | 60535265 | 4.8 hours | ~10.1 units | 4.9 | 15.0 | CONVERGED |

**Training Analysis**:
- All models achieved strong energy gaps (9.8-10.1 units)
- Positive energies: 4.6-4.9 (low energy for correct transformations)
- Negative energies: 14.5-15.0 (high energy for incorrect transformations)
- Margin values occasionally > 0, indicating contrastive loss active
- **Conclusion**: Training succeeded - models learned to distinguish correct from incorrect transformations

**Example from distribute model (final 100 steps)**:
```
[EnergyMonitor] Average energy gap (last 100 steps): 10.000, PosE=4.82, NegE=14.47, Margin=0.3456
```

---

## Root Cause Analysis

### Primary Issue: Inference Failure Despite Training Success

The **critical disconnect** is between training performance and evaluation performance:

1. **Training**: Models achieve 9-10 unit energy gaps
   - Positive samples (correct transformations): Energy ~5
   - Negative samples (incorrect transformations): Energy ~15
   - Clear discrimination during training

2. **Evaluation**: Models achieve 6% accuracy on single-rule tasks
   - Expected: Inference should follow energy gradient to low-energy (correct) solutions
   - Actual: Solutions are essentially random

### Evidence from Prior Audits

**AUDIT-003 Finding** (pre-retraining):
> "Single-rule accuracy investigation: Training converged for all 4 rules with 9-10 unit energy gaps. Root cause of low accuracy (distribute=2.1%, isolate=7.0%, divide=5.7%) is **IRED inference converging to local minima in embedding space**. combine works (100%) because output≈input. Requires inference strategy improvements (multi-start, momentum, more landscapes)."

**Key Insight**: AUDIT-003 diagnosed the problem as IRED inference getting stuck in local minima. However:
- The combine model went from 100% accuracy (when input≈output) to 6.3% accuracy after DATAGEN-002 fixed the format mismatch
- This suggests DATAGEN-002 (changing distribute/combine from bare expressions to full equations) may have inadvertently broken the one working case

### DATAGEN-002 Impact Assessment

**DATAGEN-002 Change**:
- **Before**: distribute generated `2*(3*x+5)` -> `6*x+10` (bare expressions)
- **After**: distribute generates `2*(3*x+5)=d` -> `6*x+10=d` (full equations)

**Hypothesis**: The combine rule previously worked (100% accuracy) because:
1. Input and output were nearly identical expressions
2. IRED inference could find the trivial solution quickly
3. Adding `=` signs and equation structure increased the embedding space complexity
4. Now combine has the same local minima problem as other rules

### Inference Architecture Issues

**IRED Inference Process**:
1. Initialize latent embedding from noisy input
2. Iteratively refine via gradient descent on energy landscape
3. Decode final embedding to output equation

**Known Problems**:
1. **Local minima**: Energy landscape has many local optima
2. **Step size**: Fixed step sizes may overshoot or undershoot
3. **Initialization**: Single random initialization per problem
4. **Iterations**: Fixed iteration count (50 steps per AUDIT-002 fix)
5. **Gradient quality**: Embeddings may not have smooth energy gradients

### Why Multi-Rule Shows 0% vs Single-Rule 6%

**Single-rule problems**:
- One energy landscape to traverse
- Simpler transformation (one rule application)
- Still mostly fails (6% suggests ~1/16 random chance on hypothetical 16-choice space)

**Multi-rule problems**:
- Requires composing 2-4 energy models: E_total = E_rule1 + E_rule2 + ... + E_ruleN
- Each rule's energy landscape has local minima
- Composed landscape is product of multiple broken landscapes
- **Result**: Essentially impossible to find global minimum through gradient descent

### Why Constraints Show 0%

Constrained problems add additional energy terms for constraint violations:
- E_total = E_transformation + E_positivity + E_integer
- Even more complex energy landscape
- More opportunities for local minima
- Inference completely fails

---

## Comparison to Original Research Hypothesis

**Original Hypothesis**: Compositional EBMs would enable:
1. Learning individual transformation rules independently
2. Composing rules at inference time for multi-step problems
3. Better generalization than monolithic models

**Results**: Hypothesis **rejected**:
1. Individual rules train successfully but fail at inference (6% accuracy)
2. Composition completely fails (0% on all multi-rule problems)
3. Previous comparison (job 60158011) showed compositional barely better than monolithic (4.9% vs 4.5%)

---

## Critical Questions Answered

### 1. Is the compositional approach fundamentally broken, or is this fixable?

**Answer**: The approach is **fundamentally broken as currently implemented**. The issue is not training (which works) but inference (which fails). Possible paths forward:

**Path A - Inference Improvements** (high effort, uncertain success):
- Multi-start inference (try 10+ random initializations, pick lowest energy)
- Adaptive step sizes (line search, momentum)
- Better initialization heuristics (start from input embedding, not random)
- More sophisticated inference (simulated annealing, genetic algorithms)
- Increase iterations from 50 to 500+

**Path B - Architecture Changes** (very high effort):
- Switch from IRED to different energy-based model architecture
- Use diffusion models instead of gradient-based inference
- Hybrid approach: neural energy models + symbolic search

**Path C - Abandon Compositional EBM Approach** (low effort):
- Acknowledge that EBM inference is too brittle for this domain
- Pursue monolithic models or different approaches (transformers, graph neural nets)

### 2. What specifically needs to be addressed to improve performance?

**Immediate Issues**:
1. **IRED inference local minima problem** (AUDIT-003) - Most critical
2. **exp_007 missing results** - Need to understand monolithic baseline
3. **Inference hyperparameters** - May need 10x more iterations (500 vs 50)

**Deeper Issues**:
1. **Energy landscape smoothness** - May need architectural changes to embedding space
2. **Composition assumption** - Additive energy composition may not preserve landscape quality
3. **Evaluation validity** - Need to verify DATAGEN fixes actually produced correct test data

### 3. Should we focus on single-rule performance first, or attack multi-rule directly?

**Answer**: **Focus on single-rule first**. Rationale:
- Multi-rule 0% accuracy is a **cascading failure** from single-rule 6% accuracy
- If P(single-rule success) = 0.06, then P(2-rule success) = 0.06^2 = 0.0036 = 0.36%
- Observed 0% on 1000 problems is consistent with 0.36% true success rate
- **Cannot fix composition until individual rules work**

### 4. Are the DATAGEN fixes working as intended?

**Mixed Results**:
- **Working**: Dataset generation succeeded (no crashes, 1000 problems per experiment)
- **Working**: Validation shows 0 errors via SymPy check (DATAGEN-001)
- **Working**: Training converged with expected energy gaps
- **Concerning**: combine dropped from 100% to 6.3% after DATAGEN-002 format change
- **Unknown**: Haven't manually inspected test datasets to verify correctness

**Recommendation**: Manually inspect 10-20 test problems to verify:
1. Equations are mathematically correct (input/target equivalence)
2. Equation format matches training data format
3. Solutions are actually achievable via single rule application

---

## Recommendations

### Immediate Actions (Next 1-2 Days)

1. **Verify Test Dataset Correctness**
   - Manually inspect test datasets in `results/test_datasets/*.json`
   - Pick 10 random problems per rule type
   - Verify mathematical correctness with independent SymPy evaluation
   - Check format matches training data

2. **Investigate exp_007 Missing Results**
   - Debug why evaluation report is empty
   - Re-run if needed to get monolithic baseline comparison
   - Critical for understanding if issue is compositional-specific or affects monolithic too

3. **Run Inference Diagnostics**
   - Add logging to IRED inference: energy at each step, gradient norms, final embedding
   - Visualize energy landscape for 5-10 failed problems
   - Check if inference is stuck at initialization, oscillating, or slowly drifting

### Short-term Experiments (Next 1 Week)

4. **Inference Improvement Experiments**
   - **Exp A**: Multi-start inference (10 random seeds, pick lowest energy)
   - **Exp B**: Increase iterations from 50 to 500
   - **Exp C**: Add momentum (running average of gradients)
   - **Exp D**: Better initialization (initialize from input embedding instead of random)

5. **Sanity Check Experiment**
   - Train model on 100 problems only
   - Test on same 100 problems (train == test)
   - If still fails, issue is inference not generalization
   - If succeeds, issue is generalization/overfitting

### Long-term Decision Points (Next 2 Weeks)

6. **Decision Point: Continue or Pivot?**
   - If short-term experiments raise single-rule accuracy to >50%: Continue with multi-rule composition
   - If short-term experiments show <20% improvement: Consider pivoting to different approach
   - If inference diagnostics show insurmountable landscape problems: Abandon EBM approach

7. **Alternative Approaches to Evaluate** (if pivoting):
   - Seq2seq transformers with rule labels
   - Graph neural networks on equation parse trees
   - Hybrid symbolic-neural approaches
   - Monolithic diffusion models (not compositional)

---

## Next Phase Recommendation

**Recommended Phase**: `DEBUG` or `INFERENCE_IMPROVEMENT`

**Rationale**:
- Current results show fundamental inference failure
- Training works, so no need to modify training pipeline
- Need focused investigation on IRED inference behavior
- Should not proceed to more experiments until root cause is addressed

**Next Actions**:
1. Set phase to `DEBUG` in pipeline.json
2. Add detailed inference logging to IRED
3. Run diagnostic experiment on 10-50 problems with full energy trajectory logging
4. Analyze energy landscapes and gradient behavior
5. Design targeted fix based on diagnostic findings

---

## Files Created/Modified

**Created**:
- `/Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm/documentation/evaluation-analysis.md` (this file)

**To Update**:
- `projects/algebra-ebm/.state/pipeline.json` (set phase, next_action, update results)
- `projects/algebra-ebm/documentation/debugging.md` (add inference failure entry)

---

## Appendix: Detailed Logs

### Experiment 001 Final Output
```
SINGLE-RULE EVALUATION RESULTS
--------------------------------------------------
distribute  : Accuracy=0.060, Invalid=0.000, DistImprove=-1.1%
combine     : Accuracy=0.063, Invalid=0.000, DistImprove=-4.1%
isolate     : Accuracy=0.061, Invalid=0.000, DistImprove=-2.6%
divide      : Accuracy=0.067, Invalid=0.000, DistImprove=2.5%
Average     : Accuracy=0.063
Average     : DistImprove=-1.3%
```

### Training Convergence Example (distribute, final steps)
```
[EnergyMonitor] Average energy gap (last 100 steps): 9.997, PosE=5.21, NegE=14.75, Margin=0.4633
[EnergyMonitor] Average energy gap (last 100 steps): 10.000, PosE=4.82, NegE=14.47, Margin=0.3456
```

### Known Issues from AUDIT-003
> "Root cause of low accuracy (distribute=2.1%, isolate=7.0%, divide=5.7%) is IRED inference converging to local minima in embedding space. combine works (100%) because output≈input. Requires inference strategy improvements (multi-start, momentum, more landscapes)."

**Post-DATAGEN Update**: combine no longer works (6.3%), suggesting DATAGEN-002 format change increased problem difficulty.
