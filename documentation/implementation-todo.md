# Implementation TODO - Inference Diagnostics

**Created**: 2026-02-16
**Phase**: DEBUG
**Focus**: IRED Inference Diagnostics and Improvements

---

## Overview

This TODO tracks implementation of inference diagnostics to understand why IRED fails (6% single-rule accuracy) despite successful training (9-10 unit energy gaps).

**Critical Question**: Why does inference produce essentially random results when training shows clear energy discrimination?

---

## Task Breakdown

### ✅ T1: Add Inference Logging Infrastructure

**Priority**: CRITICAL (blocks all other tasks)
**Time Estimate**: 2 hours
**File**: `src/algebra/algebra_evaluation.py`

**Implementation Details**:

1. Create `InferenceDiagnostics` class to track per-problem trajectories
2. Modify `AlgebraInference.solve_equation()` to accept `diagnostics` parameter
3. Modify `evaluate_model()` to accept `--enable_diagnostics` flag
4. Add logging to IRED inference loop

**Definition of Done**:
- [x] Logging integrated into inference loop (trajectory list in info dict)
- [x] `--enable_diagnostics` flag working in eval_algebra.py
- [x] JSON files generated with correct format in diagnostics_dir
- [x] Per-iteration data collected: energy, gradient_norm, embedding_distance, step_size, accepted

**Status**: COMPLETED (2026-02-16T23:30:00Z)

---

### ✅ T2: Implement Multi-Start Inference

**Priority**: HIGH
**Time Estimate**: 3 hours
**Files**: `src/algebra/algebra_evaluation.py`, `src/algebra/algebra_inference.py`
**Depends On**: T1

**Implementation Details**:
- Add `num_starts` parameter to `InferenceConfig`
- Loop N times with different random seeds
- Keep result with lowest final energy
- Log which start index won

**Definition of Done**:
- [x] `num_inference_starts` parameter added to evaluate_model
- [x] Multi-start loop implemented in evaluate_model
- [x] Best result selection by lowest energy working
- [x] Multi-start metadata tracked (winning_start_idx, best_energy)
- [x] Different random seeds per start (idx * 1000 + start_idx)

**Status**: COMPLETED (2026-02-16T23:30:00Z)

---

### ✅ T3: Make Iteration Count Configurable

**Priority**: HIGH
**Time Estimate**: 1 hour
**Files**: `eval_algebra.py`, `src/algebra/algebra_inference.py`
**Depends On**: T1

**Implementation Details**:
- Add `--max_inference_iterations` command-line argument
- Pass to InferenceConfig
- Log actual iteration count used

**Definition of Done**:
- [x] Already existed as `--inference_T` in eval_algebra.py
- [x] Maps to `max_iterations` in InferenceConfig via inference_params['T']
- [x] Can be configured via command-line or config files
- [x] Ready to test with 50, 100, 500 iterations in diagnostic experiments

**Status**: COMPLETED (2026-02-16T23:30:00Z)

---

### ☐ T4: Add Momentum to Gradient Descent (Optional)

**Priority**: MEDIUM
**Time Estimate**: 2 hours
**File**: `src/algebra/algebra_inference.py`
**Depends On**: T1-T3 (analyze results first)

**Implementation Details**:
- Add `momentum` parameter to `InferenceConfig`
- Modify IRED gradient update loop with velocity tracking
- Log velocity norms

**Definition of Done**:
- [ ] Momentum parameter added
- [ ] Velocity tracking implemented
- [ ] Tested on oscillating problems
- [ ] Measurable impact quantified

**Note**: Only implement if diagnostic experiments show gradient oscillation.

---

### ☐ T5: Implement Input-Guided Initialization (Optional)

**Priority**: MEDIUM
**Time Estimate**: 2 hours
**File**: `src/algebra/algebra_inference.py`
**Depends On**: T1-T3 (analyze results first)

**Implementation Details**:
- Add `use_input_init` parameter to `InferenceConfig`
- Initialize from input equation embedding instead of random
- Compare accuracy: random init vs input init

**Definition of Done**:
- [ ] Input init parameter added
- [ ] Initialization logic modified
- [ ] Tested on 100 problems
- [ ] Accuracy comparison quantified

**Note**: Only implement if random init shows high initial distance to target.

---

### ✅ T6: Create Diagnostic Experiment Configs

**Priority**: HIGH
**Time Estimate**: 1 hour
**Files**: `projects/algebra-ebm/configs/exp_diag_*.json`
**Depends On**: T1

**Create 4 Config Files**:
1. `exp_diag_baseline.json` - Current settings (50 iters, 1 start)
2. `exp_diag_multistart.json` - 10 starts, 50 iters each
3. `exp_diag_longrun.json` - 1 start, 500 iters
4. `exp_diag_combined.json` - 10 starts, 100 iters each

All use distribute rule on 100 problems with diagnostics enabled.

**Definition of Done**:
- [x] All 4 config files created in projects/algebra-ebm/configs/
- [x] JSON format validated
- [x] Configs reference correct model paths (/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results)
- [x] All configs enable diagnostics with appropriate output directories

**Status**: COMPLETED (2026-02-16T23:40:00Z)

---

### ✅ T7: Create Diagnostic SLURM Scripts

**Priority**: HIGH
**Time Estimate**: 1 hour
**Files**: `projects/algebra-ebm/slurm/eval_diag_*.sbatch`
**Depends On**: T6

**Create 4 Scripts**:
1. `eval_diag_baseline.sbatch` - 4 hour time limit
2. `eval_diag_multistart.sbatch` - 4 hour time limit
3. `eval_diag_longrun.sbatch` - 6 hour time limit (500 iterations)
4. `eval_diag_combined.sbatch` - 8 hour time limit (10 starts x 100 iterations)

All configured for gpu partition, 1 GPU, proper rsync back to home directory.

**Definition of Done**:
- [x] All 4 sbatch scripts created in projects/algebra-ebm/slurm/
- [x] Scripts configured with correct parameters matching config files
- [x] Proper GPU setup (module load cuda, nvidia-smi check)
- [x] Git clone workflow with commit checkout
- [x] Results synced back via rsync
- [x] Scripts made executable (chmod +x)
- [x] Ready for cluster submission
- [x] All 4 jobs submitted successfully (Job IDs: 60662646, 60662654, 60662655, 60662660)

**Status**: COMPLETED (2026-02-16T23:45:00Z)

---

### ☐ T8: Debug exp_007 Missing Results

**Priority**: LOW
**Time Estimate**: 3 hours
**Files**: `eval_algebra.py`
**Depends On**: None (can run in parallel)

**Investigation Steps**:
1. Read comparison evaluation logic
2. Check SLURM logs for exceptions
3. Add debug logging
4. Test locally
5. Re-run on cluster

**Definition of Done**:
- [ ] Root cause identified
- [ ] Fix implemented and tested
- [ ] exp_007 re-run produces results
- [ ] Monolithic baseline metrics available

---

### ☐ T9: Manual Dataset Verification

**Priority**: LOW
**Time Estimate**: 2 hours
**Files**: `results/test_datasets/*.json`
**Depends On**: None (independent)

**Verification Process**:
- Sample 10 problems per rule type
- Verify mathematical correctness with SymPy
- Check format matches training data
- Document findings

**Definition of Done**:
- [ ] Verification script created
- [ ] 70+ problems verified
- [ ] Results documented
- [ ] Confidence level determined

---

## Implementation Order

**Week 1** (First 3 days):
1. T1 (Logging) - Day 1
2. T2 (Multi-start) - Day 2
3. T3 (Iterations) - Day 2
4. T6 (Configs) - Day 3
5. T7 (SLURM scripts) - Day 3
6. Submit diagnostic experiments - Day 3

**Week 1** (Last 4 days):
7. T9 (Verification) - Parallel with experiments
8. Analyze diagnostic results
9. Decide on T4, T5 based on results

**Week 2**:
10. T8 (exp_007) - When time permits
11. Implement T4, T5 if warranted
12. Re-evaluate and iterate

---

## Success Metrics

### After T1-T3:
- Can visualize energy trajectories for failed problems
- Understand if stuck at init, oscillating, or slowly drifting
- Quantify local minima hypothesis
- Measure convergence rate

### After T4-T5:
- Momentum reduces oscillation (if applicable)
- Input init improves initial distance by >20%
- Combined improvements yield >10% absolute accuracy gain

### After T6-T7:
- 4 diagnostic experiments complete successfully
- Diagnostic logs available for analysis
- Clear understanding of failure modes
- Data-driven decision on next steps

### After T8:
- Monolithic baseline results available
- Fair comparison: compositional vs monolithic

### After T9:
- High confidence in test dataset correctness
- Rule out dataset errors as root cause

---

## Notes

### Why This Order?
- T1 is critical path - everything needs logging
- T2-T3 are quick wins if local minima hypothesis is correct
- T4-T5 are contingent on diagnostic results
- T6-T7 enable systematic experimentation
- T8-T9 are independent and can run in parallel

### What If Diagnostics Show No Improvement?
If multi-start and increased iterations don't help (< 5% improvement):
- Problem is not simple local minima
- May be decoder bottleneck
- May be energy landscape smoothness
- May need architectural changes
- Would trigger decision point: continue debugging or pivot

### What If Improvements Are Marginal (10-20%)?
- Single-rule goes from 6% to 15-25%
- Still far from target 85%
- Would need to assess if path to 85% is plausible
- May need combination of fixes
- Would inform 2-week decision point

### What If Improvements Are Large (>30%)?
- Single-rule goes from 6% to 35%+
- Confirms approach is viable with fixes
- Would continue with multi-rule composition
- High confidence in eventual success
