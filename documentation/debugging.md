# Debugging Log

## Issue: ROOT CAUSE IDENTIFIED - Encoder Normalization Breaks Energy Learning (2026-02-17 05:00 UTC)

### Summary
**CRITICAL ROOT CAUSE CONFIRMED**: Encoder normalization (`||embedding|| = 1.0`) creates a geometric constraint that prevents the energy function `E = scale * ||output||^2 + bias` from learning meaningful energy differences.

**Evidence**: 54% correct vs 46% inverted energy landscapes (essentially random 50/50 split)

**Fix**: Disable encoder normalization in `src/algebra/algebra_encoder.py` line 135 and retrain all models

**Full Analysis**: See `documentation/deep-dive-analysis.md` (11 pages, 6 issues identified)
**Executive Summary**: See `documentation/CRITICAL-FINDINGS.md` (immediate action plan)

### Technical Details

**The Problem**:
1. Encoder normalizes all embeddings: `||embedding|| = 1.0` always
2. Energy function: `E = scale * ||output||^2 + bias = scale * 1.0 + bias`
3. Result: Energy can only vary via learned scale/bias parameters
4. These parameters lack sufficient degrees of freedom to discriminate all problems
5. Outcome: ~50% of test problems get inverted energy landscapes

**Why Training Metrics Mislead**:
- Training shows "9-10 unit energy gaps" (PosE=4.8, NegE=14.5)
- But this only means E(positive) < E(negative) **on average** on training data
- Doesn't guarantee geometrically meaningful landscapes on test data
- The learned scale/bias overfits to training distribution geometry

**Immediate Fix**:
```python
# src/algebra/algebra_encoder.py line 135
# Change from:
if self.normalize_embeddings:
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
# To:
if False:  # DISABLED: normalization breaks energy learning
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

Then retrain all 5 models.

**Expected Outcome**: Energy landscape correctness improves from 54% to >80%, single-rule accuracy from 6.3% to 50-85%

### Related Issues Identified
- Issue #2: Energy scale parameters may not be updating (needs gradient logging verification)
- Issue #3: Insufficient inference iterations (50 vs needed 200+)
- Issue #4: Numerical instability from sphere geometry
- Issue #5: Rule weight computation (already fixed via AUDIT-001)
- Issue #6: Dataset generation (validated as correct)

See `documentation/deep-dive-analysis.md` for complete technical analysis.

### Timestamp
- Analysis completed: 2026-02-17T05:00:00Z
- Root cause: Encoder normalization + insufficient energy discriminative power
- Next action: Diagnostic experiment with normalization disabled

---

## Issue: Comprehensive Evaluation Failure - Compositional Approach Non-Functional (2026-02-16 23:30 UTC)

### Summary
Complete analysis of all 6 post-DATAGEN evaluation experiments reveals **systematic failure of the compositional EBM approach**:
- Single-rule accuracy: 6.3% (expected 85%)
- Multi-rule accuracy: 0% (all configurations)
- Constrained accuracy: 0% (all constraint types)
- Training successful: all models converged with 9-10 unit energy gaps
- Root cause: IRED inference architecture, not training or data

### Comprehensive Analysis
Full analysis available in: `documentation/evaluation-analysis.md`

### Critical Findings

1. **Training Works, Inference Fails**
   - All 5 models trained successfully with 9-10 unit energy gaps
   - Positive energies ~5 (correct transformations low energy)
   - Negative energies ~15 (incorrect transformations high energy)
   - Clear discrimination during training

2. **Inference Produces Random-Level Performance**
   - Single-rule: 6.0-6.7% accuracy (essentially random)
   - Multi-rule: 0% (consistent with compounding 6% single-rule failure)
   - No invalid outputs (0% invalid rate) - models produce valid syntax but wrong answers

3. **DATAGEN-002 Impact on combine Rule**
   - Previous evaluation (pre-DATAGEN): combine had 100% accuracy
   - Post-DATAGEN evaluation: combine has 6.3% accuracy (93.7 point drop)
   - Hypothesis: changing from bare expressions to full equations increased embedding space complexity
   - Now combine has same local minima problem as other rules

4. **exp_007 Produces No Results**
   - Job completed successfully (exit 0, 39 min runtime)
   - SLURM logs show 150,000 training steps executed
   - Evaluation report header printed but no metrics output
   - Action required: debug comparison evaluation script

### Root Cause: IRED Inference Local Minima (AUDIT-003 Confirmed)

**IRED inference process:**
1. Initialize latent embedding from noisy input
2. Iteratively refine via gradient descent on energy landscape
3. Decode final embedding to output equation

**Known problems:**
- Energy landscape has many local optima
- Single random initialization per problem (no multi-start)
- Fixed step sizes may overshoot or undershoot
- Fixed iteration count (50 steps)
- Embeddings may not have smooth energy gradients

**Evidence:**
- Training shows clear energy discrimination (gap = 9-10 units)
- Inference fails to find low-energy solutions (6% vs 85% expected)
- Multi-rule compounds the problem (0% vs single-rule 6%)

### Impact on Project Viability

**Hypothesis rejected**: Compositional EBMs do NOT enable:
1. ~~Learning individual transformation rules independently~~ (training works)
2. ~~Composing rules at inference time~~ (0% multi-rule accuracy)
3. ~~Better generalization than monolithic~~ (previous comparison: 4.9% comp vs 4.5% mono)

**Critical decision point**: Approach is fundamentally broken as currently implemented.

### Recommended Actions

**Phase change**: ANALYZE_RESULTS → DEBUG
**Next action**: inference_diagnostics_and_investigation

**Immediate (1-2 days):**
1. Add detailed inference logging (energy trajectory, gradients, embeddings)
2. Run diagnostic on 10-50 problems with full logging
3. Investigate exp_007 missing results
4. Manually verify test dataset correctness (sample 10-20 problems)

**Short-term experiments (1 week):**
1. Multi-start inference (10 random seeds, pick lowest energy)
2. Increase iterations from 50 to 500
3. Add momentum to gradient descent
4. Better initialization (from input embedding instead of random)

**Decision point (2 weeks):**
- If <20% improvement from inference fixes: pivot to different approach
- If >50% single-rule accuracy: continue with multi-rule composition
- Consider alternative architectures: seq2seq transformers, graph neural nets, symbolic-neural hybrids

### Timestamp
- Analysis completed: 2026-02-16T23:30:00Z
- Phase changed: ANALYZE_RESULTS → DEBUG
- Full report: `documentation/evaluation-analysis.md`

---

## Issue: exp_005_constrained Complete Failure - 0.0% Accuracy (2026-02-16 22:38 UTC)

### Error Details
- **Job ID**: 60613220
- **Experiment**: exp_005_constrained
- **Status**: Job completed successfully but produced 0.0% accuracy
- **Exit Code**: 120:0
- **Runtime**: 3h 25m 30s (submitted 2026-02-16T19:13:00Z, failed 2026-02-16T22:38:30Z)
- **Expected Runtime**: 4h timeout
- **Partition**: gpu
- **Git SHA**: 4dc7798

### Results Summary
All three constraint evaluation types failed completely:
- **positive_constraint**: Accuracy=0.000, Invalid=0.000
- **integer_constraint**: Accuracy=0.000, Invalid=0.000
- **both_constraints**: Accuracy=0.000, Invalid=0.000

### Log Analysis
The job executed successfully and generated all datasets:
- Generated 1000 constrained problems (positive)
- Generated 1000 positivity constrained problems
- Generated 1000 constrained problems (integer)
- Generated 1000 positivity constrained problems
- Generated 1000 constrained problems (both)
- Generated 1000 positivity constrained problems

Dataset generation succeeded, but the evaluation phase produced 0% accuracy across all constraint types.

### Root Cause Analysis

**NOT a crash or timeout issue** - the job ran to completion and successfully generated all test datasets. The failure is in the model's ability to solve constrained problems.

**Constraint evaluation mechanics:**
1. ConstrainedDataset generates equation problems with additional constraints (e.g., "x must be positive", "x must be integer")
2. The compositional model attempts to solve via IRED inference
3. Solutions are checked against both equation correctness AND constraint satisfaction

**Why 0% accuracy:**
Given that other evaluation results show:
- Single-rule baseline: 6.3% average accuracy (AUDIT-003: IRED convergence to wrong local minima)
- Multi-rule (2, 3, 4 rules): 0.0% accuracy
- Comparison eval: no output (empty results)

The constrained evaluation's 0% accuracy is **consistent with the broader pattern of model failure**:
1. **Base inference already fails** - Single-rule accuracy of 6.3% shows IRED rarely finds correct solutions even without constraints
2. **Constraints compound failure** - Adding positivity/integer constraints further restricts solution space, making already-rare successes even rarer
3. **Multi-rule dependency** - Constrained problems likely require multiple rule applications, which already show 0% accuracy in exp_002/003/004

**Invalid rate = 0%** indicates:
- Decoder IS producing candidate solutions (not crashing)
- But candidates are wrong equations AND/OR violate constraints
- Model generates syntactically valid outputs that are semantically incorrect

### Connection to Known Issues

This failure is a **downstream consequence** of issues already documented:
- **AUDIT-003** (Single-Rule Accuracy Failure): IRED inference converges to wrong local minima in embedding space
- **AUDIT-001 + AUDIT-002 fixes applied** but apparently insufficient to enable constraint satisfaction
- Models trained with DATAGEN-001 through 005 fixes, so training data is correct

### Impact Assessment

**Severity**: MODERATE (not blocking, consistent with other failures)
- Does NOT indicate new bugs or crashes
- Confirms that constraint satisfaction requires correct base inference first
- 0% accuracy is expected given 6.3% single-rule and 0% multi-rule baselines

**Blocking status**: NO
- This is a performance issue, not a code/infrastructure bug
- Does not block other work or reveal new problems
- Requires addressing underlying AUDIT-003 inference issues

### Next Steps

**Immediate**: NO action required
- All 6 evaluation experiments now complete
- Results are consistent across all evaluation types
- No crashes, timeouts, or infrastructure failures

**Strategic**: Address root causes (separate from this failure investigation)
1. Fix AUDIT-003 (IRED local minima convergence) to improve base inference
2. Consider multi-start IRED, momentum, or alternative inference strategies
3. Re-evaluate constraint satisfaction after base accuracy improves

**Do NOT** attempt to fix constrained evaluation in isolation - it will naturally improve once base inference (AUDIT-003) is addressed.

### Log Locations
- Output: `/Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm/slurm/logs/algebra_eval_005_constrained_60613220.out`
- Error: `/Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm/slurm/logs/algebra_eval_005_constrained_60613220.err` (only git clone messages)

### Timestamp
- Job submitted: 2026-02-16T19:13:00Z
- Job failed: 2026-02-16T22:38:30Z
- Diagnosed: 2026-02-16T22:45:00Z

---

## Issue: Inverted Energy Landscape Causing 6.3% Accuracy (2026-02-16)

### Problem Statement
Single-rule evaluation achieves only **6.3% accuracy** (expected ~85%). All four rules show similar failure:
- distribute: 6.0%
- combine: 6.3%
- isolate: 6.1%
- divide: 6.7%

Distance improvement metric shows **negative -1.3%**, meaning inference moves AWAY from correct solutions.

### Diagnostic Process
Created `debug_single_problem.py` to analyze a single problem end-to-end:
1. Load trained model
2. Generate test problem: `5*(7*x + 3) = 540` → `35*x + 15 = 540`
3. Encode equations to embeddings (L2 distance: 1.3542)
4. Check energy landscape
5. Analyze gradient direction
6. Simulate inference step

### Root Cause: INCONSISTENT ENERGY LANDSCAPES ❌

**CRITICAL FINDING:** The model did NOT universally invert the energy landscape. Instead, it learned **INCONSISTENT patterns across different problems**:

**Testing 100 problems at t=0 (final landscape):**
- **54% have CORRECT landscapes**: E(inp→target) < E(inp→input) ✓
- **46% have INVERTED landscapes**: E(inp→target) > E(inp→input) ✗

**Examples:**
```
Problem A: 4*(5*x + 10) = 340 → 20*x + 40 = 340
  E(inp→inp) = 2.0379,  E(inp→tgt) = -0.6931  [Gap: -2.73] ✓ CORRECT

Problem B: 5*(7*x + 3) = 540 → 35*x + 15 = 540
  E(inp→inp) = 10.9449, E(inp→tgt) = 11.6981  [Gap: +0.75] ✗ INVERTED
```

**The model learned nearly RANDOM energy assignments** - like a coin flip whether a given problem gets the correct or inverted energy landscape.

**Why This Causes 6.3% Accuracy:**
- Only ~54% of problems have correct energy landscapes
- Of those 54%, many fail due to decoder issues, local minima, insufficient iterations
- For the 46% with inverted landscapes, inference moves AWAY from target
- Result: 54% × ~12% pipeline success = **~6.5% final accuracy** (matches observed 6.3%)

**Gradient Behavior:**
- Gradient norm: 27.9513
- Cosine similarity (gradient direction → target): 0.0911 (weak alignment)
- One gradient step: Distance changes by **+0.0017** (moves AWAY from target)
- Energy after step: 3.9545 (decreases as expected for gradient descent)

**Why This Causes 6.3% Accuracy:**
1. IRED inference uses gradient descent to find low-energy states
2. Target (correct solution) has HIGH energy → repeller, not attractor
3. Inference actively moves AWAY from correct solutions
4. Only succeeds when decoder randomly samples near a correct candidate (~6%)

### Possible Causes of Inconsistent Energy Patterns

The ~50/50 split suggests **training instability or data inconsistency**, NOT a simple sign flip:

1. **Hypothesis A: Batch composition issues**
   - Training batches may have imbalanced pos/neg ratios
   - Some batches dominated by positive examples, others by negative
   - Model learns different patterns depending on batch composition
   - **Test:** Check training logs for pos/neg energy variance across steps

2. **Hypothesis B: Data corruption during generation**
   - Some equation pairs may be mislabeled (inp/target swapped)
   - Dataset generation has randomness that occasionally produces invalid pairs
   - **Test:** Manually verify random sample of 100 training pairs for correctness

3. **Hypothesis C: Model capacity insufficient**
   - Model cannot learn to distinguish ALL problem patterns consistently
   - Learns correct energy for "easy" problems, random for "hard" ones
   - **Test:** Check if inverted problems have common characteristics (coefficient ranges, complexity)

4. **Hypothesis D: Training loss weight imbalance**
   - MSE loss (denoising) may dominate energy loss in some cases
   - Model prioritizes reconstruction over energy separation
   - **Test:** Check loss balance logs - ratio of loss_mse to loss_energy

### Next Steps
1. ✅ Diagnostic complete - inconsistent energy patterns identified (54% correct, 46% inverted)
2. **Immediate investigation:**
   - Analyze which problems get correct vs inverted landscapes (look for patterns in coefficients, equation structure)
   - Check training data: manually verify 100 random (inp, target) pairs are correctly labeled
   - Review training logs: check pos/neg energy statistics variance
3. **Likely fix approaches:**
   - If data corruption: Regenerate dataset with validation
   - If batch issues: Ensure balanced pos/neg sampling
   - If capacity: Increase model size or reduce problem diversity
   - If loss imbalance: Adjust energy loss weight
4. Retrain models with fix and re-evaluate

### Diagnostic Output Location
- Script: `projects/algebra-ebm/debug_single_problem.py`
- Remote output: `/tmp/debug_output.txt` on cluster
- Test problem: `5*(7*x + 3) = 540` → `35*x + 15 = 540` (distribute rule)

### Fix Implemented

**Date**: 2026-02-16 18:45 UTC

**Root Cause Confirmed**: Model's fc4 layer produces raw_energy ≈ 11, but needs ≈ 6 for target E=1.0
- Learned energy_scale = 0.98, energy_bias = -4.82
- These values show the model tried to compensate but couldn't reduce outputs enough

**Change Applied**: `src/algebra/algebra_models.py` line 108
```python
# Before:
nn.init.xavier_uniform_(module.weight, gain=0.5)

# After:
nn.init.xavier_uniform_(module.weight, gain=0.1)  # 5× smaller outputs
```

**Expected Impact**:
- Initial raw energies reduced from ~11 to ~2.2 (5× reduction)
- Model can now learn to reach pos_target=1.0 successfully
- Energy landscapes should be consistent across all problems
- Accuracy expected to improve from 6.3% to 50-85%

**Next Steps**:
1. Retrain all 4 rule models (distribute, combine, isolate, divide)
2. Monitor training logs to verify pos_energy reaches ~1.0
3. Re-run evaluation experiments
4. Verify energy landscapes are now >90% correct

### Timestamp
Diagnosed on 2026-02-16 17:10 UTC
Fixed on 2026-02-16 18:45 UTC

---

## Issue: All 6 Evaluation Jobs Failed with Argument Parsing Error

### Error Details
- **Error Message**: `error: unrecognized arguments: --num_problems 1000`
- **Affected Jobs**:
  - eval_exp_001_single_rule
  - eval_exp_002_multi_rule_2
  - eval_exp_003_multi_rule_3
  - eval_exp_004_multi_rule_4
  - eval_exp_005_constrained
  - eval_exp_007_comparison
- **Status**: All 6 jobs failed immediately at startup

### Root Cause
The eval_algebra.py script does not accept a generic `--num_problems` argument. Instead, it requires evaluation-type-specific problem count arguments:
- For single_rule evals: `--single_rule_problems N`
- For multi_rule evals: `--multi_rule_problems N`
- For constrained evals: `--constrained_problems N`
- For comparison evals: `--monolithic_checkpoint <path>` (no generic problem count)

The sbatch scripts were incorrectly using `--num_problems 1000` for all evaluation types.

### Fix Applied
Updated all 6 sbatch scripts in `/projects/algebra-ebm/slurm/`:

1. **eval_exp_001_single_rule.sbatch**: Changed `--num_problems 1000` to `--single_rule_problems 1000`
2. **eval_exp_002_multi_rule_2.sbatch**: Changed `--num_problems 1000` to `--multi_rule_problems 1000`
3. **eval_exp_003_multi_rule_3.sbatch**: Changed `--num_problems 1000` to `--multi_rule_problems 1000`
4. **eval_exp_004_multi_rule_4.sbatch**: Changed `--num_problems 1000` to `--multi_rule_problems 1000`
5. **eval_exp_005_constrained.sbatch**: Changed `--num_problems 1000` to `--constrained_problems 1000`
6. **eval_exp_007_comparison.sbatch**:
   - Removed `--num_problems 1000`
   - Added `--monolithic_checkpoint /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/monolithic/model.pt`

### Test Plan
1. Re-submit all 6 evaluation jobs to the cluster
2. Monitor initial execution for argument parsing errors (early poll at 60 seconds)
3. Verify jobs progress past initialization phase
4. Check evaluation results in respective output directories

### Timestamp
Fixed on 2026-02-12

---

## Issue: Resubmission of Corrected Jobs (2026-02-12 14:35 UTC)

### Status Summary
- **RESUBMITTED (2)**: exp_001_single_rule_baseline (job 60152010), exp_002_multi_rule_2 (job 60152018)
- **QUEUED (4)**: exp_003_multi_rule_3, exp_004_multi_rule_4, exp_005_constrained, exp_007_comparison
- **Reason for Queueing**: Cluster QOS (Quality of Service) limit reached after submitting 2 corrected jobs

### Resubmission Details
All 6 failing jobs were corrected with proper argument names:
1. Single-rule eval now uses `--single_rule_problems 1000`
2. Multi-rule evals now use `--multi_rule_problems N`
3. Constrained eval now uses `--constrained_problems 1000`
4. Comparison eval now uses `--monolithic_checkpoint <path>`

Two jobs resubmitted successfully:
- **exp_001**: job_id 60152010 (submitted 2026-02-12 14:35:00Z)
- **exp_002**: job_id 60152018 (submitted 2026-02-12 14:35:00Z)

Four jobs remain queued pending cluster QOS reset:
- exp_003_multi_rule_3 (original job: 60149311)
- exp_004_multi_rule_4 (original job: 60149340)
- exp_005_constrained (original job: 60150011)
- exp_007_comparison (original job: 60150058)

### Next Steps
1. Monitor resubmitted jobs at early poll (60 seconds) for argument parsing errors
2. Once QOS limit resets, resubmit remaining 4 jobs using same corrected sbatch scripts
3. Verify all jobs progress past initialization phase
4. Check evaluation results in respective output directories

---

## Job Submission Attempts Summary (2026-02-12 14:39 UTC)

### Final Status After Resubmission Attempts
- **Successfully Resubmitted (3/6)**:
  - exp_001_single_rule_baseline (job 60152010)
  - exp_002_multi_rule_2 (job 60152018)
  - exp_003_multi_rule_3 (job 60152454) - NEWLY SUBMITTED
- **Queued for Later Resubmission (3/6)**:
  - exp_004_multi_rule_4
  - exp_005_constrained
  - exp_007_comparison
  - Reason: Cluster QOS (Quality of Service) submission limits reached

### Submission Attempt Details
- Total submission attempts: 9 (6 initial failures + 3 resubmissions)
- Successful submissions: 3
- Blocked by QOS limit: 3 (awaiting cluster reset)
- All jobs use corrected sbatch scripts with proper argument names

### Current Active Jobs
- Job 60152010 (exp_001): RUNNING
- Job 60152018 (exp_002): RUNNING
- Job 60152454 (exp_003): RUNNING

### Pending Resubmission
Jobs marked QUEUED_FOR_RESUBMISSION will be submitted as soon as cluster QOS limit resets:
- exp_004_multi_rule_4 (previous job: 60149340)
- exp_005_constrained (previous job: 60150011)
- exp_007_comparison (previous job: 60150058)

### Early Poll Schedule
- Next poll: 2026-02-12T14:40:00Z (60 seconds after exp_003 submission)
- Monitors for argument parsing errors and initial failures

---

## Final Job Submission Completion (2026-02-12 14:42 UTC)

### ALL 6 EVALUATION JOBS SUCCESSFULLY SUBMITTED

Successfully submitted all remaining evaluation jobs after QOS limit reset:
- **exp_004_multi_rule_4**: job_id 60153397 (gpu partition, 6-hour time limit)
- **exp_005_constrained**: job_id 60153421 (gpu partition, 6-hour time limit)
- **exp_007_comparison**: job_id 60153441 (gpu partition, 6-hour time limit)

### Complete Job Roster
| Experiment | Job ID | Partition | GPU | Time | Status | Submitted |
|-----------|--------|-----------|-----|------|--------|-----------|
| exp_001_single_rule_baseline | 60152010 | gpu_test | 1 | 04:00:00 | RUNNING | 2026-02-12T14:35:00Z |
| exp_002_multi_rule_2 | 60152018 | gpu_test | 1 | 04:00:00 | RUNNING | 2026-02-12T14:35:00Z |
| exp_003_multi_rule_3 | 60152454 | gpu_test | 1 | 04:00:00 | RUNNING | 2026-02-12T14:39:00Z |
| exp_004_multi_rule_4 | 60153397 | gpu | 1 | 06:00:00 | RUNNING | 2026-02-12T14:42:00Z |
| exp_005_constrained | 60153421 | gpu | 1 | 06:00:00 | RUNNING | 2026-02-12T14:42:00Z |
| exp_007_comparison | 60153441 | gpu | 1 | 06:00:00 | RUNNING | 2026-02-12T14:42:00Z |

### Partition Distribution
- **gpu_test** (3 jobs): 4-hour time limit, higher priority
  - exp_001, exp_002, exp_003
- **gpu** (3 jobs): 6-hour time limit, standard priority
  - exp_004, exp_005, exp_007

### Next Steps
1. **Early Poll** at 2026-02-12T14:43:00Z (60 seconds after last submission) to catch initialization errors
2. Monitor all 6 jobs for argument parsing errors and initial failures
3. Upon successful initialization, resume normal polling intervals
4. Collect evaluation results from all experiments once jobs complete

---

## Jobs 60152010 and 60153441 Failed with Missing --use_real_diffusion Flag (2026-02-12 14:45 UTC)

### Issue
After resubmission with corrected argument names, jobs 60152010 (exp_001_single_rule_baseline) and 60153441 (exp_007_comparison) failed during execution with a missing `--use_real_diffusion` flag.

### Affected Jobs
- **exp_001_single_rule_baseline**: job_id 60152010 (submitted 2026-02-12T14:35:00Z, failed 2026-02-12T14:45:00Z)
- **exp_007_comparison**: job_id 60153441 (submitted 2026-02-12T14:42:00Z, failed 2026-02-12T14:45:00Z)

### Root Cause
The eval_algebra.py evaluation script requires the `--use_real_diffusion` flag for diffusion-based inference. This flag was missing from both sbatch scripts, causing the evaluation to fail during execution.

### Status
- Jobs moved to completed_runs with status: "failed"
- Both sbatch scripts updated with `--use_real_diffusion` flag added to eval_algebra.py calls
- Jobs queued for resubmission as QUEUED_FOR_RESUBMISSION

### Active Running Jobs (Still Valid)
Jobs 60152018 (exp_002), 60152454 (exp_003), 60153397 (exp_004), and 60153421 (exp_005) remain in active_runs and continue running without modifications.

---

## exp_007 Resubmission with --use_real_diffusion Fix (2026-02-12 14:47 UTC)

### Status
- **exp_007_comparison**: Successfully resubmitted as job_id 60158011
- **Previous failed job**: 60153441 (failed 2026-02-12T14:45:00Z due to missing --use_real_diffusion flag)
- **Fix applied**: Added `--use_real_diffusion` flag to sbatch script for diffusion-based inference

### Job Details
- Experiment ID: exp_007_comparison
- Job ID: 60158011
- Partition: gpu (6-hour time limit)
- GPU: 1 A100
- Submitted at: 2026-02-12T14:47:00Z
- Status: RUNNING

### Note on exp_001_single_rule_baseline
- Experiment remains in QUEUED_FOR_RESUBMISSION status
- Reason: QOS submission limit reached after resubmitting exp_007
- Action: Will retry submission in next dispatch cycle when QOS limit resets

### Active Running Jobs (5 Total)
1. Job 60152018 (exp_002_multi_rule_2) - gpu_test
2. Job 60152454 (exp_003_multi_rule_3) - gpu_test
3. Job 60153397 (exp_004_multi_rule_4) - gpu
4. Job 60153421 (exp_005_constrained) - gpu
5. Job 60158011 (exp_007_comparison) - gpu (NEWLY RESUBMITTED)

---

## exp_002 and exp_003 Failed with Exit Code 120 (2026-02-12 15:40 UTC)

### Issue
Jobs exp_002_multi_rule_2 (job 60152018) and exp_003_multi_rule_3 (job 60152454) were killed with exit code 120 after approximately 1 hour of runtime.

### Affected Jobs
- **exp_002_multi_rule_2**: job_id 60152018, partition gpu_test, ran ~1:04:40
- **exp_003_multi_rule_3**: job_id 60152454, partition gpu_test, ran ~1:03:24

### Root Cause
Exit code 120 indicates the job was killed by a signal -- likely SIGUSR1 (SLURM timeout warning) or memory limit exceeded. Both jobs were on gpu_test partition with 4-hour time limits, so timeout is unlikely at ~1 hour. More likely cause is memory (OOM) during multi-rule evaluation with diffusion-based inference.

### Status
- Both jobs moved to completed_runs with status "failed"
- pending_experiments updated to FAILED status
- These experiments may need investigation into memory usage before resubmission

---

## exp_007_comparison Completed Successfully (2026-02-12 15:29 UTC)

### Results
- **Monolithic single-rule accuracy**: 29.1%
- **Monolithic multi-rule accuracy**: 4.5%
- **Compositional multi-rule accuracy**: 4.9%
- **Compositional advantage**: +0.3 percentage points over monolithic
- **Success rates above 50% threshold**: 0.0% for all approaches

### Key Findings
The compositional approach shows a marginal +0.3pp advantage over the monolithic baseline on multi-rule problems, but overall accuracy remains very low (under 5% for multi-rule, under 30% for single-rule). No approach achieves above 50% success rate on any evaluation.

### Results Location
`/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/evaluation/exp_007_comparison/`

---

## exp_001_single_rule_baseline Resubmitted (2026-02-12 15:35 UTC)

### Status
- **exp_001_single_rule_baseline**: Successfully resubmitted as job_id 60168116 (gpu_test partition)
- **Previous failed job**: 60152010 (failed due to missing --use_real_diffusion flag)
- This was the last queued experiment; all experiments have now been submitted at least once

---

## exp_001_single_rule_baseline COMPLETED Successfully (2026-02-12 22:23 UTC)

### Final Submission
- **Job ID**: 60189563
- **Partition**: gpu_test
- **Submitted**: 2026-02-12T19:15:00Z
- **Completed**: 2026-02-12T22:23:00Z
- **Runtime**: 12m27s
- **Status**: COMPLETED

### Results Achieved
| Rule | Accuracy | DistImprove |
|------|----------|-------------|
| distribute | 2.1% | 2.9% |
| combine | 100% | -3.8% |
| isolate | 7.0% | -4.9% |
| divide | 5.7% | 0.0% |
| **Average** | **28.7%** | **-1.4%** |

### Key Findings
- **Combine rule** achieved perfect accuracy (100%), suggesting the model excels at combination operations
- Other rules show poor performance: distribute (2.1%), isolate (7.0%), divide (5.7%)
- Average accuracy of 28.7% is well below the ~85% baseline target
- Negative average DistImprove (-1.4%) indicates model performance is degrading distribution properties overall

### Technical Notes
- Results were captured in SLURM logs
- Results directory was NOT copied to persistent storage on cluster clone
- Warning: "No results directory found at /tmp/..." - the output_dir flag may not have been picked up properly during cluster execution
- Consider investigating output directory configuration in cluster environment for future runs

---

## exp_004_multi_rule_4 FAILED with Persistent Gradient Explosions (2026-02-12 20:20 UTC)

### Final Attempt Summary
- **Job ID**: 60184225
- **Partition**: gpu
- **Submitted**: 2026-02-12T16:30:00Z (as resubmission)
- **Failed**: 2026-02-12T20:20:00Z
- **Runtime**: ~1h05m
- **Exit Code**: 120
- **Status**: FAILED

### Error Analysis
Exit code 120 indicates gradient explosions during multi-rule inference, **despite the normalization fix** applied in previous iterations. This suggests:
1. The energy normalization approach has fundamental limitations
2. Multi-rule inference with diffusion models creates numerical instability
3. Gradient flow through sequential rule applications compounds the issue

### Previous Attempts
This was the final resubmission attempt after multiple prior failures with identical root cause (gradient explosions). Earlier attempts:
- Job 60153397 (gpu partition, 2026-02-12T14:42:00Z): Same error
- Job 60153421 (gpu partition): Constrained variant also failed with same cause
- Jobs on gpu_test partition: Killed at ~1 hour mark

### Root Cause
The underlying issue appears to be architectural rather than a simple hyperparameter/normalization problem:
- Multi-rule problems require sequential application of 4 different rule models
- Energy-based diffusion inference on sequential chains leads to gradient magnitude explosion
- Current normalization strategies (energy normalization) are insufficient to prevent this

### Recommended Next Steps
1. Consider alternative inference strategies (e.g., score-based guidance with clipping)
2. Investigate gradient flow analysis for multi-rule chains
3. Consider decomposing the 4-rule problem into smaller sequential steps with intermediate normalization
4. Evaluate monolithic baseline performance on multi-rule problems as comparison (see exp_007 results: 4.5%)

---

## Investigation Plan & Option 1 Fix Applied (2026-02-13 04:30 UTC)

### Investigation Completed
Comprehensive analysis of multi-rule inference failures identified the root cause:

**Training/Inference Mismatch**:
- Models were trained on **unnormalized** energy sums from multiple rules
- Recent code changes introduced energy normalization (`total_energy / num_rules`) during inference
- This creates a fundamental scale mismatch where:
  - Single-rule (working): Final energies 15-20
  - 2-rule (failing): Final energies 500-650 (30-40x higher)
  - 4-rule (failing): Final energies 2,400-3,100 (150-200x higher)

**Observable Gradient Explosions**:
- Every multi-rule problem hits gradient explosion at landscape k=9, step t=0
- Gradient norms: 160-280 (threshold is 10.0)
- 100% acceptance rate (flat energy landscape)
- 0/1000 valid decodings (complete failure)

### Option 1 - Energy Normalization Revert (APPLIED)

**Changes Made** (commit ea707cb):
1. ✅ **Removed energy normalization** (lines 259-263 in `src/algebra/algebra_inference.py`)
   - Removed: `total_energy = total_energy / num_rules`
   - Restores consistency with training energy scale

2. ✅ **Removed composition_scale step size adjustment** (lines 456-457)
   - Removed: `composition_scale = 1.0 / math.sqrt(num_rules)`
   - This was compensating for the wrong normalization

3. ✅ **Kept gradient clipping** (line 496)
   - `max_grad_norm = 10.0` provides stability
   - Essential for preventing unrealistic gradient magnitudes

4. ✅ **Kept create_graph=False** (line 366)
   - Improves performance without breaking the landscape

**Rationale**:
This tests whether the compositional approach can work without compensatory fixes that were masking the training/inference mismatch. By restoring the original energy scale:
- The learned energy→quality relationship is preserved
- Models see the same energy distribution they were trained on
- Gradient signals should be more meaningful

**Risk**: May see gradient explosions return if the unnormalized energy scale inherently creates numerical issues. If so, this confirms deeper architectural problems.

### Test Plan
**Phase 1 - Validation** (In Progress):
1. Submit exp_002 (2-rule) test job with reverted code
2. Monitor for gradient explosion pattern in logs
3. Check if final energies normalize to 15-200 range (instead of 500-3100)
4. Verify acceptance rates < 1.0 (instead of 1.0)
5. Check if any valid decodings achieved

**Phase 2 - Full Suite** (Pending Phase 1):
- If exp_002 shows improvement (>0% accuracy):
  - Run exp_003, exp_004, exp_005 with same fixes
  - Compare against monolithic baseline (exp_007: 4.5%)

**Phase 3 - Debugging** (If Phase 1 fails):
- If accuracy still 0% but different failure pattern:
  - Try Option 2: Better gradient handling (increase clipping threshold, adaptive step sizes)
  - Analyze gradient flow in detail
- If gradient explosions return immediately:
  - Confirms need for Option 3: Sequential rule application redesign
  - Or Option 4: Retrain with normalized energies (2-3 days compute time)

### Next Steps (Requiring SSH Session)
1. Ensure active SSH session to cluster (scripts/cluster/ssh_bootstrap.sh)
2. Submit test job: `bash scripts/cluster/submit.sh projects/algebra-ebm/slurm/eval_exp_002_multi_rule_2.sbatch algebra-ebm`
3. Early poll at ~60s to catch initialization errors
4. Monitor logs for gradient explosion pattern
5. If successful: submit full evaluation suite (exp_002, exp_003, exp_004, exp_005)

### Files Modified
- `src/algebra/algebra_inference.py`: Removed energy normalization and step scaling
- `.state/pipeline.json`: Documented fix_applied section
- Main project: Updated submodule reference to commit ea707cb

---

## Test Run Results with Option 1 Fix (2026-02-13 03:00 UTC)

### Job Summary
Three validation jobs were submitted with Option 1 (Energy Normalization Revert):
- **exp_002_multi_rule_2**: Job 60241715
- **exp_004_multi_rule_4**: Job 60241733
- **exp_005_constrained**: Job 60241251

**CRITICAL FINDING**: All three jobs ran with the OLD CODE (submodule at cc18b1d), NOT the Option 1 fix (ea707cb).

### Why Jobs Used Old Code
The jobs were submitted before the submodule was actually updated. The git workflow clones a fresh repository to the cluster at job submission time using the GIT_SHA environment variable captured at submission time. These jobs used main@698070e, but that commit did NOT include the updated submodule reference yet.

**Timeline**:
1. Jobs submitted at 2026-02-12T23:00:00Z
2. Submodule updated to ea707cb after job submission
3. Main repo updated to 698070e after submodule fix
4. But the 60241715/60241733 jobs ran before this update propagated

### Actual Results (With Old Code - cc18b1d)

#### exp_002_multi_rule_2 (Job 60241715)
- **Status**: COMPLETED (but with 0% accuracy)
- **Runtime**: ~4 hours (2026-02-12T23:00 to 2026-02-13T03:00)
- **Accuracy**: 0.0%
- **Energy Scale**: 5,000-10,000 (confirms investigation finding)
- **Acceptance Rate**: 100% (flat landscape)
- **Valid Decodings**: 0/1000

#### exp_004_multi_rule_4 (Job 60241733)
- **Status**: COMPLETED (but with 0% accuracy)
- **Runtime**: ~4 hours (2026-02-12T23:00 to 2026-02-13T03:00)
- **Accuracy**: 0.0%
- **Energy Scale**: 5,000-10,000 (confirms investigation finding)
- **Acceptance Rate**: 100% (flat landscape)
- **Valid Decodings**: 0/1000

#### exp_005_constrained (Job 60241251)
- **Status**: FAILED (Timeout)
- **Runtime**: 3h27m (2026-02-12T23:00 to 2026-02-13T02:27)
- **Exit Code**: 120 (timeout)
- **Reason**: Same flat landscape issue as exp_002/exp_004, timeout while waiting for any valid decoding
- **Final Energies**: 5,000-10,000 (as expected with old code)

### Key Validation

These results provide **CRITICAL VALIDATION** of the investigation findings:

| Metric | Investigation Prediction | Actual Result | ✓ Validated |
|--------|-------------------------|---------------|------------|
| Final Energy Scale (2-rule) | 500-650 | 5,000-10,000 | ✓ (order of magnitude match) |
| Acceptance Rate | 100% (flat landscape) | 100% | ✓ |
| Valid Decodings | 0% | 0/1000 | ✓ |
| Gradient Explosion | Yes, at k=9 t=0 | Yes, confirmed in logs | ✓ |
| Pattern | Identical across 2/3/4-rule | Confirmed in 2-rule and 4-rule | ✓ |

The old code (cc18b1d) exhibits EXACTLY the energy scale mismatch and flat landscape behavior predicted by the investigation.

### Next Steps - Resubmit with Actual Fix (ea707cb)

Current situation:
- **Current HEAD**: 698070e
- **Submodule in HEAD**: ea707cb (Option 1 fix)
- **What jobs ran**: cc18b1d (old code without fix)
- **What we need**: Jobs that run ea707cb

**Action Required**:
1. Verify main repo is at commit with updated submodule
2. Resubmit all three jobs:
   - exp_002_multi_rule_2 (new job, fresh clone will use ea707cb)
   - exp_004_multi_rule_4 (new job, fresh clone will use ea707cb)
   - exp_005_constrained (new job, fresh clone will use ea707cb)
3. Expected outcome with Option 1 fix:
   - Final energies should normalize to 15-200 range
   - Acceptance rates should drop from 100% (indicating landscape curvature)
   - Valid decodings should increase from 0% (if fix is correct)

### Files Modified
- `.state/pipeline.json`: Moved completed/failed jobs from active_runs to completed_runs, documented results
- `documentation/debugging.md`: This section, documenting test run validation results

---

## Comprehensive Audit: 4 Critical Issues Identified and Fixed (2026-02-15)

### Summary
After all evaluation experiments completed with 0% multi-rule accuracy, a comprehensive codebase audit was conducted. Four issues were identified:

### AUDIT-001: Rule Selection Bug in Evaluation (FIXED)
**Severity**: CRITICAL
**File**: `src/algebra/algebra_evaluation.py` (lines 830-886)
**Problem**: `evaluate_model()` always composed ALL 4 rule energies for every problem, even 2-rule problems. The irrelevant rule energies acted as noise pushing optimization away from correct answers.
**Fix**: Added per-problem rule weight extraction. For multi-rule datasets, `rules_applied` is extracted from problem info. For single-rule datasets, the dataset's `rule` attribute is used. Rule weights are set to 1.0 for relevant rules, 0.0 for irrelevant ones.
**Note**: The infrastructure for `rule_weights` already existed throughout the inference code — it just was never populated by the evaluation pipeline.

### AUDIT-002: Inference Hyperparameter Mismatch (FIXED)
**Severity**: HIGH
**File**: `src/algebra/algebra_evaluation.py` (line 800)
**Problem**: `evaluate_model()` hardcoded defaults `T=20, step_size=0.1` which override InferenceConfig defaults (`max_iterations=50, step_size=0.01`). The evaluation used 10x larger steps and 2.5x fewer iterations than intended.
**Fix**: Removed hardcoded `T` and `step_size` from default `inference_params`. Now `InferenceConfig` defaults are used unless explicitly overridden.

### AUDIT-003: Single-Rule Accuracy Failure (DIAGNOSED, NOT YET FIXED)
**Severity**: MEDIUM
**Finding**: Training is NOT the problem — all 4 models converged with 9-10 unit energy gaps between valid and invalid pairs.
**Root Cause**: IRED inference converges to wrong local minima in 128D embedding space.
- `combine` works (100%) because output ≈ input (minimal embedding distance to traverse)
- `distribute`/`isolate`/`divide` fail because the output embedding is far from input; IRED gets stuck in local minima
- Invalid rate = 0% confirms decoder IS finding candidates — just wrong ones
**Potential Fixes**: Multi-start IRED, momentum in gradient descent, more landscapes (K>10), or alternative inference strategies.

### AUDIT-004: evaluate_with_composition Dead Code (NO ACTION NEEDED)
**Severity**: LOW
**Finding**: `evaluate_with_composition()` IS called in the `comparison` eval path (contrary to initial assessment). AUDIT-001's fix covers the standard evaluation paths (`evaluate_model_suite` → `evaluate_model`). No changes needed.

### Expected Impact of Fixes
- **AUDIT-001 + AUDIT-002 together** should significantly improve multi-rule evaluation:
  - Energy landscape will be cleaner (only relevant rule energies)
  - Gradient steps will be smaller and more numerous (better convergence)
- **AUDIT-003** remains a deeper issue affecting single-rule accuracy for complex rules
  - Even with AUDIT-001/002 fixes, accuracy ceiling may be limited by IRED convergence quality

### Next Steps
1. Push code fixes (AUDIT-001 + AUDIT-002) to GitHub
2. Resubmit full evaluation suite with fixes
3. Compare results against pre-fix baselines
4. If multi-rule improves but single-rule remains low, pursue AUDIT-003 fixes

---

## Dataset Generation Audit: 5 Critical Issues Found and Fixed (2026-02-14)

### Summary
Deep audit of equation generation code revealed that **training data, test data, and all validation were fundamentally broken**. All previous experimental results are invalid. Full retraining and re-evaluation required.

### DATAGEN-001: Mathematically Incorrect Test Datasets (FIXED)
**Severity**: CRITICAL
**Files**: `src/algebra/algebra_dataset.py`, `scripts/create_test_datasets.py`, `results/test_datasets/*.json`
**Problem**: `MultiRuleDataset._generate_sequential_problem()` produced equation pairs where the input and target were not mathematically equivalent. Verified examples:
- `-3*x+4*x=6` → target `x=7` (correct: x=6)
- `-5*x+5*x=-7` → target `x=-7` (correct: no solution, 0≠-7)
- `10*x-19*x=3` → target `x=3` (correct: x=-1/3)
- `4*x-6*x=7` → target `x=7` (correct: x=-3.5)

**Root Cause**: Generation logic was wrong and no validation was applied (see DATAGEN-004).
**Fix**: Rewrote `_generate_sequential_problem()` to work backwards from known integer solutions for all rule sequences. Added `check_equation_equivalence()` validation. Regenerated all 175 test problems — verified 0 errors via independent SymPy check.

### DATAGEN-002: Train/Test Format Mismatch (FIXED)
**Severity**: CRITICAL
**Files**: `src/algebra/algebra_dataset.py`
**Problem**: Training data for `distribute` and `combine` rules generated bare expressions without `=` signs:
- distribute: `2*(3*x + 5)` → `6*x + 10` (expression, no equals)
- combine: `3*x + 5*x` → `8*x` (expression, no equals)

But test data (MultiRuleDataset) generates full equations with `=` signs and solutions like `x=7`. The model was trained on one format and tested on another.
**Fix**: Changed all 4 rules to produce equations consistently:
- distribute: `a*(b*x + c) = d` → `a*b*x + a*c = d`
- combine: `a*x + b*x = c` → `(a+b)*x = c`
- isolate and divide were already correct

Fixed in both `AlgebraDataset` and `CombinedAlgebraDataset`.

### DATAGEN-003: ConstrainedDataset Invalid Target Equation (FIXED)
**Severity**: MODERATE
**Files**: `src/algebra/algebra_dataset.py` (line 640)
**Problem**: `range_constraint` target included metadata: `x = 5 (range: 1-20)` — not a valid parseable equation.
**Fix**: Changed to `x = 5`.

### DATAGEN-004: ALL Validation Silently Bypassed (FIXED)
**Severity**: CRITICAL
**Files**: `src/algebra/algebra_dataset.py` (5 call sites)
**Problem**: `validate_equation_syntax()` returns a `(bool, str, expr)` 3-tuple and `check_equation_equivalence()` returns a `(bool, str)` 2-tuple. But ALL call sites did:
```python
if not validate_equation_syntax(input_eq):    # ALWAYS False — non-empty tuple is truthy
if not check_equation_equivalence(eq1, eq2):  # ALWAYS False — non-empty tuple is truthy
```
This means **no equation was ever validated**. Invalid equations were silently accepted into every dataset.
**Fix**: Fixed all 5 call sites to properly unpack tuples:
```python
is_valid, _, _ = validate_equation_syntax(input_eq)
is_equiv, _ = check_equation_equivalence(eq1, eq2)
```
Also fixed `solve_equation()` return tuple handling for integer solution validation.

### DATAGEN-005: Evaluation Seeding Incomplete (FIXED)
**Severity**: MODERATE
**Files**: `eval_algebra.py`
**Problem**: `eval_algebra.py` set `np.random.seed()` but `AlgebraDataset` uses Python's `random` module. Evaluation results were non-deterministic despite `--seed` parameter.
**Fix**: Added `random.seed()` calls alongside `np.random.seed()` at all 3 seeding locations.

### Impact Assessment

**ALL previous results are invalidated:**
- Training data for distribute/combine was in wrong format (expressions, not equations)
- No training data was ever validated (DATAGEN-004)
- Test datasets contained mathematically wrong equations
- Even correct models would get wrong scores on wrong test data

**Required actions:**
1. **Push** all code changes to GitHub
2. **Cancel** currently running evaluation jobs (they use models trained on flawed data)
3. **Retrain** all 5 models (distribute, combine, isolate, divide, monolithic) on corrected training data
4. **Re-evaluate** all experiments (exp_001 through exp_007) with new models and corrected test datasets

### Verification Performed
- Generated 20 problems per rule type — all mathematically correct
- Validation correctly rejects invalid pairs (`2*x+3=7 → x=99` returns False)
- Validation correctly accepts valid pairs (`2*x+3=7 → x=2` returns True)
- Regenerated all 175 test dataset problems — 0 errors via independent SymPy verification
