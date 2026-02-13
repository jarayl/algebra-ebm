# Debugging Log

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
