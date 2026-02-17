# Action Queue - Algebra EBM

**Last Updated**: 2026-02-16
**Current Phase**: DEBUG
**Priority**: Inference diagnostics implementation

---

## High Priority (Do First)

### T1: Add Inference Logging Infrastructure
**Status**: PENDING
**Assigned**: Next dispatch
**Blockers**: None
**Time Estimate**: 2 hours
**Description**: Modify `src/algebra/algebra_evaluation.py` to add detailed inference logging
**Implementation**:
- Log energy at each iteration
- Log gradient norms
- Log embedding evolution (L2 distance from start)
- Log acceptance rates (if using Langevin)
- Log final decoding attempts
- Save to: `results/evaluation/{experiment_id}/diagnostics/problem_{i}_trajectory.json`

### T2: Implement Multi-Start Inference
**Status**: PENDING
**Assigned**: After T1
**Blockers**: T1 (logging needed to verify)
**Time Estimate**: 3 hours
**Description**: Modify evaluation to support multiple random initializations
**Implementation**:
- Add `--num_starts` parameter (default 10)
- Run inference from N different random seeds
- Keep result with lowest final energy
- Log which start index won
**Expected Impact**: If >20% improvement, confirms local minima hypothesis

### T3: Make Iteration Count Configurable
**Status**: PENDING
**Assigned**: After T1
**Blockers**: T1 (logging needed to analyze)
**Time Estimate**: 1 hour
**Description**: Make `max_iterations` configurable in evaluation
**Implementation**:
- Add `--max_inference_iterations` parameter
- Default 50, allow up to 500
- Log iteration count used per problem
**Expected Impact**: If 500 >> 50, need more optimization steps

---

## Medium Priority (Do Second)

### T4: Add Momentum to Gradient Descent
**Status**: PENDING
**Assigned**: After T1-T3 analysis
**Blockers**: T1, T2, T3 (need to see if simpler fixes work first)
**Time Estimate**: 2 hours
**Description**: Add momentum to IRED gradient descent
**Implementation**:
- Add `momentum` parameter to InferenceConfig (default 0.9)
- Update gradient step: `velocity = momentum * velocity + grad`
- Log velocity norms
**Expected Impact**: May help escape local minima

### T5: Implement Input-Guided Initialization
**Status**: PENDING
**Assigned**: After T1-T3 analysis
**Blockers**: T1, T2, T3 (optional enhancement)
**Time Estimate**: 2 hours
**Description**: Start inference from input equation embedding instead of random
**Implementation**:
- Encode input equation through model
- Use as starting point for IRED
- Compare results against random init
- Add flag: `--use_input_init`
**Expected Impact**: May provide better starting point than random noise

### T6: Create Diagnostic Experiment Configs
**Status**: PENDING
**Assigned**: After T1 implemented
**Blockers**: T1 (need logging infrastructure first)
**Time Estimate**: 1 hour
**Description**: Create 4 diagnostic experiment configs in `configs/`
**Configs**:
1. `exp_diag_baseline.json` - Current settings (50 iters, 1 start)
2. `exp_diag_multistart.json` - 10 starts, 50 iters each
3. `exp_diag_longrun.json` - 1 start, 500 iters
4. `exp_diag_combined.json` - 10 starts, 100 iters each

**Experiment Details**:
- Use exp_001 test dataset (single-rule problems, 1000 problems)
- Enable diagnostic logging
- Use distribute rule only (for speed)
- Evaluate on 100 problems (not 1000) for faster iteration

### T7: Create Diagnostic SLURM Scripts
**Status**: PENDING
**Assigned**: After T6
**Blockers**: T6 (configs must exist first)
**Time Estimate**: 1 hour
**Description**: Create sbatch scripts for diagnostic experiments
**Scripts**:
- `slurm/eval_diag_baseline.sbatch`
- `slurm/eval_diag_multistart.sbatch`
- `slurm/eval_diag_longrun.sbatch`
- `slurm/eval_diag_combined.sbatch`

**Configuration**:
- Partition: gpu (not gpu_test, to avoid QOS issues)
- Resources: 1 GPU, 2 CPUs, 16GB RAM
- Time limit: 2 hours
- Auto-clone repo and checkout current commit

---

## Low Priority (Do Third)

### T8: Debug exp_007 Missing Results
**Status**: PENDING
**Assigned**: After diagnostic experiments submitted
**Blockers**: T1-T7 (higher priority)
**Time Estimate**: 3 hours
**Description**: Investigate why exp_007 comparison evaluation produced no output
**Investigation**:
- Check evaluation script logic for comparison mode
- Look for silent exceptions or early exits
- Add debug logging
- Re-run exp_007 with fixes
**Importance**: Need monolithic baseline for fair comparison

### T9: Manual Dataset Verification
**Status**: PENDING
**Assigned**: After diagnostic experiments analyzed
**Blockers**: None (can run in parallel)
**Time Estimate**: 2 hours
**Description**: Manually verify test dataset correctness
**Process**:
- Sample 10-20 problems per rule type from `results/test_datasets/`
- Verify mathematical correctness with independent SymPy
- Check format matches training data format
- Verify single rule application achieves target
- Document any errors found
**Deliverable**: `documentation/dataset-verification.md`

---

## Completed Tasks

None yet - this is the initial queue setup.

---

## Blocked Tasks

None - all tasks have clear dependency chains.

---

## Dependencies

```
T1 (Logging)
  ├─> T2 (Multi-start) - needs logging to verify
  ├─> T3 (Iterations) - needs logging to analyze
  ├─> T6 (Configs) - needs logging infrastructure
  │   └─> T7 (SLURM scripts) - needs configs
  └─> T4, T5 (Momentum, Init) - wait for T1-T3 results

T8 (exp_007) - independent, can run anytime
T9 (Verification) - independent, can run in parallel
```

---

## Success Criteria

### T1-T3 Success Criteria:
- Logging produces valid JSON files for each problem
- Multi-start shows energy variance across starts (confirms local minima)
- Iteration count experiment shows convergence curves
- Can analyze specific failure modes (oscillation, drift, stuck)

### T4-T5 Success Criteria:
- Momentum reduces gradient oscillation (if observed)
- Input init reduces initial distance to target
- Measurable accuracy improvement (target: >10% absolute gain)

### T6-T7 Success Criteria:
- All 4 diagnostic experiments run successfully on cluster
- Logs captured and analyzable
- Results inform next steps (continue with current approach or pivot)

### T8 Success Criteria:
- exp_007 produces evaluation metrics
- Monolithic baseline results available for comparison
- Understand compositional advantage (or lack thereof)

### T9 Success Criteria:
- 20+ problems manually verified
- All verified problems mathematically correct
- Format matches training data
- Confidence that test datasets are valid

---

## Timeline

**Week 1** (2026-02-16 to 2026-02-23):
- Complete T1-T7 (implementation + experiment submission)
- Analyze diagnostic experiment results
- Complete T9 (manual verification)

**Week 2** (2026-02-23 to 2026-03-02):
- Based on Week 1 results, implement fixes (T4, T5, or architectural changes)
- Re-evaluate single-rule performance
- Target: >50% single-rule accuracy
- Complete T8 (exp_007 debug)

**Week 3** (2026-03-02 to 2026-03-09):
- If single-rule >50%: tackle multi-rule composition
- If single-rule <50%: evaluate pivot options
- Decision point: continue or pivot

---

## Notes

### Why T1 (Logging) is Critical:
Without detailed logging, we're flying blind. Current eval只 shows final accuracy (6%), but we need to understand:
- Where does inference fail? (initialization, middle, end)
- What does energy landscape look like?
- Are gradients vanishing/exploding?
- Is decoder the bottleneck or is inference stuck?

### Why T2 (Multi-Start) is High Priority:
Multi-start is the simplest way to test AUDIT-003's local minima hypothesis:
- If multi-start dramatically improves accuracy → confirms local minima
- If multi-start doesn't help → problem is elsewhere (decoder, energy landscape smoothness)

### Why T6-T7 Before T4-T5:
We need to run diagnostic experiments to inform whether T4-T5 are worth implementing:
- If diagnostics show oscillation → momentum helps (do T4)
- If diagnostics show poor init → input-guided init helps (do T5)
- If diagnostics show flat landscape → neither helps (pivot to architectural changes)

### Why T8 is Lower Priority:
exp_007 is important for comparison, but doesn't block inference diagnostics. Previous run showed compositional (4.9%) ≈ monolithic (4.5%), so we already know compositional doesn't have huge advantage. Fixing inference is more urgent.

### Why T9 Can Run in Parallel:
Manual verification is independent and doesn't block anything. If datasets are wrong, that's a critical finding. But DATAGEN fixes were validated programmatically (0 errors via SymPy), so high confidence they're correct. This is belt-and-suspenders verification.
