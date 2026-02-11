# Algebra EBM: Comprehensive Experiment Plan
**Date:** 2026-02-11
**Status:** READY TO EXECUTE

## Overview
Execute comprehensive evaluation suite for Algebra EBM compositional vs monolithic models across single-rule, multi-rule, and constrained problem domains.

## Experimental Design

### Phase 1: Foundation & Baseline (Quick Validation)
**Experiment:** `exp_006_quick_validation`
- **Objective:** Verify evaluation infrastructure is working
- **Configuration:**
  - Small dataset (100 samples per test)
  - All evaluation modes
- **Success Criteria:**
  - All imports load successfully
  - Dataset generation completes without errors
  - Evaluation runs produce numeric results

### Phase 2: Single-Rule Baseline
**Experiment:** `exp_001_single_rule_baseline`
- **Objective:** Establish baseline performance on single-rule problems
- **Test Sets:** distribute, combine, isolate, divide
- **Configuration:**
  - 1,000 problems per rule
  - Seed: 42 for reproducibility
- **Expected Results:** ~85% accuracy per rule
- **Rationale:**
  - Single rule is simplest case
  - Should show strong performance
  - Establishes baseline for comparison

### Phase 3: Multi-Rule Composition (Progressive Difficulty)
**Experiments:**
- `exp_002_multi_rule_2`: 2-rule composition
- `exp_003_multi_rule_3`: 3-rule composition
- `exp_004_multi_rule_4`: 4-rule composition

- **Objective:** Evaluate compositional capability across increasing complexity
- **Configuration:**
  - 1,000 problems per num_rules level
  - Seed: 42 for reproducibility
- **Expected Results:**
  - 2-rule: ~70-80% accuracy
  - 3-rule: ~50-70% accuracy
  - 4-rule: ~40-60% accuracy
- **Rationale:**
  - Tests core compositional capability
  - Progressive difficulty allows identifying breakdown points
  - Aligns with paper's primary claims

### Phase 4: Constrained Inference
**Experiment:** `exp_005_constrained`
- **Objective:** Evaluate model's ability to handle constraints
- **Constraints Tested:**
  - Positivity (all variables non-negative)
  - Integerness (all variables are integers)
- **Configuration:**
  - 1,000 problems with constraints
  - Seed: 42 for reproducibility
- **Expected Results:** ~50-70% accuracy with constraints
- **Rationale:**
  - Tests real-world applicability
  - Constraints affect solution space significantly

## Evaluation Metrics

For each experiment, we collect:
1. **Accuracy Metrics:**
   - Overall accuracy
   - Per-rule accuracy breakdown (for multi-rule experiments)
   - Accuracy by problem difficulty

2. **Computational Metrics:**
   - Inference time per problem
   - Total evaluation time
   - Energy evaluations required

3. **Quality Metrics:**
   - Energy gap between correct and incorrect solutions
   - Solution validity (satisfaction of constraints)
   - Solution distribution analysis

## Success Criteria

### Minimum Success
- All experiments complete without crashes
- Numeric results produced for all test sets
- Results saved in JSON format

### Expected Success
- Single-rule accuracy: 80-90%
- Multi-rule 2-rule: 60-80%
- Multi-rule 3-rule: 40-70%
- Multi-rule 4-rule: 30-60%

### Optimal Success
- Results align with paper proposal Section 6
- Clear performance degradation pattern as rules increase
- Consistent results across random seeds (when tested)

## Data Organization

Results will be saved to:
```
projects/algebra-ebm/runs/<run_id>/
├── results/
│   ├── single_rule_<rule>.json
│   ├── multi_rule_2.json
│   ├── multi_rule_3.json
│   ├── multi_rule_4.json
│   ├── constrained.json
│   └── summary.json
└── logs/
    └── evaluation_<timestamp>.log
```

## Experiment Execution Order

1. **exp_006_quick_validation** - First (quick)
2. **exp_001_single_rule_baseline** - After validation passes
3. **exp_002_multi_rule_2** - Can run in parallel with others
4. **exp_003_multi_rule_3** - Can run in parallel with others
5. **exp_004_multi_rule_4** - Can run in parallel with others
6. **exp_005_constrained** - Can run in parallel with others or after Phase 2

## Known Considerations

### Code Quality
- Recent fixes to eval_algebra.py include:
  - Type annotation improvements (Union types)
  - Better error handling for dataset access
  - Logging initialization fixes

### Data Generation
- Random seed is used for reproducibility
- Each rule uses offset seed for deterministic variety
- Total samples needed: ~6,000 (across all experiments)

### Inference
- Uses AlgebraInference for solution search
- Energy-based model ranking
- Configurable sampling parameters

## Next Steps After Experiments

1. **Analysis:**
   - Compare actual vs expected accuracies
   - Identify performance bottlenecks
   - Check for systematic errors

2. **Debugging (if needed):**
   - Detailed accuracy breakdown by problem type
   - Energy landscape visualization
   - Solution validity analysis

3. **Optimization (if needed):**
   - Tune inference parameters
   - Adjust energy scale handling
   - Improve multi-rule composition

4. **Reporting:**
   - Generate comparison report
   - Create performance visualizations
   - Document findings

## Risk Assessment

**Low Risk:**
- Single-rule evaluation (well-established)
- Dataset generation (tested in prior work)
- Result saving (standard JSON format)

**Medium Risk:**
- Multi-rule evaluation (more complex inference)
- Constrained inference (additional solver overhead)
- Memory usage with large datasets

**Mitigation:**
- Start with small dataset (quick validation)
- Monitor error logs during execution
- Validate intermediate results

