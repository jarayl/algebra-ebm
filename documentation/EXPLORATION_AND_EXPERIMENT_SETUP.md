# Algebra EBM: Codebase Exploration & Experiment Setup
**Date:** 2026-02-11
**Status:** Setup Complete, Awaiting Model Training

## Executive Summary

I've explored the algebra-ebm codebase and set up a comprehensive experimental pipeline. The implementation is **nearly complete** but requires **model training** before evaluation experiments can run.

### Key Findings from Code Analysis

1. **Strong Evaluation Framework:** `eval_algebra.py` is comprehensive and well-structured
2. **Multiple Evaluation Modes:** Single-rule, multi-rule (2/3/4 rules), constrained, monolithic, and compositional
3. **Recent Code Fixes:** Type annotations and error handling improvements to evaluation code
4. **Missing Models:** No trained models exist yet - these must be generated before evaluation

## Codebase Structure

### Source Code Organization
```
src/algebra/
├── algebra_models.py         # Core EBM model definitions
├── algebra_inference.py      # Inference engine with sampling/optimization
├── algebra_evaluation.py      # Comprehensive evaluation framework
├── algebra_dataset.py        # Dataset generation for all test scenarios
├── algebra_encoder.py        # Equation encoding/decoding
└── algebra_constraints.py    # Constraint handling for inference
```

### Training Scripts
```
train_algebra.py                  # Train compositional models (per rule)
train_algebra_monolithic.py      # Train monolithic baseline
train_models_local.py            # Utility for local training
```

### Evaluation & Testing
```
eval_algebra.py                  # Main evaluation orchestrator
tests/                           # Comprehensive test suite
```

## Recent Implementation Improvements

The recent commit "Fix algebra implementation" includes:

### 1. Type Annotation Improvements
- Added `Union` types for flexible handling of different dataset types
- Added `Optional` type hints for parameters that can be None
- Type annotations now more precise and match actual usage

### 2. Error Handling Enhancements
- Better handling of optional dataset metadata
- Improved getattr patterns to safely check for methods
- Consistent error messages for debugging

### 3. Logging Fixes
- Proper logger initialization in validation functions
- Removed redundant logger creation in utility functions

## Experimental Pipeline Setup

### Phase 1: Model Training (BLOCKING)
**Status:** PENDING - Must complete before evaluation

Required steps:
1. Train distribute rule model
2. Train combine rule model
3. Train isolate rule model
4. Train divide rule model
5. Train monolithic baseline model

**Estimated Time:**
- Local (CPU): ~8-12 hours per model (40-60 hours total)
- Cluster (GPU): ~1-2 hours per model (5-10 hours total)

**Recommendation:** Use cluster with SLURM for GPU acceleration

### Phase 2: Evaluation Experiments (BLOCKED until training complete)

#### Experiment 1: Single-Rule Baseline (exp_001)
- **Purpose:** Establish baseline on simplest task
- **Test Sets:** distribute, combine, isolate, divide (separate)
- **Configuration:** 1,000 problems per rule, seed=42
- **Expected Accuracy:** ~85% per rule
- **Duration:** ~30 minutes total

#### Experiment 2-4: Multi-Rule Composition (exp_002-exp_004)
- **Purpose:** Test compositional capability
- **Test Sets:** 2-rule, 3-rule, 4-rule compositions
- **Configuration:** 1,000 problems per set, seed=42
- **Expected Accuracy:**
  - 2-rule: 70-80%
  - 3-rule: 50-70%
  - 4-rule: 40-60%
- **Duration:** ~1 hour total

#### Experiment 5: Constrained Inference (exp_005)
- **Purpose:** Test real-world applicability
- **Constraints:** Positivity, integerness
- **Configuration:** 1,000 problems with constraints
- **Expected Accuracy:** 50-70%
- **Duration:** ~30 minutes

#### Experiment 6: Compositional vs Monolithic Comparison (exp_007)
- **Purpose:** Core research question validation
- **Comparison:** Compositional model vs monolithic baseline
- **Configuration:** Multi-rule problems across difficulty levels
- **Expected Outcome:** Compositional outperforms monolithic
- **Duration:** ~45 minutes

## Pipeline Configuration Files

### .state/pipeline.json
Central orchestration file with:
- Training phase requirements
- 6 evaluation experiments (BLOCKED pending training)
- Dependency tracking
- User input prompt for training method selection

### documentation/experiment-plan.md
Detailed experimental design including:
- Phase-by-phase breakdown
- Success criteria
- Risk assessment
- Data organization
- Known considerations

### run_experiments.py
Python orchestration script with:
- Automatic experiment execution
- Result tracking
- Run directory management
- Logging and error handling

## How to Proceed

### Option 1: Train Models Locally (Not Recommended)
```bash
cd projects/algebra-ebm

# Train compositional models (one at a time)
python train_algebra.py --rule distribute --epochs 50
python train_algebra.py --rule combine --epochs 50
python train_algebra.py --rule isolate --epochs 50
python train_algebra.py --rule divide --epochs 50

# Train monolithic baseline
python train_algebra_monolithic.py --epochs 50

# Run evaluation experiments
python run_experiments.py  # Runs all 6 experiments
```

### Option 2: Train on Cluster with SLURM (Recommended)
```bash
# Update pipeline to indicate cluster submission
# Submit SLURM jobs for training
cd /Users/mkrasnow/Desktop/research-repo

# Follow cluster submission protocol in CLAUDE.md
# Models will train in parallel on GPU nodes
# Monitor with SLURM commands (squeue, sacct)

# After training completes:
cd projects/algebra-ebm
python run_experiments.py  # Run evaluation experiments
```

## Critical Success Criteria

### For Training Phase
✓ All 5 models train without OOM errors
✓ Models save to results/{rule}/model.pt
✓ Training curves show reasonable convergence
✓ Final loss values reasonable (dataset-dependent)

### For Evaluation Phase
✓ All 6 experiments complete without crashes
✓ Numeric results saved to JSON
✓ Single-rule accuracy 75-95%
✓ Multi-rule accuracy degradation with difficulty
✓ Constrained inference produces valid solutions

### For Research Validation
✓ Results align with proposal Section 6 targets
✓ Compositional model is competitive or better on multi-rule
✓ Clear performance patterns across difficulty levels
✓ Computational efficiency reasonable

## Important Notes

### Code Quality
- Type annotations are correct and comprehensive
- Error handling is robust with clear failure messages
- Logging is informative for debugging

### Dataset Generation
- Deterministic with seed control
- Creates separate test datasets for each rule
- Validates problem generation before evaluation

### Inference Parameters
- Configurable temperature schedule
- Energy-based ranking of solutions
- Multi-sample evaluation for robustness

## Next Steps

1. **Decide on training approach:**
   - Local (quick start, slow)
   - Cluster (recommended, fast)

2. **Train models:**
   - Execute training scripts or SLURM jobs
   - Monitor for errors/convergence

3. **Run evaluation experiments:**
   - `python run_experiments.py` (all experiments)
   - Or individual experiments with `--experiment` flag

4. **Analyze results:**
   - Check JSON output files in runs/ directory
   - Compare with expected accuracies
   - Generate comparison report

5. **Debug if needed:**
   - Check documentation/debugging.md for common issues
   - Review evaluation logs for details
   - Adjust inference parameters if needed

## File Locations

**Configuration:**
- Pipeline: `projects/algebra-ebm/.state/pipeline.json`
- Experiment Plan: `projects/algebra-ebm/documentation/experiment-plan.md`
- This Summary: `projects/algebra-ebm/documentation/EXPLORATION_AND_EXPERIMENT_SETUP.md`

**Scripts:**
- Training: `projects/algebra-ebm/train_algebra.py`
- Evaluation: `projects/algebra-ebm/eval_algebra.py`
- Orchestration: `projects/algebra-ebm/run_experiments.py`

**Results:**
- Model checkpoints: `projects/algebra-ebm/results/{rule}/model.pt`
- Experiment results: `projects/algebra-ebm/runs/{experiment_id}_{timestamp}/results/`
- Logs: `projects/algebra-ebm/runs/{experiment_id}_{timestamp}/logs/`

## Questions & Troubleshooting

**Q: Why are models not loading?**
A: Models must be trained first. Run training scripts before evaluation.

**Q: Can I train locally?**
A: Yes, but it's very slow (~40-60 hours). Cluster is recommended (~5-10 hours).

**Q: What if training fails?**
A: Check logs in training output directory, review requirements.txt, verify GPU/CPU availability.

**Q: Can experiments run in parallel?**
A: Experiments can run sequentially (slower) or in a batch if you have multiple GPUs.

**Q: How do I monitor progress?**
A: Check run_experiments.py logs, check individual experiment logs in runs/ subdirectories.

