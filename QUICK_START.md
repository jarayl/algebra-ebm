# Quick Start Guide: Algebra EBM Experiments

## Status
✓ Codebase explored
✓ Experiment pipeline configured
✓ Evaluation framework ready
⏳ **Awaiting model training** (5 models required)

## What's Ready to Go
- Comprehensive evaluation suite in `eval_algebra.py`
- 6 planned experiments with clear objectives
- Experiment orchestration script `run_experiments.py`
- Detailed experiment plan and documentation

## What Needs to Happen Next
1. **Train 5 models** (distribute, combine, isolate, divide, monolithic)
2. **Run 6 evaluation experiments** (single-rule, multi-rule 2/3/4, constrained, comparison)

## Quick Commands

### Check Current Status
```bash
cd projects/algebra-ebm
cat .state/pipeline.json | grep -A 2 '"status"'
```

### Option A: Train Models Locally (Fast Start)
```bash
cd projects/algebra-ebm

# Train each rule model
python train_algebra.py --rule distribute --epochs 50 --batch_size 32
python train_algebra.py --rule combine --epochs 50 --batch_size 32
python train_algebra.py --rule isolate --epochs 50 --batch_size 32
python train_algebra.py --rule divide --epochs 50 --batch_size 32

# Train monolithic baseline
python train_algebra_monolithic.py --epochs 50 --batch_size 32

# After training completes, run evaluations
python run_experiments.py  # Runs all 6 experiments

# Or run specific experiment
python run_experiments.py --experiment exp_001_single_rule_baseline
```

### Option B: Train Models on Cluster (Recommended)
```bash
cd /Users/mkrasnow/Desktop/research-repo

# Use the /dispatch skill or SLURM submission
# Submit training jobs as SLURM batch array
# Monitor with: squeue -u $USER
# Check results: sacct -j <job_id>

# After training on cluster, run evaluations locally
cd projects/algebra-ebm
python run_experiments.py
```

## Expected Timeline

### Local Training
- Per model: 8-12 hours (CPU)
- Total: 40-60 hours
- Evaluation: 2-3 hours
- **Total: 2-3 days**

### Cluster Training (Recommended)
- Per model: 1-2 hours (GPU)
- Total: 5-10 hours (parallel)
- Evaluation: 2-3 hours
- **Total: 6-13 hours**

## Expected Results

| Test Set | Expected Accuracy |
|----------|------------------|
| Single-rule | 80-90% |
| 2-rule composition | 60-80% |
| 3-rule composition | 40-70% |
| 4-rule composition | 30-60% |
| Constrained | 50-70% |

## Troubleshooting

### Models Not Loading
**Problem:** "CRITICAL: No models loaded from ./results"
**Solution:** Run training scripts first

### Training Out of Memory
**Problem:** CUDA out of memory or CPU memory errors
**Solution:** Reduce batch_size, use cluster with GPU

### Import Errors
**Problem:** "No module named 'src.algebra'"
**Solution:** Run from projects/algebra-ebm directory

### Results Empty
**Problem:** Evaluation runs but produces empty JSON
**Solution:** Check eval_algebra.py logs, verify dataset generation

## File Structure
```
projects/algebra-ebm/
├── .state/
│   └── pipeline.json              # Central configuration
├── documentation/
│   ├── experiment-plan.md         # Detailed plan
│   ├── EXPLORATION_AND_EXPERIMENT_SETUP.md  # Full analysis
│   └── QUICK_START.md             # This file
├── src/algebra/                   # Core implementation
├── eval_algebra.py                # Evaluation script
├── train_algebra.py               # Compositional training
├── train_algebra_monolithic.py    # Monolithic training
├── run_experiments.py             # Experiment orchestrator
├── results/                       # (Will be populated with trained models)
└── runs/                          # (Will contain experiment results)
```

## Key Decisions Made

1. **Evaluation Scope:** 6 comprehensive experiments covering all test scenarios
2. **Dataset Sizes:** 1,000 problems per test (balance between coverage and speed)
3. **Random Seed:** Fixed seed (42) for reproducibility
4. **Model Storage:** Standard PyTorch format in results/{rule}/
5. **Result Organization:** Timestamped runs/ directories for traceability

## Validation Checklist

Before running full experiments:
```bash
cd projects/algebra-ebm

# ✓ Check imports work
python -c "from src.algebra.algebra_dataset import AlgebraDataset; print('✓ Imports OK')"

# ✓ Verify training script exists
test -f train_algebra.py && echo "✓ train_algebra.py found"
test -f train_algebra_monolithic.py && echo "✓ train_algebra_monolithic.py found"

# ✓ Verify evaluation script exists
test -f eval_algebra.py && echo "✓ eval_algebra.py found"

# ✓ Verify experiment orchestrator exists
test -f run_experiments.py && echo "✓ run_experiments.py found"
```

## Contact & Help

See `EXPLORATION_AND_EXPERIMENT_SETUP.md` for:
- Detailed technical analysis
- Risk assessment
- Debugging guidance
- Success criteria

