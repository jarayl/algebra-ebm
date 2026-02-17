# Algebra-EBM Retraining Plan

## Issue Summary
Single-rule evaluation achieved only 6.3% accuracy due to inconsistent energy landscapes:
- 54% of problems had correct energy ordering (E_target < E_input)
- 46% had inverted energy ordering (E_target > E_input)

## Root Cause
Model's fc4 output layer produced raw energies ~11 instead of ~6 needed for target E=1.0:
- Learned energy_scale = 0.98, energy_bias = -4.82
- Model compensated with large negative bias but couldn't fully suppress energies
- High-coefficient problems (>300) more likely to have inverted landscapes

## Fix Applied
**File**: `src/algebra/algebra_models.py` line 108

**Change**:
```python
# Before:
nn.init.xavier_uniform_(module.weight, gain=0.5)

# After:
nn.init.xavier_uniform_(module.weight, gain=0.1)  # 5× smaller initial outputs
```

## Expected Results After Retraining

### Energy Targets
- Positive energies: ~1.0 (currently ~6.0)
- Negative energies: ~15.0 (currently ~15.0) ✓
- Energy gap: ~14.0 (currently ~9.0)

### Accuracy Improvement
- Current: 6.3% (only 54% problems have correct landscapes)
- Expected: **50-85%** (>90% problems should have correct landscapes)

### Training Monitoring
Watch for these metrics in training logs:
```
[EnergyMonitor] PosE should decrease from ~6 to ~1-2
[EnergyMonitor] NegE should stay around 15
[EnergyMonitor] Gap should increase from ~9 to ~13-14
```

## Retraining Steps

### 1. Verify Fix is in Place
```bash
grep -n "gain=0.1" src/algebra/algebra_models.py
# Should show line 108 with gain=0.1
```

### 2. Retrain All 4 Models
```bash
# Submit training jobs for all rules
sbatch slurm/train_distribute.sbatch
sbatch slurm/train_combine.sbatch
sbatch slurm/train_isolate.sbatch
sbatch slurm/train_divide.sbatch
```

**Estimated time**: ~6 hours per model (50K steps)

### 3. Monitor Training Progress
Check logs every ~10K steps:
```bash
tail -100 slurm/logs/algebra_train_distribute_*.out | grep EnergyMonitor
```

Look for:
- PosE trending down toward 1-2
- NegE stable around 15
- Gap increasing toward 13-14

### 4. Validate After Training
Run diagnostic to check energy landscape consistency:
```bash
cd projects/algebra-ebm
python debug_timestep_energy.py  # Should show E_target < E_input at t=0
python analyze_problem_patterns.py  # Should show >90% correct landscapes
```

### 5. Re-run Evaluation
```bash
# Submit all 6 evaluation experiments
sbatch slurm/eval_exp_001_single_rule.sbatch
sbatch slurm/eval_exp_002_multi_rule_2.sbatch
sbatch slurm/eval_exp_003_multi_rule_3.sbatch
sbatch slurm/eval_exp_004_multi_rule_4.sbatch
sbatch slurm/eval_exp_005_constrained.sbatch
sbatch slurm/eval_exp_007_comparison.sbatch
```

### 6. Verify Results
Expected accuracy improvements:
- Single-rule: 6.3% → **70-85%**
- Multi-rule (2): ? → **50-70%** (compositional)
- Multi-rule (3): ? → **40-60%**
- Multi-rule (4): ? → **30-50%**

## Success Criteria

✅ Training converges with PosE < 2.0, NegE ~ 15.0
✅ >90% of test problems have E_target < E_input
✅ Single-rule accuracy >70%
✅ No energy landscape inversions for high-coefficient problems

## Rollback Plan

If retraining fails or makes things worse:
1. Revert `src/algebra/algebra_models.py` line 108 to `gain=0.5`
2. Consider alternative fixes:
   - Add explicit L2 regularization on fc4 outputs
   - Adjust contrastive loss weights
   - Increase model capacity (hidden_dim 512 → 1024)

## Timeline

- **Fix applied**: 2026-02-16 18:45 UTC
- **Retraining start**: TBD (waiting for user approval)
- **Retraining complete**: ~24 hours after start (4 models × 6 hours)
- **Evaluation complete**: ~4 hours after retraining
- **Results analysis**: ~1 hour

**Total estimated time**: ~29 hours from start to verified results
