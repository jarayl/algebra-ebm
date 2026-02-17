# Diagnostic Experiment: Encoder Normalization Fix

**Date**: 2026-02-17
**Purpose**: Verify that encoder normalization is the root cause of energy landscape failure
**Status**: READY TO RUN

---

## Background

Deep dive analysis identified **CRITICAL ROOT CAUSE**:
- Encoder normalizes all embeddings to unit L2 norm (`||embedding|| = 1.0`)
- Energy function `E = scale * ||output||^2 + bias` cannot discriminate (always ≈ 1.0)
- Result: 54% correct vs 46% inverted energy landscapes (random 50/50 split)

**Full Analysis**: `documentation/deep-dive-analysis.md`
**Summary**: `documentation/CRITICAL-FINDINGS.md`

---

## Modifications Applied

### ✅ Code Changed

**File**: `src/algebra/algebra_encoder.py` line 135

**Before**:
```python
if self.normalize_embeddings:
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

**After**:
```python
# DIAGNOSTIC EXPERIMENT (2026-02-17): Normalization DISABLED
if False:  # self.normalize_embeddings - DISABLED FOR DIAGNOSTIC
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

---

## Running the Diagnostic

### Step 1: Train Diagnostic Model (~3 hours)

```bash
cd /Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm

# Run training script
bash scripts/diagnostic_no_norm_train.sh
```

**What it does**:
- Trains distribute model for 10,000 steps
- Batch size 2048, timesteps 10
- Saves to `results/diagnostic_no_norm/`
- Logs to `results/diagnostic_no_norm/training.log`

**Expected output**:
- Model converges (loss decreases)
- Energy scale parameter learns meaningful value (not stuck at 1.0)
- Energy gaps ~9-10 units (same as baseline)

### Step 2: Evaluate Model (~10 minutes)

```bash
# Run evaluation script
bash scripts/diagnostic_no_norm_eval.sh
```

**What it does**:
- Tests on 100 single-rule problems
- Enables diagnostic logging
- Saves to `results/diagnostic_no_norm/evaluation/`

**Expected output**:
- Accuracy metrics
- Energy trajectory logs
- Diagnostic data for each problem

### Step 3: Analyze Energy Landscapes (~1 minute)

```bash
# Analyze energy landscape quality
python scripts/analyze_energy_landscapes.py results/diagnostic_no_norm/evaluation
```

**What it does**:
- Counts correct vs inverted energy landscapes
- Calculates percentage correct
- Compares to baseline (54%)
- Provides decision criteria

**Expected output**:
```
ENERGY LANDSCAPE QUALITY ANALYSIS
==================================
Total Problems: 100
  Correct Landscapes:   82 (82.0%) - E(inp→target) < E(inp→input) ✓
  Inverted Landscapes:  18 (18.0%) - E(inp→target) > E(inp→input) ✗

BASELINE COMPARISON
===================
Baseline (with normalization):    54.0% correct
Current  (without normalization): 82.0% correct
Improvement:                      +28.0 percentage points

DECISION: ✓ SUCCESS - ROOT CAUSE CONFIRMED
```

---

## Decision Criteria

### IF Energy Landscapes >80% Correct ✓

**Conclusion**: ROOT CAUSE CONFIRMED - Normalization was breaking energy learning

**Next Step**: Full Retraining (T0b)
- Retrain all 5 models without normalization
- Expected runtime: ~30 hours (5 models × 6 hours)
- Command: `bash scripts/full_retrain_no_norm.sh`

**Expected Outcomes**:
- Single-rule accuracy: 50-85% (up from 6.3%)
- Multi-rule accuracy: 10-30% (up from 0%)

### IF Energy Landscapes 60-80% Correct ⚠

**Conclusion**: PARTIAL FIX - Normalization contributed but not sole cause

**Next Step**: Investigate Issue #2 (energy scale parameter learning)
- Add gradient logging to training
- Verify energy_scale/energy_bias are updating
- Check if parameters are in optimizer

**Actions**:
1. Examine training logs for energy_scale values
2. Add gradient logging to train_algebra.py
3. Consider full retraining anyway (may still improve)

### IF Energy Landscapes <60% Correct ✗

**Conclusion**: HYPOTHESIS REJECTED - Normalization not the main issue

**Next Steps**:
1. Investigate Issue #2 (energy scale stuck at initialization)
2. Investigate Issue #3 (insufficient inference iterations)
3. Re-examine training logs for anomalies

**Alternative Hypotheses**:
- Energy scale/bias parameters not in optimizer
- Energy function architecture fundamentally flawed
- Decoder introducing errors (not energy landscape)

---

## Monitoring Progress

### During Training

```bash
# Watch training progress
tail -f results/diagnostic_no_norm/training.log

# Check energy scale parameter value
grep -i "energy_scale" results/diagnostic_no_norm/training.log
```

### After Training

```bash
# Check final model
ls -lh results/diagnostic_no_norm/distribute/model.pt

# Verify training completed
tail -20 results/diagnostic_no_norm/training.log
```

### After Evaluation

```bash
# Check accuracy
grep "Accuracy" results/diagnostic_no_norm/evaluation/evaluation.log

# Check diagnostic data
ls results/diagnostic_no_norm/evaluation/diagnostics/
```

---

## Troubleshooting

### Error: "Model not found"

**Cause**: Training didn't complete or saved to wrong location

**Fix**:
```bash
# Check if model exists
ls results/diagnostic_no_norm/distribute/

# Check training logs for errors
tail -50 results/diagnostic_no_norm/training.log
```

### Error: "No trajectory files found"

**Cause**: Evaluation didn't run with diagnostics enabled

**Fix**:
```bash
# Re-run evaluation with diagnostics
python eval_algebra.py \
    --model_dir results/diagnostic_no_norm \
    --eval_type single_rule \
    --rule distribute \
    --single_rule_problems 100 \
    --enable_diagnostics
```

### Error: "Normalization still enabled"

**Cause**: Code change wasn't applied correctly

**Fix**:
```bash
# Verify modification
grep -A 2 "DISABLED FOR DIAGNOSTIC" src/algebra/algebra_encoder.py

# Should output:
# if False:  # self.normalize_embeddings - DISABLED FOR DIAGNOSTIC
#     embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

---

## Timeline

- **Training**: ~3 hours
- **Evaluation**: ~10 minutes
- **Analysis**: ~1 minute
- **Total**: ~3.5 hours

---

## Success Metrics

### Baseline (Current - With Normalization)
- Energy landscape correctness: **54%** (random)
- Single-rule accuracy: **6.3%**
- Multi-rule accuracy: **0%**

### Target (After Fix - Without Normalization)
- Energy landscape correctness: **>80%**
- Single-rule accuracy: **>30%**
- Proves normalization was root cause

### Stretch Goal
- Energy landscape correctness: **>90%**
- Single-rule accuracy: **>50%**
- Would indicate very strong fix

---

## Next Steps After Diagnostic

If successful (>80% energy landscapes):

1. **Update documentation** with diagnostic results
2. **Run full retraining** (T0b) - all 5 models without normalization
3. **Full re-evaluation** (T0c) - complete test suite
4. **Update phase** to EVALUATE or SUCCESS based on results

---

## Files Created

**Scripts**:
- `scripts/diagnostic_no_norm_train.sh` - Training script
- `scripts/diagnostic_no_norm_eval.sh` - Evaluation script
- `scripts/analyze_energy_landscapes.py` - Analysis script

**Documentation**:
- `RUN-DIAGNOSTIC.md` - This file
- `documentation/deep-dive-analysis.md` - Full technical analysis
- `documentation/CRITICAL-FINDINGS.md` - Executive summary

**Modified Code**:
- `src/algebra/algebra_encoder.py` - Normalization disabled

---

## Contact

For questions or issues, refer to:
- Full analysis: `documentation/deep-dive-analysis.md`
- Debugging history: `documentation/debugging.md`
- Action queue: `documentation/queue.md`

---

**Ready to run!** Execute `bash scripts/diagnostic_no_norm_train.sh` to begin.
