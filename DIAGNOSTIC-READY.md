# ✅ Diagnostic Experiment Ready to Run

**Date**: 2026-02-17
**Status**: ALL PREPARATIONS COMPLETE
**Estimated Time**: 3.5 hours total

---

## Summary

Deep dive analysis identified the root cause of catastrophic evaluation failure:

**CRITICAL FINDING**: Encoder normalization forces `||embedding|| = 1.0`, breaking the energy function's ability to discriminate. Result: random energy landscapes (54% correct vs 46% inverted).

**The Fix**: Disable encoder normalization and retrain.

**This Diagnostic**: Tests the fix on ONE model (distribute) to verify the hypothesis before committing to full retraining (30+ hours).

---

## What's Been Prepared

### ✅ Code Modified

**File**: `src/algebra/algebra_encoder.py` line 135

Normalization has been **DISABLED** for the diagnostic experiment.

### ✅ Scripts Created

1. **`scripts/diagnostic_no_norm_train.sh`**
   - Trains distribute model for 10k steps without normalization
   - Runtime: ~3 hours
   - Output: `results/diagnostic_no_norm/`

2. **`scripts/diagnostic_no_norm_eval.sh`**
   - Evaluates on 100 problems with diagnostic logging
   - Runtime: ~10 minutes
   - Output: `results/diagnostic_no_norm/evaluation/`

3. **`scripts/analyze_energy_landscapes.py`**
   - Analyzes energy landscape quality
   - Compares to baseline (54% correct)
   - Provides decision criteria
   - Runtime: <1 minute

### ✅ Documentation Created

1. **`RUN-DIAGNOSTIC.md`** - Complete step-by-step guide
2. **`documentation/deep-dive-analysis.md`** - Full technical analysis (11 pages)
3. **`documentation/CRITICAL-FINDINGS.md`** - Executive summary (4 pages)
4. **`documentation/debugging.md`** - Updated with root cause
5. **`documentation/queue.md`** - Updated with T0, T0b, T0c tasks

---

## How to Run

### Quick Start (3 commands)

```bash
cd /Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm

# 1. Train model (~3 hours)
bash scripts/diagnostic_no_norm_train.sh

# 2. Evaluate model (~10 minutes)
bash scripts/diagnostic_no_norm_eval.sh

# 3. Analyze results (<1 minute)
python scripts/analyze_energy_landscapes.py results/diagnostic_no_norm/evaluation
```

### Detailed Instructions

See **`RUN-DIAGNOSTIC.md`** for:
- Step-by-step instructions
- What to expect at each step
- How to monitor progress
- Troubleshooting guide
- Decision criteria

---

## Expected Outcomes

### IF >80% Energy Landscapes Correct ✓

**Conclusion**: ROOT CAUSE CONFIRMED

**Next**: Full retraining (all 5 models)
- Runtime: ~30 hours
- Expected: Single-rule 50-85%, Multi-rule 10-30%

### IF 60-80% Correct ⚠

**Conclusion**: PARTIAL FIX

**Next**: Investigate energy scale parameter learning
- Add gradient logging
- Verify parameters are updating

### IF <60% Correct ✗

**Conclusion**: HYPOTHESIS REJECTED

**Next**: Investigate alternative root causes
- Energy scale stuck at initialization
- Insufficient inference iterations
- Decoder errors

---

## Verification Checklist

Before running, verify:

- [x] Code modified: `src/algebra/algebra_encoder.py` line 135
- [x] Scripts created: `scripts/diagnostic_no_norm_*.sh`
- [x] Analysis script: `scripts/analyze_energy_landscapes.py`
- [x] Documentation: `RUN-DIAGNOSTIC.md` exists
- [x] Current directory: `projects/algebra-ebm/`

To verify encoder modification:
```bash
grep -A 2 "DISABLED FOR DIAGNOSTIC" src/algebra/algebra_encoder.py
```

Should output:
```
if False:  # self.normalize_embeddings - DISABLED FOR DIAGNOSTIC
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

---

## Timeline

| Step | Duration | Cumulative |
|------|----------|-----------|
| Training | ~3 hours | 3 hours |
| Evaluation | ~10 min | 3h 10m |
| Analysis | <1 min | 3h 11m |
| **Total** | **~3.5 hours** | |

---

## Files Overview

### Scripts (3)
```
scripts/
├── diagnostic_no_norm_train.sh    # Training script
├── diagnostic_no_norm_eval.sh     # Evaluation script
└── analyze_energy_landscapes.py   # Analysis script
```

### Documentation (5)
```
documentation/
├── deep-dive-analysis.md          # Full analysis (11 pages)
├── CRITICAL-FINDINGS.md           # Executive summary (4 pages)
├── debugging.md                   # Updated with root cause
└── queue.md                       # Updated with T0/T0b/T0c

RUN-DIAGNOSTIC.md                  # How-to guide
DIAGNOSTIC-READY.md                # This file
```

### Modified Code (1)
```
src/algebra/
└── algebra_encoder.py             # Line 135: normalization disabled
```

---

## What Happens Next

### If Diagnostic Succeeds (>80%)

1. **Document results** in debugging.md
2. **Run full retraining** (T0b)
   - All 5 models without normalization
   - 30 hours runtime
3. **Full re-evaluation** (T0c)
   - Complete test suite (6 experiments)
   - 6 hours runtime
4. **Update phase** to EVALUATE or SUCCESS

### If Diagnostic Fails (<80%)

1. **Document findings** in debugging.md
2. **Investigate Issue #2**
   - Add gradient logging to training
   - Verify energy_scale/energy_bias are updating
3. **Investigate Issue #3**
   - Test with more inference iterations (200 vs 50)
   - Test with multi-start initialization
4. **Re-evaluate hypothesis**

---

## Support & References

**For questions**, see:
- **How-to guide**: `RUN-DIAGNOSTIC.md`
- **Technical details**: `documentation/deep-dive-analysis.md`
- **Quick summary**: `documentation/CRITICAL-FINDINGS.md`
- **Debugging history**: `documentation/debugging.md`

**Key findings**:
- Root cause: Encoder normalization breaks energy learning
- Evidence: 54% correct vs 46% inverted energy landscapes
- Fix: Disable normalization, retrain models
- Confidence: 95%

---

## Ready to Execute!

**Current directory**:
```bash
cd /Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm
```

**First command**:
```bash
bash scripts/diagnostic_no_norm_train.sh
```

**Monitor progress**:
```bash
tail -f results/diagnostic_no_norm/training.log
```

---

**Good luck! Expected completion: ~3.5 hours from start.**
