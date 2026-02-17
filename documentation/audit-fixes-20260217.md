# Critical Fixes Applied - 2026-02-17

## Context
After discovering 5 critical dataset bugs (DATAGEN-001 through 005), we conducted a comprehensive codebase audit that revealed multiple training and inference misconfigurations explaining the catastrophic 6.3% accuracy (expected 85%).

## Fixes Applied

### **FIX #1: AUDIT-005 - Inference Normalization Mismatch (CRITICAL)**

**Root Cause**: Models trained on unit-norm embeddings (||z|| = 1.0) but inference operated in unbounded space (||z|| ≈ 11.3).

**Impact**:
- Energy landscapes trained on unit hypersphere but evaluated off-sphere
- 10x magnitude mismatch between training and inference
- Decoder nearest-neighbor search failed due to distance inflation
- Gradients pointed in wrong directions (not constrained to tangent space)

**Fix Applied** (`src/algebra/algebra_inference.py`):
```python
# Line 424: Initialize from normalized noise
out = torch.randn(batch_size, 128, device=self.device)
out = torch.nn.functional.normalize(out, p=2, dim=-1)  # NEW: Project to unit sphere
out = out.requires_grad_(True)

# Line 511: Re-normalize after each gradient step
out_new = out - current_step_size * grad
out_new = torch.nn.functional.normalize(out_new, p=2, dim=-1)  # NEW: Stay on sphere
```

**Expected Result**: Single-rule accuracy should jump from 6.3% to 50-85% if this is the root cause.

---

### **FIX #2: Energy Scale Clamping Too Restrictive**

**Root Cause**: `energy_scale` parameter clamped to [0.1, 10.0], preventing model from learning correct energy scaling for unit-norm inputs.

**Evidence**:
- With unit-norm inputs producing `raw_energy ≈ 1.5`
- To achieve `neg_target = 15.0`, need `scale = 10.0` (at upper bound!)
- Models hitting upper bound during training, unable to learn proper scale

**Fix Applied** (`src/algebra/algebra_models.py:213`):
```python
# Changed from:
# clamped_energy_scale = torch.clamp(self.energy_scale, min=0.1, max=10.0)
# To:
clamped_energy_scale = torch.clamp(self.energy_scale, min=0.001, max=1000.0)
```

**Impact**: Allows model to learn energy scales appropriate for unit-norm embeddings.

---

### **FIX #3: FC4 Initialization Too Conservative**

**Root Cause**: Final layer initialized with gain=0.1 (very small), limiting network expressiveness. This was a "symptom fix" for energy scale issues, not addressing root cause.

**Fix Applied** (`src/algebra/algebra_models.py:110`):
```python
# Changed from:
# nn.init.xavier_uniform_(module.weight, gain=0.1)
# To:
nn.init.xavier_uniform_(module.weight, gain=0.5)
```

**Impact**: Improves network capacity to learn discriminative features. With energy_scale clamp widened, network can now express stronger patterns.

---

## Issues NOT Fixed (Require Retraining)

### **ISSUE #4: Inconsistent Energy Landscapes (54% correct, 46% inverted)**
- **Status**: Requires investigation
- **Evidence**: Testing showed 54% of problems have E(correct) < E(wrong), but 46% are inverted
- **Next Steps**:
  1. Run Validation Experiment V1 (energy landscape sanity check)
  2. If confirmed, investigate data labeling or training stability
  3. May require retraining with fixes

### **ISSUE #5: Composition-Aware Loss Targets Too Small**
- **Status**: Requires retraining
- **Evidence**: Single-rule models trained with targets divided by 4 (gap = 2.5 vs 10.0)
- **Next Steps**: Consider using standard contrastive targets (pos=1.0, neg=15.0, margin=10.0) for all models

---

## Validation Plan

### Immediate Testing (With Existing Models)
1. **Quick test**: Run eval on distribute rule with 100 problems
   - **Expected**: Accuracy jumps from 6% to 50-85%
   - **If not**: AUDIT-005 was not the sole root cause

2. **Validation Experiments** (designed by audit team):
   - V1: Energy landscape sanity check (10 min, CPU)
   - V2: Energy gap magnitude verification (10 min, CPU)
   - V4: Trivial problem test (20 min, GPU)
   - V6: Embedding round-trip test (10 min, CPU)

### If Fixes Work (Accuracy >50%)
1. Run full evaluation suite (exp_001 through exp_005)
2. Complete remaining validation experiments (V3, V5, V7)
3. Consider additional inference improvements:
   - Multi-start from multiple random initializations
   - Momentum-based optimization
   - Better temperature schedules

### If Fixes Don't Work (Accuracy <20%)
1. Run all validation experiments to diagnose remaining issues
2. Investigate Issue #4 (inverted landscapes) - likely requires retraining
3. Consider architectural pivot if problems persist

---

## Git Commits

**Submodule commit** (algebra-ebm): `2e2a6ea`
- Fixed inference normalization (AUDIT-005)
- Widened energy_scale clamp
- Increased FC4 initialization gain

**Main repo commit**: `2e8d634`
- Updated submodule reference to include fixes

---

## Next Steps

1. **IMMEDIATE**: Test fixes with existing models
   ```bash
   cd projects/algebra-ebm
   python eval_algebra.py \
     --model_dir results \
     --eval_type single_rule \
     --rule distribute \
     --single_rule_problems 100 \
     --seed 42
   ```

2. **If successful**: Cancel running diagnostic experiments (no longer needed) and run full eval suite

3. **If unsuccessful**: Execute validation experiments V1-V7 to identify remaining issues

4. **Document results**: Update pipeline.json and debugging.md with findings

---

## Files Modified

- `src/algebra/algebra_inference.py`: Inference normalization fixes
- `src/algebra/algebra_models.py`: Energy scale and FC4 initialization fixes
- `documentation/audit-fixes-20260217.md`: This document

## Audit Reports Referenced

- Training audit report (agent ae1a033)
- Evaluation/inference audit report (agent afff964)
- Representation audit report (agent a83cfe2)
- Validation experiments design (agent ab8333e)
