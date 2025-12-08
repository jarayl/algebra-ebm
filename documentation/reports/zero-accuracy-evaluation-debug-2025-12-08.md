# Deep Research: Zero Accuracy Evaluation Failure

**Date:** 2025-12-08  
**Research Question:** Why does evaluation return 0.000 accuracy with flat energy landscape?

## Executive Summary

The evaluation is returning **0.0 accuracy** due to **three critical failures working in combination**:

1. **FLAT ENERGY LANDSCAPE (Primary)**: The model outputs identical energies for all inputs (std=nan, min=max=mean). This indicates the model was either not trained properly or the `energy_scale` and `energy_bias` parameters were not learned during training. The energy landscape has no discriminative power.

2. **DECODER DISTANCE THRESHOLD MISMATCH (Secondary)**: The decoder's distance threshold (default 35.0) is just below the actual decoding distances (~36.2), causing all decodings to be rejected as invalid despite being the "best" matches available.

3. **MISSING MODEL CHECKPOINTS (Contextual)**: The models are on the cluster but the evaluation code assumes they're loaded correctly. The loaded model likely has untrained `energy_scale=1.0` and `energy_bias=0.0` (default initialization values), confirming the model was never trained or the checkpoint doesn't contain these parameters.

**Critical Finding**: This matches the known issue documented in `documentation/reports/energy-landscape-flatness-research-2025-12-06.md`. The flat energy landscape was already identified as a systemic problem requiring loss scale rebalancing during training.

---

## Research Scope

### Original Question
Why does evaluation return 0.000 accuracy with warning "No valid decoding found. Best distance: 36.1951"?

### Sub-Questions Investigated
1. What causes energy statistics to show std=nan (identical min/max/mean)?
2. Why does the decoder reject all predictions with distance 36.2 when threshold is 35.0?
3. How are `energy_scale` and `energy_bias` initialized and trained?
4. What is the relationship between flat energy landscapes and decoding failures?
5. What are the required fixes to make evaluation work?

### Files/Systems Analyzed
- `src/algebra/algebra_models.py` - Energy computation and initialization
- `src/algebra/algebra_inference.py` - Model loading and inference
- `src/algebra/algebra_evaluation.py` - Evaluation pipeline
- `src/algebra/algebra_encoder.py` - Decoder implementation
- `eval_algebra.py` - Evaluation script
- `documentation/reports/energy-landscape-flatness-research-2025-12-06.md` - Known issues

---

## Key Findings

### Finding 1: Flat Energy Landscape - All Energies Identical

**Evidence:**
- Log output: `min=1.356076e+01, max=1.356076e+01, mean=1.356076e+01, std=nan`
- `src/algebra/algebra_models.py:216-220` - Flat landscape detection code
- `src/algebra/algebra_models.py:85-86` - `energy_scale` and `energy_bias` parameters

**Analysis:**

The energy statistics show **std=nan** which occurs when all values in a batch are identical. Looking at the detection code:

```python
# src/algebra/algebra_models.py:216-220
if energy_stats['std'] < 1e-6 or energy_stats['min'] == energy_stats['max']:
    logger.error(f"FLAT ENERGY LANDSCAPE DETECTED in {self.rule_name or 'unnamed'} model!")
    logger.error(f"All energies are identical: {energy_stats['mean']:.6f}. This breaks inference completely.")
    logger.error(f"Model parameters: energy_scale={self.energy_scale.item():.6f}, energy_bias={self.energy_bias.item():.6f}")
    logger.error("Possible causes: 1) Model undertrained 2) FiLM layers dominating input 3) Incorrect loss function")
```

The energy is computed as:
```python
# src/algebra/algebra_models.py:200
energy = self.energy_scale * raw_energy + self.energy_bias
```

When the model outputs identical energies for ALL inputs regardless of their content, this indicates:
1. The network weights produce identical outputs (raw_energy is constant)
2. OR the `energy_scale ≈ 0` making all energies collapse to `energy_bias`
3. OR the model was never trained and still at initialization

**Why This Breaks Inference:**
- Inference uses energy gradients to optimize: `out = out - step_size * grad_energy`
- With flat landscape, all gradients are zero (no variation to optimize)
- The optimization accepts every step (acceptance_rate=1.000) because all energies are equal
- But the resulting embeddings are meaningless random walks

**Confidence:** High - The logs explicitly show this condition and the code has detection for it.

---

### Finding 2: Decoder Distance Threshold Just Below Actual Distances

**Evidence:**
- Warning: `No valid decoding found. Best distance: 36.1951`
- `eval_algebra.py:649` - Default threshold: `distance_threshold=35.0`
- `src/algebra/algebra_inference.py:663` - Emergency threshold comment

**Analysis:**

The decoder rejects predictions when distance > threshold:

```python
# src/algebra/algebra_inference.py:737-745
if decoded_eq is not None and distance <= distance_threshold:
    result['success'] = True
    result['output_equation'] = decoded_eq
    result['decoding_distance'] = distance
    logger.info(f"Solution found: '{decoded_eq}' (distance: {distance:.4f})")
else:
    logger.warning(f"No valid decoding found. Best distance: {distance:.4f}")
    result['output_equation'] = decoded_eq  # May be None
    result['decoding_distance'] = distance
```

The issue is **threshold mismatch**:
- Threshold: 35.0
- Actual distance: 36.1951
- Gap: 1.2 units

This is a **symptom**, not the root cause. The distances are high because:
1. Flat energy landscape produces random/meaningless embeddings
2. Random embeddings don't match any valid equation in the decoder's candidate set
3. Best match is still far away (36.2 distance units)

The codebase already recognized this, with emergency increases:
```python
# src/algebra/algebra_inference.py:663
distance_threshold: float = 50.0,  # EMERGENCY: Further increased due to flat energy landscape causing massive distances
```

**However**, the evaluation script uses the default 35.0, not the emergency 50.0!

**Confidence:** High - Exact distance (36.1951) vs threshold (35.0) is explicit in logs.

---

### Finding 3: Model Checkpoint Loading Missing energy_scale/energy_bias

**Evidence:**
- `src/algebra/algebra_inference.py:954-1087` - Model loading with parameter initialization
- `src/algebra/algebra_inference.py:959-965` - Default initialization for missing parameters
- `src/algebra/algebra_models.py:85-86` - Default values

**Analysis:**

The model loading code has extensive handling for missing `energy_scale` and `energy_bias`:

```python
# src/algebra/algebra_inference.py:959-965
if 'ebm.energy_scale' in missing_keys.missing_keys:
    wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
    logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")

if 'ebm.energy_bias' in missing_keys.missing_keys:
    wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
    logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
```

This pattern appears **5 times** in the loading code, suggesting it's a common issue. The checkpoints on the cluster likely:
1. Were saved before `energy_scale`/`energy_bias` parameters were added to the model
2. OR were saved from training that didn't update these parameters (per the flat landscape issue)
3. The loading code initializes them to defaults (1.0, 0.0) which produces no energy scaling

Even with `energy_scale=1.0`, if the raw network output is constant (FiLM domination or untrained weights), the energy will still be flat.

**Confidence:** Medium-High - The code structure strongly suggests this is a real issue, but can't verify without cluster access.

---

### Finding 4: Evaluation Uses Wrong Distance Threshold

**Evidence:**
- `eval_algebra.py:649` - Sets `distance_threshold=35.0`
- `src/algebra/algebra_inference.py:663` - Default is `50.0` (emergency value)
- Logs show: `Best distance: 36.1951`

**Analysis:**

There's a disconnect between the inference module's updated default and the evaluation script:

```python
# eval_algebra.py:649 (WRONG - uses old value)
decoder = create_decoder_with_default_candidates(encoder, distance_threshold=35.0)

# vs

# src/algebra/algebra_inference.py:663 (CORRECT - emergency value)
distance_threshold: float = 50.0,  # EMERGENCY: Further increased due to flat energy landscape
```

The evaluation script was not updated when the emergency threshold increase was made. Even worse, the comment in inference.py explicitly states:

```python
# src/algebra/algebra_inference.py:712-713
if distance_threshold >= 3.0:  # Warn for any significantly elevated threshold
    logger.warning(f"Using elevated distance threshold {distance_threshold:.1f} "
                 f"(normal: 1.5). If >= 50.0, this indicates FLAT ENERGY LANDSCAPE requiring model retraining.")
```

**The threshold >= 50.0 is explicitly a warning sign of flat energy landscape**, yet the eval script uses 35.0 which is still too low!

**Confidence:** High - Direct mismatch in source code.

---

### Finding 5: The "Invalid Rate: 0.000" Paradox

**Evidence:**
- Log: `Accuracy: 0.000, Invalid Rate: 0.000, L2 Distance: 1.000`
- All predictions rejected but none marked invalid

**Analysis:**

This reveals a subtle issue in the metrics:
- **Accuracy: 0.000** - No predictions matched targets
- **Invalid Rate: 0.000** - No predictions were syntactically invalid
- **L2 Distance: 1.000** - Normalized distance metric

The predictions are being made and are syntactically valid equations, but they're:
1. Wrong answers (accuracy 0%)
2. NOT being checked for correctness (because they were rejected by distance threshold)

Looking at the evaluation code:
```python
# src/algebra/algebra_evaluation.py:90-94
if pred_eq is None:
    result['error'] = 'No prediction (failed decoding)'
    detailed_results.append(result)
    continue
```

When decoding fails (distance > threshold), `pred_eq` might still be set (the "best" candidate) but marked as failed. This allows syntax checking but prevents equivalence checking.

**Confidence:** Medium - Inference based on code logic.

---

## Failure Cascade

```
Training Issues (Root Cause)
    ↓
Loss scale imbalance (MSE 99.7%, Energy 0.3%)
    ↓
energy_scale and energy_bias not learned
    ↓
Checkpoint saved without proper energy parameters
    ↓
Model loaded with default energy_scale=1.0, energy_bias=0.0
    ↓
FLAT ENERGY LANDSCAPE (std=nan)
    ↓
Inference produces random embeddings
    ↓
Decoder distance > threshold (36.2 > 35.0)
    ↓
All predictions rejected
    ↓
ACCURACY = 0.000
```

---

## Recommendations

### IMMEDIATE FIX #1: Update Distance Threshold

**File**: `eval_algebra.py:649`

**Change:**
```python
# BEFORE
decoder = create_decoder_with_default_candidates(encoder, distance_threshold=35.0)

# AFTER
decoder = create_decoder_with_default_candidates(encoder, distance_threshold=50.0)
```

**Expected Impact**: Will allow decoding to succeed and show actual accuracy (likely still low due to flat landscape).

---

### IMMEDIATE FIX #2: Add Energy Parameter Verification

**File**: `src/algebra/algebra_inference.py` (add after line 1095)

```python
# Verify energy parameters are reasonable
for rule_name, wrapper in rule_models.items():
    scale = wrapper.ebm.energy_scale.item()
    bias = wrapper.ebm.energy_bias.item()
    
    if scale == 1.0 and bias == 0.0:
        logger.error(f"⚠️  CRITICAL: Model '{rule_name}' has default energy parameters!")
        logger.error(f"   energy_scale=1.0, energy_bias=0.0 (never trained)")
        logger.error(f"   This will cause FLAT ENERGY LANDSCAPE and zero accuracy")
        
        raise ValueError(f"Model '{rule_name}' has untrained energy parameters")
```

**Expected Impact**: Prevents wasting compute on broken models.

---

### TRAINING FIX: Implement Adaptive Loss Scaling

See `energy-landscape-flatness-research-2025-12-06.md:497-531` for complete implementation.

**Summary**: Balance MSE and energy loss contributions during training.

---

## Additional Context

### Why std=nan Instead of 0.0?

When all values are identical, PyTorch returns `nan` for standard deviation (mathematically undefined for constant sequences).

### The Acceptance Rate Mystery

`Acceptance rate: 1.000` (100%) is a RED FLAG:
- All gradient steps accepted because all energies are equal
- High acceptance usually means flat landscape
- Should be 20-60% for healthy optimization

### Why L2 Distance = 1.000?

Normalized metric indicating predictions are maximally far from targets.

---

## Sources Consulted

### Files Read
- `src/algebra/algebra_models.py` (501 lines)
- `src/algebra/algebra_inference.py` (1199 lines)
- `src/algebra/algebra_evaluation.py` (973 lines)
- `eval_algebra.py` (752 lines)
- `documentation/reports/energy-landscape-flatness-research-2025-12-06.md` (913 lines)

**Total**: ~3,500 lines analyzed

---

## Conclusion

The zero accuracy is caused by:
1. **Flat energy landscape** from training issues (root cause)
2. **Distance threshold too low** (configuration drift)  
3. **Missing/untrained parameters** (checkpoint issues)

**Quick fix**: Update threshold to 50.0 in eval_algebra.py:649  
**Real fix**: Retrain models with adaptive loss scaling

The evaluation code is working correctly - it's detecting that the models are broken.