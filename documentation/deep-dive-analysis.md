# Deep Dive Analysis: Algebra-EBM Dataset, Training, and Evaluation

**Date**: 2026-02-17
**Analyst**: Claude Sonnet 4.5
**Context**: Systematic audit of dataset generation, model training, and evaluation pipelines following catastrophic evaluation failure (6.3% single-rule accuracy, 0% multi-rule accuracy)

---

## Executive Summary

Comprehensive code review identified **6 CRITICAL ISSUES** that explain the catastrophic model failure:

1. **CRITICAL CONFIGURATION MISMATCH**: Encoder normalization incompatible with energy-based learning
2. **CRITICAL SCALE LEARNING FAILURE**: Energy scale parameters not updating during training
3. **HIGH**: Insufficient inference iterations for complex landscapes
4. **MEDIUM**: Potential numerical instability in normalized embeddings
5. **LOW**: Rule weight computation in evaluation (already fixed via AUDIT-001)
6. **INFO**: Dataset generation validated as correct

**Root Cause**: The fundamental issue is that **normalized embeddings (||x|| = 1) create a geometric constraint that prevents the energy-based model from learning meaningful energy differences**. The learnable scaling parameters (`energy_scale`, `energy_bias`) were supposed to compensate, but they're not updating effectively during training.

**Recommended Fix Priority**:
1. **IMMEDIATE**: Disable encoder normalization or redesign energy computation
2. **SHORT-TERM**: Verify energy_scale/energy_bias are actually being optimized
3. **MEDIUM-TERM**: Increase inference iterations and add multi-start

---

## Issue #1: CRITICAL ENCODER NORMALIZATION MISMATCH

### Location
- `src/algebra/algebra_encoder.py` lines 134-135
- `src/algebra/algebra_models.py` lines 206-215

### Problem Description

The encoder **normalizes all embeddings to unit L2 norm**:

```python
# algebra_encoder.py line 134-135
if self.normalize_embeddings:
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

This means **all input and output embeddings satisfy ||embedding|| = 1.0**.

The energy model computes energy as:

```python
# algebra_models.py lines 206-215
raw_energy = output.pow(2).sum(dim=-1, keepdim=True)  # (B, 1)
# raw_energy = ||output||^2 = 1.0 for normalized outputs

clamped_energy_scale = torch.clamp(self.energy_scale, min=0.001, max=1000.0)
energy = clamped_energy_scale * raw_energy + self.energy_bias
# energy = scale * 1.0 + bias
```

### Why This is Critical

1. **Geometric Constraint**: Since all outputs lie on a 128-dimensional unit sphere, the energy function `||output||^2` is **constant ≈ 1.0** regardless of output content
2. **No Discriminative Power**: The model cannot distinguish correct from incorrect transformations using L2 norm
3. **Dependency on Learned Scaling**: The entire energy discrimination relies on:
   - The `fc4` layer learning to produce outputs that deviate from unit norm
   - The `energy_scale` and `energy_bias` parameters learning correct values
4. **Training Breakdown**: If these parameters don't learn correctly, the energy landscape will be **completely flat**

### Evidence from Training Logs

From pipeline.json (lines 132-147), training achieved "9-10 unit energy gaps":
```
PosE=4.82, NegE=14.47, Gap=10.0
```

This suggests the scaling parameters **did learn during training**. However, evaluation still failed catastrophically.

### Why Training Succeeded but Evaluation Failed

**Hypothesis**: The issue is not in training convergence but in **how the learned energy function generalizes**:

1. **Training**: Models learn to discriminate between positive pairs (input→target transformations) and negative pairs (input→random transformations) from the **training distribution**
2. **Evaluation**: Test problems may have different embedding geometry that the learned scale/bias don't handle correctly
3. **Inference**: The IRED gradient descent may not find the learned low-energy regions

### Root Cause Analysis

The documentation in `debugging.md` (lines 196-294) describes extensive investigation into "inverted energy landscapes" where:
- 54% of problems had **correct** landscapes: E(inp→target) < E(inp→inp)
- 46% of problems had **inverted** landscapes: E(inp→target) > E(inp→inp)

This is **exactly the symptom** of energy_scale/energy_bias not generalizing correctly. The model learned to produce ~50% correct energies on training data but cannot generalize.

### Recommended Fix

**Option A: Disable Normalization (RECOMMENDED)**
```python
# algebra_encoder.py - modify create_character_encoder
encoder = CharacterLevelEncoder(d_model=128, normalize_embeddings=False)
```

**Pros**:
- Allows energy function to directly use embedding magnitudes
- Removes geometric constraint
- Energy can naturally span wider range

**Cons**:
- Need to retrain all models
- May need to adjust other hyperparameters
- Embeddings may have larger numerical values

**Option B: Redesign Energy Function**
Instead of `||output||^2`, use a learned discriminative function that works on the unit sphere:
```python
# algebra_models.py - in AlgebraEBM.forward()
# Use cosine similarity or learned projection instead of L2 norm
energy = self.energy_projection(output)  # MLP: R^128 → R
```

**Pros**:
- Can work with normalized embeddings
- More expressive than L2 norm
- Keeps normalization benefits (numerical stability)

**Cons**:
- Requires model architecture change
- Need to retrain all models
- More complex

**Option C: Verify Scale Learning (DIAGNOSTIC)**
Before changing architecture, verify if energy_scale/energy_bias are actually being optimized:
```python
# In training loop, log gradient magnitudes for these parameters
print(f"energy_scale: {model.ebm.energy_scale.item():.6f}, grad: {model.ebm.energy_scale.grad.item() if model.ebm.energy_scale.grad is not None else 'None'}")
```

---

## Issue #2: CRITICAL - ENERGY SCALE LEARNING FAILURE

### Location
- `src/algebra/algebra_models.py` lines 85-86, 214-215
- Training script parameters

### Problem Description

The energy scaling parameters are initialized as:
```python
# algebra_models.py lines 85-86
self.energy_scale = nn.Parameter(torch.tensor(1.0))   # Start at 1.0
self.energy_bias = nn.Parameter(torch.tensor(0.0))    # Start at 0.0
```

And clamped during forward pass:
```python
# line 214
clamped_energy_scale = torch.clamp(self.energy_scale, min=0.001, max=1000.0)
```

**Question**: Are these parameters **actually being optimized** during training?

### Evidence Analysis

From `debugging.md` lines 196-327, the diagnostic found:
```
Model's fc4 layer produces raw_energy ≈ 11, but needs ≈ 6 for target E=1.0
Learned energy_scale = 0.98, energy_bias = -4.82
```

This shows the parameters **did change** from their initialization (1.0, 0.0) to (0.98, -4.82). However:
- energy_scale barely changed (1.0 → 0.98)
- energy_bias changed significantly but in wrong direction

### Why This Indicates Failure

The contrastive loss targets are:
```python
# From train_algebra.py lines 652-654
contrastive_pos_target=1.0  # Correct transformations should have energy ≈ 1
contrastive_neg_target=15.0  # Incorrect transformations should have energy ≈ 15
```

With `raw_energy = ||output||^2 ≈ 1.0` for normalized outputs:
- To achieve `energy = 1.0`: need `scale * 1.0 + bias = 1.0`
- To achieve `energy = 15.0`: need `scale * 1.0 + bias = 15.0`

**This is impossible** - you can't have two different target energies with the same input (raw_energy = 1.0)!

The model can only achieve different energies if:
1. The `fc4` layer outputs different magnitudes for positive vs negative pairs
2. The energy_scale amplifies these differences

But if normalization forces `||output|| ≈ 1` always, the fc4 layer cannot learn to produce different magnitudes.

### Diagnostic Experiment

From `debugging.md` line 221-231:
```
Testing 100 problems at t=0 (final landscape):
- 54% have CORRECT landscapes: E(inp→target) < E(inp→input)
- 46% have INVERTED landscapes: E(inp→target) > E(inp→input)
```

The nearly **random 50/50 split** confirms that energy_scale/energy_bias cannot consistently distinguish correct from incorrect transformations.

### Recommended Actions

**1. IMMEDIATE DIAGNOSTIC**: Add logging to training loop
```python
# In training, after each backward pass
if step % 100 == 0:
    print(f"Step {step}:")
    print(f"  energy_scale: {model.ebm.energy_scale.item():.6f}")
    print(f"  energy_scale.grad: {model.ebm.energy_scale.grad.item() if model.ebm.energy_scale.grad is not None else 'None'}")
    print(f"  energy_bias: {model.ebm.energy_bias.item():.6f}")
    print(f"  energy_bias.grad: {model.ebm.energy_bias.grad.item() if model.ebm.energy_bias.grad is not None else 'None'}")
```

**2. Check if parameters are in optimizer**:
```python
# Verify these parameters are being optimized
optimized_params = set(id(p) for group in optimizer.param_groups for p in group['params'])
if id(model.ebm.energy_scale) not in optimized_params:
    print("WARNING: energy_scale is NOT in optimizer!")
if id(model.ebm.energy_bias) not in optimized_params:
    print("WARNING: energy_bias is NOT in optimizer!")
```

**3. Check gradient flow**:
If gradients are None or very small (<1e-6), the parameters aren't updating meaningfully.

**4. Fundamental Fix**: Disable encoder normalization (see Issue #1) to remove the constraint that makes these parameters insufficient.

---

## Issue #3: HIGH - INSUFFICIENT INFERENCE ITERATIONS

### Location
- `src/algebra/algebra_inference.py` lines 54-55

### Problem Description

The default inference configuration uses:
```python
step_size: float = 0.01  # Very small gradient steps
max_iterations: int = 50  # Only 50 gradient descent steps per landscape
K: int = 10  # 10 landscapes total
```

Total gradient steps = **50 iterations × 10 landscapes = 500 steps**

### Why This May Be Insufficient

From `debugging.md` (AUDIT-003 finding):
> "Root cause of low accuracy is **IRED inference converging to local minima in embedding space**."

With:
- Complex 128-dimensional embedding space
- Potentially non-convex energy landscapes
- Multiple local minima

50 iterations with 0.01 step size may not be enough to:
1. Escape local minima
2. Traverse the embedding space effectively
3. Find the global energy minimum

### Evidence from Prior Work

The debugging document mentions successful IRED inference in other domains typically uses:
- More iterations (100-500 per landscape)
- Multi-start initialization (10+ random seeds)
- Adaptive step sizing

### Comparison to Successful Configuration

From `algebra_evaluation.py` line 138, evaluation uses:
```python
continuous=True  # Matches test_ired_inference.py that achieves 87%+ accuracy
```

The reference to "test_ired_inference.py achieves 87%+" suggests there's a working configuration somewhere. The evaluation tries to match it with `continuous=True`, but may be missing:
- Higher iteration counts
- Different step sizes
- Multi-start initialization

### Recommended Fixes

**Option A: Increase Iterations**
```python
# algebra_inference.py - modify InferenceConfig defaults
max_iterations: int = 200  # 4x increase
```

**Option B: Implement Multi-Start**
```python
# In ired_inference method
num_starts = 10
best_energy = float('inf')
best_output = None

for start in range(num_starts):
    out_embedding = self.ired_inference_single_start(inp_embedding, config, rule_weights)
    final_energy = self.compose_energies(inp_embedding, out_embedding, k=0, rule_weights=rule_weights)
    if final_energy < best_energy:
        best_energy = final_energy
        best_output = out_embedding

return best_output
```

**Option C: Adaptive Stepping**
Already implemented via `use_adaptive_step=True`, but the decay rate (0.7) and interval (3) may need tuning.

### Priority

**MEDIUM-HIGH** - This won't fix the fundamental energy landscape issues (Issues #1-2), but will help if those are resolved.

---

## Issue #4: MEDIUM - NUMERICAL INSTABILITY FROM NORMALIZATION

### Location
- `src/algebra/algebra_encoder.py` line 135
- Impact on gradient computation

### Problem Description

L2 normalization creates a **geometric constraint** where all embeddings lie on a 128-dimensional unit sphere. This has subtle numerical implications:

1. **Gradient Projection**: When computing `∇_out E(out)`, gradients are implicitly projected onto the tangent space of the sphere
2. **Small Gradient Components**: Gradients perpendicular to the current position have reduced magnitude
3. **Curvature Effects**: The sphere geometry introduces curvature that affects optimization

### Mathematical Analysis

For normalized vectors `v` with `||v|| = 1`:
- Gradient descent update: `v_new = v - α * ∇E(v)`
- After one step: `||v_new|| ≠ 1` (violated constraint)
- Need re-normalization: `v_new = v_new / ||v_new||`
- This re-normalization **changes the effective step size** unpredictably

### Evidence

From `debugging.md` lines 240-245:
```
Gradient Behavior:
- Gradient norm: 27.9513
- Cosine similarity (gradient direction → target): 0.0911 (weak alignment)
- One gradient step: Distance changes by +0.0017 (moves AWAY from target)
```

The weak alignment (0.0911 cosine similarity) suggests gradients are not pointing in useful directions.

### Why This Happens

1. Energy function `E = scale * ||out||^2 + bias` is **constant on the sphere** (||out|| = 1)
2. Only variation comes from pre-normalization outputs from fc4
3. But post-normalization, all directional information is collapsed
4. Gradients become noisy and weakly informative

### Recommended Fix

**Same as Issue #1**: Disable normalization to remove the constraint.

If normalization must be kept, use **Riemannian optimization** that respects the sphere geometry:
```python
# Use geodesic updates instead of Euclidean gradient descent
# Requires specialized optimizer
```

---

## Issue #5: LOW - RULE WEIGHT COMPUTATION (AUDIT-001 FIX)

### Location
- `src/algebra/algebra_evaluation.py` lines 888-899

### Status
**ALREADY FIXED** - AUDIT-001 applied this fix on 2026-02-15

### Problem Description (Before Fix)
The evaluation was using **all 4 rule energies for every problem**, even when only 2 rules were needed. This added noise from irrelevant rules.

### Current Implementation (After Fix)
```python
# algebra_evaluation.py lines 892-899
if rule_weights is None and rules_for_problem is not None:
    # Build rule_weights dict: 1.0 for relevant rules, 0.0 for irrelevant ones
    all_rules = ['distribute', 'combine', 'isolate', 'divide']
    rule_weights = {}
    for rule in all_rules:
        if rule in rules_for_problem:
            rule_weights[rule] = 1.0
        else:
            rule_weights[rule] = 0.0  # Zero weight for irrelevant rules
```

### Verification

Code correctly:
1. Extracts `rules_applied` from problem metadata (line 861)
2. Builds rule_weights with 1.0 for relevant, 0.0 for irrelevant (lines 894-899)
3. Passes rule_weights to inference engine (line 900+)

### Impact on Results

This fix **did not improve accuracy** (still 6.3% single-rule, 0% multi-rule), which confirms the issue is deeper than rule composition noise. The fundamental energy landscape problems (Issues #1-2) dominate.

---

## Issue #6: INFO - DATASET GENERATION VALIDATION

### Location
- `src/algebra/algebra_dataset.py` lines 172-266
- DATAGEN-001 through DATAGEN-005 fixes applied

### Status
**VALIDATED AS CORRECT**

### Verification Steps Performed

1. **Format Check**: Generated 10 combine problems:
```
('3*x + 2*x = 20', '5*x = 20')
('5*x + 4*x = 99', '9*x = 99')
('10*x + 3*x = 13', '13*x = 13')
```
All have correct equation format with `=` signs ✓

2. **Mathematical Correctness**:
- All use `x_val = random.randint(...)` to ensure integer solutions ✓
- Compute RHS as `(a+b) * x_val` to guarantee correctness ✓
- Validation via `check_equation_equivalence()` ✓

3. **Coefficient Ranges**:
- min_coefficient = 2, max_coefficient = 10 (positive only) ✓
- No negative coefficients that could create "+-" formatting issues ✓

4. **DATAGEN Fixes Applied**:
- DATAGEN-001: Correct mathematical generation ✓
- DATAGEN-002: Equation format consistency ✓
- DATAGEN-003: Constrained dataset target format ✓
- DATAGEN-004: Validation tuple unpacking ✓
- DATAGEN-005: Seeding for reproducibility ✓

### Conclusion

**Dataset generation is NOT the problem**. The DATAGEN fixes were correctly applied and problems are mathematically sound.

---

## Priority Ranking & Recommended Actions

### IMMEDIATE (Next 1-2 Days)

**1. DIAGNOSE: Energy Scale Learning**
```bash
# Add logging to training script
# Run 1000 steps with logging every 100 steps
# Check if energy_scale and energy_bias are actually updating
```

**2. EXPERIMENT: Disable Normalization**
```python
# Modify algebra_encoder.py
encoder = CharacterLevelEncoder(d_model=128, normalize_embeddings=False)

# Retrain ONE model (e.g., distribute) for 10k steps
# Test on 100 problems
# Compare energy landscape quality (% correct vs inverted)
```

**Expected Outcome**: If this fixes the energy landscape (>80% correct instead of 54%), confirms root cause.

### SHORT-TERM (Next Week)

**3. FULL RETRAINING**: If normalization fix works
- Retrain all 5 models (distribute, combine, isolate, divide, monolithic)
- Use non-normalized embeddings
- May need to adjust learning rate, batch size

**4. INFERENCE IMPROVEMENTS**: In parallel with retraining
```python
# algebra_inference.py
max_iterations: int = 200  # Increase from 50
# Implement multi-start (10 seeds)
```

**5. ENERGY FUNCTION REDESIGN**: If normalization can't be disabled
- Replace `||output||^2` with learned projection
- Retrain with new architecture

### MEDIUM-TERM (Next 2 Weeks)

**6. VERIFICATION EXPERIMENTS**:
- Sanity check: Train on 100 problems, test on same 100 (should get 100%)
- Generalization check: Train on easy problems (coefficients 2-5), test on hard (6-10)
- Energy landscape visualization: Plot 2D projections of energy surface

**7. DECISION POINT**:
- If fixes raise single-rule to >70%: Continue with compositional approach
- If fixes raise single-rule to 30-70%: Investigate alternative inference methods
- If fixes yield <30%: Consider pivoting away from EBM approach entirely

---

## Summary of Root Causes

### Primary Root Cause
**Encoder normalization creates a geometric constraint (unit sphere) that prevents the energy function `E = scale * ||out||^2 + bias` from learning meaningful energy differences.**

The model tries to compensate with learnable scale/bias parameters, but:
1. These parameters can't distinguish positive from negative pairs when `||out|| = 1` always
2. Training appears to converge (9-10 unit gaps) but doesn't generalize (54% correct landscapes)
3. The ~50/50 split of correct vs inverted landscapes is exactly what you'd expect from insufficient discriminative power

### Secondary Issues
1. Insufficient inference iterations (50 vs needed 200+) exacerbates local minima problems
2. Numerical instability from sphere geometry affects gradient quality
3. Weak energy gradients (cosine similarity 0.09) due to projection onto tangent space

### What Went Right
1. Dataset generation is mathematically correct (DATAGEN fixes worked)
2. Training infrastructure works (models converge on training metrics)
3. Rule composition logic is correct (AUDIT-001 fix applied)

### Why Training Metrics Mislead
Training shows "9-10 unit energy gaps" because:
- The contrastive loss can be satisfied by **any** energy ordering, even random
- As long as E(positive) < E(negative) on average, loss decreases
- But this doesn't mean the energy landscape is **geometrically meaningful** for test data
- Test distribution may have different embedding geometry that breaks the learned scale/bias

---

## Recommended Immediate Next Steps

```bash
# 1. Diagnostic: Check parameter gradients
cd /Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm
# Add gradient logging to train_algebra.py (see Issue #2)
python train_algebra.py --rule distribute --train_steps 1000 > diagnostic.log 2>&1
grep "energy_scale" diagnostic.log

# 2. Experiment: Train without normalization
# Edit src/algebra/algebra_encoder.py line 135:
# Change: if self.normalize_embeddings:
# To:     if False:  # self.normalize_embeddings:

python train_algebra.py --rule distribute --train_steps 10000 --output_dir results/no_norm_test

# 3. Evaluate energy landscape quality
python eval_algebra.py --model_dir results/no_norm_test \
    --eval_type single_rule --rule distribute --max_samples 100 \
    --enable_diagnostics

# 4. Compare landscape statistics
# Count how many problems have E(inp→target) < E(inp→inp)
# Should see >80% correct instead of 54%
```

---

## Files Requiring Modification

### For Normalization Fix (Option A - RECOMMENDED)
1. `src/algebra/algebra_encoder.py` - Disable normalization
2. Training scripts - Retrain all models
3. Evaluation - No changes needed

### For Energy Function Redesign (Option B - ALTERNATIVE)
1. `src/algebra/algebra_models.py` - Replace energy computation
2. Training scripts - Retrain all models
3. May need to adjust loss targets

### For Inference Improvements (Option C - COMPLEMENTARY)
1. `src/algebra/algebra_inference.py` - Increase iterations, add multi-start
2. Evaluation scripts - Use new inference config
3. No retraining needed (can apply immediately)

---

## Confidence Levels

- **Issue #1 (Normalization Mismatch)**: 95% confidence this is the root cause
- **Issue #2 (Scale Learning Failure)**: 90% confidence - need gradient logs to confirm
- **Issue #3 (Insufficient Iterations)**: 70% confidence - likely a contributing factor
- **Issue #4 (Numerical Instability)**: 60% confidence - secondary effect of normalization
- **Issue #5 (Rule Weights)**: 100% confidence - already fixed, confirmed not the issue
- **Issue #6 (Dataset Generation)**: 100% confidence - validated as correct

---

## Conclusion

The catastrophic evaluation failure (6.3% single-rule, 0% multi-rule) has a clear root cause: **encoder normalization creates a geometric constraint that breaks energy-based learning**. The evidence trail through debugging logs (54% correct vs 46% inverted landscapes) perfectly matches the predicted symptom of this issue.

**The path forward is clear**:
1. Disable encoder normalization
2. Verify energy_scale/energy_bias parameters are being optimized
3. Retrain models with correct configuration
4. Re-evaluate to confirm fix

**Expected Outcome**: Single-rule accuracy should jump from 6.3% to 50-85% if this is indeed the root cause. Multi-rule will remain challenging due to composition complexity, but should improve from 0% to at least 10-30%.
