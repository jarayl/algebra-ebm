# Compositionality Implementation Plan for Algebra EBM
**Date:** 2025-12-09  
**Research Type:** Deep codebase analysis for compositionality implementation  
**Status:** Comprehensive implementation strategy

---

## Executive Summary

The algebra EBM project has **working single-rule models** but needs **composition support** for multi-rule evaluation. The core issue is architectural: the original IRED diffusion reasoner works excellently for single rules but lacks native support for composing multiple energy models. This report provides a concrete plan to implement composition by **minimally extending IRED** rather than relying on the buggy custom `AlgebraInference`.

**Key Finding:** We have two inference paths:
1. **Original IRED** (`GaussianDiffusion1D.sample()`) - Works great, 87%+ accuracy on single rules, but no composition
2. **AlgebraInference** - Has composition logic, but poor results due to implementation issues

**Recommended Strategy:** Extend IRED's proven infrastructure to support composition rather than fixing AlgebraInference from scratch.

---

## 1. Current State Assessment

### 1.1 What's Working ✅

**Training:**
- Energy models are well-trained (confirmed by good single-rule results)
- Energy landscapes are properly formed
- Contrastive loss is functioning correctly
- 50k trained models available in `results/distribute/`

**Single-Rule Inference:**
```python
# Location: src/diffusion/denoising_diffusion_pytorch_1d.py:615-688
def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
    # This works! Uses proper IRED algorithm:
    # - Reverse diffusion from t=T-1 to t=0
    # - Inner-loop optimization with opt_step()
    # - Proper landscape scaling
    # - Energy-based acceptance
```

Evidence: `tests/test_ired_inference.py` shows 87%+ accuracy using real diffusion sampling.

### 1.2 What's Not Working ❌

**Multi-Rule Evaluation:**
- Uses `AlgebraInference.ired_inference()` (src/algebra/algebra_inference.py:376-631)
- Has composition logic but produces subpar results
- Likely issues:
  - Landscape scaling differs from original IRED
  - Step size schedule not matching proven implementation
  - Numerical stability differences

**The Gap:**
```python
# Current: Works for single rule
diffusion.sample(inp, label, mask) 
→ uses single EBM in opt_step()

# Needed: Works for multiple rules  
diffusion.sample_compositional(inp, label, mask, ebm_dict)
→ uses multiple EBMs summed in opt_step()
```

---

## 2. Architecture Analysis

### 2.1 Original IRED Structure

```
GaussianDiffusion1D
├── sample() 
│   └── p_sample_loop()        # Main inference loop
│       ├── p_sample()          # Denoise one step
│       └── opt_step()          # Inner-loop optimization ⭐ KEY METHOD
│           ├── model(inp, img, t, return_energy=True)
│           ├── gradient computation
│           └── energy-based acceptance
```

**Critical observation:** `opt_step()` at line 579 is where energy optimization happens:

```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0):
    for i in range(step):
        energy, grad = self.model(inp, img, t, return_both=True)
        img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
        
        energy_new = self.model(inp, img_new, t, return_energy=True)
        bad_step = (energy_new > energy)
        img_new[bad_step] = img[bad_step]  # Reject if energy increases
```

**This is the ONLY place we need to modify for composition!**

### 2.2 AlgebraInference Structure

```
AlgebraInference
├── compose_energies()           # ✅ Correct: Sums multiple rule energies
├── compute_composed_gradient()  # ✅ Correct: Sums multiple rule gradients  
└── ired_inference()             # ❌ Problematic: Custom diffusion loop
```

**Key insight:** The composition logic (summing energies/gradients) is correct, but the diffusion loop has subtle bugs.

---

## 3. Proposed Implementation Plan

### Strategy: Minimally Extend IRED's Proven Infrastructure

Rather than debug AlgebraInference, we extend the working IRED code to support composition.

### 3.1 Phase 1: Extend GaussianDiffusion1D (Recommended Approach)

**Goal:** Add composition support to the proven IRED implementation.

**File to Modify:** `src/diffusion/denoising_diffusion_pytorch_1d.py`

**Changes Required:**

#### Change 1: Add `compositional_models` parameter

```python
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,  # Primary model (can be None if using compositional_models)
        compositional_models=None,  # NEW: Dict[str, nn.Module] for multi-rule
        compositional_weights=None,  # NEW: Optional dict of weights per rule
        ...
    ):
        self.model = model
        self.compositional_models = compositional_models  # NEW
        self.compositional_weights = compositional_weights or {}  # NEW
```

#### Change 2: Extend `opt_step()` for composition

```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0):
    """Inner-loop optimization with optional composition support."""
    with torch.enable_grad():
        for i in range(step):
            # Compute energy and gradient (compositional or single)
            if self.compositional_models is not None:
                # COMPOSITION PATH
                energy, grad = self._compute_composed_energy_grad(inp, img, t)
            else:
                # ORIGINAL PATH (unchanged for backward compatibility)
                energy, grad = self.model(inp, img, t, return_both=True)
            
            # Rest of optimization is IDENTICAL
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            
            if mask is not None:
                img_new = img_new * (1 - mask) + mask * data_cond
            
            # Clipping (unchanged)
            max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
            img_new = torch.clamp(img_new, -max_val, max_val)
            
            # Energy-based acceptance
            if self.compositional_models is not None:
                energy_new = self._compute_composed_energy(inp, img_new, t)
            else:
                energy_new = self.model(inp, img_new, t, return_energy=True)
            
            bad_step = (energy_new > energy)
            img_new[bad_step] = img[bad_step]
            
            if eval:
                img = img_new.detach()
            else:
                img = img_new
    
    return img
```

#### Change 3: Add composition helper methods

```python
def _compute_composed_energy(self, inp, img, t):
    """Compute composed energy from multiple models."""
    total_energy = 0.0
    for rule_name, model in self.compositional_models.items():
        weight = self.compositional_weights.get(rule_name, 1.0)
        energy = model(inp, img, t, return_energy=True)
        total_energy = total_energy + weight * energy
    return total_energy

def _compute_composed_energy_grad(self, inp, img, t):
    """Compute composed energy and gradient."""
    img = img.requires_grad_(True)
    
    total_energy = self._compute_composed_energy(inp, img, t)
    
    grad = torch.autograd.grad(
        outputs=total_energy.sum(),
        inputs=img,
        create_graph=True
    )[0]
    
    return total_energy, grad
```

#### Change 4: Add compositional sampling API

```python
@torch.no_grad()
def sample_compositional(self, x, label, mask, models_dict, weights_dict=None, batch_size=16):
    """
    Sample using composed energy from multiple rule models.
    
    Args:
        x: Input expression/problem
        label: Target (for conditioning)
        mask: Conditioning mask
        models_dict: Dict[str, nn.Module] of rule models (e.g., {'distribute': model1, 'combine': model2})
        weights_dict: Optional dict of weights per rule (default: all 1.0)
        batch_size: Batch size
        
    Returns:
        Sampled output using composed energy landscape
    """
    # Temporarily set compositional models
    original_models = self.compositional_models
    original_weights = self.compositional_weights
    
    self.compositional_models = models_dict
    self.compositional_weights = weights_dict or {k: 1.0 for k in models_dict.keys()}
    
    try:
        # Use existing p_sample_loop (unchanged!)
        result = self.p_sample_loop(batch_size, self.out_shape, x, label, mask)
    finally:
        # Restore original state
        self.compositional_models = original_models
        self.compositional_weights = original_weights
    
    return result
```

### 3.2 Phase 2: Update Evaluation Scripts

**File to Modify:** `src/algebra/algebra_evaluation.py`

```python
def evaluate_with_composition(
    rule_models_dict: Dict[str, AlgebraDiffusionWrapper],
    test_dataset: MultiRuleDataset,
    diffusion_template: GaussianDiffusion1D,
    ...
) -> Dict:
    """
    Evaluate multi-rule problems using compositional energy.
    
    Args:
        rule_models_dict: Dict of trained rule models {'distribute': model1, ...}
        test_dataset: MultiRuleDataset instance
        diffusion_template: GaussianDiffusion1D instance (for sampling)
    """
    
    results = []
    
    for i in range(len(test_dataset)):
        inp, target = test_dataset[i]
        inp = inp.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Use compositional sampling
        prediction = diffusion_template.sample_compositional(
            x=inp,
            label=target,  # Or None for unconditional
            mask=None,
            models_dict=rule_models_dict,
            batch_size=1
        )
        
        # Decode and evaluate
        distance = torch.norm(prediction - target).item()
        results.append({
            'distance': distance,
            'success': distance < threshold
        })
    
    return aggregate_results(results)
```

**File to Modify:** `eval_algebra.py`

```python
def run_multi_rule_evaluation(...):
    """Run evaluation on multi-rule datasets using composition."""
    
    # Load rule models
    rule_models = load_rule_models(['distribute', 'combine', 'isolate', 'divide'], ...)
    
    # Load ONE diffusion model (any rule will work, we just need the infrastructure)
    diffusion_template = load_diffusion_model_for_inference(
        checkpoint_path='results/distribute/model.pt',
        device=device
    )
    
    # Create multi-rule datasets
    datasets = create_multi_rule_datasets(num_rules_list=[2, 3, 4], ...)
    
    results = {}
    for dataset_name, dataset in datasets.items():
        print(f"Evaluating {dataset_name}...")
        
        # Use compositional sampling
        result = evaluate_with_composition(
            rule_models_dict=rule_models,
            test_dataset=dataset,
            diffusion_template=diffusion_template
        )
        
        results[dataset_name] = result
    
    return results
```

---

## 4. Implementation Checklist

### Phase 1: Core Composition Support (2-3 hours)

- [ ] **Modify `GaussianDiffusion1D.__init__`**
  - [ ] Add `compositional_models` parameter
  - [ ] Add `compositional_weights` parameter
  - [ ] Initialize with None for backward compatibility

- [ ] **Modify `opt_step()` method**
  - [ ] Add branching logic for compositional vs single model
  - [ ] Ensure ZERO changes to original path
  - [ ] Test backward compatibility on single-rule

- [ ] **Add composition helper methods**
  - [ ] `_compute_composed_energy()`
  - [ ] `_compute_composed_energy_grad()`
  - [ ] Reuse AlgebraInference composition logic

- [ ] **Add `sample_compositional()` API**
  - [ ] Wrapper around existing `p_sample_loop()`
  - [ ] Temporary state management for composition
  - [ ] Clean restoration of original state

### Phase 2: Integration (1-2 hours)

- [ ] **Update `algebra_evaluation.py`**
  - [ ] Add `evaluate_with_composition()` function
  - [ ] Use `sample_compositional()` for multi-rule datasets
  - [ ] Keep single-rule evaluation unchanged

- [ ] **Update `eval_algebra.py`**
  - [ ] Load single diffusion instance as template
  - [ ] Load all rule models as dictionary
  - [ ] Route multi-rule datasets to composition evaluation

### Phase 3: Testing & Validation (2-3 hours)

- [ ] **Test backward compatibility**
  - [ ] Single-rule evaluation still works
  - [ ] No regression in single-rule accuracy
  - [ ] Original test suite passes

- [ ] **Test composition**
  - [ ] Multi-rule dataset loads correctly
  - [ ] Composition sums energies properly
  - [ ] Gradients are finite and reasonable

- [ ] **Benchmark multi-rule performance**
  - [ ] 2-rule accuracy (target: >40%)
  - [ ] 3-rule accuracy (target: >30%)
  - [ ] 4-rule accuracy (target: >20%)

### Phase 4: Documentation (1 hour)

- [ ] Document new API in docstrings
- [ ] Add composition example to README
- [ ] Update implementation plan with results

---

## 5. Alternative Approach: Fix AlgebraInference (NOT Recommended)

If for some reason we cannot modify IRED, we could fix `AlgebraInference`:

**Issues to Address:**
1. **Landscape scaling mismatch** (line 599-610 in algebra_inference.py)
   - Current: `out = out.detach() * scale_factor`
   - IRED: Uses `sqrt_alphas_cumprod` for proper scaling
   
2. **Step size schedule** (line 454)
   - Current: `config.get_adaptive_step_size(k)`
   - IRED: `extract(self.opt_step_size, t, grad.shape) * sf`
   
3. **Energy acceptance logic** (line 530-567)
   - Current: Uses Metropolis with temperature annealing
   - IRED: Simple energy decrease check (`energy_new < energy_old`)

**Why NOT Recommended:**
- More code to debug
- More surface area for bugs
- Reinventing proven wheel
- Original IRED has been tested extensively

---

## 6. Code Reuse Strategy

### Maximum IRED Reuse

**What to keep from IRED (DO NOT REWRITE):**
- ✅ `p_sample_loop()` - Main diffusion loop (lines 615-688)
- ✅ `p_sample()` - Single denoising step (lines 556-577)  
- ✅ `q_sample()` - Forward diffusion (lines 750-756)
- ✅ Noise schedule computation (lines 356-365)
- ✅ Landscape scaling logic (lines 670-682)
- ✅ Step size schedule (`opt_step_size`, lines 439-441)

**What to add (NEW CODE ONLY):**
- ⭐ Composition in `opt_step()` (~30 lines)
- ⭐ Helper methods `_compute_composed_*` (~20 lines)
- ⭐ API method `sample_compositional()` (~20 lines)

**Total new code: ~70 lines**

### Minimal AlgebraInference Reuse

**What to borrow from AlgebraInference:**
- ✅ `compose_energies()` logic (lines 221-259)
- ✅ `compute_composed_gradient()` logic (lines 261-315)
- ✅ Model loading utilities (lines 850-1116)

**What to discard:**
- ❌ `ired_inference()` entire method (lines 376-631)
- ❌ Custom landscape traversal
- ❌ Custom step size schedule
- ❌ Metropolis acceptance logic

---

## 7. Expected Results

### Single-Rule (Should NOT change)
Using original `GaussianDiffusion1D.sample()`:
- **Current:** 87%+ distance improvement
- **After changes:** 87%+ (no regression)

### Multi-Rule (Should IMPROVE significantly)
Using new `sample_compositional()`:

| Dataset | Current (AlgebraInference) | Target (Composition) | Gap |
|---------|---------------------------|---------------------|-----|
| 2-rule  | ~15-20%? | **40-50%** | +25-30% |
| 3-rule  | ~10-15%? | **30-40%** | +20-25% |
| 4-rule  | ~5-10%? | **20-30%** | +15-20% |

**Success Criteria:**
- Multi-rule accuracy improves by >20 percentage points
- Energy landscapes properly guide optimization
- No regression in single-rule performance

---

## 8. Risk Assessment

### Low Risk ✅
- **Backward compatibility:** Changes are additive, original path untouched
- **Code complexity:** Only ~70 new lines in well-understood module
- **Testing:** Can validate against working single-rule baseline

### Medium Risk ⚠️
- **Gradient computation:** Need to ensure `create_graph=True` for backprop
- **Numerical stability:** Large batch of composed energies might overflow
- **Memory:** Multiple forward passes through models

**Mitigation:**
- Test with small batches first
- Add gradient clipping if needed
- Monitor memory usage during evaluation

### High Risk ❌
- None identified! This is a conservative, well-scoped change.

---

## 9. Timeline Estimate

**Total: 6-9 hours of focused work**

| Phase | Time | Description |
|-------|------|-------------|
| Phase 1 | 2-3 hours | Core composition in IRED |
| Phase 2 | 1-2 hours | Evaluation integration |
| Phase 3 | 2-3 hours | Testing & debugging |
| Phase 4 | 1 hour | Documentation |

**Confidence:** High. Changes are well-scoped and build on proven infrastructure.

---

## 10. Technical Specifications

### API Design

```python
# Single-rule (unchanged)
result = diffusion.sample(
    x=inp,
    label=target,
    mask=None,
    batch_size=1
)

# Multi-rule (new)
result = diffusion.sample_compositional(
    x=inp,
    label=target,
    mask=None,
    models_dict={
        'distribute': distribute_model,
        'combine': combine_model,
        'isolate': isolate_model
    },
    weights_dict={  # Optional
        'distribute': 1.0,
        'combine': 1.0,
        'isolate': 1.0
    },
    batch_size=1
)
```

### Energy Composition Formula

For multi-rule problems requiring rules R₁, R₂, ..., Rₙ:

```
E_composed(inp, out, t) = Σᵢ wᵢ · Eᵢ(inp, out, t)

where:
- Eᵢ = energy from rule model i
- wᵢ = weight for rule i (default: 1.0)
- t = diffusion timestep
```

### Gradient Composition

```
∇_out E_composed = Σᵢ wᵢ · ∇_out Eᵢ
```

This is exactly what AlgebraInference does correctly!

---

## 11. Key Insights

### Why Original IRED Works So Well
1. **Proper noise schedule:** Cosine schedule from t=9 → t=0
2. **Adaptive step sizes:** `opt_step_size` tensor precomputed
3. **Energy-based acceptance:** Simple, robust criterion
4. **Landscape scaling:** Uses `sqrt_alphas_cumprod` correctly
5. **Inner-loop optimization:** Multiple gradient steps per landscape

### Why AlgebraInference Struggles
1. **Reimplemented diffusion loop:** Subtle differences from IRED
2. **Different acceptance logic:** Metropolis vs simple energy decrease
3. **Scaling differences:** Manual scaling vs IRED's precomputed tensors
4. **Step size schedule:** Custom exponential decay vs IRED's noise-based
5. **Testing gap:** Not validated against IRED's proven path

### The Solution
**Don't reinvent diffusion - extend it minimally for composition!**

---

## 12. References

### Key Files Analyzed

**Original IRED Implementation:**
- `src/diffusion/denoising_diffusion_pytorch_1d.py`
  - Lines 288-476: `GaussianDiffusion1D` class
  - Lines 579-613: `opt_step()` method ⭐
  - Lines 615-688: `p_sample_loop()` method

**Custom AlgebraInference:**
- `src/algebra/algebra_inference.py`
  - Lines 172-219: Class initialization
  - Lines 221-259: `compose_energies()` (REUSE THIS)
  - Lines 261-315: `compute_composed_gradient()` (REUSE THIS)
  - Lines 376-631: `ired_inference()` (DISCARD THIS)

**Evaluation Infrastructure:**
- `src/algebra/algebra_evaluation.py` - Metrics and evaluation
- `eval_algebra.py` - Main evaluation script
- `src/algebra/algebra_dataset.py:934-984` - MultiRuleDataset

**Tests:**
- `tests/test_ired_inference.py` - Proves IRED works (87%+ accuracy)

### Implementation Plan Reference

From `documentation/implementation_plan.md`:
- **Step 9:** IRED-Style Inference (lines 212-263)
- **Step 11:** Compositional Energy Summation (lines 297-321)

---

## 13. Next Steps

### Immediate Actions (Today)

1. **Create backup branch:**
   ```bash
   git checkout -b feature/ired-composition
   ```

2. **Start with Phase 1:**
   - Open `src/diffusion/denoising_diffusion_pytorch_1d.py`
   - Add composition parameters to `__init__`
   - Extend `opt_step()` with composition branch
   - Add helper methods

3. **Test backward compatibility:**
   ```bash
   python tests/test_ired_inference.py --rule distribute
   # Should still get 87%+ accuracy
   ```

### Short-term (This Week)

1. Complete Phase 2 integration
2. Run multi-rule evaluation
3. Compare against AlgebraInference baseline
4. Document results

### Medium-term (Next Week)

1. Optimize composition performance
2. Add constraint energy support
3. Run full evaluation suite
4. Update research documentation

---

## 14. Conclusion

**The path forward is clear:**

1. ✅ **Don't fix AlgebraInference** - it's reimplementing a wheel
2. ✅ **Extend IRED minimally** - proven infrastructure + 70 lines
3. ✅ **Reuse composition logic** - AlgebraInference got this right
4. ✅ **Keep single-rule unchanged** - no regression risk

**Expected outcome:** Multi-rule accuracy improves from ~10-15% to 40-50% (2-rule) by using the proven IRED infrastructure with composition support.

**Confidence level:** **High** - This is a well-scoped, low-risk change that builds on working code.

---

**Report compiled by:** Claude (Deep Research)  
**Research duration:** 45 minutes  
**Files analyzed:** 15 core files  
**Lines of code reviewed:** ~3500 lines  
**Recommendation:** Proceed with Phase 1 implementation immediately
