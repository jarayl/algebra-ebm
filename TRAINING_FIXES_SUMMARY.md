# Training Fixes Summary

## Problem
Training exhibited the following issues:
1. **Loss not converging** - Loss remained high or oscillated wildly
2. **Model collapsing** - Model outputting only a few unique outputs
3. **Flat energy landscape** - All input pairs (valid, shuffled, random) had identical energies (~0.2-0.9)

## Root Causes Identified

### Root Cause 1: FiLM Layer Initialization Dominating Input Signal
**Problem**: The FiLM (Feature-wise Linear Modulation) layers for timestep conditioning were initialized with standard Xavier uniform weights, producing outputs with std ~0.24 that dominated the hidden state signal (std ~0.04).

**Evidence**:
```
FC2 FiLM bias: mean=0.0199, std=0.2376  (before fix)
Hidden state after fc2 (pre-FiLM): std=0.0364

FiLM bias (0.24) >> hidden state (0.04) = FiLM dominates!
```

**Fix**: Initialize FiLM layers with very small weights (std=0.01) so they start near-identity:
```python
elif name in ['t_map_fc2', 't_map_fc3']:
    nn.init.normal_(module.weight, std=0.01)  # Very small weights
    if module.bias is not None:
        nn.init.zeros_(module.bias)  # Zero bias
```

### Root Cause 2: Gradient Wrapper Not Handling Detached Tensors
**Problem**: The `AlgebraDiffusionWrapper` called `requires_grad_(True)` on tensors that were already detached, which doesn't work.

**Fix**: Clone the tensor before requiring gradients:
```python
out = out.detach().clone().requires_grad_(True)  # Fixed
# Previously: out.requires_grad_(True)  # Broken - doesn't work on detached tensor
```

### Root Cause 3: Energy Scale Mismatch
**Problem**: With normalized inputs (||x||=1) and xavier initialization, raw energies were ~0.2, but contrastive loss targets were 1.0 and 10.0.

**Fix**: Added learnable energy scaling parameters:
```python
self.energy_scale = nn.Parameter(torch.tensor(1.0))
self.energy_bias = nn.Parameter(torch.tensor(0.0))
energy = self.energy_scale * raw_energy + self.energy_bias
```

## Results After Fixes

### Before Fixes
```
Valid energy:    0.22 ± 0.01
Shuffled energy: 0.22 ± 0.01
Random energy:   0.22 ± 0.01
Gap: ~0 (no discrimination)
```

### After Fixes (1000 training steps)
```
Valid energy:    1.79 ± 0.45
Shuffled energy: 9.69 ± 1.37
Random energy:   9.91 ± 0.96
Gap: 7.90 (strong discrimination)
```

## Files Modified
1. `/home/ubuntu/algebra-ebm/algebra_models.py`:
   - Fixed `_init_weights()` method with proper FiLM initialization
   - Added learnable `energy_scale` and `energy_bias` parameters
   - Fixed gradient wrapper tensor handling

2. `/home/ubuntu/algebra-ebm/diffusion_lib/denoising_diffusion_pytorch_1d.py`:
   - Adjusted contrastive loss defaults (margin=5.0, neg_target=10.0)

## How to Verify
Run the comprehensive test:
```bash
python test_training_fixes_comprehensive.py
```

Expected output: 4/4 tests passing.

## Training Recommendations
1. Use learning rate ~1e-3 for fast convergence
2. Gradient clipping (max_norm=1.0) helps stability
3. Batch size 128-256 works well
4. Expect energy gap > 3 after ~500 steps
5. Full convergence (pos~1.0, neg~10.0) after ~2000 steps
