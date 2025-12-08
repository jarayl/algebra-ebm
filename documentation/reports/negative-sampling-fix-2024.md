# Energy Gap Plateau Fix: Negative Sampling Strategy

## Issue Identified
The energy gap was plateauing at ~6.5 instead of reaching 8+ during training.

## Root Cause Analysis
The negative sampling strategy was corrupting **noise** instead of **x_start**:

```python
# OLD (BROKEN):
xmin_noise = self.q_sample(x_start=x_start, t=t, noise=noise * 3.0)
```

The diffusion formula is:
```
q_sample(x, t, n) = α_t * x + σ_t * n
```

Where:
- α_t (alpha) decreases from 1.0 → 0.0 as t increases
- σ_t (noise scale) increases from 0.0 → 1.0 as t increases

### Problem: Timestep-Dependent Separation

| Timestep | α (alpha) | σ (noise) | OLD Distance | NEW Distance |
|----------|-----------|-----------|--------------|--------------|
| 10       | 0.999     | 0.02      | 0.44         | **15.50**    |
| 50       | 0.996     | 0.09      | 1.46         | **16.14**    |
| 100      | 0.986     | 0.17      | 2.65         | **15.75**    |
| 500      | 0.702     | 0.71      | 11.36        | 11.49        |
| 800      | 0.305     | 0.95      | 15.69        | 4.83         |

**Critical Issue**: At t=10 (final denoising step), OLD method gives nearly identical positive/negative samples (dist=0.44), making it impossible for the EBM to learn fine discrimination!

## Fix Applied

```python
# NEW (FIXED):
if strategy_name == 'heavy_gaussian':
    # Corrupt x_start, then apply diffusion with SAME noise
    corruption_noise = torch.randn_like(x_start) * 2.0
    x_corrupted = x_start + corruption_noise
    xmin_noise = self.q_sample(x_start=x_corrupted, t=t, noise=noise)
```

This ensures:
1. Corruption is applied in **data space** (x_start), not noise space
2. Same diffusion noise is used for both positive and negative
3. Samples remain distinguishable at **all timesteps**, especially low t where precision matters

## Verification Results

### Isolated EBM Training (2000 steps):
```
Step    0: gap=0.74
Step  400: gap=8.84  
Step 1800: gap=8.91

Final: E(correct)=1.03, E(wrong)=9.89, Gap=8.86 ✓
```

## Files Modified
- `src/diffusion/denoising_diffusion_pytorch_1d.py` (lines 880-915)

## Strategies Updated
- `heavy_gaussian`: Now corrupts x_start + 2.0*randn
- `extreme_gaussian`: Now corrupts x_start + 4.0*randn  
- `pure_random`: Now uses random x_start + same diffusion noise
- `semantic`: Now properly applies q_sample after permutation
- Fallback: Now uses shuffled x_starts (semantically wrong answers)
