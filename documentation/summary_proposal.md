# Project Summary: Operator-Level Compositional Energy-Based Reasoning

## Core Goal
Test whether **modular energy-based models** can achieve **zero-shot compositional generalization** in symbolic algebra by learning individual rule energies and composing them at inference time.

## Key Innovation
**Train separately, compose at test time:**
- Train one energy function per algebraic rule (distribute, combine like terms, isolate variable, divide coefficient)
- Each energy only sees single-step problems during training
- At test time, **sum these energies** to solve multi-step equations requiring 2-4 rules in sequence
- This extends IRED (Iterative Reasoning via Energy Diffusion), which mentioned compositional possibilities but never implemented them

## Three Evaluation Axes
1. **Zero-shot generalization**: Can the model solve multi-rule problems it never trained on?
2. **Runtime constraint control**: Can you inject new constraints (e.g., "solution must be positive") without retraining?
3. **Inference quality**: Does IRED's annealed optimization work on composed landscapes?

## Process

### Training Phase
1. Generate single-rule equation pairs: `input_state → output_state` after one rule application
2. Train 4 separate energy models (one per rule) using:
   - **Denoising loss**: Predict noise in corrupted solutions
   - **Contrastive loss**: Ensure valid solutions have lower energy than invalid ones
3. Use IRED's annealed landscape approach (K=10 progressively sharper energy surfaces)

### Inference Phase
1. Start with random noise in embedding space
2. For each landscape k=1 to K:
   - Compute **composed energy** = sum of all rule energies
   - Run gradient descent to minimize this sum
   - Scale output for next landscape level
3. Decode final embedding back to equation string
4. Verify correctness using SymPy

## Expected Results
| Model | Single-Rule | Multi-Rule |
|-------|-------------|------------|
| Monolithic IRED | ~90% | ~20-30% |
| **Modular Sum (proposed)** | ~85% | **~50-60%** |

The 2-3x improvement on multi-rule problems would demonstrate compositional benefit.

## Why This Matters
- Shows EBMs can achieve **modular reasoning** similar to neuro-symbolic systems (like Neural Logic Machines) but through continuous optimization rather than program execution
- Provides **flexible constraint injection** at test time without retraining
- First empirical test of rule-level energy composition in a reasoning domain

## Timeline
3-4 months for proof-of-concept implementation and evaluation on linear algebra problems.