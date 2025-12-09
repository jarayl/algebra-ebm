# Compositionality Evaluation Implementation Plan
**Date:** 2025-12-09  
**Purpose:** Implement and evaluate compositional energy-based models vs monolithic baseline  
**Status:** Implementation Roadmap

---

## Executive Summary

This plan outlines the implementation of a comprehensive evaluation comparing **compositional iRED** (multiple specialized energy models combined) against **monolithic iRED** (single unified model). The goal is to demonstrate that compositionality enables superior zero-shot generalization on multi-rule problems.

**Key Objectives:**
1. Train monolithic iRED baseline (single model for all 4 rules)
2. Implement energy composition mechanism (combine multiple iRED models)
3. Create evaluation comparing both approaches on multi-rule problems
4. Demonstrate compositional advantage: ~25-30 percentage point improvement

**Expected Results:**
| Approach | Single-Rule Acc | Multi-Rule Acc | Advantage |
|----------|----------------|----------------|-----------|
| Monolithic | ~90% | ~20-30% | Baseline |
| **Compositional** | **~85%** | **~50-60%** | **+25-30 points** 🎯 |

---

## Current State Assessment

### ✅ Already Implemented
1. **Training Infrastructure**
   - `train_algebra.py` - Single-rule training (4 separate models)
   - `train_algebra_monolithic.py` - Monolithic training (all rules combined)
   - `CombinedAlgebraDataset` class (lines 637-899 in algebra_dataset.py)

2. **Evaluation Infrastructure**
   - `eval_algebra.py` - Main evaluation script with monolithic support
   - `algebra_evaluation.py` - Evaluation functions including `run_monolithic_evaluation()`
   - `evaluate_with_real_diffusion()` - Proven inference method (87%+ accuracy)

3. **Test Datasets**
   - 2-rule problems: 100 samples
   - 3-rule problems: 50 samples  
   - 4-rule problems: 25 samples
   - Location: `results/test_datasets/`

4. **Infrastructure Code**
   - `AlgebraInference` class with composition logic (algebra_inference.py:172-847)
   - `compose_energies()` and `compute_composed_gradient()` methods
   - Model loading utilities

### ❌ Not Yet Implemented
1. **Compositional Sampling in GaussianDiffusion1D**
   - `sample_compositional()` method
   - Composition support in `opt_step()` method
   - Energy summation in diffusion loop

2. **Monolithic Model Training**
   - No trained monolithic checkpoint exists yet
   - Need to run: `python train_algebra_monolithic.py --train_steps 200000`

3. **Evaluation Comparison Script**
   - No direct comparison of compositional vs monolithic
   - Need script to run both and generate comparison report

4. **Integration in eval_algebra.py**
   - Compositional evaluation mode (`--eval_type comparison`)
   - Proper routing to composition vs monolithic

---

## Implementation Plan

### Phase 1: Implement Compositional Sampling (4-6 hours)

**Goal:** Extend `GaussianDiffusion1D` to support energy composition during inference.

#### Task 1.1: Modify GaussianDiffusion1D Class
**File:** `src/diffusion/denoising_diffusion_pytorch_1d.py`

**Changes:**
```python
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        compositional_models=None,  # NEW: Dict[str, nn.Module]
        compositional_weights=None,  # NEW: Optional weights
        ...
    ):
        self.model = model
        self.compositional_models = compositional_models
        self.compositional_weights = compositional_weights or {}
```

#### Task 1.2: Extend opt_step() for Composition
**Location:** `src/diffusion/denoising_diffusion_pytorch_1d.py` around line 579

**Add branching logic:**
```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0):
    """Inner-loop optimization with optional composition support."""
    with torch.enable_grad():
        for i in range(step):
            # COMPOSITION PATH vs SINGLE MODEL PATH
            if self.compositional_models is not None:
                energy, grad = self._compute_composed_energy_grad(inp, img, t)
            else:
                energy, grad = self.model(inp, img, t, return_both=True)
            
            # Rest of optimization (unchanged)
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            # ... (existing code for clipping, acceptance, etc.)
```

#### Task 1.3: Add Composition Helper Methods
**Add to GaussianDiffusion1D:**
```python
def _compute_composed_energy(self, inp, img, t):
    """Sum energies from multiple models."""
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

#### Task 1.4: Add sample_compositional() API
```python
@torch.no_grad()
def sample_compositional(self, x, label, mask, models_dict, weights_dict=None, batch_size=16):
    """
    Sample using composed energy from multiple rule models.
    
    Args:
        x: Input equation embedding
        label: Target (for conditioning)
        mask: Conditioning mask
        models_dict: Dict[str, nn.Module] of rule models
        weights_dict: Optional weights per rule
        batch_size: Batch size
        
    Returns:
        Sampled output using composed energy
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

**Validation:**
```bash
# Test backward compatibility (should still work)
python tests/test_ired_inference.py --rule distribute

# Test composition (new functionality)
python -c "
from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
import torch

# Create test models
model1 = AlgebraDiffusionWrapper(AlgebraEBM(128, 128, 'distribute'))
model2 = AlgebraDiffusionWrapper(AlgebraEBM(128, 128, 'combine'))

# Create diffusion with composition
diffusion = GaussianDiffusion1D(model1, timesteps=10)

# Test compositional sampling
inp = torch.randn(1, 128)
result = diffusion.sample_compositional(
    x=inp,
    label=None,
    mask=None,
    models_dict={'distribute': model1, 'combine': model2},
    batch_size=1
)
print(f'Compositional sampling works! Result shape: {result.shape}')
"
```

---

### Phase 2: Train Monolithic Baseline (4-8 hours GPU time)

**Goal:** Train single unified model on all 4 rules for fair comparison.

#### Task 2.1: Launch Monolithic Training
```bash
python train_algebra_monolithic.py \
    --train_steps 200000 \
    --problems_per_rule 50000 \
    --batch_size 2048 \
    --timesteps 10 \
    --supervise-energy-landscape True \
    --use-contrastive-energy-loss True \
    --use-innerloop-opt True \
    --amp True \
    --fp16 True \
    --results_folder ./results/monolithic \
    --save_and_sample_every 5000
```

**Expected Duration:** 4-8 hours on GPU

**Checkpoints Generated:**
- `results/monolithic/model-5.pt` (25k steps)
- `results/monolithic/model-10.pt` (50k steps)
- `results/monolithic/model-20.pt` (100k steps)
- `results/monolithic/model.pt` (200k steps - FINAL)

**Monitoring:**
```bash
# Watch training progress
tail -f ./results/monolithic/log.txt

# Key metrics to watch:
# - Loss decreasing steadily
# - Energy gap increasing (target: >8 units)
# - No NaN/Inf values
```

#### Task 2.2: Validate Monolithic Training
```python
# Quick validation after training
python eval_algebra.py \
    --eval_type monolithic \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --num_samples 100 \
    --output_dir ./eval_results_quick

# Expected: >80% single-rule accuracy
```

---

### Phase 3: Integrate Compositional Evaluation (2-3 hours)

**Goal:** Add compositional evaluation to main evaluation script.

#### Task 3.1: Add evaluate_with_composition() Function
**File:** `src/algebra/algebra_evaluation.py`

**Location:** After line 1807 (after existing `evaluate_with_composition` stub)

**Implementation:**
```python
def evaluate_with_composition_real_diffusion(
    rule_checkpoints: Dict[str, str],
    test_dataset: Union[MultiRuleDataset, ConstrainedDataset],
    encoder: Any,
    decoder: Optional[EquationDecoder] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate compositional approach using real diffusion sampling.
    
    Args:
        rule_checkpoints: Dict mapping rule names to checkpoint paths
        test_dataset: Multi-rule test dataset
        encoder: Equation encoder
        decoder: Equation decoder
        device: Device for computation
        max_samples: Max samples to evaluate
        
    Returns:
        Evaluation results in standard format
    """
    logger.info("Evaluating compositional approach with real diffusion")
    
    # Load rule models
    rule_models = {}
    diffusion_template = None
    
    for rule_name, checkpoint_path in rule_checkpoints.items():
        logger.info(f"Loading {rule_name} model from {checkpoint_path}")
        diffusion, ebm = load_diffusion_model_for_inference(checkpoint_path, device)
        rule_models[rule_name] = ebm
        
        # Use first model as diffusion template
        if diffusion_template is None:
            diffusion_template = diffusion
    
    # Limit samples
    num_samples = min(len(test_dataset), max_samples) if max_samples else len(test_dataset)
    
    # Storage for results
    results = []
    predicted_embeddings = []
    target_embeddings = []
    predicted_equations = []
    target_equations = []
    
    logger.info(f"Evaluating {num_samples} samples with compositional sampling...")
    start_time = time.time()
    
    for idx in range(num_samples):
        try:
            # Get data
            sample = test_dataset[idx]
            inp = sample[0].unsqueeze(0).to(device)
            target = sample[1].unsqueeze(0).to(device)
            
            # Get problem info
            problem_info = test_dataset.get_problem_info(idx)
            input_eq_str = problem_info['input_equation']
            target_eq_str = problem_info['target_equation']
            rules_applied = problem_info['rules_applied']
            
            # Get active rule models for this problem
            active_models = {rule: rule_models[rule] for rule in rules_applied if rule in rule_models}
            
            # Run compositional sampling
            with torch.no_grad():
                pred_embedding = diffusion_template.sample_compositional(
                    x=inp,
                    label=target,
                    mask=None,
                    models_dict=active_models,
                    batch_size=1
                )
            
            # Compute distance
            final_dist = (pred_embedding - target).norm().item()
            
            # Decode if decoder available
            pred_eq_str = None
            if decoder is not None:
                pred_eq_str, decoding_distance = decoder.decode_embedding(pred_embedding.squeeze(0).cpu())
            
            result = {
                'index': idx,
                'input_equation': input_eq_str,
                'target_equation': target_eq_str,
                'predicted_equation': pred_eq_str,
                'rules_applied': rules_applied,
                'final_distance': final_dist,
                'success': final_dist < 2.0  # Standard threshold
            }
            results.append(result)
            
            predicted_embeddings.append(pred_embedding.detach().cpu())
            target_embeddings.append(target.detach().cpu())
            predicted_equations.append(pred_eq_str)
            target_equations.append(target_eq_str)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{num_samples}...")
                
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            results.append({'index': idx, 'error': str(e), 'success': False})
    
    eval_time = time.time() - start_time
    
    # Compute metrics
    valid_results = [r for r in results if 'error' not in r]
    
    # Symbolic equivalence
    symbolic_results = {'symbolic_equivalence_rate': 0.0}
    if any(eq is not None for eq in predicted_equations):
        valid_pred = [p for p in predicted_equations if p is not None]
        valid_target = [t for i, t in enumerate(target_equations) if predicted_equations[i] is not None]
        if valid_pred:
            symbolic_results = compute_symbolic_equivalence(valid_pred, valid_target)
    
    # Embedding distances
    embedding_results = {'mean_l2_distance': 0.0}
    if predicted_embeddings and target_embeddings:
        pred_tensor = torch.cat(predicted_embeddings, dim=0)
        target_tensor = torch.cat(target_embeddings, dim=0)
        embedding_results = compute_embedding_distances(pred_tensor, target_tensor)
    
    # Invalid rate
    validity_results = compute_invalid_rate(predicted_equations)
    
    # Build results
    evaluation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'compositional_real_diffusion',
        'dataset_type': type(test_dataset).__name__,
        'num_samples_evaluated': num_samples,
        'num_valid_results': len(valid_results),
        'evaluation_time_seconds': eval_time,
        'rule_models_used': list(rule_models.keys()),
        'symbolic_equivalence': symbolic_results,
        'embedding_distances': embedding_results,
        'validity': validity_results,
        'summary': {
            'accuracy': symbolic_results['symbolic_equivalence_rate'],
            'invalid_rate': validity_results['invalid_rate'],
            'mean_l2_distance': embedding_results['mean_l2_distance'],
            'total_evaluated': num_samples
        },
        'individual_results': results
    }
    
    logger.info(f"Compositional evaluation complete: {symbolic_results['symbolic_equivalence_rate']:.1%} accuracy")
    
    return evaluation_results
```

#### Task 3.2: Update eval_algebra.py for Comparison Mode
**File:** `eval_algebra.py`

**Add comparison option:**
```python
# In main() function, add elif branch:
elif args.eval_type == 'comparison':
    # Run both monolithic and compositional evaluations
    logger.info("="*60)
    logger.info("RUNNING COMPARISON: Monolithic vs Compositional")
    logger.info("="*60)
    
    results = {}
    
    # 1. Monolithic evaluation
    logger.info("\n[1/2] Monolithic Evaluation")
    mono_results = run_monolithic_evaluation(
        monolithic_checkpoint=args.monolithic_checkpoint,
        output_dir=args.output_dir,
        num_samples=args.max_samples if args.max_samples else 1000
    )
    results['monolithic'] = mono_results
    
    # 2. Compositional evaluation
    logger.info("\n[2/2] Compositional Evaluation")
    
    # Load rule checkpoints
    rule_checkpoints = {}
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        checkpoint_path = Path(args.model_dir) / rule / 'model.pt'
        if checkpoint_path.exists():
            rule_checkpoints[rule] = str(checkpoint_path)
    
    comp_results = {}
    
    # Test on multi-rule datasets
    for num_rules in [2, 3, 4]:
        test_dataset = MultiRuleDataset(
            num_rules=num_rules,
            split='test',
            num_problems=args.max_samples if args.max_samples else 1000,
            d_model=128
        )
        
        result = evaluate_with_composition_real_diffusion(
            rule_checkpoints=rule_checkpoints,
            test_dataset=test_dataset,
            encoder=encoder,
            decoder=decoder,
            max_samples=args.max_samples
        )
        
        comp_results[f'multi_rule_{num_rules}'] = result
    
    results['compositional'] = comp_results
    
    # Generate comparison report
    generate_comparison_report(results, args.output_dir)
```

**Add to argument parser:**
```python
parser.add_argument(
    '--eval_type',
    type=str,
    default='single_rule',
    choices=['single_rule', 'multi_rule', 'monolithic', 'comparison', 'full'],
    help='Evaluation type'
)
```

---

### Phase 4: Comparison and Reporting (2-3 hours)

**Goal:** Generate comprehensive comparison report.

#### Task 4.1: Create Comparison Script
**File:** `scripts/compare_monolithic_vs_compositional.py`

**Create new script:**
```python
#!/usr/bin/env python3
"""
Compare Monolithic vs Compositional IRED

Runs systematic comparison and generates markdown report.

Usage:
    python scripts/compare_monolithic_vs_compositional.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results \
        --num_samples 1000 \
        --output_dir ./comparison_results
"""

import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monolithic_checkpoint', required=True)
    parser.add_argument('--compositional_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', default='./comparison_results')
    args = parser.parse_args()
    
    # Run comparison evaluation
    os.system(f"""
    python eval_algebra.py \
        --eval_type comparison \
        --monolithic_checkpoint {args.monolithic_checkpoint} \
        --model_dir {args.compositional_dir} \
        --max_samples {args.num_samples} \
        --output_dir {args.output_dir}
    """)
    
    print(f"\nComparison complete! Results in: {args.output_dir}")

if __name__ == '__main__':
    main()
```

#### Task 4.2: Generate Comparison Report Function
**Add to eval_algebra.py:**
```python
def generate_comparison_report(results: Dict, output_dir: str):
    """Generate markdown comparison report."""
    
    mono = results['monolithic']
    comp = results['compositional']
    
    # Calculate averages
    mono_single_avg = np.mean([
        mono[f'single_rule_{rule}']['summary']['accuracy']
        for rule in ['distribute', 'combine', 'isolate', 'divide']
        if f'single_rule_{rule}' in mono
    ])
    
    mono_multi_avg = np.mean([
        mono[f'multi_rule_{n}']['summary']['accuracy']
        for n in [2, 3, 4]
        if f'multi_rule_{n}' in mono
    ])
    
    comp_multi_avg = np.mean([
        comp[f'multi_rule_{n}']['summary']['accuracy']
        for n in [2, 3, 4]
        if f'multi_rule_{n}' in comp
    ])
    
    advantage = (comp_multi_avg - mono_multi_avg) * 100
    
    # Generate report
    report = f"""# Monolithic vs Compositional Comparison

## Overall Results

| Model | Single-Rule Acc | Multi-Rule Acc | Advantage |
|-------|----------------|----------------|-----------|
| Monolithic | {mono_single_avg:.1%} | {mono_multi_avg:.1%} | Baseline |
| **Compositional** | **~{mono_single_avg:.1%}*** | **{comp_multi_avg:.1%}** | **+{advantage:.1f}%** 🎯 |

*Compositional uses rule-specific models for single-rule

## Multi-Rule Breakdown

| Rules | Monolithic | Compositional | Advantage |
|-------|-----------|--------------|-----------|
"""
    
    for n in [2, 3, 4]:
        mono_acc = mono.get(f'multi_rule_{n}', {}).get('summary', {}).get('accuracy', 0)
        comp_acc = comp.get(f'multi_rule_{n}', {}).get('summary', {}).get('accuracy', 0)
        adv = (comp_acc - mono_acc) * 100
        report += f"| {n}-rule | {mono_acc:.1%} | {comp_acc:.1%} | **+{adv:.1f}%** |\n"
    
    report += f"""
## Interpretation

{'✅ **Compositional Advantage CONFIRMED!**' if advantage > 20 else '⚠️ **Partial Advantage**'}

"""
    
    if advantage > 20:
        report += """
The compositional approach demonstrates clear superiority:
- Achieves {advantage:.0f}% absolute improvement on multi-rule problems
- Successfully demonstrates zero-shot compositional reasoning
- Validates modular energy function hypothesis
"""
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/comparison_report.md', 'w') as f:
        f.write(report)
    
    # Save JSON
    with open(f'{output_dir}/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(report)
```

---

### Phase 5: Testing and Validation (2-3 hours)

**Goal:** Verify all components work correctly.

#### Task 5.1: Unit Tests
```bash
# Test compositional sampling
python tests/test_compositional_sampling.py

# Test monolithic evaluation
python tests/test_monolithic_evaluation.py

# Test backward compatibility
python tests/test_ired_inference.py
```

#### Task 5.2: Integration Test
```bash
# Small-scale end-to-end test
python eval_algebra.py \
    --eval_type comparison \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --model_dir ./results \
    --max_samples 10 \
    --output_dir ./test_comparison
```

#### Task 5.3: Full Comparison Run
```bash
# Full evaluation (1000 samples each)
python scripts/compare_monolithic_vs_compositional.py \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --compositional_dir ./results \
    --num_samples 1000 \
    --output_dir ./final_comparison_results

# View report
cat ./final_comparison_results/comparison_report.md
```

---

## Expected Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| **Phase 1** | Implement compositional sampling | 4-6 hours | None |
| **Phase 2** | Train monolithic model | 4-8 hours | Phase 1 (optional) |
| **Phase 3** | Integration | 2-3 hours | Phase 1 |
| **Phase 4** | Reporting | 2-3 hours | Phase 3 |
| **Phase 5** | Testing | 2-3 hours | All above |

**Total Time:** 14-23 hours (~2-3 days)

**Critical Path:** Phase 1 → Phase 3 → Phase 5 (can run Phase 2 in parallel)

---

## Success Criteria

### Minimum Success ✅
- Compositional sampling works without errors
- Monolithic model achieves >80% single-rule accuracy
- Compositional outperforms monolithic by >15 points on multi-rule

### Full Success 🎯
- Compositional sampling integrates seamlessly with IRED
- Monolithic achieves ~90% single-rule accuracy
- Compositional outperforms monolithic by >25 points on multi-rule
- Results match proposal expectations (50-60% multi-rule)

### Exceptional Success 🚀
- Compositional achieves >60% on 2-rule problems
- Clean, reproducible comparison pipeline
- Comprehensive analysis and visualization
- Publication-ready results

---

## Risk Mitigation

### Risk 1: Compositional sampling has bugs
**Mitigation:** 
- Test backward compatibility first
- Start with 2-model composition
- Add extensive logging

### Risk 2: Monolithic training fails
**Mitigation:**
- Monitor training closely
- Use proven hyperparameters
- Save frequent checkpoints

### Risk 3: Performance gap is small
**Mitigation:**
- Verify both implementations are correct
- Check test dataset diversity
- Review energy landscapes

---

## Next Steps

### Immediate (Today)
1. Implement Phase 1 (compositional sampling)
2. Start Phase 2 (monolithic training) in parallel
3. Create git branch: `feature/compositionality-evaluation`

### This Week
1. Complete Phase 3 (integration)
2. Run Phase 5 (testing)
3. Generate Phase 4 (reports)

### Next Week
1. Analyze results
2. Write up findings
3. Update research documentation

---

## References

### Key Files
- `src/diffusion/denoising_diffusion_pytorch_1d.py` - IRED diffusion (lines 288-688)
- `src/algebra/algebra_evaluation.py` - Evaluation functions
- `eval_algebra.py` - Main evaluation script
- `train_algebra_monolithic.py` - Monolithic training
- `documentation/reports/compositionality-implementation-plan-2025-12-09.md` - Detailed design
- `documentation/reports/monolithic-ired-baseline-plan-2025-12-09.md` - Monolithic design

### Existing Test Datasets
- `results/test_datasets/2_rule_test_dataset.json` - 100 samples
- `results/test_datasets/3_rule_test_dataset.json` - 50 samples
- `results/test_datasets/4_rule_test_dataset.json` - 25 samples

---

**Status:** Ready for implementation  
**Priority:** High (core research contribution)  
**Complexity:** Medium (well-scoped, building on proven infrastructure)
