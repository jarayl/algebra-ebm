# Monolithic IRED Baseline Implementation Plan
**Date:** 2025-12-09  
**Research Type:** Deep research on monolithic baseline for composition comparison  
**Status:** Comprehensive implementation strategy

---

## Executive Summary

To validate the compositionality hypothesis, we need a **monolithic IRED baseline** - a single unified EBM trained on all 4 rules combined. This baseline will demonstrate that **composition is superior** to a single large model by showing the expected performance gap:

- **Monolithic:** ~90% single-rule, ~20-30% multi-rule  
- **Compositional:** ~85% single-rule, ~50-60% multi-rule

**Key Finding:** The monolithic baseline should be **straightforward to implement** by creating a combined dataset and reusing the existing training infrastructure. The challenge is in **fair comparison** and **systematic evaluation**.

---

## 1. Research Scope

### Original Question
Create a monolithic IRED baseline that:
1. Trains on all 4 rules simultaneously (200k problems total)
2. Uses the same architecture as rule-specific models
3. Can be fairly compared against compositional approach
4. Demonstrates compositional advantage on multi-rule problems

### Sub-Questions Investigated
1. How do we combine all 4 rule datasets into one?
2. What training modifications are needed?
3. How do we ensure fair comparison (same data, same compute)?
4. What evaluation metrics demonstrate compositional advantage?
5. How do we systematically compare performance?

### Files/Systems Analyzed
- `train_algebra.py` - Single-rule training script (850 lines)
- `src/algebra/algebra_dataset.py` - Dataset classes (2700+ lines)
- `src/algebra/algebra_models.py` - EBM architecture (200+ lines)
- `eval_algebra.py` - Evaluation infrastructure (600+ lines)
- `documentation/implementation_plan.md` - Step 17: Monolithic baseline (lines 512-523)

---

## 2. Key Findings

### Finding 1: Combined Dataset Strategy

**Evidence:**
- `src/algebra/algebra_dataset.py:26-48` - AlgebraDataset class structure
- `train_algebra.py:508-556` - Dataset creation logic
- `documentation/implementation_plan.md:520` - "Dataset combines all 4 rule types: 200k problems total"

**Analysis:**

The most straightforward approach is to create a **`CombinedAlgebraDataset`** class that:
1. Generates 50k problems per rule (50k × 4 = 200k total)
2. Shuffles problems from all rules together
3. Provides uniform sampling during training

**Implementation:**
```python
class CombinedAlgebraDataset(data.Dataset):
    """
    Combined dataset for monolithic IRED baseline.
    
    Generates problems from all 4 rules uniformly:
    - 50k distribute problems
    - 50k combine problems  
    - 50k isolate problems
    - 50k divide problems
    Total: 200k problems (same as 4x rule-specific training)
    
    This ensures fair comparison with compositional approach.
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    
    def __init__(
        self,
        split: str = 'train',
        problems_per_rule: int = 50000,
        d_model: int = 128,
        **kwargs  # Pass through variability parameters
    ):
        super().__init__()
        
        self.split = split
        self.problems_per_rule = problems_per_rule
        self.d_model = d_model
        
        # Dataset interface requirements
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder (shared across all rules)
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Generate problems from all 4 rules
        self.equation_pairs = []
        self.rule_labels = []  # Track which rule each problem came from
        
        for rule in self.VALID_RULES:
            print(f"Generating {problems_per_rule} {rule} problems...")
            
            # Create temporary single-rule dataset
            rule_dataset = AlgebraDataset(
                rule=rule,
                split=split,
                num_problems=problems_per_rule,
                d_model=d_model,
                **kwargs
            )
            
            # Collect equation pairs
            self.equation_pairs.extend(rule_dataset.equation_pairs)
            self.rule_labels.extend([rule] * len(rule_dataset.equation_pairs))
        
        # Shuffle to mix rules uniformly
        combined = list(zip(self.equation_pairs, self.rule_labels))
        random.shuffle(combined)
        self.equation_pairs, self.rule_labels = zip(*combined)
        
        print(f"Combined dataset: {len(self.equation_pairs)} total problems")
        print(f"  Per-rule breakdown: {self._count_per_rule()}")
    
    def _count_per_rule(self) -> Dict[str, int]:
        """Count how many problems from each rule."""
        counts = defaultdict(int)
        for rule in self.rule_labels:
            counts[rule] += 1
        return dict(counts)
    
    def __len__(self) -> int:
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded equation pair at index."""
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode both equations
        inp_emb = self.encoder(input_eq)
        target_emb = self.encoder(target_eq)
        
        return inp_emb, target_emb
```

**Confidence:** High - This is a straightforward extension of existing dataset infrastructure.

---

### Finding 2: Training Script Modifications

**Evidence:**
- `train_algebra.py:454-850` - Main training function
- `train_algebra.py:508-556` - Dataset creation
- `train_algebra.py:573-583` - Model initialization

**Analysis:**

Create **`train_algebra_monolithic.py`** by:
1. Copying `train_algebra.py` as template
2. Replacing single-rule dataset with `CombinedAlgebraDataset`
3. Removing `--rule` argument (trains on all rules)
4. Updating results folder to `results/monolithic/`

**Key Differences:**
```python
# ORIGINAL (train_algebra.py)
parser.add_argument(
    '--rule', 
    type=str,
    required=True,  # ❌ Required
    choices=['distribute', 'combine', 'isolate', 'divide'],
    help='Algebraic rule to train'
)

# MONOLITHIC (train_algebra_monolithic.py)
# No --rule argument needed!

# Dataset creation:
# ORIGINAL
dataset = AlgebraDataset(
    rule=args.rule,  # ❌ Single rule
    split=args.split,
    num_problems=args.num_problems,
    d_model=args.d_model
)

# MONOLITHIC  
dataset = CombinedAlgebraDataset(
    split=args.split,
    problems_per_rule=args.num_problems,  # ✅ 50k per rule
    d_model=args.d_model
)

# Model initialization:
# ORIGINAL
ebm = AlgebraEBM(
    inp_dim=dataset.inp_dim,
    out_dim=dataset.out_dim,
    rule_name=args.rule  # ❌ Rule-specific name
)

# MONOLITHIC
ebm = AlgebraEBM(
    inp_dim=dataset.inp_dim,
    out_dim=dataset.out_dim,
    rule_name='monolithic'  # ✅ Generic name
)

# Results folder:
# ORIGINAL: './results/{rule_name}'
# MONOLITHIC: './results/monolithic'
```

**Confidence:** High - Minimal changes to proven training infrastructure.

---

### Finding 3: Fair Comparison Requirements

**Evidence:**
- `documentation/implementation_plan.md:520` - "Same architecture as AlgebraEBM"
- `train_algebra.py:239-241` - `num_problems=50000` default
- `src/algebra/algebra_models.py:41-88` - AlgebraEBM architecture

**Analysis:**

For **fair comparison**, monolithic and compositional must have:

| Criterion | Compositional (4 models) | Monolithic (1 model) | Fair? |
|-----------|--------------------------|----------------------|-------|
| **Total training data** | 4 × 50k = 200k | 4 × 50k = 200k | ✅ Equal |
| **Architecture** | AlgebraEBM (512 hidden) | AlgebraEBM (512 hidden) | ✅ Equal |
| **Training steps** | 50k per model | 50k total | ⚠️ **Must decide** |
| **Total compute** | 4 × 50k = 200k steps | 50k steps | ❌ **Imbalance!** |
| **Parameters per model** | ~800k each | ~800k | ✅ Equal arch |
| **Total parameters** | 4 × 800k = 3.2M | 800k | ⚠️ **4x difference** |

**Critical Decision: Training Steps**

**Option A: Equal steps per model (Recommended)**
```bash
# Compositional: 50k steps × 4 models = 200k total steps
python train_algebra.py --rule distribute --train_steps 50000
python train_algebra.py --rule combine --train_steps 50000
python train_algebra.py --rule isolate --train_steps 50000
python train_algebra.py --rule divide --train_steps 50000

# Monolithic: 200k steps to match total compute
python train_algebra_monolithic.py --train_steps 200000
```

**Rationale:** Monolithic sees 4x more diverse data, so needs 4x more steps for fair comparison.

**Option B: Equal steps total (Alternative)**
```bash
# Compositional: 50k steps × 4 models
# Monolithic: 50k steps × 1 model
python train_algebra_monolithic.py --train_steps 50000
```

**Rationale:** Each model gets same number of gradient updates.

**Recommendation:** **Option A** - Monolithic gets 200k steps to match total compute budget. This is fairer because monolithic sees more diverse data per step.

**Confidence:** Medium - This is a design choice, but Option A is more principled.

---

### Finding 4: Evaluation Strategy

**Evidence:**
- `eval_algebra.py:58-101` - Single-rule dataset creation
- `eval_algebra.py:104-146` - Multi-rule dataset creation  
- `src/algebra/algebra_evaluation.py:52-99` - Evaluation infrastructure
- `documentation/implementation_plan.md:429-439` - Expected results

**Analysis:**

We need **two types of evaluations**:

#### 4.1 Single-Rule Evaluation

**Test on each rule separately:**
```python
def evaluate_monolithic_single_rule(
    monolithic_model: AlgebraDiffusionWrapper,
    rule: str,
    num_samples: int = 1000
) -> Dict:
    """
    Evaluate monolithic model on single-rule problems.
    
    This tests if monolithic can match rule-specific performance.
    Expected: ~90% accuracy (should be good at individual rules)
    """
    
    # Create test dataset for this specific rule
    test_dataset = AlgebraDataset(
        rule=rule,
        split='test',
        num_problems=num_samples,
        d_model=128
    )
    
    # Use REAL diffusion sampling (proven to work)
    results = evaluate_with_real_diffusion(
        model=monolithic_model,
        test_dataset=test_dataset,
        diffusion_template=diffusion  # GaussianDiffusion1D instance
    )
    
    return {
        'rule': rule,
        'accuracy': results['accuracy'],
        'mean_distance': results['mean_distance'],
        'samples': num_samples
    }
```

**Run for all 4 rules:**
```python
single_rule_results = {}
for rule in ['distribute', 'combine', 'isolate', 'divide']:
    results = evaluate_monolithic_single_rule(
        monolithic_model, 
        rule=rule,
        num_samples=1000
    )
    single_rule_results[rule] = results
    print(f"{rule}: {results['accuracy']:.1%}")
```

#### 4.2 Multi-Rule Evaluation

**Test on compositional problems:**
```python
def evaluate_monolithic_multi_rule(
    monolithic_model: AlgebraDiffusionWrapper,
    num_rules: int,
    num_samples: int = 1000
) -> Dict:
    """
    Evaluate monolithic model on multi-rule problems.
    
    This is where monolithic should FAIL compared to composition.
    Expected: ~20-30% accuracy (poor generalization)
    """
    
    # Create multi-rule test dataset
    test_dataset = MultiRuleDataset(
        num_rules=num_rules,
        split='test',
        num_problems=num_samples,
        d_model=128
    )
    
    # Use REAL diffusion sampling
    results = evaluate_with_real_diffusion(
        model=monolithic_model,
        test_dataset=test_dataset,
        diffusion_template=diffusion
    )
    
    return {
        'num_rules': num_rules,
        'accuracy': results['accuracy'],
        'mean_distance': results['mean_distance'],
        'samples': num_samples
    }
```

**Run for 2, 3, 4 rule problems:**
```python
multi_rule_results = {}
for num_rules in [2, 3, 4]:
    results = evaluate_monolithic_multi_rule(
        monolithic_model,
        num_rules=num_rules,
        num_samples=1000
    )
    multi_rule_results[num_rules] = results
    print(f"{num_rules}-rule: {results['accuracy']:.1%}")
```

**Confidence:** High - Evaluation follows proven patterns from existing code.

---

### Finding 5: Comparison Framework

**Evidence:**
- `documentation/implementation_plan.md:718-729` - Expected results table
- `scripts/compositional_analysis.py:61-79` - Report generation

**Analysis:**

Create **comprehensive comparison script** `compare_monolithic_vs_compositional.py`:

```python
#!/usr/bin/env python3
"""
Comparison Script: Monolithic vs Compositional IRED

Systematically compares performance of:
1. Monolithic: Single EBM trained on all 4 rules
2. Compositional: 4 separate EBMs composed at inference

Expected Results (from proposal):
- Single-Rule: Monolithic ~90%, Compositional ~85%  
- Multi-Rule: Monolithic ~20-30%, Compositional ~50-60%

Key Insight: Monolithic is slightly better on single rules,
but MUCH worse on multi-rule (composition advantage!)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

def load_monolithic_model(checkpoint_path: str):
    """Load trained monolithic model."""
    diffusion, ebm = load_diffusion_model_for_inference(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return diffusion, ebm

def load_compositional_models(results_dir: str):
    """Load 4 rule-specific models for composition."""
    models = {}
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        checkpoint_path = f"{results_dir}/{rule}/model.pt"
        diffusion, ebm = load_diffusion_model_for_inference(checkpoint_path)
        models[rule] = {'diffusion': diffusion, 'ebm': ebm}
    return models

def evaluate_single_rule(
    model_name: str,
    model,
    rule: str,
    num_samples: int = 1000
) -> Dict:
    """Evaluate on single-rule problems."""
    print(f"\n[{model_name}] Evaluating {rule} (single-rule)...")
    
    test_dataset = AlgebraDataset(
        rule=rule,
        split='test',
        num_problems=num_samples,
        d_model=128
    )
    
    results = evaluate_with_real_diffusion(
        model=model,
        test_dataset=test_dataset
    )
    
    return {
        'model': model_name,
        'rule': rule,
        'accuracy': results['accuracy'],
        'mean_distance': results['mean_distance']
    }

def evaluate_multi_rule(
    model_name: str,
    model,  # Can be single model or dict of models
    num_rules: int,
    num_samples: int = 1000,
    is_compositional: bool = False
) -> Dict:
    """Evaluate on multi-rule problems."""
    print(f"\n[{model_name}] Evaluating {num_rules}-rule (multi-rule)...")
    
    test_dataset = MultiRuleDataset(
        num_rules=num_rules,
        split='test',
        num_problems=num_samples,
        d_model=128
    )
    
    if is_compositional:
        # Use compositional sampling with multiple models
        results = evaluate_with_composition(
            rule_models_dict=model,
            test_dataset=test_dataset
        )
    else:
        # Use single monolithic model
        results = evaluate_with_real_diffusion(
            model=model,
            test_dataset=test_dataset
        )
    
    return {
        'model': model_name,
        'num_rules': num_rules,
        'accuracy': results['accuracy'],
        'mean_distance': results['mean_distance']
    }

def generate_comparison_table(all_results: Dict) -> str:
    """Generate comparison table matching proposal format."""
    
    mono_single = all_results['monolithic']['single_rule']
    comp_single = all_results['compositional']['single_rule']
    mono_multi = all_results['monolithic']['multi_rule']
    comp_multi = all_results['compositional']['multi_rule']
    
    # Average single-rule accuracies
    mono_single_avg = sum(r['accuracy'] for r in mono_single.values()) / 4
    comp_single_avg = sum(r['accuracy'] for r in comp_single.values()) / 4
    
    # Average multi-rule accuracies (2, 3, 4 rules)
    mono_multi_avg = sum(r['accuracy'] for r in mono_multi.values()) / 3
    comp_multi_avg = sum(r['accuracy'] for r in comp_multi.values()) / 3
    
    table = f"""
# Monolithic vs Compositional IRED Comparison

## Overall Results

| Model               | Single-Rule Acc | Multi-Rule Acc | Advantage |
|---------------------|----------------|----------------|-----------|
| Monolithic IRED     | {mono_single_avg:.1%}           | {mono_multi_avg:.1%}           | {(mono_multi_avg - comp_multi_avg)*100:+.1f}% |
| **Compositional**   | **{comp_single_avg:.1%}**       | **{comp_multi_avg:.1%}**       | **Baseline** |

**Key Insight:** Compositional achieves {(comp_multi_avg - mono_multi_avg) / mono_multi_avg * 100:+.0f}% relative improvement on multi-rule problems!

## Single-Rule Breakdown

| Rule       | Monolithic | Compositional | Difference |
|------------|-----------|--------------|------------|
| Distribute | {mono_single['distribute']['accuracy']:.1%}       | {comp_single['distribute']['accuracy']:.1%}          | {(mono_single['distribute']['accuracy'] - comp_single['distribute']['accuracy'])*100:+.1f}% |
| Combine    | {mono_single['combine']['accuracy']:.1%}       | {comp_single['combine']['accuracy']:.1%}          | {(mono_single['combine']['accuracy'] - comp_single['combine']['accuracy'])*100:+.1f}% |
| Isolate    | {mono_single['isolate']['accuracy']:.1%}       | {comp_single['isolate']['accuracy']:.1%}          | {(mono_single['isolate']['accuracy'] - comp_single['isolate']['accuracy'])*100:+.1f}% |
| Divide     | {mono_single['divide']['accuracy']:.1%}       | {comp_single['divide']['accuracy']:.1%}          | {(mono_single['divide']['accuracy'] - comp_single['divide']['accuracy'])*100:+.1f}% |

## Multi-Rule Breakdown

| Rules | Monolithic | Compositional | Compositional Advantage |
|-------|-----------|--------------|------------------------|
| 2-rule | {mono_multi[2]['accuracy']:.1%}       | {comp_multi[2]['accuracy']:.1%}          | **{(comp_multi[2]['accuracy'] - mono_multi[2]['accuracy'])*100:+.1f}%** 🎯 |
| 3-rule | {mono_multi[3]['accuracy']:.1%}       | {comp_multi[3]['accuracy']:.1%}          | **{(comp_multi[3]['accuracy'] - mono_multi[3]['accuracy'])*100:+.1f}%** 🎯 |
| 4-rule | {mono_multi[4]['accuracy']:.1%}       | {comp_multi[4]['accuracy']:.1%}          | **{(comp_multi[4]['accuracy'] - mono_multi[4]['accuracy'])*100:+.1f}%** 🎯 |

## Statistical Summary

**Single-Rule Performance:**
- Monolithic is {abs((mono_single_avg - comp_single_avg) / comp_single_avg * 100):.1f}% {'better' if mono_single_avg > comp_single_avg else 'worse'} than compositional
- Expected: ~5% better (monolithic trains on all rules together)

**Multi-Rule Performance (KEY RESULT):**
- Compositional is {(comp_multi_avg - mono_multi_avg) / mono_multi_avg * 100:+.1f}% better than monolithic
- Expected: ~100-150% better (composition enables zero-shot generalization)
- **Result:** {'✅ Hypothesis CONFIRMED' if comp_multi_avg > mono_multi_avg * 1.5 else '⚠️ Hypothesis PARTIAL' if comp_multi_avg > mono_multi_avg else '❌ Hypothesis FAILED'}

## Interpretation

{interpret_results(mono_single_avg, comp_single_avg, mono_multi_avg, comp_multi_avg)}
"""
    
    return table

def interpret_results(mono_single, comp_single, mono_multi, comp_multi):
    """Provide interpretation of results."""
    
    if comp_multi > mono_multi * 1.5:
        return """
✅ **Compositional Advantage CONFIRMED!**

The compositional approach demonstrates clear superiority on multi-rule problems:
1. Maintains competitive single-rule performance (~5% difference)
2. Achieves >50% relative improvement on multi-rule generalization
3. Successfully demonstrates zero-shot compositional reasoning

This validates the core hypothesis: modular energy functions can be composed
at inference time to solve novel multi-rule problems without retraining.
"""
    elif comp_multi > mono_multi:
        return """
⚠️ **Compositional Advantage PARTIAL**

The compositional approach shows some improvement but less than expected:
1. Single-rule performance is comparable
2. Multi-rule improvement exists but is modest (<50% relative)
3. May indicate implementation issues or insufficient training

Recommendations:
- Verify compositional inference is using proper energy summation
- Check that monolithic model is properly trained
- Consider increasing training steps or model capacity
"""
    else:
        return """
❌ **Compositional Advantage NOT OBSERVED**

The compositional approach does not outperform monolithic on multi-rule:
1. This suggests a fundamental issue with the implementation
2. Possible causes:
   - Compositional inference has bugs (energy summation, scaling)
   - Training issues (flat energy landscapes, poor contrastive loss)
   - Evaluation bugs (decoder mismatch, threshold issues)

CRITICAL: Debug compositional inference and training before drawing conclusions.
"""

def main():
    parser = argparse.ArgumentParser(description='Compare Monolithic vs Compositional')
    parser.add_argument('--monolithic_checkpoint', type=str, required=True,
                        help='Path to monolithic model checkpoint')
    parser.add_argument('--compositional_dir', type=str, required=True,
                        help='Directory containing 4 rule-specific models')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples per evaluation')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Monolithic vs Compositional IRED Comparison")
    print("=" * 60)
    
    # Load models
    print("\n[Setup] Loading models...")
    monolithic_diffusion, monolithic_ebm = load_monolithic_model(args.monolithic_checkpoint)
    compositional_models = load_compositional_models(args.compositional_dir)
    
    all_results = {
        'monolithic': {'single_rule': {}, 'multi_rule': {}},
        'compositional': {'single_rule': {}, 'multi_rule': {}}
    }
    
    # ===== SINGLE-RULE EVALUATION =====
    print("\n" + "=" * 60)
    print("SINGLE-RULE EVALUATION")
    print("=" * 60)
    
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        # Monolithic
        mono_result = evaluate_single_rule(
            'Monolithic',
            monolithic_diffusion,
            rule,
            args.num_samples
        )
        all_results['monolithic']['single_rule'][rule] = mono_result
        
        # Compositional (use rule-specific model)
        comp_result = evaluate_single_rule(
            'Compositional',
            compositional_models[rule]['diffusion'],
            rule,
            args.num_samples
        )
        all_results['compositional']['single_rule'][rule] = comp_result
        
        print(f"\n{rule.upper()}")
        print(f"  Monolithic:    {mono_result['accuracy']:.1%}")
        print(f"  Compositional: {comp_result['accuracy']:.1%}")
        print(f"  Difference:    {(mono_result['accuracy'] - comp_result['accuracy'])*100:+.1f}%")
    
    # ===== MULTI-RULE EVALUATION =====
    print("\n" + "=" * 60)
    print("MULTI-RULE EVALUATION (KEY COMPARISON)")
    print("=" * 60)
    
    for num_rules in [2, 3, 4]:
        # Monolithic
        mono_result = evaluate_multi_rule(
            'Monolithic',
            monolithic_diffusion,
            num_rules,
            args.num_samples,
            is_compositional=False
        )
        all_results['monolithic']['multi_rule'][num_rules] = mono_result
        
        # Compositional
        comp_result = evaluate_multi_rule(
            'Compositional',
            compositional_models,
            num_rules,
            args.num_samples,
            is_compositional=True
        )
        all_results['compositional']['multi_rule'][num_rules] = comp_result
        
        advantage = (comp_result['accuracy'] - mono_result['accuracy']) * 100
        print(f"\n{num_rules}-RULE")
        print(f"  Monolithic:    {mono_result['accuracy']:.1%}")
        print(f"  Compositional: {comp_result['accuracy']:.1%}")
        print(f"  Advantage:     {advantage:+.1f}% {'🎯' if advantage > 20 else '⚠️' if advantage > 0 else '❌'}")
    
    # ===== GENERATE REPORT =====
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw results
    results_path = f"{args.output_dir}/comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Raw results saved to: {results_path}")
    
    # Generate markdown report
    report = generate_comparison_table(all_results)
    report_path = f"{args.output_dir}/comparison_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Comparison report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"\nView results:")
    print(f"  cat {report_path}")

if __name__ == '__main__':
    main()
```

**Confidence:** High - This provides systematic, reproducible comparison.

---

## 3. Implementation Plan

### Phase 1: Dataset Creation (1-2 hours)

**File to Create:** `src/algebra/algebra_dataset.py` (add new class)

**Task 1.1: Implement `CombinedAlgebraDataset`**
```python
class CombinedAlgebraDataset(data.Dataset):
    """
    Combined dataset for monolithic baseline.
    Generates 50k problems per rule (200k total).
    """
    # Implementation from Finding 1
```

**Validation:**
```python
# Test dataset creation
dataset = CombinedAlgebraDataset(
    split='train',
    problems_per_rule=1000,  # Small test
    d_model=128
)

assert len(dataset) == 4000  # 1000 × 4 rules
assert dataset.inp_dim == 128
assert dataset.out_dim == 128

# Check rule distribution
counts = dataset._count_per_rule()
for rule in ['distribute', 'combine', 'isolate', 'divide']:
    assert counts[rule] == 1000
```

---

### Phase 2: Training Script (1 hour)

**File to Create:** `train_algebra_monolithic.py`

**Task 2.1: Copy and modify training script**
```bash
# Copy as template
cp train_algebra.py train_algebra_monolithic.py

# Key modifications:
# 1. Remove --rule argument
# 2. Change dataset to CombinedAlgebraDataset
# 3. Update model rule_name to 'monolithic'
# 4. Update results folder to './results/monolithic'
```

**Task 2.2: Update argument parser**
```python
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Monolithic Algebra EBM (All Rules Combined)'
    )
    
    # Remove --rule argument (not needed!)
    # Add clarifying help text
    parser.add_argument(
        '--train_steps',
        type=int,
        default=200000,  # 4x single-rule for fair comparison
        help='Training steps (default 200k = 4x50k for fair comparison)'
    )
    
    parser.add_argument(
        '--problems_per_rule',
        type=int,
        default=50000,
        help='Problems per rule (4 rules × 50k = 200k total)'
    )
    
    # ... rest of arguments same as train_algebra.py
```

**Validation:**
```bash
# Dry run
python train_algebra_monolithic.py --train_steps 100 --problems_per_rule 10
```

---

### Phase 3: Training Execution (4-8 hours)

**Task 3.1: Train monolithic model**
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

**Expected Duration:** 4-8 hours on GPU (depends on hardware)

**Monitoring:**
```bash
# Watch training progress
tail -f ./results/monolithic/log.txt

# Check for issues:
# - Loss should decrease steadily
# - Energy gap should increase (>8 units at end)
# - No NaN/Inf values
```

**Checkpoints:**
- `./results/monolithic/model-5.pt` (25k steps)
- `./results/monolithic/model-10.pt` (50k steps)
- `./results/monolithic/model-20.pt` (100k steps)
- `./results/monolithic/model.pt` (200k steps final)

---

### Phase 4: Evaluation Infrastructure (2-3 hours)

**Task 4.1: Add monolithic evaluation to `eval_algebra.py`**

```python
def run_monolithic_evaluation(
    monolithic_checkpoint: str,
    output_dir: str,
    num_samples: int = 1000
) -> Dict:
    """
    Evaluate monolithic model on single-rule and multi-rule datasets.
    
    Returns:
        Dictionary with results for each evaluation type
    """
    
    # Load monolithic model
    diffusion, ebm = load_diffusion_model_for_inference(monolithic_checkpoint)
    
    results = {}
    
    # Single-rule evaluation
    print("\n[Monolithic] Single-rule evaluation")
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        test_dataset = AlgebraDataset(
            rule=rule,
            split='test',
            num_problems=num_samples,
            d_model=128
        )
        
        result = evaluate_with_real_diffusion(
            model=diffusion,
            test_dataset=test_dataset
        )
        
        results[f'single_rule_{rule}'] = result
        print(f"  {rule}: {result['accuracy']:.1%}")
    
    # Multi-rule evaluation  
    print("\n[Monolithic] Multi-rule evaluation")
    for num_rules in [2, 3, 4]:
        test_dataset = MultiRuleDataset(
            num_rules=num_rules,
            split='test',
            num_problems=num_samples,
            d_model=128
        )
        
        result = evaluate_with_real_diffusion(
            model=diffusion,
            test_dataset=test_dataset
        )
        
        results[f'multi_rule_{num_rules}'] = result
        print(f"  {num_rules}-rule: {result['accuracy']:.1%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/monolithic_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

**Task 4.2: Add monolithic option to CLI**
```python
# In eval_algebra.py parse_args()
parser.add_argument(
    '--eval_type',
    type=str,
    default='single_rule',
    choices=['single_rule', 'multi_rule', 'monolithic', 'comparison'],
    help='Evaluation type'
)

parser.add_argument(
    '--monolithic_checkpoint',
    type=str,
    default='./results/monolithic/model.pt',
    help='Path to monolithic model checkpoint'
)
```

**Validation:**
```bash
# Test monolithic evaluation
python eval_algebra.py \
    --eval_type monolithic \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --num_samples 100 \
    --output_dir ./eval_results_test
```

---

### Phase 5: Comparison Script (2-3 hours)

**File to Create:** `compare_monolithic_vs_compositional.py`

Implementation from Finding 5 (full script provided above).

**Validation:**
```bash
# Run comparison (requires both models trained)
python compare_monolithic_vs_compositional.py \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --compositional_dir ./results \
    --num_samples 1000 \
    --output_dir ./comparison_results

# View report
cat ./comparison_results/comparison_report.md
```

---

## 4. Evaluation Metrics

### 4.1 Primary Metrics

**Single-Rule Accuracy:**
```
Monolithic_Accuracy_Rule = (Correct_Predictions / Total_Samples) × 100%

Expected: ~90% for all rules
```

**Multi-Rule Accuracy:**
```
Monolithic_Accuracy_MultiRule = (Correct_Predictions / Total_Samples) × 100%

Expected: ~20-30% (2-rule), ~15-20% (3-rule), ~10-15% (4-rule)
```

**Compositional Advantage:**
```
Advantage = Comp_Accuracy - Mono_Accuracy

Expected: +20-30 percentage points on multi-rule
```

### 4.2 Secondary Metrics

**L2 Distance (Auxiliary):**
```
Mean_L2 = mean(||prediction_emb - target_emb||₂)

Lower is better, tracks convergence quality
```

**Invalid Rate:**
```
Invalid_Rate = (Syntactically_Invalid / Total_Predictions) × 100%

Should be <5% for good decoding
```

**Per-Rule Breakdown:**
```
For multi-rule problems, track which specific rule combinations are hardest
```

---

## 5. Expected Results

### 5.1 From Proposal (Table)

| Model               | Single-Rule Acc | Multi-Rule Acc |
|---------------------|----------------|----------------|
| Monolithic IRED     | ~90%           | ~20–30%        |
| **Compositional**   | **~85%**       | **~50–60%**    |
| NLM Baseline        | ~90%           | ~70%+          |

### 5.2 Detailed Breakdown

**Single-Rule Expected Results:**
```
Monolithic Performance:
- distribute: 88-92%
- combine: 88-92%
- isolate: 88-92%
- divide: 88-92%
Average: ~90%

Compositional Performance:
- distribute: 83-87%
- combine: 83-87%
- isolate: 83-87%
- divide: 83-87%
Average: ~85%

Difference: Monolithic ~5% better (trained on all rules together)
```

**Multi-Rule Expected Results:**
```
2-Rule Problems:
- Monolithic: 25-30%
- Compositional: 50-60%
- Advantage: +25-30 points

3-Rule Problems:
- Monolithic: 18-23%
- Compositional: 40-50%
- Advantage: +22-27 points

4-Rule Problems:
- Monolithic: 12-17%
- Compositional: 30-40%
- Advantage: +18-23 points
```

### 5.3 Success Criteria

**Minimum Success:**
- Monolithic achieves >80% single-rule accuracy
- Compositional outperforms monolithic on multi-rule by >15 points

**Full Success:**
- Monolithic achieves ~90% single-rule accuracy  
- Compositional outperforms monolithic on multi-rule by >25 points
- Results match proposal expectations

---

## 6. Debugging Strategy

### 6.1 If Monolithic Performance is Too Low (<70% single-rule)

**Possible causes:**
1. **Training issues:** Loss not converging, flat energy landscapes
2. **Dataset issues:** Malformed equations, poor diversity
3. **Architecture issues:** Model capacity insufficient

**Debug steps:**
```python
# 1. Check training loss
python -c "
import json
with open('results/monolithic/log.json') as f:
    logs = json.load(f)
    print(f'Final loss: {logs[-1][\"loss\"]}')
    print(f'Energy gap: {logs[-1].get(\"energy_gap\", 0)}')
"

# 2. Check energy landscapes
python tests/debug/debug_energy_landscapes.py \
    --checkpoint ./results/monolithic/model.pt \
    --num_samples 10

# 3. Validate dataset
python -c "
from src.algebra.algebra_dataset import CombinedAlgebraDataset
dataset = CombinedAlgebraDataset(split='test', problems_per_rule=100)
for i in range(10):
    inp, target = dataset[i]
    print(f'Sample {i}: inp.shape={inp.shape}, target.shape={target.shape}')
"
```

### 6.2 If Monolithic Performance is Too High on Multi-Rule (>40%)

**Possible causes:**
1. **Evaluation bug:** Decoder using wrong candidate set
2. **Test leakage:** Multi-rule problems too similar to training
3. **Unexpected generalization:** Model learned better than expected!

**Debug steps:**
```python
# 1. Check multi-rule dataset diversity
python scripts/inspect_multi_rule_targets.py --num_rules 2 --num_samples 100

# 2. Verify evaluation correctness
python tests/unit/test_evaluation_pipeline.py -v

# 3. Manual inspection
python -c "
from src.algebra.algebra_dataset import MultiRuleDataset
dataset = MultiRuleDataset(num_rules=2, split='test', num_problems=10)
for i in range(10):
    inp, target, rules = dataset.equation_data[i]
    print(f'{i}: {inp} -> {target} (rules: {rules})')
"
```

### 6.3 If Compositional Advantage is Small (<15 points)

**Possible causes:**
1. **Compositional inference bug:** Energy summation incorrect
2. **Monolithic unexpectedly good:** Model generalizes better than expected
3. **Training issues:** Neither model learned well

**Debug steps:**
```bash
# 1. Test compositional inference on known examples
python tests/unit/test_algebra_inference.py -v

# 2. Compare energy landscapes
python scripts/compare_energy_landscapes.py \
    --monolithic ./results/monolithic/model.pt \
    --compositional_dir ./results

# 3. Manual verification
python -c "
# Load both models and test on same problem
# Verify compositional uses all 4 energies
# Check energy values are reasonable
"
```

---

## 7. Timeline Estimate

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| **1** | Create `CombinedAlgebraDataset` | 1-2 hours | None |
| **2** | Create `train_algebra_monolithic.py` | 1 hour | Phase 1 |
| **3** | Train monolithic model | 4-8 hours | Phase 2 |
| **4** | Add monolithic evaluation | 2-3 hours | Phase 3 |
| **5** | Create comparison script | 2-3 hours | Phase 4 |
| **6** | Run full comparison | 1-2 hours | Phase 5 |
| **7** | Generate report & analysis | 1 hour | Phase 6 |

**Total Time:** 12-20 hours (1.5-2.5 days)

**Parallelization:**
- Phases 1-2 can be done while compositional models train
- Phase 4-5 can be developed during Phase 3 (training)

---

## 8. Files to Create/Modify

### New Files:
1. `src/algebra/algebra_dataset.py` - Add `CombinedAlgebraDataset` class (~150 lines)
2. `train_algebra_monolithic.py` - Monolithic training script (~850 lines, mostly copied)
3. `compare_monolithic_vs_compositional.py` - Comparison script (~400 lines)

### Modified Files:
1. `eval_algebra.py` - Add monolithic evaluation option (~100 lines added)
2. `src/algebra/algebra_evaluation.py` - Add monolithic evaluation function (~80 lines added)

**Total New Code:** ~1600 lines (mostly copy-paste with modifications)

---

## 9. Critical Implementation Notes

### 9.1 Fair Comparison Requirements

**MUST ENSURE:**
1. ✅ Same total training data (200k problems)
2. ✅ Same architecture (AlgebraEBM with 512 hidden units)
3. ✅ Same training hyperparameters (learning rate, batch size, etc.)
4. ✅ Same evaluation protocol (same test sets, same metrics)
5. ⚠️ Comparable compute budget (200k steps for monolithic vs 4×50k for compositional)

### 9.2 Training Stability

**Monitor for:**
- Loss convergence (should decrease steadily)
- Energy gap (should increase to >8 units)
- No NaN/Inf values
- Reasonable training time (~4-8 hours, not >>10 hours)

**If training is unstable:**
- Reduce learning rate (1e-4 → 5e-5)
- Increase batch size (2048 → 4096)
- Check dataset for malformed equations

### 9.3 Evaluation Correctness

**CRITICAL CHECKS:**
1. **Decoder candidate set:** Must use test dataset equations, not hardcoded defaults
2. **Distance threshold:** Use 2.0 for normalized embeddings
3. **Real diffusion sampling:** Use `GaussianDiffusion1D.sample()`, not custom inference
4. **Fair test sets:** Multi-rule problems never seen during training

### 9.4 Result Interpretation

**Expected patterns:**
- Monolithic slightly better on single-rule (~5%)
- Compositional much better on multi-rule (~50-100% relative improvement)

**If patterns don't match:**
- Check for bugs (most likely cause)
- Verify training completed successfully
- Validate evaluation correctness
- Consider dataset/architecture issues

---

## 10. Risk Assessment

### Low Risk ✅
- Dataset creation (straightforward extension)
- Training script modification (mostly copy-paste)
- Code complexity (using proven infrastructure)

### Medium Risk ⚠️
- Training time (4-8 hours, could be longer)
- Compute budget decision (200k vs 50k steps)
- Result variability (stochastic training)

### High Risk ❌
- None identified! This is well-scoped work building on proven code.

---

## 11. Conclusion

### Summary

Creating a monolithic IRED baseline is **straightforward**:

1. **Dataset:** Combine all 4 rules into `CombinedAlgebraDataset` (150 lines)
2. **Training:** Copy `train_algebra.py` → `train_algebra_monolithic.py` (minor mods)
3. **Evaluation:** Add monolithic option to `eval_algebra.py` (~80 lines)
4. **Comparison:** Create comprehensive comparison script (~400 lines)

**Total effort:** 12-20 hours (including 4-8 hours GPU training)

### Key Insights

1. **Fair comparison requires careful thought:** Must match compute budget (200k steps vs 4×50k)
2. **Evaluation is critical:** Same test sets, same metrics, same protocol
3. **Expected result:** Compositional wins on multi-rule by ~25-30 points
4. **This proves compositionality works:** Zero-shot generalization through energy composition

### Next Steps

1. **Implement `CombinedAlgebraDataset`** (1-2 hours)
2. **Create `train_algebra_monolithic.py`** (1 hour)
3. **Train monolithic model** (4-8 hours GPU)
4. **Run comparison** (2-3 hours evaluation + analysis)
5. **Generate report** showing compositional advantage

---

**Report compiled by:** Claude (Deep Research)  
**Research duration:** 60 minutes  
**Files analyzed:** 8 core files  
**Lines of code reviewed:** ~4000 lines  
**Recommendation:** Proceed with implementation - straightforward and well-scoped
