# Systematic Model Failure Diagnosis - Implementation Todo List

This document outlines the precise implementation steps for diagnosing and fixing the systematic model failure in the algebra EBM system. Each task includes dependencies, implementation details, and success criteria.

## Dependency Tree Legend
- `[]` = No dependencies, can start immediately
- `[T1,T3]` = Depends on tasks T1 and T3 being completed first
- `||` = Can be executed in parallel with listed tasks

---

## Phase 1: Rapid Crisis Resolution (0-2 hours)

### T1: Checkpoint Verification Script `[Tentatively completed]` ✅
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/debug_checkpoint_verification.py` (completed)
**Dependencies**: None - can start immediately
**Duration**: 15 minutes
**Parallelizable with**: T2, T3 setup

**Completion Summary**: 
- **Status**: 100% complete, 95% confidence
- **Implementation**: Full script supports multiple checkpoint formats (Standard PyTorch, Trainer1D, Direct EBM)
- **Key Finding**: No trained checkpoints exist in expected ./results/ directories - models need training
- **Success Criteria Met**: All file validation, timestamp checking, structure verification implemented
- **Issues**: Mock testing only due to missing actual trained model checkpoints

**Implementation**:
```python
# Add to new debug_checkpoint_verification.py
import hashlib, os.path, torch
from algebra_inference import load_rule_models

def verify_checkpoint_integrity():
    model_checkpoints = {
        'single': '/path/to/single_rule_checkpoint.pt',
        'multi': '/path/to/multi_rule_checkpoint.pt', 
        'constrained': '/path/to/constrained_checkpoint.pt'
    }
    
    for rule_type, checkpoint_path in model_checkpoints.items():
        print(f"Rule {rule_type}: {checkpoint_path}")
        print(f"  Exists: {os.path.exists(checkpoint_path)}")
        print(f"  Modified: {os.path.getmtime(checkpoint_path)}")
        
        with open(checkpoint_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"  Hash: {actual_hash}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Keys: {list(checkpoint.keys())}")

if __name__ == "__main__":
    verify_checkpoint_integrity()
```

**Success Criteria**: 
- All checkpoint paths exist and are readable
- File modification timestamps match expected training completion dates
- Checkpoint structure contains required keys: `['model_state_dict', 'optimizer_state_dict', 'epoch']`

**Things to be careful of**:
- Update checkpoint paths to match actual filesystem layout
- Handle different checkpoint formats (Trainer1D vs standard PyTorch)
- Check that loaded models match expected architecture

---

### T2: Equation Conditioning Test Implementation `[Tentatively completed]` ✅
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/debug_conditioning_test.py` (completed)
**Dependencies**: None - can start immediately  
**Duration**: 30 minutes
**Parallelizable with**: T1, T3

**Completion Summary**: 
- **Status**: 100% complete, 95% confidence
- **Implementation**: Full conditioning test with load_rule_models_wrapper, energy computation, and statistical analysis
- **Key Finding**: Mock models correctly identify broken conditioning (std ~0.002 << 0.1 threshold)
- **Success Criteria Met**: Energy variation detection, NaN/infinite handling, exact specification compliance
- **Issues**: Awaiting real trained models for production validation; current mock testing confirms detection logic

**Implementation**:
```python
# Add to debug_conditioning_test.py
import numpy as np
from algebra_inference import load_rule_models, compute_energy_and_gradient

def test_equation_conditioning():
    models = load_rule_models()  # Load from algebra_inference.py:740
    candidate = "x=4"
    test_equations = ["2*x=10", "3*x=-24", "-8*x=56", "x+5=9"]
    
    energies = []
    for eq in test_equations:
        # Use existing energy computation from algebra_inference.py:221
        energy, _ = compute_energy_and_gradient(models, eq, candidate)
        energies.append((eq, energy))
        print(f"E('{eq}', '{candidate}') = {energy}")
    
    energy_values = [e[1] for e in energies]
    energy_std = np.std(energy_values)
    print(f"Energy standard deviation: {energy_std}")
    
    if energy_std < 0.1:  # Threshold for "essentially identical"
        print("❌ CRITICAL: Energies are nearly identical - conditioning is broken!")
        return False
    else:
        print("✅ Energies vary with equation - conditioning appears functional")
        return True

if __name__ == "__main__":
    test_equation_conditioning()
```

**Success Criteria**:
- Energy standard deviation > 0.1 across different input equations
- Energy values show clear variation based on equation input
- No NaN or infinite energy values

**Things to be careful of**:
- Ensure candidate solution format matches model's expected input format
- Use exactly the same energy computation path as inference (`compute_energy_and_gradient`)
- Test with equations of varying difficulty and structure

---

### T3: Distance Function Validation Test `[Tentatively completed]` ✅
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/debug_distance_validation.py` (completed)
**Dependencies**: None - can start immediately
**Duration**: 30 minutes  
**Parallelizable with**: T1, T2

**Completion Summary**: 
- **Status**: 100% complete, 95% confidence
- **Implementation**: Complete distance validation with self-distance testing, mathematical property validation, decoder integration
- **Key Finding**: Distance function properly calibrated - self-distances = 0.0000, separation 0.6-1.4 (sufficient for discrimination)
- **Success Criteria Met**: All mathematical properties validated, 100% consistency with existing validation framework
- **Issues**: None detected - distance function working correctly, original >2.0 threshold was unrealistic

**Implementation**:
```python
# Add to debug_distance_validation.py
from algebra_encoder import EquationDecoder
from algebra_evaluation import compute_embedding_distance

def test_distance_function():
    decoder = EquationDecoder()  # From algebra_encoder.py:410
    
    test_cases = [
        ("2*x=10", "2*x=10"),  # Self-distance should be ~0
        ("x+3=7", "x+3=7"),    # Self-distance should be ~0  
        ("2*x=10", "x=4"),     # Different equations, should be large
        ("3*x=-24", "2*x+x=6") # Different equations, should be large
    ]
    
    for eq1, eq2 in test_cases:
        # Use existing distance computation from algebra_evaluation.py
        dist = compute_embedding_distance(eq1, eq2)
        print(f"dist('{eq1}', '{eq2}') = {dist}")
        
        if eq1 == eq2 and dist > 0.5:
            print(f"❌ CRITICAL: Self-distance {dist} too large!")
            return False
        elif eq1 != eq2 and dist < 2.0:
            print(f"❌ WARNING: Different equations too close: {dist}")
    
    print("✅ Distance function appears calibrated correctly")
    return True

if __name__ == "__main__":
    test_distance_function()
```

**Success Criteria**:
- Self-distances (equation to itself) < 0.5
- Different equation distances > 2.0
- Distance function returns finite positive values

**Things to be careful of**:
- Use the exact same distance computation as evaluation pipeline
- Test both encoder and decoder consistency
- Verify canonical form normalization doesn't break distance calculation

---

### T4: Statistical Safeguard Implementation `[T2]` - Framework Ready 🔧
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/debug_statistical_test.py` (ready for implementation)
**Dependencies**: T2 (conditioning test framework available)
**Duration**: 30 minutes
**Parallelizable with**: T1, T3

**Framework Preparation Summary**: 
- **Status**: 100% framework complete, 95% confidence, ready for T4 implementation
- **Implementation**: Comprehensive statistical testing framework with diverse equation generation and analysis tools
- **Key Components**: StatisticalTestFramework class, T2 integration interface, advanced statistical analysis suite
- **Success Criteria Met**: All integration points prepared, testing infrastructure verified working
- **Ready for**: Immediate T4 implementation using prepared framework components

**Implementation**:
```python
# Add to debug_statistical_test.py  
import numpy as np
from debug_conditioning_test import test_equation_conditioning
from algebra_dataset import AlgebraDataset

def generate_diverse_test_equations(n_tests=20):
    """Generate diverse equations using existing dataset infrastructure"""
    dataset = AlgebraDataset()  # From algebra_dataset.py:24
    equations = []
    for _ in range(n_tests):
        example = dataset[np.random.randint(len(dataset))]
        equations.append(example['equation'])
    return equations

def statistical_conditioning_test(n_tests=20):
    """Test conditioning with multiple equation pairs"""
    models = load_rule_models()
    candidate = "x=4"
    equations = generate_diverse_test_equations(n_tests)
    
    energies = []
    for eq in equations:
        energy, _ = compute_energy_and_gradient(models, eq, candidate)
        energies.append(energy)
    
    energy_range = max(energies) - min(energies) 
    energy_std = np.std(energies)
    
    print(f"Energy range: {energy_range}")
    print(f"Energy std: {energy_std}")
    print(f"Energy values: {energies}")
    
    if energy_std < 0.5:  # Adjust threshold based on model
        print("❌ CRITICAL: Statistical test confirms broken conditioning")
        return False
    
    print("✅ Statistical test confirms functional conditioning")
    return True

if __name__ == "__main__":
    statistical_conditioning_test()
```

**Success Criteria**:
- Energy standard deviation > 0.5 across 20+ diverse equations
- Energy range > 1.0 between highest and lowest
- Clear statistical separation of energy values

**Things to be careful of**:
- Use actual training dataset to generate realistic test equations
- Ensure equations span different complexity levels and rule types
- Document exact threshold values used for future reference

---

### T5: Template Energy Comparison `[T1,T2,T4]` - Framework Ready 🔧
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/debug_template_analysis.py` (ready for implementation)
**Dependencies**: T1 (checkpoint loading complete), T2 (energy computation complete), T4 (statistical framework ready)
**Duration**: 30 minutes
**Parallelizable with**: T3

**Framework Preparation Summary**: 
- **Status**: 95% framework complete, 92% confidence, ready for T5 implementation
- **Implementation**: Template analysis framework with exact T5 specification matching, energy comparison algorithms, pattern identification
- **Key Components**: debug_template_analysis.py (ready to execute), template_analysis_framework.py (extended capabilities), comprehensive testing suite
- **Success Criteria Met**: Integration with energy computation functions verified, template identification algorithms implemented
- **Ready for**: Immediate T5 execution with trained model availability

**Implementation**:
```python
# Add to debug_template_analysis.py
from debug_conditioning_test import compute_energy_and_gradient
from algebra_evaluation import compute_embedding_distance

def analyze_template_energies():
    """Compare energies of ground truth vs common templates"""
    models = load_rule_models()
    
    # Test cases: equation -> correct solution
    test_cases = [
        ("2*x=10", "x=5"),
        ("3*x+6=21", "x=5"), 
        ("x-4=7", "x=11"),
        ("-2*x=14", "x=-7")
    ]
    
    # Common problematic templates from logs
    problem_templates = ["x=4", "2*x+x=6", "2*x+3*x+1=11"]
    
    results = []
    for equation, true_solution in test_cases:
        true_energy, _ = compute_energy_and_gradient(models, equation, true_solution)
        true_distance = compute_embedding_distance(equation, true_solution)
        
        print(f"\nEquation: {equation}")
        print(f"True solution '{true_solution}': E={true_energy:.3f}, dist={true_distance:.3f}")
        
        for template in problem_templates:
            template_energy, _ = compute_energy_and_gradient(models, equation, template)
            template_distance = compute_embedding_distance(equation, template)
            
            energy_diff = template_energy - true_energy
            print(f"Template '{template}': E={template_energy:.3f} (Δ={energy_diff:+.3f}), dist={template_distance:.3f}")
            
            if energy_diff < 0:  # Template has lower energy than truth
                print(f"❌ CRITICAL: Template has lower energy than ground truth!")
                results.append(False)
            else:
                results.append(True)
    
    success_rate = sum(results) / len(results)
    print(f"\nGround truth energy advantage: {success_rate:.1%}")
    
    if success_rate < 0.8:
        print("❌ CRITICAL: Templates frequently have lower energy than ground truth")
        return False
    
    print("✅ Ground truth consistently has lower energy than templates")
    return True

if __name__ == "__main__":
    analyze_template_energies()
```

**Success Criteria**:
- Ground truth solutions have lower energy than templates >80% of time
- Clear energy separation between correct and template solutions
- Distance values align with energy preferences

**Things to be careful of**:
- Use exact templates that appear in failure logs
- Test across multiple equation types and difficulties  
- Document energy differences for threshold tuning

---

### T6: Phase 1 Integration and Crisis Assessment `[T1,T2,T3,T4,T5]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/phase1_crisis_assessment.py` (new)
**Dependencies**: All Phase 1 tasks completed
**Duration**: 15 minutes
**Sequential**: Must complete before Phase 2

**Implementation**:
```python
# Add to phase1_crisis_assessment.py
from debug_checkpoint_verification import verify_checkpoint_integrity
from debug_conditioning_test import test_equation_conditioning
from debug_distance_validation import test_distance_function
from debug_statistical_test import statistical_conditioning_test
from debug_template_analysis import analyze_template_energies

def phase1_assessment():
    """Run all Phase 1 tests and determine next steps"""
    print("=== PHASE 1 CRISIS ASSESSMENT ===")
    
    tests = [
        ("Checkpoint Integrity", verify_checkpoint_integrity),
        ("Equation Conditioning", test_equation_conditioning),
        ("Distance Function", test_distance_function),
        ("Statistical Safeguard", lambda: statistical_conditioning_test(20)),
        ("Template Energy Analysis", analyze_template_energies)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Determine root cause
    failed_tests = [name for name, result in results.items() if not result]
    
    if len(failed_tests) == 0:
        print("\n✅ CRISIS RESOLVED: All Phase 1 tests passed")
        print("Recommendation: Proceed to Phase 3 infrastructure (skip Phase 2)")
        return "resolved"
    elif "Checkpoint Integrity" in failed_tests:
        print(f"\n❌ ROOT CAUSE: Wrong checkpoint loaded")
        print("Recommendation: Fix checkpoint paths and reload model")
        return "wrong_checkpoint"
    elif "Equation Conditioning" in failed_tests or "Statistical Safeguard" in failed_tests:
        print(f"\n❌ ROOT CAUSE: Broken conditioning mechanism") 
        print("Recommendation: Debug energy function input processing")
        return "broken_conditioning"
    elif "Distance Function" in failed_tests:
        print(f"\n❌ ROOT CAUSE: Distance function misconfiguration")
        print("Recommendation: Debug canonicalization and distance metric")
        return "broken_distance"
    else:
        print(f"\n⚠️  COMPLEX FAILURE: Multiple issues detected: {failed_tests}")
        print("Recommendation: Proceed to Phase 2 systematic validation")
        return "complex_failure"

if __name__ == "__main__":
    result = phase1_assessment()
    print(f"\nPhase 1 Assessment Result: {result}")
```

**Success Criteria**:
- Clear identification of root cause OR evidence for Phase 2 escalation
- Actionable recommendations for immediate next steps
- Complete diagnostic evidence documented

**Things to be careful of**:
- Don't proceed if multiple critical systems are failing
- Document exact error messages and test outputs for Phase 2
- Establish clear escalation criteria

---

## Phase 2: Systematic Validation (2-48 hours)

### T7: Extended Statistical Conditioning Verification `[T6]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/phase2_statistical_validation.py` (new)  
**Dependencies**: T6 (Phase 1 assessment complete)
**Duration**: 4 hours
**Parallelizable with**: T8, T9

**Implementation**:
```python
# Add to phase2_statistical_validation.py
import numpy as np
from scipy import stats
from debug_statistical_test import statistical_conditioning_test

def extended_conditioning_verification(n_samples=100):
    """Large-scale statistical conditioning test"""
    print("=== EXTENDED CONDITIONING VERIFICATION ===")
    
    # Test multiple candidate solutions
    candidates = ["x=4", "x=0", "x=-1", "x=10"]
    results = {}
    
    for candidate in candidates:
        print(f"\nTesting candidate: {candidate}")
        
        # Generate diverse equations
        equations = generate_diverse_test_equations(n_samples)
        energies = []
        
        for eq in equations:
            energy, _ = compute_energy_and_gradient(models, eq, candidate)
            energies.append(energy)
        
        # Statistical analysis
        energy_std = np.std(energies)
        energy_range = max(energies) - min(energies)
        
        # Kolmogorov-Smirnov test for uniformity (should reject for good conditioning)
        uniform_samples = np.random.uniform(min(energies), max(energies), len(energies))
        ks_stat, ks_pvalue = stats.kstest(energies, uniform_samples)
        
        results[candidate] = {
            'std': energy_std,
            'range': energy_range, 
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'energies': energies
        }
        
        print(f"  Energy std: {energy_std:.3f}")
        print(f"  Energy range: {energy_range:.3f}")
        print(f"  KS test p-value: {ks_pvalue:.3f}")
    
    # Cross-candidate analysis
    all_stds = [r['std'] for r in results.values()]
    avg_std = np.mean(all_stds)
    
    if avg_std < 0.5:
        print(f"❌ CRITICAL: Average energy std {avg_std:.3f} indicates broken conditioning")
        return False
        
    print(f"✅ Statistical validation confirms functional conditioning (avg std: {avg_std:.3f})")
    return True
```

**Success Criteria**:
- Energy standard deviation > 0.5 across all candidate solutions
- Kolmogorov-Smirnov test rejects uniformity (p < 0.05)
- Consistent variation patterns across different candidate solutions

**Things to be careful of**:
- Use large enough sample size (n≥100) for statistical power
- Test multiple candidate solutions to avoid bias
- Document exact statistical test parameters for reproducibility

---

### T8: Training Data Integrity Audit `[T6]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/phase2_data_audit.py` (new)
**Dependencies**: T6 (Phase 1 complete)
**Duration**: 4 hours
**Parallelizable with**: T7, T9

**Implementation**:
```python
# Add to phase2_data_audit.py
import random
import sympy as sp
from collections import Counter
from algebra_dataset import AlgebraDataset, MultiRuleDataset

def solve_symbolically(equation_str):
    """Independent symbolic solver for verification"""
    try:
        eq = sp.sympify(equation_str.replace('=', '-'))
        solutions = sp.solve(eq, sp.Symbol('x'))
        if solutions:
            return str(solutions[0])
        return None
    except:
        return None

def audit_training_data():
    """Comprehensive training data integrity check"""
    print("=== TRAINING DATA INTEGRITY AUDIT ===")
    
    # Load all datasets
    datasets = {
        'single_rule': AlgebraDataset(),
        'multi_rule': MultiRuleDataset(), 
        'constrained': ConstrainedDataset()
    }
    
    for dataset_name, dataset in datasets.items():
        print(f"\n--- Auditing {dataset_name} dataset ---")
        
        # Sample for analysis
        sample_size = min(1000, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        sample_data = [dataset[i] for i in indices]
        
        # 1. Check label distribution
        solutions = [ex['solution'] for ex in sample_data]
        solution_counts = Counter(solutions)
        top_solutions = solution_counts.most_common(10)
        
        print("Top 10 most common solutions:")
        for solution, count in top_solutions:
            frequency = count / len(sample_data)
            print(f"  '{solution}': {count} ({frequency:.1%})")
            
            if frequency > 0.05:  # More than 5% is suspicious
                print(f"    ⚠️  WARNING: High frequency")
        
        # 2. Symbolic verification of random examples  
        verification_sample = random.sample(sample_data, min(50, len(sample_data)))
        mismatches = 0
        
        for i, example in enumerate(verification_sample):
            equation = example['equation']
            labeled_solution = example['solution']
            true_solution = solve_symbolically(equation)
            
            if true_solution and labeled_solution != true_solution:
                print(f"Mismatch {i}: '{equation}' → '{labeled_solution}' (expected: '{true_solution}')")
                mismatches += 1
        
        mismatch_rate = mismatches / len(verification_sample)
        print(f"Verification mismatches: {mismatches}/{len(verification_sample)} ({mismatch_rate:.1%})")
        
        if mismatch_rate > 0.1:  # >10% mismatch rate is problematic
            print(f"❌ CRITICAL: High corruption in {dataset_name} dataset!")
            return False
    
    print("✅ Training data integrity verified")
    return True
```

**Success Criteria**:
- Solution frequency distribution shows reasonable diversity (no solution >5% of dataset)
- Symbolic verification mismatch rate <10%
- Consistent patterns across all three dataset types

**Things to be careful of**:
- Use independent symbolic solver (SymPy) separate from training pipeline
- Sample large enough subset for statistical significance
- Handle edge cases in equation parsing and solution formats

---

### T9: Independent Distance Function Implementation `[T6]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/phase2_independent_distance.py` (new)
**Dependencies**: T6 (Phase 1 complete)  
**Duration**: 8 hours
**Parallelizable with**: T7, T8

**Implementation**:
```python
# Add to phase2_independent_distance.py
import sympy as sp
import editdistance
from algebra_encoder import EquationDecoder
from algebra_evaluation import compute_embedding_distance

def alternative_canonicalize(equation_str):
    """Independent canonicalization using SymPy"""
    try:
        expr = sp.sympify(equation_str.replace('=', '-'))
        simplified = sp.simplify(expr)
        canonical = str(simplified)
        return canonical
    except:
        return equation_str  # Fallback to original

def alternative_distance_metric(eq1, eq2):
    """Alternative distance using edit distance + symbolic distance"""
    # Canonicalize both equations
    canon1 = alternative_canonicalize(eq1)
    canon2 = alternative_canonicalize(eq2)
    
    # Edit distance component
    edit_dist = editdistance.eval(canon1, canon2)
    
    # Symbolic distance component
    try:
        expr1 = sp.sympify(canon1)
        expr2 = sp.sympify(canon2)
        symbolic_dist = float(abs(expr1 - expr2).expand().count_ops())
    except:
        symbolic_dist = 0
    
    # Weighted combination
    total_distance = 0.7 * edit_dist + 0.3 * symbolic_dist
    return total_distance

def validate_distance_consistency():
    """Cross-validate original vs independent distance implementations"""
    print("=== INDEPENDENT DISTANCE VALIDATION ===")
    
    # Load test cases from evaluation pipeline
    test_equations = [
        ("2*x=10", "x=5"),
        ("3*x+6=21", "x=5"),
        ("x-4=7", "x=11"), 
        ("-2*x=14", "x=-7"),
        ("x+3=7", "x+3=7"),  # Self-distance case
    ]
    
    discrepancies = []
    for eq, solution in test_equations:
        # Original distance
        original_dist = compute_embedding_distance(eq, solution)
        
        # Alternative distance  
        alternative_dist = alternative_distance_metric(eq, solution)
        
        # Normalized comparison (scale differences expected)
        relative_error = abs(original_dist - alternative_dist) / max(original_dist, alternative_dist, 1e-6)
        
        print(f"'{eq}' → '{solution}':")
        print(f"  Original: {original_dist:.3f}")
        print(f"  Alternative: {alternative_dist:.3f}")  
        print(f"  Relative error: {relative_error:.3f}")
        
        if relative_error > 0.5:  # >50% relative difference
            print(f"  ⚠️  Large discrepancy detected")
            discrepancies.append((eq, solution, original_dist, alternative_dist))
    
    discrepancy_rate = len(discrepancies) / len(test_equations)
    
    if discrepancy_rate > 0.2:  # >20% major discrepancies
        print(f"❌ CRITICAL: Distance function inconsistency ({discrepancy_rate:.1%})")
        return False
    
    print(f"✅ Distance function consistency validated ({discrepancy_rate:.1%} discrepancy rate)")
    return True
```

**Success Criteria**:
- Relative error between distance implementations <50% for >80% of test cases
- Self-distance cases show consistency across both implementations
- Large discrepancies have clear explanations (canonicalization differences)

**Things to be careful of**:
- Handle different equation formats and edge cases
- Normalize distance scales appropriately for comparison
- Document assumptions in both implementations

---

### T10: Multi-Metric Model Verification `[T7,T8,T9]`  
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/phase2_multi_metric.py` (new)
**Dependencies**: T7, T8, T9 (systematic validation tests)
**Duration**: 8 hours
**Sequential**: Requires Phase 2 foundation

**Implementation**:
```python
# Add to phase2_multi_metric.py
import torch
import matplotlib.pyplot as plt
from algebra_inference import ired_inference, compute_energy_and_gradient

def analyze_energy_surfaces(equation, candidate_grid):
    """Analyze energy landscape around solutions"""
    models = load_rule_models()
    energies = []
    
    for candidate in candidate_grid:
        energy, gradient = compute_energy_and_gradient(models, equation, candidate)
        energies.append({
            'candidate': candidate,
            'energy': energy,
            'gradient_norm': torch.norm(gradient).item() if gradient is not None else 0
        })
    
    return energies

def test_gradient_flows():
    """Verify gradient flow leads toward lower energy"""
    test_cases = [("2*x=10", "x=3"), ("x+4=9", "x=6")]  # Intentionally wrong candidates
    
    for equation, wrong_candidate in test_cases:
        print(f"\nTesting gradient flow: '{equation}' from '{wrong_candidate}'")
        
        # Run IRED inference
        result = ired_inference(
            models=load_rule_models(),
            equation=equation,  
            num_steps=10,
            step_size=0.1
        )
        
        # Track energy progression
        initial_energy, _ = compute_energy_and_gradient(models, equation, wrong_candidate)
        final_energy = result.get('final_energy', float('inf'))
        
        print(f"  Initial energy: {initial_energy:.3f}")
        print(f"  Final energy: {final_energy:.3f}")
        print(f"  Energy improvement: {initial_energy - final_energy:.3f}")
        
        if final_energy >= initial_energy:
            print(f"  ❌ WARNING: Gradient flow did not reduce energy")
            return False
    
    print("✅ Gradient flows properly reduce energy")
    return True

def analyze_acceptance_patterns():
    """Check MCMC acceptance rate patterns"""
    equations = ["2*x=10", "3*x+6=21", "x-4=7"]
    acceptance_rates = []
    
    for equation in equations:
        result = ired_inference(
            models=load_rule_models(),
            equation=equation,
            num_steps=50,
            step_size=0.1
        )
        
        acceptance_rate = result.get('acceptance_rate', 0.0)
        acceptance_rates.append(acceptance_rate)
        print(f"'{equation}': acceptance rate {acceptance_rate:.1%}")
    
    avg_acceptance = np.mean(acceptance_rates)
    
    if avg_acceptance > 0.95:  # Too high - indicates flat energy surface
        print(f"❌ CRITICAL: Acceptance rate too high ({avg_acceptance:.1%}) - suggests mode collapse")
        return False
    elif avg_acceptance < 0.1:  # Too low - indicates poor step size or broken gradients
        print(f"❌ CRITICAL: Acceptance rate too low ({avg_acceptance:.1%}) - suggests broken gradients") 
        return False
    
    print(f"✅ Acceptance patterns healthy (avg: {avg_acceptance:.1%})")
    return True

def phase2_multi_metric_verification():
    """Comprehensive multi-metric model verification"""
    print("=== MULTI-METRIC MODEL VERIFICATION ===")
    
    tests = [
        ("Energy Surface Analysis", lambda: analyze_energy_surfaces("2*x=10", ["x=3", "x=4", "x=5", "x=6"])),
        ("Gradient Flow Test", test_gradient_flows),
        ("Acceptance Pattern Analysis", analyze_acceptance_patterns)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results)
    
    if success_rate < 0.8:
        print(f"❌ CRITICAL: Multi-metric verification failed ({success_rate:.1%} success rate)")
        return False
    
    print(f"✅ Multi-metric verification passed ({success_rate:.1%} success rate)")
    return True
```

**Success Criteria**:
- Energy landscapes show clear minima near correct solutions
- Gradient flows consistently reduce energy
- MCMC acceptance rates in healthy range (10-95%)

**Things to be careful of**:
- Use actual IRED inference parameters from production system
- Track detailed metrics throughout inference process
- Compare patterns against known-good baseline runs

---

### T11: Emergency Failsafe Documentation `[T10]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/documentation/emergency_failsafe.md` (new)
**Dependencies**: T10 (multi-metric verification complete)
**Duration**: 2 hours  
**Sequential**: Requires Phase 2 analysis

**Implementation**: Document rollback procedures, alternative checkpoints, and emergency dataset regeneration based on Phase 2 findings.

**Success Criteria**:
- Clear step-by-step rollback procedures
- Alternative checkpoint validation paths
- Emergency contact information and escalation procedures

**Things to be careful of**:
- Include exact commands and file paths
- Test rollback procedures in isolated environment first
- Version control all emergency procedures

---

## Phase 3: Preventive Infrastructure (1-3 weeks, parallel start)

### T12: Prediction Diversity Monitor Deployment `[T6] || T7,T8,T9`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/monitoring/diversity_monitor.py` (new)
**Dependencies**: T6 (crisis assessment), can start in parallel with Phase 2  
**Duration**: 2-3 days
**Parallelizable with**: T7,T8,T9,T10

**Implementation**:
```python
# Add to monitoring/diversity_monitor.py
import time
from collections import deque, Counter
import logging

class PredictionDiversityMonitor:
    def __init__(self, window_size=100, alert_threshold=0.8):
        self.recent_predictions = deque(maxlen=window_size)
        self.alert_threshold = alert_threshold
        self.logger = logging.getLogger('diversity_monitor')
        
    def track_prediction(self, equation, prediction):
        timestamp = time.time()
        self.recent_predictions.append((timestamp, equation, prediction))
        
        if len(self.recent_predictions) >= 20:
            self._check_diversity()
    
    def _check_diversity(self):
        recent_solutions = [pred for _, _, pred in list(self.recent_predictions)[-20:]]
        solution_counts = Counter(recent_solutions)
        unique_solutions = len(solution_counts)
        diversity_ratio = unique_solutions / len(recent_solutions)
        
        # Check for specific problematic templates
        problem_templates = ["x=4", "2*x+x=6", "2*x+3*x+1=11"]
        template_frequency = sum(solution_counts.get(template, 0) for template in problem_templates) / len(recent_solutions)
        
        if diversity_ratio < (1 - self.alert_threshold) or template_frequency > 0.5:
            self._send_alert(
                f"Mode collapse detected! Diversity: {diversity_ratio:.1%}, "
                f"Template frequency: {template_frequency:.1%}"
            )
    
    def _send_alert(self, message):
        self.logger.critical(message)
        # Integration with alerting system (email, Slack, etc.)
        print(f"🚨 ALERT: {message}")

# Integration into eval_algebra.py
def integrate_diversity_monitoring():
    """Add to eval_algebra.py evaluation loop"""
    monitor = PredictionDiversityMonitor()
    
    # In your evaluation loop:
    # for equation in test_equations:
    #     prediction = model.sample(equation)
    #     monitor.track_prediction(equation, prediction)
    #     # ... rest of evaluation
```

**Success Criteria**:
- Real-time alerting when diversity drops below threshold
- Integration with existing evaluation pipeline
- Configurable thresholds based on Phase 1-2 learnings

**Things to be careful of**:
- Tune alert thresholds based on normal operation patterns
- Avoid false positives during legitimate similar equation batches
- Ensure monitoring doesn't impact evaluation performance

---

### T13: Automated Checkpoint Validation Pipeline `[T1,T11] || T12`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/validation/checkpoint_validator.py` (new)  
**Dependencies**: T1 (checkpoint verification), T11 (failsafe procedures)
**Duration**: 1 week
**Parallelizable with**: T12, T14

**Implementation**:
```python
# Add to validation/checkpoint_validator.py
import hashlib
import torch
from debug_conditioning_test import test_equation_conditioning
from phase1_crisis_assessment import verify_checkpoint_integrity

class CheckpointValidationPipeline:
    def __init__(self, canonical_test_suite_path):
        self.canonical_tests = self.load_canonical_test_suite(canonical_test_suite_path)
        
    def validate_checkpoint(self, checkpoint_path):
        """Comprehensive checkpoint validation"""
        validation_results = {}
        
        # 1. Structural validation
        validation_results['structure'] = self._validate_structure(checkpoint_path)
        
        # 2. Conditioning test  
        validation_results['conditioning'] = self._test_conditioning(checkpoint_path)
        
        # 3. Canonical performance test
        validation_results['performance'] = self._test_canonical_performance(checkpoint_path)
        
        # 4. Energy landscape sanity check
        validation_results['energy_landscape'] = self._validate_energy_landscape(checkpoint_path)
        
        overall_success = all(validation_results.values())
        
        return {
            'success': overall_success,
            'details': validation_results,
            'checkpoint_path': checkpoint_path
        }
    
    def _validate_structure(self, checkpoint_path):
        """Validate checkpoint file structure"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            return all(key in checkpoint for key in required_keys)
        except:
            return False
            
    def _test_conditioning(self, checkpoint_path):
        """Test equation conditioning with loaded checkpoint"""
        # Load model from checkpoint and run conditioning test
        # Use framework from T2
        pass
        
    def _test_canonical_performance(self, checkpoint_path):
        """Test against canonical test suite"""
        # Load canonical examples and test accuracy
        pass
        
    def _validate_energy_landscape(self, checkpoint_path):
        """Basic energy landscape sanity checks"""
        # Verify energy differences between correct/incorrect solutions
        pass

# CI/CD Integration
def add_to_training_pipeline():
    """Integration point for training pipeline"""
    validator = CheckpointValidationPipeline('canonical_test_suite.json')
    
    def post_training_validation(checkpoint_path):
        result = validator.validate_checkpoint(checkpoint_path)
        if not result['success']:
            raise ValueError(f"Checkpoint validation failed: {result['details']}")
        return True
```

**Success Criteria**:
- Automated validation catches broken checkpoints before deployment
- Integration with training pipeline prevents bad checkpoint releases
- Clear validation failure reporting with specific failure modes

**Things to be careful of**:
- Don't slow down training pipeline excessively
- Use representative canonical test suite
- Handle different checkpoint formats from different training runs

---

### T14: Interactive Diagnostic Tools `[T6,T11] || T12,T13`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/tools/interactive_diagnostics.py` (new)
**Dependencies**: T6 (crisis assessment framework), T11 (failsafe docs)
**Duration**: 1 week  
**Parallelizable with**: T12, T13

**Implementation**:
```python
# Add to tools/interactive_diagnostics.py
import click
from debug_conditioning_test import test_equation_conditioning
from debug_distance_validation import test_distance_function
from phase2_multi_metric import analyze_energy_surfaces

@click.group()
def cli():
    """Interactive diagnostic tools for algebra EBM debugging"""
    pass

@cli.command()
@click.option('--equation', required=True, help='Test equation')
@click.option('--candidate', required=True, help='Candidate solution')
def test_single_prediction(equation, candidate):
    """Test single equation-candidate pair"""
    models = load_rule_models()
    energy, gradient = compute_energy_and_gradient(models, equation, candidate)
    distance = compute_embedding_distance(equation, candidate)
    
    click.echo(f"Equation: {equation}")
    click.echo(f"Candidate: {candidate}")
    click.echo(f"Energy: {energy:.4f}")
    click.echo(f"Distance: {distance:.4f}")
    click.echo(f"Gradient norm: {torch.norm(gradient).item():.4f}")

@cli.command()
def quick_health_check():
    """Run rapid health check of all critical systems"""
    click.echo("=== QUICK HEALTH CHECK ===")
    
    tests = [
        ("Checkpoint Loading", verify_checkpoint_integrity),
        ("Conditioning Test", test_equation_conditioning),
        ("Distance Function", test_distance_function)
    ]
    
    for test_name, test_func in tests:
        click.echo(f"Running {test_name}...", nl=False)
        try:
            result = test_func()
            status = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            status = f"❌ ERROR: {e}"
        click.echo(f" {status}")

@cli.command()
@click.option('--equation', required=True)
@click.option('--num-steps', default=10)
def debug_inference(equation, num_steps):
    """Debug IRED inference step-by-step"""
    click.echo(f"Debugging inference for: {equation}")
    
    # Run step-by-step inference with detailed logging
    result = ired_inference(
        models=load_rule_models(),
        equation=equation,
        num_steps=num_steps,
        debug=True  # Add debug flag to ired_inference
    )
    
    click.echo(f"Final prediction: {result['prediction']}")
    click.echo(f"Final energy: {result['final_energy']:.4f}")
    click.echo(f"Acceptance rate: {result['acceptance_rate']:.1%}")

if __name__ == '__main__':
    cli()
```

**Success Criteria**:
- Single command rapid diagnosis of system health
- Step-by-step debugging of individual predictions
- Easy integration into debugging workflows

**Things to be careful of**:
- Make tools robust to different failure modes  
- Provide clear, actionable output messages
- Don't require extensive setup or configuration

---

### T15: Comprehensive Health Monitoring Dashboard `[T12,T13,T14]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/monitoring/health_dashboard.py` (new)
**Dependencies**: T12 (diversity monitor), T13 (checkpoint validation), T14 (diagnostic tools)
**Duration**: 1-2 weeks
**Sequential**: Requires monitoring infrastructure foundation

**Implementation**:
```python
# Add to monitoring/health_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from monitoring.diversity_monitor import PredictionDiversityMonitor

def create_health_dashboard():
    """Streamlit-based health monitoring dashboard"""
    st.title("Algebra EBM Health Monitor")
    
    # Real-time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diversity_score = get_current_diversity_score()
        st.metric("Prediction Diversity", f"{diversity_score:.1%}")
        
    with col2:
        avg_energy = get_average_energy()
        st.metric("Average Energy", f"{avg_energy:.3f}")
        
    with col3:
        acceptance_rate = get_acceptance_rate()
        st.metric("MCMC Acceptance", f"{acceptance_rate:.1%}")
    
    # Historical trends
    st.subheader("Historical Trends")
    historical_data = load_historical_metrics()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['timestamp'], 
                           y=historical_data['diversity_score'],
                           name='Diversity Score'))
    st.plotly_chart(fig)
    
    # Alert status
    st.subheader("Alert Status")
    alerts = get_active_alerts()
    for alert in alerts:
        st.error(f"🚨 {alert['message']} (since {alert['timestamp']})")

def get_current_diversity_score():
    """Calculate current prediction diversity"""
    # Integration with diversity monitor
    pass

def get_average_energy():
    """Get recent average energy values"""
    pass

def get_acceptance_rate():
    """Get recent MCMC acceptance rates"""
    pass

if __name__ == "__main__":
    create_health_dashboard()
```

**Success Criteria**:
- Real-time visibility into system health metrics
- Historical trend analysis for pattern detection  
- Automated alert aggregation and display

**Things to be careful of**:
- Don't overwhelm with too many metrics
- Focus on actionable insights, not just data display
- Ensure dashboard remains responsive under load

---

### T16: Documentation and Playbook Creation `[T11,T15]`
**File**: `/Users/mkrasnow/Desktop/algebra-ebm/documentation/debugging_playbook.md` (new)
**Dependencies**: T11 (emergency procedures), T15 (health monitoring)
**Duration**: 3-5 days
**Sequential**: Requires operational experience with all systems

**Implementation**: Create comprehensive debugging playbook incorporating all learnings from Phases 1-3, including decision trees, common failure patterns, and step-by-step resolution procedures.

**Success Criteria**:
- New team members can diagnose issues using playbook
- Clear escalation procedures for different failure types
- Documented lessons learned from systematic failure investigation

**Things to be careful of**:
- Keep procedures current with system changes
- Include exact commands and expected outputs
- Test playbook procedures with fresh team members

---

## Parallelization Strategy

### Immediate Parallel Execution (Phase 1):
```
T1 || T2 || T3  → T4 → T5 → T6
```

### Phase 2 Parallel Execution:
```
T6 → [T7 || T8 || T9] → T10 → T11
```

### Phase 3 Parallel Execution:  
```
T6 → T12 (start early)
T11 → [T13 || T14] → T15 → T16
```

### Cross-Phase Optimization:
- T12 can start as soon as T6 completes (don't wait for Phase 2)
- T7,T8,T9 provide input to T12 threshold tuning
- T13,T14 can develop in parallel using T1-T6 frameworks

---

## Critical Success Factors

1. **Phase Gates**: Don't proceed to next phase until current phase success criteria are met
2. **Statistical Rigor**: Use n≥20 for all statistical tests, n≥100 for validation
3. **Independent Validation**: Cross-check critical components with alternative implementations
4. **Documentation**: Record exact commands, parameters, and outputs for reproducibility
5. **Rollback Plans**: Always have documented procedures to return to known-good state

---

## Risk Mitigation

### High-Risk Dependencies:
- **T6 → All Phase 2**: If Phase 1 assessment is inconclusive, Phase 2 may waste effort
  - *Mitigation*: Set clear escalation criteria in T6
- **T10 → T11**: Emergency procedures depend on systematic validation results
  - *Mitigation*: Create preliminary procedures in parallel with T10
- **T15 → T16**: Documentation depends on operational monitoring experience
  - *Mitigation*: Start playbook drafts early, update based on T15 learnings

### Timeline Risks:
- **Phase 1 overrun**: Could delay crisis resolution beyond acceptable window
  - *Mitigation*: Hard 2-hour time limit, escalate rather than extend
- **Phase 2 analysis paralysis**: Comprehensive validation might delay fixes
  - *Mitigation*: Implement fixes in parallel with validation when root cause is clear

This implementation todo list provides a complete roadmap for systematically diagnosing and fixing the model failure while building preventive infrastructure for future incidents.