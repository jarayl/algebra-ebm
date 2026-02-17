#!/usr/bin/env python3
"""
Validate training data - check for mislabeled or corrupted equation pairs.
"""
import sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).parent))

from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_encoder import check_equation_equivalence

def validate_pair(input_eq, target_eq, rule):
    """Validate that a pair represents correct rule application."""
    # Check equivalence
    try:
        is_equiv, _ = check_equation_equivalence(input_eq, target_eq)
    except:
        return False, "equivalence_check_failed"
    
    if not is_equiv:
        return False, "not_equivalent"
    
    # Check rule-specific patterns
    if rule == 'distribute':
        # Input should have pattern: a*(b*x + c) = d
        # Target should have pattern: a*b*x + a*c = d
        if '*(' not in input_eq or '+' not in input_eq:
            return False, "wrong_input_format"
        if '*(' in target_eq:
            return False, "wrong_target_format"
    
    return True, "valid"

def main():
    print("="*80)
    print("TRAINING DATA VALIDATION")
    print("="*80)
    
    rule = 'distribute'
    
    # Sample from training set
    print(f"\nLoading training dataset for '{rule}'...")
    dataset = AlgebraDataset(rule=rule, split='train', num_problems=1000, d_model=128)
    
    # Random sample
    sample_size = 100
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    
    print(f"Validating {sample_size} random training examples...\n")
    
    valid_count = 0
    invalid_count = 0
    error_types = {}
    invalid_examples = []
    
    for i in indices:
        input_eq, target_eq = dataset.get_equation_pair(i)
        is_valid, error_type = validate_pair(input_eq, target_eq, rule)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if len(invalid_examples) < 10:
                invalid_examples.append((input_eq, target_eq, error_type))
    
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Valid pairs:   {valid_count}/{sample_size} ({valid_count/sample_size*100:.1f}%)")
    print(f"Invalid pairs: {invalid_count}/{sample_size} ({invalid_count/sample_size*100:.1f}%)")
    
    if error_types:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    
    if invalid_examples:
        print(f"\n{'='*80}")
        print("INVALID EXAMPLES:")
        print("="*80)
        for i, (inp, tgt, err) in enumerate(invalid_examples):
            print(f"{i+1}. Error: {err}")
            print(f"   Input:  {inp}")
            print(f"   Target: {tgt}")
            print()
    
    print("="*80)
    print("CONCLUSION:")
    print("="*80)
    if invalid_count == 0:
        print("✓ All sampled training data is valid - no data corruption detected")
    else:
        print(f"✗ Found {invalid_count} invalid pairs - DATA CORRUPTION LIKELY")
        print("  This could cause inconsistent energy learning!")

if __name__ == "__main__":
    main()
