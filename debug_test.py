#!/usr/bin/env python3
"""
Debug script to reproduce the tuple unpacking error
"""

import sys
sys.path.append('.')

from algebra_encoder import validate_equation_syntax

# Test the function that's causing the error
try:
    result = validate_equation_syntax("2*x+3=7")
    print(f"validate_equation_syntax returned: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    
    # Try unpacking like in the code
    is_valid, error_msg, parsed_expr = result
    print(f"Successfully unpacked: is_valid={is_valid}, error_msg={error_msg}, parsed_expr={parsed_expr}")
    
except Exception as e:
    print(f"Error in validate_equation_syntax: {e}")
    import traceback
    traceback.print_exc()

# Test a potentially problematic case
print("\nTesting constrained dataset creation...")
try:
    from algebra_dataset import ConstrainedDataset
    dataset = ConstrainedDataset(
        num_rules=2,
        constraints=['positive'],
        split='test',
        num_problems=10,
        d_model=128,
        seed=42
    )
    
    # Try getting problem info
    problem_info = dataset.get_problem_info(0)
    print(f"Problem info keys: {problem_info.keys()}")
    print(f"Problem info: {problem_info}")
    
except Exception as e:
    print(f"Error in constrained dataset: {e}")
    import traceback
    traceback.print_exc()