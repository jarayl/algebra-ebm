#!/usr/bin/env python3
"""
Final test for enhanced calibration functionality using a rule that generates equations.
"""

import sys
sys.path.insert(0, 'src')

from algebra.algebra_dataset import AlgebraDataset

def test_dataset_with_equations():
    """Test dataset with rule that generates equations (with =)."""
    
    # Test with isolate rule which generates equations like "2*x + 3 = 7"
    print("Testing dataset with isolate rule...")
    dataset = AlgebraDataset(rule='isolate', num_problems=10)
    
    print("\nTesting get_equation_pair method:")
    for i in range(min(3, len(dataset))):
        inp_eq, out_eq = dataset.get_equation_pair(i)
        print(f"  {i}: {inp_eq} -> {out_eq}")
        
        assert isinstance(inp_eq, str)
        assert isinstance(out_eq, str)
        assert len(inp_eq) > 0
        assert len(out_eq) > 0
        # isolate rule should have = in both input and output
        assert '=' in inp_eq, f"Input equation should have '=': {inp_eq}"
        assert '=' in out_eq, f"Output equation should have '=': {out_eq}"
    
    print("✓ get_equation_pair method works correctly")
    
    print("\nTesting get_problem_info method:")
    for i in range(min(3, len(dataset))):
        problem_info = dataset.get_problem_info(i)
        print(f"  {i}: {problem_info}")
        
        assert isinstance(problem_info, dict)
        assert 'input_equation' in problem_info
        assert 'target_equation' in problem_info
        assert isinstance(problem_info['input_equation'], str)
        assert isinstance(problem_info['target_equation'], str)
    
    print("✓ get_problem_info method works correctly")
    
    print("\nAll dataset interface tests passed!")

if __name__ == "__main__":
    test_dataset_with_equations()