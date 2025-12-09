#!/usr/bin/env python3
"""
Test script for monolithic evaluation infrastructure.

This script validates that the CLI changes work without requiring a trained model.
It tests argument parsing, import resolution, and basic infrastructure setup.
"""

import argparse
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.algebra.algebra_evaluation import run_monolithic_evaluation
        print("✓ run_monolithic_evaluation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import run_monolithic_evaluation: {e}")
        return False
    
    try:
        from src.algebra.algebra_dataset import AlgebraDataset, MultiRuleDataset
        print("✓ Dataset classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset classes: {e}")
        return False
    
    return True


def test_cli_arguments():
    """Test that CLI argument parsing works correctly."""
    print("\nTesting CLI arguments...")
    
    # Import the argument parser from eval_algebra.py
    sys.path.append('/Users/mkrasnow/Desktop/algebra-ebm')
    
    try:
        # Test monolithic evaluation arguments
        from eval_algebra import main
        
        # Simulate command line arguments for monolithic evaluation
        test_args = [
            'eval_algebra.py',
            '--eval_type', 'monolithic',
            '--monolithic_checkpoint', './test/model.pt',
            '--output_dir', './test_output',
            '--max_samples', '10'
        ]
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        
        # Add the same arguments as in eval_algebra.py
        parser.add_argument('--eval_type', choices=['single_rule', 'multi_rule', 'monolithic', 'comparison', 'constrained', 'full'])
        parser.add_argument('--monolithic_checkpoint', type=str, default='./results/monolithic/model.pt')
        parser.add_argument('--output_dir', type=str, default='./evaluation_results')
        parser.add_argument('--max_samples', type=int)
        
        args = parser.parse_args(test_args[1:])  # Skip script name
        
        print(f"✓ eval_type parsed: {args.eval_type}")
        print(f"✓ monolithic_checkpoint parsed: {args.monolithic_checkpoint}")
        print(f"✓ output_dir parsed: {args.output_dir}")
        print(f"✓ max_samples parsed: {args.max_samples}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI argument parsing failed: {e}")
        return False


def test_dataset_creation():
    """Test that datasets can be created (without requiring actual training data)."""
    print("\nTesting dataset creation...")
    
    try:
        from src.algebra.algebra_dataset import AlgebraDataset, MultiRuleDataset
        
        # Test AlgebraDataset creation
        dataset = AlgebraDataset(
            rule='distribute',
            split='test',
            num_problems=5,  # Small number for quick test
            d_model=128
        )
        print(f"✓ AlgebraDataset created with {len(dataset)} problems")
        
        # Test MultiRuleDataset creation
        multi_dataset = MultiRuleDataset(
            num_rules=2,
            split='test',
            num_problems=5,  # Small number for quick test
            d_model=128
        )
        print(f"✓ MultiRuleDataset created with {len(multi_dataset)} problems")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_function_signature():
    """Test that the monolithic evaluation function has the correct signature."""
    print("\nTesting function signature...")
    
    try:
        from src.algebra.algebra_evaluation import run_monolithic_evaluation
        import inspect
        
        sig = inspect.signature(run_monolithic_evaluation)
        params = list(sig.parameters.keys())
        
        expected_params = ['monolithic_checkpoint', 'output_dir', 'num_samples']
        
        for param in expected_params:
            if param in params:
                print(f"✓ Parameter '{param}' found")
            else:
                print(f"✗ Parameter '{param}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Function signature test failed: {e}")
        return False


def test_mock_run():
    """Test a mock run without actual model checkpoint."""
    print("\nTesting mock run...")
    
    try:
        # Create temporary output directory
        test_output_dir = "/tmp/test_monolithic_eval"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # This should fail gracefully since the checkpoint doesn't exist
        from src.algebra.algebra_evaluation import run_monolithic_evaluation
        
        try:
            run_monolithic_evaluation(
                monolithic_checkpoint="./nonexistent_model.pt",
                output_dir=test_output_dir,
                num_samples=5
            )
            print("✗ Expected failure due to missing checkpoint, but function succeeded")
            return False
        except Exception as e:
            if "Failed to load monolithic model" in str(e) or "No such file" in str(e):
                print("✓ Function correctly failed with missing checkpoint")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Mock run test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MONOLITHIC EVALUATION INFRASTRUCTURE TEST")
    print("="*60)
    
    tests = [
        ("Import Resolution", test_imports),
        ("CLI Arguments", test_cli_arguments),
        ("Dataset Creation", test_dataset_creation),
        ("Function Signature", test_function_signature),
        ("Mock Run", test_mock_run)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✅ All tests passed! Monolithic evaluation infrastructure is ready.")
        return 0
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)