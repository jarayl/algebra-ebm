#!/usr/bin/env python3
"""
Safety and Regression Testing for Dataset Variability Enhancement

Tests that the enhanced dataset generation doesn't introduce regressions
and maintains compatibility with existing training workflows.

Covers:
1. API compatibility 
2. Default behavior preservation
3. Performance regression checks
4. Edge case handling
5. Security validation
"""

import sys
import os
import time
import torch
import traceback
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from algebra_dataset import AlgebraDataset, DatasetVariabilityValidator
    from algebra_encoder import create_character_encoder
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

def test_api_compatibility():
    """Test that all existing APIs remain functional and unchanged."""
    print("=== Testing API Compatibility ===")
    
    try:
        # Test original constructor signature still works
        dataset = AlgebraDataset('distribute', 'train', 100)
        assert len(dataset) == 100
        assert dataset.rule == 'distribute'
        assert dataset.split == 'train'
        
        # Test dataset interface properties
        assert hasattr(dataset, 'inp_dim')
        assert hasattr(dataset, 'out_dim')
        assert dataset.inp_dim == 128
        assert dataset.out_dim == 128
        
        # Test __getitem__ returns correct format
        sample = dataset[0]
        assert len(sample) == 2
        assert isinstance(sample[0], torch.Tensor)
        assert isinstance(sample[1], torch.Tensor)
        assert sample[0].shape == (128,)
        assert sample[1].shape == (128,)
        
        # Test all rule types still work
        for rule in ['distribute', 'combine', 'isolate', 'divide']:
            rule_dataset = AlgebraDataset(rule, num_problems=10)
            assert len(rule_dataset) == 10
            sample = rule_dataset[0]
            assert sample[0].shape == (128,)
        
        print("✓ API compatibility preserved")
        return True
        
    except Exception as e:
        print(f"✗ API compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_default_behavior_preservation():
    """Test that default behavior is identical to original implementation."""
    print("\n=== Testing Default Behavior Preservation ===")
    
    try:
        # Create dataset with explicit defaults matching original behavior
        dataset1 = AlgebraDataset(
            rule='combine',
            num_problems=50,
            enable_stratified_sampling=False,
            enable_solution_first=False
        )
        
        # Create dataset using original-style constructor
        dataset2 = AlgebraDataset('combine', num_problems=50)
        
        # Both should have identical configuration
        assert dataset1.enable_stratified_sampling == dataset2.enable_stratified_sampling == False
        assert dataset1.enable_solution_first == dataset2.enable_solution_first == False
        assert dataset1.coeff_range == dataset2.coeff_range == [-10, 10]
        assert len(dataset1) == len(dataset2) == 50
        
        # Test that coefficient generation uses original range
        coeffs = []
        for _ in range(100):
            coeff = dataset1._generate_random_coefficients(1)
            coeffs.append(coeff)
        
        # All coefficients should be within original range
        assert all(-10 <= c <= 10 for c in coeffs), "Coefficients outside original range"
        
        # Test that no adaptive generation is triggered
        assert dataset1.enable_adaptive_generation == False
        assert dataset1.validator is None
        
        print("✓ Default behavior preserved")
        return True
        
    except Exception as e:
        print(f"✗ Default behavior test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_regression():
    """Test that enhanced features don't cause significant performance regression."""
    print("\n=== Testing Performance Regression ===")
    
    try:
        # Benchmark original dataset creation (reduced size for quick testing)
        start_time = time.time()
        original_dataset = AlgebraDataset(
            'distribute', 
            num_problems=100,
            enable_stratified_sampling=False,
            enable_solution_first=False
        )
        original_time = time.time() - start_time
        
        # Benchmark enhanced dataset creation
        start_time = time.time()
        enhanced_dataset = AlgebraDataset(
            'distribute',
            num_problems=100, 
            enable_stratified_sampling=True,
            enable_solution_first=True
        )
        enhanced_time = time.time() - start_time
        
        # Performance overhead should be reasonable (< 50% increase)
        performance_overhead = (enhanced_time - original_time) / original_time
        
        print(f"Original dataset creation: {original_time:.3f}s")
        print(f"Enhanced dataset creation: {enhanced_time:.3f}s")  
        print(f"Performance overhead: {performance_overhead:.1%}")
        
        assert performance_overhead < 0.5, f"Performance regression too high: {performance_overhead:.1%}"
        
        # Test data loading performance
        start_time = time.time()
        for i in range(min(100, len(original_dataset))):
            _ = original_dataset[i]
        original_load_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(min(100, len(enhanced_dataset))):
            _ = enhanced_dataset[i]
        enhanced_load_time = time.time() - start_time
        
        load_overhead = (enhanced_load_time - original_load_time) / original_load_time
        print(f"Data loading overhead: {load_overhead:.1%}")
        
        assert load_overhead < 0.2, f"Data loading regression too high: {load_overhead:.1%}"
        
        print("✓ Performance regression within acceptable limits")
        return True
        
    except Exception as e:
        print(f"✗ Performance regression test failed: {e}")
        traceback.print_exc()
        return False

def test_edge_case_handling():
    """Test edge cases and error handling robustness."""
    print("\n=== Testing Edge Case Handling ===")
    
    try:
        # Test invalid rule types
        try:
            AlgebraDataset('invalid_rule')
            assert False, "Should have raised ValueError for invalid rule"
        except ValueError:
            pass
        
        # Test invalid ranges
        try:
            AlgebraDataset('distribute', stratified_ranges={'invalid': [5, 2]})  # min > max
            assert False, "Should have raised ValueError for invalid range"
        except ValueError:
            pass
        
        # Test invalid distributions
        try:
            AlgebraDataset('distribute', 
                         enable_stratified_sampling=True,
                         stratified_distribution={'basic': 0.5, 'extended': 0.7})  # sum > 1
            assert False, "Should have raised ValueError for invalid distribution"
        except ValueError:
            pass
        
        # Test empty datasets
        tiny_dataset = AlgebraDataset('distribute', num_problems=1)
        assert len(tiny_dataset) == 1
        sample = tiny_dataset[0]
        assert sample[0].shape == (128,)
        
        # Test validator edge cases
        validator = DatasetVariabilityValidator()
        
        # Test with empty/invalid equations
        assert validator.extract_solution("") is None
        assert validator.extract_solution("invalid equation") is None
        assert validator.extract_coefficients("") == []
        
        # Test with tuple inputs
        assert validator.extract_solution(("2*x=4", "x=2")) == 2.0
        assert len(validator.extract_coefficients(("3*x+5=8", "3*x=3"))) > 0
        
        print("✓ Edge case handling robust")
        return True
        
    except Exception as e:
        print(f"✗ Edge case handling test failed: {e}")
        traceback.print_exc()
        return False

def test_security_validation():
    """Test that security vulnerabilities have been addressed."""
    print("\n=== Testing Security Validation ===")
    
    try:
        validator = DatasetVariabilityValidator()
        
        # Test input length limits (prevents ReDoS)
        long_input = "x=" + "1" * 2000  # Very long equation
        result = validator.extract_solution(long_input)
        assert result is None, "Should reject overly long inputs"
        
        # Test coefficient extraction length limits
        coeffs = validator.extract_coefficients("x=" + "1" * 1000)
        assert len(coeffs) == 0, "Should reject overly long coefficient inputs"
        
        # Test that dangerous patterns would be blocked by algebra_encoder validation
        dangerous_patterns = [
            "__import__('os')",
            "exec('rm -rf /')",
            "eval('malicious code')"
        ]
        
        for pattern in dangerous_patterns:
            # These should be safely handled by the equation validation
            result = validator.extract_solution(pattern)
            assert result is None, f"Dangerous pattern should be rejected: {pattern}"
        
        # Test tuple handling doesn't introduce vulnerabilities
        malicious_tuple = ("safe_eq", "__import__('os')")
        result = validator.extract_solution(malicious_tuple)
        # Should extract from first element only, second element ignored
        assert result is None or isinstance(result, (int, float))
        
        print("✓ Security validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Security validation test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_and_resource_safety():
    """Test memory usage and resource management."""
    print("\n=== Testing Memory and Resource Safety ===")
    
    try:
        # Test that coverage history is properly limited
        dataset = AlgebraDataset(
            'distribute',
            num_problems=50,  # Small dataset for quick test
            enable_stratified_sampling=True,
            enable_solution_first=True
        )
        
        # Coverage history should be limited
        assert hasattr(dataset, '_max_history_size')
        assert dataset._max_history_size == 100
        assert len(dataset._coverage_history) <= dataset._max_history_size
        
        # Test that larger datasets don't cause memory issues
        # (This creates the dataset but doesn't load all data into memory)
        large_dataset = AlgebraDataset('combine', num_problems=1000)
        assert len(large_dataset) == 1000
        
        # Should be able to access any index without loading everything
        sample_first = large_dataset[0]
        sample_middle = large_dataset[500] 
        sample_last = large_dataset[999]
        
        assert all(s[0].shape == (128,) for s in [sample_first, sample_middle, sample_last])
        
        # Test thread safety primitives exist
        assert hasattr(dataset, '_adjustment_lock')
        
        print("✓ Memory and resource safety validated")
        return True
        
    except Exception as e:
        print(f"✗ Memory safety test failed: {e}")
        traceback.print_exc()
        return False

def test_training_compatibility():
    """Test compatibility with existing training scripts and workflows."""
    print("\n=== Testing Training Compatibility ===")
    
    try:
        # Test dataset creation with training script parameters
        training_params = {
            'rule': 'isolate',
            'split': 'train', 
            'num_problems': 100,
            'd_model': 128,
            'enable_stratified_sampling': True,
            'stratified_ranges': {
                'basic': [-5, 5],
                'extended': [-20, 20],
                'challenge': [-50, 50]
            },
            'stratified_distribution': {
                'basic': 0.4,
                'extended': 0.4,
                'challenge': 0.2
            }
        }
        
        dataset = AlgebraDataset(**training_params)
        
        # Test PyTorch DataLoader compatibility
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))
        
        assert len(batch) == 2  # input and target
        assert batch[0].shape == (4, 128)  # batch of inputs
        assert batch[1].shape == (4, 128)  # batch of targets
        
        # Test that encoder compatibility is maintained
        encoder = create_character_encoder(d_model=128)
        sample_input, sample_target = dataset[0]
        
        # Encoder should be compatible with dataset outputs
        assert sample_input.shape == (encoder.d_model,)
        assert sample_target.shape == (encoder.d_model,)
        
        print("✓ Training compatibility maintained")
        return True
        
    except Exception as e:
        print(f"✗ Training compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all safety and regression tests."""
    print("Dataset Variability Enhancement - Safety & Regression Test")
    print("=" * 70)
    
    tests = [
        test_api_compatibility,
        test_default_behavior_preservation,
        test_performance_regression,
        test_edge_case_handling,
        test_security_validation,
        test_memory_and_resource_safety,
        test_training_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Safety & Regression Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🔒 ALL SAFETY TESTS PASSED!")
        print("Dataset variability enhancement is safe for production deployment.")
        return True
    else:
        print("⚠️  Some safety tests failed.")
        print("Review failed tests before deploying to production.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)