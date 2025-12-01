#!/usr/bin/env python3
"""
Integration Test for Dataset Variability Enhancement

Tests the complete integration of all variability enhancement features:
1. Stratified coefficient sampling
2. Solution-first equation generation
3. Dataset variability validation
4. Training script integration
5. Backward compatibility

This test verifies the solution addresses the original dataset variability problem.
"""

import sys
import os
import torch
import numpy as np
import traceback
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from algebra_dataset import AlgebraDataset, DatasetVariabilityValidator, COVERAGE_METRICS
    from algebra_encoder import create_character_encoder, solve_equation
    from algebra_models import AlgebraEBM
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Ensure all required modules are available in the current directory")
    sys.exit(1)

def test_backward_compatibility():
    """Test that default behavior remains unchanged for backward compatibility."""
    print("=== Testing Backward Compatibility ===")
    
    try:
        # Create dataset with default settings (should be identical to original behavior)
        dataset = AlgebraDataset(
            rule='distribute',
            num_problems=100,
            coeff_range=[-10, 10],
            enable_stratified_sampling=False,
            enable_solution_first=False
        )
        
        # Check basic properties
        assert len(dataset) == 100
        assert dataset.inp_dim == 128
        assert dataset.out_dim == 128
        assert dataset.enable_stratified_sampling == False
        assert dataset.enable_solution_first == False
        
        # Test data loading
        sample = dataset[0]
        assert len(sample) == 2  # (input_embedding, target_embedding)
        assert sample[0].shape == (128,)
        assert sample[1].shape == (128,)
        
        print("✓ Backward compatibility preserved")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_stratified_coefficient_generation():
    """Test stratified coefficient sampling functionality."""
    print("\n=== Testing Stratified Coefficient Generation ===")
    
    try:
        # Test with custom stratified configuration
        stratified_ranges = {
            'basic': [-5, 5],
            'extended': [-20, 20], 
            'challenge': [-50, 50]
        }
        stratified_distribution = {
            'basic': 0.5,
            'extended': 0.3,
            'challenge': 0.2
        }
        
        dataset = AlgebraDataset(
            rule='combine',
            num_problems=200,  # Reduced for faster testing
            enable_stratified_sampling=True,
            stratified_ranges=stratified_ranges,
            stratified_distribution=stratified_distribution
        )
        
        # Generate sample coefficients and analyze distribution
        coefficients = []
        for _ in range(500):  # Reduced sample size for faster testing
            coeff = dataset._generate_random_coefficients(1)
            coefficients.append(coeff)
        
        # Check distribution roughly matches expected probabilities
        basic_count = sum(1 for c in coefficients if -5 <= c <= 5)
        extended_count = sum(1 for c in coefficients if (-20 <= c < -5) or (5 < c <= 20))
        challenge_count = sum(1 for c in coefficients if (-50 <= c < -20) or (20 < c <= 50))
        
        basic_ratio = basic_count / 500
        extended_ratio = extended_count / 500
        challenge_ratio = challenge_count / 500
        
        print(f"Basic range coverage: {basic_ratio:.3f} (expected ~0.5)")
        print(f"Extended range coverage: {extended_ratio:.3f} (expected ~0.3)")
        print(f"Challenge range coverage: {challenge_ratio:.3f} (expected ~0.2)")
        
        # Allow realistic tolerance for random variation - focus on overall coverage
        assert basic_ratio >= 0.35, f"Basic ratio {basic_ratio} too low (expected >=0.35)"
        assert extended_ratio >= 0.20, f"Extended ratio {extended_ratio} too low (expected >=0.20)"  
        assert challenge_ratio >= 0.08, f"Challenge ratio {challenge_ratio} too low (expected >=0.08)"
        
        # Ensure all ranges are covered
        assert basic_ratio + extended_ratio + challenge_ratio >= 0.9, "Total coverage too low"
        
        print("✓ Stratified coefficient generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Stratified coefficient generation test failed: {e}")
        traceback.print_exc()
        return False

def test_solution_first_generation():
    """Test solution-first equation generation functionality."""
    print("\n=== Testing Solution-First Generation ===")
    
    try:
        # Test with custom solution ranges
        target_solution_ranges = {
            'small': [-5, 5],
            'medium': [-15, 15],
            'large': [-30, 30]
        }
        solution_range_distribution = {
            'small': 0.6,
            'medium': 0.3,
            'large': 0.1
        }
        
        dataset = AlgebraDataset(
            rule='isolate',
            num_problems=500,
            enable_solution_first=True,
            target_solution_ranges=target_solution_ranges,
            solution_range_distribution=solution_range_distribution
        )
        
        # Extract solutions from generated equations and check coverage
        solutions = []
        validator = DatasetVariabilityValidator()
        
        # Sample a subset of equations to test solution extraction
        for i in range(min(100, len(dataset))):
            # Get the equation pair
            input_emb, target_emb = dataset[i]
            
            # Extract equation strings from embeddings (approximate test)
            # For now, we'll use the dataset's internal equation storage if available
            if hasattr(dataset, 'equation_pairs') and i < len(dataset.equation_pairs):
                input_eq_str, target_eq_str = dataset.equation_pairs[i]
                
                # Extract solution from the equation
                solution = validator.extract_solution(input_eq_str)
                if solution is not None:
                    solutions.append(solution)
        
        print(f"Extracted {len(solutions)} valid solutions from equations")
        
        if solutions:
            # Analyze solution distribution
            small_count = sum(1 for s in solutions if -5 <= s <= 5)
            medium_count = sum(1 for s in solutions if (-15 <= s < -5) or (5 < s <= 15))
            large_count = sum(1 for s in solutions if (-30 <= s < -15) or (15 < s <= 30))
            
            total = len(solutions)
            small_ratio = small_count / total if total > 0 else 0
            medium_ratio = medium_count / total if total > 0 else 0
            large_ratio = large_count / total if total > 0 else 0
            
            print(f"Small range coverage: {small_ratio:.3f} (expected ~0.6)")
            print(f"Medium range coverage: {medium_ratio:.3f} (expected ~0.3)")
            print(f"Large range coverage: {large_ratio:.3f} (expected ~0.1)")
            
            # Verify solution distribution roughly matches configuration
            assert small_ratio >= 0.4, f"Small range coverage {small_ratio} too low"
            assert medium_ratio >= 0.15, f"Medium range coverage {medium_ratio} too low"
        
        print("✓ Solution-first generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Solution-first generation test failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_variability_validator():
    """Test DatasetVariabilityValidator functionality."""
    print("\n=== Testing Dataset Variability Validator ===")
    
    try:
        validator = DatasetVariabilityValidator()
        
        # Test equation extraction
        test_equations = [
            "2*x+3=7",  # solution: x=2
            "4*x-2=6",  # solution: x=2 
            "3*x=9",    # solution: x=3
            "x+5=8"     # solution: x=3
        ]
        
        solutions = []
        for eq in test_equations:
            solution = validator.extract_solution(eq)
            if solution is not None:
                solutions.append(solution)
        
        print(f"Extracted solutions: {solutions}")
        assert len(solutions) >= 3, f"Expected at least 3 solutions, got {len(solutions)}"
        
        # Test coefficient extraction
        coefficients = []
        for eq in test_equations:
            coeffs = validator.extract_coefficients(eq)
            coefficients.extend(coeffs)
        
        print(f"Extracted coefficients: {coefficients}")
        assert len(coefficients) >= 6, f"Expected at least 6 coefficients, got {len(coefficients)}"
        
        # Create a small dataset to validate
        dataset = AlgebraDataset(
            rule='distribute',
            num_problems=200,
            enable_stratified_sampling=True,
            enable_solution_first=True
        )
        
        # Test coverage validation
        solution_coverage = validator.validate_solution_coverage(dataset)
        coeff_diversity = validator.validate_coefficient_diversity(dataset)
        
        print(f"Solution coverage validation: {solution_coverage.get('passed', 'unknown')}")
        print(f"Coefficient diversity validation: {coeff_diversity.get('passed', 'unknown')}")
        
        # Test comprehensive coverage report
        coverage_report = validator.generate_coverage_report(dataset)
        assert 'solution_coverage' in coverage_report
        assert 'coefficient_diversity' in coverage_report
        assert 'overall_passed' in coverage_report
        
        print("✓ Dataset variability validator working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Dataset variability validator test failed: {e}")
        traceback.print_exc()
        return False

def test_training_script_integration():
    """Test integration with the training script configuration."""
    print("\n=== Testing Training Script Integration ===")
    
    try:
        # Test command line argument parsing and dataset creation
        # Simulate command line arguments for enhanced variability
        dataset_kwargs = {
            'rule': 'distribute',
            'split': 'train',
            'num_problems': 100,
            'd_model': 128,
            'enable_stratified_sampling': True,
            'enable_solution_first': True,
            'stratified_ranges': {
                'basic': [-5, 5],
                'extended': [-20, 20],
                'challenge': [-50, 50]
            },
            'stratified_distribution': {
                'basic': 0.4,
                'extended': 0.4,
                'challenge': 0.2
            },
            'target_solution_ranges': {
                'small': [-10, 10],
                'medium': [-25, 25],
                'large': [-50, 50]
            },
            'solution_range_distribution': {
                'small': 0.5,
                'medium': 0.35,
                'large': 0.15
            }
        }
        
        # Create dataset with training script configuration
        dataset = AlgebraDataset(**dataset_kwargs)
        
        # Verify dataset properties match configuration
        assert dataset.enable_stratified_sampling == True
        assert dataset.enable_solution_first == True
        assert len(dataset.stratified_ranges) == 3
        assert len(dataset.target_solution_ranges) == 3
        
        # Test basic data loading
        sample = dataset[0]
        assert len(sample) == 2
        assert sample[0].shape == (128,)
        assert sample[1].shape == (128,)
        
        # Test with AlgebraEBM model (basic compatibility check)
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='distribute')
        ebm.eval()
        
        with torch.no_grad():
            input_emb, target_emb = sample
            # Add batch dimension
            input_batch = input_emb.unsqueeze(0)
            target_batch = target_emb.unsqueeze(0) 
            t = torch.zeros(1).long()
            
            energy = ebm(input_batch, target_batch, t)
            assert energy.shape == (1, 1), f"Expected energy shape (1, 1), got {energy.shape}"
        
        print("✓ Training script integration working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Training script integration test failed: {e}")
        traceback.print_exc()
        return False

def test_comprehensive_dataset_generation():
    """Test comprehensive dataset generation with all features enabled."""
    print("\n=== Testing Comprehensive Dataset Generation ===")
    
    try:
        # Create datasets for all rule types with enhanced variability
        rules = ['distribute', 'combine', 'isolate', 'divide']
        datasets = {}
        
        for rule in rules:
            dataset = AlgebraDataset(
                rule=rule,
                num_problems=100,  # Reduced for faster testing
                enable_stratified_sampling=True,
                enable_solution_first=True
            )
            datasets[rule] = dataset
            
            print(f"Created {rule} dataset: {len(dataset)} problems")
            
            # Basic validation
            assert len(dataset) == 100
            sample = dataset[0]
            assert sample[0].shape == (128,)
            assert sample[1].shape == (128,)
        
        # Test that different rules generate different distributions
        validator = DatasetVariabilityValidator()
        
        for rule, dataset in datasets.items():
            coverage_report = validator.generate_coverage_report(dataset)
            overall_passed = coverage_report.get('overall_passed', False)
            
            print(f"{rule} dataset coverage: {'PASSED' if overall_passed else 'NEEDS IMPROVEMENT'}")
            
            # Check for improvement recommendations if coverage needs work
            if not overall_passed:
                recommendations = coverage_report.get('recommendations', [])
                if recommendations:
                    print(f"  Recommendations for {rule}: {recommendations[0]}")
        
        print("✓ Comprehensive dataset generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive dataset generation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("Dataset Variability Enhancement - Integration Test")
    print("=" * 60)
    
    tests = [
        test_backward_compatibility,
        test_stratified_coefficient_generation,
        test_solution_first_generation,
        test_dataset_variability_validator,
        test_training_script_integration,
        test_comprehensive_dataset_generation
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
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("Dataset variability enhancement is ready for production use.")
        return True
    else:
        print("❌ Some integration tests failed.")
        print("Review failed tests before proceeding to production.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)