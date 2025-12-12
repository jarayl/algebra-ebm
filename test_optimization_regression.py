#!/usr/bin/env python3
"""
Comprehensive regression test for normalization optimizations.

Ensures that all optimizations maintain mathematical correctness across
various edge cases and configurations.
"""

import torch
import numpy as np
import pytest
from typing import Dict, List

from src.algebra.algebra_encoder import CharacterLevelEncoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig

# Import the optimized implementation
import sys
sys.path.append('.')
from optimized_compose_energies_demo import OptimizedAlgebraInference


def create_test_models(rule_names: List[str], device: str = 'cpu') -> Dict[str, AlgebraDiffusionWrapper]:
    """Create test models."""
    rule_models = {}
    for rule in rule_names:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.to(device)
        wrapper.eval()
        rule_models[rule] = wrapper
    return rule_models


class TestNormalizationOptimizations:
    """Comprehensive test suite for optimization correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test setup
        self.rule_names = ['distribute', 'combine', 'isolate', 'divide']
        self.rule_models = create_test_models(self.rule_names, self.device)
        self.encoder = CharacterLevelEncoder()
        self.config = InferenceConfig(K=5, max_iterations=10)
        
        # Create both implementations
        self.baseline = AlgebraInference(self.rule_models, self.encoder, config=self.config, device=self.device)
        self.optimized = OptimizedAlgebraInference(self.rule_models, self.encoder, config=self.config, device=self.device)
    
    def test_basic_equivalence(self):
        """Test basic mathematical equivalence."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        baseline_result = self.baseline.compose_energies(inp, out, k, normalize=True)
        optimized_result = self.optimized.compose_energies_optimized(inp, out, k, normalize=True, _skip_cache=True)
        
        max_diff = torch.abs(baseline_result - optimized_result).max().item()
        assert max_diff < 1e-6, f"Basic equivalence failed: max_diff={max_diff}"
        assert baseline_result.shape == optimized_result.shape
    
    def test_different_batch_sizes(self):
        """Test correctness across different batch sizes."""
        k = 1
        
        for batch_size in [1, 4, 16, 32, 64]:
            inp = torch.randn(batch_size, 128, device=self.device)
            out = torch.randn(batch_size, 128, device=self.device)
            
            baseline_result = self.baseline.compose_energies(inp, out, k, normalize=True)
            optimized_result = self.optimized.compose_energies_optimized(inp, out, k, normalize=True, _skip_cache=True)
            
            max_diff = torch.abs(baseline_result - optimized_result).max().item()
            assert max_diff < 1e-6, f"Batch size {batch_size} failed: max_diff={max_diff}"
    
    def test_custom_rule_weights(self):
        """Test with various rule weight configurations."""
        batch_size = 16
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        weight_configs = [
            {'distribute': 2.0, 'combine': 0.5, 'isolate': 1.5, 'divide': 0.8},
            {'distribute': 0.1, 'combine': 3.0},  # Partial weights
            {},  # Empty weights
            {rule: 1.0 for rule in self.rule_names},  # All 1.0
        ]
        
        for rule_weights in weight_configs:
            baseline_result = self.baseline.compose_energies(inp, out, k, rule_weights=rule_weights, normalize=True)
            optimized_result = self.optimized.compose_energies_optimized(inp, out, k, rule_weights=rule_weights, 
                                                                        normalize=True, _skip_cache=True)
            
            max_diff = torch.abs(baseline_result - optimized_result).max().item()
            assert max_diff < 1e-6, f"Rule weights {rule_weights} failed: max_diff={max_diff}"
    
    def test_no_normalization(self):
        """Test equivalence when normalization is disabled."""
        batch_size = 16
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        baseline_result = self.baseline.compose_energies(inp, out, k, normalize=False)
        optimized_result = self.optimized.compose_energies_optimized(inp, out, k, normalize=False, _skip_cache=True)
        
        max_diff = torch.abs(baseline_result - optimized_result).max().item()
        assert max_diff < 1e-6, f"No normalization failed: max_diff={max_diff}"
    
    def test_single_rule(self):
        """Test with single rule model."""
        single_rule_models = {'distribute': self.rule_models['distribute']}
        baseline_single = AlgebraInference(single_rule_models, self.encoder, config=self.config, device=self.device)
        optimized_single = OptimizedAlgebraInference(single_rule_models, self.encoder, config=self.config, device=self.device)
        
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        baseline_result = baseline_single.compose_energies(inp, out, k, normalize=True)
        optimized_result = optimized_single.compose_energies_optimized(inp, out, k, normalize=True, _skip_cache=True)
        
        max_diff = torch.abs(baseline_result - optimized_result).max().item()
        assert max_diff < 1e-6, f"Single rule failed: max_diff={max_diff}"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through optimized implementation."""
        batch_size = 4
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
        k = 2
        
        # Test baseline
        baseline_energy = self.baseline.compose_energies(inp, out, k, normalize=True)
        baseline_loss = baseline_energy.sum()
        baseline_loss.backward()
        baseline_grad = out.grad.clone()
        out.grad.zero_()
        
        # Test optimized
        optimized_energy = self.optimized.compose_energies_optimized(inp, out, k, normalize=True, _skip_cache=True)
        optimized_loss = optimized_energy.sum()
        optimized_loss.backward()
        optimized_grad = out.grad.clone()
        
        # Compare gradients
        grad_diff = torch.abs(baseline_grad - optimized_grad).max().item()
        assert grad_diff < 1e-6, f"Gradient flow failed: max_grad_diff={grad_diff}"
    
    def test_caching_consistency(self):
        """Test that caching doesn't affect results."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        # First call (cache miss)
        result_1 = self.optimized.compose_energies_optimized(inp, out, k, normalize=True)
        
        # Second call (cache hit)
        result_2 = self.optimized.compose_energies_optimized(inp, out, k, normalize=True)
        
        # Third call (no cache)
        result_3 = self.optimized.compose_energies_optimized(inp, out, k, normalize=True, _skip_cache=True)
        
        # All should be identical
        diff_1_2 = torch.abs(result_1 - result_2).max().item()
        diff_1_3 = torch.abs(result_1 - result_3).max().item()
        
        assert diff_1_2 < 1e-7, f"Cache consistency failed: cached vs uncached diff={diff_1_2}"
        assert diff_1_3 < 1e-7, f"Cache vs no-cache failed: diff={diff_1_3}"
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        batch_size = 8
        k = 2
        
        # Test with very small values
        inp_small = torch.full((batch_size, 128), 1e-6, device=self.device)
        out_small = torch.full((batch_size, 128), 1e-6, device=self.device)
        
        baseline_small = self.baseline.compose_energies(inp_small, out_small, k, normalize=True)
        optimized_small = self.optimized.compose_energies_optimized(inp_small, out_small, k, normalize=True, _skip_cache=True)
        
        assert torch.isfinite(baseline_small).all(), "Baseline should produce finite values"
        assert torch.isfinite(optimized_small).all(), "Optimized should produce finite values"
        
        # Test with large values  
        inp_large = torch.full((batch_size, 128), 100.0, device=self.device)
        out_large = torch.full((batch_size, 128), 100.0, device=self.device)
        
        baseline_large = self.baseline.compose_energies(inp_large, out_large, k, normalize=True)
        optimized_large = self.optimized.compose_energies_optimized(inp_large, out_large, k, normalize=True, _skip_cache=True)
        
        assert torch.isfinite(baseline_large).all(), "Large values should be handled"
        assert torch.isfinite(optimized_large).all(), "Large values should be handled"
    
    def test_calibration_scales(self):
        """Test correctness with calibration scales."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 1
        
        calibration_scales = {
            'distribute': 1.5,
            'combine': 0.8,
            'isolate': 2.0
        }
        
        baseline_result = self.baseline.compose_energies(inp, out, k, calibration_scales=calibration_scales, normalize=True)
        optimized_result = self.optimized.compose_energies_optimized(inp, out, k, calibration_scales=calibration_scales, 
                                                                    normalize=True, _skip_cache=True)
        
        max_diff = torch.abs(baseline_result - optimized_result).max().item()
        assert max_diff < 1e-6, f"Calibration scales failed: max_diff={max_diff}"


def run_performance_test():
    """Run performance regression test."""
    print("\n" + "="*60)
    print("PERFORMANCE REGRESSION TEST")
    print("="*60)
    
    device = 'cpu'
    rule_names = ['distribute', 'combine', 'isolate', 'divide']
    rule_models = create_test_models(rule_names, device)
    encoder = CharacterLevelEncoder()
    config = InferenceConfig(K=5, max_iterations=10)
    
    baseline = AlgebraInference(rule_models, encoder, config=config, device=device)
    optimized = OptimizedAlgebraInference(rule_models, encoder, config=config, device=device)
    
    # Performance test
    batch_size = 32
    inp = torch.randn(batch_size, 128, device=device)
    out = torch.randn(batch_size, 128, device=device)
    k = 2
    iterations = 200
    
    import time
    
    # Baseline
    start_time = time.perf_counter()
    for _ in range(iterations):
        baseline_result = baseline.compose_energies(inp, out, k, normalize=True)
    baseline_time = time.perf_counter() - start_time
    
    # Optimized
    start_time = time.perf_counter()
    for _ in range(iterations):
        optimized_result = optimized.compose_energies_optimized(inp, out, k, normalize=True)
    optimized_time = time.perf_counter() - start_time
    
    speedup = baseline_time / optimized_time
    
    # Verify correctness
    max_diff = torch.abs(baseline_result - optimized_result).max().item()
    
    print(f"Baseline time:    {baseline_time*1000:.2f} ms")
    print(f"Optimized time:   {optimized_time*1000:.2f} ms")
    print(f"Speedup:          {speedup:.2f}x ({(speedup-1)*100:.1f}%)")
    print(f"Max difference:   {max_diff:.2e}")
    print(f"Correctness:      {'✅ PASS' if max_diff < 1e-6 else '❌ FAIL'}")
    
    assert speedup >= 0.95, f"Performance regression detected: {speedup:.2f}x"
    assert max_diff < 1e-6, f"Correctness regression detected: {max_diff:.2e}"
    
    print("✅ Performance regression test PASSED")
    
    return {'speedup': speedup, 'correctness': max_diff < 1e-6}


if __name__ == "__main__":
    print("Running normalization optimization regression tests...")
    
    # Run correctness tests
    test_suite = TestNormalizationOptimizations()
    test_suite.setup_method()
    
    test_methods = [
        ('Basic equivalence', test_suite.test_basic_equivalence),
        ('Different batch sizes', test_suite.test_different_batch_sizes),
        ('Custom rule weights', test_suite.test_custom_rule_weights),
        ('No normalization', test_suite.test_no_normalization),
        ('Single rule', test_suite.test_single_rule),
        ('Gradient flow', test_suite.test_gradient_flow),
        ('Caching consistency', test_suite.test_caching_consistency),
        ('Numerical stability', test_suite.test_numerical_stability),
        ('Calibration scales', test_suite.test_calibration_scales),
    ]
    
    passed = 0
    failed = 0
    
    print("\nCorrectness Tests:")
    print("-" * 40)
    
    for test_name, test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed += 1
    
    print(f"\nCorrectness Results: {passed} passed, {failed} failed")
    
    # Run performance test
    performance_result = run_performance_test()
    
    if failed == 0 and performance_result['correctness'] and performance_result['speedup'] >= 0.95:
        print("\n🎉 ALL TESTS PASSED - Optimizations are correct and performant!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Review the implementation")
        exit(1)