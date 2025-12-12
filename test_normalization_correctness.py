#!/usr/bin/env python3
"""
Correctness test for normalization optimization.

Validates that the optimized compose_energies implementation produces
mathematically equivalent results to the baseline implementation.
"""

import torch
import pytest
import numpy as np
from typing import Dict, List
import logging

from src.algebra.algebra_encoder import CharacterLevelEncoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig

# Disable logging for cleaner test output
logging.basicConfig(level=logging.CRITICAL)


def create_baseline_compose_energies(inference: AlgebraInference):
    """Create baseline implementation for comparison."""
    
    def baseline_compose_energies(
        inp: torch.Tensor,
        out: torch.Tensor, 
        k: int,
        rule_weights=None,
        t=None,
        normalize: bool = True,
        calibration_scales=None
    ):
        """Baseline implementation matching original logic."""
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in inference.rule_models.keys()}
        
        if t is None:
            t = torch.full((inp.shape[0],), k, dtype=torch.long, device=inp.device)
        
        # Collect individual energies in original order
        individual_energies = []
        rule_names = list(inference.rule_models.keys())
        
        for rule_name in rule_names:
            model = inference.rule_models[rule_name]
            energy = model(inp, out, t, return_energy=True)
            
            if calibration_scales is not None and rule_name in calibration_scales:
                energy = energy * calibration_scales[rule_name]
            
            individual_energies.append(energy)
        
        if not normalize:
            # Original weighted summation
            total_energy = 0.0
            for rule_name, energy in zip(rule_names, individual_energies):
                weight = rule_weights.get(rule_name, 1.0)
                total_energy += weight * energy
            return total_energy
        
        # Single rule case
        if len(individual_energies) == 1:
            weight = rule_weights.get(rule_names[0], 1.0)
            return weight * individual_energies[0]
        
        # Normalization (original implementation)
        energy_tensor = torch.stack(individual_energies, dim=-1)
        mean = energy_tensor.mean(dim=-1, keepdim=True)
        std = energy_tensor.std(dim=-1, keepdim=True, unbiased=False)
        
        epsilon = 1e-6
        std_safe = std + epsilon
        
        normalized_energies = (energy_tensor - mean) / std_safe
        
        # Re-scale to [1.0, 15.0]
        target_min, target_max = 1.0, 15.0
        target_scale = (target_max - target_min) / 4.0
        target_offset = (target_min + target_max) / 2.0
        
        rescaled_energies = target_scale * normalized_energies + target_offset
        
        # Original loop-based weight application
        total_energy = torch.zeros(inp.shape[0], 1, device=inp.device, dtype=inp.dtype)
        for i, rule_name in enumerate(rule_names):
            weight = rule_weights.get(rule_name, 1.0)
            total_energy += weight * rescaled_energies[:, :, i]
        
        return total_energy
    
    return baseline_compose_energies


class TestNormalizationCorrectness:
    """Test suite for normalization optimization correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create test models
        self.rule_names = ['distribute', 'combine', 'isolate', 'divide']
        self.rule_models = {}
        for rule in self.rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.to(self.device)
            wrapper.eval()
            self.rule_models[rule] = wrapper
        
        # Create inference engine
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5, max_iterations=10)
        self.inference = AlgebraInference(self.rule_models, encoder, config=config, device=self.device)
        
        # Create baseline implementation
        self.baseline_fn = create_baseline_compose_energies(self.inference)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_basic_correctness(self):
        """Test basic correctness with default parameters."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        t = torch.full((batch_size,), k, dtype=torch.long, device=self.device)
        
        # Disable caching for fair comparison
        optimized_energy = self.inference.compose_energies(inp, out, k, t=t, _skip_cache=True)
        baseline_energy = self.baseline_fn(inp, out, k, t=t)
        
        # Check mathematical equivalence
        max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
        assert max_diff < 1e-5, f"Energy difference too large: {max_diff}"
        
        # Check shapes match
        assert optimized_energy.shape == baseline_energy.shape
        assert optimized_energy.device == baseline_energy.device
    
    def test_no_normalization(self):
        """Test correctness when normalization is disabled."""
        batch_size = 16
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 1
        
        optimized_energy = self.inference.compose_energies(inp, out, k, normalize=False)
        baseline_energy = self.baseline_fn(inp, out, k, normalize=False)
        
        max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
        assert max_diff < 1e-5, f"No-normalization energy difference: {max_diff}"
    
    def test_different_batch_sizes(self):
        """Test correctness across different batch sizes."""
        for batch_size in [1, 4, 16, 32, 64]:
            inp = torch.randn(batch_size, 128, device=self.device)
            out = torch.randn(batch_size, 128, device=self.device)
            k = 3
            
            optimized_energy = self.inference.compose_energies(inp, out, k, _skip_cache=True)
            baseline_energy = self.baseline_fn(inp, out, k)
            
            max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
            assert max_diff < 1e-5, f"Batch size {batch_size} difference: {max_diff}"
    
    def test_custom_rule_weights(self):
        """Test correctness with custom rule weights."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        # Test various weight configurations
        weight_configs = [
            {'distribute': 2.0, 'combine': 0.5, 'isolate': 1.5, 'divide': 0.8},
            {'distribute': 0.1, 'combine': 3.0},  # Partial weights
            {},  # Empty weights (should default to 1.0)
        ]
        
        for rule_weights in weight_configs:
            optimized_energy = self.inference.compose_energies(
                inp, out, k, rule_weights=rule_weights, _skip_cache=True
            )
            baseline_energy = self.baseline_fn(inp, out, k, rule_weights=rule_weights)
            
            max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
            assert max_diff < 1e-5, f"Rule weights {rule_weights} difference: {max_diff}"
    
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
        
        optimized_energy = self.inference.compose_energies(
            inp, out, k, calibration_scales=calibration_scales, _skip_cache=True
        )
        baseline_energy = self.baseline_fn(inp, out, k, calibration_scales=calibration_scales)
        
        max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
        assert max_diff < 1e-5, f"Calibration scales difference: {max_diff}"
    
    def test_single_rule(self):
        """Test correctness with single rule model."""
        # Create inference with only one rule
        single_rule_models = {'distribute': self.rule_models['distribute']}
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5)
        single_inference = AlgebraInference(single_rule_models, encoder, config=config, device=self.device)
        single_baseline = create_baseline_compose_energies(single_inference)
        
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        optimized_energy = single_inference.compose_energies(inp, out, k, _skip_cache=True)
        baseline_energy = single_baseline(inp, out, k)
        
        max_diff = torch.abs(optimized_energy - baseline_energy).max().item()
        assert max_diff < 1e-5, f"Single rule difference: {max_diff}"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through optimized implementation."""
        batch_size = 4
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
        k = 2
        
        # Compute energy and backpropagate
        energy = self.inference.compose_energies(inp, out, k, _skip_cache=True)
        loss = energy.sum()
        loss.backward()
        
        # Check that gradients exist and are reasonable
        assert out.grad is not None, "Gradients should flow to output"
        assert not torch.isnan(out.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(out.grad).any(), "Gradients should not contain Inf"
        
        # Check gradient magnitude is reasonable
        grad_norm = torch.norm(out.grad).item()
        assert 1e-8 < grad_norm < 1e4, f"Gradient norm should be reasonable: {grad_norm}"
    
    def test_caching_consistency(self):
        """Test that caching doesn't affect results."""
        batch_size = 8
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        
        # First call (cache miss)
        energy_1 = self.inference.compose_energies(inp, out, k)
        
        # Second call (cache hit)
        energy_2 = self.inference.compose_energies(inp, out, k)
        
        # Third call without cache
        energy_3 = self.inference.compose_energies(inp, out, k, _skip_cache=True)
        
        # All should be identical
        assert torch.allclose(energy_1, energy_2, atol=1e-7), "Cached result should match"
        assert torch.allclose(energy_1, energy_3, atol=1e-7), "No-cache result should match"
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        batch_size = 8
        k = 2
        
        # Test with very small values
        inp_small = torch.full((batch_size, 128), 1e-6, device=self.device)
        out_small = torch.full((batch_size, 128), 1e-6, device=self.device)
        
        energy_small = self.inference.compose_energies(inp_small, out_small, k, _skip_cache=True)
        assert torch.isfinite(energy_small).all(), "Small values should produce finite energy"
        
        # Test with large values
        inp_large = torch.full((batch_size, 128), 100.0, device=self.device)
        out_large = torch.full((batch_size, 128), 100.0, device=self.device)
        
        energy_large = self.inference.compose_energies(inp_large, out_large, k, _skip_cache=True)
        assert torch.isfinite(energy_large).all(), "Large values should produce finite energy"
        
        # Test with mixed signs
        inp_mixed = torch.randn(batch_size, 128, device=self.device) * 10
        out_mixed = torch.randn(batch_size, 128, device=self.device) * 10
        
        energy_mixed = self.inference.compose_energies(inp_mixed, out_mixed, k, _skip_cache=True)
        assert torch.isfinite(energy_mixed).all(), "Mixed values should produce finite energy"


def run_performance_regression_test():
    """Run performance regression test to ensure optimizations don't hurt performance."""
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test setup
    rule_names = ['distribute', 'combine', 'isolate', 'divide']
    rule_models = {}
    for rule in rule_names:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.to(device)
        wrapper.eval()
        rule_models[rule] = wrapper
    
    encoder = CharacterLevelEncoder()
    config = InferenceConfig(K=5, max_iterations=10)
    inference = AlgebraInference(rule_models, encoder, config=config, device=device)
    baseline_fn = create_baseline_compose_energies(inference)
    
    # Test inputs
    batch_size = 32
    inp = torch.randn(batch_size, 128, device=device)
    out = torch.randn(batch_size, 128, device=device)
    k = 2
    t = torch.full((batch_size,), k, dtype=torch.long, device=device)
    
    iterations = 100
    
    # Benchmark baseline
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        baseline_energy = baseline_fn(inp, out, k, t=t)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    baseline_time = time.perf_counter() - start_time
    
    # Benchmark optimized
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        optimized_energy = inference.compose_energies(inp, out, k, t=t, _skip_cache=True)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    optimized_time = time.perf_counter() - start_time
    
    speedup = baseline_time / optimized_time
    
    print(f"\n🏃 PERFORMANCE REGRESSION TEST:")
    print(f"   Baseline time:    {baseline_time*1000:.2f} ms")
    print(f"   Optimized time:   {optimized_time*1000:.2f} ms")
    print(f"   Speedup:          {speedup:.2f}x")
    
    # Verify correctness
    max_diff = torch.abs(baseline_energy - optimized_energy).max().item()
    print(f"   Max difference:   {max_diff:.2e}")
    
    # Assert performance improvement and correctness
    assert speedup >= 0.95, f"Performance regression detected: {speedup:.2f}x"
    assert max_diff < 1e-5, f"Correctness regression detected: {max_diff:.2e}"
    
    print("   Status:           ✅ PASS")
    
    return {'speedup': speedup, 'max_diff': max_diff}


if __name__ == "__main__":
    # Run correctness tests
    test_instance = TestNormalizationCorrectness()
    test_instance.setup_method()
    
    print("Running normalization optimization correctness tests...")
    
    test_methods = [
        test_instance.test_basic_correctness,
        test_instance.test_no_normalization,
        test_instance.test_different_batch_sizes,
        test_instance.test_custom_rule_weights,
        test_instance.test_calibration_scales,
        test_instance.test_single_rule,
        test_instance.test_gradient_flow,
        test_instance.test_caching_consistency,
        test_instance.test_numerical_stability
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    # Run performance regression test
    regression_results = run_performance_regression_test()
    
    if failed == 0 and regression_results['speedup'] >= 0.95:
        print("\n🎉 ALL TESTS PASSED - Optimization is correct and performant!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please review the implementation")
        exit(1)