#!/usr/bin/env python3
"""
Final energy caching test with accurate tracking.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algebra.algebra_inference import AlgebraInference, InferenceConfig
from algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from algebra.algebra_encoder import CharacterLevelEncoder

class PreciseEnergyTracker:
    """Precise energy computation tracker."""
    
    def __init__(self, inference_engine):
        self.inference = inference_engine
        self.total_energy_calls = 0
        self.gradient_calls = 0
        self.energy_and_gradient_calls = 0
        
        # Patch the methods
        self._original_compose_energies = inference_engine.compose_energies
        self._original_compute_energy_and_gradient = inference_engine.compute_energy_and_gradient
        self._original_compute_composed_gradient = inference_engine.compute_composed_gradient
        
        inference_engine.compose_energies = self._track_compose_energies
        inference_engine.compute_energy_and_gradient = self._track_energy_and_gradient
        inference_engine.compute_composed_gradient = self._track_gradient
    
    def _track_compose_energies(self, *args, **kwargs):
        """Track direct compose_energies calls."""
        self.total_energy_calls += 1
        return self._original_compose_energies(*args, **kwargs)
        
    def _track_energy_and_gradient(self, *args, **kwargs):
        """Track energy_and_gradient calls."""
        self.energy_and_gradient_calls += 1
        return self._original_compute_energy_and_gradient(*args, **kwargs)
        
    def _track_gradient(self, *args, **kwargs):
        """Track gradient-only calls."""
        self.gradient_calls += 1
        return self._original_compute_composed_gradient(*args, **kwargs)
    
    def reset(self):
        """Reset counters."""
        self.total_energy_calls = 0
        self.gradient_calls = 0 
        self.energy_and_gradient_calls = 0
        
    def get_summary(self):
        """Get tracking summary."""
        return {
            'total_energy_calls': self.total_energy_calls,
            'energy_and_gradient_calls': self.energy_and_gradient_calls,
            'gradient_only_calls': self.gradient_calls,
            'metropolis_energy_calls': self.total_energy_calls - self.energy_and_gradient_calls - self.gradient_calls
        }

def test_precise_energy_counting():
    """Test with precise energy call counting."""
    
    print("Testing precise energy computation counting...")
    
    # Create test setup
    rule_models = {}
    for rule_name in ['distribute']:
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models[rule_name] = wrapper
    
    encoder = CharacterLevelEncoder(d_model=128)
    
    config = InferenceConfig(
        step_size=0.1,
        max_iterations=5,  # 5 iterations per landscape  
        K=3,  # 3 landscapes
        use_adaptive_step=False
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
    tracker = PreciseEnergyTracker(inference)
    
    # Run test
    tracker.reset()
    result = inference.solve_equation("x+1=2")
    
    summary = tracker.get_summary()
    info = result['inference_info']
    
    # Analysis
    total_iterations = config.K * config.max_iterations
    print(f"\nConfiguration:")
    print(f"- Total iterations: {total_iterations} ({config.K} landscapes × {config.max_iterations} iterations)")
    print(f"- Total steps run: {info['total_steps']}")
    print(f"- Accepted steps: {info['accepted_steps']}")
    
    print(f"\nEnergy computation calls:")
    print(f"- Total compose_energies calls: {summary['total_energy_calls']}")
    print(f"- Energy+gradient calls: {summary['energy_and_gradient_calls']}")
    print(f"- Gradient-only calls: {summary['gradient_only_calls']}")
    print(f"- Metropolis energy calls: {summary['metropolis_energy_calls']}")
    
    # Calculate expected calls without caching
    # Without caching: every iteration needs energy+gradient (1 energy call) + metropolis (1 energy call) = 2 per iteration
    expected_without_caching = total_iterations * 2
    
    # With caching: first iteration of each landscape needs energy+gradient, rest need gradient-only
    # Plus all iterations need metropolis energy
    landscapes_first_iterations = config.K  # First iteration of each landscape
    cached_iterations = total_iterations - landscapes_first_iterations  # Iterations that can use cache
    expected_with_perfect_caching = landscapes_first_iterations * 2 + cached_iterations * 1  # 2 for first, 1 for cached
    
    actual_calls = summary['total_energy_calls']
    
    print(f"\nAnalysis:")
    # Exclude final statistics call (1 extra call for final energy reporting)
    algorithm_calls = actual_calls - 1  # Subtract the final statistics call
    
    print(f"- Expected without caching: {expected_without_caching}")
    print(f"- Expected with perfect caching: {expected_with_perfect_caching}")
    print(f"- Actual calls (including 1 final stats): {actual_calls}")
    print(f"- Algorithm calls (excluding final stats): {algorithm_calls}")
    print(f"- Improvement vs no caching: {(expected_without_caching - algorithm_calls) / expected_without_caching * 100:.1f}%")
    print(f"- Efficiency vs perfect caching: {(algorithm_calls - expected_with_perfect_caching) / expected_with_perfect_caching * 100:.1f}% overhead")
    
    # Detailed breakdown
    print(f"\nDetailed breakdown:")
    print(f"- Landscapes: {config.K}")
    print(f"- First iterations (no cache): {config.K} × 2 calls = {config.K * 2}")
    print(f"- Cached iterations: {cached_iterations} × 1 call = {cached_iterations}")
    print(f"- Expected total: {config.K * 2 + cached_iterations}")
    
    algorithm_calls = actual_calls - 1  # Exclude final statistics call
    improvement_percentage = (expected_without_caching - algorithm_calls) / expected_without_caching * 100
    success = improvement_percentage >= 25.0  # 25% improvement target
    
    return {
        'success': success,
        'actual_calls': algorithm_calls,  # Report algorithm calls only
        'expected_without_caching': expected_without_caching,
        'improvement_percentage': improvement_percentage,
        'summary': summary
    }

def main():
    """Main test function."""
    result = test_precise_energy_counting()
    
    print(f"\n{'='*60}")
    print("ENERGY CACHING TEST RESULT")
    print(f"{'='*60}")
    
    improvement = result['improvement_percentage']
    if result['success']:
        print(f"✅ SUCCESS: {improvement:.1f}% improvement achieved!")
    else:
        print(f"❌ INSUFFICIENT: Only {improvement:.1f}% improvement")
        
    return result['success']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)