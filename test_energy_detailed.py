#!/usr/bin/env python3
"""
Detailed energy computation analysis to understand the caching optimization.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algebra.algebra_inference import AlgebraInference, InferenceConfig
from algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from algebra.algebra_encoder import CharacterLevelEncoder

class DetailedEnergyTracker:
    """Track where energy computations happen."""
    
    def __init__(self, inference_engine):
        self.inference = inference_engine
        self.call_trace = []
        
        # Track all energy computation methods
        self._original_compose_energies = inference_engine.compose_energies
        self._original_compute_energy_and_gradient = inference_engine.compute_energy_and_gradient
        self._original_compute_composed_gradient = inference_engine.compute_composed_gradient
        
        inference_engine.compose_energies = self._track_compose_energies
        inference_engine.compute_energy_and_gradient = self._track_compute_energy_and_gradient
        inference_engine.compute_composed_gradient = self._track_compute_composed_gradient
        
    def _track_compose_energies(self, *args, **kwargs):
        """Track compose_energies calls."""
        self.call_trace.append('compose_energies')
        return self._original_compose_energies(*args, **kwargs)
        
    def _track_compute_energy_and_gradient(self, *args, **kwargs):
        """Track compute_energy_and_gradient calls."""
        self.call_trace.append('compute_energy_and_gradient')
        return self._original_compute_energy_and_gradient(*args, **kwargs)
        
    def _track_compute_composed_gradient(self, *args, **kwargs):
        """Track compute_composed_gradient calls.""" 
        self.call_trace.append('compute_composed_gradient')
        return self._original_compute_composed_gradient(*args, **kwargs)
    
    def reset_trace(self):
        """Reset the call trace."""
        self.call_trace = []
    
    def get_trace_summary(self):
        """Get summary of calls."""
        from collections import Counter
        counter = Counter(self.call_trace)
        return dict(counter)

def analyze_detailed_energy_flow():
    """Analyze the detailed flow of energy computations."""
    
    print("Analyzing detailed energy computation flow...")
    
    # Create test setup
    rule_models = {}
    for rule_name in ['distribute']:  # Just one rule for simplicity
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models[rule_name] = wrapper
    
    encoder = CharacterLevelEncoder(d_model=128)
    
    # Very small config for detailed analysis
    config = InferenceConfig(
        step_size=0.1,
        max_iterations=3,  # Just 3 iterations per landscape
        K=2,  # Just 2 landscapes
        use_adaptive_step=False
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
    tracker = DetailedEnergyTracker(inference)
    
    # Run inference
    tracker.reset_trace()
    result = inference.solve_equation("x=1")
    
    # Analyze the trace
    trace = tracker.call_trace
    summary = tracker.get_trace_summary()
    
    print(f"\nTotal iterations: {config.K * config.max_iterations} = {config.K} landscapes × {config.max_iterations} iterations")
    print(f"Call trace ({len(trace)} calls total):")
    
    for i, call in enumerate(trace):
        print(f"  {i+1}: {call}")
    
    print(f"\nSummary:")
    for method, count in summary.items():
        print(f"  {method}: {count} calls")
        
    print(f"\nAccepted steps: {result['inference_info']['accepted_steps']}")
    print(f"Total steps: {result['inference_info']['total_steps']}")
    
    # Analysis
    expected_energy_for_gradient = summary.get('compute_energy_and_gradient', 0) + summary.get('compute_composed_gradient', 0)
    actual_compose_energies = summary.get('compose_energies', 0)
    
    print(f"\nAnalysis:")
    print(f"- Expected energy calls for gradients: {expected_energy_for_gradient}")
    print(f"- Actual compose_energies calls: {actual_compose_energies}")
    print(f"- Extra energy calls: {actual_compose_energies - expected_energy_for_gradient}")
    
    # The extra calls should be for Metropolis acceptance (1 per iteration)
    expected_total = expected_energy_for_gradient + config.K * config.max_iterations
    print(f"- Expected total (gradients + Metropolis): {expected_total}")
    print(f"- Efficiency vs expected: {(expected_total - actual_compose_energies) / expected_total * 100:.1f}%")

if __name__ == "__main__":
    analyze_detailed_energy_flow()