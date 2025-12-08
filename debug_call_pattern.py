#!/usr/bin/env python3
"""
Debug the exact call pattern to understand the energy computations.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algebra.algebra_inference import AlgebraInference, InferenceConfig
from algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from algebra.algebra_encoder import CharacterLevelEncoder

class CallPatternTracker:
    """Track the exact pattern of method calls."""
    
    def __init__(self, inference_engine):
        self.call_stack = []
        self.inference = inference_engine
        
        # Patch all relevant methods
        self._original_compose_energies = inference_engine.compose_energies
        self._original_compute_energy_and_gradient = inference_engine.compute_energy_and_gradient
        self._original_compute_composed_gradient = inference_engine.compute_composed_gradient
        
        inference_engine.compose_energies = self._track_compose_energies
        inference_engine.compute_energy_and_gradient = self._track_energy_and_gradient
        inference_engine.compute_composed_gradient = self._track_gradient
    
    def _track_compose_energies(self, *args, **kwargs):
        """Track compose_energies calls."""
        # Record the call with context
        import traceback
        stack = traceback.extract_stack()
        # Find the caller that's not this tracker
        caller = None
        for frame in reversed(stack[:-1]):  # Skip this frame
            if 'track_' not in frame.name and 'CallPatternTracker' not in frame.filename:
                caller = frame.name
                break
        
        self.call_stack.append(f"compose_energies (from {caller})")
        return self._original_compose_energies(*args, **kwargs)
        
    def _track_energy_and_gradient(self, *args, **kwargs):
        """Track energy_and_gradient calls."""
        self.call_stack.append("compute_energy_and_gradient")
        return self._original_compute_energy_and_gradient(*args, **kwargs)
        
    def _track_gradient(self, *args, **kwargs):
        """Track gradient calls."""
        self.call_stack.append("compute_composed_gradient")
        return self._original_compute_composed_gradient(*args, **kwargs)
    
    def reset(self):
        """Reset call stack."""
        self.call_stack = []
        
    def print_call_pattern(self):
        """Print the call pattern."""
        print("Call pattern:")
        for i, call in enumerate(self.call_stack):
            print(f"  {i+1}: {call}")

def debug_call_pattern():
    """Debug the exact call pattern."""
    
    print("Debugging exact call pattern...")
    
    # Simple test setup
    rule_models = {}
    for rule_name in ['distribute']:
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models[rule_name] = wrapper
    
    encoder = CharacterLevelEncoder(d_model=128)
    
    # Very small config for analysis
    config = InferenceConfig(
        step_size=0.1,
        max_iterations=2,  # Just 2 iterations
        K=2,  # Just 2 landscapes
        use_adaptive_step=False
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
    tracker = CallPatternTracker(inference)
    
    # Run test
    tracker.reset()
    result = inference.solve_equation("x=1")
    
    # Print results
    tracker.print_call_pattern()
    
    # Count specific patterns
    total_calls = len([c for c in tracker.call_stack if 'compose_energies' in c])
    energy_grad_calls = len([c for c in tracker.call_stack if 'compute_energy_and_gradient' in c])
    grad_only_calls = len([c for c in tracker.call_stack if 'compute_composed_gradient' in c])
    
    print(f"\nSummary:")
    print(f"Total energy calls: {total_calls}")
    print(f"Energy+gradient calls: {energy_grad_calls}")
    print(f"Gradient-only calls: {grad_only_calls}")
    print(f"Expected: 2 landscapes × 2 iterations = 4 iterations")
    print(f"Expected calls: 4 gradient + 4 metropolis + 1 final = 9 total")

if __name__ == "__main__":
    debug_call_pattern()