#!/usr/bin/env python3
"""
Debug the energy caching logic to see why it's not working.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algebra.algebra_inference import AlgebraInference, InferenceConfig
from algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from algebra.algebra_encoder import CharacterLevelEncoder

# Monkey patch the inference to add debugging
original_ired_inference = None

def debug_ired_inference(self, inp_embedding, config=None, rule_weights=None):
    """Debug version of ired_inference that prints cache status."""
    
    # Use provided config or fall back to instance config
    if config is None:
        config = self.config
    
    batch_size = inp_embedding.shape[0]
    
    # Initialize from noise  
    out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
    
    # Track optimization statistics
    info = {
        'energy_history': [],
        'step_sizes': [],
        'landscape_transitions': [],
        'gradient_norms': [],
        'accepted_steps': 0,
        'total_steps': 0
    }
    
    # Initialize energy caching variables
    have_cached_energy = False
    cached_energy_val = None
    
    print(f"Starting inference with {config.K} landscapes, {config.max_iterations} iterations each")
    
    # Iterate through K landscapes
    for k in range(config.K):
        sigma_k = torch.sqrt(1 - self.alphas_cumprod[k]).item()
        
        # Adaptive step size using config method
        current_step_size = config.get_adaptive_step_size(k)
        info['step_sizes'].append(current_step_size)
        
        print(f"\nLandscape {k}:")
        
        # Pre-allocate timestep tensor for this landscape
        timestep_tensor = torch.full((batch_size,), k, dtype=torch.long, device=inp_embedding.device)
        
        # Reset energy cache when starting new landscape
        have_cached_energy = False
        cached_energy_val = None
        print(f"  Cache reset for new landscape")
        
        # max_iterations gradient descent steps in this landscape
        for t in range(config.max_iterations):
            print(f"  Iteration {t}: cache={have_cached_energy}, cached_val={cached_energy_val}")
            
            # Energy caching optimization
            if have_cached_energy:
                print(f"    Using cached energy: {cached_energy_val}")
                energy_before_val = cached_energy_val
                # Only compute gradient
                grad = self.compute_composed_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
            else:
                print(f"    Computing fresh energy and gradient")
                # Compute both energy and gradient atomically
                energy_current, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                energy_before_val = energy_current.item()
            
            grad_norm = torch.norm(grad).item()
            info['energy_history'].append(energy_before_val)
            info['gradient_norms'].append(grad_norm)
            
            # Gradient descent step
            out_new = out - current_step_size * grad
            
            # Metropolis acceptance criteria
            print(f"    Computing energy for new state")
            energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights, timestep_tensor)
            energy_after_val = energy_after.item()
            delta_E = energy_after_val - energy_before_val
            
            # Simple acceptance (always accept for debugging)
            accepted = True
            print(f"    Step accepted: {accepted}, new energy: {energy_after_val}")
            
            if accepted:
                # Update state
                out = out_new.detach().requires_grad_(True)
                
                # Cache energy for next iteration
                have_cached_energy = True
                cached_energy_val = energy_after_val
                info['accepted_steps'] += 1
                print(f"    Cached energy for next iteration: {cached_energy_val}")
            else:
                # Invalidate cache
                have_cached_energy = False
                cached_energy_val = None
                print(f"    Cache invalidated due to rejection")
                
            info['total_steps'] += 1
        
        info['landscape_transitions'].append(k)
    
    # Final statistics
    final_k = k
    final_timestep_tensor = torch.full((batch_size,), final_k, dtype=torch.long, device=inp_embedding.device)
    info['final_energy'] = self.compose_energies(inp_embedding, out, final_k, rule_weights, final_timestep_tensor).item()
    info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
    
    return out.detach(), info

def test_cache_debugging():
    """Test with detailed cache debugging."""
    
    # Create simple test setup
    rule_models = {}
    for rule_name in ['distribute']:
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models[rule_name] = wrapper
    
    encoder = CharacterLevelEncoder(d_model=128)
    
    config = InferenceConfig(
        step_size=0.1,
        max_iterations=2,  # Just 2 iterations per landscape
        K=2,  # Just 2 landscapes
        use_adaptive_step=False
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
    
    # Replace the ired_inference method with our debug version
    global original_ired_inference
    original_ired_inference = inference.ired_inference
    inference.ired_inference = debug_ired_inference.__get__(inference, AlgebraInference)
    
    # Run inference
    result = inference.solve_equation("x=1")
    
    print(f"\nFinal result:")
    print(f"Accepted steps: {result['inference_info']['accepted_steps']}")
    print(f"Total steps: {result['inference_info']['total_steps']}")

if __name__ == "__main__":
    test_cache_debugging()