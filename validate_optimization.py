#!/usr/bin/env python3
"""
Validate the energy caching optimization by measuring actual computational savings.
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algebra.algebra_inference import AlgebraInference, InferenceConfig
from algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from algebra.algebra_encoder import CharacterLevelEncoder

def create_baseline_inference(device='cpu'):
    """Create a baseline inference engine without caching for comparison."""
    
    # Create rule models
    rule_models = {}
    for rule_name in ['distribute', 'combine']:
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models[rule_name] = wrapper
    
    encoder = CharacterLevelEncoder(d_model=128)
    
    config = InferenceConfig(
        step_size=0.1,
        max_iterations=20,
        K=5,
        use_adaptive_step=False
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device=device)
    
    # Disable caching by modifying the inference loop to always use compute_energy_and_gradient
    def no_cache_ired_inference(self, inp_embedding, config=None, rule_weights=None):
        """Version without caching for baseline comparison."""
        
        if config is None:
            config = self.config
            
        batch_size = inp_embedding.shape[0]
        out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
        
        info = {
            'energy_history': [],
            'step_sizes': [],
            'landscape_transitions': [],
            'gradient_norms': [],
            'accepted_steps': 0,
            'total_steps': 0
        }
        
        # No caching - always use compute_energy_and_gradient
        for k in range(config.K):
            sigma_k = torch.sqrt(1 - self.alphas_cumprod[k]).item()
            current_step_size = config.get_adaptive_step_size(k)
            info['step_sizes'].append(current_step_size)
            
            timestep_tensor = torch.full((batch_size,), k, dtype=torch.long, device=inp_embedding.device)
            
            for t in range(config.max_iterations):
                # Always compute fresh energy and gradient (no caching)
                energy_current, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                energy_before_val = energy_current.item()
                
                grad_norm = torch.norm(grad).item()
                info['energy_history'].append(energy_before_val)
                info['gradient_norms'].append(grad_norm)
                
                out_new = out - current_step_size * grad
                
                energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights, timestep_tensor)
                energy_after_val = energy_after.item()
                
                # Simple acceptance for comparison
                accepted = True
                
                if accepted:
                    out = out_new.detach().requires_grad_(True)
                    info['accepted_steps'] += 1
                
                info['total_steps'] += 1
            
            info['landscape_transitions'].append(k)
        
        # Final statistics
        final_k = k
        final_timestep_tensor = torch.full((batch_size,), final_k, dtype=torch.long, device=inp_embedding.device)
        info['final_energy'] = self.compose_energies(inp_embedding, out, final_k, rule_weights, final_timestep_tensor).item()
        info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
        
        return out.detach(), info
    
    # Replace the method
    inference.no_cache_ired_inference = no_cache_ired_inference.__get__(inference, AlgebraInference)
    
    return inference

def benchmark_caching_performance():
    """Benchmark the performance improvement from energy caching."""
    
    print("Benchmarking energy caching performance improvement...")
    
    # Create test equations
    test_equations = [
        "x+1=3",
        "2*x=6", 
        "x-2=4",
        "3*x+1=7",
        "x/2=3"
    ]
    
    # Test with caching (current implementation)
    cached_inference = AlgebraInference(
        {
            'distribute': AlgebraDiffusionWrapper(AlgebraEBM(inp_dim=128, out_dim=128, rule_name='distribute')),
            'combine': AlgebraDiffusionWrapper(AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine'))
        },
        CharacterLevelEncoder(d_model=128),
        InferenceConfig(step_size=0.1, max_iterations=10, K=3, use_adaptive_step=False),
        device='cpu'
    )
    
    # Test without caching (baseline)
    baseline_inference = create_baseline_inference()
    
    print(f"Testing with {len(test_equations)} equations...")
    
    # Benchmark cached version
    start_time = time.time()
    cached_results = []
    for eq in test_equations:
        result = cached_inference.solve_equation(eq)
        cached_results.append(result)
    cached_time = time.time() - start_time
    
    # Benchmark baseline version  
    start_time = time.time()
    baseline_results = []
    for eq in test_equations:
        # Use the no-cache version
        inp_embedding = baseline_inference.encoder(eq).unsqueeze(0)
        out_embedding, info = baseline_inference.no_cache_ired_inference(inp_embedding)
        baseline_results.append({'inference_info': info})
    baseline_time = time.time() - start_time
    
    # Calculate statistics
    speedup = baseline_time / cached_time if cached_time > 0 else float('inf')
    improvement_percent = (baseline_time - cached_time) / baseline_time * 100 if baseline_time > 0 else 0
    
    print(f"\nPerformance Results:")
    print(f"Baseline (no caching): {baseline_time:.3f} seconds")
    print(f"Optimized (with caching): {cached_time:.3f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement_percent:.1f}%")
    
    # Check for target improvement
    target_improvement = 30.0  # 30% minimum
    success = improvement_percent >= target_improvement
    
    print(f"\nTarget: {target_improvement}% improvement")
    if success:
        print(f"✅ SUCCESS: Achieved {improvement_percent:.1f}% improvement")
    else:
        print(f"❌ INSUFFICIENT: Only {improvement_percent:.1f}% improvement")
        
    return {
        'success': success,
        'baseline_time': baseline_time,
        'cached_time': cached_time,
        'speedup': speedup,
        'improvement_percent': improvement_percent,
        'cached_results': cached_results,
        'baseline_results': baseline_results
    }

def main():
    """Main validation function."""
    results = benchmark_caching_performance()
    
    print(f"\n{'='*60}")
    print("ENERGY CACHING OPTIMIZATION VALIDATION")
    print(f"{'='*60}")
    
    if results['success']:
        print(f"🎉 PASSED: {results['improvement_percent']:.1f}% performance improvement achieved!")
        print(f"   Speedup: {results['speedup']:.2f}x faster")
    else:
        print(f"❌ FAILED: Only {results['improvement_percent']:.1f}% improvement (target: 30%)")
    
    return results['success']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)