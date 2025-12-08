#!/usr/bin/env python3
"""
Test the clean energy caching optimization.

This tests the minimal, targeted optimization implemented in algebra_inference.py
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import logging
from unittest.mock import patch

# Set logging to minimal to avoid noise
logging.basicConfig(level=logging.CRITICAL)

# Import inference components
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_encoder import CharacterLevelEncoder


class CleanOptimizationTester:
    """Test the clean energy caching optimization."""
    
    def create_test_models(self, rule_names: List[str]) -> Dict[str, AlgebraDiffusionWrapper]:
        """Create test models."""
        rule_models = {}
        for rule in rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.eval()
            rule_models[rule] = wrapper
        return rule_models
    
    def create_original_inference(self, rule_models, encoder, config):
        """
        Create a version that simulates the original behavior WITHOUT energy caching
        to get a true baseline comparison.
        """
        
        class OriginalInference(AlgebraInference):
            """Original inference without energy caching optimization."""
            
            def ired_inference_original(self, inp_embedding, config=None, rule_weights=None):
                """Original inference logic without energy caching."""
                
                if config is None:
                    config = self.config
                
                if len(inp_embedding.shape) != 2 or inp_embedding.shape[1] != 128:
                    raise ValueError(f"inp_embedding must have shape (B, 128), got {inp_embedding.shape}")
                
                batch_size = inp_embedding.shape[0]
                out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
                
                info = {
                    'energy_history': [],
                    'step_sizes': [],
                    'accepted_steps': 0,
                    'total_steps': 0
                }
                
                # Original behavior: NO energy caching
                for k in range(config.K):
                    current_step_size = config.get_adaptive_step_size(k)
                    info['step_sizes'].append(current_step_size)
                    
                    # Allocate timestep tensor each time (original behavior)
                    timestep_tensor = torch.full((batch_size,), k, dtype=torch.long, device=inp_embedding.device)
                    
                    for t in range(config.max_iterations):
                        
                        # ORIGINAL: Always compute both energy and gradient (no caching)
                        energy_current, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                        energy_before_val = energy_current.item()
                        
                        info['energy_history'].append(energy_before_val)
                        
                        # Gradient descent
                        out_new = out - current_step_size * grad
                        
                        # ORIGINAL: Always compute energy for new state (no caching)
                        energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights, timestep_tensor)
                        energy_after_val = energy_after.item()
                        
                        # Simple acceptance
                        delta_E = energy_after_val - energy_before_val
                        temperature = 0.5  # Simplified
                        
                        if delta_E <= 0:
                            accept_prob = 1.0
                        else:
                            accept_prob = np.exp(-delta_E / temperature)
                        
                        import random
                        accepted = random.random() < accept_prob
                        
                        if accepted:
                            out = out_new.detach().requires_grad_(True)
                            info['accepted_steps'] += 1
                        
                        info['total_steps'] += 1
                        
                        # Early break for testing speed
                        if t >= 4:
                            break
                    
                    # Early break for testing speed
                    if k >= 2:
                        break
                
                info['final_energy'] = energy_before_val
                info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
                
                return out.detach(), info
        
        return OriginalInference(rule_models, encoder, config=config, device='cpu')
    
    def benchmark_clean_optimization(self, iterations: int = 5):
        """Benchmark clean energy caching optimization."""
        
        print("\n" + "="*70)
        print("CLEAN ENERGY CACHING OPTIMIZATION TEST")
        print("="*70)
        
        # Setup
        rule_names = ['distribute', 'combine']  # Fewer rules for faster testing
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=3, max_iterations=5, step_size=0.01)  # Small for testing
        test_equations = ["x+1=2", "2*x=4"]  # Simple equations
        
        rule_models = self.create_test_models(rule_names)
        
        print(f"Configuration:")
        print(f"  Rules: {len(rule_names)}")
        print(f"  Landscapes (K): {config.K}")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Equations: {len(test_equations)}")
        print(f"  Iterations per test: {iterations}")
        
        all_results = []
        
        for equation in test_equations:
            print(f"\nTesting: '{equation}'")
            
            # Test original (no caching)
            print("  Original (no energy caching)...")
            original_inference = self.create_original_inference(rule_models, encoder, config)
            
            original_times = []
            for i in range(iterations):
                try:
                    with torch.no_grad():
                        inp_embedding = encoder(equation).unsqueeze(0)
                    
                    # Suppress noise
                    import sys, os
                    original_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    
                    start_time = time.perf_counter()
                    try:
                        out, info = original_inference.ired_inference_original(inp_embedding)
                    finally:
                        sys.stderr.close()
                        sys.stderr = original_stderr
                    end_time = time.perf_counter()
                    
                    original_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Original run {i+1} failed: {e}")
            
            # Test optimized (with caching)
            print("  Optimized (with energy caching)...")
            optimized_inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
            
            optimized_times = []
            
            for i in range(iterations):
                try:
                    with torch.no_grad():
                        inp_embedding = encoder(equation).unsqueeze(0)
                    
                    # Suppress noise
                    import sys, os
                    original_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    
                    start_time = time.perf_counter()
                    try:
                        out, info = optimized_inference.ired_inference(inp_embedding)
                    finally:
                        sys.stderr.close()
                        sys.stderr = original_stderr
                    end_time = time.perf_counter()
                    
                    optimized_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Optimized run {i+1} failed: {e}")
            
            # Calculate results
            if original_times and optimized_times:
                original_avg = np.mean(original_times)
                optimized_avg = np.mean(optimized_times)
                speedup = original_avg / optimized_avg
                speedup_pct = (speedup - 1.0) * 100
                
                result = {
                    'equation': equation,
                    'original_avg_time': original_avg,
                    'optimized_avg_time': optimized_avg,
                    'speedup_factor': speedup,
                    'speedup_percentage': speedup_pct,
                    'original_times': original_times,
                    'optimized_times': optimized_times
                }
                
                all_results.append(result)
                
                print(f"    Original: {original_avg:.4f}s")
                print(f"    Optimized: {optimized_avg:.4f}s")
                print(f"    Speedup: {speedup:.2f}x ({speedup_pct:+.1f}%)")
        
        # Overall analysis
        if all_results:
            overall_speedups = [r['speedup_percentage'] for r in all_results]
            avg_speedup = np.mean(overall_speedups)
            min_speedup = np.min(overall_speedups)
            max_speedup = np.max(overall_speedups)
            
            print(f"\n" + "="*50)
            print("OPTIMIZATION RESULTS:")
            print("="*50)
            print(f"Average speedup:      {avg_speedup:+.1f}%")
            print(f"Speedup range:        {min_speedup:+.1f}% to {max_speedup:+.1f}%")
            
            # Determine success
            target_speedup = 30
            target_met = avg_speedup >= target_speedup
            
            print(f"\nTARGET ASSESSMENT:")
            if target_met:
                print(f"✅ SUCCESS: {avg_speedup:.1f}% average speedup EXCEEDS {target_speedup}% target!")
                status = "success"
            elif avg_speedup >= 15:
                print(f"🟡 PARTIAL: {avg_speedup:.1f}% average speedup shows progress toward {target_speedup}% target")
                status = "partial"
            elif avg_speedup >= 0:
                print(f"🔶 MINOR IMPROVEMENT: {avg_speedup:.1f}% speedup below {target_speedup}% target")
                status = "minor_improvement"
            else:
                print(f"❌ REGRESSION: {avg_speedup:.1f}% performance loss")
                status = "regression"
            
            return {
                'status': status,
                'average_speedup_percentage': avg_speedup,
                'speedup_range': [min_speedup, max_speedup],
                'target_speedup': target_speedup,
                'target_met': target_met,
                'detailed_results': all_results
            }
        else:
            print("❌ No successful tests completed")
            return {'status': 'failed'}


def main():
    """Main test execution."""
    
    print("Clean Energy Caching Optimization Test")
    print("="*40)
    print("Testing minimal energy caching implementation")
    print("Target: 30%+ speedup through reduced energy computations")
    
    tester = CleanOptimizationTester()
    results = tester.benchmark_clean_optimization(iterations=3)
    
    # Save results
    output_file = 'clean_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()