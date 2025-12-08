#!/usr/bin/env python3
"""
Test the effectiveness of energy caching optimizations.

Compares performance before and after optimization improvements.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import logging

# Set logging to minimal to avoid noise
logging.basicConfig(level=logging.CRITICAL)

# Import inference components
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_encoder import CharacterLevelEncoder


class OptimizationTester:
    """Test energy caching optimization effectiveness."""
    
    def __init__(self):
        self.results = {}
    
    def create_test_models(self, rule_names: List[str]) -> Dict[str, AlgebraDiffusionWrapper]:
        """Create test models."""
        rule_models = {}
        for rule in rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.eval()
            rule_models[rule] = wrapper
        return rule_models
    
    def create_baseline_inference(self, rule_models, encoder, config):
        """
        Create baseline inference by temporarily disabling optimizations.
        We'll modify the inference to simulate the original behavior.
        """
        
        class BaselineInference(AlgebraInference):
            """Baseline inference without enhanced optimizations for comparison."""
            
            def ired_inference_baseline(self, inp_embedding, config=None, rule_weights=None):
                """Simplified inference without enhanced caching for baseline comparison."""
                
                if config is None:
                    config = self.config
                
                # Input validation (simplified)
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
                
                # Simple caching (original level)
                cached_energy = None
                
                for k in range(config.K):
                    current_step_size = config.get_adaptive_step_size(k)
                    info['step_sizes'].append(current_step_size)
                    
                    # Allocate timestep tensor each time (no pre-allocation)
                    timestep_tensor = torch.full((batch_size,), k, dtype=torch.long, device=inp_embedding.device)
                    
                    # Reset cache each landscape
                    cached_energy = None
                    
                    for t in range(config.max_iterations):
                        
                        # Basic caching - only cache within single iteration
                        if cached_energy is not None:
                            energy_before_val = cached_energy
                            grad = self.compute_composed_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                        else:
                            energy_current, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                            energy_before_val = energy_current.item()
                        
                        info['energy_history'].append(energy_before_val)
                        
                        # Gradient descent
                        out_new = out - current_step_size * grad
                        
                        # Always compute energy for new state (no caching)
                        energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights, timestep_tensor)
                        energy_after_val = energy_after.item()
                        
                        # Simple acceptance
                        delta_E = energy_after_val - energy_before_val
                        temperature = 0.5  # Simplified temperature
                        
                        if delta_E <= 0:
                            accept_prob = 1.0
                        else:
                            accept_prob = np.exp(-delta_E / temperature)
                        
                        import random
                        accepted = random.random() < accept_prob
                        
                        if accepted:
                            out = out_new.detach().requires_grad_(True)
                            cached_energy = energy_after_val
                            info['accepted_steps'] += 1
                        else:
                            cached_energy = None  # Invalidate cache
                        
                        info['total_steps'] += 1
                        
                        # Early break to keep test fast
                        if t >= 3:  # Limit iterations for testing
                            break
                    
                    # Early break to keep test fast  
                    if k >= 2:  # Limit landscapes for testing
                        break
                
                info['final_energy'] = energy_before_val
                info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
                
                return out.detach(), info
        
        return BaselineInference(rule_models, encoder, config=config, device='cpu')
    
    def benchmark_inference_performance(self, iterations: int = 5):
        """Benchmark inference performance with and without optimizations."""
        
        print("\n" + "="*70)
        print("ENERGY CACHING OPTIMIZATION EFFECTIVENESS TEST")
        print("="*70)
        
        # Setup
        rule_names = ['distribute', 'combine', 'isolate']
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=3, max_iterations=6, step_size=0.01)  # Small for quick testing
        test_equations = ["x+1=2", "2*x+3=7", "x*x=4"]
        
        rule_models = self.create_test_models(rule_names)
        
        print(f"Test configuration:")
        print(f"  Rules: {rule_names}")
        print(f"  Landscapes (K): {config.K}")
        print(f"  Max iterations per landscape: {config.max_iterations}")
        print(f"  Test equations: {len(test_equations)}")
        print(f"  Iterations per test: {iterations}")
        
        all_results = []
        
        for equation in test_equations:
            print(f"\nTesting equation: '{equation}'")
            
            # Test baseline (without enhanced optimizations)
            print("  Running baseline implementation...")
            baseline_inference = self.create_baseline_inference(rule_models, encoder, config)
            
            baseline_times = []
            for i in range(iterations):
                try:
                    with torch.no_grad():
                        inp_embedding = encoder(equation).unsqueeze(0)
                    
                    # Suppress stderr to avoid noise
                    import sys
                    import os
                    original_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    
                    start_time = time.perf_counter()
                    try:
                        out_baseline, info_baseline = baseline_inference.ired_inference_baseline(inp_embedding)
                    finally:
                        sys.stderr.close()
                        sys.stderr = original_stderr
                    end_time = time.perf_counter()
                    
                    baseline_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Baseline run {i+1} failed: {e}")
            
            # Test optimized (with enhanced caching)
            print("  Running optimized implementation...")
            optimized_inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
            
            optimized_times = []
            cache_stats = []
            
            for i in range(iterations):
                try:
                    with torch.no_grad():
                        inp_embedding = encoder(equation).unsqueeze(0)
                    
                    # Suppress stderr
                    import sys
                    import os
                    original_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    
                    start_time = time.perf_counter()
                    try:
                        out_optimized, info_optimized = optimized_inference.ired_inference(inp_embedding)
                    finally:
                        sys.stderr.close()
                        sys.stderr = original_stderr
                    end_time = time.perf_counter()
                    
                    optimized_times.append(end_time - start_time)
                    cache_stats.append({
                        'cache_hit_rate': info_optimized.get('cache_hit_rate', 0),
                        'cache_hits': info_optimized.get('cache_hits', 0),
                        'cache_misses': info_optimized.get('cache_misses', 0)
                    })
                    
                except Exception as e:
                    print(f"    Optimized run {i+1} failed: {e}")
            
            # Calculate results for this equation
            if baseline_times and optimized_times:
                baseline_avg = np.mean(baseline_times)
                optimized_avg = np.mean(optimized_times)
                speedup = baseline_avg / optimized_avg
                speedup_pct = (speedup - 1.0) * 100
                
                avg_cache_hit_rate = np.mean([s['cache_hit_rate'] for s in cache_stats])
                total_cache_hits = sum(s['cache_hits'] for s in cache_stats)
                total_cache_misses = sum(s['cache_misses'] for s in cache_stats)
                
                result = {
                    'equation': equation,
                    'baseline_avg_time': baseline_avg,
                    'optimized_avg_time': optimized_avg,
                    'speedup_factor': speedup,
                    'speedup_percentage': speedup_pct,
                    'cache_hit_rate': avg_cache_hit_rate,
                    'total_cache_hits': total_cache_hits,
                    'total_cache_misses': total_cache_misses,
                    'baseline_times': baseline_times,
                    'optimized_times': optimized_times
                }
                
                all_results.append(result)
                
                print(f"    Baseline: {baseline_avg:.4f}s")
                print(f"    Optimized: {optimized_avg:.4f}s") 
                print(f"    Speedup: {speedup:.2f}x ({speedup_pct:.1f}%)")
                print(f"    Cache hit rate: {avg_cache_hit_rate:.1%}")
        
        # Overall analysis
        if all_results:
            overall_speedups = [r['speedup_percentage'] for r in all_results]
            overall_cache_rates = [r['cache_hit_rate'] for r in all_results]
            
            avg_speedup = np.mean(overall_speedups)
            min_speedup = np.min(overall_speedups)
            max_speedup = np.max(overall_speedups)
            avg_cache_rate = np.mean(overall_cache_rates)
            
            print(f"\n" + "="*50)
            print("OVERALL OPTIMIZATION RESULTS:")
            print("="*50)
            print(f"Average speedup:      {avg_speedup:.1f}%")
            print(f"Speedup range:        {min_speedup:.1f}% to {max_speedup:.1f}%")
            print(f"Average cache hit rate: {avg_cache_rate:.1%}")
            
            # Determine success
            target_speedup = 30  # 30% minimum target
            target_met = avg_speedup >= target_speedup
            
            print(f"\nTARGET ASSESSMENT:")
            if target_met:
                print(f"✅ SUCCESS: {avg_speedup:.1f}% average speedup EXCEEDS {target_speedup}% target!")
                status = "success"
            elif avg_speedup >= 20:
                print(f"🟡 PARTIAL: {avg_speedup:.1f}% average speedup approaches {target_speedup}% target")
                status = "partial"
            else:
                print(f"❌ BELOW TARGET: {avg_speedup:.1f}% average speedup below {target_speedup}% target")
                status = "below_target"
            
            # Analysis
            print(f"\nOPTIMIZATION ANALYSIS:")
            if avg_cache_rate > 0.3:
                print(f"+ Good cache effectiveness: {avg_cache_rate:.1%} hit rate")
            else:
                print(f"- Low cache effectiveness: {avg_cache_rate:.1%} hit rate")
                
            if avg_speedup > 0:
                print(f"+ Performance improvement achieved")
            else:
                print(f"- Performance regression detected")
            
            return {
                'status': status,
                'average_speedup_percentage': avg_speedup,
                'speedup_range': [min_speedup, max_speedup],
                'average_cache_hit_rate': avg_cache_rate,
                'target_speedup': target_speedup,
                'target_met': target_met,
                'detailed_results': all_results,
                'summary': {
                    'equations_tested': len(all_results),
                    'iterations_per_test': iterations,
                    'total_tests': len(all_results) * iterations * 2  # baseline + optimized
                }
            }
        
        else:
            print("❌ No successful tests completed")
            return {'status': 'failed', 'error': 'No successful benchmark runs'}


def main():
    """Main test execution."""
    
    print("Energy Caching Optimization Effectiveness Test")
    print("="*50)
    print("Testing implemented optimizations in algebra_inference.py")
    print("Target: 30-50% inference speedup through reduced energy computations")
    
    tester = OptimizationTester()
    results = tester.benchmark_inference_performance(iterations=3)  # Quick test
    
    # Save results
    output_file = 'optimization_effectiveness_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()