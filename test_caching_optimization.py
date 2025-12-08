#!/usr/bin/env python3
"""
Test and optimize energy caching in IRED inference.

Focus on caching improvements independent of model quality.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import logging
from unittest.mock import patch

# Import inference components
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_encoder import CharacterLevelEncoder

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


class CachingProfiler:
    """Test current caching and implement optimizations."""
    
    def __init__(self):
        self.call_counts = {}
        self.timing_data = {}
    
    def create_instrumented_models(self, rule_names: List[str]) -> Dict[str, AlgebraDiffusionWrapper]:
        """Create models with call counting instrumentation."""
        rule_models = {}
        
        for rule in rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.eval()
            
            # Instrument the forward method
            original_forward = wrapper.forward
            call_count_key = f"{rule}_calls"
            self.call_counts[call_count_key] = 0
            
            def create_instrumented_forward(rule_name):
                def instrumented_forward(*args, **kwargs):
                    self.call_counts[f"{rule_name}_calls"] += 1
                    return original_forward(*args, **kwargs)
                return instrumented_forward
            
            wrapper.forward = create_instrumented_forward(rule)
            rule_models[rule] = wrapper
            
        return rule_models
    
    def reset_counters(self):
        """Reset all call counters."""
        for key in self.call_counts:
            self.call_counts[key] = 0
    
    def benchmark_current_implementation(self, iterations: int = 10) -> Dict:
        """Benchmark current caching implementation."""
        
        print("Benchmarking current energy caching implementation...")
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate']
        rule_models = self.create_instrumented_models(rule_names)
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5, max_iterations=10, step_size=0.01)
        inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
        
        # Test equation
        test_equation = "x+1=2"
        
        # Benchmark multiple runs
        times = []
        call_counts = []
        
        for i in range(iterations):
            self.reset_counters()
            
            try:
                with torch.no_grad():
                    inp_embedding = encoder(test_equation).unsqueeze(0)
                
                start_time = time.perf_counter()
                
                # Suppress the flat energy warnings for cleaner output
                with patch('src.algebra.algebra_models.logger.error'):
                    out_embedding, info = inference.ired_inference(inp_embedding)
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                total_calls = sum(self.call_counts.values())
                call_counts.append(total_calls)
                
            except Exception as e:
                print(f"Iteration {i} failed: {e}")
                continue
        
        if not times:
            return {'error': 'All benchmarks failed'}
        
        return {
            'current_implementation': {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_calls': np.mean(call_counts),
                'std_calls': np.std(call_counts),
                'raw_times': times,
                'raw_calls': call_counts
            }
        }


class OptimizedAlgebraInference(AlgebraInference):
    """Optimized version with enhanced energy caching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enhanced caching state
        self.energy_cache = {}
        self.gradient_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _compute_cache_key(self, inp: torch.Tensor, out: torch.Tensor, k: int) -> str:
        """Compute cache key for energy/gradient caching."""
        # Use tensor hashes for cache key (note: this is for profiling, not production)
        inp_hash = hash(tuple(inp.flatten().detach().numpy()))
        out_hash = hash(tuple(out.flatten().detach().numpy()))
        return f"{inp_hash}_{out_hash}_{k}"
    
    def compose_energies_optimized(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights=None,
        t=None
    ) -> torch.Tensor:
        """Enhanced energy composition with better caching."""
        
        # Check cache first
        cache_key = self._compute_cache_key(inp, out, k)
        if cache_key in self.energy_cache:
            self.cache_hits += 1
            return self.energy_cache[cache_key]
        
        # Cache miss - compute energy
        self.cache_misses += 1
        energy = self.compose_energies(inp, out, k, rule_weights, t)
        
        # Store in cache
        self.energy_cache[cache_key] = energy
        
        # Limit cache size to prevent memory issues
        if len(self.energy_cache) > 1000:
            # Remove oldest entries (simple LRU)
            oldest_keys = list(self.energy_cache.keys())[:100]
            for old_key in oldest_keys:
                del self.energy_cache[old_key]
        
        return energy
    
    def compute_energy_and_gradient_optimized(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights=None,
        t=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized energy and gradient computation with caching."""
        
        # Check if we have both energy and gradient cached
        cache_key = self._compute_cache_key(inp, out, k)
        
        if cache_key in self.energy_cache and cache_key in self.gradient_cache:
            self.cache_hits += 2  # Both energy and gradient from cache
            return self.energy_cache[cache_key], self.gradient_cache[cache_key]
        
        # Compute both (existing logic)
        energy, grad = self.compute_energy_and_gradient(inp, out, k, rule_weights, t)
        
        # Cache both results
        self.energy_cache[cache_key] = energy
        self.gradient_cache[cache_key] = grad
        
        # Manage cache size
        if len(self.energy_cache) > 1000:
            oldest_keys = list(self.energy_cache.keys())[:100]
            for old_key in oldest_keys:
                self.energy_cache.pop(old_key, None)
                self.gradient_cache.pop(old_key, None)
        
        return energy, grad
    
    def ired_inference_optimized(self, inp_embedding: torch.Tensor, config=None, rule_weights=None):
        """IRED inference with enhanced caching optimizations."""
        
        # Reset cache for new inference
        self.energy_cache.clear()
        self.gradient_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Use provided config or fall back to instance config
        if config is None:
            config = self.config
        else:
            if config.K != len(self.alphas_cumprod):
                raise ValueError(f"config.K={config.K} does not match precomputed alphas_cumprod length={len(self.alphas_cumprod)}")
        
        # Input validation (same as original)
        if len(inp_embedding.shape) != 2 or inp_embedding.shape[1] != 128:
            raise ValueError(f"inp_embedding must have shape (B, 128), got {inp_embedding.shape}")
        if len(self.rule_models) == 0:
            raise ValueError("No rule models loaded")
        if not torch.is_tensor(inp_embedding):
            raise TypeError(f"inp_embedding must be torch.Tensor, got {type(inp_embedding)}")
        if torch.isnan(inp_embedding).any():
            raise ValueError("inp_embedding contains NaN values")
        if torch.isinf(inp_embedding).any():
            raise ValueError("inp_embedding contains Inf values")
        
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
        
        # OPTIMIZATION 1: Pre-allocate tensors to avoid repeated allocation
        timestep_tensors = {}
        for k in range(config.K):
            timestep_tensors[k] = torch.full((batch_size,), k, dtype=torch.long, device=inp_embedding.device)
        
        # OPTIMIZATION 2: Enhanced energy caching with per-landscape cache persistence
        landscape_energy_cache = {}
        
        # Iterate through K landscapes
        for k in range(config.K):
            sigma_k = torch.sqrt(1 - self.alphas_cumprod[k]).item()
            current_step_size = config.get_adaptive_step_size(k)
            info['step_sizes'].append(current_step_size)
            
            # Get pre-allocated tensor
            timestep_tensor = timestep_tensors[k]
            
            # OPTIMIZATION 3: Track energy for current state
            current_energy = None
            
            # Gradient descent steps in this landscape
            for t in range(config.max_iterations):
                
                # OPTIMIZATION 4: Reuse previous energy computation when possible
                if current_energy is None:
                    # Need to compute energy for current state
                    current_energy = self.compose_energies_optimized(
                        inp_embedding, out, k, rule_weights, timestep_tensor
                    )
                
                energy_before_val = current_energy.item()
                info['energy_history'].append(energy_before_val)
                
                # Compute gradient (this always requires fresh computation due to autograd)
                grad = self.compute_composed_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                grad_norm = torch.norm(grad).item()
                info['gradient_norms'].append(grad_norm)
                
                # Safety checks (same as original)
                if grad_norm > 100.0:
                    logger.warning(f"Gradient explosion at landscape {k}, step {t}")
                    info['convergence_reason'] = f'gradient_explosion_k{k}_t{t}'
                    break
                
                # Energy stagnation detection
                if len(info['energy_history']) >= 10:
                    recent_energies = info['energy_history'][-10:]
                    energy_std = torch.tensor(recent_energies).std().item()
                    if energy_std < 1e-6 and grad_norm < 1e-4:
                        logger.info(f"Convergence at landscape {k}, step {t}")
                        info['convergence_reason'] = f'converged_k{k}_t{t}'
                        break
                
                # Gradient descent step
                out_new = out - current_step_size * grad
                
                # OPTIMIZATION 5: Compute energy for new state
                energy_after = self.compose_energies_optimized(
                    inp_embedding, out_new, k, rule_weights, timestep_tensor
                )
                energy_after_val = energy_after.item()
                delta_E = energy_after_val - energy_before_val
                
                # Metropolis acceptance (same logic as original)
                LANDSCAPE_DECAY = -0.05
                ITERATION_DECAY = -0.02
                MIN_TEMPERATURE = 0.1
                MAX_ENERGY_DELTA_MULTIPLIER = 50.0
                
                temperature = 1.0 * np.exp(LANDSCAPE_DECAY * k) * np.exp(ITERATION_DECAY * t / config.max_iterations)
                temperature = max(temperature, MIN_TEMPERATURE)
                
                if delta_E <= 0:
                    accept_prob = 1.0
                else:
                    clipped_delta_E = min(delta_E, MAX_ENERGY_DELTA_MULTIPLIER * temperature)
                    accept_prob = np.exp(-clipped_delta_E / temperature)
                
                import random
                accepted = random.random() < accept_prob
                
                if accepted:
                    out = out_new.detach().requires_grad_(True)
                    current_energy = energy_after  # OPTIMIZATION 6: Carry forward the computed energy
                    info['accepted_steps'] += 1
                    
                    if config.should_early_stop(energy_after_val):
                        logger.debug(f"Early stopping at landscape {k}, step {t}")
                        break
                else:
                    # Step rejected - current_energy stays the same (no recomputation needed)
                    pass
                
                info['total_steps'] += 1
            
            info['landscape_transitions'].append(k)
            
            # Check for convergence between landscapes
            if k >= 2 and len(info['gradient_norms']) >= 20:
                recent_grads = info['gradient_norms'][-20:]
                avg_grad_norm = sum(recent_grads) / len(recent_grads)
                max_grad_norm = max(recent_grads)
                
                recent_energies = info['energy_history'][-20:]
                energy_range = max(recent_energies) - min(recent_energies)
                
                if avg_grad_norm < 1e-3 and max_grad_norm < 1e-2 and energy_range < 0.01:
                    logger.info(f"Overall convergence achieved after landscape {k}")
                    info['convergence_reason'] = f'overall_convergence_k{k}'
                    break
            
            if 'convergence_reason' in info:
                break
            
            # Scale for next landscape (same as original)
            if k < config.K - 1:
                sigma_k_next = torch.sqrt(1 - self.alphas_cumprod[k + 1]).item()
                
                if sigma_k > 1e-8:
                    scale_factor = sigma_k_next / sigma_k
                else:
                    scale_factor = 1.0
                
                out = out.detach() * scale_factor
                out = out.requires_grad_(True)
                current_energy = None  # Invalidate energy cache due to scaling
        
        # Final statistics
        final_k = k
        final_timestep_tensor = torch.full((batch_size,), final_k, dtype=torch.long, device=inp_embedding.device)
        info['final_energy'] = self.compose_energies(inp_embedding, out, final_k, rule_weights, final_timestep_tensor).item()
        info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
        
        # Cache statistics
        info['cache_hits'] = self.cache_hits
        info['cache_misses'] = self.cache_misses
        info['cache_hit_rate'] = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        return out.detach(), info


def benchmark_optimizations():
    """Benchmark original vs optimized implementations."""
    
    print("\n" + "="*60)
    print("ENERGY CACHING OPTIMIZATION BENCHMARK")
    print("="*60)
    
    # Setup
    rule_names = ['distribute', 'combine']
    encoder = CharacterLevelEncoder()
    config = InferenceConfig(K=3, max_iterations=5, step_size=0.01)  # Small for testing
    test_equation = "x+1=2"
    
    # Create models
    rule_models = {}
    for rule in rule_names:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.eval()
        rule_models[rule] = wrapper
    
    # Benchmark original implementation
    print("\n1. Benchmarking ORIGINAL implementation...")
    original_inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
    
    original_times = []
    iterations = 5
    
    for i in range(iterations):
        try:
            with torch.no_grad():
                inp_embedding = encoder(test_equation).unsqueeze(0)
            
            start_time = time.perf_counter()
            
            with patch('src.algebra.algebra_models.logger.error'):  # Suppress flat energy warnings
                out_embedding, info = original_inference.ired_inference(inp_embedding)
            
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
            
        except Exception as e:
            print(f"Original iteration {i} failed: {e}")
    
    # Benchmark optimized implementation
    print("2. Benchmarking OPTIMIZED implementation...")
    optimized_inference = OptimizedAlgebraInference(rule_models, encoder, config=config, device='cpu')
    
    optimized_times = []
    cache_stats = []
    
    for i in range(iterations):
        try:
            with torch.no_grad():
                inp_embedding = encoder(test_equation).unsqueeze(0)
            
            start_time = time.perf_counter()
            
            with patch('src.algebra.algebra_models.logger.error'):  # Suppress flat energy warnings
                out_embedding, info = optimized_inference.ired_inference_optimized(inp_embedding)
            
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
            cache_stats.append({
                'cache_hits': info.get('cache_hits', 0),
                'cache_misses': info.get('cache_misses', 0),
                'cache_hit_rate': info.get('cache_hit_rate', 0)
            })
            
        except Exception as e:
            print(f"Optimized iteration {i} failed: {e}")
    
    # Calculate results
    if original_times and optimized_times:
        original_avg = np.mean(original_times)
        optimized_avg = np.mean(optimized_times)
        speedup = original_avg / optimized_avg
        speedup_pct = (speedup - 1.0) * 100
        
        avg_cache_hit_rate = np.mean([s['cache_hit_rate'] for s in cache_stats if s['cache_hit_rate'] > 0])
        
        print(f"\n" + "="*40)
        print("RESULTS:")
        print(f"Original avg time:    {original_avg:.4f}s")
        print(f"Optimized avg time:   {optimized_avg:.4f}s")
        print(f"Speedup factor:       {speedup:.2f}x")
        print(f"Speedup percentage:   {speedup_pct:.1f}%")
        print(f"Avg cache hit rate:   {avg_cache_hit_rate*100:.1f}%")
        
        # Determine if target is met
        target_speedup = 30  # 30% minimum
        
        if speedup_pct >= target_speedup:
            print(f"\n✅ SUCCESS: {speedup_pct:.1f}% speedup exceeds {target_speedup}% target!")
            status = "success"
        elif speedup_pct >= 20:
            print(f"\n🟡 PARTIAL: {speedup_pct:.1f}% speedup approaches {target_speedup}% target")
            status = "partial"
        else:
            print(f"\n❌ BELOW TARGET: {speedup_pct:.1f}% speedup below {target_speedup}% target")
            status = "below_target"
        
        return {
            'status': status,
            'original_time': original_avg,
            'optimized_time': optimized_avg,
            'speedup_factor': speedup,
            'speedup_percentage': speedup_pct,
            'target_speedup': target_speedup,
            'cache_hit_rate': avg_cache_hit_rate,
            'raw_data': {
                'original_times': original_times,
                'optimized_times': optimized_times,
                'cache_stats': cache_stats
            }
        }
    
    else:
        print("❌ BENCHMARK FAILED: Unable to collect timing data")
        return {'status': 'failed', 'error': 'No timing data collected'}


def main():
    """Main optimization testing."""
    
    print("Testing Energy Caching Optimizations")
    print("="*50)
    
    # Run benchmarks
    results = benchmark_optimizations()
    
    # Save results
    with open('energy_caching_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: energy_caching_optimization_results.json")
    
    return results


if __name__ == "__main__":
    import numpy as np
    main()