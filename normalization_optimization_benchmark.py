#!/usr/bin/env python3
"""
Comprehensive benchmark for normalization performance optimization.

This script measures the performance impact of the optimized compose_energies method
with vectorized operations, caching, and reduced tensor allocations.

Key optimizations tested:
1. Normalization statistics caching
2. Vectorized tensor operations 
3. Pre-computed weight tensors
4. Reduced memory allocations
"""

import torch
import time
import logging
import statistics
import json
from typing import Dict, List, Tuple
from pathlib import Path
from unittest.mock import patch

# Import existing components
from src.algebra.algebra_encoder import CharacterLevelEncoder, ASTEncoder, EquationDecoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig

# Set minimal logging for cleaner output
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmark for normalization optimization."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def create_test_models(self, rule_names: List[str]) -> Dict[str, AlgebraDiffusionWrapper]:
        """Create test models for benchmarking."""
        rule_models = {}
        for rule in rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.to(self.device)
            wrapper.eval()
            rule_models[rule] = wrapper
        return rule_models
    
    def create_baseline_compose_energies(self, inference: AlgebraInference):
        """Create baseline compose_energies function without optimizations."""
        
        def baseline_compose_energies(
            inp: torch.Tensor,
            out: torch.Tensor, 
            k: int,
            rule_weights=None,
            t=None,
            normalize: bool = True,
            calibration_scales=None
        ):
            """Original implementation without optimizations."""
            if rule_weights is None:
                rule_weights = {rule: 1.0 for rule in inference.rule_models.keys()}
            
            # Use pre-allocated tensor if provided, otherwise allocate new one
            if t is None:
                t = torch.full((inp.shape[0],), k, dtype=torch.long, device=inp.device)
            
            # Collect all individual energies for normalization (original order)
            individual_energies = []
            rule_names = list(inference.rule_models.keys())
            
            for rule_name in rule_names:
                model = inference.rule_models[rule_name]
                energy = model(inp, out, t, return_energy=True)  # (B, 1)
                
                # Apply calibration scales if provided
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
            
            # Handle single rule case - no normalization needed
            if len(individual_energies) == 1:
                weight = rule_weights.get(rule_names[0], 1.0)
                return weight * individual_energies[0]
            
            # Stack energies for efficient normalization (original implementation)
            energy_tensor = torch.stack(individual_energies, dim=-1)  # (B, 1, N_rules)
            
            # Compute z-score normalization: (x - μ) / σ (original implementation)
            mean = energy_tensor.mean(dim=-1, keepdim=True)  # (B, 1, 1)
            std = energy_tensor.std(dim=-1, keepdim=True, unbiased=False)  # (B, 1, 1)
            
            # Add epsilon for numerical stability
            epsilon = 1e-6
            std_safe = std + epsilon
            
            # Apply z-score normalization
            normalized_energies = (energy_tensor - mean) / std_safe  # (B, 1, N_rules)
            
            # Re-scale to target range [1.0, 15.0]
            target_min, target_max = 1.0, 15.0
            target_scale = (target_max - target_min) / 4.0
            target_offset = (target_min + target_max) / 2.0
            
            rescaled_energies = target_scale * normalized_energies + target_offset  # (B, 1, N_rules)
            
            # Apply rule weights to normalized energies and sum (original loop-based approach)
            total_energy = torch.zeros(inp.shape[0], 1, device=inp.device, dtype=inp.dtype)
            for i, rule_name in enumerate(rule_names):
                weight = rule_weights.get(rule_name, 1.0)
                total_energy += weight * rescaled_energies[:, :, i]  # (B, 1)
            
            return total_energy
        
        return baseline_compose_energies
    
    def benchmark_compose_energies_microbench(self, iterations: int = 1000) -> Dict:
        """Microbenchmark for compose_energies performance."""
        print(f"Running compose_energies microbenchmark ({iterations} iterations)...")
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate', 'divide']
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5, max_iterations=10)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device=self.device)
        baseline_fn = self.create_baseline_compose_energies(inference)
        
        # Create test inputs
        batch_size = 32
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        t = torch.full((batch_size,), k, dtype=torch.long, device=self.device)
        
        # Benchmark baseline implementation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            baseline_energy = baseline_fn(inp, out, k, t=t)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        baseline_time = time.perf_counter() - start_time
        
        # Benchmark optimized implementation (with cache)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            optimized_energy = inference.compose_energies(inp, out, k, t=t)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        optimized_time = time.perf_counter() - start_time
        
        # Benchmark optimized implementation (without cache for fair comparison)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            no_cache_energy = inference.compose_energies(inp, out, k, t=t, _skip_cache=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_cache_time = time.perf_counter() - start_time
        
        # Validate correctness
        energy_diff = torch.abs(baseline_energy - optimized_energy).max().item()
        correctness_ok = energy_diff < 1e-5
        
        speedup_with_cache = baseline_time / optimized_time
        speedup_without_cache = baseline_time / no_cache_time
        
        return {
            'baseline_time_ms': baseline_time * 1000,
            'optimized_time_ms': optimized_time * 1000, 
            'no_cache_time_ms': no_cache_time * 1000,
            'speedup_with_cache': speedup_with_cache,
            'speedup_without_cache': speedup_without_cache,
            'speedup_percentage_with_cache': (speedup_with_cache - 1.0) * 100,
            'speedup_percentage_without_cache': (speedup_without_cache - 1.0) * 100,
            'correctness_check': correctness_ok,
            'max_energy_diff': energy_diff,
            'iterations': iterations
        }
    
    def benchmark_normalization_overhead(self, iterations: int = 500) -> Dict:
        """Measure normalization overhead specifically."""
        print(f"Benchmarking normalization overhead ({iterations} iterations)...")
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate', 'divide']
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5, max_iterations=10)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device=self.device)
        
        # Create test inputs
        batch_size = 32
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        t = torch.full((batch_size,), k, dtype=torch.long, device=self.device)
        
        # Benchmark without normalization
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            no_norm_energy = inference.compose_energies(inp, out, k, t=t, normalize=False)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_norm_time = time.perf_counter() - start_time
        
        # Benchmark with normalization (optimized)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            norm_energy = inference.compose_energies(inp, out, k, t=t, normalize=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        norm_time = time.perf_counter() - start_time
        
        normalization_overhead = norm_time / no_norm_time
        
        return {
            'no_normalization_time_ms': no_norm_time * 1000,
            'with_normalization_time_ms': norm_time * 1000,
            'normalization_overhead': normalization_overhead,
            'normalization_overhead_percentage': (normalization_overhead - 1.0) * 100,
            'iterations': iterations
        }
    
    def benchmark_caching_effectiveness(self, iterations: int = 200) -> Dict:
        """Measure caching effectiveness across repeated calls."""
        print(f"Benchmarking caching effectiveness ({iterations} iterations)...")
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate']
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=3, max_iterations=5)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device=self.device)
        
        # Create test inputs that will hit the cache
        batch_size = 16
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 1
        t = torch.full((batch_size,), k, dtype=torch.long, device=self.device)
        
        # Cold start - first call will miss cache
        _ = inference.compose_energies(inp, out, k, t=t)
        
        # Benchmark with warm cache
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Use same inputs to hit cache
            cached_energy = inference.compose_energies(inp, out, k, t=t)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cached_time = time.perf_counter() - start_time
        
        # Benchmark with cache disabled  
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            no_cache_energy = inference.compose_energies(inp, out, k, t=t, _skip_cache=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_cache_time = time.perf_counter() - start_time
        
        cache_speedup = no_cache_time / cached_time
        
        return {
            'cached_time_ms': cached_time * 1000,
            'no_cache_time_ms': no_cache_time * 1000,
            'cache_speedup': cache_speedup,
            'cache_speedup_percentage': (cache_speedup - 1.0) * 100,
            'iterations': iterations
        }
    
    def benchmark_full_inference_impact(self, iterations: int = 5) -> Dict:
        """Measure impact on full inference pipeline."""
        print(f"Benchmarking full inference impact ({iterations} iterations)...")
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate']
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        
        # Use smaller config for faster testing
        config = InferenceConfig(K=3, max_iterations=10, step_size=0.01)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device=self.device)
        
        # Test equation
        test_equation = "x+2=5"
        
        # Benchmark multiple runs
        times = []
        cache_hit_rates = []
        
        for i in range(iterations):
            try:
                with torch.no_grad():
                    inp_embedding = encoder(test_equation).unsqueeze(0)
                    inp_embedding = inp_embedding.to(self.device)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                # Suppress logging for cleaner output
                with patch('src.algebra.algebra_models.logger.error'):
                    out_embedding, info = inference.ired_inference(inp_embedding)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                cache_hit_rates.append(info.get('cache_hit_rate', 0.0))
                
            except Exception as e:
                print(f"Full inference iteration {i} failed: {e}")
                continue
        
        if not times:
            return {'error': 'All benchmarks failed'}
        
        return {
            'avg_time': statistics.mean(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'avg_cache_hit_rate': statistics.mean(cache_hit_rates),
            'iterations_completed': len(times),
            'raw_times': times
        }
    
    def analyze_memory_usage(self) -> Dict:
        """Analyze memory usage patterns."""
        print("Analyzing memory usage...")
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory analysis'}
        
        # Create test setup
        rule_names = ['distribute', 'combine', 'isolate', 'divide']
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=5, max_iterations=10)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device=self.device)
        baseline_fn = self.create_baseline_compose_energies(inference)
        
        # Test inputs
        batch_size = 32
        inp = torch.randn(batch_size, 128, device=self.device)
        out = torch.randn(batch_size, 128, device=self.device)
        k = 2
        t = torch.full((batch_size,), k, dtype=torch.long, device=self.device)
        
        # Measure baseline memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        for _ in range(100):
            baseline_energy = baseline_fn(inp, out, k, t=t)
        
        baseline_memory = torch.cuda.max_memory_allocated()
        
        # Measure optimized memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        for _ in range(100):
            optimized_energy = inference.compose_energies(inp, out, k, t=t)
        
        optimized_memory = torch.cuda.max_memory_allocated()
        
        memory_reduction = (baseline_memory - optimized_memory) / baseline_memory
        
        return {
            'baseline_memory_mb': baseline_memory / (1024 * 1024),
            'optimized_memory_mb': optimized_memory / (1024 * 1024),
            'memory_reduction_percentage': memory_reduction * 100,
            'memory_savings_mb': (baseline_memory - optimized_memory) / (1024 * 1024)
        }


def main():
    """Run comprehensive normalization optimization benchmark."""
    print("="*80)
    print("NORMALIZATION PERFORMANCE OPTIMIZATION BENCHMARK")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    benchmark = PerformanceBenchmark(device)
    results = {}
    
    # 1. Microbenchmark compose_energies
    print("\n" + "1. COMPOSE_ENERGIES MICROBENCHMARK".ljust(60, '-'))
    results['compose_energies'] = benchmark.benchmark_compose_energies_microbench(iterations=1000)
    
    # 2. Normalization overhead
    print("\n" + "2. NORMALIZATION OVERHEAD ANALYSIS".ljust(60, '-'))
    results['normalization_overhead'] = benchmark.benchmark_normalization_overhead(iterations=500)
    
    # 3. Caching effectiveness
    print("\n" + "3. CACHING EFFECTIVENESS".ljust(60, '-'))
    results['caching'] = benchmark.benchmark_caching_effectiveness(iterations=200)
    
    # 4. Full inference impact
    print("\n" + "4. FULL INFERENCE IMPACT".ljust(60, '-'))
    results['full_inference'] = benchmark.benchmark_full_inference_impact(iterations=5)
    
    # 5. Memory analysis
    if torch.cuda.is_available():
        print("\n" + "5. MEMORY USAGE ANALYSIS".ljust(60, '-'))
        results['memory'] = benchmark.analyze_memory_usage()
    
    # Print results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Compose energies results
    ce = results['compose_energies']
    print(f"\n🚀 COMPOSE_ENERGIES PERFORMANCE:")
    print(f"   Baseline time:               {ce['baseline_time_ms']:.2f} ms")
    print(f"   Optimized time (with cache): {ce['optimized_time_ms']:.2f} ms")
    print(f"   Optimized time (no cache):   {ce['no_cache_time_ms']:.2f} ms")
    print(f"   Speedup (with cache):        {ce['speedup_with_cache']:.2f}x ({ce['speedup_percentage_with_cache']:.1f}%)")
    print(f"   Speedup (no cache):          {ce['speedup_without_cache']:.2f}x ({ce['speedup_percentage_without_cache']:.1f}%)")
    print(f"   Correctness check:           {'✅ PASS' if ce['correctness_check'] else '❌ FAIL'}")
    
    # Normalization overhead
    no = results['normalization_overhead']
    print(f"\n📊 NORMALIZATION OVERHEAD:")
    print(f"   Without normalization:       {no['no_normalization_time_ms']:.2f} ms")
    print(f"   With normalization:          {no['with_normalization_time_ms']:.2f} ms")
    print(f"   Normalization overhead:      {no['normalization_overhead']:.2f}x ({no['normalization_overhead_percentage']:.1f}%)")
    
    # Caching effectiveness
    cache = results['caching']
    print(f"\n💾 CACHING EFFECTIVENESS:")
    print(f"   Cached calls time:           {cache['cached_time_ms']:.2f} ms")
    print(f"   No cache time:               {cache['no_cache_time_ms']:.2f} ms")
    print(f"   Cache speedup:               {cache['cache_speedup']:.2f}x ({cache['cache_speedup_percentage']:.1f}%)")
    
    # Full inference impact
    fi = results['full_inference']
    if 'error' not in fi:
        print(f"\n🎯 FULL INFERENCE IMPACT:")
        print(f"   Average inference time:      {fi['avg_time']:.4f}s")
        print(f"   Standard deviation:          {fi['std_time']:.4f}s")
        print(f"   Min/Max time:                {fi['min_time']:.4f}s / {fi['max_time']:.4f}s")
        print(f"   Average cache hit rate:      {fi['avg_cache_hit_rate']*100:.1f}%")
    
    # Memory usage
    if 'memory' in results and 'error' not in results['memory']:
        mem = results['memory']
        print(f"\n🧠 MEMORY USAGE:")
        print(f"   Baseline memory:             {mem['baseline_memory_mb']:.2f} MB")
        print(f"   Optimized memory:            {mem['optimized_memory_mb']:.2f} MB")
        print(f"   Memory reduction:            {mem['memory_reduction_percentage']:.1f}%")
        print(f"   Memory savings:              {mem['memory_savings_mb']:.2f} MB")
    
    # Target achievement analysis
    print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")
    target_reduction = 80  # Target: reduce 5-10x overhead to <2x overhead
    
    # Current overhead is normalization_overhead, target is <2x
    current_overhead = no['normalization_overhead']
    if current_overhead < 2.0:
        overhead_status = "✅ TARGET MET"
    elif current_overhead < 3.0:
        overhead_status = "🟡 CLOSE TO TARGET"
    else:
        overhead_status = "❌ NEEDS MORE WORK"
    
    print(f"   Current normalization overhead: {current_overhead:.2f}x")
    print(f"   Target overhead:                <2.0x")
    print(f"   Status:                         {overhead_status}")
    
    # Overall speedup
    overall_speedup = ce['speedup_percentage_with_cache']
    if overall_speedup >= 50:
        speedup_status = "✅ EXCELLENT"
    elif overall_speedup >= 30:
        speedup_status = "✅ GOOD"
    elif overall_speedup >= 10:
        speedup_status = "🟡 MODERATE"
    else:
        speedup_status = "❌ INSUFFICIENT"
    
    print(f"   Overall compose_energies speedup: {overall_speedup:.1f}%")
    print(f"   Status:                           {speedup_status}")
    
    print("\n" + "="*80)
    
    # Save detailed results
    with open('normalization_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📝 Detailed results saved to: normalization_optimization_results.json")
    
    return results


if __name__ == "__main__":
    main()