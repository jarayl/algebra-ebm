#!/usr/bin/env python3
"""
Benchmark script to measure the tensor pre-allocation optimization impact.

This script compares the performance of the original implementation (allocating
timestep tensors for each compose_energies call) vs the optimized implementation 
(pre-allocating tensors once per landscape).
"""

import torch
import time
import logging
import statistics
from typing import Dict, List, Tuple
from pathlib import Path

# Import existing components
from src.algebra.algebra_encoder import CharacterLevelEncoder, ASTEncoder, EquationDecoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_inference_engine(device='cuda' if torch.cuda.is_available() else 'cpu') -> AlgebraInference:
    """Create a mock inference engine for benchmarking."""
    # Create mock models
    rule_models = {}
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.to(device)
        wrapper.eval()
        rule_models[rule] = wrapper
    
    # Create encoder
    encoder = CharacterLevelEncoder()
    encoder.to(device)
    encoder.eval()
    
    # Create inference engine with small config for fast testing
    config = InferenceConfig(
        K=10,
        max_iterations=20,
        step_size=0.01
    )
    
    inference = AlgebraInference(rule_models, encoder, config=config, device=device)
    return inference


def benchmark_compose_energies_old(
    inference: AlgebraInference,
    inp: torch.Tensor,
    out: torch.Tensor,
    k: int,
    iterations: int = 100
) -> float:
    """Benchmark the OLD way: allocate tensor for each compose_energies call."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        # This simulates the old behavior - always allocate new tensor
        _ = inference.compose_energies(inp, out, k, rule_weights=None, t=None)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    return end_time - start_time


def benchmark_compose_energies_new(
    inference: AlgebraInference,
    inp: torch.Tensor,
    out: torch.Tensor,
    k: int,
    iterations: int = 100
) -> float:
    """Benchmark the NEW way: pre-allocate tensor and reuse."""
    # Pre-allocate timestep tensor (optimization)
    timestep_tensor = torch.full((inp.shape[0],), k, dtype=torch.long, device=inp.device)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        # This uses the optimized behavior - reuse pre-allocated tensor
        _ = inference.compose_energies(inp, out, k, rule_weights=None, t=timestep_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    return end_time - start_time


def run_tensor_allocation_microbenchmark(device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Microbenchmark pure tensor allocation overhead."""
    logger.info("Running tensor allocation microbenchmark...")
    
    batch_size = 32
    k = 5
    iterations = 1000
    
    # Benchmark raw tensor allocation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        t = torch.full((batch_size,), k, dtype=torch.long, device=device)
        # Force actual allocation by accessing tensor
        _ = t.sum().item()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    allocation_time = end_time - start_time
    per_allocation_us = (allocation_time * 1_000_000) / iterations
    
    return {
        'total_time_ms': allocation_time * 1000,
        'per_allocation_us': per_allocation_us,
        'allocations_per_second': iterations / allocation_time
    }


def run_full_inference_benchmark(iterations: int = 5) -> Dict:
    """Benchmark full inference pipeline to estimate real-world impact."""
    logger.info(f"Running full inference benchmark ({iterations} iterations)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = create_mock_inference_engine(device)
    
    # Create test input
    test_equation = "2*x+4=8"
    
    times = []
    for i in range(iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        result = inference.solve_equation(test_equation)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        logger.info(f"Iteration {i+1}: {times[-1]:.4f}s, Success: {result['success']}")
    
    return {
        'mean_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min_time': min(times),
        'max_time': max(times),
        'total_iterations': iterations
    }


def estimate_allocation_impact() -> Dict:
    """Estimate the theoretical impact of tensor pre-allocation optimization."""
    logger.info("Estimating theoretical allocation impact...")
    
    # Analysis parameters based on the code
    K = 10  # Number of landscapes
    max_iterations = 50  # Gradient steps per landscape
    num_rules = 4  # Number of rule models
    
    # Calculate allocation counts
    allocations_per_gradient_step = num_rules  # One per rule model in compose_energies
    total_gradient_steps = K * max_iterations
    total_allocations_old = total_gradient_steps * allocations_per_gradient_step
    total_allocations_new = K  # One pre-allocation per landscape
    
    allocation_reduction = total_allocations_old - total_allocations_new
    reduction_percentage = (allocation_reduction / total_allocations_old) * 100
    
    return {
        'landscapes': K,
        'max_iterations': max_iterations,
        'num_rules': num_rules,
        'allocations_old': total_allocations_old,
        'allocations_new': total_allocations_new,
        'allocation_reduction': allocation_reduction,
        'reduction_percentage': reduction_percentage
    }


def main():
    """Run comprehensive tensor optimization benchmark."""
    logger.info("Starting tensor pre-allocation optimization benchmark...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    results = {}
    
    # 1. Theoretical analysis
    results['theoretical'] = estimate_allocation_impact()
    
    # 2. Microbenchmark tensor allocation overhead
    results['tensor_allocation'] = run_tensor_allocation_microbenchmark(device)
    
    # 3. Benchmark compose_energies calls
    logger.info("Benchmarking compose_energies performance...")
    inference = create_mock_inference_engine(device)
    
    batch_size = 32
    inp = torch.randn(batch_size, 128, device=device)
    out = torch.randn(batch_size, 128, device=device)
    k = 5
    
    # Compare old vs new approach
    old_time = benchmark_compose_energies_old(inference, inp, out, k, iterations=200)
    new_time = benchmark_compose_energies_new(inference, inp, out, k, iterations=200)
    
    speedup_factor = old_time / new_time
    speedup_percentage = ((old_time - new_time) / old_time) * 100
    
    results['compose_energies'] = {
        'old_time_ms': old_time * 1000,
        'new_time_ms': new_time * 1000,
        'speedup_factor': speedup_factor,
        'speedup_percentage': speedup_percentage
    }
    
    # 4. Full inference benchmark
    results['full_inference'] = run_full_inference_benchmark(iterations=5)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("TENSOR PRE-ALLOCATION OPTIMIZATION ANALYSIS")
    print("="*80)
    
    print("\n1. THEORETICAL IMPACT:")
    th = results['theoretical']
    print(f"   Landscapes: {th['landscapes']}")
    print(f"   Max iterations per landscape: {th['max_iterations']}")
    print(f"   Number of rule models: {th['num_rules']}")
    print(f"   Tensor allocations (old): {th['allocations_old']:,}")
    print(f"   Tensor allocations (new): {th['allocations_new']:,}")
    print(f"   Allocation reduction: {th['allocation_reduction']:,} ({th['reduction_percentage']:.1f}%)")
    
    print("\n2. TENSOR ALLOCATION MICROBENCHMARK:")
    ta = results['tensor_allocation']
    print(f"   Time per allocation: {ta['per_allocation_us']:.2f} μs")
    print(f"   Allocations per second: {ta['allocations_per_second']:,.0f}")
    print(f"   Total benchmark time: {ta['total_time_ms']:.2f} ms")
    
    print("\n3. COMPOSE_ENERGIES PERFORMANCE:")
    ce = results['compose_energies']
    print(f"   Old approach: {ce['old_time_ms']:.2f} ms")
    print(f"   New approach: {ce['new_time_ms']:.2f} ms")
    print(f"   Speedup factor: {ce['speedup_factor']:.3f}x")
    print(f"   Speedup percentage: {ce['speedup_percentage']:.2f}%")
    
    print("\n4. FULL INFERENCE BENCHMARK:")
    fi = results['full_inference']
    print(f"   Mean time: {fi['mean_time']:.4f}s")
    print(f"   Std deviation: {fi['std_time']:.4f}s")
    print(f"   Min time: {fi['min_time']:.4f}s")
    print(f"   Max time: {fi['max_time']:.4f}s")
    
    print("\n5. TARGET ACHIEVEMENT ANALYSIS:")
    # Theoretical speedup estimate
    theoretical_time_saved = th['allocation_reduction'] * ta['per_allocation_us'] / 1_000_000  # Convert to seconds
    baseline_inference_time = fi['mean_time']
    theoretical_speedup = (theoretical_time_saved / baseline_inference_time) * 100
    
    print(f"   Theoretical time saved per inference: {theoretical_time_saved*1000:.2f} ms")
    print(f"   Baseline inference time: {baseline_inference_time:.4f}s")
    print(f"   Theoretical speedup: {theoretical_speedup:.2f}%")
    print(f"   Target speedup: 1-2%")
    print(f"   Target achieved: {'YES' if 1.0 <= theoretical_speedup <= 3.0 else 'NO'}")
    
    # Practical compose_energies speedup
    if ce['speedup_percentage'] > 0:
        print(f"   Measured compose_energies speedup: {ce['speedup_percentage']:.2f}%")
        print(f"   Optimization successful: {'YES' if ce['speedup_percentage'] > 0.5 else 'MARGINAL'}")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    main()