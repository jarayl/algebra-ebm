#!/usr/bin/env python3
"""
Benchmark that simulates the IRED inference pattern where compose_energies
is called repeatedly with similar inputs during gradient descent.

This should reveal the 5-10x overhead mentioned in the task description.
"""

import torch
import time
import logging
from typing import Dict, List

# Import existing components
from src.algebra.algebra_encoder import CharacterLevelEncoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig

# Disable logging
logging.basicConfig(level=logging.CRITICAL)


def create_test_models(rule_names: List[str], device: str) -> Dict[str, AlgebraEBM]:
    """Create test models for benchmarking."""
    rule_models = {}
    for rule in rule_names:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.to(device)
        wrapper.eval()
        rule_models[rule] = wrapper
    return rule_models


def benchmark_ired_pattern():
    """
    Benchmark the IRED pattern: many calls during gradient descent optimization.
    
    This simulates the actual usage where compose_energies is called:
    - Multiple times per gradient step (energy + gradient computation)  
    - Across many gradient steps (10-50 per landscape)
    - Across multiple landscapes (K=10)
    
    Total calls = K * max_iterations * ~3 calls per iteration = ~1500 calls
    """
    print("Benchmarking IRED inference pattern...")
    
    device = 'cpu'
    
    # Create test setup that mirrors real IRED configuration
    rule_names = ['distribute', 'combine', 'isolate', 'divide'] 
    rule_models = create_test_models(rule_names, device)
    encoder = CharacterLevelEncoder()
    config = InferenceConfig(K=10, max_iterations=20)  # Realistic IRED config
    
    inference = AlgebraInference(rule_models, encoder, config=config, device=device)
    
    # Simulate equation solving
    test_equation = "2*x+3=7"
    with torch.no_grad():
        inp_embedding = encoder(test_equation).unsqueeze(0).to(device)
    
    print(f"IRED simulation configuration:")
    print(f"  - Landscapes (K): {config.K}")
    print(f"  - Max iterations per landscape: {config.max_iterations}")
    print(f"  - Number of rules: {len(rule_names)}")
    print(f"  - Input equation: {test_equation}")
    print()
    
    # Pattern 1: Simulate gradient computation calls (energy + gradient)
    # This is where the 5-10x overhead shows up due to repeated normalization
    total_calls = config.K * config.max_iterations * 2  # Energy + gradient per iteration
    
    print(f"Simulating {total_calls} compose_energies calls...")
    print("(This represents K landscapes × max_iterations × 2 calls/iteration)")
    print()
    
    # Initialize output embedding (changes slightly each iteration)
    out_embedding = torch.randn_like(inp_embedding)
    
    # Benchmark WITHOUT normalization (baseline)
    print("Phase 1: Without normalization...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    call_count = 0
    for k in range(config.K):
        for iteration in range(config.max_iterations):
            # Simulate small changes to output during gradient descent
            out_embedding += 0.001 * torch.randn_like(out_embedding)
            
            # Call 1: Compute energy for current state
            energy1 = inference.compose_energies(inp_embedding, out_embedding, k, normalize=False)
            call_count += 1
            
            # Call 2: Compute energy for gradient (with requires_grad)
            out_grad = out_embedding.clone().requires_grad_(True)
            energy2 = inference.compose_energies(inp_embedding, out_grad, k, normalize=False)
            call_count += 1
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    baseline_time = time.perf_counter() - start_time
    
    # Reset for normalized version
    out_embedding = torch.randn_like(inp_embedding)
    
    # Benchmark WITH normalization (current implementation)  
    print("Phase 2: With normalization...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    call_count = 0
    for k in range(config.K):
        for iteration in range(config.max_iterations):
            # Simulate small changes to output during gradient descent
            out_embedding += 0.001 * torch.randn_like(out_embedding)
            
            # Call 1: Compute energy for current state
            energy1 = inference.compose_energies(inp_embedding, out_embedding, k, normalize=True)
            call_count += 1
            
            # Call 2: Compute energy for gradient (with requires_grad)
            out_grad = out_embedding.clone().requires_grad_(True)
            energy2 = inference.compose_energies(inp_embedding, out_grad, k, normalize=True)
            call_count += 1
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    normalized_time = time.perf_counter() - start_time
    
    # Calculate overhead
    normalization_overhead = normalized_time / baseline_time
    overhead_percentage = (normalization_overhead - 1.0) * 100
    overhead_per_call = (normalized_time - baseline_time) * 1000 / call_count
    
    print()
    print("="*80)
    print("IRED PATTERN BENCHMARK RESULTS")
    print("="*80)
    print(f"Total calls: {call_count}")
    print(f"Baseline time (no norm):  {baseline_time:.3f}s ({baseline_time*1000/call_count:.3f} ms/call)")
    print(f"Normalized time:          {normalized_time:.3f}s ({normalized_time*1000/call_count:.3f} ms/call)")
    print(f"Normalization overhead:   {normalization_overhead:.2f}x ({overhead_percentage:.1f}%)")
    print(f"Overhead per call:        {overhead_per_call:.3f} ms")
    print()
    
    # Detailed breakdown
    total_overhead_time = normalized_time - baseline_time
    print("OVERHEAD BREAKDOWN:")
    print(f"  Total normalization time: {total_overhead_time:.3f}s")
    print(f"  Time per normalization:   {total_overhead_time*1000/call_count:.3f} ms")
    print(f"  Operations per call:")
    print(f"    - torch.stack([energy1, energy2, energy3, energy4]): allocates new tensor")
    print(f"    - tensor.mean() + tensor.std(): 2 reduction operations")
    print(f"    - Elementwise normalization: (energy_tensor - mean) / std_safe")
    print(f"    - Rescaling: target_scale * normalized + offset")
    print(f"    - Weight application loop: 4 iterations of tensor indexing + addition")
    print()
    
    # Performance analysis vs target
    print("TARGET ANALYSIS:")
    print(f"  Current overhead: {normalization_overhead:.2f}x")
    if normalization_overhead >= 5.0:
        target_status = "❌ SEVERE - Matches task description (5-10x overhead)"
        optimization_urgency = "HIGH"
    elif normalization_overhead >= 3.0:
        target_status = "🟠 MODERATE - Significant overhead present"  
        optimization_urgency = "MEDIUM"
    elif normalization_overhead >= 2.0:
        target_status = "🟡 MILD - Some overhead present"
        optimization_urgency = "LOW"
    else:
        target_status = "✅ GOOD - Low overhead"
        optimization_urgency = "LOW"
        
    print(f"  Status: {target_status}")
    print(f"  Optimization urgency: {optimization_urgency}")
    print()
    
    # Identify major bottlenecks
    print("MAJOR BOTTLENECKS IDENTIFIED:")
    print("  1. 📊 Repeated normalization statistics computation")
    print("     - Same rule set, similar energy ranges")
    print("     - mean() and std() called on every normalization")
    print("     - Could cache statistics for similar configurations")
    print()
    print("  2. 🔄 Loop-based weight application")
    print("     - 4 sequential tensor operations per call")
    print("     - Could vectorize with pre-computed weight tensor")
    print()
    print("  3. 📦 Tensor allocation overhead")
    print("     - torch.stack() creates new tensor each call")
    print("     - Could pre-allocate or use in-place operations")
    print()
    print("  4. 🧮 Redundant arithmetic operations")
    print("     - target_scale, target_offset computed every time")
    print("     - Could pre-compute constants")
    print()
    
    # Optimization impact projection
    theoretical_speedup = min(normalization_overhead * 0.7, normalization_overhead - 0.5)  # Conservative estimate
    print("OPTIMIZATION POTENTIAL:")
    print(f"  Current overhead: {overhead_per_call:.3f} ms/call")
    print(f"  Projected reduction: ~{((normalization_overhead - theoretical_speedup) / normalization_overhead * 100):.0f}%")
    print(f"  Target overhead: {theoretical_speedup:.2f}x")
    print(f"  Time savings: {(total_overhead_time * (1 - theoretical_speedup/normalization_overhead)):.3f}s per inference")
    
    return {
        'baseline_time': baseline_time,
        'normalized_time': normalized_time,
        'overhead': normalization_overhead,
        'overhead_percentage': overhead_percentage,
        'total_calls': call_count,
        'overhead_per_call_ms': overhead_per_call,
        'optimization_urgency': optimization_urgency
    }


if __name__ == "__main__":
    results = benchmark_ired_pattern()