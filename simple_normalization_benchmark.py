#!/usr/bin/env python3
"""
Simple benchmark to demonstrate current normalization performance.
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


def create_test_models(rule_names: List[str], device: str) -> Dict[str, AlgebraDiffusionWrapper]:
    """Create test models for benchmarking."""
    rule_models = {}
    for rule in rule_names:
        ebm = AlgebraEBM(rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        wrapper.to(device)
        wrapper.eval()
        rule_models[rule] = wrapper
    return rule_models


def benchmark_current_implementation():
    """Benchmark the current compose_energies implementation."""
    print("Benchmarking current compose_energies implementation...")
    
    device = 'cpu'  # Use CPU for simplicity
    
    # Create test setup
    rule_names = ['distribute', 'combine', 'isolate', 'divide']
    rule_models = create_test_models(rule_names, device)
    encoder = CharacterLevelEncoder()
    config = InferenceConfig(K=5, max_iterations=10)
    
    inference = AlgebraInference(rule_models, encoder, config=config, device=device)
    
    # Create test inputs
    batch_size = 32
    inp = torch.randn(batch_size, 128, device=device)
    out = torch.randn(batch_size, 128, device=device)
    k = 2
    iterations = 500
    
    print(f"Test configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of rules: {len(rule_names)}")
    print(f"  - Iterations: {iterations}")
    print(f"  - Device: {device}")
    print()
    
    # Benchmark compose_energies with normalization
    print("Benchmarking compose_energies with normalization...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for i in range(iterations):
        energy = inference.compose_energies(inp, out, k, normalize=True)
        if i == 0:
            print(f"  First call energy shape: {energy.shape}")
            print(f"  First call energy sample: {energy[0].item():.4f}")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    norm_time = time.perf_counter() - start_time
    
    # Benchmark compose_energies without normalization
    print("Benchmarking compose_energies without normalization...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for i in range(iterations):
        energy = inference.compose_energies(inp, out, k, normalize=False)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    no_norm_time = time.perf_counter() - start_time
    
    # Calculate overhead
    normalization_overhead = norm_time / no_norm_time
    overhead_percentage = (normalization_overhead - 1.0) * 100
    
    print()
    print("="*60)
    print("RESULTS:")
    print(f"  Without normalization: {no_norm_time*1000:.2f} ms ({no_norm_time*1000/iterations:.3f} ms/call)")
    print(f"  With normalization:    {norm_time*1000:.2f} ms ({norm_time*1000/iterations:.3f} ms/call)")
    print(f"  Normalization overhead: {normalization_overhead:.2f}x ({overhead_percentage:.1f}%)")
    print()
    
    # Target analysis
    print("TARGET ANALYSIS:")
    print(f"  Current overhead: {normalization_overhead:.2f}x")
    print(f"  Target overhead: <2.0x")
    
    if normalization_overhead < 2.0:
        status = "✅ ALREADY MEETING TARGET"
    elif normalization_overhead < 3.0:
        status = "🟡 CLOSE TO TARGET"
    elif normalization_overhead < 5.0:
        status = "🟠 MODERATE OPTIMIZATION NEEDED"
    else:
        status = "❌ SIGNIFICANT OPTIMIZATION NEEDED"
    
    print(f"  Status: {status}")
    print()
    
    # Bottleneck analysis
    print("BOTTLENECK ANALYSIS:")
    overhead_ms = (norm_time - no_norm_time) * 1000
    per_call_overhead = overhead_ms / iterations
    print(f"  Total normalization overhead: {overhead_ms:.2f} ms")
    print(f"  Per-call overhead: {per_call_overhead:.3f} ms")
    print(f"  Operations per call:")
    print(f"    - torch.stack(): 1 call")
    print(f"    - tensor.mean(): 1 call") 
    print(f"    - tensor.std(): 1 call")
    print(f"    - Element-wise ops: 3-4 calls")
    print(f"    - Loop-based weight application: {len(rule_names)} iterations")
    print()
    
    print("OPTIMIZATION OPPORTUNITIES:")
    print("  1. 🎯 Cache normalization statistics when inp/out/rules are same")
    print("  2. 🚀 Vectorize weight application (eliminate loop)")
    print("  3. 📦 Pre-compute weight tensors")
    print("  4. 🧠 Reduce memory allocations")
    print("  5. ⚡ Use more efficient tensor operations")
    
    return {
        'no_norm_time_ms': no_norm_time * 1000,
        'norm_time_ms': norm_time * 1000,
        'overhead': normalization_overhead,
        'overhead_percentage': overhead_percentage,
        'per_call_overhead_ms': per_call_overhead,
        'iterations': iterations,
        'batch_size': batch_size,
        'num_rules': len(rule_names)
    }


if __name__ == "__main__":
    results = benchmark_current_implementation()