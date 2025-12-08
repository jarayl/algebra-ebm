#!/usr/bin/env python3
"""Quick performance test for the optimized dataset generation."""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from src.algebra.algebra_dataset import AlgebraDataset

print("Testing performance after optimization...")

# Test original dataset (baseline)
start = time.time()
original = AlgebraDataset('distribute', num_problems=500, 
                         enable_stratified_sampling=False, 
                         enable_solution_first=False)
original_time = time.time() - start

# Test enhanced dataset 
start = time.time()
enhanced = AlgebraDataset('distribute', num_problems=500,
                         enable_stratified_sampling=True,
                         enable_solution_first=True)
enhanced_time = time.time() - start

overhead = ((enhanced_time - original_time) / original_time) * 100

print(f"Original: {original_time:.3f}s")
print(f"Enhanced: {enhanced_time:.3f}s")
print(f"Overhead: {overhead:.1f}%")

if overhead < 50:
    print("✓ Performance optimization successful")
else:
    print("✗ Still high overhead")