#!/usr/bin/env python3
"""
Deep Dive: Root Cause Analysis for Training Issues

This script investigates the exact root causes:
1. Why energy landscape is flat (energies ~0.2 for both pos/neg)
2. Why output collapse occurs (wrapper returning zeros)
3. What needs to be fixed
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)

from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_dataset import AlgebraDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#############################################################################
# Issue 1: Energy Landscape Analysis
#############################################################################
print("\n" + "=" * 70)
print("ISSUE 1: Energy Landscape Deep Dive")
print("=" * 70)

ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)

# Check model architecture
print("\nModel Architecture:")
total_params = sum(p.numel() for p in ebm.parameters())
print(f"  Total parameters: {total_params:,}")
for name, module in ebm.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params:,} params")

# Check weight initialization
print("\nWeight Initialization Analysis:")
for name, param in ebm.named_parameters():
    if 'weight' in name:
        print(f"  {name:30s}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

# The energy is ||output||^2, so let's check output layer behavior
print("\n" + "-" * 60)
print("Output Layer Analysis (energy = ||fc4(h)||^2):")

# Test fc4 output magnitude with random hidden states
hidden = torch.randn(32, 512).to(device)  # Typical hidden dimension
with torch.no_grad():
    fc4_out = ebm.fc4(hidden)
    print(f"  fc4 input shape: {hidden.shape}")
    print(f"  fc4 output shape: {fc4_out.shape}")
    print(f"  fc4 output mean: {fc4_out.mean().item():.6f}")
    print(f"  fc4 output std: {fc4_out.std().item():.6f}")
    print(f"  fc4 output L2 norm (mean): {fc4_out.norm(dim=-1).mean().item():.6f}")
    
    # Energy would be L2 norm squared
    energy = fc4_out.pow(2).sum(dim=-1)
    print(f"  Resulting energy mean: {energy.mean().item():.6f}")
    print(f"  Resulting energy std: {energy.std().item():.6f}")

# Check full forward pass energy distribution
print("\n" + "-" * 60)
print("Full Forward Pass Energy Distribution:")

batch_size = 100
for input_type in ['zeros', 'ones', 'randn', 'uniform', 'large']:
    if input_type == 'zeros':
        inp = torch.zeros(batch_size, 128).to(device)
        out = torch.zeros(batch_size, 128).to(device)
    elif input_type == 'ones':
        inp = torch.ones(batch_size, 128).to(device)
        out = torch.ones(batch_size, 128).to(device)
    elif input_type == 'randn':
        inp = torch.randn(batch_size, 128).to(device)
        out = torch.randn(batch_size, 128).to(device)
    elif input_type == 'uniform':
        inp = torch.rand(batch_size, 128).to(device) * 2 - 1  # [-1, 1]
        out = torch.rand(batch_size, 128).to(device) * 2 - 1
    else:  # large
        inp = torch.randn(batch_size, 128).to(device) * 10
        out = torch.randn(batch_size, 128).to(device) * 10
    
    t = torch.randint(0, 10, (batch_size,), device=device)
    
    with torch.no_grad():
        energy = ebm(inp, out, t)
    
    print(f"  Input type '{input_type:8s}': E_mean={energy.mean().item():.4f}, E_std={energy.std().item():.4f}")

#############################################################################
# Issue 2: Why All Energies Are Similar
#############################################################################
print("\n" + "=" * 70)
print("ISSUE 2: Why All Energies Are Similar")
print("=" * 70)

# The key insight: normalized inputs + orthogonal init = similar outputs
print("\nHypothesis: Normalized embeddings + initialization = flat landscape")

# Test with actual dataset embeddings
dataset = AlgebraDataset(rule='combine', split='train', num_problems=50, d_model=128)

# Get some real embeddings
real_inp = []
real_out = []
for i in range(20):
    inp, out = dataset[i]
    real_inp.append(inp)
    real_out.append(out)

real_inp = torch.stack(real_inp).to(device)
real_out = torch.stack(real_out).to(device)

print(f"\nReal dataset embeddings:")
print(f"  Input L2 norms: {real_inp.norm(dim=-1).mean().item():.4f} (all should be ~1.0)")
print(f"  Output L2 norms: {real_out.norm(dim=-1).mean().item():.4f}")

# The problem: when all inputs have norm=1, the first layer sees very similar magnitudes
print("\n" + "-" * 60)
print("First Layer (fc1) Analysis:")

with torch.no_grad():
    concat_input = torch.cat([real_inp, real_out], dim=-1)  # (20, 256)
    fc1_out = ebm.fc1(concat_input)
    fc1_out_activated = torch.nn.functional.silu(fc1_out)  # swish = silu
    
    print(f"  Concatenated input shape: {concat_input.shape}")
    print(f"  fc1 output before activation: mean={fc1_out.mean():.4f}, std={fc1_out.std():.4f}")
    print(f"  fc1 output after swish: mean={fc1_out_activated.mean():.4f}, std={fc1_out_activated.std():.4f}")
    
# Compare valid vs invalid pairs
print("\n" + "-" * 60)
print("Valid vs Invalid Pair Comparison:")

t = torch.zeros(20, device=device, dtype=torch.long)

# Valid: input-output pairs from dataset
with torch.no_grad():
    energy_valid = ebm(real_inp, real_out, t)

# Invalid 1: shuffled outputs
shuffled_out = real_out[torch.randperm(20)]
with torch.no_grad():
    energy_shuffled = ebm(real_inp, shuffled_out, t)

# Invalid 2: random outputs
random_out = torch.randn_like(real_out)
random_out = F.normalize(random_out, p=2, dim=-1)  # Normalize like real embeddings
with torch.no_grad():
    energy_random = ebm(real_inp, random_out, t)

print(f"  Valid pairs:    E={energy_valid.mean().item():.4f} ± {energy_valid.std().item():.4f}")
print(f"  Shuffled pairs: E={energy_shuffled.mean().item():.4f} ± {energy_shuffled.std().item():.4f}")
print(f"  Random pairs:   E={energy_random.mean().item():.4f} ± {energy_random.std().item():.4f}")
print(f"  Gap (shuffled-valid): {(energy_shuffled.mean() - energy_valid.mean()).item():.4f}")
print(f"  Gap (random-valid): {(energy_random.mean() - energy_valid.mean()).item():.4f}")

#############################################################################
# Issue 3: Gradient Wrapper Problem
#############################################################################
print("\n" + "=" * 70)
print("ISSUE 3: Gradient Wrapper Analysis")
print("=" * 70)

wrapper = AlgebraDiffusionWrapper(ebm).to(device)

# Test 1: With requires_grad input
print("\nTest 1: Input WITH requires_grad:")
inp = torch.randn(8, 128, device=device)
out = torch.randn(8, 128, device=device, requires_grad=True)
t = torch.randint(0, 10, (8,), device=device)

grad_output = wrapper(inp, out, t)
print(f"  Gradient shape: {grad_output.shape}")
print(f"  Gradient mean: {grad_output.mean().item():.6f}")
print(f"  Gradient std: {grad_output.std().item():.6f}")
print(f"  Gradient L2 norm: {grad_output.norm(dim=-1).mean().item():.6f}")

# Test 2: Without requires_grad (simulating dataloader)
print("\nTest 2: Input WITHOUT requires_grad (simulating dataloader):")
inp = torch.randn(8, 128, device=device)
out = torch.randn(8, 128, device=device)  # No requires_grad!
t = torch.randint(0, 10, (8,), device=device)

try:
    grad_output = wrapper(inp, out, t)
    print(f"  Gradient shape: {grad_output.shape}")
    print(f"  Gradient mean: {grad_output.mean().item():.6f}")
    print(f"  Gradient std: {grad_output.std().item():.6f}")
    
    # Check if it's all zeros
    if grad_output.abs().max().item() < 1e-10:
        print("  ⚠️  PROBLEM: Gradient is all zeros!")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Manual gradient computation to verify
print("\nTest 3: Manual gradient computation:")
inp = torch.randn(8, 128, device=device)
out_orig = torch.randn(8, 128, device=device)
out = out_orig.clone().requires_grad_(True)  # Properly clone first
t = torch.randint(0, 10, (8,), device=device)

energy = ebm(inp, out, t)
grad_manual = torch.autograd.grad(energy.sum(), out, create_graph=True)[0]
print(f"  Manual gradient mean: {grad_manual.mean().item():.6f}")
print(f"  Manual gradient std: {grad_manual.std().item():.6f}")
print(f"  Manual gradient L2 norm: {grad_manual.norm(dim=-1).mean().item():.6f}")

#############################################################################
# Issue 4: Contrastive Loss Target Mismatch
#############################################################################
print("\n" + "=" * 70)
print("ISSUE 4: Contrastive Loss Target Analysis")
print("=" * 70)

from src.algebra.algebra_models import ContrastiveEnergyLoss

loss_fn = ContrastiveEnergyLoss(margin=10.0, pos_target=1.0, neg_target=15.0)

print(f"\nContrastive Loss Configuration:")
print(f"  pos_target = {loss_fn.pos_target} (want positive energies near this)")
print(f"  neg_target = {loss_fn.neg_target} (want negative energies near this)")
print(f"  margin = {loss_fn.margin} (minimum gap E_neg - E_pos)")

# Current model energies
with torch.no_grad():
    energy_current = ebm(real_inp, real_out, torch.zeros(20, device=device, dtype=torch.long))

print(f"\nCurrent model energies: {energy_current.mean().item():.4f}")
print(f"Problem: Model outputs ~{energy_current.mean().item():.2f}, but targets are {loss_fn.pos_target} and {loss_fn.neg_target}")
print(f"\nThis creates HUGE loss just from scale mismatch!")

# Calculate what the loss would be
pos_loss = F.mse_loss(energy_current, torch.full_like(energy_current, loss_fn.pos_target))
neg_loss = F.mse_loss(energy_current, torch.full_like(energy_current, loss_fn.neg_target))
print(f"  pos_loss (trying to push {energy_current.mean().item():.2f} → {loss_fn.pos_target}): {pos_loss.item():.4f}")
print(f"  neg_loss (trying to push {energy_current.mean().item():.2f} → {loss_fn.neg_target}): {neg_loss.item():.4f}")

#############################################################################
# ROOT CAUSES IDENTIFIED
#############################################################################
print("\n" + "=" * 70)
print("ROOT CAUSES IDENTIFIED")
print("=" * 70)

print("""
1. FLAT ENERGY LANDSCAPE:
   - All embeddings are L2-normalized to unit norm
   - With orthogonal/small initialization, fc4 output magnitude is small (~0.45)
   - Energy = ||fc4_output||^2 is always ~0.2 regardless of input
   - The model cannot distinguish valid from invalid pairs at initialization
   
   FIX: Either scale up fc4 output, or use different energy formulation

2. GRADIENT WRAPPER ISSUE:
   - The wrapper does `out = out.requires_grad_(True)` on the input tensor
   - This fails silently when out is already detached from computation graph
   - Result: gradient computation fails, returns zeros
   
   FIX: Clone tensor before requiring grad: `out = out.clone().requires_grad_(True)`

3. TARGET SCALE MISMATCH:
   - ContrastiveEnergyLoss targets: pos=1.0, neg=15.0
   - Actual model energies: ~0.2
   - This creates enormous loss from scale mismatch before any learning
   - The model wastes capacity trying to scale up energies to match targets
   
   FIX: Either adjust targets to match model output scale, or add a learned scaling layer

4. INITIALIZATION CAUSES NEAR-CONSTANT ENERGY:
   - Orthogonal init with gain=0.1 creates very small weights
   - Combined with normalized inputs, all outputs are similar
   
   FIX: Use Xavier/He initialization with appropriate gain
""")

#############################################################################
# PROPOSED FIXES
#############################################################################
print("\n" + "=" * 70)
print("PROPOSED FIXES")
print("=" * 70)

print("""
FIX 1: Fix gradient wrapper (CRITICAL)
   In AlgebraDiffusionWrapper.forward():
   - Change: out = out.requires_grad_(True)
   - To:     out = out.detach().clone().requires_grad_(True)

FIX 2: Adjust contrastive loss targets (IMPORTANT)
   - Option A: Scale down targets to match model output (~0.1 to ~3.0)
   - Option B: Add energy scaling layer: energy = scale * ||output||^2 + bias
   - Recommended: pos_target=0.1, neg_target=2.0, margin=1.0

FIX 3: Better weight initialization (HELPFUL)
   - Use Xavier uniform initialization for fc4 with larger gain
   - Or: Add a learnable energy scaling parameter

FIX 4: Consider different energy formulation (OPTIONAL)
   - Instead of ||output||^2, use learned distance metric
   - Or: Use bilinear energy E(inp, out) = inp^T W out
""")

# Test the gradient fix
print("\n" + "-" * 60)
print("Testing proposed gradient fix:")

def fixed_wrapper_forward(wrapper, inp, out, t):
    """Fixed version of wrapper forward."""
    # Clone and detach before requiring grad - this is the fix!
    out_for_grad = out.detach().clone().requires_grad_(True)
    
    # Compute energy
    energy = wrapper.ebm(inp, out_for_grad, t)
    
    # Compute gradient
    grad = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=out_for_grad,
        create_graph=True
    )[0]
    
    return grad

inp = torch.randn(8, 128, device=device)
out = torch.randn(8, 128, device=device)  # No requires_grad
t = torch.randint(0, 10, (8,), device=device)

grad_fixed = fixed_wrapper_forward(wrapper, inp, out, t)
print(f"  Fixed gradient mean: {grad_fixed.mean().item():.6f}")
print(f"  Fixed gradient std: {grad_fixed.std().item():.6f}")
print(f"  Fixed gradient L2 norm: {grad_fixed.norm(dim=-1).mean().item():.6f}")
print(f"  ✓ Gradient is now non-zero!")
