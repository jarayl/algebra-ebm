#!/usr/bin/env python3
"""
Debug baseline evaluation to understand what's happening.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
import numpy as np

def load_checkpoint_fixed(checkpoint_path, d_model=128):
    """Load checkpoint with proper key extraction."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Create fresh model
    ebm = AlgebraEBM(
        inp_dim=d_model,
        out_dim=d_model,
        rule_name='distribute'
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract EBM state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    
    # Handle _orig_mod.model.ebm. prefix
    ebm_state_dict = {}
    for k, v in state_dict.items():
        if '_orig_mod.model.ebm.' in k:
            new_key = k.replace('_orig_mod.model.ebm.', '')
            ebm_state_dict[new_key] = v
        elif 'model.ebm.' in k:
            new_key = k.replace('model.ebm.', '')
            ebm_state_dict[new_key] = v
        elif k.startswith('ebm.'):
            new_key = k.replace('ebm.', '')
            ebm_state_dict[new_key] = v
    
    if not ebm_state_dict:
        print("No EBM keys found. Available keys:")
        print(list(state_dict.keys())[:10])
        return None
    
    print(f"Found {len(ebm_state_dict)} EBM parameters")
    ebm.load_state_dict(ebm_state_dict)
    ebm.eval()
    return ebm

def debug_model():
    """Debug what the model is doing."""
    print("="*60)
    print("DEBUGGING BASELINE EVALUATION")
    print("="*60)
    
    # Load model
    ebm = load_checkpoint_fixed('results/distribute/model-12.pt')
    if ebm is None:
        print("Failed to load checkpoint")
        return
    
    # Test basic model functionality
    inp = torch.randn(1, 128)
    out = torch.randn(1, 128)
    t = torch.tensor([5.0])
    
    with torch.no_grad():
        energy = ebm(inp, out, t)
        print(f"Model energy output: {energy.item():.4f}")
    
    # Test dataset
    dataset = AlgebraDataset(
        rule='distribute',
        split='test',
        num_problems=5,
        d_model=128
    )
    
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample input norm: {sample[0].norm().item():.4f}")
    print(f"Sample target norm: {sample[1].norm().item():.4f}")
    
    # Test without diffusion - just look at raw distances
    print("\nRaw distances (without diffusion):")
    for i in range(5):
        sample = dataset[i]
        inp = sample[0]
        target = sample[1]
        
        # Random noise
        noise = torch.randn(128)
        initial_dist = (noise - target).norm().item()
        
        # Target should be distance 0 from itself
        target_dist = (target - target).norm().item()
        
        # Input distance from target
        input_dist = (inp - target).norm().item()
        
        print(f"  Sample {i}: noise_dist={initial_dist:.2f}, input_dist={input_dist:.2f}, target_dist={target_dist:.4f}")
    
    # Test diffusion setup
    model_wrapper = AlgebraDiffusionWrapper(ebm)
    
    # Test with minimal diffusion
    print("\nTesting minimal inference...")
    sample = dataset[0]
    inp = sample[0].unsqueeze(0)
    target = sample[1].unsqueeze(0)
    
    # Manual gradient step
    x = torch.randn(1, 128)
    x.requires_grad_(True)
    t = torch.tensor([0.0])  # Final timestep
    
    energy = ebm(inp, x, t)
    print(f"Initial energy: {energy.item():.4f}")
    
    grad = torch.autograd.grad(energy, x)[0]
    print(f"Gradient norm: {grad.norm().item():.4f}")
    
    # Take gradient step
    with torch.no_grad():
        x_new = x - 0.01 * grad
        energy_new = ebm(inp, x_new, t)
        print(f"After gradient step: energy={energy_new.item():.4f}")
        
        # Distance to target
        dist_before = (x.detach() - target).norm().item()
        dist_after = (x_new - target).norm().item()
        print(f"Distance: {dist_before:.4f} → {dist_after:.4f} (improvement: {dist_before - dist_after:.4f})")

if __name__ == '__main__':
    debug_model()