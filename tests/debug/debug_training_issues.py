#!/usr/bin/env python3
"""
Comprehensive Training Diagnostics for Algebra EBM

This script systematically identifies common training issues:
1. Energy landscape flatness (energies too similar for pos/neg)
2. Model output collapse (limited diversity in outputs)
3. Gradient flow problems (vanishing/exploding gradients)
4. Loss component imbalance (MSE dominating energy loss)
5. Encoder/decoder issues (embedding space problems)

Run: python debug_training_issues.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import sys
import os

# Prevent threading issues
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("=" * 70)
print("ALGEBRA EBM TRAINING DIAGNOSTICS")
print("=" * 70)

# Import project modules
try:
    from algebra_dataset import AlgebraDataset
    from algebra_models import AlgebraEBM, AlgebraDiffusionWrapper, ContrastiveEnergyLoss
    from algebra_encoder import create_character_encoder
    from dataset import NoisyWrapper
    from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#############################################################################
# TEST 1: Energy Landscape Analysis
#############################################################################
def test_energy_landscape():
    """Test if energy function produces distinct values for pos/neg samples."""
    print("\n" + "=" * 70)
    print("TEST 1: Energy Landscape Analysis")
    print("=" * 70)
    
    # Create model
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    
    # Generate test data
    batch_size = 64
    inp = torch.randn(batch_size, 128).to(device)
    
    # Positive: small perturbation from input (should be valid transformation)
    out_pos = inp + torch.randn_like(inp) * 0.1
    
    # Negative: large perturbation (should be invalid)
    out_neg_light = inp + torch.randn_like(inp) * 1.0
    out_neg_heavy = inp + torch.randn_like(inp) * 3.0
    out_neg_random = torch.randn_like(inp)  # Completely random
    
    # Test at different timesteps
    timesteps = [0, 4, 9]
    
    print("\nEnergy values at different noise levels:")
    print("-" * 60)
    
    issues_found = []
    
    for t_val in timesteps:
        t = torch.full((batch_size,), t_val, device=device)
        
        with torch.no_grad():
            e_pos = ebm(inp, out_pos, t)
            e_neg_light = ebm(inp, out_neg_light, t)
            e_neg_heavy = ebm(inp, out_neg_heavy, t)
            e_neg_random = ebm(inp, out_neg_random, t)
        
        print(f"\nTimestep t={t_val}:")
        print(f"  E_pos (small perturbation):  {e_pos.mean().item():.4f} ± {e_pos.std().item():.4f}")
        print(f"  E_neg (light noise x1.0):    {e_neg_light.mean().item():.4f} ± {e_neg_light.std().item():.4f}")
        print(f"  E_neg (heavy noise x3.0):    {e_neg_heavy.mean().item():.4f} ± {e_neg_heavy.std().item():.4f}")
        print(f"  E_neg (random):              {e_neg_random.mean().item():.4f} ± {e_neg_random.std().item():.4f}")
        
        # Check for issues
        gap = e_neg_random.mean() - e_pos.mean()
        ratio = e_neg_random.mean() / (e_pos.mean() + 1e-8)
        
        print(f"  Gap (random - pos): {gap.item():.4f}, Ratio: {ratio.item():.4f}")
        
        if abs(gap.item()) < 1.0:
            issues_found.append(f"t={t_val}: Energy gap too small ({gap.item():.4f})")
        if ratio.item() < 1.5:
            issues_found.append(f"t={t_val}: Energy ratio too small ({ratio.item():.4f})")
    
    # Check energy variance
    print("\n" + "-" * 60)
    print("Energy variance analysis (untrained model):")
    
    all_energies = []
    for _ in range(10):
        inp = torch.randn(batch_size, 128).to(device)
        out = torch.randn(batch_size, 128).to(device)
        t = torch.randint(0, 10, (batch_size,), device=device)
        with torch.no_grad():
            e = ebm(inp, out, t)
            all_energies.append(e)
    
    all_e = torch.cat(all_energies)
    print(f"  Overall energy range: [{all_e.min().item():.4f}, {all_e.max().item():.4f}]")
    print(f"  Overall energy mean: {all_e.mean().item():.4f}")
    print(f"  Overall energy std: {all_e.std().item():.4f}")
    
    if all_e.std().item() < 0.1:
        issues_found.append(f"Energy variance too low: std={all_e.std().item():.4f}")
    
    if issues_found:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✓ Energy landscape appears reasonable for untrained model")
    
    return len(issues_found) == 0

#############################################################################
# TEST 2: Output Diversity / Model Collapse
#############################################################################
def test_output_diversity():
    """Check if model produces diverse outputs or collapses to few patterns."""
    print("\n" + "=" * 70)
    print("TEST 2: Output Diversity Analysis")
    print("=" * 70)
    
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    wrapper = AlgebraDiffusionWrapper(ebm).to(device)
    
    # Different input scenarios
    batch_size = 32
    
    # Scenario 1: Same input, different outputs expected
    print("\nScenario 1: Same input, different random targets")
    inp_fixed = torch.randn(1, 128).to(device).expand(batch_size, -1)
    out_varied = torch.randn(batch_size, 128).to(device)
    t = torch.full((batch_size,), 5, device=device)
    
    with torch.no_grad():
        pred = wrapper(inp_fixed, out_varied, t)
        energies = ebm(inp_fixed, out_varied, t)
    
    # Check output diversity
    pred_mean = pred.mean(dim=0)
    pred_centered = pred - pred_mean
    pred_var = (pred_centered ** 2).mean()
    
    # Check if all predictions are too similar
    pairwise_dists = torch.cdist(pred, pred)
    avg_dist = pairwise_dists[~torch.eye(batch_size, dtype=bool, device=device)].mean()
    
    print(f"  Prediction variance: {pred_var.item():.6f}")
    print(f"  Average pairwise distance: {avg_dist.item():.6f}")
    print(f"  Energy variance: {energies.std().item():.6f}")
    
    issues = []
    if pred_var.item() < 0.01:
        issues.append(f"Prediction variance too low: {pred_var.item():.6f}")
    if avg_dist.item() < 0.1:
        issues.append(f"Outputs too similar: avg_dist={avg_dist.item():.6f}")
    
    # Scenario 2: Check unique outputs
    print("\nScenario 2: Unique output analysis")
    unique_outputs = torch.unique(pred.round(decimals=2), dim=0)
    uniqueness_ratio = len(unique_outputs) / batch_size
    print(f"  Unique outputs: {len(unique_outputs)}/{batch_size} ({uniqueness_ratio*100:.1f}%)")
    
    if uniqueness_ratio < 0.5:
        issues.append(f"Output collapse detected: only {uniqueness_ratio*100:.1f}% unique")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Output diversity appears reasonable")
    
    return len(issues) == 0

#############################################################################
# TEST 3: Gradient Flow Analysis
#############################################################################
def test_gradient_flow():
    """Analyze gradient magnitudes through the model."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Flow Analysis")
    print("=" * 70)
    
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    
    batch_size = 32
    inp = torch.randn(batch_size, 128, device=device, requires_grad=True)
    out = torch.randn(batch_size, 128, device=device, requires_grad=True)
    t = torch.randint(0, 10, (batch_size,), device=device)
    
    # Forward pass
    energy = ebm(inp, out, t)
    loss = energy.mean()
    
    # Backward pass
    loss.backward()
    
    print("\nGradient magnitudes by layer:")
    print("-" * 60)
    
    issues = []
    gradient_stats = {}
    
    for name, param in ebm.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            gradient_stats[name] = grad_norm
            
            status = "✓"
            if grad_norm < 1e-7:
                status = "⚠️ VANISHING"
                issues.append(f"{name}: vanishing gradient ({grad_norm:.2e})")
            elif grad_norm > 100:
                status = "⚠️ EXPLODING"
                issues.append(f"{name}: exploding gradient ({grad_norm:.2e})")
            
            print(f"  {name:30s}: norm={grad_norm:10.6f}, mean={grad_mean:10.6f}, std={grad_std:10.6f} {status}")
    
    # Check input gradients
    if inp.grad is not None:
        inp_grad_norm = inp.grad.norm().item()
        print(f"\n  Input gradient norm: {inp_grad_norm:.6f}")
        if inp_grad_norm < 1e-7:
            issues.append(f"Input gradient vanishing: {inp_grad_norm:.2e}")
    
    if out.grad is not None:
        out_grad_norm = out.grad.norm().item()
        print(f"  Output gradient norm: {out_grad_norm:.6f}")
        if out_grad_norm < 1e-7:
            issues.append(f"Output gradient vanishing: {out_grad_norm:.2e}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Gradient flow appears healthy")
    
    return len(issues) == 0

#############################################################################
# TEST 4: Contrastive Loss Behavior
#############################################################################
def test_contrastive_loss():
    """Test ContrastiveEnergyLoss behavior and sensitivity."""
    print("\n" + "=" * 70)
    print("TEST 4: Contrastive Loss Analysis")
    print("=" * 70)
    
    loss_fn = ContrastiveEnergyLoss(margin=10.0, pos_target=1.0, neg_target=15.0)
    
    print("\nContrastive loss configuration:")
    print(f"  pos_target: {loss_fn.pos_target}")
    print(f"  neg_target: {loss_fn.neg_target}")
    print(f"  margin: {loss_fn.margin}")
    
    # Test scenarios
    scenarios = [
        ("Ideal separation", torch.tensor([[1.0]]), torch.tensor([[15.0]])),
        ("Both low (flat landscape)", torch.tensor([[0.5]]), torch.tensor([[0.8]])),
        ("Both high", torch.tensor([[10.0]]), torch.tensor([[12.0]])),
        ("Reversed (neg < pos)", torch.tensor([[10.0]]), torch.tensor([[5.0]])),
        ("Good gap but wrong scale", torch.tensor([[100.0]]), torch.tensor([[150.0]])),
    ]
    
    print("\nLoss values for different energy configurations:")
    print("-" * 70)
    
    issues = []
    
    for name, pos_e, neg_e in scenarios:
        pos_e = pos_e.float()
        neg_e = neg_e.float()
        loss, metrics = loss_fn.compute_loss(pos_e, neg_e, return_metrics=True)
        
        print(f"\n{name}:")
        print(f"  E_pos={pos_e.item():.1f}, E_neg={neg_e.item():.1f}")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Components: pos_loss={metrics['pos_loss']:.4f}, neg_loss={metrics['neg_loss']:.4f}, margin_loss={metrics['margin_loss']:.4f}")
        print(f"  Gap: {metrics['energy_gap']:.4f}, Ratio: {metrics['energy_ratio']:.4f}")
        
        # Check for problematic cases
        if name == "Both low (flat landscape)" and loss.item() < 1.0:
            issues.append("Loss too low for flat landscape case")
    
    # Test gradient flow through contrastive loss
    print("\n" + "-" * 70)
    print("Gradient flow through ContrastiveEnergyLoss:")
    
    pos_e = torch.tensor([[5.0]], requires_grad=True)
    neg_e = torch.tensor([[8.0]], requires_grad=True)
    
    loss, _ = loss_fn.compute_loss(pos_e, neg_e, return_metrics=True)
    loss.backward()
    
    print(f"  d(loss)/d(E_pos) = {pos_e.grad.item():.6f}")
    print(f"  d(loss)/d(E_neg) = {neg_e.grad.item():.6f}")
    
    if abs(pos_e.grad.item()) < 1e-6:
        issues.append("Zero gradient for positive energy")
    if abs(neg_e.grad.item()) < 1e-6:
        issues.append("Zero gradient for negative energy")
    
    # Expected: pos gradient should be positive (push pos_e down)
    # neg gradient should be negative (push neg_e up)
    if pos_e.grad.item() < 0:
        issues.append(f"Wrong sign for pos gradient: {pos_e.grad.item()}")
    if neg_e.grad.item() > 0:
        issues.append(f"Wrong sign for neg gradient: {neg_e.grad.item()}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Contrastive loss behavior is correct")
    
    return len(issues) == 0

#############################################################################
# TEST 5: Dataset and Encoding Analysis
#############################################################################
def test_dataset_encoding():
    """Analyze dataset and encoding properties."""
    print("\n" + "=" * 70)
    print("TEST 5: Dataset and Encoding Analysis")
    print("=" * 70)
    
    issues = []
    
    # Create small dataset
    print("\nCreating test dataset...")
    dataset = AlgebraDataset(rule='combine', split='train', num_problems=100, d_model=128)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  inp_dim: {dataset.inp_dim}")
    print(f"  out_dim: {dataset.out_dim}")
    
    # Analyze embedding properties
    print("\nEmbedding statistics:")
    print("-" * 60)
    
    inp_embeddings = []
    out_embeddings = []
    
    for i in range(min(50, len(dataset))):
        result = dataset[i]
        if len(result) == 3:
            inp, out, mask = result
        else:
            inp, out = result
        inp_embeddings.append(inp)
        out_embeddings.append(out)
    
    inp_tensor = torch.stack(inp_embeddings)
    out_tensor = torch.stack(out_embeddings)
    
    print(f"\nInput embeddings:")
    print(f"  Shape: {inp_tensor.shape}")
    print(f"  Mean: {inp_tensor.mean().item():.6f}")
    print(f"  Std: {inp_tensor.std().item():.6f}")
    print(f"  Range: [{inp_tensor.min().item():.4f}, {inp_tensor.max().item():.4f}]")
    print(f"  L2 norms: mean={inp_tensor.norm(dim=-1).mean().item():.4f}, std={inp_tensor.norm(dim=-1).std().item():.4f}")
    
    print(f"\nOutput embeddings:")
    print(f"  Shape: {out_tensor.shape}")
    print(f"  Mean: {out_tensor.mean().item():.6f}")
    print(f"  Std: {out_tensor.std().item():.6f}")
    print(f"  Range: [{out_tensor.min().item():.4f}, {out_tensor.max().item():.4f}]")
    print(f"  L2 norms: mean={out_tensor.norm(dim=-1).mean().item():.4f}, std={out_tensor.norm(dim=-1).std().item():.4f}")
    
    # Check for issues
    if inp_tensor.std().item() < 0.01:
        issues.append(f"Input embeddings have low variance: {inp_tensor.std().item():.6f}")
    if out_tensor.std().item() < 0.01:
        issues.append(f"Output embeddings have low variance: {out_tensor.std().item():.6f}")
    
    # Check embedding similarity
    inp_out_sim = F.cosine_similarity(inp_tensor, out_tensor, dim=-1)
    print(f"\nInput-Output similarity (cosine):")
    print(f"  Mean: {inp_out_sim.mean().item():.4f}")
    print(f"  Std: {inp_out_sim.std().item():.4f}")
    
    if inp_out_sim.mean().item() > 0.95:
        issues.append(f"Input/Output too similar: cosine={inp_out_sim.mean().item():.4f}")
    
    # Check unique embeddings
    unique_inp = torch.unique(inp_tensor.round(decimals=3), dim=0)
    unique_out = torch.unique(out_tensor.round(decimals=3), dim=0)
    print(f"\nUnique embeddings (rounded to 3 decimals):")
    print(f"  Inputs: {len(unique_inp)}/{len(inp_tensor)} ({100*len(unique_inp)/len(inp_tensor):.1f}%)")
    print(f"  Outputs: {len(unique_out)}/{len(out_tensor)} ({100*len(unique_out)/len(out_tensor):.1f}%)")
    
    if len(unique_inp) < len(inp_tensor) * 0.9:
        issues.append(f"Many duplicate input embeddings: {len(unique_inp)}/{len(inp_tensor)}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Dataset and encoding properties look reasonable")
    
    return len(issues) == 0

#############################################################################
# TEST 6: Mini Training Loop Analysis  
#############################################################################
def test_mini_training():
    """Run a few training steps and analyze behavior."""
    print("\n" + "=" * 70)
    print("TEST 6: Mini Training Loop Analysis")
    print("=" * 70)
    
    issues = []
    
    # Setup
    dataset = AlgebraDataset(rule='combine', split='train', num_problems=500, d_model=128)
    noisy_dataset = NoisyWrapper(dataset, timesteps=10)
    
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    wrapper = AlgebraDiffusionWrapper(ebm).to(device)
    
    diffusion = GaussianDiffusion1D(
        wrapper,
        seq_length=128,
        objective='pred_noise',
        timesteps=10,
        sampling_timesteps=10,
        supervise_energy_landscape=True,
        use_contrastive_energy_loss=True,
        use_innerloop_opt=True,
        step_size_multiplier=0.1,
        continuous=True
    ).to(device)
    
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    
    # Collect metrics over mini training
    print("\nRunning 50 training steps...")
    print("-" * 60)
    
    losses = []
    mse_losses = []
    energy_losses = []
    energy_gaps = []
    grad_norms = []
    
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True)
    
    for step, batch in enumerate(dataloader):
        if step >= 50:
            break
        
        # Handle both 2-tuple and 3-tuple returns
        if len(batch) == 3:
            inp, out, mask = batch
        else:
            inp, out = batch
            mask = None
        
        inp = inp.to(device)
        out = out.to(device)
        mask = mask.to(device) if mask is not None else None
        
        optimizer.zero_grad()
        
        loss, (loss_mse, loss_energy, loss_opt) = diffusion(inp, out, mask)
        loss.backward()
        
        # Track gradient norm
        total_norm = 0
        for p in diffusion.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        optimizer.step()
        
        losses.append(loss.item())
        mse_losses.append(loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse)
        if isinstance(loss_energy, torch.Tensor):
            energy_losses.append(loss_energy.item())
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, mse={loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse:.4f}, energy={loss_energy.item() if isinstance(loss_energy, torch.Tensor) else loss_energy:.4f}, grad_norm={total_norm:.4f}")
    
    print("\n" + "-" * 60)
    print("Training statistics over 50 steps:")
    
    losses = np.array(losses)
    mse_losses = np.array(mse_losses)
    energy_losses = np.array(energy_losses) if energy_losses else np.array([0])
    grad_norms = np.array(grad_norms)
    
    print(f"  Total loss: {losses[0]:.4f} -> {losses[-1]:.4f} (change: {losses[-1] - losses[0]:.4f})")
    print(f"  MSE loss: {mse_losses[0]:.4f} -> {mse_losses[-1]:.4f}")
    print(f"  Energy loss: {energy_losses[0]:.4f} -> {energy_losses[-1]:.4f}")
    print(f"  Gradient norm: mean={grad_norms.mean():.4f}, max={grad_norms.max():.4f}")
    
    # Check for issues
    if losses[-1] > losses[0] * 1.5:
        issues.append(f"Loss increasing: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    if np.std(losses) < 1e-6:
        issues.append(f"Loss not changing (stuck): std={np.std(losses):.6f}")
    
    if grad_norms.max() > 100:
        issues.append(f"Gradient explosion: max_norm={grad_norms.max():.4f}")
    
    if grad_norms.mean() < 1e-6:
        issues.append(f"Vanishing gradients: mean_norm={grad_norms.mean():.6f}")
    
    # Check loss balance
    mse_to_energy_ratio = mse_losses.mean() / (energy_losses.mean() + 1e-8)
    print(f"\n  MSE/Energy ratio: {mse_to_energy_ratio:.2f}")
    
    if mse_to_energy_ratio > 100:
        issues.append(f"MSE dominates energy loss by {mse_to_energy_ratio:.0f}x")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Mini training loop looks reasonable")
    
    return len(issues) == 0

#############################################################################
# TEST 7: FiLM Conditioning Analysis
#############################################################################
def test_film_conditioning():
    """Test if FiLM conditioning is working properly."""
    print("\n" + "=" * 70)
    print("TEST 7: FiLM (Time) Conditioning Analysis")
    print("=" * 70)
    
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    
    batch_size = 32
    inp = torch.randn(batch_size, 128).to(device)
    out = torch.randn(batch_size, 128).to(device)
    
    # Test energy at different timesteps for same input
    print("\nEnergy variation across timesteps (same input):")
    print("-" * 60)
    
    energies_by_t = {}
    for t_val in range(10):
        t = torch.full((batch_size,), t_val, device=device)
        with torch.no_grad():
            e = ebm(inp, out, t)
        energies_by_t[t_val] = e.mean().item()
        print(f"  t={t_val}: E={e.mean().item():.4f} ± {e.std().item():.4f}")
    
    # Check if energies vary with timestep
    e_values = list(energies_by_t.values())
    e_range = max(e_values) - min(e_values)
    e_std = np.std(e_values)
    
    print(f"\n  Energy range across timesteps: {e_range:.4f}")
    print(f"  Energy std across timesteps: {e_std:.4f}")
    
    issues = []
    if e_range < 0.1:
        issues.append(f"FiLM conditioning may be weak: energy range={e_range:.4f}")
    
    # Check FiLM parameters
    print("\nFiLM parameter statistics:")
    with torch.no_grad():
        t = torch.arange(10, device=device).float()
        t_emb = ebm.time_mlp(t)
        
        fc2_params = ebm.t_map_fc2(t_emb)
        fc2_gain, fc2_bias = torch.chunk(fc2_params, 2, dim=-1)
        
        fc3_params = ebm.t_map_fc3(t_emb)
        fc3_gain, fc3_bias = torch.chunk(fc3_params, 2, dim=-1)
    
    print(f"  FC2 gain range: [{fc2_gain.min().item():.4f}, {fc2_gain.max().item():.4f}]")
    print(f"  FC2 bias range: [{fc2_bias.min().item():.4f}, {fc2_bias.max().item():.4f}]")
    print(f"  FC3 gain range: [{fc3_gain.min().item():.4f}, {fc3_gain.max().item():.4f}]")
    print(f"  FC3 bias range: [{fc3_bias.min().item():.4f}, {fc3_bias.max().item():.4f}]")
    
    # Gain should be around 0 (since we add 1 to it)
    if fc2_gain.abs().max().item() > 10:
        issues.append(f"FC2 gain values too large: max={fc2_gain.abs().max().item():.4f}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ FiLM conditioning appears functional")
    
    return len(issues) == 0

#############################################################################
# TEST 8: Diffusion Wrapper Analysis
#############################################################################
def test_diffusion_wrapper():
    """Test AlgebraDiffusionWrapper gradient computation."""
    print("\n" + "=" * 70)
    print("TEST 8: Diffusion Wrapper Gradient Analysis")
    print("=" * 70)
    
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
    wrapper = AlgebraDiffusionWrapper(ebm).to(device)
    
    batch_size = 32
    inp = torch.randn(batch_size, 128).to(device)
    out = torch.randn(batch_size, 128, requires_grad=True).to(device)
    t = torch.randint(0, 10, (batch_size,), device=device)
    
    # Get wrapper output (should be gradient of energy)
    grad_output = wrapper(inp, out, t)
    
    print(f"\nWrapper output (energy gradient) statistics:")
    print(f"  Shape: {grad_output.shape}")
    print(f"  Mean: {grad_output.mean().item():.6f}")
    print(f"  Std: {grad_output.std().item():.6f}")
    print(f"  Range: [{grad_output.min().item():.4f}, {grad_output.max().item():.4f}]")
    print(f"  L2 norm (mean): {grad_output.norm(dim=-1).mean().item():.4f}")
    
    issues = []
    
    if grad_output.std().item() < 1e-6:
        issues.append(f"Gradient output has no variance: std={grad_output.std().item():.6f}")
    
    if grad_output.norm(dim=-1).mean().item() < 1e-6:
        issues.append(f"Gradient magnitude too small: {grad_output.norm(dim=-1).mean().item():.6f}")
    
    # Test return_energy mode
    energy = wrapper(inp, out, t, return_energy=True)
    print(f"\nEnergy output:")
    print(f"  Shape: {energy.shape}")
    print(f"  Mean: {energy.mean().item():.4f}")
    print(f"  Std: {energy.std().item():.4f}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Diffusion wrapper appears functional")
    
    return len(issues) == 0

#############################################################################
# RUN ALL TESTS
#############################################################################
def main():
    results = {}
    
    results['energy_landscape'] = test_energy_landscape()
    results['output_diversity'] = test_output_diversity()
    results['gradient_flow'] = test_gradient_flow()
    results['contrastive_loss'] = test_contrastive_loss()
    results['dataset_encoding'] = test_dataset_encoding()
    results['film_conditioning'] = test_film_conditioning()
    results['diffusion_wrapper'] = test_diffusion_wrapper()
    results['mini_training'] = test_mini_training()
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {name:25s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        if not results['energy_landscape']:
            print("""
• ENERGY LANDSCAPE ISSUES:
  - Energies may not differentiate between valid/invalid transformations
  - Consider increasing model capacity or checking initialization
  - The energy function should produce LOW energy for valid pairs
    and HIGH energy for invalid pairs
""")
        
        if not results['output_diversity']:
            print("""
• OUTPUT COLLAPSE:
  - Model may be producing similar outputs regardless of input
  - Check for mode collapse in the energy function
  - Consider adding diversity-promoting regularization
""")
        
        if not results['gradient_flow']:
            print("""
• GRADIENT FLOW ISSUES:
  - Vanishing gradients prevent learning
  - Check layer initializations and activation functions
  - Consider gradient clipping or different optimizer
""")
        
        if not results['contrastive_loss']:
            print("""
• CONTRASTIVE LOSS ISSUES:
  - Loss may not be providing correct training signal
  - Check target energy values (pos_target, neg_target)
  - Verify margin is appropriate for energy scale
""")
        
        if not results['mini_training']:
            print("""
• TRAINING ISSUES:
  - Loss may not be decreasing or is unstable
  - MSE may be dominating energy loss
  - Check learning rate and loss balancing
""")

if __name__ == "__main__":
    main()
