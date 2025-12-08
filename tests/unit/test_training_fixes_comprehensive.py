#!/usr/bin/env python
"""
Comprehensive test for training fixes in AlgebraEBM.

Tests the following fixes:
1. FiLM layer initialization - prevents FiLM from dominating input signal
2. Gradient wrapper fix - handles detached tensors properly
3. Learnable energy scaling - allows matching contrastive targets
4. Proper weight initialization - Xavier uniform with controlled gains

Run with: python tests/unit/test_training_fixes_comprehensive.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper, ContrastiveEnergyLoss
from src.algebra.algebra_dataset import AlgebraDataset

def test_film_initialization():
    """Test that FiLM layers are initialized near-identity."""
    print("=" * 60)
    print("TEST 1: FiLM Initialization")
    print("=" * 60)
    
    model = AlgebraEBM(inp_dim=128, out_dim=128)
    
    t = torch.zeros(1, dtype=torch.long)
    t_emb = model.time_mlp(t)
    
    fc2_params = model.t_map_fc2(t_emb)
    fc2_gain, fc2_bias = torch.chunk(fc2_params, 2, dim=-1)
    
    fc3_params = model.t_map_fc3(t_emb)
    fc3_gain, fc3_bias = torch.chunk(fc3_params, 2, dim=-1)
    
    # FiLM should be near-identity: gain ≈ 0, bias ≈ 0
    # So effective gain (gain + 1) ≈ 1
    gain2_ok = abs(fc2_gain.mean().item()) < 0.1 and fc2_gain.std().item() < 0.1
    gain3_ok = abs(fc3_gain.mean().item()) < 0.1 and fc3_gain.std().item() < 0.1
    bias2_ok = abs(fc2_bias.mean().item()) < 0.1 and fc2_bias.std().item() < 0.1
    bias3_ok = abs(fc3_bias.mean().item()) < 0.1 and fc3_bias.std().item() < 0.1
    
    print(f"FC2 gain: mean={fc2_gain.mean():.4f}, std={fc2_gain.std():.4f} {'✓' if gain2_ok else '✗'}")
    print(f"FC2 bias: mean={fc2_bias.mean():.4f}, std={fc2_bias.std():.4f} {'✓' if bias2_ok else '✗'}")
    print(f"FC3 gain: mean={fc3_gain.mean():.4f}, std={fc3_gain.std():.4f} {'✓' if gain3_ok else '✗'}")
    print(f"FC3 bias: mean={fc3_bias.mean():.4f}, std={fc3_bias.std():.4f} {'✓' if bias3_ok else '✗'}")
    
    passed = gain2_ok and gain3_ok and bias2_ok and bias3_ok
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gradient_flow():
    """Test that gradients flow through the diffusion wrapper."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow through Wrapper")
    print("=" * 60)
    
    model = AlgebraEBM(inp_dim=128, out_dim=128)
    wrapper = AlgebraDiffusionWrapper(model)
    
    # Random inputs
    inp = F.normalize(torch.randn(10, 128), dim=-1)
    out = F.normalize(torch.randn(10, 128), dim=-1)
    t = torch.randint(0, 100, (10,))
    
    # Forward pass
    energy = wrapper(inp, out, t, return_energy=True)
    
    # Backward pass
    loss = energy.mean()
    loss.backward()
    
    # Check gradients exist
    has_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads.append(name)
    
    # Should have gradients on key layers
    key_layers = ['fc1', 'fc2', 'fc3', 'fc4', 'energy_scale', 'energy_bias']
    found = sum(1 for k in key_layers if any(k in name for name in has_grads))
    
    print(f"Parameters with gradients: {len(has_grads)}/{len(list(model.parameters()))}")
    print(f"Key layers with gradients: {found}/{len(key_layers)}")
    
    passed = found >= len(key_layers) - 1  # Allow one missing
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_energy_discrimination():
    """Test that untrained model doesn't collapse to constant output."""
    print("\n" + "=" * 60)
    print("TEST 3: Initial Energy Discrimination")
    print("=" * 60)
    
    model = AlgebraEBM(inp_dim=128, out_dim=128)
    dataset = AlgebraDataset(rule='combine', split='train', num_problems=200, d_model=128)
    
    batch_size = 100
    inp_list, out_list = [], []
    for i in range(batch_size):
        inp_i, out_i = dataset[i]
        inp_list.append(inp_i)
        out_list.append(out_i)
    inp = torch.stack(inp_list)
    out = torch.stack(out_list)
    
    neg_out = out[torch.randperm(batch_size)]
    random_out = F.normalize(torch.randn_like(out), dim=-1)
    
    t = torch.zeros(batch_size, dtype=torch.long)
    
    with torch.no_grad():
        e_valid = model(inp, out, t)
        e_neg = model(inp, neg_out, t)
        e_rand = model(inp, random_out, t)
    
    # Check that there's variance in energy (not constant)
    var_ok = e_valid.std() > 0.001 or e_neg.std() > 0.001 or e_rand.std() > 0.001
    
    print(f"Valid energy: mean={e_valid.mean():.4f}, std={e_valid.std():.4f}")
    print(f"Shuffled energy: mean={e_neg.mean():.4f}, std={e_neg.std():.4f}")
    print(f"Random energy: mean={e_rand.mean():.4f}, std={e_rand.std():.4f}")
    print(f"Energy has variance: {var_ok}")
    
    # At init, we don't expect discrimination, but at least not constant
    print(f"\nResult: {'PASS' if var_ok else 'FAIL'}")
    return var_ok


def test_training_convergence():
    """Test that model learns to discriminate valid from invalid pairs."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Convergence (1000 steps)")
    print("=" * 60)
    
    model = AlgebraEBM(inp_dim=128, out_dim=128)
    wrapper = AlgebraDiffusionWrapper(model)
    dataset = AlgebraDataset(rule='combine', split='train', num_problems=2000, d_model=128)
    loss_fn = ContrastiveEnergyLoss(margin=5.0, pos_target=1.0, neg_target=10.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    batch_size = 128
    
    losses = []
    for step in range(1000):
        indices = torch.randint(0, len(dataset), (batch_size,))
        inp_list, out_list = [], []
        for idx in indices:
            inp, out = dataset[idx.item()]
            inp_list.append(inp)
            out_list.append(out)
        inp = torch.stack(inp_list)
        out = torch.stack(out_list)
        
        neg_out = out[torch.randperm(batch_size)]
        
        t = torch.randint(0, 100, (batch_size,))
        pos_energy = wrapper(inp, out, t, return_energy=True)
        neg_energy = wrapper(inp, neg_out, t, return_energy=True)
        
        loss = loss_fn.compute_loss(pos_energy, neg_energy)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 200 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, gap={neg_energy.mean().item() - pos_energy.mean().item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_inp, test_out = [], []
        for i in range(100):
            inp_i, out_i = dataset[i]
            test_inp.append(inp_i)
            test_out.append(out_i)
        test_inp = torch.stack(test_inp)
        test_out = torch.stack(test_out)
        
        neg_out = test_out[torch.randperm(100)]
        t = torch.zeros(100, dtype=torch.long)
        
        pos_e = wrapper(test_inp, test_out, t, return_energy=True)
        neg_e = wrapper(test_inp, neg_out, t, return_energy=True)
    
    gap = neg_e.mean() - pos_e.mean()
    print(f"\nFinal results:")
    print(f"  Valid energy: {pos_e.mean():.4f}")
    print(f"  Shuffled energy: {neg_e.mean():.4f}")
    print(f"  Energy gap: {gap:.4f}")
    
    # Should have gap > 3 after 1000 steps
    passed = gap > 3.0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (gap > 3.0 required)")
    return passed


def main():
    print("Running comprehensive training fix tests...")
    print("=" * 60)
    
    results = []
    
    results.append(("FiLM Initialization", test_film_initialization()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Energy Discrimination", test_energy_discrimination()))
    results.append(("Training Convergence", test_training_convergence()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ All training fixes verified successfully!")
    else:
        print("\n✗ Some tests failed - review the output above")


if __name__ == "__main__":
    main()
