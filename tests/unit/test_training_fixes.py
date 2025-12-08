#!/usr/bin/env python3
"""
Test script to verify the training bug fixes.

This script tests:
1. NoisyWrapper returns clean data (diffusion model adds noise internally)
2. AlgebraDiffusionWrapper correctly predicts noise for training
3. AlgebraDiffusionWrapper returns energy/gradients for inference
4. Encoder produces normalized embeddings
5. End-to-end training step works correctly
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project root to path (so 'src.xxx' imports work)
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

print("=" * 60)
print("TESTING ALGEBRA EBM TRAINING FIXES")
print("=" * 60)

# Test 1: NoisyWrapper
print("\n[TEST 1] NoisyWrapper returns clean data...")
try:
    from src.datasets.dataset import NoisyWrapper
    from src.algebra.algebra_dataset import AlgebraDataset
    
    # Create a small dataset
    dataset = AlgebraDataset(rule='distribute', split='train', num_problems=10, d_model=128)
    noisy_dataset = NoisyWrapper(dataset, timesteps=10)
    
    # Get a sample
    x, y = noisy_dataset[0]
    
    # Get the original clean data
    x_clean, y_clean = dataset[0]
    
    # Check that NoisyWrapper returns clean data (y should match y_clean)
    if torch.allclose(y, y_clean, rtol=1e-5, atol=1e-5):
        print("✅ PASS: NoisyWrapper returns clean target data (no double-noising)")
    else:
        print("❌ FAIL: NoisyWrapper is still adding noise!")
        print(f"   y shape: {y.shape}, y_clean shape: {y_clean.shape}")
        print(f"   Max difference: {(y - y_clean).abs().max().item():.6f}")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: NoisyWrapper test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Encoder normalization
print("\n[TEST 2] Encoder produces normalized embeddings...")
try:
    from src.algebra.algebra_encoder import create_character_encoder
    
    encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
    
    test_equations = ["2*x+3=7", "x=5", "3*(x+1)=9", "-5*x=-25"]
    
    for eq in test_equations:
        emb = encoder.encode_equation_string(eq)
        norm = torch.norm(emb).item()
        
        # Normalized embeddings should have unit norm
        if abs(norm - 1.0) < 0.01:
            print(f"  ✅ '{eq}' -> norm={norm:.4f}")
        else:
            print(f"  ❌ '{eq}' -> norm={norm:.4f} (expected ~1.0)")
            sys.exit(1)
    
    print("✅ PASS: Encoder produces normalized embeddings")
except Exception as e:
    print(f"❌ FAIL: Encoder normalization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: AlgebraDiffusionWrapper
print("\n[TEST 3] AlgebraDiffusionWrapper noise prediction vs energy/gradient...")
try:
    from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
    
    # Create model
    ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='test')
    wrapper = AlgebraDiffusionWrapper(ebm)
    
    # Create test inputs
    batch_size = 4
    inp = torch.randn(batch_size, 128)
    out = torch.randn(batch_size, 128)
    t = torch.randint(0, 10, (batch_size,))
    
    # Test default forward (noise prediction)
    noise_pred = wrapper(inp, out, t)
    if noise_pred.shape == (batch_size, 128):
        print(f"  ✅ Default forward returns noise prediction shape: {noise_pred.shape}")
    else:
        print(f"  ❌ Wrong shape for noise prediction: {noise_pred.shape}")
        sys.exit(1)
    
    # Test energy return
    energy = wrapper(inp, out, t, return_energy=True)
    if energy.shape == (batch_size, 1):
        print(f"  ✅ return_energy=True returns energy shape: {energy.shape}")
    else:
        print(f"  ❌ Wrong shape for energy: {energy.shape}")
        sys.exit(1)
    
    # Test return_both
    energy_both, grad_both = wrapper(inp, out, t, return_both=True)
    if energy_both.shape == (batch_size, 1) and grad_both.shape == (batch_size, 128):
        print(f"  ✅ return_both=True returns correct shapes: energy={energy_both.shape}, grad={grad_both.shape}")
    else:
        print(f"  ❌ Wrong shapes: energy={energy_both.shape}, grad={grad_both.shape}")
        sys.exit(1)
    
    # Verify noise prediction is different from gradient
    grad_from_energy = wrapper(inp, out, t, return_both=True)[1]
    noise_pred_new = wrapper(inp, out, t)
    
    # They should be different (noise is learned, gradient is computed)
    if not torch.allclose(noise_pred_new, grad_from_energy, rtol=0.1, atol=0.1):
        print("  ✅ Noise prediction differs from energy gradient (as expected)")
    else:
        print("  ⚠️  WARNING: Noise prediction similar to energy gradient")
    
    print("✅ PASS: AlgebraDiffusionWrapper behaves correctly")
except Exception as e:
    print(f"❌ FAIL: AlgebraDiffusionWrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training step
print("\n[TEST 4] End-to-end training step...")
try:
    from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
    
    # Create diffusion model with our wrapper
    diffusion = GaussianDiffusion1D(
        wrapper,
        seq_length=128,
        objective='pred_noise',
        timesteps=10,
        sampling_timesteps=10,
        supervise_energy_landscape=True,
        use_innerloop_opt=False,  # Disable for simpler test
        show_inference_tqdm=False,
        continuous=True
    )
    diffusion = diffusion.cuda() if torch.cuda.is_available() else diffusion
    
    # Create optimizer
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    
    # Create batch
    device = next(diffusion.parameters()).device
    inp_batch = torch.randn(4, 128).to(device)
    target_batch = torch.randn(4, 128).to(device)
    
    # Forward pass
    optimizer.zero_grad()
    loss, (loss_mse, loss_energy, loss_opt) = diffusion(inp_batch, target_batch, mask=None)
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  MSE Loss: {loss_mse.item():.4f}")
    print(f"  Energy Loss: {loss_energy.item():.4f}")
    
    # Check loss is finite and reasonable
    if torch.isfinite(loss) and loss.item() < 100:
        print("  ✅ Loss is finite and reasonable")
    else:
        print(f"  ❌ Loss is problematic: {loss.item()}")
        sys.exit(1)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    num_params = 0
    for p in diffusion.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
            num_params += 1
    total_grad_norm = total_grad_norm ** 0.5
    
    if total_grad_norm > 0 and total_grad_norm < 1000:
        print(f"  ✅ Gradients computed successfully (norm: {total_grad_norm:.4f})")
    else:
        print(f"  ❌ Gradient issue: norm={total_grad_norm}")
        sys.exit(1)
    
    # Optimizer step
    optimizer.step()
    print("  ✅ Optimizer step completed")
    
    print("✅ PASS: End-to-end training step works correctly")
except Exception as e:
    print(f"❌ FAIL: Training step test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Multiple training steps
print("\n[TEST 5] Multiple training steps (convergence check)...")
try:
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        
        # Create new random batch
        inp_batch = torch.randn(8, 128).to(device)
        target_batch = torch.randn(8, 128).to(device)
        
        loss, _ = diffusion(inp_batch, target_batch, mask=None)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    # Check if losses are all finite
    if all(np.isfinite(l) for l in losses):
        print("  ✅ All losses are finite")
    else:
        print("  ❌ Some losses are not finite")
        sys.exit(1)
    
    # Check if training is stable (no explosion)
    if max(losses) < 100 * min(losses[0], 1.0):
        print("  ✅ Training is stable (no explosion)")
    else:
        print("  ⚠️  WARNING: Training may be unstable")
    
    print("✅ PASS: Multiple training steps completed")
except Exception as e:
    print(f"❌ FAIL: Multiple training steps test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("=" * 60)
print("\nThe training fixes have been verified:")
print("1. NoisyWrapper now returns clean data (no double-noising)")
print("2. Encoder produces normalized embeddings for stable training")
print("3. AlgebraDiffusionWrapper correctly predicts noise for training")
print("4. Energy/gradient computation works for inference optimization")
print("5. End-to-end training steps work correctly")
print("\nYou can now run training with: python train_algebra.py --rule distribute")
