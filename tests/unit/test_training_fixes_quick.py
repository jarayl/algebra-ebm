#!/usr/bin/env python3
"""
Quick Validation Test: Verify training fixes work properly

Run: python tests/unit/test_training_fixes_quick.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("=" * 70)
print("QUICK TRAINING VALIDATION TEST")
print("=" * 70)

from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper, ContrastiveEnergyLoss
from src.datasets.dataset import NoisyWrapper
from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#############################################################################
# Setup
#############################################################################
print("\n" + "-" * 70)
print("Setting up model and data...")

# Create dataset
dataset = AlgebraDataset(rule='combine', split='train', num_problems=1000, d_model=128)
noisy_dataset = NoisyWrapper(dataset, timesteps=10)

# Create model
ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine').to(device)
wrapper = AlgebraDiffusionWrapper(ebm).to(device)

# Create diffusion model
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

# Optimizer
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

#############################################################################
# Train for 200 steps
#############################################################################
print("\n" + "-" * 70)
print("Training for 200 steps...")

batch_size = 64
dataloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True)

losses = []
mse_losses = []
energy_losses = []
energy_gaps = []

for step, batch in enumerate(dataloader):
    if step >= 200:
        break
    
    inp, out = batch
    inp = inp.to(device)
    out = out.to(device)
    mask = None
    
    optimizer.zero_grad()
    loss, (loss_mse, loss_energy, loss_opt) = diffusion(inp, out, mask)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=10.0)
    
    optimizer.step()
    
    losses.append(loss.item())
    mse_losses.append(loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse)
    if torch.is_tensor(loss_energy):
        energy_losses.append(loss_energy.item())
    
    if step % 50 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, mse={loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse:.4f}, energy={loss_energy.item() if torch.is_tensor(loss_energy) else loss_energy:.4f}")

#############################################################################
# Evaluate energy gap after training
#############################################################################
print("\n" + "-" * 70)
print("Evaluating energy gap after training...")

# Get some test samples
test_inp = []
test_out = []
for i in range(50):
    inp, out = dataset[i]
    test_inp.append(inp)
    test_out.append(out)

test_inp = torch.stack(test_inp).to(device)
test_out = torch.stack(test_out).to(device)
t = torch.zeros(50, device=device, dtype=torch.long)

# Valid pairs
with torch.no_grad():
    energy_valid = ebm(test_inp, test_out, t)

# Invalid pairs (shuffled)
shuffled_out = test_out[torch.randperm(50)]
with torch.no_grad():
    energy_shuffled = ebm(test_inp, shuffled_out, t)

# Invalid pairs (random)
random_out = torch.randn_like(test_out)
random_out = F.normalize(random_out, p=2, dim=-1)
with torch.no_grad():
    energy_random = ebm(test_inp, random_out, t)

print(f"\nEnergy values (lower = more valid):")
print(f"  Valid pairs:    E = {energy_valid.mean().item():.4f} ± {energy_valid.std().item():.4f}")
print(f"  Shuffled pairs: E = {energy_shuffled.mean().item():.4f} ± {energy_shuffled.std().item():.4f}")
print(f"  Random pairs:   E = {energy_random.mean().item():.4f} ± {energy_random.std().item():.4f}")

gap_valid_shuffled = energy_shuffled.mean() - energy_valid.mean()
gap_valid_random = energy_random.mean() - energy_valid.mean()

print(f"\nEnergy gaps (positive = model learned distinction):")
print(f"  Shuffled - Valid: {gap_valid_shuffled.item():.4f}")
print(f"  Random - Valid:   {gap_valid_random.item():.4f}")

#############################################################################
# Summary
#############################################################################
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"\nLoss trajectory:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Improvement:  {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

print(f"\nEnergy loss trajectory:")
print(f"  Initial: {energy_losses[0]:.4f}")
print(f"  Final:   {energy_losses[-1]:.4f}")

if gap_valid_shuffled.item() > 0 or gap_valid_random.item() > 0:
    print(f"\n✓ SUCCESS: Model is learning to distinguish valid from invalid pairs!")
    print(f"  The energy for invalid pairs is higher than for valid pairs.")
else:
    print(f"\n⚠️ WARNING: Model hasn't yet learned to distinguish valid from invalid pairs.")
    print(f"  This may require more training steps or hyperparameter tuning.")

# Check contrastive loss targets
print(f"\n" + "-" * 70)
print("Contrastive Loss Target Check:")
print(f"  Target for valid (pos_target): 1.0")
print(f"  Target for invalid (neg_target): 10.0")
print(f"  Actual valid energy: {energy_valid.mean().item():.4f}")
print(f"  Actual shuffled energy: {energy_shuffled.mean().item():.4f}")

if energy_valid.mean().item() < 3.0:
    print(f"  ✓ Valid energies are in reasonable range (< 3.0)")
else:
    print(f"  ⚠️ Valid energies are still high - may need more training")

# Check learnable parameters
print(f"\nLearnable energy scaling:")
print(f"  energy_scale: {ebm.energy_scale.item():.4f}")
print(f"  energy_bias: {ebm.energy_bias.item():.4f}")
