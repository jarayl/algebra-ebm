"""
Comprehensive diagnosis of why trained model gets 0% accuracy at inference.

This investigates:
1. Does the energy correctly distinguish valid vs invalid solutions?
2. Does gradient descent on energy move toward valid solutions?
3. Is the MSE loss objective conflicting with energy-based inference?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, '/home/ubuntu/algebra-ebm')

from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper, ContrastiveEnergyLoss
from src.algebra.algebra_encoder import create_character_encoder
from src.algebra.algebra_dataset import generate_single_transformation
from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
print("\n" + "="*70)
print("Creating and training model...")
print("="*70)

ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name='combine')
model = AlgebraDiffusionWrapper(ebm).to(device)
encoder = create_character_encoder(d_model=128)

# Initialize diffusion with IRED settings
diffusion = GaussianDiffusion1D(
    model=model,
    seq_length=128,
    timesteps=10,
    use_innerloop_opt=True,
    step_size_multiplier=0.1,
    use_contrastive_energy_loss=True
).to(device)

# Create simple training data
print("\nGenerating training data...")
from torch.utils.data import Dataset, DataLoader

class SimpleAlgebraDataset(Dataset):
    def __init__(self, num_samples=1000, rule='combine'):
        self.data = []
        self.encoder = create_character_encoder(d_model=128)
        
        for _ in range(num_samples):
            try:
                inp_eq, out_eq, rule_name = generate_single_transformation(rule_type=rule)
                inp_emb = self.encoder.encode(inp_eq)
                out_emb = self.encoder.encode(out_eq)
                self.data.append((
                    torch.tensor(inp_emb, dtype=torch.float32),
                    torch.tensor(out_emb, dtype=torch.float32)
                ))
            except:
                continue
                
        print(f"Generated {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inp, out = self.data[idx]
        # Return (target, condition, mask) format expected by Trainer1D
        return out, inp, torch.zeros(1)

dataset = SimpleAlgebraDataset(num_samples=500)

# Quick train for 500 steps
print("\n" + "="*70)
print("Training for 500 steps...")
print("="*70)

trainer = Trainer1D(
    diffusion_model=diffusion,
    dataset=dataset,
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=500,
    gradient_accumulate_every=1,
    save_and_sample_every=10000,  # Don't save during debug
    results_folder='./debug_results'
)

# Train (suppress output)
import warnings
warnings.filterwarnings('ignore')
trainer.train()

# Now diagnose the trained model
print("\n" + "="*70)
print("DIAGNOSIS 1: Energy discrimination")
print("="*70)

# Get a batch of real data
test_inp, test_out = dataset.data[0]
test_inp = test_inp.unsqueeze(0).to(device)
test_out = test_out.unsqueeze(0).to(device)

# Compute energy at various timesteps
for t_val in [0, 4, 9]:
    t = torch.tensor([t_val], device=device)
    
    # Energy for correct output
    energy_correct = ebm(test_inp, test_out, t).item()
    
    # Energy for random output
    random_out = torch.randn_like(test_out)
    energy_random = ebm(test_inp, random_out, t).item()
    
    # Energy for slightly perturbed output
    perturbed_out = test_out + 0.1 * torch.randn_like(test_out)
    energy_perturbed = ebm(test_inp, perturbed_out, t).item()
    
    print(f"\nTimestep t={t_val}:")
    print(f"  Energy (correct output):   {energy_correct:.4f}")
    print(f"  Energy (random output):    {energy_random:.4f}")
    print(f"  Energy (perturbed 0.1):    {energy_perturbed:.4f}")
    print(f"  → Ratio (random/correct):  {energy_random/max(energy_correct, 1e-6):.2f}x")

print("\n" + "="*70)
print("DIAGNOSIS 2: Gradient descent direction")
print("="*70)

# Start from random noise and do gradient descent
t = torch.tensor([0], device=device)
x = torch.randn(1, 128, device=device)

# Record trajectory
trajectory = [x.clone()]
energies = []

# Get target direction (ideal)
target_direction = (test_out - x) / torch.norm(test_out - x)

print("\nGradient descent from noise toward target:")
print(f"Initial distance to target: {torch.norm(test_out - x).item():.4f}")

for step in range(20):
    x.requires_grad_(True)
    energy = ebm(test_inp, x, t)
    grad = torch.autograd.grad(energy.sum(), x)[0]
    
    energies.append(energy.item())
    
    # Take step
    step_size = 0.5
    x_new = x - step_size * grad
    
    # Compute metrics
    dist_to_target = torch.norm(test_out - x_new).item()
    grad_dot_target = (grad.flatten() @ target_direction.flatten()).item()
    
    if step % 5 == 0:
        print(f"  Step {step}: energy={energy.item():.4f}, "
              f"dist_to_target={dist_to_target:.4f}, "
              f"grad·target_dir={grad_dot_target:.4f}")
    
    x = x_new.detach()
    trajectory.append(x.clone())

print(f"Final distance to target: {torch.norm(test_out - x).item():.4f}")

# Check if we actually got closer
initial_dist = torch.norm(test_out - trajectory[0]).item()
final_dist = torch.norm(test_out - trajectory[-1]).item()
print(f"\n→ {'IMPROVED' if final_dist < initial_dist else 'GOT WORSE'}: "
      f"{initial_dist:.4f} → {final_dist:.4f}")

print("\n" + "="*70)
print("DIAGNOSIS 3: MSE loss objective vs Energy")
print("="*70)

# The key insight: MSE loss trains the model to output predicted x0 (denoised)
# But the energy should be LOW for correct outputs
# These could conflict!

# During training:
# - loss_mse trains model to predict x0 from x_t
# - loss_energy trains model to have low energy for valid pairs

# At inference:
# - We use gradient of energy to refine predictions
# But wait - what IS the model outputting?

# Let's check what the diffusion model actually computes
print("\nChecking diffusion model predictions...")

with torch.no_grad():
    t = torch.tensor([5], device=device)
    
    # Noisy input at timestep t
    noise = torch.randn_like(test_out)
    alphas_cumprod_t = diffusion.alphas_cumprod[t].item()
    x_t = np.sqrt(alphas_cumprod_t) * test_out + np.sqrt(1 - alphas_cumprod_t) * noise
    
    # What does the model predict?
    # The model in this setup returns the GRADIENT of energy
    # NOT the denoised x0 or noise!
    output = model(test_inp, x_t, t)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output norm: {torch.norm(output).item():.4f}")
    
    # The diffusion code expects model output to be:
    # - pred_noise (if objective='pred_noise')
    # - pred_x0 (if objective='pred_x0')
    # But our model outputs GRADIENT!
    
    print(f"\nDiffusion objective: {diffusion.objective}")

print("\n" + "="*70)
print("DIAGNOSIS 4: Checking model_predictions flow")
print("="*70)

# This is the critical path - what does model_predictions do?
with torch.no_grad():
    t = torch.tensor([5], device=device)
    
    # Create noisy x_t
    noise = torch.randn_like(test_out)
    alphas_cumprod_t = diffusion.alphas_cumprod[t].item()
    x_t = np.sqrt(alphas_cumprod_t) * test_out + np.sqrt(1 - alphas_cumprod_t) * noise
    
    # Get model prediction
    preds = diffusion.model_predictions(test_inp, x_t, t)
    
    print(f"Predicted noise shape: {preds.pred_noise.shape}")
    print(f"Predicted x_start shape: {preds.pred_x_start.shape}")
    
    # Compare predicted x_start to actual
    pred_error = torch.norm(preds.pred_x_start - test_out).item()
    print(f"\nError in predicted x_start: {pred_error:.4f}")
    print(f"Actual x_start norm: {torch.norm(test_out).item():.4f}")
    print(f"Predicted x_start norm: {torch.norm(preds.pred_x_start).item():.4f}")

print("\n" + "="*70)
print("DIAGNOSIS 5: Full inference loop test")
print("="*70)

# Run actual sampling
diffusion.show_inference_tqdm = False
diffusion.use_innerloop_opt = True

samples = diffusion.p_sample_loop(
    batch_size=1,
    shape=(128,),
    inp=test_inp,
    cond=None,
    mask=None,
    return_traj=True
)

print(f"Trajectory shape: {samples.shape}")

# Check distance to target at each timestep
print("\nSampling trajectory distances to target:")
for i, sample in enumerate(samples):
    if i % 2 == 0:  # Every other step
        dist = torch.norm(sample - test_out).item()
        print(f"  Timestep {9-i}: distance to target = {dist:.4f}")

final_dist = torch.norm(samples[-1] - test_out).item()
random_baseline = torch.norm(torch.randn_like(test_out) - test_out).item()

print(f"\nFinal distance: {final_dist:.4f}")
print(f"Random baseline: {random_baseline:.4f}")
print(f"→ {'BETTER than random' if final_dist < random_baseline else 'WORSE than random'}")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("""
Key findings to check:
1. Does energy distinguish valid vs invalid? (Should be yes if training works)
2. Does gradient point toward valid solutions? (Critical for inference)
3. Is the model output being interpreted correctly by diffusion?
4. Does the inference loop actually use energy gradients?
""")
