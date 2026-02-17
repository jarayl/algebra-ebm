#!/usr/bin/env python3
"""
Debug a single problem end-to-end to understand failure mode.
Run this ON THE CLUSTER where models are available.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Also add current directory in case we're being called from elsewhere
import os
sys.path.insert(0, os.getcwd())

from src.algebra.algebra_encoder import create_character_encoder
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_inference import load_rule_models

def main():
    print("="*80)
    print("SINGLE PROBLEM DEEP DIVE")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rule = 'distribute'

    # Load model
    print(f"\n1. Loading {rule} model...")
    models = load_rule_models(
        rule_names=[rule],
        model_dir='./results',
        device=device
    )

    if not models or rule not in models:
        print(f"ERROR: Could not load {rule} model")
        return

    model = models[rule]
    print(f"✓ Model loaded on {device}")

    # Create dataset
    print(f"\n2. Creating test problem...")
    dataset = AlgebraDataset(
        rule=rule,
        split='test',
        num_problems=1,
        d_model=128
    )

    input_eq, target_eq = dataset.get_equation_pair(0)
    print(f"Input:  {input_eq}")
    print(f"Target: {target_eq}")

    # Encode
    print(f"\n3. Encoding equations...")
    encoder = create_character_encoder(d_model=128)
    input_emb = encoder(input_eq).to(device)
    target_emb = encoder(target_eq).to(device)

    print(f"Input embedding norm:  {input_emb.norm().item():.4f}")
    print(f"Target embedding norm: {target_emb.norm().item():.4f}")

    initial_dist = (input_emb - target_emb).norm().item()
    print(f"Initial L2 distance: {initial_dist:.4f}")

    # Check energies
    print(f"\n4. Checking energy landscape...")

    # Model may be wrapped - try to get the EBM
    ebm_model = model
    if hasattr(model, 'model') and hasattr(model.model, 'ebm'):
        ebm_model = model.model.ebm
        print(f"Extracted EBM from wrapper")
    elif hasattr(model, 'ebm'):
        ebm_model = model.ebm
        print(f"Extracted EBM from model")

    # AlgebraEBM expects (inp, out, t) where:
    # - inp: input equation embedding (fixed)
    # - out: candidate output embedding (variable)
    # - t: timestep (0 = final landscape)
    timestep = torch.tensor([0.0], device=device)  # Final landscape

    with torch.no_grad():
        # Energy when starting point equals input (should this be high or low?)
        energy_input = ebm_model(input_emb.unsqueeze(0), input_emb.unsqueeze(0), timestep).item()
        # Energy when output equals target (should be LOW - this is the goal)
        energy_target = ebm_model(input_emb.unsqueeze(0), target_emb.unsqueeze(0), timestep).item()

    print(f"Energy(input):  {energy_input:.4f}")
    print(f"Energy(target): {energy_target:.4f}")
    print(f"Energy gap:     {energy_target - energy_input:.4f}")

    if energy_target < energy_input:
        print("✓ Target has LOWER energy (good - inference should move toward it)")
    else:
        print("✗ Target has HIGHER energy (BAD - inference will move away!)")
        print("  This means the model did NOT learn the correct energy landscape!")

    # Check gradient direction
    print(f"\n5. Checking gradient direction...")
    # Gradient w.r.t. output embedding (the variable being optimized)
    output_var = input_emb.clone().detach().requires_grad_(True)
    timestep_grad = torch.tensor([0.0], device=device)
    energy = ebm_model(input_emb.unsqueeze(0), output_var.unsqueeze(0), timestep_grad)
    energy.backward()

    gradient = output_var.grad
    gradient_norm = gradient.norm().item()

    # Direction to target
    direction_to_target = target_emb - input_emb
    direction_to_target_norm = F.normalize(direction_to_target.unsqueeze(0), p=2, dim=-1).squeeze(0)

    # Gradient direction (negative = decrease energy)
    gradient_direction = F.normalize(-gradient.unsqueeze(0), p=2, dim=-1).squeeze(0)

    cosine_sim = (gradient_direction * direction_to_target_norm).sum().item()

    print(f"Gradient norm: {gradient_norm:.4f}")
    print(f"Cosine similarity (grad direction vs target direction): {cosine_sim:.4f}")

    if cosine_sim > 0.5:
        print("✓ Gradient points toward target (good)")
    elif cosine_sim > 0:
        print("~ Gradient points somewhat toward target")
    else:
        print("✗ Gradient points AWAY from target (bad)")

    # Simulate one gradient step
    print(f"\n6. Simulating inference step...")
    step_size = 0.01
    next_emb = input_emb - step_size * gradient
    next_emb = F.normalize(next_emb.unsqueeze(0), p=2, dim=-1).squeeze(0)

    next_dist = (next_emb - target_emb).norm().item()
    dist_change = next_dist - initial_dist
    improvement = (initial_dist - next_dist) / initial_dist

    print(f"Distance before step: {initial_dist:.4f}")
    print(f"Distance after step:  {next_dist:.4f}")
    print(f"Distance change:      {dist_change:+.4f}")
    print(f"Improvement:          {improvement*100:.2f}%")

    if improvement > 0:
        print("✓ Step REDUCES distance (good)")
    else:
        print("✗ Step INCREASES distance (bad)")

    # Energy after step
    with torch.no_grad():
        energy_next = ebm_model(input_emb.unsqueeze(0), next_emb.unsqueeze(0), timestep).item()

    print(f"\nEnergy after step: {energy_next:.4f}")
    print(f"Energy change:     {energy_next - energy_input:+.4f}")

    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)

    if energy_target >= energy_input:
        print("🔴 ROOT CAUSE: Model assigns HIGHER energy to correct target!")
        print("   The trained model has inverted energy landscape.")
        print("   This is a fundamental training failure.")
    elif improvement <= 0:
        print("🔴 ROOT CAUSE: Gradient steps move AWAY from target!")
        print("   Even though target has lower energy, gradient descent")
        print("   is not finding the path to it.")
    else:
        print("🟢 Energy landscape looks correct - need to check other issues:")
        print("   - Decoder coverage (is target in candidates?)")
        print("   - Inference hyperparameters (enough iterations?)")
        print("   - Local minima trapping")

if __name__ == "__main__":
    main()
