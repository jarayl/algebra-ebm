#!/usr/bin/env python3
"""
Test energy landscape across all timesteps to identify if the inversion
is timestep-specific or universal.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.algebra.algebra_encoder import create_character_encoder
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_inference import load_rule_models

def main():
    print("="*80)
    print("TIMESTEP-DEPENDENT ENERGY ANALYSIS")
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

    # Extract EBM
    ebm_model = model
    if hasattr(model, 'model') and hasattr(model.model, 'ebm'):
        ebm_model = model.model.ebm
    elif hasattr(model, 'ebm'):
        ebm_model = model.ebm
    print(f"✓ Extracted EBM")

    # Create test problem
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
    encoder = create_character_encoder(d_model=128)
    input_emb = encoder(input_eq).to(device)
    target_emb = encoder(target_eq).to(device)

    print(f"\n3. Testing energy across ALL timesteps (0-9)...")
    print(f"{'Timestep':<10} {'E(inp→inp)':<12} {'E(inp→tgt)':<12} {'Gap':<12} {'Sign':<10}")
    print("-" * 80)

    for t_val in range(10):
        timestep = torch.tensor([float(t_val)], device=device)

        with torch.no_grad():
            # Energy when output = input (no transformation)
            energy_input = ebm_model(
                input_emb.unsqueeze(0),
                input_emb.unsqueeze(0),
                timestep
            ).item()

            # Energy when output = target (correct transformation)
            energy_target = ebm_model(
                input_emb.unsqueeze(0),
                target_emb.unsqueeze(0),
                timestep
            ).item()

        gap = energy_target - energy_input
        sign = "✓ GOOD" if energy_target < energy_input else "✗ BAD"

        print(f"t={t_val:<8} {energy_input:<12.4f} {energy_target:<12.4f} {gap:+12.4f} {sign}")

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("If E(inp→tgt) < E(inp→inp) at ALL timesteps:")
    print("  → Energy landscape is UNIVERSALLY INVERTED")
    print("  → Training bug affects all timesteps")
    print()
    print("If E(inp→tgt) < E(inp→inp) at SOME timesteps only:")
    print("  → Timestep-dependent issue")
    print("  → May be related to diffusion noise schedule")
    print()
    print("Expected: E(inp→tgt) should be LOWER than E(inp→inp)")
    print("  because (inp→tgt) applies the distribute rule correctly")

if __name__ == "__main__":
    main()
