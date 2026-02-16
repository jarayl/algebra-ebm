#!/usr/bin/env python3
"""
Debug script to diagnose why single-rule evaluation is failing (6.3% accuracy).

This script will:
1. Load a trained model
2. Create a simple test problem
3. Run inference step-by-step with detailed logging
4. Compare training vs inference energy landscapes
5. Check decoder functionality
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.algebra.algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_inference import load_rule_models, AlgebraInference, InferenceConfig

def test_model_loading(model_dir='./results', rule='distribute'):
    """Test 1: Verify model loads correctly"""
    print("="*80)
    print("TEST 1: Model Loading")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    models = load_rule_models(
        rule_names=[rule],
        model_dir=model_dir,
        device=device
    )

    if not models or rule not in models:
        print(f"❌ Failed to load {rule} model")
        return None

    model = models[rule]
    print(f"✅ Loaded {rule} model successfully")
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    return models


def test_dataset_generation(rule='distribute', num_problems=5):
    """Test 2: Check dataset generation with fixed format"""
    print("\n" + "="*80)
    print("TEST 2: Dataset Generation (DATAGEN-002 format)")
    print("="*80)

    dataset = AlgebraDataset(
        rule=rule,
        split='test',
        num_problems=num_problems,
        d_model=128
    )

    print(f"Generated {len(dataset)} test problems")
    print("\nSample problems:")

    for i in range(min(3, len(dataset))):
        input_eq, target_eq = dataset.get_equation_pair(i)
        print(f"  {i+1}. Input:  {input_eq}")
        print(f"     Target: {target_eq}")
        print()

    return dataset


def test_encoder_decoder(encoder, decoder, dataset):
    """Test 3: Verify encoder/decoder functionality"""
    print("\n" + "="*80)
    print("TEST 3: Encoder/Decoder")
    print("="*80)

    input_eq, target_eq = dataset.get_equation_pair(0)

    # Encode
    input_emb = encoder(input_eq)
    target_emb = encoder(target_eq)

    print(f"Input equation: {input_eq}")
    print(f"Input embedding shape: {input_emb.shape}")
    print(f"Input embedding norm: {input_emb.norm().item():.4f}")

    print(f"\nTarget equation: {target_eq}")
    print(f"Target embedding shape: {target_emb.shape}")
    print(f"Target embedding norm: {target_emb.norm().item():.4f}")

    # Check L2 distance
    l2_dist = (input_emb - target_emb).norm().item()
    print(f"\nL2 distance (input → target): {l2_dist:.4f}")

    # Decode
    decoded_input, dist_input = decoder.decode_embedding(input_emb)
    decoded_target, dist_target = decoder.decode_embedding(target_emb)

    print(f"\nDecoded input:  {decoded_input} (distance: {dist_input:.4f})")
    print(f"Decoded target: {decoded_target} (distance: {dist_target:.4f})")

    # Check if target is in candidate set
    print(f"\nDecoder has {len(decoder.candidate_equations)} candidates")
    if target_eq in decoder.candidate_equations:
        print(f"✅ Target equation '{target_eq}' IS in decoder candidates")
    else:
        print(f"❌ Target equation '{target_eq}' NOT in decoder candidates")
        # Find closest match
        closest_idx = np.argmin([
            (encoder(cand) - target_emb).norm().item()
            for cand in decoder.candidate_equations[:100]  # Check first 100
        ])
        closest_cand = decoder.candidate_equations[closest_idx]
        closest_dist = (encoder(closest_cand) - target_emb).norm().item()
        print(f"   Closest candidate: '{closest_cand}' (distance: {closest_dist:.4f})")


def test_energy_landscape(model, encoder, dataset, device='cpu'):
    """Test 4: Check energy landscape"""
    print("\n" + "="*80)
    print("TEST 4: Energy Landscape Analysis")
    print("="*80)

    input_eq, target_eq = dataset.get_equation_pair(0)

    input_emb = encoder(input_eq).to(device)
    target_emb = encoder(target_eq).to(device)

    print(f"Problem: {input_eq} → {target_eq}")

    # Test energies at different points
    with torch.no_grad():
        energy_input = model(input_emb.unsqueeze(0)).item()
        energy_target = model(target_emb.unsqueeze(0)).item()

        # Random point
        random_emb = torch.randn_like(input_emb)
        random_emb = F.normalize(random_emb, p=2, dim=-1)
        energy_random = model(random_emb.unsqueeze(0)).item()

        # Midpoint
        mid_emb = (input_emb + target_emb) / 2
        mid_emb = F.normalize(mid_emb, p=2, dim=-1)
        energy_mid = model(mid_emb.unsqueeze(0)).item()

    print(f"\nEnergy at input:   {energy_input:.4f}")
    print(f"Energy at target:  {energy_target:.4f}")
    print(f"Energy at random:  {energy_random:.4f}")
    print(f"Energy at midpoint: {energy_mid:.4f}")
    print(f"\nEnergy gap (input → target): {energy_target - energy_input:.4f}")

    if energy_target < energy_input:
        print("✅ Target has LOWER energy (correct - inference should move toward it)")
    else:
        print("❌ Target has HIGHER energy (problem - inference will move away!)")


def test_gradient_direction(model, encoder, dataset, device='cpu'):
    """Test 5: Check gradient direction"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Direction Analysis")
    print("="*80)

    input_eq, target_eq = dataset.get_equation_pair(0)

    input_emb = encoder(input_eq).to(device)
    target_emb = encoder(target_eq).to(device)

    # Compute gradient at input
    input_emb_var = input_emb.clone().detach().requires_grad_(True)
    energy = model(input_emb_var.unsqueeze(0))
    energy.backward()

    gradient = input_emb_var.grad
    gradient_norm = gradient.norm().item()

    print(f"Gradient norm at input: {gradient_norm:.4f}")

    # Direction toward target
    direction_to_target = target_emb - input_emb
    direction_to_target_normalized = F.normalize(direction_to_target.unsqueeze(0), p=2, dim=-1).squeeze(0)

    # Gradient direction (negative gradient = direction of decrease)
    gradient_direction = F.normalize(-gradient.unsqueeze(0), p=2, dim=-1).squeeze(0)

    # Cosine similarity
    cosine_sim = (gradient_direction * direction_to_target_normalized).sum().item()

    print(f"\nCosine similarity (gradient direction, direction to target): {cosine_sim:.4f}")

    if cosine_sim > 0.5:
        print("✅ Gradient points TOWARD target (angle < 60°)")
    elif cosine_sim > 0:
        print("⚠️  Gradient points somewhat toward target (60° < angle < 90°)")
    elif cosine_sim > -0.5:
        print("❌ Gradient points somewhat AWAY from target (90° < angle < 120°)")
    else:
        print("❌ Gradient points strongly AWAY from target (angle > 120°)")

    # Step in gradient direction
    step_size = 0.01
    next_emb = input_emb - step_size * gradient
    next_emb = F.normalize(next_emb.unsqueeze(0), p=2, dim=-1).squeeze(0)

    dist_before = (input_emb - target_emb).norm().item()
    dist_after = (next_emb - target_emb).norm().item()

    print(f"\nDistance to target before step: {dist_before:.4f}")
    print(f"Distance to target after step:  {dist_after:.4f}")
    print(f"Distance change: {dist_after - dist_before:.4f}")

    if dist_after < dist_before:
        print("✅ Gradient step REDUCES distance to target")
    else:
        print("❌ Gradient step INCREASES distance to target")


def test_full_inference(models, encoder, decoder, dataset, device='cpu'):
    """Test 6: Run full inference with logging"""
    print("\n" + "="*80)
    print("TEST 6: Full Inference Pipeline")
    print("="*80)

    input_eq, target_eq = dataset.get_equation_pair(0)

    print(f"Problem: {input_eq} → {target_eq}")

    inference_engine = AlgebraInference(
        rule_models=models,
        encoder=encoder,
        decoder=decoder
    )

    config = InferenceConfig(
        max_iterations=50,
        step_size=0.01,
        use_adaptive_step=False
    )

    # Run inference with single rule
    rule = list(models.keys())[0]
    rule_weights = {rule: 1.0}

    result = inference_engine.solve_equation(
        input_eq,
        config=config,
        rule_weights=rule_weights,
        distance_threshold=2.0
    )

    print(f"\nInference completed:")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Output equation: {result.get('output_equation', 'None')}")
    print(f"  Decoding distance: {result.get('decoding_distance', float('inf')):.4f}")

    # Check if output is correct
    output_eq = result.get('output_equation')
    if output_eq == target_eq:
        print(f"✅ Output MATCHES target exactly")
    else:
        print(f"❌ Output does NOT match target")
        print(f"   Expected: {target_eq}")
        print(f"   Got:      {output_eq}")

    # Check inference info
    if 'inference_info' in result:
        info = result['inference_info']
        print(f"\nInference statistics:")
        print(f"  Final energy: {info.get('final_energy', 'N/A')}")
        print(f"  Acceptance rate: {info.get('acceptance_rate', 'N/A')}")
        print(f"  Iterations: {info.get('num_iterations', 'N/A')}")


def main():
    """Run all diagnostic tests"""
    print("ALGEBRA EBM INFERENCE DIAGNOSTICS")
    print("="*80)
    print("Testing why single-rule evaluation achieves only 6.3% accuracy")
    print("="*80)

    rule = 'distribute'
    model_dir = './results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test 1: Load model
    models = test_model_loading(model_dir, rule)
    if not models:
        print("\n❌ CRITICAL: Cannot proceed without model")
        return

    # Test 2: Generate dataset
    dataset = test_dataset_generation(rule, num_problems=5)

    # Test 3: Encoder/decoder
    encoder = create_character_encoder(d_model=128)
    decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
    test_encoder_decoder(encoder, decoder, dataset)

    # Test 4: Energy landscape
    model = models[rule].to(device)
    test_energy_landscape(model, encoder, dataset, device)

    # Test 5: Gradient direction
    test_gradient_direction(model, encoder, dataset, device)

    # Test 6: Full inference
    test_full_inference(models, encoder, decoder, dataset, device)

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
