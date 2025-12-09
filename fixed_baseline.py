#!/usr/bin/env python3
"""
Fixed baseline evaluation using proper IRED sampling methodology.
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
    
    # Extract EBM state dict from 'ema' which is typically better
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
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

def proper_ired_evaluation(num_samples=50):
    """
    Proper IRED evaluation using the same methodology as training.
    
    Key insight: IRED evaluates by:
    1. Starting from noise
    2. Using diffusion.sample() to converge to solution  
    3. Measuring if the result is close to the target
    """
    print("="*60)
    print("PROPER IRED BASELINE EVALUATION")
    print("="*60)
    
    # Load model (try EMA first as it's usually better)
    ebm = load_checkpoint_fixed('results/distribute/model-12.pt')
    if ebm is None:
        print("Failed to load checkpoint")
        return None
    
    # Create diffusion with same settings as training
    model_wrapper = AlgebraDiffusionWrapper(ebm)
    diffusion = GaussianDiffusion1D(
        model_wrapper,
        seq_length=128,
        timesteps=10,
        sampling_timesteps=10,
        supervise_energy_landscape=True,
        use_innerloop_opt=True,
        use_contrastive_energy_loss=True,
        step_size_multiplier=0.1,
        show_inference_tqdm=False,
        continuous=True
    )
    
    # Load test data
    dataset = AlgebraDataset(
        rule='distribute',
        split='test',
        num_problems=num_samples,
        d_model=128
    )
    
    print(f"Evaluating on {min(num_samples, len(dataset))} samples...")
    
    results = []
    distances_threshold = 0.5  # Distance threshold for "correct"
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        inp = sample[0].unsqueeze(0)  # Input equation (1, 128)
        target = sample[1].unsqueeze(0)  # Target solution (1, 128)
        
        # IRED inference: sample() method runs full annealing
        # x=inp provides the conditioning (input equation)
        # label=None for unconditional generation (find any valid solution)
        with torch.no_grad():
            # This is the key: diffusion.sample() implements full IRED
            generated = diffusion.sample(
                x=inp,        # Input equation to solve
                label=None,   # Unconditional (find solution for equation)
                mask=None,
                batch_size=1
            )
        
        # Measure success: is generated solution close to target?
        distance_to_target = (generated - target).norm().item()
        is_correct = distance_to_target < distances_threshold
        
        results.append({
            'distance_to_target': distance_to_target,
            'is_correct': is_correct
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples")
    
    # Calculate accuracy
    accuracy = np.mean([r['is_correct'] for r in results])
    mean_distance = np.mean([r['distance_to_target'] for r in results])
    std_distance = np.std([r['distance_to_target'] for r in results])
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Accuracy (distance < {distances_threshold}): {accuracy*100:.1f}%")
    print(f"Mean distance to target: {mean_distance:.4f} ± {std_distance:.4f}")
    print(f"Number of samples: {len(results)}")
    
    # Try different thresholds
    for threshold in [0.1, 0.2, 0.5, 1.0, 2.0]:
        acc = np.mean([r['distance_to_target'] < threshold for r in results])
        print(f"Accuracy (distance < {threshold:3.1f}): {acc*100:.1f}%")
    
    if accuracy >= 0.87:
        print(f"\n✓ BASELINE MEETS 87%+ TARGET (with threshold {distances_threshold})")
    else:
        print(f"\n✗ BASELINE BELOW 87% TARGET (with threshold {distances_threshold})")
        # Check if any threshold gives 87%+
        for threshold in [0.5, 1.0, 2.0, 3.0]:
            acc = np.mean([r['distance_to_target'] < threshold for r in results])
            if acc >= 0.87:
                print(f"  ✓ Would meet 87%+ target with threshold {threshold}")
                break
    
    return {
        'accuracy': accuracy,
        'mean_distance': mean_distance,
        'threshold_used': distances_threshold,
        'num_samples': len(results),
        'meets_target': accuracy >= 0.87,
        'all_distances': [r['distance_to_target'] for r in results]
    }

def analyze_training_performance():
    """Check what performance the model achieved during training."""
    print("\nChecking training performance from latest checkpoint...")
    
    try:
        checkpoint = torch.load('results/distribute/model-12.pt', map_location='cpu')
        step = checkpoint.get('step', 'unknown')
        print(f"Checkpoint from training step: {step}")
        
        # Look for validation metrics if saved
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print("Training metrics:", metrics)
    except Exception as e:
        print(f"Could not load training metrics: {e}")

if __name__ == '__main__':
    analyze_training_performance()
    results = proper_ired_evaluation(50)
    if results:
        print(f"\nSUMMARY:")
        print(f"Final baseline accuracy: {results['accuracy']*100:.1f}%")
        print(f"Target met: {results['meets_target']}")