#!/usr/bin/env python3
"""
Comprehensive Testing Script for Algebra EBM Training

This script verifies:
1. Energy model training correctness
2. Good solutions have much lower energy than bad solutions (energy separation)
3. Energy landscape is smooth at early inference (high k, near t=T-1) but sharp at final inference (low k, near t=0)
4. Target solution is the local minima in the final energy landscape (t=0)

Usage:
    # Train for 5000 steps and run all tests
    python tests/test_algebra_ebm_training.py --train_steps 5000 --rule distribute
    
    # Train for 10000 steps with verbose output
    python tests/test_algebra_ebm_training.py --train_steps 10000 --rule distribute --verbose
    
    # Load existing checkpoint and test
    python tests/test_algebra_ebm_training.py --load_checkpoint results/distribute/model-5.pt --rule distribute
"""

import os
import sysw
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_encoder import create_character_encoder
from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from src.datasets.dataset import NoisyWrapper


class AlgebraEBMTester:
    """
    Comprehensive testing class for Algebra EBM models.
    
    Tests:
    1. Training correctness - loss decreases, model parameters update
    2. Energy separation - good solutions have lower energy than bad
    3. Landscape smoothness - smooth at high k (early inference), sharp at low k (final inference)
    4. Local minima - target is minima in final landscape (t=0)
    """
    
    def __init__(
        self,
        rule: str = 'distribute',
        d_model: int = 128,
        timesteps: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = False
    ):
        self.rule = rule
        self.d_model = d_model
        self.timesteps = timesteps
        self.device = device
        self.verbose = verbose
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Create models
        self.ebm = None
        self.diffusion = None
        self.trainer = None
        
        # Store test results
        self.results = {
            'training': {},
            'energy_separation': {},
            'landscape_smoothness': {},
            'local_minima': {}
        }
        
        print(f"[AlgebraEBMTester] Initialized for rule '{rule}' on {device}")
    
    def setup_model(self) -> None:
        """Initialize the EBM model and diffusion wrapper."""
        print("[Setup] Creating AlgebraEBM model...")
        
        self.ebm = AlgebraEBM(
            inp_dim=self.d_model,
            out_dim=self.d_model,
            rule_name=self.rule
        ).to(self.device)
        
        self.model_wrapper = AlgebraDiffusionWrapper(self.ebm)
        
        print(f"[Setup] Model created with {sum(p.numel() for p in self.ebm.parameters())} parameters")
    
    def setup_dataset(self, num_problems: int = 10000) -> AlgebraDataset:
        """Create training dataset."""
        print(f"[Setup] Creating dataset with {num_problems} problems...")
        
        dataset = AlgebraDataset(
            rule=self.rule,
            split='train',
            num_problems=num_problems,
            d_model=self.d_model
        )
        
        return dataset
    
    def train_model(
        self,
        train_steps: int = 5000,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        results_folder: str = None,
        num_problems: int = 50000
    ) -> Dict:
        """
        Train the EBM model and track training metrics.
        
        Returns training statistics including loss history.
        """
        print(f"\n{'='*60}")
        print(f"[Training] Starting training for {train_steps} steps")
        print(f"{'='*60}")
        
        if self.ebm is None:
            self.setup_model()
        
        # Create dataset
        dataset = self.setup_dataset(num_problems=num_problems)
        noisy_dataset = NoisyWrapper(dataset, timesteps=self.timesteps)
        
        # Setup diffusion
        # NOTE: continuous=True is important for algebra embeddings
        # This matches train_algebra.py production training configuration
        self.diffusion = GaussianDiffusion1D(
            self.model_wrapper,
            seq_length=self.d_model,
            timesteps=self.timesteps,
            sampling_timesteps=self.timesteps,
            supervise_energy_landscape=True,
            use_innerloop_opt=True,
            use_contrastive_energy_loss=True,
            enable_loss_balance_monitoring=True,
            step_size_multiplier=0.1,
            show_inference_tqdm=False,
            continuous=True  # IMPORTANT: Must match train_algebra.py for consistent behavior
        ).to(self.device)
        
        # Setup trainer
        if results_folder is None:
            results_folder = f'./test_results/{self.rule}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        os.makedirs(results_folder, exist_ok=True)
        
        self.trainer = Trainer1D(
            self.diffusion,
            noisy_dataset,
            train_batch_size=batch_size,
            train_lr=learning_rate,
            train_num_steps=train_steps,
            gradient_accumulate_every=1,
            results_folder=results_folder,
            save_and_sample_every=max(1000, train_steps // 5),
            amp=True,
            fp16=True,
            data_workers=4
        )
        
        # Store initial parameters for comparison
        initial_params = {name: param.clone().detach() 
                         for name, param in self.ebm.named_parameters()}
        
        # Train
        print(f"[Training] Training with batch_size={batch_size}, lr={learning_rate}")
        self.trainer.train()
        
        # Verify parameters changed
        params_changed = 0
        for name, param in self.ebm.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed += 1
        
        training_results = {
            'train_steps': train_steps,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'parameters_updated': params_changed,
            'total_parameters': len(initial_params),
            'results_folder': results_folder
        }
        
        self.results['training'] = training_results
        
        print(f"\n[Training] Complete!")
        print(f"  - Parameters updated: {params_changed}/{len(initial_params)}")
        
        return training_results
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model checkpoint."""
        print(f"[Load] Loading checkpoint from {checkpoint_path}")
        
        if self.ebm is None:
            self.setup_model()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            self.ebm.load_state_dict(checkpoint['model'])
        elif 'ema' in checkpoint:
            self.ebm.load_state_dict(checkpoint['ema'])
        else:
            # Try loading directly
            self.ebm.load_state_dict(checkpoint)
        
        self.ebm.eval()
        print("[Load] Checkpoint loaded successfully")
    
    def test_energy_separation(
        self,
        num_samples: int = 100,
        margin_threshold: float = 0.5,
        noise_scale: float = 0.5
    ) -> Dict:
        """
        Test 1: Verify good solutions have much lower energy than bad solutions.
        
        Creates positive (correct transformation) and negative (corrupted) samples
        and measures the energy gap between them.
        
        Note: The AlgebraDataset returns (input, target) pairs. We generate negative
        samples by adding noise to the target embeddings.
        """
        print(f"\n{'='*60}")
        print("[Test 1] Energy Separation Test")
        print(f"{'='*60}")
        
        if self.ebm is None:
            raise RuntimeError("Model not loaded. Train or load checkpoint first.")
        
        self.ebm.eval()
        
        # Create test dataset
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        pos_energies = []
        neg_energies = []
        energy_gaps = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                # Get sample (inp, target) - dataset returns 2 tensors
                sample = dataset[i]
                inp = sample[0].unsqueeze(0).to(self.device)
                pos = sample[1].unsqueeze(0).to(self.device)  # Correct target
                
                # Generate negative sample by corrupting the target
                # This simulates an incorrect transformation
                noise = torch.randn_like(pos) * noise_scale
                neg = pos + noise
                
                # t=0 is the FINAL timestep in diffusion (sharpest landscape)
                # During inference, we go from t=T-1 (noisy) down to t=0 (clean)
                t = torch.tensor([0], device=self.device, dtype=torch.float)
                
                # Compute energies
                E_pos = self.ebm(inp, pos, t).item()
                E_neg = self.ebm(inp, neg, t).item()
                
                pos_energies.append(E_pos)
                neg_energies.append(E_neg)
                energy_gaps.append(E_neg - E_pos)
        
        # Handle empty results
        if len(pos_energies) == 0:
            print("\n  ✗ ERROR: No samples processed!")
            results = {
                'num_samples': 0,
                'error': 'No samples processed',
                'passed': False
            }
            self.results['energy_separation'] = results
            return results
        
        # Compute statistics
        results = {
            'num_samples': len(pos_energies),
            'pos_energy_mean': float(np.mean(pos_energies)),
            'pos_energy_std': float(np.std(pos_energies)),
            'neg_energy_mean': float(np.mean(neg_energies)),
            'neg_energy_std': float(np.std(neg_energies)),
            'energy_gap_mean': float(np.mean(energy_gaps)),
            'energy_gap_std': float(np.std(energy_gaps)),
            'energy_gap_min': float(np.min(energy_gaps)),
            'energy_gap_max': float(np.max(energy_gaps)),
            'separation_rate': float(np.mean([g > margin_threshold for g in energy_gaps])),
            'noise_scale': noise_scale,
            'passed': float(np.mean(energy_gaps)) > margin_threshold
        }
        
        self.results['energy_separation'] = results
        
        # Print results
        print(f"\n  Positive (correct) energy: {results['pos_energy_mean']:.4f} ± {results['pos_energy_std']:.4f}")
        print(f"  Negative (corrupted) energy: {results['neg_energy_mean']:.4f} ± {results['neg_energy_std']:.4f}")
        print(f"  Energy gap (neg - pos): {results['energy_gap_mean']:.4f} ± {results['energy_gap_std']:.4f}")
        print(f"  Separation rate (gap > {margin_threshold}): {results['separation_rate']*100:.1f}%")
        
        if results['passed']:
            print(f"\n  ✓ PASSED: Mean energy gap ({results['energy_gap_mean']:.4f}) > threshold ({margin_threshold})")
        else:
            print(f"\n  ✗ FAILED: Mean energy gap ({results['energy_gap_mean']:.4f}) <= threshold ({margin_threshold})")
        
        return results
    
    def test_landscape_smoothness(
        self,
        num_samples: int = 20,
        perturbation_scales: List[float] = None
    ) -> Dict:
        """
        Test 2: Verify landscape is smooth at high k (early inference) and sharp at low k (final inference).
        
        In IRED/diffusion:
        - t=0 (low k) = FINAL inference step = clean/sharp landscape
        - t=T-1 (high k) = INITIAL inference step = noisy/smooth landscape
        
        Inference iterates: t = T-1, T-2, ..., 1, 0 (high to low)
        
        We measure energy sensitivity to perturbations at each timestep.
        High k (early) should have LOW sensitivity (smooth landscape).
        Low k (final) should have HIGH sensitivity (sharp landscape).
        """
        print(f"\n{'='*60}")
        print("[Test 2] Landscape Smoothness Test")
        print(f"{'='*60}")
        
        if perturbation_scales is None:
            perturbation_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        if self.ebm is None:
            raise RuntimeError("Model not loaded. Train or load checkpoint first.")
        
        self.ebm.eval()
        
        # Create test samples
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        # Measure energy sensitivity at different timesteps
        timestep_sensitivities = {}
        
        actual_samples = min(num_samples, len(dataset))
        if actual_samples == 0:
            print("\n  ✗ ERROR: No samples in dataset!")
            results = {'error': 'No samples', 'passed': False}
            self.results['landscape_smoothness'] = results
            return results
        
        with torch.no_grad():
            for k in range(self.timesteps):
                sensitivities = []
                
                for i in range(actual_samples):
                    sample = dataset[i]
                    inp = sample[0].unsqueeze(0).to(self.device)
                    target = sample[1].unsqueeze(0).to(self.device)
                    
                    t = torch.tensor([k], device=self.device, dtype=torch.float)
                    
                    # Base energy at target
                    E_base = self.ebm(inp, target, t).item()
                    
                    # Energy changes with perturbations
                    for scale in perturbation_scales:
                        perturbation = torch.randn_like(target) * scale
                        perturbed = target + perturbation
                        E_perturbed = self.ebm(inp, perturbed, t).item()
                        
                        # Sensitivity = |ΔE| / perturbation_norm
                        delta_E = abs(E_perturbed - E_base)
                        pert_norm = perturbation.norm().item()
                        if pert_norm > 1e-8:
                            sensitivities.append(delta_E / pert_norm)
                
                if sensitivities:
                    timestep_sensitivities[k] = {
                        'mean': float(np.mean(sensitivities)),
                        'std': float(np.std(sensitivities)),
                        'min': float(np.min(sensitivities)),
                        'max': float(np.max(sensitivities))
                    }
                else:
                    timestep_sensitivities[k] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                    }
        
        # For correct IRED behavior: sensitivity should DECREASE as k increases
        # (low k = final/sharp = HIGH sensitivity, high k = initial/smooth = LOW sensitivity)
        k_values = list(range(self.timesteps))
        mean_sensitivities = [timestep_sensitivities[k]['mean'] for k in k_values]
        
        # Compute correlation between k and sensitivity
        # Expect NEGATIVE correlation (sensitivity decreases as k increases)
        if len(set(mean_sensitivities)) > 1:  # Need variance for correlation
            correlation = float(np.corrcoef(k_values, mean_sensitivities)[0, 1])
        else:
            correlation = 0.0
        
        # In IRED diffusion: t=0 is FINAL (sharp), t=T-1 is INITIAL (smooth)
        # So we expect LOW k to be SHARPER (higher sensitivity) than HIGH k
        # "early" in inference = HIGH k values, "late" in inference = LOW k values
        third = max(1, self.timesteps // 3)
        # High k (first entries) should be SMOOTH (low sensitivity) - this is "early" in inference
        high_k_sensitivity = float(np.mean(mean_sensitivities[-third:]))  # k near T-1
        # Low k (last entries) should be SHARP (high sensitivity) - this is "late" in inference  
        low_k_sensitivity = float(np.mean(mean_sensitivities[:third]))  # k near 0
        
        # Ratio should be > 1 if low k is sharper than high k (correct IRED behavior)
        sharpening_ratio = low_k_sensitivity / (high_k_sensitivity + 1e-8)

        results = {
            'timestep_sensitivities': timestep_sensitivities,
            'correlation_k_sensitivity': correlation,
            'high_k_sensitivity': high_k_sensitivity,
            'low_k_sensitivity': low_k_sensitivity,
            'sharpening_ratio': sharpening_ratio,
            'passed': sharpening_ratio > 1.2  # Low k (final) should be sharper than high k (initial)
        }
        
        self.results['landscape_smoothness'] = results
        
        # Print results
        print(f"\n  Sensitivity by timestep k:")
        for k in [0, self.timesteps//2, self.timesteps-1]:
            stats = timestep_sensitivities[k]
            print(f"    k={k}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        
        print(f"\n  IRED semantics: t=0 (final/sharp), t=T-1 (initial/smooth)")
        print(f"  High k (k>{self.timesteps - third}, initial/noisy) sensitivity: {high_k_sensitivity:.6f}")
        print(f"  Low k (k<{third}, final/clean) sensitivity: {low_k_sensitivity:.6f}")
        print(f"  Sharpening ratio (low_k/high_k): {sharpening_ratio:.2f}x")
        print(f"  Correlation (k vs sensitivity): {correlation:.4f}")

        if results['passed']:
            print(f"\n  ✓ PASSED: Low k is sharper than high k (ratio={sharpening_ratio:.2f})")
        else:
            print(f"\n  ✗ FAILED: Low k is not sharper than high k (ratio={sharpening_ratio:.2f})")
            print(f"    NOTE: In IRED, we expect t=0 (low k) to be sharp, t=T-1 (high k) to be smooth.")
        
        return results
    
    def test_local_minima(
        self,
        num_samples: int = 50,
        neighborhood_radius: float = 0.1,
        num_neighbors: int = 20
    ) -> Dict:
        """
        Test 3: Verify target solution is a local minimum in the final landscape.
        
        For each sample, compare target energy with random perturbations
        at the final timestep. Target should have lower energy than neighbors.
        """
        print(f"\n{'='*60}")
        print("[Test 3] Local Minima Test (Final Landscape)")
        print(f"{'='*60}")
        
        if self.ebm is None:
            raise RuntimeError("Model not loaded. Train or load checkpoint first.")
        
        self.ebm.eval()
        
        # Create test samples
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        is_local_minimum = []
        energy_ranks = []  # Rank of target among neighbors (0 = lowest)
        target_vs_neighbor_gaps = []
        
        actual_samples = min(num_samples, len(dataset))
        if actual_samples == 0:
            print("\n  ✗ ERROR: No samples in dataset!")
            results = {'error': 'No samples', 'passed': False}
            self.results['local_minima'] = results
            return results
        
        with torch.no_grad():
            for i in range(actual_samples):
                sample = dataset[i]
                inp = sample[0].unsqueeze(0).to(self.device)
                target = sample[1].unsqueeze(0).to(self.device)
                
                # t=0 is the FINAL timestep in IRED (sharpest landscape)
                t = torch.tensor([0], device=self.device, dtype=torch.float)

                # Energy at target
                E_target = self.ebm(inp, target, t).item()                # Energy at random neighbors
                neighbor_energies = []
                for _ in range(num_neighbors):
                    perturbation = torch.randn_like(target) * neighborhood_radius
                    neighbor = target + perturbation
                    E_neighbor = self.ebm(inp, neighbor, t).item()
                    neighbor_energies.append(E_neighbor)
                
                # Check if target is the minimum
                min_neighbor_energy = min(neighbor_energies)
                mean_neighbor_energy = float(np.mean(neighbor_energies))
                
                is_min = E_target <= min_neighbor_energy
                is_local_minimum.append(is_min)
                
                # Rank (how many neighbors have lower energy)
                rank = sum(1 for e in neighbor_energies if e < E_target)
                energy_ranks.append(rank)
                
                # Gap between target and mean neighbor
                target_vs_neighbor_gaps.append(mean_neighbor_energy - E_target)
        
        # Compute statistics
        results = {
            'num_samples': len(is_local_minimum),
            'local_minimum_rate': float(np.mean(is_local_minimum)),
            'mean_rank': float(np.mean(energy_ranks)),
            'median_rank': float(np.median(energy_ranks)),
            'mean_gap_vs_neighbors': float(np.mean(target_vs_neighbor_gaps)),
            'std_gap_vs_neighbors': float(np.std(target_vs_neighbor_gaps)),
            'neighborhood_radius': neighborhood_radius,
            'num_neighbors': num_neighbors,
            'passed': float(np.mean(is_local_minimum)) > 0.5  # At least 50% should be local minima
        }
        
        self.results['local_minima'] = results
        
        # Print results
        print(f"\n  Local minimum rate: {results['local_minimum_rate']*100:.1f}%")
        print(f"  Mean rank among {num_neighbors} neighbors: {results['mean_rank']:.2f}")
        print(f"  Mean gap (neighbors - target): {results['mean_gap_vs_neighbors']:.4f}")
        print(f"  Neighborhood radius: {neighborhood_radius}")
        
        if results['passed']:
            print(f"\n  ✓ PASSED: {results['local_minimum_rate']*100:.1f}% of targets are local minima")
        else:
            print(f"\n  ✗ FAILED: Only {results['local_minimum_rate']*100:.1f}% of targets are local minima")
        
        return results
    
    def test_gradient_optimization(
        self,
        num_samples: int = 20,
        num_steps: int = 50,
        step_size: float = 0.01
    ) -> Dict:
        """
        Test 4: Verify gradient descent from noise converges toward target.
        
        Starting from random initialization, use gradient descent on energy
        and check if we get closer to the target solution.
        """
        print(f"\n{'='*60}")
        print("[Test 4] Gradient Optimization Test")
        print(f"{'='*60}")
        
        if self.ebm is None:
            raise RuntimeError("Model not loaded. Train or load checkpoint first.")
        
        self.ebm.eval()
        
        # Create test samples
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        distance_improvements = []
        energy_decreases = []
        final_distances = []
        
        actual_samples = min(num_samples, len(dataset))
        if actual_samples == 0:
            print("\n  ✗ ERROR: No samples in dataset!")
            results = {'error': 'No samples', 'passed': False}
            self.results['gradient_optimization'] = results
            return results
        
        for i in range(actual_samples):
            sample = dataset[i]
            inp = sample[0].unsqueeze(0).to(self.device)
            target = sample[1].unsqueeze(0).to(self.device)
            
            # t=0 is the FINAL timestep in IRED (sharpest landscape)
            t = torch.tensor([0], device=self.device, dtype=torch.float)
            
            # Start from noise
            current = torch.randn_like(target)
            initial_dist = (current - target).norm().item()
            
            with torch.no_grad():
                initial_energy = self.ebm(inp, current, t).item()
            
            # Gradient descent
            for step in range(num_steps):
                current.requires_grad_(True)
                energy = self.ebm(inp, current, t)
                
                grad = torch.autograd.grad(energy.sum(), current)[0]
                
                with torch.no_grad():
                    current = current - step_size * grad
                    current = current.detach()
            
            with torch.no_grad():
                final_energy = self.ebm(inp, current, t).item()
            
            final_dist = (current - target).norm().item()
            
            # Avoid division by zero
            if initial_dist > 1e-8:
                distance_improvements.append((initial_dist - final_dist) / initial_dist)
            else:
                distance_improvements.append(0.0)
            energy_decreases.append(initial_energy - final_energy)
            final_distances.append(final_dist)
        
        results = {
            'num_samples': len(distance_improvements),
            'mean_distance_improvement': float(np.mean(distance_improvements)),
            'std_distance_improvement': float(np.std(distance_improvements)),
            'mean_energy_decrease': float(np.mean(energy_decreases)),
            'mean_final_distance': float(np.mean(final_distances)),
            'num_steps': num_steps,
            'step_size': step_size,
            'passed': float(np.mean(distance_improvements)) > 0  # Should get closer on average
        }
        
        self.results['gradient_optimization'] = results
        
        print(f"\n  Mean distance improvement: {results['mean_distance_improvement']*100:.1f}%")
        print(f"  Mean energy decrease: {results['mean_energy_decrease']:.4f}")
        print(f"  Mean final distance to target: {results['mean_final_distance']:.4f}")
        
        if results['passed']:
            print(f"\n  ✓ PASSED: Optimization improves distance by {results['mean_distance_improvement']*100:.1f}%")
        else:
            print(f"\n  ✗ FAILED: Optimization does not improve distance")
        
        return results
    
    def generate_visualization(self, save_path: str = None) -> None:
        """Generate visualization plots for all test results."""
        print(f"\n{'='*60}")
        print("[Visualization] Generating plots...")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Energy Separation
        ax1 = axes[0, 0]
        if self.results.get('energy_separation'):
            res = self.results['energy_separation']
            categories = ['Positive\n(Correct)', 'Negative\n(Incorrect)']
            means = [res['pos_energy_mean'], res['neg_energy_mean']]
            stds = [res['pos_energy_std'], res['neg_energy_std']]
            colors = ['green', 'red']
            
            bars = ax1.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax1.set_ylabel('Energy')
            ax1.set_title(f'Energy Separation\n(Gap: {res["energy_gap_mean"]:.2f})')
            ax1.axhline(y=res['pos_energy_mean'], color='green', linestyle='--', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Landscape Smoothness
        ax2 = axes[0, 1]
        if self.results.get('landscape_smoothness'):
            res = self.results['landscape_smoothness']
            k_values = list(res['timestep_sensitivities'].keys())
            sensitivities = [res['timestep_sensitivities'][k]['mean'] for k in k_values]
            errors = [res['timestep_sensitivities'][k]['std'] for k in k_values]
            
            ax2.errorbar(k_values, sensitivities, yerr=errors, marker='o', capsize=3)
            ax2.set_xlabel('Timestep k (low k = final/sharp, high k = initial/smooth)')
            ax2.set_ylabel('Energy Sensitivity')
            ax2.set_title(f'Landscape Sharpness vs Timestep\n(Sharpening ratio: {res["sharpening_ratio"]:.2f}x)')
            ax2.grid(True, alpha=0.3)
            # Add annotation for IRED semantics
            ax2.annotate('← FINAL (t=0)', xy=(0, sensitivities[0]), fontsize=8, ha='left')
            ax2.annotate('INITIAL (t=T-1) →', xy=(max(k_values), sensitivities[-1]), fontsize=8, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Local Minima Distribution
        ax3 = axes[1, 0]
        if self.results.get('local_minima'):
            res = self.results['local_minima']
            categories = ['Local\nMinimum', 'Not\nMinimum']
            values = [res['local_minimum_rate'] * 100, (1 - res['local_minimum_rate']) * 100]
            colors = ['green', 'red']
            
            ax3.bar(categories, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title(f'Local Minima Rate\n({res["local_minimum_rate"]*100:.1f}% success)')
            ax3.set_ylim(0, 100)
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"Test Summary for Rule: {self.rule}\n"
        summary_text += "=" * 40 + "\n\n"
        
        tests = [
            ('Energy Separation', 'energy_separation'),
            ('Landscape Smoothness', 'landscape_smoothness'),
            ('Local Minima', 'local_minima'),
            ('Gradient Optimization', 'gradient_optimization')
        ]
        
        for test_name, key in tests:
            if self.results.get(key):
                status = "✓ PASSED" if self.results[key].get('passed', False) else "✗ FAILED"
                summary_text += f"{test_name}: {status}\n"
            else:
                summary_text += f"{test_name}: Not run\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return combined results."""
        print(f"\n{'='*60}")
        print(f"Running All Tests for Rule: {self.rule}")
        print(f"{'='*60}")
        
        # Run tests
        self.test_energy_separation()
        self.test_landscape_smoothness()
        self.test_local_minima()
        self.test_gradient_optimization()
        
        # Summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        
        all_passed = True
        for test_name, results in self.results.items():
            if results and 'passed' in results:
                status = "✓ PASSED" if results['passed'] else "✗ FAILED"
                print(f"  {test_name}: {status}")
                if not results['passed']:
                    all_passed = False
        
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        return self.results
    
    def save_results(self, save_path: str) -> None:
        """Save test results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results_json = convert_types(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"[Save] Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Algebra EBM Training')
    parser.add_argument('--rule', type=str, default='distribute',
                        choices=['distribute', 'combine', 'isolate', 'divide'],
                        help='Algebraic rule to test')
    parser.add_argument('--train_steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Training learning rate')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load instead of training')
    parser.add_argument('--results_folder', type=str, default=None,
                        help='Folder to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (only valid with --load_checkpoint)')
    parser.add_argument('--save_plots', type=str, default=None,
                        help='Path to save visualization plots')
    parser.add_argument('--num_problems', type=int, default=50000,
                        help='Number of problems to generate for training dataset')
    parser.add_argument('--test_samples', type=int, default=50,
                        help='Number of samples for each test')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: use minimal training (100 steps, 1000 problems)')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        print("[Quick Mode] Using minimal parameters for fast debugging")
        args.train_steps = 100
        args.num_problems = 1000
        args.test_samples = 10
        args.batch_size = 64
    
    # Create tester
    tester = AlgebraEBMTester(
        rule=args.rule,
        verbose=args.verbose
    )
    
    # Train or load model
    if args.load_checkpoint:
        tester.load_checkpoint(args.load_checkpoint)
    else:
        tester.train_model(
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            results_folder=args.results_folder,
            num_problems=args.num_problems
        )
    
    # Run all tests with configurable sample size
    tester.test_energy_separation(num_samples=args.test_samples)
    tester.test_landscape_smoothness(num_samples=min(20, args.test_samples))
    tester.test_local_minima(num_samples=args.test_samples)
    tester.test_gradient_optimization(num_samples=min(20, args.test_samples))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, results in tester.results.items():
        if results and 'passed' in results:
            status = "✓ PASSED" if results['passed'] else "✗ FAILED"
            print(f"  {test_name}: {status}")
            if not results['passed']:
                all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Generate visualization
    if args.save_plots:
        tester.generate_visualization(save_path=args.save_plots)
    else:
        # Default save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./test_results/ebm_test_{args.rule}_{timestamp}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tester.generate_visualization(save_path=save_path)
    
    # Save results
    results_path = f'./test_results/ebm_test_{args.rule}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    tester.save_results(results_path)
    
    return tester.results


if __name__ == '__main__':
    main()
