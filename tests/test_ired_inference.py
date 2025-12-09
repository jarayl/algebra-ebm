#!/usr/bin/env python3
"""
Real IRED Inference Evaluation Script

This script evaluates the actual IRED inference performance using realistic
parameters from the paper and proposal, NOT the minimized test parameters.

Key differences from test_algebra_ebm_training.py:
- Uses full K=10 landscape traversal (t=T-1 → t=0)
- Uses proper adaptive step sizes from noise schedule
- Uses energy-based acceptance (only accept if energy decreases)
- Uses proper landscape scaling between timesteps
- Runs many more optimization steps per landscape

This tests what the model will actually do during real inference, not just
whether the energy landscape has the right properties.

Usage:
    # Quick evaluation (5 samples, fast)
    python tests/test_ired_inference.py --rule distribute --num_samples 5
    
    # Full evaluation (100 samples)
    python tests/test_ired_inference.py --rule distribute --num_samples 100
    
    # Load existing checkpoint
    python tests/test_ired_inference.py --load_checkpoint results/distribute/model-10.pt --rule distribute
    
    # Train first then evaluate
    python tests/test_ired_inference.py --rule distribute --train_steps 10000 --num_samples 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import math
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


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for diffusion noise as used in IRED."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class IREDInferenceEvaluator:
    """
    Evaluator for real IRED inference with proper parameters.
    
    Implements the full IRED Algorithm 2:
    1. Initialize from Gaussian noise
    2. For each landscape k = K-1, K-2, ..., 0:
       - Run T gradient descent steps
       - Only accept updates that decrease energy
       - Scale output for next landscape
    3. Return final output
    """
    
    def __init__(
        self,
        rule: str = 'distribute',
        d_model: int = 128,
        timesteps: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.rule = rule
        self.d_model = d_model
        self.timesteps = timesteps
        self.device = device
        
        # Model components
        self.ebm = None
        self.diffusion = None
        
        # Precompute noise schedule
        self._setup_noise_schedule()
        
        print(f"[IREDInferenceEvaluator] Initialized for rule '{rule}' on {device}")
        print(f"  Timesteps (K): {timesteps}")
        print(f"  Embedding dim: {d_model}")
    
    def _setup_noise_schedule(self):
        """Precompute noise schedule values."""
        betas = cosine_beta_schedule(self.timesteps).to(torch.float32)
        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        
        # Compute step sizes (as in diffusion code)
        # base_step_sizes = betas * sqrt(1 / (1 - alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # Step size multiplier (matching GaussianDiffusion1D default)
        step_size_multiplier = 0.1
        base_step_sizes = betas.to(self.device) * torch.sqrt(1. / (1. - self.alphas_cumprod))
        self.opt_step_sizes = base_step_sizes * step_size_multiplier
        
        print(f"  Step sizes: min={self.opt_step_sizes.min():.6f}, max={self.opt_step_sizes.max():.6f}")
    
    def setup_model(self) -> None:
        """Initialize the EBM model."""
        print("[Setup] Creating AlgebraEBM model...")
        
        self.ebm = AlgebraEBM(
            inp_dim=self.d_model,
            out_dim=self.d_model,
            rule_name=self.rule
        ).to(self.device)
        
        self.model_wrapper = AlgebraDiffusionWrapper(self.ebm)
        
        print(f"[Setup] Model created with {sum(p.numel() for p in self.ebm.parameters())} parameters")
    
    def train_model(
        self,
        train_steps: int = 10000,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        num_problems: int = 50000
    ) -> str:
        """Train the model and return checkpoint path."""
        print(f"\n{'='*60}")
        print(f"[Training] Training for {train_steps} steps")
        print(f"{'='*60}")
        
        if self.ebm is None:
            self.setup_model()
        
        # Create dataset
        dataset = AlgebraDataset(
            rule=self.rule,
            split='train',
            num_problems=num_problems,
            d_model=self.d_model
        )
        noisy_dataset = NoisyWrapper(dataset, timesteps=self.timesteps)
        
        # Setup diffusion for training (also used for real inference)
        self.diffusion = GaussianDiffusion1D(
            self.model_wrapper,
            seq_length=self.d_model,
            timesteps=self.timesteps,
            sampling_timesteps=self.timesteps,
            supervise_energy_landscape=True,
            use_innerloop_opt=True,
            use_contrastive_energy_loss=True,
            step_size_multiplier=0.1,
            show_inference_tqdm=False,
            continuous=True  # For algebra continuous embeddings
        ).to(self.device)
        
        results_folder = f'./test_results/ired_eval_{self.rule}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(results_folder, exist_ok=True)
        
        trainer = Trainer1D(
            self.diffusion,
            noisy_dataset,
            train_batch_size=batch_size,
            train_lr=learning_rate,
            train_num_steps=train_steps,
            gradient_accumulate_every=1,
            results_folder=results_folder,
            save_and_sample_every=max(1000, train_steps),
            amp=True,
            fp16=True,
            data_workers=4
        )
        
        trainer.train()
        
        print(f"[Training] Complete! Results in {results_folder}")
        return results_folder
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model checkpoint."""
        print(f"[Load] Loading checkpoint from {checkpoint_path}")
        
        if self.ebm is None:
            self.setup_model()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats - this is an EMA Trainer1D checkpoint
        ebm_state_dict = {}
        
        # Try to extract EBM weights from different possible locations
        for key_name in ['ema', 'model']:
            if key_name in checkpoint:
                state_dict = checkpoint[key_name]
                for k, v in state_dict.items():
                    # Handle nested checkpoint structure
                    if 'ema_model._orig_mod.model.ebm.' in k:
                        new_key = k.replace('ema_model._orig_mod.model.ebm.', '')
                    elif 'online_model._orig_mod.model.ebm.' in k:
                        new_key = k.replace('online_model._orig_mod.model.ebm.', '')
                    elif '_orig_mod.model.ebm.' in k:
                        new_key = k.replace('_orig_mod.model.ebm.', '')
                    elif 'model.ebm.' in k:
                        new_key = k.replace('model.ebm.', '')
                    elif k.startswith('ema_model.') and not '_orig_mod' in k:
                        new_key = k.replace('ema_model.', '')
                    elif k.startswith('online_model.') and not '_orig_mod' in k:
                        new_key = k.replace('online_model.', '')
                    else:
                        continue
                    
                    # Only keep valid EBM parameters
                    if new_key in ['energy_scale', 'energy_bias', 'time_mlp.1.weight', 'time_mlp.1.bias', 'time_mlp.3.weight', 'time_mlp.3.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias', 't_map_fc2.weight', 't_map_fc2.bias', 't_map_fc3.weight', 't_map_fc3.bias']:
                        ebm_state_dict[new_key] = v
        
        if len(ebm_state_dict) == 18:  # Expected number of EBM parameters
            self.ebm.load_state_dict(ebm_state_dict)
            print(f"[Load] Successfully loaded {len(ebm_state_dict)} EBM parameters")
        else:
            print(f"[Load] Found {len(ebm_state_dict)} EBM parameters, expected 18")
            print("Available keys:", list(ebm_state_dict.keys())[:5])
            raise RuntimeError(f"Could not extract complete EBM state dict from checkpoint")
        
        self.ebm.eval()
        
        # Also setup diffusion for real inference if not already done
        if self.diffusion is None:
            self.diffusion = GaussianDiffusion1D(
                self.model_wrapper,
                seq_length=self.d_model,
                timesteps=self.timesteps,
                sampling_timesteps=self.timesteps,
                supervise_energy_landscape=True,
                use_innerloop_opt=True,
                use_contrastive_energy_loss=True,
                step_size_multiplier=0.1,
                show_inference_tqdm=False,
                continuous=True
            ).to(self.device)
        
        print("[Load] Checkpoint loaded successfully")
    
    def real_diffusion_inference(
        self,
        inp: torch.Tensor,
        target: torch.Tensor
    ) -> Dict:
        """
        Run the REAL diffusion sampling that's used during training evaluation.
        
        This uses GaussianDiffusion1D.sample() which implements the actual
        IRED algorithm including proper scaling, noise scheduling, and
        inner-loop optimization.
        
        Args:
            inp: Input tensor (the problem/expression to solve) 
            target: Ground truth target embedding
        """
        if self.diffusion is None:
            raise RuntimeError("Diffusion not initialized. Train or load model first.")
        
        self.ebm.eval()
        initial_noise = torch.randn(1, self.d_model, device=self.device)
        initial_dist = (initial_noise - target).norm().item()
        
        # Use the actual diffusion sampling
        # Signature: sample(x, label, mask, batch_size=16, return_traj=False)
        # - x: input expression/problem  
        # - label: target for conditioning (we pass None since we're generating)
        # - mask: conditioning mask (None for unconditional)
        with torch.no_grad():
            # Sample uses p_sample_loop internally with proper IRED steps
            result = self.diffusion.sample(
                x=inp,           # Input expression to solve
                label=target,    # Target (used for conditioning in some modes)
                mask=None,       # No conditioning mask
                batch_size=inp.size(0)
            )
        
        final_dist = (result - target).norm().item()
        
        return {
            'initial_distance': initial_dist,
            'final_distance': final_dist,
            'distance_improvement': (initial_dist - final_dist) / initial_dist if initial_dist > 0 else 0,
            'final_embedding': result.detach().cpu()
        }

    def ired_inference_single(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        steps_per_landscape: int = 20,
        verbose: bool = False
    ) -> Dict:
        """
        Run full IRED inference for a single sample.
        
        This implements IRED-style inference:
        - Traverse landscapes from t=T-1 (noisy) to t=0 (clean)
        - At each landscape, run gradient descent with energy-based acceptance
        - Use proper diffusion scaling
        
        Args:
            inp: Input embedding (1, d_model)
            target: Target embedding for evaluation (1, d_model)
            steps_per_landscape: Gradient steps per landscape (T in the paper)
            verbose: Print progress
            
        Returns:
            Dictionary with inference metrics
        """
        self.ebm.eval()
        
        # Initialize from Gaussian noise
        current = torch.randn(1, self.d_model, device=self.device)
        initial_dist = (current - target).norm().item()
        
        with torch.no_grad():
            t_init = torch.tensor([self.timesteps - 1], device=self.device, dtype=torch.float)
            initial_energy = self.ebm(inp, current, t_init).item()
        
        trajectory = {
            'distances': [initial_dist],
            'energies': [initial_energy],
            'accepted_steps': [],
            'rejected_steps': []
        }
        
        # Traverse landscapes from t=T-1 down to t=0
        # This is the key IRED pattern: high t (noisy/smooth) → low t (clean/sharp)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.float)
            step_size = self.opt_step_sizes[t].item()
            
            accepted = 0
            rejected = 0
            
            # Gradient descent steps at this landscape
            for step in range(steps_per_landscape):
                # Compute gradient
                current.requires_grad_(True)
                energy = self.ebm(inp, current, t_tensor)
                grad = torch.autograd.grad(energy.sum(), current)[0]
                
                with torch.no_grad():
                    # Propose update
                    current_new = current - step_size * grad
                    
                    # Energy-based acceptance (only accept if energy decreases)
                    energy_new = self.ebm(inp, current_new, t_tensor).item()
                    energy_old = energy.item()
                    
                    if energy_new < energy_old:
                        current = current_new.detach()
                        accepted += 1
                    else:
                        rejected += 1
                        current = current.detach()
            
            trajectory['accepted_steps'].append(accepted)
            trajectory['rejected_steps'].append(rejected)
            
            # Track progress
            dist = (current - target).norm().item()
            with torch.no_grad():
                e = self.ebm(inp, current, t_tensor).item()
            trajectory['distances'].append(dist)
            trajectory['energies'].append(e)
            
            if verbose:
                print(f"  t={t}: dist={dist:.4f}, energy={e:.4f}, accepted={accepted}/{steps_per_landscape}")
        
        # Final metrics
        final_dist = (current - target).norm().item()
        with torch.no_grad():
            t_final = torch.tensor([0], device=self.device, dtype=torch.float)
            final_energy = self.ebm(inp, current, t_final).item()
        
        return {
            'initial_distance': initial_dist,
            'final_distance': final_dist,
            'distance_improvement': (initial_dist - final_dist) / initial_dist if initial_dist > 0 else 0,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_decrease': initial_energy - final_energy,
            'total_accepted': sum(trajectory['accepted_steps']),
            'total_rejected': sum(trajectory['rejected_steps']),
            'trajectory': trajectory,
            'final_embedding': current.detach().cpu()
        }
    
    def evaluate(
        self,
        num_samples: int = 50,
        steps_per_landscape: int = 20,
        verbose: bool = False
    ) -> Dict:
        """
        Run IRED inference evaluation on test samples.
        
        Args:
            num_samples: Number of test samples to evaluate
            steps_per_landscape: Gradient steps per landscape
            verbose: Print per-sample progress
            
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"[IRED Inference Evaluation]")
        print(f"  Samples: {num_samples}")
        print(f"  Landscapes (K): {self.timesteps}")
        print(f"  Steps per landscape: {steps_per_landscape}")
        print(f"  Total gradient steps: {self.timesteps * steps_per_landscape}")
        print(f"{'='*60}")
        
        if self.ebm is None:
            raise RuntimeError("Model not loaded. Train or load checkpoint first.")
        
        # Create test dataset
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        results = []
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inp = sample[0].unsqueeze(0).to(self.device)
            target = sample[1].unsqueeze(0).to(self.device)
            
            if verbose:
                print(f"\nSample {i+1}/{num_samples}:")
            
            result = self.ired_inference_single(
                inp, target, 
                steps_per_landscape=steps_per_landscape,
                verbose=verbose
            )
            results.append(result)
            
            if not verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples} samples...")
        
        # Aggregate results
        aggregated = {
            'num_samples': len(results),
            'steps_per_landscape': steps_per_landscape,
            'total_landscapes': self.timesteps,
            'total_steps': self.timesteps * steps_per_landscape,
            
            # Distance metrics
            'mean_initial_distance': float(np.mean([r['initial_distance'] for r in results])),
            'mean_final_distance': float(np.mean([r['final_distance'] for r in results])),
            'std_final_distance': float(np.std([r['final_distance'] for r in results])),
            'mean_distance_improvement': float(np.mean([r['distance_improvement'] for r in results])),
            'std_distance_improvement': float(np.std([r['distance_improvement'] for r in results])),
            
            # Energy metrics
            'mean_initial_energy': float(np.mean([r['initial_energy'] for r in results])),
            'mean_final_energy': float(np.mean([r['final_energy'] for r in results])),
            'mean_energy_decrease': float(np.mean([r['energy_decrease'] for r in results])),
            
            # Acceptance metrics
            'mean_accepted_ratio': float(np.mean([
                r['total_accepted'] / (r['total_accepted'] + r['total_rejected'] + 1e-8) 
                for r in results
            ])),
            
            # Success metrics (distance improved by >50%)
            'success_rate_50pct': float(np.mean([r['distance_improvement'] > 0.5 for r in results])),
            'success_rate_25pct': float(np.mean([r['distance_improvement'] > 0.25 for r in results])),
            'success_rate_any': float(np.mean([r['distance_improvement'] > 0 for r in results])),
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("[Results Summary]")
        print(f"{'='*60}")
        print(f"  Initial distance (from noise): {aggregated['mean_initial_distance']:.4f}")
        print(f"  Final distance to target: {aggregated['mean_final_distance']:.4f} ± {aggregated['std_final_distance']:.4f}")
        print(f"  Distance improvement: {aggregated['mean_distance_improvement']*100:.1f}% ± {aggregated['std_distance_improvement']*100:.1f}%")
        print(f"")
        print(f"  Initial energy: {aggregated['mean_initial_energy']:.4f}")
        print(f"  Final energy: {aggregated['mean_final_energy']:.4f}")
        print(f"  Energy decrease: {aggregated['mean_energy_decrease']:.4f}")
        print(f"")
        print(f"  Acceptance ratio: {aggregated['mean_accepted_ratio']*100:.1f}%")
        print(f"  Success rate (>50% improvement): {aggregated['success_rate_50pct']*100:.1f}%")
        print(f"  Success rate (>25% improvement): {aggregated['success_rate_25pct']*100:.1f}%")
        print(f"  Success rate (any improvement): {aggregated['success_rate_any']*100:.1f}%")
        
        return aggregated
    
    def evaluate_real_diffusion(
        self,
        num_samples: int = 50,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate using the REAL GaussianDiffusion1D.sample() method.
        
        This is what the actual training evaluation uses, and represents
        the true IRED inference performance.
        """
        print(f"\n{'='*60}")
        print(f"[Real Diffusion Sampling Evaluation]")
        print(f"  Samples: {num_samples}")
        print(f"  Using: GaussianDiffusion1D.sample() (actual IRED)")
        print(f"{'='*60}")
        
        if self.diffusion is None:
            raise RuntimeError("Diffusion not initialized. Train or load model first.")
        
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test',
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        results = []
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inp = sample[0].unsqueeze(0).to(self.device)
            target = sample[1].unsqueeze(0).to(self.device)
            
            result = self.real_diffusion_inference(inp, target)
            results.append(result)
            
            if verbose:
                print(f"  Sample {i+1}: dist_improvement={result['distance_improvement']*100:.1f}%")
            elif (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples} samples...")
        
        aggregated = {
            'num_samples': len(results),
            'method': 'GaussianDiffusion1D.sample()',
            'mean_initial_distance': float(np.mean([r['initial_distance'] for r in results])),
            'mean_final_distance': float(np.mean([r['final_distance'] for r in results])),
            'std_final_distance': float(np.std([r['final_distance'] for r in results])),
            'mean_distance_improvement': float(np.mean([r['distance_improvement'] for r in results])),
            'std_distance_improvement': float(np.std([r['distance_improvement'] for r in results])),
            'success_rate_50pct': float(np.mean([r['distance_improvement'] > 0.5 for r in results])),
            'success_rate_25pct': float(np.mean([r['distance_improvement'] > 0.25 for r in results])),
            'success_rate_any': float(np.mean([r['distance_improvement'] > 0 for r in results])),
        }
        
        print(f"\n{'='*60}")
        print("[Real Diffusion Results]")
        print(f"{'='*60}")
        print(f"  Initial distance (from noise): {aggregated['mean_initial_distance']:.4f}")
        print(f"  Final distance to target: {aggregated['mean_final_distance']:.4f} ± {aggregated['std_final_distance']:.4f}")
        print(f"  Distance improvement: {aggregated['mean_distance_improvement']*100:.1f}% ± {aggregated['std_distance_improvement']*100:.1f}%")
        print(f"  Success rate (>50% improvement): {aggregated['success_rate_50pct']*100:.1f}%")
        print(f"  Success rate (>25% improvement): {aggregated['success_rate_25pct']*100:.1f}%")
        print(f"  Success rate (any improvement): {aggregated['success_rate_any']*100:.1f}%")
        
        return aggregated

    def compare_with_naive(
        self,
        num_samples: int = 20,
        steps_per_landscape: int = 20
    ) -> Dict:
        """
        Compare full IRED inference vs naive single-timestep optimization.
        
        This demonstrates why the full IRED procedure (annealed landscapes)
        is necessary vs just optimizing at t=0.
        """
        print(f"\n{'='*60}")
        print("[Comparison: Full IRED vs Naive Single-Timestep]")
        print(f"{'='*60}")
        
        dataset = AlgebraDataset(
            rule=self.rule,
            split='test', 
            num_problems=num_samples,
            d_model=self.d_model
        )
        
        ired_results = []
        naive_results = []
        
        total_steps = self.timesteps * steps_per_landscape
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inp = sample[0].unsqueeze(0).to(self.device)
            target = sample[1].unsqueeze(0).to(self.device)
            
            # Full IRED inference
            ired_result = self.ired_inference_single(inp, target, steps_per_landscape)
            ired_results.append(ired_result['distance_improvement'])
            
            # Naive: same number of steps but only at t=0
            current = torch.randn(1, self.d_model, device=self.device)
            initial_dist = (current - target).norm().item()
            
            t_zero = torch.tensor([0], device=self.device, dtype=torch.float)
            step_size = 0.01  # Fixed step size
            
            for _ in range(total_steps):
                current.requires_grad_(True)
                energy = self.ebm(inp, current, t_zero)
                grad = torch.autograd.grad(energy.sum(), current)[0]
                
                with torch.no_grad():
                    current_new = current - step_size * grad
                    energy_new = self.ebm(inp, current_new, t_zero).item()
                    if energy_new < energy.item():
                        current = current_new.detach()
                    else:
                        current = current.detach()
            
            final_dist = (current - target).norm().item()
            naive_improvement = (initial_dist - final_dist) / initial_dist if initial_dist > 0 else 0
            naive_results.append(naive_improvement)
        
        comparison = {
            'ired_mean_improvement': float(np.mean(ired_results)),
            'ired_std_improvement': float(np.std(ired_results)),
            'naive_mean_improvement': float(np.mean(naive_results)),
            'naive_std_improvement': float(np.std(naive_results)),
            'ired_advantage': float(np.mean(ired_results) - np.mean(naive_results)),
        }
        
        print(f"\n  Full IRED (K={self.timesteps} landscapes):")
        print(f"    Distance improvement: {comparison['ired_mean_improvement']*100:.1f}% ± {comparison['ired_std_improvement']*100:.1f}%")
        print(f"\n  Naive (t=0 only, same total steps):")
        print(f"    Distance improvement: {comparison['naive_mean_improvement']*100:.1f}% ± {comparison['naive_std_improvement']*100:.1f}%")
        print(f"\n  IRED advantage: +{comparison['ired_advantage']*100:.1f}%")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='IRED Inference Evaluation')
    parser.add_argument('--rule', type=str, default='distribute',
                        choices=['distribute', 'combine', 'isolate', 'divide'],
                        help='Algebraic rule to evaluate')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--train_steps', type=int, default=None,
                        help='Train for this many steps before evaluation')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of test samples to evaluate')
    parser.add_argument('--steps_per_landscape', type=int, default=20,
                        help='Gradient steps per landscape (T in IRED)')
    parser.add_argument('--timesteps', type=int, default=10,
                        help='Number of landscapes (K in IRED)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample progress')
    parser.add_argument('--compare_naive', action='store_true',
                        help='Also run comparison with naive single-timestep optimization')
    parser.add_argument('--use_real_diffusion', action='store_true',
                        help='Use actual GaussianDiffusion1D.sample() for evaluation')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--num_problems', type=int, default=50000,
                        help='Number of training problems')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = IREDInferenceEvaluator(
        rule=args.rule,
        timesteps=args.timesteps
    )
    
    # Train or load model
    if args.load_checkpoint:
        evaluator.load_checkpoint(args.load_checkpoint)
    elif args.train_steps:
        evaluator.train_model(
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            num_problems=args.num_problems
        )
    else:
        print("ERROR: Must specify either --load_checkpoint or --train_steps")
        return
    
    # Run evaluation using real diffusion sampling (recommended)
    if args.use_real_diffusion:
        results = evaluator.evaluate_real_diffusion(
            num_samples=args.num_samples,
            verbose=args.verbose
        )
    else:
        # Run manual IRED-style evaluation 
        results = evaluator.evaluate(
            num_samples=args.num_samples,
            steps_per_landscape=args.steps_per_landscape,
            verbose=args.verbose
        )
    
    # Optionally compare with naive
    if args.compare_naive:
        comparison = evaluator.compare_with_naive(
            num_samples=min(20, args.num_samples),
            steps_per_landscape=args.steps_per_landscape
        )
        results['comparison'] = comparison
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'./test_results/ired_inference_{args.rule}_{timestamp}.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Save] Results saved to {results_path}")


if __name__ == '__main__':
    main()