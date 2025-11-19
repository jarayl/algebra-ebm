#!/usr/bin/env python3
"""
Main Training Script for Algebra EBM Models

Trains individual rule-specific energy-based models for algebraic reasoning.
This script implements Step 8 of Phase 3 from the implementation plan.

Usage:
    python train_algebra.py --rule distribute --batch_size 2048 --timesteps 10
    python train_algebra.py --rule combine --train_steps 50000
    python train_algebra.py --rule isolate --supervise-energy-landscape True
    python train_algebra.py --rule divide --use-innerloop-opt True

Trains 4 separate models (one per rule):
- distribute: Distribution of multiplication over addition  
- combine: Combining like terms
- isolate: Isolating variables through addition/subtraction
- divide: Dividing coefficients

Models are saved to ./results/{rule_name}/
"""

import os
import os.path as osp
import argparse
import torch

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    print('Warning: MKL not initialized.')

# IRED Infrastructure
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from dataset import NoisyWrapper

# Algebra-specific components  
from algebra_dataset import AlgebraDataset
from algebra_models import AlgebraEBM, AlgebraDiffusionWrapper


def str2bool(x):
    """Convert string arguments to boolean values."""
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid boolean value: {}'.format(x))


def parse_args():
    """Parse command-line arguments for algebra training."""
    parser = argparse.ArgumentParser(
        description='Train Rule-Specific Algebra EBM Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument(
        '--rule', 
        type=str,
        required=True,
        choices=['distribute', 'combine', 'isolate', 'divide'],
        help='Algebraic rule to train: distribute, combine, isolate, or divide'
    )
    
    parser.add_argument(
        '--split',
        type=str, 
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to use for training'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='Training batch size (requires ~16GB GPU for default)'
    )
    
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=256,
        help='Validation batch size'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int, 
        default=10,
        help='Number of diffusion timesteps (K landscapes)'
    )
    
    parser.add_argument(
        '--train_steps',
        type=int,
        default=50000,
        help='Total number of training steps'
    )
    
    parser.add_argument(
        '--train_lr',
        type=float,
        default=1e-4,
        help='Training learning rate'
    )
    
    parser.add_argument(
        '--ema_decay',
        type=float,
        default=0.995,
        help='Exponential moving average decay for model weights'
    )
    
    parser.add_argument(
        '--gradient_accumulate_every',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--num_problems',
        type=int,
        default=50000,
        help='Number of problems to generate per rule'
    )
    
    parser.add_argument(
        '--d_model',
        type=int,
        default=128,
        help='Model embedding dimension'
    )
    
    # IRED-specific parameters
    parser.add_argument(
        '--supervise-energy-landscape',
        type=str2bool,
        default=True,
        help='Enable contrastive landscape loss'
    )
    
    parser.add_argument(
        '--use-innerloop-opt', 
        type=str2bool,
        default=True,
        help='Enable T-step optimization during training'
    )
    
    # I/O parameters
    parser.add_argument(
        '--results_folder',
        type=str,
        default=None,
        help='Results folder (default: ./results/{rule_name})'
    )
    
    parser.add_argument(
        '--save_and_sample_every',
        type=int,
        default=1000,
        help='Save model checkpoint every N steps'
    )
    
    parser.add_argument(
        '--data_workers',
        type=int,
        default=None,
        help='Number of data loading workers (default: auto)'
    )
    
    # Debugging and evaluation
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run one evaluation before training'
    )
    
    parser.add_argument(
        '--load_milestone',
        type=str,
        default=None,
        help='Load model from checkpoint path'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print(f"Starting algebra EBM training for rule: {args.rule}")
    
    # Validate GPU memory for batch size
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if args.batch_size >= 2048 and gpu_memory_gb < 15:
            print(f"Warning: Batch size {args.batch_size} may cause OOM on {gpu_memory_gb:.1f}GB GPU")
            print("Consider reducing batch size with --batch_size")
    
    # Set up results directory
    if args.results_folder is None:
        args.results_folder = osp.join('results', args.rule)
    
    try:
        os.makedirs(args.results_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating results directory: {e}")
        return
    
    print(f"Results will be saved to: {args.results_folder}")
    
    # Create algebra dataset for specified rule
    print(f"Creating dataset for rule '{args.rule}' with {args.num_problems} problems...")
    try:
        dataset = AlgebraDataset(
            rule=args.rule,
            split=args.split,
            num_problems=args.num_problems,
            d_model=args.d_model
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Check that algebra_dataset.py and dependencies are properly installed")
        return
    
    # Wrap with noise corruption for IRED training
    print(f"Wrapping dataset with noise corruption (timesteps={args.timesteps})...")
    try:
        noisy_dataset = NoisyWrapper(dataset, timesteps=args.timesteps)
    except Exception as e:
        print(f"Error creating noisy wrapper: {e}")
        return
    
    print(f"Dataset ready: inp_dim={dataset.inp_dim}, out_dim={dataset.out_dim}")
    
    # Validate dataset dimensions
    if dataset.inp_dim != args.d_model or dataset.out_dim != args.d_model:
        print(f"Warning: Dataset dimensions ({dataset.inp_dim}, {dataset.out_dim}) != d_model ({args.d_model})")
    
    # Create algebra energy model
    print(f"Initializing AlgebraEBM for rule '{args.rule}'...")
    try:
        ebm = AlgebraEBM(
            inp_dim=dataset.inp_dim,
            out_dim=dataset.out_dim,
            rule_name=args.rule
        )
    except Exception as e:
        print(f"Error creating AlgebraEBM: {e}")
        print("Check that algebra_models.py is properly implemented")
        return
    
    # Wrap for diffusion training
    print("Wrapping EBM with diffusion wrapper...")
    try:
        model = AlgebraDiffusionWrapper(ebm)
    except Exception as e:
        print(f"Error creating diffusion wrapper: {e}")
        return
    
    # Create diffusion model with IRED configuration
    print("Setting up GaussianDiffusion1D...")
    try:
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=args.d_model,
            objective='pred_noise',
            timesteps=args.timesteps,
            sampling_timesteps=args.timesteps,
            supervise_energy_landscape=args.supervise_energy_landscape,
            use_innerloop_opt=args.use_innerloop_opt,
            show_inference_tqdm=False,
            continuous=True  # For continuous algebraic embeddings
        )
    except Exception as e:
        print(f"Error creating GaussianDiffusion1D: {e}")
        print("Check diffusion_lib installation and parameter compatibility")
        return
    
    # Create trainer
    print("Setting up Trainer1D...")
    try:
        trainer = Trainer1D(
            diffusion,
            noisy_dataset,
            train_batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            train_lr=args.train_lr,
            train_num_steps=args.train_steps,
            gradient_accumulate_every=args.gradient_accumulate_every,
            ema_decay=args.ema_decay,
            data_workers=args.data_workers,
            amp=False,  # No mixed precision for algebra tasks
            metric='mse',  # Use MSE for continuous embeddings
            results_folder=args.results_folder,
            save_and_sample_every=args.save_and_sample_every,
            evaluate_first=args.evaluate
        )
    except Exception as e:
        print(f"Error creating Trainer1D: {e}")
        print("Check trainer configuration and dependencies")
        return
    
    # Load checkpoint if specified
    if args.load_milestone is not None:
        print(f"Loading checkpoint from: {args.load_milestone}")
        try:
            trainer.load(args.load_milestone)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Check that {args.load_milestone} exists and is a valid checkpoint")
            return
    
    # Add minimal progress logging
    def log_progress(step, loss, metrics=None):
        if step % 100 == 0 or step in [500, 1000, 1500, 2000]:
            msg = f"Step {step:4d}: Loss={loss:8.4f}"
            if metrics:
                for key, value in metrics.items():
                    if key != 'loss':
                        msg += f", {key}={value:6.3f}"
            print(msg)
    
    # Check if trainer has callback support
    if hasattr(trainer, 'set_progress_callback'):
        trainer.set_progress_callback(log_progress)
    elif hasattr(trainer, 'add_callback'):
        trainer.add_callback('progress', log_progress)
    
    # Start training
    print(f"Starting training for {args.train_steps} steps...")
    print("=" * 60)
    try:
        trainer.train()
        print("Training completed successfully!")
        print(f"Model saved to: {args.results_folder}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Partial training results may be available in: {args.results_folder}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        print(f"Check logs in: {args.results_folder}")
        print("Verify that dependencies are properly installed and GPU has sufficient memory")
        return


if __name__ == "__main__":
    main()