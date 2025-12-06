#!/usr/bin/env python3
"""
Main Training Script for Algebra EBM Models

Trains individual rule-specific energy-based models for algebraic reasoning.
This script implements Step 8 of Phase 3 from the implementation plan.

Usage:
    python train_algebra.py --rule distribute --batch_size 2048 --timesteps 10
    python train_algebra.py --rule combine --train_steps 200000
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

# PyTorch dynamo configuration to handle .item() calls in compiled graphs
# This prevents graph breaks from scalar tensor extractions during training
try:
    torch._dynamo.config.capture_scalar_outputs = True
    print("Dynamo scalar output capture enabled for optimization")
except AttributeError:
    print("Warning: PyTorch dynamo not available, .item() calls may cause graph breaks")

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


def parse_range(range_str):
    """Parse comma-separated range string to [min, max] list."""
    try:
        parts = range_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Range must have exactly 2 values, got {len(parts)}")
        return [int(parts[0].strip()), int(parts[1].strip())]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid range format '{range_str}': {e}")


def parse_distribution(dist_str):
    """Parse comma-separated distribution string to probabilities list."""
    try:
        parts = dist_str.split(',')
        probs = [float(p.strip()) for p in parts]
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Distribution probabilities must sum to 1.0, got {total}")
        return probs
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid distribution format '{dist_str}': {e}")


def build_variability_config(args):
    """Build variability configuration from command line arguments."""
    config = {
        'enable_stratified_sampling': args.enable_stratified_sampling,
        'enable_solution_first': args.enable_solution_first
    }
    
    # Parse stratified sampling configuration
    if args.enable_stratified_sampling:
        try:
            basic_range = parse_range(args.stratified_basic_range)
            extended_range = parse_range(args.stratified_extended_range)
            challenge_range = parse_range(args.stratified_challenge_range)
            distribution_probs = parse_distribution(args.stratified_distribution)
            
            if len(distribution_probs) != 3:
                raise ValueError("Stratified distribution must have exactly 3 probabilities")
            
            config['stratified_ranges'] = {
                'basic': basic_range,
                'extended': extended_range,
                'challenge': challenge_range
            }
            config['stratified_distribution'] = {
                'basic': distribution_probs[0],
                'extended': distribution_probs[1], 
                'challenge': distribution_probs[2]
            }
            
        except ValueError as e:
            print(f"Error in stratified sampling configuration: {e}")
            return None
    
    # Parse solution-first configuration
    if args.enable_solution_first:
        try:
            small_range = parse_range(args.solution_small_range)
            medium_range = parse_range(args.solution_medium_range)
            large_range = parse_range(args.solution_large_range)
            solution_probs = parse_distribution(args.solution_range_distribution)
            
            if len(solution_probs) != 3:
                raise ValueError("Solution range distribution must have exactly 3 probabilities")
            
            config['target_solution_ranges'] = {
                'small': small_range,
                'medium': medium_range,
                'large': large_range
            }
            config['solution_range_distribution'] = {
                'small': solution_probs[0],
                'medium': solution_probs[1],
                'large': solution_probs[2]
            }
            
        except ValueError as e:
            print(f"Error in solution-first configuration: {e}")
            return None
    
    return config


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
        default=5000,
        help='Total number of training steps (1M recommended for sharp energy landscapes, 5K for quick testing, 1.3M for full IRED baseline)'
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
    
    parser.add_argument(
        '--step_size_multiplier',
        type=float,
        default=0.1,
        help='Step size scaling factor for optimization (smaller = more stable for algebra)'
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
    
    # Variability enhancement parameters
    parser.add_argument(
        '--enable_stratified_sampling',
        type=str2bool,
        default=False,
        help='Enable stratified coefficient sampling for enhanced variability'
    )
    
    parser.add_argument(
        '--stratified_basic_range',
        type=str,
        default='-5,5',
        help='Basic coefficient range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--stratified_extended_range', 
        type=str,
        default='-20,20',
        help='Extended coefficient range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--stratified_challenge_range',
        type=str,
        default='-50,50', 
        help='Challenge coefficient range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--stratified_distribution',
        type=str,
        default='0.4,0.4,0.2',
        help='Stratified distribution probabilities (comma-separated: basic,extended,challenge)'
    )
    
    parser.add_argument(
        '--enable_solution_first',
        type=str2bool,
        default=False,
        help='Enable solution-first equation generation for systematic coverage'
    )
    
    parser.add_argument(
        '--solution_small_range',
        type=str,
        default='-10,10',
        help='Small solution range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--solution_medium_range',
        type=str,
        default='-25,25', 
        help='Medium solution range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--solution_large_range',
        type=str,
        default='-50,50',
        help='Large solution range (comma-separated: min,max)'
    )
    
    parser.add_argument(
        '--solution_range_distribution',
        type=str,
        default='0.5,0.35,0.15',
        help='Solution range distribution probabilities (comma-separated: small,medium,large)'
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
    
    parser.add_argument(
        '--enable-semantic-corruption',
        type=str2bool,
        default=False,
        help='Enable semantic corruption strategy for richer negative sampling'
    )
    
    parser.add_argument(
        '--corruption-strategy-probs',
        type=str,
        default=None,
        help='JSON string specifying corruption strategy probabilities (e.g., \'{"heavy_gaussian": 0.3, "extreme_gaussian": 0.3, "pure_random": 0.2, "semantic": 0.2}\')'
    )
    
    # Performance optimization parameters
    parser.add_argument(
        '--amp',
        type=str2bool,
        default=True,
        help='Enable Automatic Mixed Precision (AMP) for ~2x speedup'
    )
    
    parser.add_argument(
        '--fp16',
        type=str2bool,
        default=True,
        help='Use FP16 mixed precision training'
    )
    
    parser.add_argument(
        '--pin_memory',
        type=str2bool,
        default=True,
        help='Pin memory in data loaders for faster GPU transfer'
    )
    
    parser.add_argument(
        '--persistent_workers',
        type=str2bool,
        default=True,
        help='Keep data loader workers persistent between epochs'
    )
    
    parser.add_argument(
        '--compile_model',
        type=str2bool,
        default=True,
        help='Use torch.compile for ~20% additional speedup (PyTorch 2.0+)'
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
    
    # Parse variability configuration
    variability_config = build_variability_config(args)
    if variability_config is None:
        print("Failed to parse variability configuration")
        return
    
    # Parse corruption strategy probabilities if provided
    corruption_strategy_probs = None
    if args.corruption_strategy_probs:
        try:
            import json
            corruption_strategy_probs = json.loads(args.corruption_strategy_probs)
            if not isinstance(corruption_strategy_probs, dict):
                print(f"Error: corruption-strategy-probs must be a JSON object, got {type(corruption_strategy_probs)}")
                return
            print(f"Using custom corruption strategy probabilities: {corruption_strategy_probs}")
        except json.JSONDecodeError as e:
            print(f"Error parsing corruption-strategy-probs JSON: {e}")
            return
        except Exception as e:
            print(f"Error processing corruption-strategy-probs: {e}")
            return
    
    # Log corruption strategy configuration
    if args.enable_semantic_corruption:
        print("Semantic corruption strategy enabled for enhanced negative sampling")
    else:
        print("Using standard noise corruption strategies only")
    
    # Create algebra dataset for specified rule
    print(f"Creating dataset for rule '{args.rule}' with {args.num_problems} problems...")
    if variability_config['enable_stratified_sampling'] or variability_config['enable_solution_first']:
        print("Enhanced variability features enabled:")
        if variability_config['enable_stratified_sampling']:
            print(f"  - Stratified coefficient sampling: {variability_config['stratified_distribution']}")
        if variability_config['enable_solution_first']:
            print(f"  - Solution-first generation: {variability_config['solution_range_distribution']}")
    
    try:
        # Build dataset arguments with variability configuration
        dataset_kwargs = {
            'rule': args.rule,
            'split': args.split,
            'num_problems': args.num_problems,
            'd_model': args.d_model,
            'enable_stratified_sampling': variability_config['enable_stratified_sampling'],
            'enable_solution_first': variability_config['enable_solution_first']
        }
        
        # Add stratified sampling parameters if enabled
        if variability_config['enable_stratified_sampling']:
            dataset_kwargs.update({
                'stratified_ranges': variability_config['stratified_ranges'],
                'stratified_distribution': variability_config['stratified_distribution']
            })
        
        # Add solution-first parameters if enabled
        if variability_config['enable_solution_first']:
            dataset_kwargs.update({
                'target_solution_ranges': variability_config['target_solution_ranges'],
                'solution_range_distribution': variability_config['solution_range_distribution']
            })
        
        dataset = AlgebraDataset(**dataset_kwargs)
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
    
    # Note: torch.compile will be applied to diffusion model after creation
    
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
            step_size_multiplier=args.step_size_multiplier,
            enable_semantic_corruption=args.enable_semantic_corruption,
            corruption_strategy_probs=corruption_strategy_probs,
            show_inference_tqdm=False,
            continuous=True  # For continuous algebraic embeddings
        )
    except Exception as e:
        print(f"Error creating GaussianDiffusion1D: {e}")
        print("Check diffusion_lib installation and parameter compatibility")
        return
    
    # Apply PyTorch compilation to diffusion model (actual hot path)
    if args.compile_model:
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            if hasattr(torch, 'compile'):
                logger.info("Compiling diffusion model with torch.compile (mode='reduce-overhead')...")
                diffusion = torch.compile(diffusion, mode='reduce-overhead')
                logger.info("✓ Diffusion model compiled successfully")
                print("✓ Diffusion model compiled successfully")
            else:
                logger.warning("⚠ torch.compile not available, skipping compilation")
                print("⚠ torch.compile not available, skipping compilation")
                args.compile_model = False
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"⚠ torch.compile failed: {e}")
            logger.warning("Continuing without compilation...")
            print(f"⚠ torch.compile failed: {e}, continuing without compilation")
            args.compile_model = False
        except Exception as e:
            logger.warning(f"⚠ Unexpected error during compilation: {e}")
            print(f"⚠ Model compilation failed: {e}, continuing without compilation")
            args.compile_model = False
    
    # Create trainer with performance optimizations
    print("Setting up Trainer1D with performance optimizations...")
    if args.amp and args.fp16:
        print("✓ Mixed precision training enabled (FP16)")
    if args.pin_memory:
        print("✓ Pinned memory enabled for faster GPU transfers")
    if args.persistent_workers and args.data_workers and args.data_workers > 0:
        print("✓ Persistent data workers enabled")
    
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
            amp=args.amp,  # Enable mixed precision for major speedup
            fp16=args.fp16,  # Use FP16 mixed precision
            pin_memory=args.pin_memory,  # Faster GPU memory transfers
            persistent_workers=args.persistent_workers,  # Keep workers alive
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
    
    # Enhanced progress logging with convergence monitoring
    loss_history = []
    training_unstable = False
    
    def log_progress(step, loss, metrics=None):
        nonlocal training_unstable
        
        # Track loss and gradient information
        loss_history.append(loss)
        
        # Check for gradient explosion through loss explosion
        if loss > 1000.0 or (len(loss_history) > 1 and loss > 10 * loss_history[-2]):
            print(f"\n⚠️  TRAINING UNSTABLE: Step {step}, Loss exploded to {loss:.2e}")
            print("Consider reducing learning rate or checking dataset quality")
            training_unstable = True
        
        # Check for training stagnation (last 1000 steps)
        elif len(loss_history) >= 1000:
            recent_losses = loss_history[-1000:]
            loss_std = (sum((l - sum(recent_losses)/len(recent_losses))**2 for l in recent_losses) / len(recent_losses))**0.5
            if loss_std < 1e-6 and loss > 0.01:
                print(f"\n📊 Training may have stagnated at step {step}: Loss std={loss_std:.2e}")
                print("Consider adjusting learning rate or optimization parameters")
        
        # Regular progress logging
        if step % 100 == 0 or step in [500, 1000, 1500, 2000] or training_unstable:
            msg = f"Step {step:4d}: Loss={loss:8.4f}"
            if metrics:
                for key, value in metrics.items():
                    if key != 'loss':
                        msg += f", {key}={value:6.3f}"
            
            # Add convergence status indicators
            if len(loss_history) >= 100:
                recent_trend = (loss_history[-1] - loss_history[-100]) / 100
                trend_indicator = "📈" if recent_trend > 0.01 else "📉" if recent_trend < -0.01 else "➡️"
                msg += f" {trend_indicator}"
                
            print(msg)
            
        # Early stopping check for severe instability
        if training_unstable and loss > 10000.0:
            print(f"\n🚨 EARLY STOPPING: Training critically unstable at step {step}")
            print("Loss has exploded beyond recovery threshold")
            if hasattr(trainer, 'stop_training'):
                trainer.stop_training()
            return False  # Signal to stop if callback supports return values
            
        return True
    
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
        
        # Post-training analysis and convergence summary
        print("\n" + "=" * 60)
        print("Training Convergence Analysis")
        print("=" * 60)
        
        if len(loss_history) > 0:
            final_loss = loss_history[-1]
            initial_loss = loss_history[0] if len(loss_history) > 1 else final_loss
            
            # Training stability assessment
            if training_unstable:
                print("🚨 Training Stability: UNSTABLE - Detected loss explosions")
                print("   Recommendation: Reduce learning rate or check dataset quality")
            elif final_loss < 0.01:
                print("✅ Training Stability: EXCELLENT - Low final loss achieved")
            elif final_loss < 0.1:
                print("✅ Training Stability: GOOD - Reasonable final loss")
            else:
                print("⚠️ Training Stability: MARGINAL - High final loss")
                print(f"   Final loss: {final_loss:.4f} - Consider longer training")
            
            # Convergence trend analysis
            if len(loss_history) >= 1000:
                recent_losses = loss_history[-1000:]
                loss_std = (sum((l - sum(recent_losses)/len(recent_losses))**2 for l in recent_losses) / len(recent_losses))**0.5
                if loss_std < 1e-6:
                    print("📊 Convergence Status: CONVERGED - Loss stabilized")
                elif len(loss_history) >= 100:
                    recent_trend = (loss_history[-1] - loss_history[-100]) / 100
                    if recent_trend < -0.001:
                        print("📉 Convergence Status: IMPROVING - Loss still decreasing")
                    elif recent_trend > 0.001:
                        print("📈 Convergence Status: DEGRADING - Loss increasing")
                    else:
                        print("➡️ Convergence Status: STABLE - Minimal change")
                        
            print(f"📈 Loss Progress: {initial_loss:.4f} → {final_loss:.4f} "
                  f"({((final_loss - initial_loss) / initial_loss * 100):+.1f}%)")
                  
        if training_unstable:
            print("\n⚠️ Training completed with instability warnings")
            print("   Consider reviewing training configuration before production use")
        else:
            print("\n✅ Training completed successfully!")
            
        print(f"📁 Model saved to: {args.results_folder}")
        
        # Report dataset variability results if adaptive generation was used
        if variability_config['enable_stratified_sampling'] or variability_config['enable_solution_first']:
            print("\n" + "=" * 60)
            print("Dataset Variability Report")
            print("=" * 60)
            
            try:
                # Get coverage history if available
                coverage_history = dataset.get_coverage_history()
                if coverage_history:
                    print(f"Adaptive generation performed {len(coverage_history)} quality checkpoints")
                    if hasattr(dataset, '_generation_stats'):
                        stats = dataset._generation_stats
                        print(f"Generation statistics:")
                        print(f"  - Total attempts: {stats.get('attempts', 0)}")
                        print(f"  - Successful equations: {stats.get('successes', 0)}")
                        print(f"  - Coverage adjustments: {stats.get('coverage_adjustments', 0)}")
                
                # Get final coverage validation
                coverage_report = dataset.validate_current_coverage()
                if 'overall_passed' in coverage_report:
                    status = "PASSED" if coverage_report['overall_passed'] else "NEEDS IMPROVEMENT"
                    print(f"Final coverage validation: {status}")
                    
                    if coverage_report.get('recommendations'):
                        print("Coverage recommendations:")
                        for rec in coverage_report['recommendations'][:3]:  # Show top 3
                            print(f"  - {rec}")
                            
            except Exception as e:
                print(f"Error generating variability report: {e}")
            
            print("=" * 60)
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