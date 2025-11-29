#!/usr/bin/env python3
"""
Local Model Training Script - Simplified Training Execution

Creates a simple, local environment-friendly script to train algebra EBM models
without requiring SLURM infrastructure. Addresses the root cause identified
in Phase 1 Crisis Assessment: missing model checkpoints.

Usage:
    python train_models_local.py                    # Train all 4 rules
    python train_models_local.py --rule distribute  # Train single rule
    python train_models_local.py --quick-test       # Quick test with minimal parameters
"""

import os
import sys
import argparse
import time
import subprocess
import torch
from pathlib import Path
from typing import List, Dict, Optional

def check_system_requirements() -> Dict[str, bool]:
    """
    Check system requirements for local training.
    
    Returns:
        Dictionary of requirement check results
    """
    print("=== SYSTEM REQUIREMENTS CHECK ===")
    
    requirements = {}
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    requirements['cuda'] = cuda_available
    print(f"CUDA Available: {'✅' if cuda_available else '❌'} {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0
        
        requirements['gpu_memory'] = memory_gb
        print(f"GPU Device: {device_name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  WARNING: GPU memory < 4GB may cause training issues")
            print("   Consider using smaller batch sizes or CPU training")
    else:
        print("⚠️  WARNING: CUDA not available, will use CPU training")
        print("   CPU training will be significantly slower")
        requirements['gpu_memory'] = 0
    
    # Check disk space in results directory
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    try:
        statvfs = os.statvfs(results_dir)
        available_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
        requirements['disk_space'] = available_gb
        print(f"Available Disk Space: {available_gb:.1f} GB")
        
        if available_gb < 2:
            print("⚠️  WARNING: Low disk space may cause training failures")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        requirements['disk_space'] = float('inf')  # Assume sufficient
    
    return requirements


def get_training_parameters(requirements: Dict[str, bool], quick_test: bool = False) -> Dict[str, any]:
    """
    Determine optimal training parameters based on system capabilities.
    
    Args:
        requirements: System requirements check results
        quick_test: Whether to use minimal parameters for testing
        
    Returns:
        Dictionary of training parameters
    """
    if quick_test:
        return {
            'batch_size': 64,
            'train_steps': 100,
            'num_problems': 500,
            'timesteps': 5,
            'save_every': 50
        }
    
    # Determine batch size based on GPU memory
    gpu_memory = requirements.get('gpu_memory', 0)
    if gpu_memory >= 16:
        batch_size = 2048  # Original default
    elif gpu_memory >= 8:
        batch_size = 1024
    elif gpu_memory >= 4:
        batch_size = 512
    else:
        batch_size = 256  # Conservative for CPU or low-memory GPU
    
    return {
        'batch_size': batch_size,
        'train_steps': 10000,  # Reduced from default 50000 for local training
        'num_problems': 10000,  # Reduced from default 50000
        'timesteps': 10,
        'save_every': 500
    }


def run_single_rule_training(rule: str, params: Dict[str, any], results_dir: str) -> bool:
    """
    Train a single rule model using the existing train_algebra.py script.
    
    Args:
        rule: Rule name (distribute, combine, isolate, divide)
        params: Training parameters
        results_dir: Directory to save results
        
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*50}")
    print(f"Training rule: {rule}")
    print(f"{'='*50}")
    
    # Create rule-specific results directory
    rule_dir = Path(results_dir) / rule
    rule_dir.mkdir(parents=True, exist_ok=True)
    
    # Build training command
    cmd = [
        sys.executable, "train_algebra.py",
        "--rule", rule,
        "--batch_size", str(params['batch_size']),
        "--train_steps", str(params['train_steps']),
        "--num_problems", str(params['num_problems']),
        "--timesteps", str(params['timesteps']),
        "--results_folder", str(rule_dir),
        "--save_and_sample_every", str(params['save_every']),
        "--supervise-energy-landscape", "True",
        "--use-innerloop-opt", "True"
    ]
    
    print(f"Training command: {' '.join(cmd)}")
    print(f"Expected output: {rule_dir / 'model.pt'}")
    
    # Run training with progress monitoring
    start_time = time.time()
    
    try:
        # Run training subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully in {duration:.0f}s")
            
            # Verify model file was created
            model_file = rule_dir / "model.pt"
            if model_file.exists():
                model_size_mb = model_file.stat().st_size / (1024**2)
                print(f"   Model saved: {model_file} ({model_size_mb:.1f} MB)")
                return True
            else:
                print(f"❌ Training reported success but no model file found at {model_file}")
                return False
        else:
            print(f"❌ Training failed with exit code {result.returncode}")
            print(f"   Duration: {duration:.0f}s")
            
            # Show error output for debugging
            if result.stderr:
                print(f"   Error output: {result.stderr[:500]}...")  # First 500 chars
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after 2 hours")
        return False
        
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False


def train_all_models(quick_test: bool = False, specific_rule: Optional[str] = None) -> Dict[str, bool]:
    """
    Train all algebra EBM models (or a specific rule).
    
    Args:
        quick_test: Whether to use minimal parameters for testing
        specific_rule: If provided, only train this specific rule
        
    Returns:
        Dictionary mapping rule names to success status
    """
    print("=== LOCAL ALGEBRA EBM TRAINING ===")
    print("Addressing Phase 1 Crisis Assessment finding: missing model checkpoints")
    print()
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Get training parameters
    params = get_training_parameters(requirements, quick_test)
    
    print(f"\n=== TRAINING PARAMETERS ===")
    print(f"Batch size: {params['batch_size']}")
    print(f"Training steps: {params['train_steps']}")
    print(f"Problems per rule: {params['num_problems']}")
    print(f"Timesteps: {params['timesteps']}")
    print(f"Save frequency: every {params['save_every']} steps")
    print()
    
    # Determine which rules to train
    all_rules = ['distribute', 'combine', 'isolate', 'divide']
    rules_to_train = [specific_rule] if specific_rule else all_rules
    
    results_dir = "./results"
    Path(results_dir).mkdir(exist_ok=True)
    
    # Train each rule
    training_results = {}
    successful_rules = []
    failed_rules = []
    
    overall_start_time = time.time()
    
    for i, rule in enumerate(rules_to_train, 1):
        print(f"\nTraining rule {i}/{len(rules_to_train)}: {rule}")
        
        success = run_single_rule_training(rule, params, results_dir)
        training_results[rule] = success
        
        if success:
            successful_rules.append(rule)
        else:
            failed_rules.append(rule)
        
        # Show intermediate progress
        print(f"Progress: {i}/{len(rules_to_train)} rules attempted")
        print(f"Successful: {len(successful_rules)} | Failed: {len(failed_rules)}")
    
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total training time: {total_duration:.0f}s ({total_duration/60:.1f}m)")
    print(f"Rules trained: {len(rules_to_train)}")
    print(f"Successful: {len(successful_rules)}")
    print(f"Failed: {len(failed_rules)}")
    
    if successful_rules:
        print(f"\n✅ Successful rules: {', '.join(successful_rules)}")
        
    if failed_rules:
        print(f"\n❌ Failed rules: {', '.join(failed_rules)}")
        print("   Check error messages above for debugging information")
        print("   You can retry individual rules with: python train_models_local.py --rule <rule_name>")
    
    # Provide next steps based on results
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    
    if len(successful_rules) == len(rules_to_train):
        print("🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("Next steps:")
        print("   1. Run Phase 1 Crisis Assessment again to verify resolution:")
        print("      python phase1_crisis_assessment.py")
        print("   2. If assessment passes, proceed to Phase 3 infrastructure")
        
    elif len(successful_rules) > 0:
        print(f"⚠️  PARTIAL SUCCESS: {len(successful_rules)}/{len(rules_to_train)} rules trained")
        print("Next steps:")
        print("   1. Retry failed rules individually")
        print("   2. Run Phase 1 Crisis Assessment to check progress")
        print("   3. Investigate specific training issues")
        
    else:
        print("❌ ALL TRAINING FAILED")
        print("Next steps:")
        print("   1. Check system requirements (GPU memory, dependencies)")
        print("   2. Try quick test: python train_models_local.py --quick-test")
        print("   3. Review training logs and error messages")
        print("   4. Consider environment or dependency issues")
    
    return training_results


def main():
    """Main entry point for local training script."""
    parser = argparse.ArgumentParser(
        description='Local training script for algebra EBM models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--rule',
        type=str,
        choices=['distribute', 'combine', 'isolate', 'divide'],
        help='Train only this specific rule (default: train all rules)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal parameters'
    )
    
    args = parser.parse_args()
    
    # Run training
    training_results = train_all_models(
        quick_test=args.quick_test,
        specific_rule=args.rule
    )
    
    # Exit with appropriate code
    successful_count = sum(training_results.values())
    total_count = len(training_results)
    
    if successful_count == total_count:
        sys.exit(0)  # All successful
    elif successful_count > 0:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # All failed


if __name__ == "__main__":
    main()