#!/usr/bin/env python3
"""
Compare Monolithic vs Compositional IRED

Runs systematic comparison between monolithic and compositional approaches
and generates comprehensive markdown report.

Usage:
    python scripts/compare_monolithic_vs_compositional.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results \
        --num_samples 1000 \
        --output_dir ./comparison_results

Example:
    # Quick test with fewer samples
    python scripts/compare_monolithic_vs_compositional.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results \
        --num_samples 100 \
        --output_dir ./test_comparison

    # Full evaluation (default)
    python scripts/compare_monolithic_vs_compositional.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results
"""

import argparse
import sys
import os
from pathlib import Path


def validate_paths(args):
    """Validate that required paths exist."""
    errors = []
    
    # Check monolithic checkpoint
    if not Path(args.monolithic_checkpoint).exists():
        errors.append(f"Monolithic checkpoint not found: {args.monolithic_checkpoint}")
    
    # Check compositional directory
    if not Path(args.compositional_dir).exists():
        errors.append(f"Compositional directory not found: {args.compositional_dir}")
    else:
        # Check for rule model checkpoints
        missing_rules = []
        for rule in ['distribute', 'combine', 'isolate', 'divide']:
            rule_checkpoint = Path(args.compositional_dir) / rule / 'model.pt'
            if not rule_checkpoint.exists():
                missing_rules.append(rule)
        
        if missing_rules:
            errors.append(f"Missing rule checkpoints in {args.compositional_dir}: {missing_rules}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(
        description='Compare monolithic vs compositional IRED performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--monolithic_checkpoint',
        type=str,
        required=True,
        help='Path to monolithic model checkpoint (e.g., ./results/monolithic/model.pt)'
    )
    
    parser.add_argument(
        '--compositional_dir', 
        type=str,
        required=True,
        help='Directory containing rule model checkpoints (e.g., ./results)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to evaluate per dataset (default: 1000)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_results',
        help='Output directory for results (default: ./comparison_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    print("Validating paths...")
    errors = validate_paths(args)
    
    if errors:
        print("❌ Path validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues and try again.")
        return 1
    
    print("✅ Path validation passed")
    
    # Build command
    eval_script = Path(__file__).parent.parent / "eval_algebra.py"
    
    cmd = [
        sys.executable, str(eval_script),
        "--eval_type", "comparison",
        "--use_real_diffusion",
        "--monolithic_checkpoint", args.monolithic_checkpoint,
        "--model_dir", args.compositional_dir,
        "--max_samples", str(args.num_samples),
        "--output_dir", args.output_dir
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Print command for transparency
    print("\n" + "="*60)
    print("RUNNING COMPARISON EVALUATION")
    print("="*60)
    print(f"Monolithic checkpoint: {args.monolithic_checkpoint}")
    print(f"Compositional directory: {args.compositional_dir}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print("\nCommand:")
    print(" ".join(cmd))
    print("="*60)
    
    # Run comparison evaluation
    try:
        import subprocess
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("✅ COMPARISON COMPLETE!")
        print("="*60)
        print(f"Results available in: {args.output_dir}")
        print(f"  - comparison_report.md (markdown report)")
        print(f"  - comparison_results.json (detailed results)")
        print("="*60)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Comparison evaluation failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Could not find eval_algebra.py at {eval_script}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())