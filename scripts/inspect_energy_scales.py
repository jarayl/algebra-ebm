#!/usr/bin/env python3
"""
Energy Scale Diagnostic Script

This script measures learned energy scales from trained algebraic rule models
to diagnose energy scale mismatch issues in compositional inference.

The core issue: Each rule model learns its own energy_scale parameter (range 0.1-10.0)
which causes naive summation to be dominated by models with larger scales.

Usage:
    python scripts/inspect_energy_scales.py --model_dir ./results
    python scripts/inspect_energy_scales.py --model_dir ./results --output scales_report.json
    python scripts/inspect_energy_scales.py --help

Example Output:
    distribute: scale=2.300, bias=0.150, range_ratio=3.78x
    combine: scale=8.700, bias=0.090, 
    isolate: scale=4.100, bias=0.200,
    divide: scale=6.500, bias=0.050,
    
    Scale Statistics:
    - Min scale: 2.300 (distribute)
    - Max scale: 8.700 (combine)
    - Range ratio: 3.78x
    - Naive summation impact: combine model dominates by factor of 3.8x
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import load_rule_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_learned_scales(model_dir: str, rules: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Extract learned energy_scale and energy_bias parameters from model checkpoints.
    
    This function handles multiple checkpoint formats and device compatibility issues
    gracefully, providing detailed diagnostic information about energy scaling parameters.
    
    Args:
        model_dir: Directory containing saved rule models
        rules: List of specific rule names to inspect (default: all standard rules)
        
    Returns:
        scales_data: Dictionary mapping rule names to their scale parameters
                    Format: {rule_name: {'scale': float, 'bias': float, 'checkpoint_path': str}}
                    
    Raises:
        ValueError: If model_dir doesn't exist or contains no valid checkpoints
        RuntimeError: If critical errors occur during model loading
    """
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
        
    logger.info(f"Inspecting energy scales in directory: {model_dir}")
    
    # Use provided rule names or default to all standard rules
    possible_rules = rules if rules is not None else ['distribute', 'combine', 'isolate', 'divide']
    scales_data = {}
    
    # Track loading statistics
    loading_stats = {
        'attempted': 0,
        'successful': 0,
        'failed': 0,
        'missing': 0
    }
    
    for rule_name in possible_rules:
        logger.info(f"Searching for {rule_name} model...")
        loading_stats['attempted'] += 1
        
        # Try multiple possible checkpoint paths (matching load_rule_models logic)
        possible_paths = [
            model_dir_path / rule_name / 'model.pt',
            model_dir_path / rule_name / 'checkpoint.pt', 
            model_dir_path / rule_name / 'model-1.pt',
            model_dir_path / rule_name / 'model-final.pt'
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                logger.info(f"  Found checkpoint: {path}")
                break
        
        if model_path is None:
            logger.warning(f"  No checkpoint found for {rule_name}")
            logger.debug(f"  Searched paths: {[str(p) for p in possible_paths]}")
            loading_stats['missing'] += 1
            continue
            
        try:
            # Load checkpoint with device compatibility and security
            device = 'cpu'  # Force CPU loading for diagnostic purposes
            
            logger.debug(f"Loading checkpoint: {model_path}")
            # FIX CRIT-005: Add weights_only=True to prevent arbitrary code execution
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            # Create model architecture to extract parameters
            ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
            wrapper = AlgebraDiffusionWrapper(ebm)
            
            # Handle different checkpoint formats (copied from load_rule_models)
            state_loaded = False
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                logger.debug(f"  Detected Trainer1D checkpoint format")
                full_state = checkpoint['model']
                
                # Extract EBM parameters with proper prefix handling
                has_orig_mod_keys = any(k.startswith('_orig_mod.model.ebm.') for k in full_state.keys())
                has_regular_keys = any(k.startswith('model.ebm.') for k in full_state.keys())
                
                if has_orig_mod_keys:
                    logger.debug("  Using '_orig_mod.model.ebm.' prefix extraction")
                    ebm_state = {
                        k.replace('_orig_mod.model.', '', 1): v
                        for k, v in full_state.items()
                        if k.startswith('_orig_mod.model.ebm.')
                    }
                elif has_regular_keys:
                    logger.debug("  Using 'model.ebm.' prefix extraction")
                    ebm_state = {
                        k.replace('model.', '', 1): v
                        for k, v in full_state.items()
                        if k.startswith('model.ebm.')
                    }
                else:
                    logger.debug("  Using direct state dict extraction")
                    ebm_state = full_state
                    
                if ebm_state:
                    missing_keys = wrapper.load_state_dict(ebm_state, strict=False)
                    state_loaded = True
                    
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                logger.debug(f"  Detected standard PyTorch checkpoint format")
                missing_keys = wrapper.load_state_dict(checkpoint['model_state_dict'], strict=False)
                state_loaded = True
                
            elif isinstance(checkpoint, dict) and any(key.startswith('ebm.') for key in checkpoint.keys()):
                logger.debug(f"  Detected direct state dict format")
                missing_keys = wrapper.load_state_dict(checkpoint, strict=False)
                state_loaded = True
                
            else:
                logger.debug(f"  Unknown format, attempting direct load")
                missing_keys = wrapper.load_state_dict(checkpoint, strict=False)
                state_loaded = True
            
            if not state_loaded:
                logger.error(f"  Failed to load state for {rule_name}")
                loading_stats['failed'] += 1
                continue
                
            # FIX CRIT-009: Add attribute existence checks before accessing energy parameters
            if hasattr(wrapper.ebm, 'energy_scale') and wrapper.ebm.energy_scale is not None:
                energy_scale = wrapper.ebm.energy_scale.item()
            else:
                logger.warning(f"  Model {rule_name} missing energy_scale parameter, using default 1.0")
                energy_scale = 1.0
                
            if hasattr(wrapper.ebm, 'energy_bias') and wrapper.ebm.energy_bias is not None:
                energy_bias = wrapper.ebm.energy_bias.item()
            else:
                logger.warning(f"  Model {rule_name} missing energy_bias parameter, using default 0.0")
                energy_bias = 0.0
            
            scales_data[rule_name] = {
                'scale': energy_scale,
                'bias': energy_bias,
                'checkpoint_path': str(model_path),
                'missing_keys': list(missing_keys.missing_keys) if hasattr(missing_keys, 'missing_keys') else [],
                'unexpected_keys': list(missing_keys.unexpected_keys) if hasattr(missing_keys, 'unexpected_keys') else []
            }
            
            logger.info(f"  Successfully extracted: scale={energy_scale:.3f}, bias={energy_bias:.3f}")
            loading_stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"  Error loading {rule_name}: {type(e).__name__}: {e}")
            loading_stats['failed'] += 1
            continue
    
    # Log loading summary
    logger.info(f"Loading summary: {loading_stats['successful']}/{loading_stats['attempted']} models loaded")
    if loading_stats['missing'] > 0:
        logger.warning(f"  {loading_stats['missing']} models missing")
    if loading_stats['failed'] > 0:
        logger.warning(f"  {loading_stats['failed']} models failed to load")
        
    if not scales_data:
        raise ValueError(f"No valid models found in {model_dir}")
        
    return scales_data


def print_scale_analysis(scales_data: Dict[str, Dict[str, float]]) -> None:
    """
    Print comprehensive analysis of energy scale data.
    
    This function provides human-readable output showing the energy scale mismatch
    problem and quantifying its impact on compositional inference.
    
    Args:
        scales_data: Dictionary of scale data from inspect_learned_scales
    """
    if not scales_data:
        print("No scale data available for analysis")
        return
        
    print("\n" + "=" * 70)
    print("ENERGY SCALE DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # Print individual model scales
    print("\nLearned Energy Scales by Rule:")
    print("-" * 50)
    
    scales = []
    biases = []
    rule_names = []
    
    for rule_name, data in sorted(scales_data.items()):
        scale = data['scale']
        bias = data['bias']
        scales.append(scale)
        biases.append(bias)
        rule_names.append(rule_name)
        
        # Add status indicators for unusual values
        status_indicators = []
        if scale < 0.5:
            status_indicators.append("LOW")
        elif scale > 8.0:
            status_indicators.append("HIGH") 
        if abs(bias) > 1.0:
            status_indicators.append("BIAS!")
            
        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""
        
        print(f"  {rule_name:>10}: scale={scale:6.3f}, bias={bias:6.3f}{status_str}")
        
        # Show checkpoint path for debugging
        if 'checkpoint_path' in data:
            print(f"             checkpoint: {Path(data['checkpoint_path']).name}")
    
    # Calculate scale statistics
    min_scale = min(scales)
    max_scale = max(scales)
    scale_range = max_scale - min_scale
    scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
    mean_scale = sum(scales) / len(scales)
    
    min_rule = rule_names[scales.index(min_scale)]
    max_rule = rule_names[scales.index(max_scale)]
    
    print(f"\nScale Statistics:")
    print("-" * 30)
    print(f"  Min scale:     {min_scale:.3f} ({min_rule})")
    print(f"  Max scale:     {max_scale:.3f} ({max_rule})")
    print(f"  Mean scale:    {mean_scale:.3f}")
    print(f"  Scale range:   {scale_range:.3f}")
    print(f"  Scale ratio:   {scale_ratio:.2f}x")
    
    # Bias statistics
    min_bias = min(biases)
    max_bias = max(biases)
    bias_range = max_bias - min_bias
    mean_bias = sum(biases) / len(biases)
    
    print(f"\nBias Statistics:")
    print("-" * 30)
    print(f"  Min bias:      {min_bias:.3f}")
    print(f"  Max bias:      {max_bias:.3f}")
    print(f"  Mean bias:     {mean_bias:.3f}")
    print(f"  Bias range:    {bias_range:.3f}")
    
    # Naive summation impact analysis
    print(f"\nNaive Summation Impact Analysis:")
    print("-" * 40)
    
    if scale_ratio > 2.0:
        dominant_contribution = (max_scale / sum(scales)) * 100
        print(f"  🚨 ENERGY SCALE MISMATCH DETECTED!")
        print(f"  The '{max_rule}' model dominates by factor {scale_ratio:.2f}x")
        print(f"  In naive summation, '{max_rule}' contributes {dominant_contribution:.1f}% of total energy")
        print(f"  Other models are effectively suppressed")
        
        # Calculate relative suppression
        suppression_factors = [max_scale / s for s in scales]
        print(f"\n  Suppression factors (relative to {max_rule}):")
        for i, (rule, factor) in enumerate(zip(rule_names, suppression_factors)):
            if rule != max_rule:
                print(f"    {rule:>10}: {factor:.2f}x suppressed")
                
        # Recommendations
        print(f"\n  💡 RECOMMENDATIONS:")
        print(f"  1. Apply scale normalization: divide each energy by its model's scale")
        print(f"  2. Use weighted summation with inverse scale weights")
        print(f"  3. Consider scale-aware training objectives")
        
    elif scale_ratio > 1.5:
        print(f"  ⚠️  Moderate scale imbalance detected (ratio: {scale_ratio:.2f}x)")
        print(f"  May cause '{max_rule}' to have outsized influence")
        print(f"  Consider monitoring composition performance")
        
    else:
        print(f"  ✅ Scale imbalance is minimal (ratio: {scale_ratio:.2f}x)")
        print(f"  Naive summation should work reasonably well")
    
    # Additional diagnostic info
    if any('missing_keys' in data and data['missing_keys'] for data in scales_data.values()):
        print(f"\n⚠️  Some models had missing parameters (initialized to defaults)")
    
    if any('unexpected_keys' in data and data['unexpected_keys'] for data in scales_data.values()):
        print(f"⚠️  Some models had unexpected parameters (ignored)")
    
    print("\n" + "=" * 70)


def save_scale_report(scales_data: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Save detailed scale analysis to JSON file.
    
    Args:
        scales_data: Dictionary of scale data from inspect_learned_scales
        output_path: Path to save JSON report
    """
    # Calculate summary statistics
    if not scales_data:
        report = {'error': 'No scale data available'}
    else:
        scales = [data['scale'] for data in scales_data.values()]
        biases = [data['bias'] for data in scales_data.values()]
        rule_names = list(scales_data.keys())
        
        min_scale = min(scales)
        max_scale = max(scales)
        scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
        
        min_rule = rule_names[scales.index(min_scale)]
        max_rule = rule_names[scales.index(max_scale)]
        
        # Create comprehensive report
        report = {
            'timestamp': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown'),
            'individual_scales': scales_data,
            'summary_statistics': {
                'scale_stats': {
                    'min': min_scale,
                    'max': max_scale,
                    'mean': sum(scales) / len(scales),
                    'ratio': scale_ratio,
                    'range': max_scale - min_scale,
                    'min_rule': min_rule,
                    'max_rule': max_rule
                },
                'bias_stats': {
                    'min': min(biases),
                    'max': max(biases),
                    'mean': sum(biases) / len(biases),
                    'range': max(biases) - min(biases)
                },
                'num_models': len(scales_data)
            },
            'impact_analysis': {
                'scale_imbalance_severity': (
                    'severe' if scale_ratio > 2.0 else
                    'moderate' if scale_ratio > 1.5 else
                    'minimal'
                ),
                'dominant_model': max_rule,
                'dominant_contribution_percent': (max_scale / sum(scales)) * 100,
                'suppression_factors': {
                    rule: max_scale / scales_data[rule]['scale']
                    for rule in rule_names
                }
            },
            'recommendations': [
                "Apply scale normalization before summation",
                "Use weighted summation with inverse scale weights", 
                "Monitor composition performance",
                "Consider scale-aware training objectives"
            ] if scale_ratio > 1.5 else [
                "Current scales are reasonably balanced",
                "Naive summation should work well"
            ]
        }
    
    # Save to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Scale report saved to: {output_path}")


def main():
    """Main entry point for the energy scale diagnostic script."""
    parser = argparse.ArgumentParser(
        description="Diagnose learned energy scales in algebraic rule models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inspect_energy_scales.py --model_dir ./results
  python scripts/inspect_energy_scales.py --model_dir ./results --output scales_report.json
  python scripts/inspect_energy_scales.py --model_dir /path/to/models --verbose
        
This script helps identify energy scale mismatch issues that can cause
certain rule models to dominate in compositional inference.
        """
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./results',
        help='Directory containing saved rule models (default: ./results)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save JSON report (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )
    
    parser.add_argument(
        '--rules',
        nargs='+',
        default=['distribute', 'combine', 'isolate', 'divide'],
        help='Specific rule names to inspect (default: all standard rules)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    try:
        logger.info("Starting energy scale diagnostic...")
        
        # FIX DEVIL-001: Pass args.rules to inspect_learned_scales function
        scales_data = inspect_learned_scales(args.model_dir, args.rules)
        
        # Print analysis to console
        print_scale_analysis(scales_data)
        
        # Save detailed report if requested
        if args.output:
            save_scale_report(scales_data, args.output)
        
        # Return appropriate exit code
        scales = [data['scale'] for data in scales_data.values()]
        if scales:
            min_scale = min(scales)
            max_scale = max(scales)
            scale_ratio = max_scale / min_scale if min_scale > 0 else float('inf')
            
            if scale_ratio > 2.0:
                logger.warning("Severe energy scale imbalance detected!")
                return 1  # Exit code indicating issue found
            elif scale_ratio > 1.5:
                logger.warning("Moderate energy scale imbalance detected")
                return 1
            else:
                logger.info("Energy scales are reasonably balanced")
                return 0
        else:
            logger.error("No valid scale data extracted")
            return 2
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 2
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 3
        
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        return 4


if __name__ == '__main__':
    exit(main())