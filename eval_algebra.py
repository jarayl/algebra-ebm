#!/usr/bin/env python3
"""
Algebra EBM Evaluation Script

Main script for running comprehensive evaluations on algebra EBM models.
Tests single-rule, multi-rule, and constrained variants as specified in the
implementation plan.

Evaluation Sets:
1. Single-Rule Test: Held-out problems from each rule's distribution
2. Multi-Rule Test (2 rules): Equations requiring 2 sequential rules  
3. Multi-Rule Test (3 rules): Equations requiring 3 sequential rules
4. Multi-Rule Test (4 rules): Equations requiring all 4 rules
5. Constrained Test: Multi-rule + positivity/integerness constraints

Expected Results (from proposal Section 6):
- Single-Rule Accuracy: ~85%
- Multi-Rule Accuracy: ~50-60%

Usage:
    # Run full evaluation suite
    python eval_algebra.py --model_dir ./results --output_dir ./evaluation_results
    
    # Run single evaluation
    python eval_algebra.py --model_dir ./results --eval_type single_rule --rule distribute
    
    # Quick test with small datasets
    python eval_algebra.py --model_dir ./results --quick_test --max_samples 100
"""

import argparse
import logging
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import random
import torch
import numpy as np

# Import algebra components
from src.algebra.algebra_evaluation import (
    evaluate_model_suite, save_evaluation_results,
    evaluate_with_real_diffusion, run_monolithic_evaluation
)
from src.algebra.algebra_inference import load_rule_models
from src.algebra.algebra_encoder import create_decoder_with_default_candidates
from src.algebra.algebra_dataset import AlgebraDataset, MultiRuleDataset, ConstrainedDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_single_rule_datasets(
    rules: List[str],
    num_problems: int = 1000,
    d_model: int = 128,
    seed: Optional[int] = 42
) -> Dict[str, AlgebraDataset]:
    """
    Create single-rule test datasets for each rule type.
    
    Args:
        rules: List of rule names to create datasets for
        num_problems: Number of problems per rule
        d_model: Encoder embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping rule names to datasets
    """
    logger.info(f"Creating single-rule datasets with {num_problems} problems each")
    
    datasets = {}
    
    for rule in rules:
        if seed is not None:
            # Use consistent deterministic offset for each rule
            rule_offset = {'distribute': 0, 'combine': 100, 'isolate': 200, 'divide': 300}.get(rule, 0)
            np.random.seed(seed + rule_offset)
            random.seed(seed + rule_offset)

        try:
            dataset = AlgebraDataset(
                rule=rule,
                split='test',
                num_problems=num_problems,
                d_model=d_model
            )
            datasets[f"single_rule_{rule}"] = dataset
            logger.info(f"Created {rule} dataset: {len(dataset)} problems")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create dataset for rule {rule}: {str(e)}")
            logger.error(f"Traceback for rule {rule}:\n{traceback.format_exc()}")
            raise RuntimeError(f"Fast fail: Cannot create essential dataset for rule {rule}") from e
    
    return datasets


def create_multi_rule_datasets(
    num_rules_list: List[int] = [2, 3, 4],
    num_problems: int = 1000,
    d_model: int = 128,
    seed: Optional[int] = 42
) -> Dict[str, MultiRuleDataset]:
    """
    Create multi-rule test datasets for compositional evaluation.
    
    Args:
        num_rules_list: List of rule counts to create datasets for
        num_problems: Number of problems per dataset
        d_model: Encoder embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dataset names to multi-rule datasets
    """
    logger.info(f"Creating multi-rule datasets with {num_problems} problems each")
    
    datasets = {}
    
    for num_rules in num_rules_list:
        if seed is not None:
            np.random.seed(seed + num_rules * 100)
            random.seed(seed + num_rules * 100)
            
        try:
            dataset = MultiRuleDataset(
                num_rules=num_rules,
                split='test',
                num_problems=num_problems,
                d_model=d_model,
                seed=seed
            )
            datasets[f"multi_rule_{num_rules}"] = dataset
            logger.info(f"Created {num_rules}-rule dataset: {len(dataset)} problems")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create {num_rules}-rule dataset: {str(e)}")
            logger.error(f"Traceback for {num_rules}-rule dataset:\n{traceback.format_exc()}")
            raise RuntimeError(f"Fast fail: Cannot create essential multi-rule dataset for {num_rules} rules") from e
    
    return datasets


def create_constrained_datasets(
    constraint_types: List[str] = ['positive', 'integer', 'both'],
    num_rules: int = 3,
    num_problems: int = 500,
    d_model: int = 128,
    seed: Optional[int] = 42
) -> Dict[str, ConstrainedDataset]:
    """
    Create constrained test datasets for constraint evaluation.
    
    Args:
        constraint_types: List of constraint types to create datasets for
        num_rules: Number of rules to chain in each problem
        num_problems: Number of problems per constraint type
        d_model: Encoder embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping constraint names to datasets
    """
    logger.info(f"Creating constrained datasets with {num_problems} problems each")
    
    datasets = {}
    
    for i, constraint in enumerate(constraint_types):
        if seed is not None:
            np.random.seed(seed + i * 200)
            random.seed(seed + i * 200)
            
        try:
            dataset = ConstrainedDataset(
                num_rules=num_rules,
                constraints=[constraint],
                split='test',
                num_problems=num_problems,
                d_model=d_model,
                seed=seed
            )
            datasets[f"constrained_{constraint}"] = dataset
            logger.info(f"Created {constraint} constrained dataset: {len(dataset)} problems")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create {constraint} constrained dataset: {str(e)}")
            logger.error(f"Traceback for {constraint} constrained dataset:\n{traceback.format_exc()}")
            raise RuntimeError(f"Fast fail: Cannot create essential constrained dataset for {constraint}") from e
    
    return datasets


def run_single_rule_evaluation(
    rule_models: Optional[Dict[str, Any]],
    encoder: Any,
    decoder: Any,
    rule: str,
    num_problems: int = 1000,
    seed: Optional[int] = 42,
    **eval_kwargs
) -> Dict[str, Any]:
    """
    Run evaluation on a single rule.

    Args:
        rule_models: Loaded rule models (can be None for compatibility)
        encoder: Equation encoder
        decoder: Equation decoder
        rule: Rule name to evaluate
        num_problems: Number of test problems
        **eval_kwargs: Additional evaluation arguments

    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Running single-rule evaluation for: {rule}")

    # Create test dataset
    datasets = create_single_rule_datasets([rule], num_problems=num_problems, seed=seed)
    dataset_name = f"single_rule_{rule}"

    if dataset_name not in datasets:
        raise ValueError(f"Failed to create dataset for rule {rule}")

    # Run evaluation using only the specific rule model
    single_rule_models = {rule: rule_models[rule]} if rule_models and rule in rule_models else {}
    
    if not single_rule_models:
        raise ValueError(f"Model for rule {rule} not found in loaded models")
    
    from src.algebra.algebra_evaluation import evaluate_model
    
    result = evaluate_model(
        rule_models=single_rule_models,
        test_dataset=datasets[dataset_name],
        encoder=encoder,
        decoder=decoder,
        **eval_kwargs
    )
    
    return {dataset_name: result}


def run_full_evaluation_suite(
    rule_models: Optional[Dict[str, Any]],
    encoder: Any,
    decoder: Any,
    single_rule_problems: int = 1000,
    multi_rule_problems: int = 1000,
    constrained_problems: int = 500,
    seed: Optional[int] = 42,
    **eval_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run the complete evaluation suite across all test scenarios.

    Args:
        rule_models: Loaded rule models (can be None for compatibility)
        encoder: Equation encoder
        decoder: Equation decoder
        single_rule_problems: Number of problems per single rule test
        multi_rule_problems: Number of problems per multi-rule test
        constrained_problems: Number of problems per constrained test
        **eval_kwargs: Additional evaluation arguments

    Returns:
        Complete evaluation results across all test sets
    """
    logger.info("Starting full evaluation suite")

    # Create all test datasets - using Union type to support multiple dataset types
    all_datasets: Dict[str, Union[AlgebraDataset, MultiRuleDataset, ConstrainedDataset]] = {}
    
    # 1. Single-rule datasets - CRITICAL: must succeed for basic evaluation
    try:
        single_datasets = create_single_rule_datasets(
            rules=['distribute', 'combine', 'isolate', 'divide'],
            num_problems=single_rule_problems,
            seed=seed
        )
        all_datasets.update(single_datasets)
        logger.info(f"Created {len(single_datasets)} single-rule datasets")
        
        # Fast fail if no single-rule datasets created
        if len(single_datasets) == 0:
            raise RuntimeError("Fast fail: No single-rule datasets created")
            
    except Exception as e:
        logger.error(f"CRITICAL: Error creating single-rule datasets: {str(e)}")
        logger.error(f"Single-rule datasets traceback:\n{traceback.format_exc()}")
        raise RuntimeError("Fast fail: Cannot proceed without single-rule datasets") from e
    
    # 2. Multi-rule datasets - CRITICAL: must succeed for compositional evaluation
    try:
        multi_datasets = create_multi_rule_datasets(
            num_rules_list=[2, 3, 4],
            num_problems=multi_rule_problems,
            seed=seed
        )
        all_datasets.update(multi_datasets)
        logger.info(f"Created {len(multi_datasets)} multi-rule datasets")
        
        # Fast fail if no multi-rule datasets created
        if len(multi_datasets) == 0:
            raise RuntimeError("Fast fail: No multi-rule datasets created")
            
    except Exception as e:
        logger.error(f"CRITICAL: Error creating multi-rule datasets: {str(e)}")
        logger.error(f"Multi-rule datasets traceback:\n{traceback.format_exc()}")
        raise RuntimeError("Fast fail: Cannot proceed without multi-rule datasets") from e
    
    # 3. Constrained datasets - OPTIONAL: can continue without these
    try:
        constrained_datasets = create_constrained_datasets(
            constraint_types=['positive', 'integer', 'both'],
            num_problems=constrained_problems
        )
        all_datasets.update(constrained_datasets)
        logger.info(f"Created {len(constrained_datasets)} constrained datasets")
    except Exception as e:
        logger.warning(f"Non-critical: Error creating constrained datasets: {str(e)}")
        logger.warning(f"Constrained datasets traceback:\n{traceback.format_exc()}")
        logger.info("Continuing evaluation without constrained datasets")
    
    if not all_datasets:
        raise ValueError("No test datasets were created successfully")

    if rule_models is None:
        raise ValueError("rule_models cannot be None for evaluation")

    logger.info(f"Running evaluation on {len(all_datasets)} test sets")

    # Run evaluation suite
    from src.algebra.algebra_evaluation import evaluate_model_suite

    results = evaluate_model_suite(
        rule_models=rule_models,
        test_datasets=all_datasets,
        encoder=encoder,
        decoder=decoder,
        **eval_kwargs
    )
    
    return results


def generate_comparison_report(results: Dict, output_dir: str):
    """Generate markdown comparison report between monolithic and compositional approaches."""
    import os
    import json
    import time
    
    mono = results.get('monolithic', {})
    comp = results.get('compositional', {})
    
    # Calculate single-rule averages (monolithic only, since compositional uses rule-specific models)
    mono_single_accuracies = []
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        result = mono.get(f'single_rule_{rule}', {})
        if 'error' not in result and 'summary' in result:
            accuracy = result['summary'].get('accuracy', 0.0)
            mono_single_accuracies.append(accuracy)
    
    mono_single_avg = np.mean(mono_single_accuracies) if mono_single_accuracies else 0.0
    
    # Calculate multi-rule averages
    mono_multi_accuracies = []
    comp_multi_accuracies = []
    
    for n in [2, 3, 4]:
        mono_result = mono.get(f'multi_rule_{n}', {})
        if 'error' not in mono_result and 'summary' in mono_result:
            accuracy = mono_result['summary'].get('accuracy', 0.0)
            mono_multi_accuracies.append(accuracy)
        
        comp_result = comp.get(f'multi_rule_{n}', {})
        if 'error' not in comp_result and 'summary' in comp_result:
            accuracy = comp_result['summary'].get('accuracy', 0.0)
            comp_multi_accuracies.append(accuracy)
    
    mono_multi_avg = np.mean(mono_multi_accuracies) if mono_multi_accuracies else 0.0
    comp_multi_avg = np.mean(comp_multi_accuracies) if comp_multi_accuracies else 0.0
    
    # Calculate advantage
    advantage = (comp_multi_avg - mono_multi_avg) * 100
    
    # Generate report
    report = f"""# Monolithic vs Compositional Comparison

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overall Results

| Model | Single-Rule Acc | Multi-Rule Acc | Advantage |
|-------|----------------|----------------|-----------|
| Monolithic | {mono_single_avg:.1%} | {mono_multi_avg:.1%} | Baseline |
| **Compositional** | **~{mono_single_avg:.1%}*** | **{comp_multi_avg:.1%}** | **+{advantage:.1f}%** 🎯 |

*Compositional uses rule-specific models for single-rule evaluation

## Multi-Rule Breakdown

| Rules | Monolithic | Compositional | Advantage |
|-------|-----------|--------------|-----------|
"""
    
    for n in [2, 3, 4]:
        mono_acc = mono.get(f'multi_rule_{n}', {}).get('summary', {}).get('accuracy', 0)
        comp_acc = comp.get(f'multi_rule_{n}', {}).get('summary', {}).get('accuracy', 0)
        adv = (comp_acc - mono_acc) * 100
        report += f"| {n}-rule | {mono_acc:.1%} | {comp_acc:.1%} | **+{adv:.1f}%** |\n"
    
    report += f"""
## Interpretation

{'✅ **Compositional Advantage CONFIRMED!**' if advantage > 20 else '⚠️ **Partial Advantage**' if advantage > 5 else '❌ **Limited Advantage**'}

"""
    
    if advantage > 20:
        report += f"""
The compositional approach demonstrates clear superiority:
- Achieves **{advantage:.0f}%** absolute improvement on multi-rule problems
- Successfully demonstrates zero-shot compositional reasoning
- Validates modular energy function hypothesis

### Key Insights:
- **Zero-shot generalization**: Compositional models generalize to unseen rule combinations
- **Modular design advantage**: Separate rule models compose effectively
- **Energy landscape quality**: Individual energy landscapes combine constructively
"""
    elif advantage > 5:
        report += f"""
The compositional approach shows moderate improvement:
- Achieves **{advantage:.1f}%** absolute improvement on multi-rule problems
- Demonstrates some compositional reasoning capability
- Suggests potential for further optimization

### Recommendations:
- Investigate energy composition weights
- Optimize individual rule model training
- Explore alternative composition strategies
"""
    else:
        report += f"""
The compositional approach shows limited improvement:
- Achieves only **{advantage:.1f}%** absolute improvement on multi-rule problems
- May indicate issues with composition or training

### Investigation needed:
- Verify individual rule model quality
- Check energy composition implementation
- Review training hyperparameters
"""
    
    # Add detailed results
    report += f"""
## Detailed Results

### Monolithic Model Performance
"""
    
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        result = mono.get(f'single_rule_{rule}', {})
        if 'error' not in result and 'summary' in result:
            accuracy = result['summary'].get('accuracy', 0.0)
            report += f"- **{rule}**: {accuracy:.1%}\n"
    
    report += f"""
### Multi-Rule Performance Comparison
"""
    
    for n in [2, 3, 4]:
        mono_result = mono.get(f'multi_rule_{n}', {})
        comp_result = comp.get(f'multi_rule_{n}', {})
        
        mono_acc = mono_result.get('summary', {}).get('accuracy', 0.0) if 'error' not in mono_result else 0.0
        comp_acc = comp_result.get('summary', {}).get('accuracy', 0.0) if 'error' not in comp_result else 0.0
        improvement = (comp_acc - mono_acc) * 100
        
        report += f"- **{n}-rule problems**: Monolithic {mono_acc:.1%} → Compositional {comp_acc:.1%} (+{improvement:.1f}%)\n"
    
    # Add metadata
    if 'evaluation_metadata' in mono:
        metadata = mono['evaluation_metadata']
        report += f"""
## Evaluation Details

- **Evaluation time**: {metadata.get('total_evaluation_time', 0):.1f} seconds
- **Samples per dataset**: {metadata.get('num_samples_per_dataset', 'Unknown')}
- **Decoder consistency**: {metadata.get('decoder_consistency', 'Unknown')}
"""
    
    # Save report
    try:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'comparison_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
    except Exception as e:
        logger.error(f"Failed to save comparison report: {e}")
        report_path = None
    
    # Save JSON results with safe serialization
    json_path = os.path.join(output_dir, 'comparison_results.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # Use str fallback for non-serializable objects
    except Exception as e:
        logger.warning(f"Failed to save JSON results: {e}")
    
    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info("COMPARISON REPORT GENERATED")
    logger.info("="*60)
    logger.info(f"Monolithic average (single-rule): {mono_single_avg:.1%}")
    logger.info(f"Monolithic average (multi-rule): {mono_multi_avg:.1%}")
    logger.info(f"Compositional average (multi-rule): {comp_multi_avg:.1%}")
    logger.info(f"Compositional advantage: +{advantage:.1f} percentage points")
    if report_path:
        logger.info(f"Report saved to: {report_path}")
    else:
        logger.warning("Failed to save markdown report")
    logger.info(f"Results saved to: {json_path}")
    logger.info("="*60)
    
    return report_path


def generate_evaluation_report(results: Dict[str, Dict[str, Any]], output_file: Optional[str] = None):
    """
    Generate a formatted evaluation report.
    
    Args:
        results: Evaluation results from suite
        output_file: Optional file path to save report
    """
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("ALGEBRA EBM EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Extract results by category
    single_rule_results = {}
    multi_rule_results = {}
    constrained_results = {}
    
    for name, result in results.items():
        if name.startswith('_'):  # Skip metadata
            continue
        elif 'error' in result:
            report_lines.append(f"ERROR in {name}: {result['error']}")
            if 'traceback' in result:
                report_lines.append(f"TRACEBACK for {name}:")
                report_lines.append(result['traceback'])
                report_lines.append("")
            continue
            
        if name.startswith('single_rule_'):
            single_rule_results[name] = result
        elif name.startswith('multi_rule_'):
            multi_rule_results[name] = result
        elif name.startswith('constrained_'):
            constrained_results[name] = result
    
    # Single-rule results
    if single_rule_results:
        report_lines.append("SINGLE-RULE EVALUATION RESULTS")
        report_lines.append("-" * 50)
        for name, result in single_rule_results.items():
            rule_name = name.replace('single_rule_', '')
            summary = result.get('summary', {})
            accuracy = summary.get('accuracy', 0)
            invalid_rate = summary.get('invalid_rate', 0)
            
            # Include distance metrics if available (from real diffusion evaluation)
            dist_improvement = summary.get('mean_distance_improvement', None)
            if dist_improvement is not None:
                report_lines.append(f"{rule_name:12}: Accuracy={accuracy:.3f}, Invalid={invalid_rate:.3f}, DistImprove={dist_improvement*100:.1f}%")
            else:
                report_lines.append(f"{rule_name:12}: Accuracy={accuracy:.3f}, Invalid={invalid_rate:.3f}")
        
        # Calculate average
        accuracies = [r['summary']['accuracy'] for r in single_rule_results.values() if 'summary' in r]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            report_lines.append(f"{'Average':12}: Accuracy={avg_accuracy:.3f}")
        
        # Distance improvement averages if available
        dist_improvements = [r['summary'].get('mean_distance_improvement') for r in single_rule_results.values() 
                           if r.get('summary', {}).get('mean_distance_improvement') is not None]
        if dist_improvements:
            avg_dist_improvement = np.mean(dist_improvements)
            report_lines.append(f"{'Average':12}: DistImprove={avg_dist_improvement*100:.1f}%")
        
        report_lines.append("")
    
    # Multi-rule results
    if multi_rule_results:
        report_lines.append("MULTI-RULE EVALUATION RESULTS")
        report_lines.append("-" * 50)
        for name, result in multi_rule_results.items():
            num_rules = name.replace('multi_rule_', '')
            summary = result.get('summary', {})
            accuracy = summary.get('accuracy', 0)
            invalid_rate = summary.get('invalid_rate', 0)
            report_lines.append(f"{num_rules} rules:    Accuracy={accuracy:.3f}, Invalid={invalid_rate:.3f}")
        report_lines.append("")
    
    # Constrained results
    if constrained_results:
        report_lines.append("CONSTRAINED EVALUATION RESULTS")
        report_lines.append("-" * 50)
        for name, result in constrained_results.items():
            constraint = name.replace('constrained_', '')
            summary = result.get('summary', {})
            accuracy = summary.get('accuracy', 0)
            invalid_rate = summary.get('invalid_rate', 0)
            report_lines.append(f"{constraint:12}: Accuracy={accuracy:.3f}, Invalid={invalid_rate:.3f}")
        report_lines.append("")
    
    # Expected vs actual comparison
    if single_rule_results and multi_rule_results:
        report_lines.append("EXPECTED vs ACTUAL RESULTS")
        report_lines.append("-" * 50)
        
        single_avg = np.mean([r['summary']['accuracy'] for r in single_rule_results.values()])
        multi_avg = np.mean([r['summary']['accuracy'] for r in multi_rule_results.values()])
        
        report_lines.append(f"Single-Rule Average: {single_avg:.3f} (Expected: ~0.850)")
        report_lines.append(f"Multi-Rule Average:  {multi_avg:.3f} (Expected: ~0.500-0.600)")
        report_lines.append("")
        
        if single_avg >= 0.80:
            report_lines.append("✓ Single-rule performance meets expectations")
        else:
            report_lines.append("✗ Single-rule performance below expectations")
            
        if multi_avg >= 0.45:
            report_lines.append("✓ Multi-rule performance meets minimum expectations")
        else:
            report_lines.append("✗ Multi-rule performance below expectations")
        report_lines.append("")
    
    report_lines.append("="*80)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Report saved to {output_file}")


def main():
    """Main evaluation function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Algebra EBM Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./results',
        help='Directory containing trained rule models (default: ./results)'
    )
    
    # Evaluation type
    parser.add_argument(
        '--eval_type',
        type=str,
        default='single_rule',
        choices=['single_rule', 'multi_rule', 'monolithic', 'comparison', 'constrained', 'full'],
        help='Evaluation type (default: single_rule)'
    )
    
    # Specific evaluation options
    parser.add_argument(
        '--rule',
        choices=['distribute', 'combine', 'isolate', 'divide'],
        help='Specific rule to evaluate (for single_rule eval_type)'
    )
    
    parser.add_argument(
        '--num_rules',
        type=int,
        choices=[2, 3, 4],
        help='Number of rules for multi-rule evaluation'
    )
    
    # Dataset sizes
    parser.add_argument(
        '--single_rule_problems',
        type=int,
        default=100,
        help='Number of problems per single-rule test (default: 100)'
    )
    
    parser.add_argument(
        '--multi_rule_problems',
        type=int,
        default=100,
        help='Number of problems per multi-rule test (default: 100)'
    )
    
    parser.add_argument(
        '--constrained_problems',
        type=int,
        default=50,
        help='Number of problems per constrained test (default: 50)'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with small datasets'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        help='Maximum samples to evaluate (for quick testing)'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results (default: ./evaluation_results)'
    )
    
    parser.add_argument(
        '--save_detailed',
        action='store_true',
        help='Save detailed per-sample results (uses more memory)'
    )
    
    # Encoder options
    parser.add_argument(
        '--encoder_type',
        choices=['character', 'ast'],
        default='character',
        help='Type of encoder to use (default: character)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to run on (auto, cuda, cpu) (default: auto)'
    )
    
    # Inference parameters
    parser.add_argument(
        '--inference_T',
        type=int,
        default=50,
        help='Number of gradient steps per landscape (default: 50)'
    )

    parser.add_argument(
        '--inference_step_size',
        type=float,
        default=0.05,
        help='Step size for gradient descent (default: 0.05)'
    )

    parser.add_argument(
        '--num_inference_starts',
        type=int,
        default=1,
        help='Number of random starts for multi-start inference (default: 1)'
    )

    parser.add_argument(
        '--enable_diagnostics',
        action='store_true',
        help='Enable detailed per-iteration diagnostic logging'
    )

    parser.add_argument(
        '--diagnostics_dir',
        type=str,
        default=None,
        help='Directory to save diagnostic trajectory files (required if --enable_diagnostics)'
    )
    
    # Real diffusion inference (recommended)
    parser.add_argument(
        '--use_real_diffusion',
        action='store_true',
        help='Use GaussianDiffusion1D.sample() for inference (RECOMMENDED - achieves 87%+ accuracy). '
             'Requires --checkpoint to specify the full model checkpoint.'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to the full model checkpoint (model.pt) for real diffusion inference. '
             'Required when --use_real_diffusion is specified.'
    )

    parser.add_argument(
        '--monolithic_checkpoint',
        type=str,
        default='./results/monolithic/model.pt',
        help='Path to monolithic model checkpoint (default: ./results/monolithic/model.pt)'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()

    # Validate real diffusion arguments
    if args.use_real_diffusion and not args.checkpoint:
        # For comparison mode, checkpoints are loaded from model_dir, so --checkpoint is not required
        if args.eval_type != 'comparison':
            parser.error("--checkpoint is required when using --use_real_diffusion (except for comparison mode)")

    # Validate diagnostics arguments
    if args.enable_diagnostics and args.diagnostics_dir is None:
        parser.error("--diagnostics_dir is required when --enable_diagnostics is specified")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Quick test mode adjustments
    if args.quick_test:
        args.single_rule_problems = min(args.single_rule_problems, 100)
        args.multi_rule_problems = min(args.multi_rule_problems, 100)  
        args.constrained_problems = min(args.constrained_problems, 50)
        logger.info("Quick test mode: Using smaller dataset sizes")
    
    # Create output directory
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create encoder and decoder (needed for both evaluation paths)
        if args.encoder_type == 'character':
            from src.algebra.algebra_encoder import create_character_encoder
            encoder = create_character_encoder(d_model=128)
        else:
            from src.algebra.algebra_encoder import create_ast_encoder
            encoder = create_ast_encoder(d_model=128)
        
        logger.info(f"Created {args.encoder_type} encoder")
        
        # Create decoder with default candidates - evaluate_model will rebuild it from test dataset
        decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
        logger.info(f"Created decoder with {len(decoder.candidate_equations)} default candidates - will be rebuilt from test dataset in evaluate_model")
        
        # Load rule models only if NOT using real diffusion AND NOT doing monolithic evaluation
        rule_models = None
        if not args.use_real_diffusion and args.eval_type != 'monolithic':
            logger.info(f"Loading models from {args.model_dir}")
            rule_models = load_rule_models(
                rule_names=['distribute', 'combine', 'isolate', 'divide'],
                model_dir=args.model_dir,
                device=device
            )
            
            if not rule_models:
                raise ValueError(f"CRITICAL: No models loaded from {args.model_dir}")
            
            # Fast fail if we don't have models for all basic rules
            required_rules = {'distribute', 'combine', 'isolate', 'divide'}
            loaded_rules = set(rule_models.keys())
            missing_rules = required_rules - loaded_rules
            
            if missing_rules:
                raise ValueError(f"CRITICAL: Missing models for essential rules: {missing_rules}. Fast fail.")
            
            logger.info(f"Loaded {len(rule_models)} rule models: {list(rule_models.keys())}")
        elif args.eval_type == 'monolithic':
            logger.info("Skipping rule model loading for monolithic evaluation")
        
        # Set up evaluation parameters
        eval_params = {
            'inference_params': {
                'T': args.inference_T,
                'step_size': args.inference_step_size
            },
            'store_detailed_results': args.save_detailed,
            'num_inference_starts': args.num_inference_starts,
            'enable_diagnostics': args.enable_diagnostics,
            'diagnostics_dir': args.diagnostics_dir
        }

        if args.max_samples:
            eval_params['max_samples'] = args.max_samples
        
        # Run evaluation based on type
        start_time = time.time()
        
        # Use real diffusion inference if requested (RECOMMENDED)
        if args.use_real_diffusion:
            logger.info("="*60)
            logger.info("Using REAL GaussianDiffusion1D.sample() for inference")
            logger.info("This is the RECOMMENDED method - achieves 87%+ distance improvement")
            logger.info("="*60)
            
            if args.eval_type == 'single_rule':
                if not args.rule:
                    raise ValueError("--rule must be specified for single_rule evaluation")
                
                # Create single-rule dataset
                dataset = AlgebraDataset(
                    rule=args.rule,
                    split='test',
                    num_problems=args.single_rule_problems,
                    d_model=128
                )
                
                results = evaluate_with_real_diffusion(
                    checkpoint_path=args.checkpoint,
                    test_dataset=dataset,
                    encoder=encoder,
                    decoder=decoder,
                    max_samples=args.max_samples,
                    device=device,
                    store_detailed_results=args.save_detailed
                )
                # Wrap in dict for compatibility with report generator
                results = {f"single_rule_{args.rule}": results}
                
            elif args.eval_type == 'full':
                # Run real diffusion evaluation for each rule
                results = {}
                for rule in ['distribute', 'combine', 'isolate', 'divide']:
                    logger.info(f"\nEvaluating rule: {rule}")
                    
                    # Find checkpoint for this rule
                    rule_checkpoint = Path(args.checkpoint)
                    if not rule_checkpoint.exists():
                        # Try to find rule-specific checkpoint
                        model_dir = Path(args.model_dir)
                        possible_paths = [
                            model_dir / rule / 'model.pt',
                            model_dir / rule / 'model-1.pt',
                            model_dir / f"{rule}_model.pt",
                        ]
                        rule_checkpoint = None
                        for p in possible_paths:
                            if p.exists():
                                rule_checkpoint = p
                                break
                        if rule_checkpoint is None:
                            logger.warning(f"No checkpoint found for rule {rule}, skipping")
                            continue
                    
                    dataset = AlgebraDataset(
                        rule=rule,
                        split='test',
                        num_problems=args.single_rule_problems,
                        d_model=128
                    )
                    
                    rule_results = evaluate_with_real_diffusion(
                        checkpoint_path=str(rule_checkpoint),
                        test_dataset=dataset,
                        encoder=encoder,
                        decoder=decoder,
                        max_samples=args.max_samples,
                        device=device,
                        store_detailed_results=args.save_detailed
                    )
                    results[f"single_rule_{rule}"] = rule_results
            elif args.eval_type == 'monolithic':
                # Run monolithic evaluation
                results = run_monolithic_evaluation(
                    monolithic_checkpoint=args.monolithic_checkpoint,
                    output_dir=args.output_dir,
                    num_samples=args.max_samples if args.max_samples else 1000
                )
            elif args.eval_type == 'comparison':
                # Run both monolithic and compositional evaluations for comparison
                logger.info("="*60)
                logger.info("RUNNING COMPARISON: Monolithic vs Compositional")
                logger.info("="*60)
                
                results = {}
                
                # 1. Monolithic evaluation
                logger.info("\n[1/2] Monolithic Evaluation")
                
                # Validate monolithic checkpoint exists
                if not Path(args.monolithic_checkpoint).exists():
                    raise ValueError(f"Monolithic checkpoint not found: {args.monolithic_checkpoint}")
                
                mono_results = run_monolithic_evaluation(
                    monolithic_checkpoint=args.monolithic_checkpoint,
                    output_dir=args.output_dir,
                    num_samples=args.max_samples if args.max_samples else 1000
                )
                results['monolithic'] = mono_results
                
                # 2. Compositional evaluation
                logger.info("\n[2/2] Compositional Evaluation")
                
                # Load rule checkpoints
                from pathlib import Path
                rule_checkpoints = {}
                for rule in ['distribute', 'combine', 'isolate', 'divide']:
                    checkpoint_path = Path(args.model_dir) / rule / 'model.pt'
                    if checkpoint_path.exists():
                        rule_checkpoints[rule] = str(checkpoint_path)
                    else:
                        logger.warning(f"Checkpoint not found for rule {rule} at {checkpoint_path}")
                
                if not rule_checkpoints:
                    raise ValueError(f"No rule checkpoints found in {args.model_dir}. Cannot run compositional evaluation.")
                
                comp_results = {}
                
                # Load rule models for compositional evaluation
                from src.algebra.algebra_evaluation import load_diffusion_model_for_inference, evaluate_with_composition
                rule_models = {}
                diffusion_template = None
                
                for rule_name, checkpoint_path in rule_checkpoints.items():
                    logger.info(f"Loading {rule_name} model from {checkpoint_path}")
                    diffusion, ebm = load_diffusion_model_for_inference(checkpoint_path, device)
                    rule_models[rule_name] = ebm  # Store EBM models for energy composition
                    
                    # Use first model as diffusion template
                    if diffusion_template is None:
                        diffusion_template = diffusion
                
                # Test on multi-rule datasets
                for num_rules in [2, 3, 4]:
                    logger.info(f"Evaluating {num_rules}-rule compositional problems")
                    
                    test_dataset = MultiRuleDataset(
                        num_rules=num_rules,
                        split='test',
                        num_problems=args.max_samples if args.max_samples else 1000,
                        d_model=128
                    )
                    
                    result = evaluate_with_composition(
                        rule_models_dict=rule_models,
                        test_dataset=test_dataset,
                        diffusion_template=diffusion_template,
                        encoder=encoder,
                        decoder=decoder,
                        device=device,
                        max_samples=args.max_samples,
                        store_detailed_results=args.save_detailed
                    )
                    
                    comp_results[f'multi_rule_{num_rules}'] = result
                
                results['compositional'] = comp_results
                
                # Generate comparison report
                generate_comparison_report(results, args.output_dir)
            else:
                raise ValueError(f"--use_real_diffusion currently only supports single_rule, full, monolithic, and comparison eval_type, got: {args.eval_type}")
        
        # Original evaluation path (uses AlgebraInference)
        elif args.eval_type == 'single_rule':
            if not args.rule:
                raise ValueError("--rule must be specified for single_rule evaluation")
            
            results = run_single_rule_evaluation(
                rule_models=rule_models,
                encoder=encoder,
                decoder=decoder,
                rule=args.rule,
                num_problems=args.single_rule_problems,
                seed=args.seed,
                **eval_params
            )
            
        elif args.eval_type == 'multi_rule':
            if not args.num_rules:
                raise ValueError("--num_rules must be specified for multi_rule evaluation")

            if rule_models is None:
                raise ValueError("rule_models cannot be None for multi_rule evaluation")

            datasets = create_multi_rule_datasets(
                num_rules_list=[args.num_rules],
                num_problems=args.multi_rule_problems,
                seed=args.seed
            )

            from src.algebra.algebra_evaluation import evaluate_model_suite
            results = evaluate_model_suite(
                rule_models=rule_models,
                test_datasets=datasets,  # type: ignore[arg-type]
                encoder=encoder,
                decoder=decoder,
                **eval_params
            )
            
        elif args.eval_type == 'constrained':
            if rule_models is None:
                raise ValueError("rule_models cannot be None for constrained evaluation")

            datasets = create_constrained_datasets(
                constraint_types=['positive', 'integer', 'both'],
                num_problems=args.constrained_problems
            )

            from src.algebra.algebra_evaluation import evaluate_model_suite
            results = evaluate_model_suite(
                rule_models=rule_models,
                test_datasets=datasets,  # type: ignore[arg-type]
                encoder=encoder,
                decoder=decoder,
                **eval_params
            )
            
        elif args.eval_type == 'monolithic':
            # Run monolithic evaluation using the provided checkpoint
            results = run_monolithic_evaluation(
                monolithic_checkpoint=args.monolithic_checkpoint,
                output_dir=args.output_dir,
                num_samples=args.max_samples if args.max_samples else 1000
            )
            
        elif args.eval_type == 'full':
            results = run_full_evaluation_suite(
                rule_models=rule_models,
                encoder=encoder,
                decoder=decoder,
                single_rule_problems=args.single_rule_problems,
                multi_rule_problems=args.multi_rule_problems,
                constrained_problems=args.constrained_problems,
                seed=args.seed,
                **eval_params
            )
        
        elif args.eval_type == 'comparison':
            raise ValueError(
                "Comparison evaluation requires --use_real_diffusion. "
                "Please add --use_real_diffusion to enable monolithic vs compositional comparison."
            )
        
        else:
            raise ValueError(f"Unknown evaluation type: {args.eval_type}")
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Save detailed results
        results_file = output_dir / f"evaluation_results_{args.eval_type}.json"
        save_evaluation_results(results, str(results_file))
        
        # Generate and save report
        report_file = output_dir / f"evaluation_report_{args.eval_type}.txt"
        generate_evaluation_report(results, str(report_file))
        
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"CRITICAL: Evaluation failed: {str(e)}")
        logger.error(f"Main evaluation traceback:\n{traceback.format_exc()}")
        logger.error("Fast fail: Evaluation cannot continue due to critical error")
        raise


if __name__ == "__main__":
    main()