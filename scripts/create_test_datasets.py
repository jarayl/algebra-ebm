#!/usr/bin/env python3
"""
Create multi-rule test datasets for benchmarking composition performance.

This script generates test datasets for 2-rule, 3-rule, and 4-rule problems
to benchmark compositional performance of algebra solvers.

Generated datasets:
- 2-rule problems (distribute+combine, 100 samples)
- 3-rule problems (distribute+combine+isolate, 50 samples)  
- 4-rule problems (all rules, 25 samples)

Usage:
    python scripts/create_test_datasets.py
    python scripts/create_test_datasets.py --output_dir results/test_datasets
    python scripts/create_test_datasets.py --seed 42 --verbose
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algebra.algebra_dataset import MultiRuleDataset


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def save_dataset(dataset: MultiRuleDataset, output_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Save dataset to JSON format compatible with MultiRuleDataset.
    
    Args:
        dataset: MultiRuleDataset instance
        output_path: Path to save the dataset
        logger: Logger instance
        
    Returns:
        Dictionary with dataset metadata
    """
    data = {
        'metadata': {
            'num_rules': dataset.num_rules,
            'num_problems': len(dataset),
            'split': dataset.split,
            'coeff_range': dataset.coeff_range,
            'd_model': dataset.d_model,
            'valid_rules': dataset.VALID_RULES
        },
        'problems': []
    }
    
    logger.info(f"Extracting {len(dataset)} problems from {dataset.num_rules}-rule dataset...")
    
    for i in range(len(dataset)):
        try:
            # Get the raw problem info
            problem_info = dataset.get_problem_info(i)
            
            # Store problem data in the format expected by MultiRuleDataset
            problem_data = {
                'id': i,
                'input_equation': problem_info['input_equation'],
                'target_equation': problem_info['target_equation'], 
                'rules_applied': problem_info['rules_applied'],
                'solution': problem_info.get('solution', None),
                'trace': problem_info.get('trace', [])
            }
            
            data['problems'].append(problem_data)
            
        except Exception as e:
            logger.warning(f"Failed to extract problem {i}: {e}")
            continue
    
    # Save to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(data['problems'])} problems to {output_path}")
    
    return {
        'file_path': output_path,
        'num_problems': len(data['problems']),
        'num_rules': dataset.num_rules,
        'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
    }


def create_2_rule_dataset(output_dir: str, seed: int, logger: logging.Logger) -> Dict[str, Any]:
    """Create 2-rule test dataset (distribute+combine, 100 samples)."""
    logger.info("Creating 2-rule dataset (distribute+combine, 100 samples)...")
    
    dataset = MultiRuleDataset(
        num_rules=2,
        split='test',
        num_problems=100,
        coeff_range=[-10, 10],
        seed=seed
    )
    
    output_path = os.path.join(output_dir, '2_rule_test_dataset.json')
    return save_dataset(dataset, output_path, logger)


def create_3_rule_dataset(output_dir: str, seed: int, logger: logging.Logger) -> Dict[str, Any]:
    """Create 3-rule test dataset (distribute+combine+isolate, 50 samples)."""
    logger.info("Creating 3-rule dataset (distribute+combine+isolate, 50 samples)...")
    
    dataset = MultiRuleDataset(
        num_rules=3,
        split='test', 
        num_problems=50,
        coeff_range=[-10, 10],
        seed=seed + 1  # Different seed for variety
    )
    
    output_path = os.path.join(output_dir, '3_rule_test_dataset.json')
    return save_dataset(dataset, output_path, logger)


def create_4_rule_dataset(output_dir: str, seed: int, logger: logging.Logger) -> Dict[str, Any]:
    """Create 4-rule test dataset (all rules, 25 samples)."""
    logger.info("Creating 4-rule dataset (all rules, 25 samples)...")
    
    dataset = MultiRuleDataset(
        num_rules=4,
        split='test',
        num_problems=25, 
        coeff_range=[-10, 10],
        seed=seed + 2  # Different seed for variety
    )
    
    output_path = os.path.join(output_dir, '4_rule_test_dataset.json')
    return save_dataset(dataset, output_path, logger)


def analyze_datasets(results: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze the generated datasets and provide summary statistics."""
    total_problems = sum(result['num_problems'] for result in results)
    total_size_mb = sum(result['file_size_mb'] for result in results)
    
    analysis = {
        'total_problems': total_problems,
        'total_file_size_mb': round(total_size_mb, 2),
        'datasets_created': len(results),
        'dataset_breakdown': {
            f"{result['num_rules']}_rules": {
                'num_problems': result['num_problems'],
                'file_path': result['file_path'],
                'file_size_mb': round(result['file_size_mb'], 2)
            }
            for result in results
        }
    }
    
    logger.info(f"Dataset creation complete!")
    logger.info(f"Total problems generated: {total_problems}")
    logger.info(f"Total file size: {total_size_mb:.2f} MB")
    logger.info("Datasets created:")
    for result in results:
        logger.info(f"  {result['num_rules']}-rule: {result['num_problems']} problems ({result['file_size_mb']:.2f} MB)")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Create multi-rule test datasets for benchmarking composition performance"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/test_datasets',
        help='Directory to save generated datasets (default: results/test_datasets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting multi-rule test dataset creation...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create all datasets
        results = []
        
        # 2-rule dataset (distribute+combine, 100 samples)
        result_2 = create_2_rule_dataset(args.output_dir, args.seed, logger)
        results.append(result_2)
        
        # 3-rule dataset (distribute+combine+isolate, 50 samples)
        result_3 = create_3_rule_dataset(args.output_dir, args.seed, logger)
        results.append(result_3)
        
        # 4-rule dataset (all rules, 25 samples)
        result_4 = create_4_rule_dataset(args.output_dir, args.seed, logger)
        results.append(result_4)
        
        # Analyze results
        analysis = analyze_datasets(results, logger)
        
        # Save summary analysis
        summary_path = os.path.join(args.output_dir, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Summary analysis saved to {summary_path}")
        logger.info("Multi-rule test dataset creation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())