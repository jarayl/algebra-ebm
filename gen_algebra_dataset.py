#!/usr/bin/env python3
"""
Generate and save algebra datasets for EBM training.

This script pre-generates algebra datasets and saves them to disk for faster
loading during training. Generates datasets for each rule type (distribute,
combine, isolate, divide) and all splits (train, test, val).

Usage:
    # Generate all datasets with default sizes
    python gen_algebra_dataset.py --all

    # Generate specific rule dataset
    python gen_algebra_dataset.py --rule distribute --split train --size 50000

    # Generate with custom parameters
    python gen_algebra_dataset.py --rule combine --split train --size 100000 --d_model 256
"""

import os
import gzip
import argparse
import pickle
import time
from algebra_dataset import AlgebraDataset, MultiRuleDataset, ConstrainedDataset

parser = argparse.ArgumentParser(description='Generate algebra datasets')

# Dataset selection
parser.add_argument('--all', action='store_true', help='Generate all single-rule datasets')
parser.add_argument('--rule', type=str, choices=['distribute', 'combine', 'isolate', 'divide'],
                   help='Single rule dataset type')
parser.add_argument('--multirule', action='store_true', help='Generate multi-rule datasets')
parser.add_argument('--constrained', action='store_true', help='Generate constrained datasets')

# Dataset parameters
parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'val'],
                   help='Dataset split (default: train)')
parser.add_argument('--size', type=int, default=None,
                   help='Number of problems to generate (default: auto based on split)')
parser.add_argument('--num_rules', type=int, default=2, choices=[2, 3, 4],
                   help='Number of rules for multi-rule dataset (default: 2)')
parser.add_argument('--constraints', type=str, nargs='+',
                   choices=['positive', 'integer', 'both'],
                   help='Constraints for constrained dataset')
parser.add_argument('--d_model', type=int, default=128,
                   help='Embedding dimension (default: 128)')
parser.add_argument('--coeff_range', type=int, nargs=2, default=[-10, 10],
                   help='Coefficient range (default: -10 10)')

# Output options
parser.add_argument('--output_dir', type=str, default='./data/algebra',
                   help='Output directory (default: ./data/algebra)')
parser.add_argument('--compress', action='store_true', default=True,
                   help='Compress with gzip (default: True)')

FLAGS = parser.parse_args()


def get_default_size(split: str, is_multirule: bool = False, is_constrained: bool = False) -> int:
    """Get default dataset size based on split and type."""
    if is_constrained:
        return {'train': 50000, 'test': 5000, 'val': 5000}[split]
    elif is_multirule:
        return {'train': 50000, 'test': 10000, 'val': 10000}[split]
    else:  # single rule
        return {'train': 50000, 'test': 10000, 'val': 10000}[split]


def save_dataset(dataset, filename: str, compress: bool = True):
    """Save dataset to disk with optional compression."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if compress:
        filename = filename + '.gz'
        with gzip.open(filename, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    return filename


def generate_single_rule_dataset(rule: str, split: str, size: int) -> AlgebraDataset:
    """Generate a single-rule algebra dataset."""
    print(f"\nGenerating {rule} dataset ({split} split)...")
    print(f"  Size: {size} problems")
    print(f"  d_model: {FLAGS.d_model}")
    print(f"  Coefficient range: {FLAGS.coeff_range}")

    start_time = time.time()

    dataset = AlgebraDataset(
        rule=rule,
        split=split,
        num_problems=size,
        coeff_range=FLAGS.coeff_range,
        d_model=FLAGS.d_model
    )

    elapsed = time.time() - start_time
    actual_size = len(dataset)
    success_rate = (actual_size / size) * 100 if size > 0 else 0

    print(f"  Generated: {actual_size}/{size} problems ({success_rate:.1f}% success)")
    print(f"  Time: {elapsed:.2f}s ({elapsed/actual_size*1000:.2f}ms per problem)")

    # Get dataset info
    info = dataset.get_rule_info()
    print(f"  Info: {info}")

    return dataset


def generate_multirule_dataset(num_rules: int, split: str, size: int) -> MultiRuleDataset:
    """Generate a multi-rule algebra dataset."""
    print(f"\nGenerating multi-rule dataset ({num_rules} rules, {split} split)...")
    print(f"  Size: {size} problems")
    print(f"  d_model: {FLAGS.d_model}")

    start_time = time.time()

    dataset = MultiRuleDataset(
        num_rules=num_rules,
        split=split,
        num_problems=size,
        coeff_range=FLAGS.coeff_range,
        d_model=FLAGS.d_model
    )

    elapsed = time.time() - start_time
    actual_size = len(dataset)

    print(f"  Generated: {actual_size}/{size} problems")
    print(f"  Time: {elapsed:.2f}s")

    # Get dataset info
    info = dataset.get_dataset_info()
    print(f"  Rule distribution: {info['rule_distribution']}")

    return dataset


def generate_constrained_dataset(
    num_rules: int,
    constraints: list,
    split: str,
    size: int
) -> ConstrainedDataset:
    """Generate a constrained algebra dataset."""
    print(f"\nGenerating constrained dataset ({num_rules} rules, {split} split)...")
    print(f"  Size: {size} problems")
    print(f"  Constraints: {constraints}")
    print(f"  d_model: {FLAGS.d_model}")

    start_time = time.time()

    dataset = ConstrainedDataset(
        num_rules=num_rules,
        constraints=constraints,
        split=split,
        num_problems=size,
        coeff_range=FLAGS.coeff_range,
        d_model=FLAGS.d_model
    )

    elapsed = time.time() - start_time
    actual_size = len(dataset)

    print(f"  Generated: {actual_size}/{size} problems")
    print(f"  Time: {elapsed:.2f}s")

    # Get constraint stats
    stats = dataset.get_constraint_stats()
    print(f"  Constraint satisfaction: {stats}")

    return dataset


def main():
    # Validate arguments
    if not (FLAGS.all or FLAGS.rule or FLAGS.multirule or FLAGS.constrained):
        parser.error("Must specify --all, --rule, --multirule, or --constrained")

    if FLAGS.constrained and not FLAGS.constraints:
        parser.error("--constraints required for constrained datasets")

    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    print("="*70)
    print("ALGEBRA DATASET GENERATION")
    print("="*70)
    print(f"Output directory: {FLAGS.output_dir}")
    print(f"Compression: {FLAGS.compress}")

    # Generate datasets
    if FLAGS.all:
        # Generate all single-rule datasets for all splits
        rules = ['distribute', 'combine', 'isolate', 'divide']
        splits = ['train', 'test', 'val']

        for rule in rules:
            for split in splits:
                size = FLAGS.size if FLAGS.size else get_default_size(split)
                dataset = generate_single_rule_dataset(rule, split, size)

                # Save dataset
                filename = os.path.join(
                    FLAGS.output_dir,
                    f'algebra_{rule}_{split}_{size}.pkl'
                )
                saved_path = save_dataset(dataset, filename, FLAGS.compress)
                print(f"  Saved to: {saved_path}")

        print("\n" + "="*70)
        print(f"All datasets generated successfully!")
        print(f"Total: {len(rules) * len(splits)} datasets")
        print("="*70)

    elif FLAGS.rule:
        # Generate specific rule dataset
        size = FLAGS.size if FLAGS.size else get_default_size(FLAGS.split)
        dataset = generate_single_rule_dataset(FLAGS.rule, FLAGS.split, size)

        # Save dataset
        filename = os.path.join(
            FLAGS.output_dir,
            f'algebra_{FLAGS.rule}_{FLAGS.split}_{size}.pkl'
        )
        saved_path = save_dataset(dataset, filename, FLAGS.compress)
        print(f"\nDataset saved to: {saved_path}")

    elif FLAGS.multirule:
        # Generate multi-rule dataset
        if FLAGS.split == 'train':
            print("Warning: MultiRuleDataset is typically used for evaluation (test/val), not training")

        size = FLAGS.size if FLAGS.size else get_default_size(FLAGS.split, is_multirule=True)
        dataset = generate_multirule_dataset(FLAGS.num_rules, FLAGS.split, size)

        # Save dataset
        filename = os.path.join(
            FLAGS.output_dir,
            f'algebra_multirule_{FLAGS.num_rules}_{FLAGS.split}_{size}.pkl'
        )
        saved_path = save_dataset(dataset, filename, FLAGS.compress)
        print(f"\nDataset saved to: {saved_path}")

    elif FLAGS.constrained:
        # Generate constrained dataset
        if FLAGS.split == 'train':
            print("Warning: ConstrainedDataset is typically used for evaluation (test/val), not training")

        size = FLAGS.size if FLAGS.size else get_default_size(FLAGS.split, is_constrained=True)
        dataset = generate_constrained_dataset(
            FLAGS.num_rules,
            FLAGS.constraints,
            FLAGS.split,
            size
        )

        # Save dataset
        constraints_str = '_'.join(FLAGS.constraints)
        filename = os.path.join(
            FLAGS.output_dir,
            f'algebra_constrained_{FLAGS.num_rules}_{constraints_str}_{FLAGS.split}_{size}.pkl'
        )
        saved_path = save_dataset(dataset, filename, FLAGS.compress)
        print(f"\nDataset saved to: {saved_path}")


if __name__ == '__main__':
    main()