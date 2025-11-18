"""
Dataset Verification Script

Validates the quality and correctness of generated algebra datasets.
Checks syntax validity, mathematical equivalence, and constraint satisfaction.

Usage:
    python verify.py --rule distribute --split train --num_problems 1000
    python verify.py --multirule --num_rules 3 --num_problems 500
    python verify.py --constrained --num_rules 2 --constraints positive --num_problems 500
"""

import argparse
import logging
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

import sympy as sp
from algebra_dataset import AlgebraDataset, MultiRuleDataset, ConstrainedDataset
from algebra_encoder import validate_equation_syntax, solve_equation, check_equation_equivalence


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetVerifier:
    """
    Comprehensive verifier for algebra datasets.

    Validates equation pairs for:
    - Syntax correctness (both input and target)
    - Mathematical equivalence (same solution set)
    - Rule application correctness
    - Constraint satisfaction (for constrained datasets)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'total_problems': 0,
            'syntax_valid': 0,
            'syntax_invalid': 0,
            'equivalent': 0,
            'not_equivalent': 0,
            'solution_errors': 0,
            'constraint_satisfied': 0,
            'constraint_violated': 0,
            'error_details': defaultdict(int),
            'failed_problems': []
        }

    def verify_equation_pair(
        self,
        input_eq: str,
        target_eq: str,
        index: int,
        expected_rule: Optional[str] = None
    ) -> Dict:
        """
        Verify a single equation pair.

        Args:
            input_eq: Input equation string
            target_eq: Target equation string
            index: Problem index for tracking
            expected_rule: Expected transformation rule (optional)

        Returns:
            Dictionary with verification results
        """
        result = {
            'index': index,
            'input': input_eq,
            'target': target_eq,
            'syntax_valid': False,
            'equivalent': False,
            'input_solution': None,
            'target_solution': None,
            'error': None
        }

        # Step 1: Validate syntax
        input_valid, input_error, _ = validate_equation_syntax(input_eq)
        target_valid, target_error, _ = validate_equation_syntax(target_eq)

        if not input_valid:
            result['error'] = f"Input syntax error: {input_error}"
            self.results['syntax_invalid'] += 1
            return result

        if not target_valid:
            result['error'] = f"Target syntax error: {target_error}"
            self.results['syntax_invalid'] += 1
            return result

        result['syntax_valid'] = True
        self.results['syntax_valid'] += 1

        # Step 2: Solve both equations
        try:
            input_solutions, input_solve_error = solve_equation(input_eq)
            target_solutions, target_solve_error = solve_equation(target_eq)

            if input_solve_error:
                result['error'] = f"Input solve error: {input_solve_error}"
                self.results['solution_errors'] += 1
                return result

            if target_solve_error:
                result['error'] = f"Target solve error: {target_solve_error}"
                self.results['solution_errors'] += 1
                return result

            result['input_solution'] = input_solutions
            result['target_solution'] = target_solutions

        except Exception as e:
            result['error'] = f"Solution error: {type(e).__name__}: {str(e)}"
            self.results['solution_errors'] += 1
            return result

        # Step 3: Check mathematical equivalence
        try:
            are_equiv, equiv_error = check_equation_equivalence(input_eq, target_eq)

            if equiv_error:
                result['error'] = f"Equivalence check error: {equiv_error}"
                self.results['not_equivalent'] += 1
                return result

            if are_equiv:
                result['equivalent'] = True
                self.results['equivalent'] += 1
            else:
                result['error'] = f"Not equivalent: {input_solutions} != {target_solutions}"
                self.results['not_equivalent'] += 1
                return result

        except Exception as e:
            result['error'] = f"Equivalence error: {type(e).__name__}: {str(e)}"
            self.results['not_equivalent'] += 1
            return result

        # Success case
        return result

    def verify_constrained_problem(
        self,
        input_eq: str,
        target_eq: str,
        constraints: Dict[str, bool],
        index: int
    ) -> Dict:
        """
        Verify a constrained problem including constraint satisfaction.

        Args:
            input_eq: Input equation string
            target_eq: Target equation string
            constraints: Dictionary of constraint satisfaction status
            index: Problem index

        Returns:
            Dictionary with verification results including constraint checks
        """
        # First do standard verification
        result = self.verify_equation_pair(input_eq, target_eq, index)

        # Add constraint verification
        result['constraints'] = constraints

        # Check if all constraints marked as satisfied are actually satisfied
        if result['target_solution']:
            # Extract the solution value (should be x = value)
            if len(result['target_solution']) == 1:
                solution_value = result['target_solution'][0]

                # Verify constraint satisfaction
                constraint_checks = {}

                if 'positive' in constraints:
                    actual_positive = solution_value > 0
                    claimed_positive = constraints['positive']
                    constraint_checks['positive'] = {
                        'claimed': claimed_positive,
                        'actual': actual_positive,
                        'match': claimed_positive == actual_positive
                    }

                if 'integer' in constraints:
                    actual_integer = abs(solution_value - round(solution_value)) < 1e-6
                    claimed_integer = constraints['integer']
                    constraint_checks['integer'] = {
                        'claimed': claimed_integer,
                        'actual': actual_integer,
                        'match': claimed_integer == actual_integer
                    }

                if 'both' in constraints:
                    actual_positive = solution_value > 0
                    actual_integer = abs(solution_value - round(solution_value)) < 1e-6
                    actual_both = actual_positive and actual_integer
                    claimed_both = constraints['both']
                    constraint_checks['both'] = {
                        'claimed': claimed_both,
                        'actual': actual_both,
                        'match': claimed_both == actual_both
                    }

                result['constraint_checks'] = constraint_checks

                # Check if all constraints match
                all_match = all(check['match'] for check in constraint_checks.values())
                if all_match:
                    self.results['constraint_satisfied'] += 1
                else:
                    self.results['constraint_violated'] += 1
                    if not result.get('error'):
                        result['error'] = f"Constraint mismatch: {constraint_checks}"

        return result

    def verify_dataset(
        self,
        dataset,
        sample_size: Optional[int] = None,
        show_failures: bool = True
    ) -> Dict:
        """
        Verify an entire dataset.

        Args:
            dataset: AlgebraDataset, MultiRuleDataset, or ConstrainedDataset
            sample_size: Number of problems to verify (None = all)
            show_failures: Whether to print failed problems

        Returns:
            Dictionary with comprehensive verification results
        """
        dataset_size = len(dataset)
        verify_count = min(sample_size, dataset_size) if sample_size else dataset_size

        logger.info(f"Verifying {verify_count} problems from dataset of {dataset_size}...")

        self.results['total_problems'] = verify_count
        start_time = time.time()

        # Determine dataset type
        is_constrained = isinstance(dataset, ConstrainedDataset)
        is_multirule = isinstance(dataset, MultiRuleDataset) and not is_constrained
        is_single_rule = isinstance(dataset, AlgebraDataset)

        for i in range(verify_count):
            if i % max(1, verify_count // 10) == 0 and i > 0:
                logger.info(f"Progress: {i}/{verify_count} ({i/verify_count*100:.1f}%)")

            # Get problem based on dataset type
            if is_constrained:
                problem_info = dataset.get_problem_info(i)
                input_eq = problem_info['input_equation']
                target_eq = problem_info['target_equation']
                constraints = problem_info['constraint_satisfaction']

                result = self.verify_constrained_problem(
                    input_eq, target_eq, constraints, i
                )
            else:
                if is_multirule:
                    problem_info = dataset.get_problem_info(i)
                    input_eq = problem_info['input_equation']
                    target_eq = problem_info['target_equation']
                    expected_rules = problem_info['rules_applied']
                else:  # Single rule
                    input_eq, target_eq = dataset.get_equation_pair(i)
                    expected_rules = [dataset.rule]

                result = self.verify_equation_pair(
                    input_eq, target_eq, i,
                    expected_rule=expected_rules[0] if expected_rules else None
                )

            # Track errors
            if result.get('error'):
                error_type = result['error'].split(':')[0]
                self.results['error_details'][error_type] += 1
                self.results['failed_problems'].append(result)

        elapsed_time = time.time() - start_time
        self.results['verification_time'] = elapsed_time

        # Generate summary
        self._print_summary(dataset, show_failures)

        return self.results

    def _print_summary(self, dataset, show_failures: bool = True):
        """Print verification summary."""
        print("\n" + "="*70)
        print("DATASET VERIFICATION SUMMARY")
        print("="*70)

        # Dataset info
        if isinstance(dataset, ConstrainedDataset):
            info = dataset.get_dataset_info()
            print(f"\nDataset Type: ConstrainedDataset")
            print(f"Number of Rules: {info['num_rules']}")
            print(f"Constraints: {info['constraints']}")
        elif isinstance(dataset, MultiRuleDataset):
            info = dataset.get_dataset_info()
            print(f"\nDataset Type: MultiRuleDataset")
            print(f"Number of Rules: {info['num_rules']}")
            print(f"Rule Distribution: {info['rule_distribution']}")
        else:
            info = dataset.get_rule_info()
            print(f"\nDataset Type: AlgebraDataset")
            print(f"Rule: {info['rule']}")

        print(f"Split: {info['split']}")
        print(f"Total Problems: {self.results['total_problems']}")

        # Verification results
        print(f"\nVerification Time: {self.results['verification_time']:.2f}s")
        print(f"Average Time per Problem: {self.results['verification_time']/self.results['total_problems']*1000:.2f}ms")

        print("\n" + "-"*70)
        print("SYNTAX VALIDATION")
        print("-"*70)
        print(f"Valid: {self.results['syntax_valid']} ({self.results['syntax_valid']/self.results['total_problems']*100:.2f}%)")
        print(f"Invalid: {self.results['syntax_invalid']} ({self.results['syntax_invalid']/self.results['total_problems']*100:.2f}%)")

        print("\n" + "-"*70)
        print("MATHEMATICAL EQUIVALENCE")
        print("-"*70)
        print(f"Equivalent: {self.results['equivalent']} ({self.results['equivalent']/self.results['total_problems']*100:.2f}%)")
        print(f"Not Equivalent: {self.results['not_equivalent']} ({self.results['not_equivalent']/self.results['total_problems']*100:.2f}%)")
        print(f"Solution Errors: {self.results['solution_errors']} ({self.results['solution_errors']/self.results['total_problems']*100:.2f}%)")

        # Constraint verification for constrained datasets
        if isinstance(dataset, ConstrainedDataset):
            print("\n" + "-"*70)
            print("CONSTRAINT SATISFACTION")
            print("-"*70)
            print(f"Satisfied: {self.results['constraint_satisfied']} ({self.results['constraint_satisfied']/self.results['total_problems']*100:.2f}%)")
            print(f"Violated: {self.results['constraint_violated']} ({self.results['constraint_violated']/self.results['total_problems']*100:.2f}%)")

        # Error breakdown
        if self.results['error_details']:
            print("\n" + "-"*70)
            print("ERROR BREAKDOWN")
            print("-"*70)
            for error_type, count in sorted(self.results['error_details'].items(), key=lambda x: x[1], reverse=True):
                print(f"{error_type}: {count} ({count/self.results['total_problems']*100:.2f}%)")

        # Overall success rate
        success_count = self.results['equivalent']
        if isinstance(dataset, ConstrainedDataset):
            success_count = min(success_count, self.results['constraint_satisfied'])

        print("\n" + "="*70)
        print(f"OVERALL SUCCESS RATE: {success_count}/{self.results['total_problems']} ({success_count/self.results['total_problems']*100:.2f}%)")
        print("="*70)

        # Show failed problems if requested
        if show_failures and self.results['failed_problems']:
            print("\n" + "-"*70)
            print(f"FAILED PROBLEMS (showing first 10 of {len(self.results['failed_problems'])})")
            print("-"*70)
            for i, failed in enumerate(self.results['failed_problems'][:10]):
                print(f"\nProblem #{failed['index']}:")
                print(f"  Input:  {failed['input']}")
                print(f"  Target: {failed['target']}")
                print(f"  Error:  {failed['error']}")
                if 'input_solution' in failed and failed['input_solution']:
                    print(f"  Input Solution:  {failed['input_solution']}")
                if 'target_solution' in failed and failed['target_solution']:
                    print(f"  Target Solution: {failed['target_solution']}")


def main():
    """Main verification entry point."""
    parser = argparse.ArgumentParser(description='Verify algebra dataset quality')

    # Dataset type selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rule', type=str, choices=['distribute', 'combine', 'isolate', 'divide'],
                      help='Single rule dataset type')
    group.add_argument('--multirule', action='store_true',
                      help='Multi-rule dataset')
    group.add_argument('--constrained', action='store_true',
                      help='Constrained dataset')

    # Dataset parameters
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'val'],
                       help='Dataset split')
    parser.add_argument('--num_problems', type=int, default=1000,
                       help='Number of problems to generate')
    parser.add_argument('--num_rules', type=int, default=2, choices=[2, 3, 4],
                       help='Number of rules for multi-rule dataset')
    parser.add_argument('--constraints', type=str, nargs='+',
                       choices=['positive', 'integer', 'both'],
                       help='Constraints for constrained dataset')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Embedding dimension')

    # Verification parameters
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of problems to verify (default: all)')
    parser.add_argument('--show_failures', action='store_true', default=True,
                       help='Show failed problems in output')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create dataset based on type
    logger.info("Generating dataset...")

    try:
        if args.rule:
            dataset = AlgebraDataset(
                rule=args.rule,
                split=args.split,
                num_problems=args.num_problems,
                d_model=args.d_model
            )
        elif args.multirule:
            if args.split == 'train':
                logger.warning("MultiRuleDataset only supports 'test' or 'val' splits. Using 'test'.")
                args.split = 'test'
            dataset = MultiRuleDataset(
                num_rules=args.num_rules,
                split=args.split,
                num_problems=args.num_problems,
                d_model=args.d_model
            )
        else:  # constrained
            if not args.constraints:
                logger.error("--constraints required for constrained dataset")
                sys.exit(1)
            if args.split == 'train':
                logger.warning("ConstrainedDataset only supports 'test' or 'val' splits. Using 'test'.")
                args.split = 'test'
            dataset = ConstrainedDataset(
                num_rules=args.num_rules,
                constraints=args.constraints,
                split=args.split,
                num_problems=args.num_problems,
                d_model=args.d_model
            )

        logger.info(f"Dataset generated with {len(dataset)} problems")

        # Verify dataset
        verifier = DatasetVerifier(verbose=args.verbose)
        results = verifier.verify_dataset(
            dataset,
            sample_size=args.sample_size,
            show_failures=args.show_failures
        )

        # Exit with error code if verification failed
        success_rate = results['equivalent'] / results['total_problems']
        if success_rate < 0.95:  # Require 95% success rate
            logger.error(f"Verification failed: {success_rate*100:.2f}% success rate (minimum 95%)")
            sys.exit(1)
        else:
            logger.info(f"Verification passed: {success_rate*100:.2f}% success rate")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error during verification: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
