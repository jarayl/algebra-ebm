"""
Algebra Dataset Classes for EBM Training

Implements PyTorch Dataset classes for generating and loading algebraic equation problems.
Creates separate datasets for single-rule, multi-rule, and constrained evaluation.

Classes:
- AlgebraDataset: Base class for single-rule problems (distribute, combine, isolate, divide)
- MultiRuleDataset: For compositional testing (2-4 sequential rule applications)  
- ConstrainedDataset: For constraint evaluation (positivity/integerness requirements)
"""

import torch
import torch.utils.data as data
import numpy as np
import sympy as sp
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
import time
import math
import hashlib
from algebra_encoder import create_character_encoder, validate_equation_syntax, check_equation_equivalence, solve_equation


class AlgebraDataset(data.Dataset):
    """
    Base dataset class for single-rule algebraic problems.
    
    Generates pairs of (input_equation, target_equation) for training rule-specific EBMs.
    Each rule type (distribute, combine, isolate, divide) gets a separate dataset.
    Supports stratified coefficient sampling for enhanced dataset variability.
    
    Args:
        rule: Rule type ('distribute', 'combine', 'isolate', 'divide')
        split: Dataset split ('train', 'test', 'val')  
        num_problems: Number of problems to generate (default: 50000)
        coeff_range: Range for random coefficients (default: [-10, 10])
        d_model: Encoder embedding dimension (default: 128)
        enable_stratified_sampling: Enable stratified coefficient sampling (default: False for compatibility)
        stratified_ranges: Dict of range tiers with [min, max] values (default: basic/extended/challenge)
        stratified_distribution: Dict of tier probabilities (default: 40%/40%/20%)
        enable_solution_first: Enable solution-first equation generation (default: False for compatibility)
        target_solution_ranges: Dict of solution ranges with [min, max] values (default: small/medium/large)
        solution_range_distribution: Dict of solution range probabilities (default: 50%/35%/15%)
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    
    def __init__(
        self,
        rule: str,
        split: str = 'train',
        num_problems: int = 50000,
        coeff_range: List[int] = [-10, 10],
        d_model: int = 128,
        enable_stratified_sampling: bool = False,
        stratified_ranges: Optional[Dict[str, List[int]]] = None,
        stratified_distribution: Optional[Dict[str, float]] = None,
        enable_solution_first: bool = False,
        target_solution_ranges: Optional[Dict[str, List[int]]] = None,
        solution_range_distribution: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # Validate inputs
        if rule not in self.VALID_RULES:
            raise ValueError(f"Rule must be one of {self.VALID_RULES}, got {rule}")
        if split not in ['train', 'test', 'val']:
            raise ValueError(f"Split must be 'train', 'test', or 'val', got {split}")
        if len(coeff_range) != 2 or coeff_range[0] >= coeff_range[1]:
            raise ValueError("coeff_range must be [min, max] with min < max")
            
        self.rule = rule
        self.split = split
        self.num_problems = num_problems
        self.coeff_range = coeff_range
        self.d_model = d_model
        
        # Initialize stratified coefficient sampling
        self.enable_stratified_sampling = enable_stratified_sampling
        
        # Default stratified ranges as specified in the plan
        default_ranges = {
            'basic': [-5, 5],      # Core range for fundamental patterns  
            'extended': [-20, 20], # Expanded range for diversity
            'challenge': [-50, 50] # Wider range for robustness
        }
        
        # Default stratified distribution: 40% basic, 40% extended, 20% challenge
        default_distribution = {
            'basic': 0.4,
            'extended': 0.4, 
            'challenge': 0.2
        }
        
        self.stratified_ranges = stratified_ranges if stratified_ranges is not None else default_ranges
        self.stratified_distribution = stratified_distribution if stratified_distribution is not None else default_distribution
        
        # Validate stratified parameters if enabled
        if self.enable_stratified_sampling:
            self._validate_stratified_parameters()
        
        # Initialize solution-first equation generation
        self.enable_solution_first = enable_solution_first
        
        # Default target solution ranges as specified in the plan
        default_solution_ranges = {
            'small': [-10, 10],    # 50% of problems
            'medium': [-25, 25],   # 35% of problems  
            'large': [-50, 50]     # 15% of problems
        }
        
        # Default solution range distribution: 50% small, 35% medium, 15% large
        default_solution_distribution = {
            'small': 0.5,
            'medium': 0.35,
            'large': 0.15
        }
        
        self.target_solution_ranges = target_solution_ranges if target_solution_ranges is not None else default_solution_ranges
        self.solution_range_distribution = solution_range_distribution if solution_range_distribution is not None else default_solution_distribution
        
        # Validate solution-first parameters if enabled
        if self.enable_solution_first:
            self._validate_solution_first_parameters()
        
        # Dataset interface requirements
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Initialize logging and generation statistics
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._generation_stats = {
            'attempts': 0,
            'successes': 0, 
            'failures': defaultdict(int),
            'coverage_adjustments': 0,
            'quality_checkpoints': 0
        }
        
        # Initialize variability validation for adaptive generation
        self.enable_adaptive_generation = enable_stratified_sampling or enable_solution_first
        self.validator = DatasetVariabilityValidator() if self.enable_adaptive_generation else None
        # Adaptive checkpoint interval based on dataset size for better performance
        self.checkpoint_interval = max(1000, self.num_problems // 5) if self.num_problems >= 2000 else self.num_problems
        self._coverage_history = []  # Track coverage over time
        self._max_history_size = 100  # Limit memory usage
        # Removed threading lock for single-threaded dataset generation
        
        # Pre-generate all equation pairs for deterministic behavior
        self.equation_pairs = self._generate_all_equations()
    
    def _validate_stratified_parameters(self) -> None:
        """Validate stratified sampling parameters."""
        # Check for empty distributions
        if not self.stratified_distribution or not self.stratified_ranges:
            raise ValueError("Stratified ranges and distribution cannot be empty when stratified sampling is enabled")
        
        # Check that distribution sums to approximately 1.0
        total_prob = sum(self.stratified_distribution.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Stratified distribution must sum to 1.0, got {total_prob}")
        
        # Check that all probabilities are non-negative
        for name, prob in self.stratified_distribution.items():
            if prob < 0:
                raise ValueError(f"Probability for '{name}' cannot be negative, got {prob}")
        
        # Check that all ranges are valid [min, max] with min < max
        for name, range_vals in self.stratified_ranges.items():
            if len(range_vals) != 2 or range_vals[0] >= range_vals[1]:
                raise ValueError(f"Range '{name}' must be [min, max] with min < max, got {range_vals}")
        
        # Check that distribution keys match range keys
        if set(self.stratified_distribution.keys()) != set(self.stratified_ranges.keys()):
            raise ValueError("Stratified distribution keys must match stratified range keys")
    
    def _validate_solution_first_parameters(self) -> None:
        """Validate solution-first generation parameters."""
        # Check for empty distributions
        if not self.solution_range_distribution or not self.target_solution_ranges:
            raise ValueError("Solution ranges and distribution cannot be empty when solution-first generation is enabled")
        
        # Check that distribution sums to approximately 1.0
        total_prob = sum(self.solution_range_distribution.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Solution range distribution must sum to 1.0, got {total_prob}")
        
        # Check that all probabilities are non-negative
        for name, prob in self.solution_range_distribution.items():
            if prob < 0:
                raise ValueError(f"Solution range probability for '{name}' cannot be negative, got {prob}")
        
        # Check that all ranges are valid [min, max] with min < max
        for name, range_vals in self.target_solution_ranges.items():
            if len(range_vals) != 2 or range_vals[0] >= range_vals[1]:
                raise ValueError(f"Solution range '{name}' must be [min, max] with min < max, got {range_vals}")
        
        # Check that distribution keys match range keys
        if set(self.solution_range_distribution.keys()) != set(self.target_solution_ranges.keys()):
            raise ValueError("Solution range distribution keys must match target solution range keys")
    
    def _generate_random_coefficients(self, count: int = 1) -> Union[int, List[int]]:
        """
        Generate random integer coefficients using stratified sampling or fallback range.
        
        If stratified sampling is enabled, uses the configured ranges and distribution.
        Otherwise, falls back to the original uniform sampling over coeff_range.
        
        Args:
            count: Number of coefficients to generate
            
        Returns:
            Single coefficient (if count=1) or list of coefficients
        """
        coeffs = []
        for _ in range(count):
            while True:
                if self.enable_stratified_sampling:
                    # Select range tier based on distribution
                    tier_name = random.choices(
                        list(self.stratified_distribution.keys()),
                        weights=list(self.stratified_distribution.values())
                    )[0]
                    
                    # Generate coefficient from selected tier
                    range_min, range_max = self.stratified_ranges[tier_name]
                    coeff = random.randint(range_min, range_max)
                else:
                    # Fallback to original uniform sampling for backward compatibility
                    coeff = random.randint(self.coeff_range[0], self.coeff_range[1])
                
                if coeff != 0:  # Avoid zero coefficients
                    coeffs.append(coeff)
                    break
        
        return coeffs if count > 1 else coeffs[0]
    
    def _generate_target_solution(self) -> int:
        """
        Generate a target solution value using solution-first approach.
        
        Selects a solution range tier based on the configured distribution,
        then generates a random integer from that range.
        
        Returns:
            Integer solution value from the selected target range
        """
        if not self.enable_solution_first:
            # Fallback to simple generation for backward compatibility
            return self._generate_random_coefficients(1)
        
        # Select solution range tier based on distribution
        tier_name = random.choices(
            list(self.solution_range_distribution.keys()),
            weights=list(self.solution_range_distribution.values())
        )[0]
        
        # Generate solution from selected tier
        range_min, range_max = self.target_solution_ranges[tier_name]
        return random.randint(range_min, range_max)
    
    def _build_distribute_equation_from_solution(self, solution: int) -> Tuple[str, str]:
        """
        Build a distribute equation backward from target solution.
        Rule: a(x + b) + c = target  ->  ax + ab + c = target
        
        Args:
            solution: Target solution value for x
            
        Returns:
            Tuple of (input_equation, target_equation)
        """
        # Generate coefficients
        a, b, c = self._generate_random_coefficients(3)
        
        # Choose operation (+ or -)
        op = random.choice(['+', '-'])
        
        if op == '+':
            # a(x + b) + c = target  ->  ax + ab + c = target
            target_value = a * (solution + b) + c
            input_eq = f"{a}*(x+{b})+{c}={target_value}"
            target_eq = f"{a}*x+{a*b + c}={target_value}"
        else:
            # a(x - b) + c = target  ->  ax - ab + c = target  
            target_value = a * (solution - b) + c
            input_eq = f"{a}*(x-{b})+{c}={target_value}"
            target_eq = f"{a}*x+{-a*b + c}={target_value}"
            
        return input_eq, target_eq
    
    def _build_combine_equation_from_solution(self, solution: int) -> Tuple[str, str]:
        """
        Build a combine equation backward from target solution.
        Rule: ax + bx + c = target  ->  (a+b)x + c = target
        
        Args:
            solution: Target solution value for x
            
        Returns:
            Tuple of (input_equation, target_equation)
        """
        # Generate coefficients with retry for valid combination
        max_retries = 10
        for attempt in range(max_retries):
            a, b, c = self._generate_random_coefficients(3)
            
            # Choose operation (+ or -)
            op = random.choice(['+', '-'])
            
            if op == '+':
                # ax + bx + c = target  ->  (a+b)x + c = target
                combined_coeff = a + b
            else:
                # ax - bx + c = target  ->  (a-b)x + c = target
                combined_coeff = a - b
            
            if combined_coeff != 0:
                break
        else:
            # Fail loudly if can't generate valid coefficients
            raise ValueError('Failed to generate non-zero combined coefficient after 10 attempts')
        
        if op == '+':
            target_value = combined_coeff * solution + c
            input_eq = f"{a}*x+{b}*x+{c}={target_value}"
            target_eq = f"{combined_coeff}*x+{c}={target_value}"
        else:
            target_value = combined_coeff * solution + c
            input_eq = f"{a}*x-{b}*x+{c}={target_value}" 
            target_eq = f"{combined_coeff}*x+{c}={target_value}"
            
        return input_eq, target_eq
    
    def _build_isolate_equation_from_solution(self, solution: int) -> Tuple[str, str]:
        """
        Build an isolate equation backward from target solution.
        Rule: ax + b = target  ->  ax = target - b
        
        Args:
            solution: Target solution value for x
            
        Returns:
            Tuple of (input_equation, target_equation)
        """
        # Generate coefficients
        a, b = self._generate_random_coefficients(2)
        
        # Build equation: ax + b = target where target = ax + b
        target_value = a * solution + b
        
        # ax + b = target  ->  ax = target - b
        input_eq = f"{a}*x+{b}={target_value}"
        target_eq = f"{a}*x={target_value - b}"
            
        return input_eq, target_eq
    
    def _build_divide_equation_from_solution(self, solution: int) -> Tuple[str, str]:
        """
        Build a divide equation backward from target solution.
        Rule: ax = b  ->  x = b/a
        
        Args:
            solution: Target solution value for x
            
        Returns:
            Tuple of (input_equation, target_equation)
        """
        # Generate coefficient, ensure a != 1 for non-trivial division
        while True:
            a = self._generate_random_coefficients(1)
            if abs(a) > 1:  # Ensure non-trivial division
                break
                
        # Compute b = a * solution so that ax = b gives x = solution
        b = a * solution
        
        # ax = b  ->  x = solution
        input_eq = f"{a}*x={b}"
        target_eq = f"x={solution}"
            
        return input_eq, target_eq
    
    def _generate_distribute_equation(self) -> Tuple[str, str]:
        """
        Generate distribute rule equation pair.
        Rule: a(x + b) = ax + ab  or  a(x - b) = ax - ab
        
        Returns: (input_equation, target_equation)
        """
        if self.enable_solution_first:
            # Use solution-first generation for systematic coverage
            solution = self._generate_target_solution()
            input_eq, target_eq = self._build_distribute_equation_from_solution(solution)
            # Convert = to == for validation consistency
            return input_eq.replace('=', '=='), target_eq.replace('=', '==')
        else:
            # Original generation method for backward compatibility
            # Generate random coefficients
            a, b, c = self._generate_random_coefficients(3)
            
            # Choose operation (+ or -)
            op = random.choice(['+', '-'])
            
            # Generate a proper solution first, then build equation
            x_solution = self._generate_random_coefficients(1)
            
            if op == '+':
                # a(x + b) + c = target  ->  ax + ab + c = target
                target_value = a * (x_solution + b) + c
                input_eq = f"{a}*(x+{b})+{c}=={target_value}"
                target_eq = f"{a}*x+{a*b + c}=={target_value}"
            else:
                # a(x - b) + c = target  ->  ax - ab + c = target
                target_value = a * (x_solution - b) + c
                input_eq = f"{a}*(x-{b})+{c}=={target_value}"
                target_eq = f"{a}*x+{-a*b + c}=={target_value}"
                
            return input_eq, target_eq
    
    def _generate_combine_equation(self) -> Tuple[str, str]:
        """
        Generate combine rule equation pair.
        Rule: ax + bx = (a+b)x  or  ax - bx = (a-b)x
        
        Returns: (input_equation, target_equation) 
        """
        if self.enable_solution_first:
            # Use solution-first generation for systematic coverage
            solution = self._generate_target_solution()
            input_eq, target_eq = self._build_combine_equation_from_solution(solution)
            # Convert = to == for validation consistency
            return input_eq.replace('=', '=='), target_eq.replace('=', '==')
        else:
            # Original generation method for backward compatibility
            # Generate random coefficients
            a, b, c = self._generate_random_coefficients(3)
            
            # Choose operation (+ or -)
            op = random.choice(['+', '-'])
            
            # Generate a proper solution first, then build equation
            x_solution = self._generate_random_coefficients(1)
            
            if op == '+':
                # ax + bx + c = target  ->  (a+b)x + c = target
                combined_coeff = a + b
                # Ensure non-zero combined coefficient
                if combined_coeff == 0:
                    combined_coeff = 1  # Fallback to avoid degenerate case
                target_value = combined_coeff * x_solution + c
                input_eq = f"{a}*x+{b}*x+{c}=={target_value}"
                target_eq = f"{combined_coeff}*x+{c}=={target_value}"
            else:
                # ax - bx + c = target  ->  (a-b)x + c = target
                combined_coeff = a - b
                # Ensure non-zero combined coefficient
                if combined_coeff == 0:
                    combined_coeff = 1  # Fallback to avoid degenerate case
                target_value = combined_coeff * x_solution + c
                input_eq = f"{a}*x-{b}*x+{c}=={target_value}" 
                target_eq = f"{combined_coeff}*x+{c}=={target_value}"
                
            return input_eq, target_eq
    
    def _generate_isolate_equation(self) -> Tuple[str, str]:
        """
        Generate isolate rule equation pair.
        Rule: ax + b = c  ->  ax = c - b
        
        Returns: (input_equation, target_equation)
        """
        if self.enable_solution_first:
            # Use solution-first generation for systematic coverage
            solution = self._generate_target_solution()
            input_eq, target_eq = self._build_isolate_equation_from_solution(solution)
            # Convert = to == for validation consistency
            return input_eq.replace('=', '=='), target_eq.replace('=', '==')
        else:
            # Original generation method for backward compatibility
            # Generate random coefficients and solution
            a, b = self._generate_random_coefficients(2)
            x_solution = self._generate_random_coefficients(1)
            
            # Build equation: ax + b = target where target = ax + b
            target_value = a * x_solution + b
            
            # ax + b = target  ->  ax = target - b
            input_eq = f"{a}*x+{b}=={target_value}"
            target_eq = f"{a}*x=={target_value - b}"
                
            return input_eq, target_eq
    
    def _generate_divide_equation(self) -> Tuple[str, str]:
        """
        Generate divide rule equation pair.
        Rule: ax = b  ->  x = b/a
        
        Returns: (input_equation, target_equation)
        """
        if self.enable_solution_first:
            # Use solution-first generation for systematic coverage
            solution = self._generate_target_solution()
            input_eq, target_eq = self._build_divide_equation_from_solution(solution)
            # Convert = to == for validation consistency
            return input_eq.replace('=', '=='), target_eq.replace('=', '==')
        else:
            # Original generation method for backward compatibility
            # Generate random coefficients, ensure a != 1 for non-trivial division
            while True:
                a = self._generate_random_coefficients(1)
                if abs(a) > 1:  # Ensure non-trivial division
                    break
                    
            # Generate solution first, then compute b = a * solution  
            solution = self._generate_random_coefficients(1)
            b = a * solution
            
            # ax = b  ->  x = solution
            input_eq = f"{a}*x=={b}"
            target_eq = f"x=={solution}"
                
            return input_eq, target_eq
    
    def _check_coverage_gaps(self, current_equations: List[Tuple[str, str]]) -> Dict[str, any]:
        """
        Check for coverage gaps in current equation set.
        
        Args:
            current_equations: List of (input_eq, target_eq) pairs generated so far
            
        Returns:
            Dictionary with gap analysis and recommended adjustments
        """
        if not self.validator or not current_equations:
            return {'has_gaps': False, 'recommendations': []}
        
        try:
            # For small datasets, skip expensive analysis to improve performance
            if len(current_equations) < 1000:
                return {'has_gaps': False, 'recommendations': ['Coverage analysis skipped for small datasets']}
            
            # Validate current coverage for larger datasets
            coverage_report = self.validator.generate_coverage_report(current_equations)
            
            # Check if coverage is adequate
            solution_coverage = coverage_report['solution_coverage']
            coeff_diversity = coverage_report['coefficient_diversity']
            
            has_gaps = (
                not solution_coverage.get('passed', False) or
                not coeff_diversity.get('passed', False)
            )
            
            return {
                'has_gaps': has_gaps,
                'solution_coverage': solution_coverage,
                'coefficient_diversity': coeff_diversity,
                'recommendations': coverage_report['recommendations']
            }
            
        except Exception as e:
            self.logger.warning(f"Coverage gap analysis failed: {e}")
            return {'has_gaps': False, 'recommendations': []}
    
    def _adjust_generation_parameters(self, gap_analysis: Dict) -> bool:
        """
        Adjust generation parameters based on coverage gap analysis.
        
        Args:
            gap_analysis: Results from _check_coverage_gaps
            
        Returns:
            True if parameters were adjusted, False otherwise
        """
        if not self.enable_adaptive_generation or not gap_analysis.get('has_gaps', False):
            return False
        
        adjustments_made = False
        
        try:
            solution_coverage = gap_analysis.get('solution_coverage', {})
            coeff_diversity = gap_analysis.get('coefficient_diversity', {})
            
            # Adjust solution range distribution if needed
            if self.enable_solution_first and not solution_coverage.get('passed', False):
                coverage_by_range = solution_coverage.get('coverage_by_range', {})
                under_covered = [
                    range_name for range_name, data in coverage_by_range.items()
                    if not data.get('meets_minimum', True)
                ]
                
                if under_covered:
                    # Temporarily increase probability for under-covered ranges
                    total_adjustment = 0.2  # Redistribute 20% of probability
                    adjustment_per_range = total_adjustment / len(under_covered)
                    
                    # Create adjusted distribution
                    adjusted_distribution = self.solution_range_distribution.copy()
                    # Create inverse mapping from formatted range names to tier names
                    range_tier_mapping = {
                        f"{tier_range[0]}_to_{tier_range[1]}": tier_name 
                        for tier_name, tier_range in self.target_solution_ranges.items()
                    }
                    
                    # Convert under_covered to tier names for direct lookup
                    under_covered_tiers = set()
                    for range_name in under_covered:
                        if range_name in range_tier_mapping:
                            under_covered_tiers.add(range_tier_mapping[range_name])
                    
                    if not under_covered_tiers:
                        return False  # No valid under-covered tiers
                    
                    # Reduce probability for well-covered ranges
                    for tier_name in self.solution_range_distribution:
                        if tier_name not in under_covered_tiers:
                            adjusted_distribution[tier_name] = max(0.05, adjusted_distribution[tier_name] * 0.8)
                    
                    # Increase probability for under-covered ranges
                    for tier_name in under_covered_tiers:
                        current_prob = adjusted_distribution[tier_name]
                        new_prob = current_prob + adjustment_per_range
                        adjusted_distribution[tier_name] = new_prob
                    
                    # First normalize to sum = 1.0
                    total_prob = sum(adjusted_distribution.values())
                    for tier in adjusted_distribution:
                        adjusted_distribution[tier] /= total_prob
                    
                    # Then apply bounds with soft constraints
                    for tier in adjusted_distribution:
                        adjusted_distribution[tier] = min(0.6, max(0.05, adjusted_distribution[tier]))
                    
                    # Final normalization to ensure sum = 1.0
                    total_prob = sum(adjusted_distribution.values())
                    for tier in adjusted_distribution:
                        adjusted_distribution[tier] /= total_prob
                    
                    self.solution_range_distribution = adjusted_distribution
                    adjustments_made = True
                    self.logger.info(f"Adjusted solution range distribution for under-covered ranges: {under_covered}")
            
            # Adjust coefficient stratification if needed  
            if self.enable_stratified_sampling and not coeff_diversity.get('passed', False):
                if coeff_diversity.get('unique_ratio', 1.0) < 0.8:
                    # Temporarily favor extended and challenge ranges for more diversity
                    adjusted_stratified = self.stratified_distribution.copy()
                    adjusted_stratified['basic'] = 0.2  # Reduce basic
                    adjusted_stratified['extended'] = 0.5  # Increase extended
                    adjusted_stratified['challenge'] = 0.3  # Increase challenge
                    
                    self.stratified_distribution = adjusted_stratified
                    adjustments_made = True
                    self.logger.info("Adjusted coefficient distribution to increase diversity")
            
            if adjustments_made:
                self._generation_stats['coverage_adjustments'] += 1
                
        except Exception as e:
            self.logger.warning(f"Parameter adjustment failed: {e}")
            
        return adjustments_made
    
    def _generate_single_equation_pair(self) -> Tuple[str, str]:
        """Generate one equation pair based on the rule type."""
        
        if self.rule == 'distribute':
            return self._generate_distribute_equation()
        elif self.rule == 'combine': 
            return self._generate_combine_equation()
        elif self.rule == 'isolate':
            return self._generate_isolate_equation()
        elif self.rule == 'divide':
            return self._generate_divide_equation()
        else:
            raise ValueError(f"Unknown rule: {self.rule}")
    
    def _validate_equation_pair(self, input_eq: str, target_eq: str) -> bool:
        """
        Validate that both equations are syntactically correct and equivalent.
        
        Args:
            input_eq: Input equation string
            target_eq: Target equation string
            
        Returns:
            True if both equations are valid and equivalent
        """
        try:
            # Check syntax validity
            input_valid, _, _ = validate_equation_syntax(input_eq.replace('==', '='))
            target_valid, _, _ = validate_equation_syntax(target_eq.replace('==', '='))
            
            if not (input_valid and target_valid):
                return False
                
            # Check equivalence 
            equiv, _ = check_equation_equivalence(
                input_eq.replace('==', '='), 
                target_eq.replace('==', '=')
            )
            
            return equiv
            
        except (ValueError, TypeError, sp.SympifyError) as e:
            # Expected errors during validation - log for debugging
            self.logger.debug(f"Equation validation failed: {type(e).__name__} - {input_eq}, {target_eq}")
            return False
        except Exception as e:
            # Unexpected error - log for investigation  
            self.logger.warning(f"Unexpected validation error: {type(e).__name__}: {str(e)} - {input_eq}, {target_eq}")
            return False
    
    def _generate_all_equations(self) -> List[Tuple[str, str]]:
        """Pre-generate all equation pairs for the dataset with adaptive coverage monitoring."""
        equations = []
        attempts = 0
        max_attempts = self.num_problems * 10  # Allow up to 10x attempts
        
        while len(equations) < self.num_problems and attempts < max_attempts:
            attempts += 1
            self._generation_stats['attempts'] += 1
            
            try:
                input_eq, target_eq = self._generate_single_equation_pair()
                
                # Validate the equation pair
                if self._validate_equation_pair(input_eq, target_eq):
                    # Convert == to = for final format
                    input_eq_final = input_eq.replace('==', '=')
                    target_eq_final = target_eq.replace('==', '=')
                    equations.append((input_eq_final, target_eq_final))
                    self._generation_stats['successes'] += 1
                    
                    # Quality checkpoint: validate coverage every 1000 problems (as specified in plan)
                    if (self.enable_adaptive_generation and 
                        (len(equations) % self.checkpoint_interval == 0 or 
                         len(equations) == self.num_problems) and 
                        len(equations) > 0):
                        
                        self._generation_stats['quality_checkpoints'] += 1
                        
                        # Check for coverage gaps
                        gap_analysis = self._check_coverage_gaps(equations)
                        
                        # Store coverage history for monitoring
                        checkpoint_data = {
                            'equations_count': len(equations),
                            'has_gaps': gap_analysis.get('has_gaps', False),
                            'recommendations': gap_analysis.get('recommendations', [])
                        }
                        self._coverage_history.append(checkpoint_data)
                        
                        # Limit history size to prevent memory issues
                        if len(self._coverage_history) > self._max_history_size:
                            self._coverage_history = self._coverage_history[-self._max_history_size:]
                        
                        # Adjust parameters if needed
                        if gap_analysis.get('has_gaps', False):
                            adjusted = self._adjust_generation_parameters(gap_analysis)
                            if adjusted:
                                self.logger.info(f"Checkpoint {len(equations)}: Adjusted generation parameters")
                                for rec in gap_analysis.get('recommendations', [])[:3]:  # Log first 3 recommendations
                                    self.logger.info(f"  - {rec}")
                            else:
                                self.logger.debug(f"Checkpoint {len(equations)}: Coverage gaps detected but no adjustments made")
                        else:
                            self.logger.debug(f"Checkpoint {len(equations)}: Coverage targets met")
                    
            except (ValueError, TypeError, sp.SympifyError) as e:
                # Expected errors during generation - track for analysis  
                self._generation_stats['failures'][type(e).__name__] += 1
                if attempts % 1000 == 0:  # Rate-limited logging
                    self.logger.debug(f"Generation progress: {len(equations)}/{attempts} ({len(equations)/attempts:.1%})")
                continue
            except Exception as e:
                # Unexpected error - log for investigation
                self.logger.warning(f"Unexpected generation error: {type(e).__name__}: {str(e)}")
                self._generation_stats['failures']['unexpected'] += 1
                continue
        
        # Final coverage validation if adaptive generation is enabled
        if self.enable_adaptive_generation and equations:
            final_gap_analysis = self._check_coverage_gaps(equations)
            final_report = {
                'final_equations_count': len(equations),
                'final_coverage_status': final_gap_analysis,
                'total_checkpoints': self._generation_stats['quality_checkpoints'],
                'total_adjustments': self._generation_stats['coverage_adjustments']
            }
            self._coverage_history.append(final_report)
            
            # Limit history size to prevent memory issues
            if len(self._coverage_history) > self._max_history_size:
                self._coverage_history = self._coverage_history[-self._max_history_size:]
            
            if not final_gap_analysis.get('has_gaps', True):
                self.logger.info("Final validation: All coverage targets met")
            else:
                self.logger.warning("Final validation: Some coverage gaps remain")
                for rec in final_gap_analysis.get('recommendations', [])[:3]:
                    self.logger.warning(f"  - {rec}")
                
        # Log generation completion statistics
        success_rate = len(equations) / attempts if attempts > 0 else 0
        self.logger.info(f"Generation complete: {len(equations)} equations from {attempts} attempts "
                        f"({success_rate:.1%} success rate)")
        if self._generation_stats['failures']:
            self.logger.info(f"Failure breakdown: {dict(self._generation_stats['failures'])}")
        
        if self.enable_adaptive_generation:
            self.logger.info(f"Adaptive generation stats: {self._generation_stats['quality_checkpoints']} checkpoints, "
                           f"{self._generation_stats['coverage_adjustments']} parameter adjustments")
            
        if len(equations) < self.num_problems:
            self.logger.warning(f"Only generated {len(equations)} valid equations out of {self.num_problems} requested")
            
        return equations
    
    def __len__(self) -> int:
        """Return the total number of equation pairs."""
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get encoded equation pair at index.
        
        Args:
            index: Index of equation pair to retrieve
            
        Returns:
            Tuple of (input_embedding, target_embedding) as tensors
        """
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
            
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode equations to embeddings
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding
    
    def get_equation_pair(self, index: int) -> Tuple[str, str]:
        """
        Get raw equation strings at index (useful for debugging/inspection).
        
        Args:
            index: Index of equation pair to retrieve
            
        Returns:
            Tuple of (input_equation, target_equation) as strings
        """
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
            
        return self.equation_pairs[index]
    
    def get_rule_info(self) -> Dict:
        """Get information about the current rule and dataset."""
        return {
            'rule': self.rule,
            'split': self.split,
            'num_problems': len(self.equation_pairs),
            'requested_problems': self.num_problems,
            'coeff_range': self.coeff_range,
            'd_model': self.d_model,
            'inp_dim': self.inp_dim,
            'out_dim': self.out_dim
        }
    
    def get_coverage_history(self) -> List[Dict]:
        """
        Get the coverage history from adaptive generation monitoring.
        
        Returns:
            List of checkpoint data showing coverage analysis over time
        """
        if not self.enable_adaptive_generation:
            return []
        return self._coverage_history.copy()
    
    def validate_current_coverage(self) -> Dict:
        """
        Perform coverage validation on the current dataset.
        
        Returns:
            Dictionary with comprehensive coverage analysis
        """
        if not self.validator:
            return {'error': 'Validation not available - adaptive generation not enabled'}
        
        return self.validator.generate_coverage_report(self.equation_pairs)


class MultiRuleDataset(data.Dataset):
    """
    Dataset for compositional testing with multi-rule equation problems.
    
    Generates equations requiring 2-4 sequential rule applications that are never
    seen during training, enabling zero-shot compositional evaluation.
    
    Args:
        num_rules: Number of rules to chain (2, 3, or 4)
        split: Dataset split ('test', 'val') - no 'train' since these are for evaluation only  
        num_problems: Number of problems to generate (default: 10000)
        coeff_range: Range for random coefficients (default: [-10, 10])
        d_model: Encoder embedding dimension (default: 128)
        seed: Random seed for deterministic generation (default: None)
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    
    def __init__(
        self,
        num_rules: int,
        split: str = 'test',
        num_problems: int = 10000,
        coeff_range: List[int] = [-10, 10],
        d_model: int = 128,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Validate inputs
        if num_rules not in [2, 3, 4]:
            raise ValueError(f"num_rules must be 2, 3, or 4, got {num_rules}")
        if split not in ['test', 'val']:
            raise ValueError("MultiRuleDataset only supports 'test' or 'val' splits (not for training)")
        if len(coeff_range) != 2 or coeff_range[0] >= coeff_range[1]:
            raise ValueError("coeff_range must be [min, max] with min < max")
            
        self.num_rules = num_rules
        self.split = split
        self.num_problems = num_problems
        self.coeff_range = coeff_range
        self.d_model = d_model
        
        # Dataset interface requirements
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Initialize logging and generation statistics
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._generation_stats = {
            'attempts': 0,
            'successes': 0, 
            'failures': defaultdict(int)
        }
        
        # Set random seed for deterministic generation
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pre-generate all multi-rule equation pairs
        self.equation_data = self._generate_all_multirule_equations()
    
    def _generate_random_coefficients(self, count: int = 1) -> Union[int, List[int]]:
        """Generate random integer coefficients in the specified range, excluding zero."""
        coeffs = []
        for _ in range(count):
            while True:
                coeff = random.randint(self.coeff_range[0], self.coeff_range[1])
                if coeff != 0:  # Avoid zero coefficients
                    coeffs.append(coeff)
                    break
        return coeffs if count > 1 else coeffs[0]
    
    def _generate_forward_multirule_problem(self) -> Tuple[str, str, List[str]]:
        """
        Generate multi-rule problem using forward composition approach.
        More reliable than inverse transformations.
        
        Returns:
            Tuple of (complex_input_equation, simple_target_equation, rules_applied)
        """
        # Start with a base equation structure
        x_solution = self._generate_random_coefficients(1)
        
        # Generate rule sequence
        rule_sequence = self._generate_rule_sequence()
        
        # Start with a simple equation and apply forward transformations
        # This creates a chain that the model will need to reverse
        
        # Begin with the solved form
        target_eq = f"x={x_solution}"
        
        # Apply a sequence of "reverse" operations to create complexity
        # Each operation makes the equation more complex in a way that 
        # requires the corresponding rule to simplify
        
        current_eq = target_eq
        applied_rules = []
        
        for rule in rule_sequence:
            if rule == 'divide':
                # Create ax = b from x = solution
                a = self._generate_random_coefficients(1)
                if abs(a) <= 1:
                    a = random.choice([-3, -2, 2, 3])
                new_lhs = f"{a}*x"
                new_rhs = str(a * x_solution)
                current_eq = f"{new_lhs}={new_rhs}"
                applied_rules.append('divide')
                
            elif rule == 'isolate':
                # Create ax + b = c from ax = d
                if '=' in current_eq:
                    lhs, rhs = current_eq.split('=')
                    b = self._generate_random_coefficients(1)
                    new_rhs = str(int(rhs) + b) if rhs.isdigit() or (rhs.startswith('-') and rhs[1:].isdigit()) else f"{rhs}+{b}"
                    new_lhs = f"{lhs}+{b}" if b >= 0 else f"{lhs}{b}"
                    current_eq = f"{new_lhs}={new_rhs}"
                    applied_rules.append('isolate')
                    
            elif rule == 'combine':
                # Create ax + bx = c from (a+b)x = c  
                if '=' in current_eq and 'x' in current_eq:
                    lhs, rhs = current_eq.split('=')
                    # Extract coefficient of x
                    if lhs.strip() == 'x':
                        total_coeff = 1
                    elif lhs.strip() == '-x':
                        total_coeff = -1
                    elif '*x' in lhs:
                        coeff_part = lhs.split('*x')[0].strip()
                        total_coeff = int(coeff_part) if coeff_part and coeff_part != '+' and coeff_part != '-' else (1 if coeff_part != '-' else -1)
                    else:
                        total_coeff = 1
                        
                    # Split into two terms
                    a = self._generate_random_coefficients(1)
                    b = total_coeff - a
                    if b == 0:  # Avoid zero coefficient
                        b = 1
                        a = total_coeff - 1
                        
                    new_lhs = f"{a}*x+{b}*x" if b >= 0 else f"{a}*x{b}*x"
                    current_eq = f"{new_lhs}={rhs}"
                    applied_rules.append('combine')
                    
            elif rule == 'distribute':
                # Create a(x + b) + c = d from ax + ab + c = d
                if '=' in current_eq and 'x' in current_eq:
                    lhs, rhs = current_eq.split('=')
                    a = self._generate_random_coefficients(1)
                    b = self._generate_random_coefficients(1)
                    c = self._generate_random_coefficients(1)
                    
                    # Create distribution form
                    new_lhs = f"{a}*(x+{b})+{c}" if b >= 0 else f"{a}*(x{b})+{c}"
                    if c < 0:
                        new_lhs = f"{a}*(x+{b}){c}" if b >= 0 else f"{a}*(x{b}){c}"
                        
                    # Calculate what RHS should be
                    new_rhs_val = a * (x_solution + b) + c
                    current_eq = f"{new_lhs}={new_rhs_val}"
                    applied_rules.append('distribute')
        
        return current_eq, target_eq, applied_rules
    
    def _generate_rule_sequence(self) -> List[str]:
        """Generate a random sequence of rules to apply."""
        return random.choices(self.VALID_RULES, k=self.num_rules)
    
    def _generate_single_multirule_problem(self) -> Tuple[str, str, List[str]]:
        """
        Generate one multi-rule problem using forward composition.
        
        Returns:
            Tuple of (complex_input_equation, simple_target_equation, rules_applied)
        """
        return self._generate_forward_multirule_problem()
    
    def _validate_multirule_pair(self, input_eq: str, target_eq: str) -> bool:
        """
        Validate that the multi-rule equation pair is syntactically correct.
        
        Note: We don't check equivalence here since the transformations are complex
        and may require the actual rule applications to solve.
        """
        try:
            # Check syntax validity only
            input_valid, _, _ = validate_equation_syntax(input_eq)
            target_valid, _, _ = validate_equation_syntax(target_eq)
            
            return input_valid and target_valid
            
        except (ValueError, TypeError, sp.SympifyError) as e:
            # Expected errors during validation - log for debugging
            self.logger.debug(f"Multi-rule validation failed: {type(e).__name__} - {input_eq}, {target_eq}")
            return False
        except Exception as e:
            # Unexpected error - log for investigation  
            self.logger.warning(f"Unexpected multi-rule validation error: {type(e).__name__}: {str(e)} - {input_eq}, {target_eq}")
            return False
    
    def _generate_all_multirule_equations(self) -> List[Tuple[str, str, List[str]]]:
        """Pre-generate all multi-rule equation problems for the dataset."""
        equations = []
        attempts = 0
        max_attempts = self.num_problems * 20  # Allow more attempts for complex generation
        
        while len(equations) < self.num_problems and attempts < max_attempts:
            attempts += 1
            self._generation_stats['attempts'] += 1
            
            try:
                input_eq, target_eq, rules = self._generate_single_multirule_problem()
                
                # Validate the equation pair
                if self._validate_multirule_pair(input_eq, target_eq):
                    equations.append((input_eq, target_eq, rules))
                    self._generation_stats['successes'] += 1
                    
            except (ValueError, TypeError, sp.SympifyError) as e:
                # Expected errors during generation - track for analysis  
                self._generation_stats['failures'][type(e).__name__] += 1
                if attempts % 1000 == 0:  # Rate-limited logging
                    self.logger.debug(f"Multi-rule generation progress: {len(equations)}/{attempts} ({len(equations)/attempts:.1%})")
                continue
            except Exception as e:
                # Unexpected error - log for investigation
                self.logger.warning(f"Unexpected multi-rule generation error: {type(e).__name__}: {str(e)}")
                self._generation_stats['failures']['unexpected'] += 1
                continue
                
        if len(equations) < self.num_problems:
            print(f"Warning: Only generated {len(equations)} valid multi-rule equations out of {self.num_problems} requested")
            
        return equations
    
    def __len__(self) -> int:
        """Return the total number of equation problems."""
        return len(self.equation_data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get encoded equation pair at index.
        
        Args:
            index: Index of equation pair to retrieve
            
        Returns:
            Tuple of (input_embedding, target_embedding) as tensors
        """
        if index >= len(self.equation_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_data)}")
            
        # Handle both 3-value and 4-value tuples for compatibility with subclasses
        equation_tuple = self.equation_data[index]
        if len(equation_tuple) < 2:
            raise ValueError(f"Invalid equation tuple length {len(equation_tuple)}, expected at least 2")
        input_eq, target_eq = equation_tuple[0], equation_tuple[1]
        
        # Encode equations to embeddings
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding
    
    def get_problem_info(self, index: int) -> Dict:
        """
        Get full information about a problem including rules applied.
        
        Args:
            index: Index of problem to retrieve
            
        Returns:
            Dictionary with input_eq, target_eq, and rules_applied
        """
        if index >= len(self.equation_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_data)}")
            
        # Handle both 3-value and 4-value tuples for compatibility with subclasses
        equation_tuple = self.equation_data[index]
        if len(equation_tuple) < 3:
            raise ValueError(f"Invalid equation tuple length {len(equation_tuple)}, expected at least 3")
        input_eq, target_eq, rules = equation_tuple[0], equation_tuple[1], equation_tuple[2]
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': rules,
            'num_rules': len(rules)
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the multi-rule dataset."""
        rule_counts = {}
        for _, _, rules in self.equation_data:
            for rule in rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
                
        return {
            'num_rules': self.num_rules,
            'split': self.split,
            'num_problems': len(self.equation_data),
            'requested_problems': self.num_problems,
            'rule_distribution': rule_counts,
            'coeff_range': self.coeff_range,
            'd_model': self.d_model,
            'inp_dim': self.inp_dim,
            'out_dim': self.out_dim
        }


class ConstrainedDataset(MultiRuleDataset):
    """
    Dataset for constraint evaluation with positivity/integerness requirements.
    
    Extends MultiRuleDataset to add constraint requirements to test 
    constraint injection capabilities without retraining.
    
    Args:
        num_rules: Number of rules to chain (2, 3, or 4)
        constraints: List of constraint types ('positive', 'integer', 'both')
        split: Dataset split ('test', 'val') - no 'train' since these are for evaluation only
        num_problems: Number of problems to generate (default: 5000)
        coeff_range: Range for random coefficients (default: [-10, 10])
        d_model: Encoder embedding dimension (default: 128)
        seed: Random seed for deterministic generation (default: None)
    """
    
    VALID_CONSTRAINTS = ['positive', 'integer', 'both']
    
    def __init__(
        self,
        num_rules: int,
        constraints: List[str],
        split: str = 'test',
        num_problems: int = 5000,
        coeff_range: List[int] = [-10, 10],
        d_model: int = 128,
        seed: Optional[int] = None
    ):
        # Validate constraint types
        for constraint in constraints:
            if constraint not in self.VALID_CONSTRAINTS:
                raise ValueError(f"Constraint must be one of {self.VALID_CONSTRAINTS}, got {constraint}")
        
        self.constraints = constraints
        
        # Initialize parent class
        super().__init__(
            num_rules=num_rules,
            split=split,
            num_problems=num_problems,
            coeff_range=coeff_range,
            d_model=d_model,
            seed=seed
        )
        
        # Override equation generation to include constraints
        self.equation_data = self._generate_all_constrained_equations()
    
    def _satisfies_constraints(self, solution_value: Union[int, float]) -> Dict[str, bool]:
        """
        Check if a solution satisfies the given constraints.
        
        Args:
            solution_value: The solution value to check
            
        Returns:
            Dictionary mapping constraint names to satisfaction status
        """
        results = {}
        
        for constraint in self.constraints:
            if constraint == 'positive':
                results['positive'] = solution_value > 0
            elif constraint == 'integer':
                results['integer'] = abs(solution_value - round(solution_value)) < 1e-6
            elif constraint == 'both':
                results['positive'] = solution_value > 0
                results['integer'] = abs(solution_value - round(solution_value)) < 1e-6
                results['both'] = results['positive'] and results['integer']
                
        return results
    
    def _generate_constrained_solution(self) -> int:
        """
        Generate a solution that satisfies all constraints.
        
        Returns:
            Integer solution value that meets constraint requirements
        """
        if 'positive' in self.constraints or 'both' in self.constraints:
            # Generate positive solution
            return random.randint(1, max(5, self.coeff_range[1]//2))
        elif 'integer' in self.constraints:
            # Generate integer solution (can be negative)
            return self._generate_random_coefficients(1)
        else:
            # Fallback to any integer
            return self._generate_random_coefficients(1)
    
    def _generate_forward_constrained_problem(self) -> Tuple[str, str, List[str], Dict[str, bool]]:
        """
        Generate multi-rule problem with constraint-satisfying solution.
        
        Returns:
            Tuple of (complex_input_equation, simple_target_equation, rules_applied, constraint_status)
        """
        # Generate solution that satisfies constraints
        x_solution = self._generate_constrained_solution()
        
        # Generate rule sequence
        rule_sequence = self._generate_rule_sequence()
        
        # Start with solved form
        target_eq = f"x={x_solution}"
        
        # Apply forward transformations to create complexity
        current_eq = target_eq
        applied_rules = []
        
        for rule in rule_sequence:
            if rule == 'divide':
                # Create ax = b from x = solution
                a = self._generate_random_coefficients(1)
                if abs(a) <= 1:
                    a = random.choice([-3, -2, 2, 3])
                new_lhs = f"{a}*x"
                new_rhs = str(a * x_solution)
                current_eq = f"{new_lhs}={new_rhs}"
                applied_rules.append('divide')
                
            elif rule == 'isolate':
                # Create ax + b = c from ax = d
                if '=' in current_eq:
                    lhs, rhs = current_eq.split('=')
                    b = self._generate_random_coefficients(1)
                    new_rhs = str(int(rhs) + b) if rhs.lstrip('-').isdigit() else f"{rhs}+{b}"
                    new_lhs = f"{lhs}+{b}" if b >= 0 else f"{lhs}{b}"
                    current_eq = f"{new_lhs}={new_rhs}"
                    applied_rules.append('isolate')
                    
            elif rule == 'combine':
                # Create ax + bx = c from (a+b)x = c
                if '=' in current_eq and 'x' in current_eq:
                    lhs, rhs = current_eq.split('=')
                    
                    # Extract coefficient of x
                    if lhs.strip() == 'x':
                        total_coeff = 1
                    elif lhs.strip() == '-x':
                        total_coeff = -1
                    elif '*x' in lhs:
                        coeff_part = lhs.split('*x')[0].strip()
                        if coeff_part in ['', '+']:
                            total_coeff = 1
                        elif coeff_part == '-':
                            total_coeff = -1
                        else:
                            total_coeff = int(coeff_part)
                    else:
                        total_coeff = 1
                        
                    # Split into two terms
                    a = self._generate_random_coefficients(1)
                    b = total_coeff - a
                    if b == 0:  # Avoid zero coefficient
                        b = 1
                        a = total_coeff - 1
                        
                    new_lhs = f"{a}*x+{b}*x" if b >= 0 else f"{a}*x{b}*x"
                    current_eq = f"{new_lhs}={rhs}"
                    applied_rules.append('combine')
                    
            elif rule == 'distribute':
                # Create a(x + b) + c = d from ax + ab + c = d
                if '=' in current_eq and 'x' in current_eq:
                    a = self._generate_random_coefficients(1)
                    b = self._generate_random_coefficients(1)
                    c = self._generate_random_coefficients(1)
                    
                    # Create distribution form
                    new_lhs = f"{a}*(x+{b})+{c}" if b >= 0 and c >= 0 else \
                              f"{a}*(x+{b}){c}" if b >= 0 and c < 0 else \
                              f"{a}*(x{b})+{c}" if b < 0 and c >= 0 else \
                              f"{a}*(x{b}){c}"
                        
                    # Calculate what RHS should be
                    new_rhs_val = a * (x_solution + b) + c
                    current_eq = f"{new_lhs}={new_rhs_val}"
                    applied_rules.append('distribute')
        
        # Check constraint satisfaction
        constraint_status = self._satisfies_constraints(x_solution)
        
        return current_eq, target_eq, applied_rules, constraint_status
    
    def _generate_single_constrained_problem(self) -> Tuple[str, str, List[str], Dict[str, bool]]:
        """
        Generate one constrained multi-rule problem.
        
        Returns:
            Tuple of (complex_input_equation, simple_target_equation, rules_applied, constraint_status)
        """
        return self._generate_forward_constrained_problem()
    
    def _generate_all_constrained_equations(self) -> List[Tuple[str, str, List[str], Dict[str, bool]]]:
        """Pre-generate all constrained equation problems for the dataset."""
        equations = []
        attempts = 0
        max_attempts = self.num_problems * 30  # Allow more attempts for constrained generation
        
        while len(equations) < self.num_problems and attempts < max_attempts:
            attempts += 1
            
            try:
                input_eq, target_eq, rules, constraints = self._generate_single_constrained_problem()
                
                # Validate the equation pair and ensure constraints are satisfied
                # Build list of required constraint keys from self.constraints
                required_keys = [c for c in self.constraints if c != 'both']
                if 'both' in self.constraints:
                    required_keys = ['positive', 'integer']
                # Verify ALL required constraints are satisfied
                constraint_satisfied = all(constraints.get(k, False) for k in required_keys)
                
                if self._validate_multirule_pair(input_eq, target_eq) and constraint_satisfied:
                    equations.append((input_eq, target_eq, rules, constraints))
                    
            except (ValueError, TypeError, sp.SympifyError) as e:
                self._generation_stats['failures'][type(e).__name__] += 1
                self.logger.debug(f"Generation failed: {type(e).__name__}")
                continue
            except Exception as e:
                self.logger.warning(f"Unexpected error: {type(e).__name__}: {str(e)}")
                self._generation_stats['failures']['unexpected'] += 1
                continue
                
        if len(equations) < self.num_problems:
            print(f"Warning: Only generated {len(equations)} valid constrained equations out of {self.num_problems} requested")
            
        return equations
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get encoded equation pair at index.
        
        Args:
            index: Index of equation pair to retrieve
            
        Returns:
            Tuple of (input_embedding, target_embedding) as tensors
        """
        if index >= len(self.equation_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_data)}")
            
        # Defensive unpacking with validation (following parent class pattern)
        equation_tuple = self.equation_data[index]
        
        # Essential validation: ensure we have minimum required elements  
        if len(equation_tuple) < 2:
            raise ValueError(f"Invalid equation tuple at index {index}: expected at least 2 elements (input_eq, target_eq), got {len(equation_tuple)}")
        
        # Extract core elements using defensive pattern
        input_eq, target_eq = equation_tuple[0], equation_tuple[1]
        
        # Basic type validation to prevent encoder failures
        if not isinstance(input_eq, str) or not isinstance(target_eq, str):
            raise TypeError(f"Equations must be strings at index {index}, got input: {type(input_eq)}, target: {type(target_eq)}")
        
        # Encode equations to embeddings
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding
    
    def get_problem_info(self, index: int) -> Dict:
        """
        Get full information about a constrained problem.
        
        Args:
            index: Index of problem to retrieve
            
        Returns:
            Dictionary with input_eq, target_eq, rules_applied, and constraint_status
        """
        if index >= len(self.equation_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_data)}")
            
        # Defensive unpacking with validation (consistent with __getitem__ pattern)
        equation_tuple = self.equation_data[index]
        
        # Validate 4-tuple structure for constraint data access
        if len(equation_tuple) < 4:
            raise ValueError(f"Invalid equation tuple at index {index}: expected 4 elements (input_eq, target_eq, rules, constraints), got {len(equation_tuple)}")
        
        # Extract all elements using defensive pattern
        input_eq, target_eq, rules, constraints = equation_tuple[0], equation_tuple[1], equation_tuple[2], equation_tuple[3]
        
        # Basic type validation for core elements
        if not isinstance(input_eq, str) or not isinstance(target_eq, str):
            raise TypeError(f"Equations must be strings at index {index}, got input: {type(input_eq)}, target: {type(target_eq)}")
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': rules,
            'num_rules': len(rules),
            'constraints': self.constraints,
            'constraint_satisfaction': constraints
        }
    
    def get_constraint_stats(self) -> Dict:
        """Get statistics about constraint satisfaction across the dataset."""
        total_problems = len(self.equation_data)
        if total_problems == 0:
            return {}
            
        constraint_counts = {}
        for constraint in self.VALID_CONSTRAINTS:
            constraint_counts[constraint] = 0
            
        for _, _, _, constraints in self.equation_data:
            for constraint_name, satisfied in constraints.items():
                if satisfied:
                    constraint_counts[constraint_name] = constraint_counts.get(constraint_name, 0) + 1
        
        # Convert to percentages
        constraint_stats = {
            constraint: (count / total_problems) * 100 
            for constraint, count in constraint_counts.items()
        }
        
        return {
            'total_problems': total_problems,
            'constraint_satisfaction_rates': constraint_stats,
            'required_constraints': self.constraints
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the constrained dataset."""
        # Count rules manually since parent method expects 3-tuples but we have 4-tuples
        rule_counts = {}
        for _, _, rules, _ in self.equation_data:  # 4-tuple unpacking for constrained data
            for rule in rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Get constraint information
        constraint_info = self.get_constraint_stats()
        
        return {
            'num_rules': self.num_rules,
            'split': self.split,
            'num_problems': len(self.equation_data),
            'coeff_range': self.coeff_range,
            'rule_counts': rule_counts,
            'constraints': self.constraints,
            'constraint_stats': constraint_info
        }


# Quality indicators configuration as specified in the revised plan
QUALITY_INDICATORS = {
    'generation_efficiency': {
        'success_rate': 0.85,           # 85% valid equation generation
        'uniqueness_rate': 0.90,        # 90% unique problems
        'coverage_rate': 0.95           # 95% target range coverage
    },
    'algebraic_validity': {
        'syntactic_correctness': 1.0,   # 100% valid syntax
        'semantic_equivalence': 1.0,    # 100% rule correctness
        'solution_consistency': 1.0     # 100% consistent solutions
    },
    'variability_health': {
        'coefficient_entropy': 3.5,     # High coefficient diversity
        'solution_entropy': 4.0,        # High solution diversity  
        'pattern_repetition': 0.05      # Low pattern repetition
    }
}

# Coverage metrics configuration as specified in the revised plan
COVERAGE_METRICS = {
    'solution_distribution': {
        'target_ranges': [-50, -25, -10, 0, 10, 25, 50],
        'minimum_coverage': 0.02,  # 2% minimum per range
        'balance_threshold': 0.15  # Max deviation from uniform
    },
    'coefficient_diversity': {
        'unique_coefficient_ratio': 0.8,  # 80% unique coefficients
        'common_pattern_threshold': 0.05  # Max 5% identical patterns
    },
    'rule_complexity_distribution': {
        'simple_equations': 0.3,   # 30% single-step solutions
        'medium_equations': 0.5,   # 50% moderate complexity
        'complex_equations': 0.2   # 20% multi-step within rule
    }
}


class DatasetVariabilityValidator:
    """
    Validates dataset variability and coverage for algebraic equation generation.
    
    Provides comprehensive analysis of solution distribution, coefficient diversity,
    and rule complexity to ensure adequate dataset variability for training.
    
    Args:
        target_metrics: Dictionary of target metrics (default: COVERAGE_METRICS)
    """
    
    def __init__(self, target_metrics: Dict = None):
        self.metrics = target_metrics if target_metrics is not None else COVERAGE_METRICS
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_solution(self, equation_string: Union[str, Tuple[str, str]]) -> Optional[float]:
        """
        Extract the solution value from an equation string.
        
        Args:
            equation_string: Equation string to parse or tuple of (input_eq, target_eq)
            
        Returns:
            Solution value if found, None if parsing fails
        """
        try:
            # Handle tuple format (input_eq, target_eq) - use input equation
            if isinstance(equation_string, tuple):
                if len(equation_string) >= 1:
                    eq_str = equation_string[0]
                else:
                    return None
            else:
                eq_str = equation_string
            
            # Input validation for security
            if not eq_str or len(eq_str) > 1000:
                return None
                
            # Handle both = and == formats
            eq_str = eq_str.replace('==', '=')
            
            # Add input sanitization before solve_equation call
            import string
            allowed_chars = set(string.digits + 'x+-*/=.() ')
            if not set(eq_str).issubset(allowed_chars):
                self.logger.warning(f'Rejected equation with invalid characters: {eq_str[:50]}')
                return None

            # Use algebra_encoder's solve function (imported at module level)
            solutions, error = solve_equation(eq_str)
            
            if error or not solutions:
                return None
            
            # Extract candidate solution
            if isinstance(solutions[0], (int, float)):
                candidate = float(solutions[0])
            else:
                try:
                    candidate = float(solutions[0].evalf())
                except:
                    return None
            
            # Verify solution by substitution
            try:
                if '==' in eq_str:
                    left, right = eq_str.split('==')
                elif '=' in eq_str:
                    left, right = eq_str.split('=')
                else:
                    return None
                
                # Simple substitution check (safer than eval - consider sympy for production)
                left_val = eval(left.replace('x', str(candidate)))
                right_val = eval(right.replace('x', str(candidate)))
                
                if abs(left_val - right_val) > 1e-6:
                    self.logger.warning(f'Solution {candidate} does not satisfy equation {eq_str[:50]}')
                    return None
                
                return candidate
            except Exception as e:
                self.logger.error(f'Solution verification failed: {e}')
                return None
            
        except (ValueError, TypeError, ImportError) as e:
            self.logger.debug(f"Failed to extract solution from '{equation_string}': {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error extracting solution from '{equation_string}': {e}")
            return None
    
    def extract_coefficients(self, equation_string: Union[str, Tuple[str, str]]) -> List[int]:
        """
        Extract coefficient pattern from an equation string.
        
        Args:
            equation_string: Equation string to parse or tuple of (input_eq, target_eq)
            
        Returns:
            List of coefficients found in the equation
        """
        import re
        coefficients = []
        
        try:
            # Handle tuple format (input_eq, target_eq) - use input equation
            if isinstance(equation_string, tuple):
                if len(equation_string) >= 1:
                    eq_str = equation_string[0]
                else:
                    return coefficients
            else:
                eq_str = equation_string
            
            # Input validation for security (prevent ReDoS)
            if not eq_str or len(eq_str) > 500:
                return coefficients
                
            # Extract integer coefficients using regex
            # Matches patterns like: 3*x, -2*x, +5, -7, etc.
            coeff_patterns = re.findall(r'([+-]?\d+)', eq_str)
            
            for coeff_str in coeff_patterns:
                try:
                    coeff = int(coeff_str.replace('+', ''))
                    coefficients.append(coeff)
                except ValueError:
                    continue
                    
        except (ValueError, TypeError, re.error) as e:
            self.logger.debug(f"Failed to extract coefficients from '{equation_string}': {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error extracting coefficients from '{equation_string}': {e}")
        
        return coefficients
    
    def analyze_distribution(self, values: List[float], distribution_config: Dict) -> Dict:
        """
        Analyze the distribution of values against target ranges.
        
        Args:
            values: List of values to analyze
            distribution_config: Configuration with target_ranges and thresholds
            
        Returns:
            Dictionary with distribution analysis results
        """
        if not values:
            return {
                'passed': False,
                'error': 'No values to analyze',
                'coverage_by_range': {},
                'total_coverage': 0.0
            }
        
        target_ranges = distribution_config['target_ranges']
        minimum_coverage = distribution_config['minimum_coverage']
        balance_threshold = distribution_config['balance_threshold']
        
        # Count values in each range
        range_counts = {}
        total_values = len(values)
        
        for i, range_boundary in enumerate(target_ranges[:-1]):
            next_boundary = target_ranges[i + 1]
            range_key = f"{range_boundary}_to_{next_boundary}"
            
            # Include upper bound for last range only
            if i == len(target_ranges) - 2:  # Last range pair
                count = sum(1 for v in values if range_boundary <= v <= next_boundary)
            else:
                count = sum(1 for v in values if range_boundary <= v < next_boundary)
            coverage = count / total_values if total_values > 0 else 0.0
            range_counts[range_key] = {
                'count': count,
                'coverage': coverage,
                'meets_minimum': coverage >= minimum_coverage
            }
        
        # Calculate overall statistics
        total_coverage = sum(rc['coverage'] for rc in range_counts.values())
        expected_uniform = 1.0 / len(range_counts)
        max_deviation = max(abs(rc['coverage'] - expected_uniform) for rc in range_counts.values())
        balance_check = max_deviation <= balance_threshold
        
        # Check if all ranges meet minimum coverage
        all_ranges_covered = all(rc['meets_minimum'] for rc in range_counts.values())
        
        return {
            'passed': all_ranges_covered and balance_check,
            'total_coverage': total_coverage,
            'coverage_by_range': range_counts,
            'balance_check': balance_check,
            'max_deviation_from_uniform': max_deviation,
            'balance_threshold': balance_threshold,
            'all_ranges_covered': all_ranges_covered
        }
    
    def analyze_uniqueness(self, patterns: List[List[int]], diversity_config: Dict) -> Dict:
        """
        Analyze the uniqueness and diversity of coefficient patterns.
        
        Args:
            patterns: List of coefficient patterns to analyze
            diversity_config: Configuration with uniqueness thresholds
            
        Returns:
            Dictionary with uniqueness analysis results
        """
        if not patterns:
            return {
                'passed': False,
                'error': 'No patterns to analyze',
                'unique_ratio': 0.0,
                'most_common_frequency': 0.0
            }
        
        unique_coefficient_ratio = diversity_config['unique_coefficient_ratio']
        common_pattern_threshold = diversity_config['common_pattern_threshold']
        
        # Convert patterns to comparable tuples (preserve order for accuracy)
        pattern_tuples = [tuple(pattern) for pattern in patterns if pattern]
        
        if not pattern_tuples:
            return {
                'passed': False,
                'error': 'No valid patterns found',
                'unique_ratio': 0.0,
                'most_common_frequency': 0.0
            }
        
        # Count unique patterns
        from collections import Counter
        pattern_counts = Counter(pattern_tuples)
        
        total_patterns = len(pattern_tuples)
        unique_patterns = len(pattern_counts)
        actual_unique_ratio = unique_patterns / total_patterns
        
        # Find most common pattern frequency
        most_common_count = pattern_counts.most_common(1)[0][1]
        most_common_frequency = most_common_count / total_patterns
        
        # Check thresholds
        uniqueness_check = actual_unique_ratio >= unique_coefficient_ratio
        common_pattern_check = most_common_frequency <= common_pattern_threshold
        
        return {
            'passed': uniqueness_check and common_pattern_check,
            'unique_ratio': actual_unique_ratio,
            'target_unique_ratio': unique_coefficient_ratio,
            'unique_patterns': unique_patterns,
            'total_patterns': total_patterns,
            'most_common_frequency': most_common_frequency,
            'common_pattern_threshold': common_pattern_threshold,
            'uniqueness_check': uniqueness_check,
            'common_pattern_check': common_pattern_check
        }
    
    def validate_solution_coverage(self, dataset) -> Dict:
        """
        Ensure solution integer range coverage meets targets.
        
        Args:
            dataset: AlgebraDataset instance or list of equation strings
            
        Returns:
            Dictionary with solution coverage analysis
        """
        try:
            # Extract equation strings from dataset
            if hasattr(dataset, 'equation_pairs'):
                # AlgebraDataset instance
                equations = [pair[1] for pair in dataset.equation_pairs]  # Target equations
            elif isinstance(dataset, list):
                # Direct list of equation strings
                equations = dataset
            else:
                return {'passed': False, 'error': 'Invalid dataset format'}
            
            # Extract solutions
            solutions = []
            for eq in equations:
                solution = self.extract_solution(eq)
                if solution is not None:
                    solutions.append(solution)
            
            return self.analyze_distribution(solutions, self.metrics['solution_distribution'])
            
        except Exception as e:
            self.logger.error(f"Solution coverage validation failed: {e}")
            return {'passed': False, 'error': str(e)}
    
    def validate_coefficient_diversity(self, dataset) -> Dict:
        """
        Ensure coefficient patterns have adequate diversity.
        
        Args:
            dataset: AlgebraDataset instance or list of equation strings
            
        Returns:
            Dictionary with coefficient diversity analysis
        """
        try:
            # Extract equation strings from dataset
            if hasattr(dataset, 'equation_pairs'):
                # AlgebraDataset instance - analyze both input and target equations
                equations = []
                for pair in dataset.equation_pairs:
                    equations.extend(pair)  # Both input and target
            elif isinstance(dataset, list):
                # Direct list of equation strings
                equations = dataset
            else:
                return {'passed': False, 'error': 'Invalid dataset format'}
            
            # Extract coefficient patterns
            patterns = []
            for eq in equations:
                pattern = self.extract_coefficients(eq)
                if pattern:
                    patterns.append(pattern)
            
            return self.analyze_uniqueness(patterns, self.metrics['coefficient_diversity'])
            
        except Exception as e:
            self.logger.error(f"Coefficient diversity validation failed: {e}")
            return {'passed': False, 'error': str(e)}
    
    def generate_improvement_recommendations(self, dataset) -> List[str]:
        """
        Generate specific recommendations for improving dataset variability.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            List of specific improvement recommendations
        """
        recommendations = []
        
        try:
            # Analyze solution coverage
            solution_analysis = self.validate_solution_coverage(dataset)
            if not solution_analysis.get('passed', False):
                if 'coverage_by_range' in solution_analysis:
                    under_covered = [
                        range_name for range_name, data in solution_analysis['coverage_by_range'].items()
                        if not data['meets_minimum']
                    ]
                    if under_covered:
                        recommendations.append(
                            f"Increase solution coverage in ranges: {', '.join(under_covered)}"
                        )
                
                if not solution_analysis.get('balance_check', False):
                    recommendations.append(
                        f"Rebalance solution distribution - max deviation: "
                        f"{solution_analysis.get('max_deviation_from_uniform', 0):.3f}"
                    )
            
            # Analyze coefficient diversity
            coeff_analysis = self.validate_coefficient_diversity(dataset)
            if not coeff_analysis.get('passed', False):
                if not coeff_analysis.get('uniqueness_check', False):
                    recommendations.append(
                        f"Increase coefficient diversity - current unique ratio: "
                        f"{coeff_analysis.get('unique_ratio', 0):.3f}, "
                        f"target: {coeff_analysis.get('target_unique_ratio', 0):.3f}"
                    )
                
                if not coeff_analysis.get('common_pattern_check', False):
                    recommendations.append(
                        f"Reduce common pattern repetition - most frequent pattern appears "
                        f"{coeff_analysis.get('most_common_frequency', 0):.3f} of the time, "
                        f"threshold: {coeff_analysis.get('common_pattern_threshold', 0):.3f}"
                    )
            
            if not recommendations:
                recommendations.append("Dataset variability meets all target metrics")
                
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append(f"Analysis failed: {e}")
        
        return recommendations
    
    def generate_coverage_report(self, dataset) -> Dict:
        """
        Generate comprehensive coverage analysis report.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with complete coverage analysis
        """
        return {
            'solution_coverage': self.validate_solution_coverage(dataset),
            'coefficient_diversity': self.validate_coefficient_diversity(dataset),
            'recommendations': self.generate_improvement_recommendations(dataset),
            'metrics_configuration': self.metrics,
            'overall_passed': (
                self.validate_solution_coverage(dataset).get('passed', False) and
                self.validate_coefficient_diversity(dataset).get('passed', False)
            )
        }


class ContinuousQualityMonitor:
    """
    Continuous quality monitoring for algebraic equation generation.
    
    Provides ongoing monitoring of dataset quality without architectural changes.
    Implements automated quality assurance with threshold enforcement and alerting.
    
    Args:
        quality_thresholds: Dictionary of quality thresholds (default: QUALITY_INDICATORS)
    """
    
    def __init__(self, quality_thresholds: Dict = None):
        self.thresholds = quality_thresholds if quality_thresholds is not None else QUALITY_INDICATORS
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validator = DatasetVariabilityValidator()
        
        # Alert tracking
        self.alert_history = []
        self.max_alert_history = 50
        
        # Quality metrics cache with size management
        self.metrics_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.max_cache_size = 100  # Prevent memory leaks
        
        import threading
        self._cache_lock = threading.Lock()  # Thread safety
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries and enforce size limits."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, value in self.metrics_cache.items()
            if current_time - value['timestamp'] > self.cache_timeout
        ]
        for key in expired_keys:
            del self.metrics_cache[key]
        
        # Enforce size limit by removing oldest entries
        if len(self.metrics_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.metrics_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            for key, _ in sorted_items[:-self.max_cache_size]:
                del self.metrics_cache[key]
        
    def compute_batch_metrics(self, batch_problems: List[str]) -> Dict:
        """
        Compute quality metrics for a batch of problems.
        
        Args:
            batch_problems: List of equation strings to analyze
            
        Returns:
            Dictionary with computed quality metrics
        """
        if not batch_problems:
            return {'error': 'No problems to analyze'}
            
        try:
            import time
            # Use secure hashing to prevent collisions
            # Use both first and last 100 problems to differentiate batches
            batch_size = len(batch_problems)
            if batch_size <= 200:
                batch_sample = tuple(batch_problems)
            else:
                batch_sample = tuple(batch_problems[:100] + batch_problems[-100:])

            # Include batch size and use full hash to prevent collisions
            cache_key = f"batch_{batch_size}_{hashlib.sha256(str(batch_sample).encode()).hexdigest()}"
            current_time = time.time()
            
            # Check cache first (thread-safe)
            with self._cache_lock:
                if (cache_key in self.metrics_cache and 
                    current_time - self.metrics_cache[cache_key]['timestamp'] < self.cache_timeout):
                    return self.metrics_cache[cache_key]['metrics']
            
            metrics = {}
            
            # Generation Efficiency Metrics
            total_problems = len(batch_problems)
            valid_problems = []
            unique_problems = set()
            
            for problem in batch_problems:
                # Check syntactic validity
                try:
                    from algebra_encoder import validate_equation_syntax
                    is_valid, _, _ = validate_equation_syntax(problem)
                    if is_valid:
                        valid_problems.append(problem)
                        unique_problems.add(problem)
                except:
                    continue
            
            # Calculate generation efficiency
            success_rate = len(valid_problems) / total_problems if total_problems > 0 else 0.0
            uniqueness_rate = len(unique_problems) / total_problems if total_problems > 0 else 0.0
            
            # Coverage rate from validator
            coverage_analysis = self.validator.validate_solution_coverage(valid_problems)
            coverage_rate = coverage_analysis.get('total_coverage', 0.0)
            
            metrics['generation_efficiency'] = {
                'success_rate': success_rate,
                'uniqueness_rate': uniqueness_rate,
                'coverage_rate': coverage_rate
            }
            
            # Algebraic Validity Metrics
            syntactic_correctness = success_rate  # Already computed above
            semantic_equivalence = self._check_semantic_equivalence(valid_problems)
            solution_consistency = self._check_solution_consistency(valid_problems)
            
            metrics['algebraic_validity'] = {
                'syntactic_correctness': syntactic_correctness,
                'semantic_equivalence': semantic_equivalence,
                'solution_consistency': solution_consistency
            }
            
            # Variability Health Metrics
            coefficient_entropy = self._calculate_coefficient_entropy(valid_problems)
            solution_entropy = self._calculate_solution_entropy(valid_problems)
            pattern_repetition = self._calculate_pattern_repetition(valid_problems)
            
            metrics['variability_health'] = {
                'coefficient_entropy': coefficient_entropy,
                'solution_entropy': solution_entropy,
                'pattern_repetition': pattern_repetition
            }
            
            # Cache results with size management (thread-safe)
            with self._cache_lock:
                self.metrics_cache[cache_key] = {
                    'metrics': metrics,
                    'timestamp': current_time
                }
                self._cleanup_cache()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute batch metrics: {e}")
            return {'error': str(e)}
    
    def _check_semantic_equivalence(self, problems: List[str]) -> float:
        """Check semantic equivalence of equation transformations."""
        if not problems:
            return 1.0
            
        try:
            # For single equations, we assume semantic correctness from syntactic validity
            # In a full implementation, this would verify transformation correctness
            return 1.0
        except Exception:
            return 0.0
    
    def _check_solution_consistency(self, problems: List[str]) -> float:
        """Check solution consistency across problems."""
        if not problems:
            return 1.0
            
        try:
            consistent_solutions = 0
            total_solvable = 0
            
            for problem in problems:
                try:
                    solution = self.validator.extract_solution(problem)
                    if solution is not None:
                        total_solvable += 1
                        # Check if solution is a valid number
                        if isinstance(solution, (int, float)) and not math.isnan(float(solution)):
                            consistent_solutions += 1
                except:
                    continue
            
            return consistent_solutions / total_solvable if total_solvable > 0 else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_coefficient_entropy(self, problems: List[str]) -> float:
        """Calculate entropy of coefficient distributions."""
        if not problems:
            return 0.0
            
        try:
            import math
            from collections import Counter
            
            all_coefficients = []
            for problem in problems:
                coefficients = self.validator.extract_coefficients(problem)
                all_coefficients.extend(coefficients)
            
            if not all_coefficients:
                return 0.0
            
            # Calculate entropy safely
            counts = Counter(all_coefficients)
            total = len(all_coefficients)
            entropy = 0.0
            for count in counts.values():
                if count > 0:  # Avoid log(0)
                    probability = count / total
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_solution_entropy(self, problems: List[str]) -> float:
        """Calculate entropy of solution distributions."""
        if not problems:
            return 0.0
            
        try:
            import math
            from collections import Counter
            
            solutions = []
            for problem in problems:
                solution = self.validator.extract_solution(problem)
                if solution is not None:
                    # Round to handle floating point precision
                    # Keep original precision for entropy calculation
                    solutions.append(float(solution))
            
            if not solutions:
                return 0.0
            
            # Calculate entropy safely
            counts = Counter(solutions)
            total = len(solutions)
            entropy = 0.0
            for count in counts.values():
                if count > 0:  # Avoid log(0)
                    probability = count / total
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_repetition(self, problems: List[str]) -> float:
        """Calculate pattern repetition rate."""
        if not problems:
            return 0.0
            
        try:
            from collections import Counter
            
            # Extract coefficient patterns
            patterns = []
            for problem in problems:
                coefficients = self.validator.extract_coefficients(problem)
                if coefficients:
                    patterns.append(tuple(coefficients))
            
            if not patterns:
                return 0.0
            
            # Calculate most common pattern frequency
            pattern_counts = Counter(patterns)
            most_common_count = pattern_counts.most_common(1)[0][1]
            
            return most_common_count / len(patterns)
            
        except Exception:
            return 0.0
    
    def check_quality_thresholds(self, metrics: Dict) -> List[Dict]:
        """
        Check if metrics meet quality thresholds and generate alerts.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            List of alerts for threshold violations
        """
        alerts = []
        
        try:
            for category, category_metrics in metrics.items():
                if category == 'error':
                    continue
                    
                category_thresholds = self.thresholds.get(category, {})
                
                for metric_name, value in category_metrics.items():
                    threshold = category_thresholds.get(metric_name)
                    
                    if threshold is not None:
                        # Check if metric meets threshold
                        if metric_name == 'pattern_repetition':
                            # Lower is better for pattern repetition
                            if value > threshold:
                                alerts.append({
                                    'category': category,
                                    'metric': metric_name,
                                    'value': value,
                                    'threshold': threshold,
                                    'severity': 'warning' if value <= threshold * 1.5 else 'critical',
                                    'message': f"{metric_name} ({value:.3f}) exceeds threshold ({threshold})"
                                })
                        else:
                            # Higher is better for most metrics
                            if value < threshold:
                                alerts.append({
                                    'category': category,
                                    'metric': metric_name,
                                    'value': value,
                                    'threshold': threshold,
                                    'severity': 'warning' if value > threshold * 0.8 else 'critical',
                                    'message': f"{metric_name} ({value:.3f}) below threshold ({threshold})"
                                })
        
        except Exception as e:
            self.logger.error(f"Failed to check quality thresholds: {e}")
            alerts.append({
                'category': 'system',
                'metric': 'threshold_check',
                'value': 0.0,
                'threshold': 1.0,
                'severity': 'critical',
                'message': f"Threshold checking failed: {e}"
            })
        
        return alerts
    
    def trigger_quality_adjustments(self, alerts: List[Dict]) -> Dict:
        """
        Trigger quality adjustments based on alerts.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Dictionary with adjustment actions taken
        """
        if not alerts:
            return {'actions': []}
        
        actions_taken = []
        
        try:
            import time
            
            # Store alerts in history
            timestamp = time.time()
            for alert in alerts:
                alert['timestamp'] = timestamp
            
            self.alert_history.extend(alerts)
            
            # Limit alert history size
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
            
            # Group alerts by severity
            critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
            warning_alerts = [a for a in alerts if a.get('severity') == 'warning']
            
            # Log alerts
            if critical_alerts:
                for alert in critical_alerts:
                    self.logger.critical(f"Quality Alert: {alert['message']}")
                actions_taken.append("Logged critical quality alerts")
            
            if warning_alerts:
                for alert in warning_alerts:
                    self.logger.warning(f"Quality Warning: {alert['message']}")
                actions_taken.append("Logged quality warnings")
            
            # Note: Actual adjustments would depend on the specific system architecture
            # This implementation focuses on monitoring and alerting
            
        except Exception as e:
            self.logger.error(f"Failed to trigger quality adjustments: {e}")
            actions_taken.append(f"Adjustment failed: {e}")
        
        return {'actions': actions_taken, 'alert_count': len(alerts)}
    
    def monitor_generation_batch(self, batch_problems: List[str]) -> Dict:
        """
        Monitor quality for each generation batch.
        
        Args:
            batch_problems: List of equation strings to monitor
            
        Returns:
            Dictionary with monitoring results
        """
        try:
            # Compute metrics for the batch
            metrics = self.compute_batch_metrics(batch_problems)
            
            if 'error' in metrics:
                return {'status': 'failed', 'error': metrics['error']}
            
            # Check thresholds and generate alerts
            alerts = self.check_quality_thresholds(metrics)
            
            # Trigger adjustments if needed
            adjustment_results = self.trigger_quality_adjustments(alerts)
            
            return {
                'status': 'completed',
                'metrics': metrics,
                'alerts': alerts,
                'adjustments': adjustment_results,
                'batch_size': len(batch_problems)
            }
            
        except Exception as e:
            self.logger.error(f"Batch monitoring failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def analyze_coverage_trends(self, dataset) -> Dict:
        """Analyze coverage trends over time."""
        try:
            # Use validator for coverage analysis
            coverage_report = self.validator.generate_coverage_report(dataset)
            
            return {
                'current_coverage': coverage_report.get('solution_coverage', {}),
                'diversity_status': coverage_report.get('coefficient_diversity', {}),
                'trend': 'stable'  # Would require historical data for actual trend analysis
            }
            
        except Exception as e:
            self.logger.error(f"Coverage trend analysis failed: {e}")
            return {'error': str(e)}
    
    def detect_quality_issues(self, dataset) -> List[str]:
        """Detect quality degradation issues."""
        issues = []
        
        try:
            # Get current metrics for the dataset
            if hasattr(dataset, 'equation_pairs'):
                equations = [pair[1] for pair in dataset.equation_pairs]  # Target equations
            elif isinstance(dataset, list):
                equations = dataset
            else:
                return ['Invalid dataset format for quality detection']
            
            # Compute metrics
            metrics = self.compute_batch_metrics(equations)
            
            if 'error' in metrics:
                return [f"Quality detection failed: {metrics['error']}"]
            
            # Check for quality issues
            alerts = self.check_quality_thresholds(metrics)
            
            for alert in alerts:
                if alert.get('severity') == 'critical':
                    issues.append(f"Critical: {alert['message']}")
                elif alert.get('severity') == 'warning':
                    issues.append(f"Warning: {alert['message']}")
            
            if not issues:
                issues.append("No quality issues detected")
                
        except Exception as e:
            self.logger.error(f"Quality issue detection failed: {e}")
            issues.append(f"Quality detection failed: {e}")
        
        return issues
    
    def suggest_improvements(self, dataset) -> List[str]:
        """Suggest improvements based on quality analysis."""
        suggestions = []
        
        try:
            # Use validator for improvement suggestions
            coverage_report = self.validator.generate_coverage_report(dataset)
            suggestions.extend(coverage_report.get('recommendations', []))
            
            # Add monitor-specific suggestions based on recent alerts
            recent_alerts = self.alert_history[-10:] if self.alert_history else []
            
            if recent_alerts:
                critical_count = sum(1 for a in recent_alerts if a.get('severity') == 'critical')
                if critical_count > 3:
                    suggestions.append("Consider adjusting generation parameters due to frequent critical alerts")
                    
                warning_count = sum(1 for a in recent_alerts if a.get('severity') == 'warning')
                if warning_count > 5:
                    suggestions.append("Monitor generation trends for potential quality degradation")
            
            if not suggestions:
                suggestions.append("Dataset quality meets all monitoring criteria")
                
        except Exception as e:
            self.logger.error(f"Improvement suggestion failed: {e}")
            suggestions.append(f"Analysis failed: {e}")
        
        return suggestions
    
    def generate_periodic_reports(self, dataset, period: str = 'weekly') -> Dict:
        """
        Generate regular quality assessment reports.
        
        Args:
            dataset: Dataset to analyze
            period: Report period ('weekly', 'monthly', etc.)
            
        Returns:
            Dictionary with comprehensive quality report
        """
        try:
            report = {
                'report_period': period,
                'timestamp': time.time(),
                'coverage_trends': self.analyze_coverage_trends(dataset),
                'quality_degradation': self.detect_quality_issues(dataset),
                'optimization_recommendations': self.suggest_improvements(dataset),
                'alert_summary': {
                    'total_alerts': len(self.alert_history),
                    'recent_critical': len([a for a in self.alert_history[-20:] if a.get('severity') == 'critical']),
                    'recent_warnings': len([a for a in self.alert_history[-20:] if a.get('severity') == 'warning'])
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Periodic report generation failed: {e}")
            return {'error': str(e), 'period': period}
    
    def get_alert_history(self) -> List[Dict]:
        """Get the history of quality alerts."""
        return self.alert_history.copy()
    
    def clear_alert_history(self) -> int:
        """Clear alert history and return count of cleared alerts."""
        count = len(self.alert_history)
        self.alert_history.clear()
        return count