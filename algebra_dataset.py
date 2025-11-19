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
from algebra_encoder import create_character_encoder, validate_equation_syntax, check_equation_equivalence


class AlgebraDataset(data.Dataset):
    """
    Base dataset class for single-rule algebraic problems.
    
    Generates pairs of (input_equation, target_equation) for training rule-specific EBMs.
    Each rule type (distribute, combine, isolate, divide) gets a separate dataset.
    
    Args:
        rule: Rule type ('distribute', 'combine', 'isolate', 'divide')
        split: Dataset split ('train', 'test', 'val')  
        num_problems: Number of problems to generate (default: 50000)
        coeff_range: Range for random coefficients (default: [-10, 10])
        d_model: Encoder embedding dimension (default: 128)
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    
    def __init__(
        self,
        rule: str,
        split: str = 'train',
        num_problems: int = 50000,
        coeff_range: List[int] = [-10, 10],
        d_model: int = 128
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
        
        # Pre-generate all equation pairs for deterministic behavior
        self.equation_pairs = self._generate_all_equations()
    
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
    
    def _generate_distribute_equation(self) -> Tuple[str, str]:
        """
        Generate distribute rule equation pair.
        Rule: a(x + b) = ax + ab  or  a(x - b) = ax - ab
        
        Returns: (input_equation, target_equation)
        """
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
        """Pre-generate all equation pairs for the dataset."""
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
                
        # Log generation completion statistics
        success_rate = len(equations) / attempts if attempts > 0 else 0
        self.logger.info(f"Generation complete: {len(equations)} equations from {attempts} attempts "
                        f"({success_rate:.1%} success rate)")
        if self._generation_stats['failures']:
            self.logger.info(f"Failure breakdown: {dict(self._generation_stats['failures'])}")
            
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
            
        input_eq, target_eq, _, _ = self.equation_data[index]
        
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
            
        input_eq, target_eq, rules, constraints = self.equation_data[index]
        
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
        base_info = super().get_dataset_info()
        constraint_info = self.get_constraint_stats()
        
        # Merge the dictionaries
        base_info.update({
            'constraints': self.constraints,
            'constraint_stats': constraint_info
        })
        
        return base_info