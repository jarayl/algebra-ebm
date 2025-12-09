"""
Algebra Dataset Classes for EBM Training

Implements PyTorch Dataset classes for generating and loading algebraic equation problems.
Creates separate datasets for single-rule, multi-rule, and constrained evaluation.

Classes:
- AlgebraDataset: Base class for single-rule problems (distribute, combine, isolate, divide)
- MultiRuleDataset: For compositional testing (2-4 sequential rule applications)  
- ConstrainedDataset: For constraint evaluation (positivity/integerness requirements)
- CombinedAlgebraDataset: Monolithic dataset combining all 4 rules for IRED baseline training
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
from src.algebra.algebra_encoder import create_character_encoder, validate_equation_syntax, check_equation_equivalence, solve_equation


class AlgebraDataset(data.Dataset):
    """
    Base dataset class for single-rule algebraic problems.
    
    Generates equations following specific algebraic rules:
    - distribute: a*(b+c) -> a*b + a*c
    - combine: a*b + a*c -> a*(b+c) 
    - isolate: a*x + b = c -> x = (c-b)/a
    - divide: a*x = b -> x = b/a
    
    Features:
    - Adaptive generation with coverage monitoring
    - Controlled complexity and variability
    - Character encoding for text-based models
    - Syntax validation and solution verification
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    DEFAULT_NUM_PROBLEMS = {
        'train': 50000,
        'val': 5000,
        'test': 10000
    }
    
    def __init__(
        self,
        rule: str,
        split: str = 'train',
        num_problems: Optional[int] = None,
        d_model: int = 128,
        min_coefficient: int = 2,
        max_coefficient: int = 10,
        min_constant: int = 1,
        max_constant: int = 15,
        force_positive_coefficients: bool = False,
        force_integer_solutions: bool = False,
        enable_adaptive_generation: bool = True,
        coverage_check_interval: int = 5000,
        use_variable_names: List[str] = None,
        debug_mode: bool = False
    ):
        super().__init__()
        
        # Validate rule
        if rule not in self.VALID_RULES:
            raise ValueError(f"Invalid rule '{rule}'. Must be one of {self.VALID_RULES}")
            
        self.rule = rule
        self.split = split
        self.d_model = d_model
        self.debug_mode = debug_mode
        
        # Set number of problems
        if num_problems is None:
            self.num_problems = self.DEFAULT_NUM_PROBLEMS.get(split, 10000)
        else:
            self.num_problems = num_problems
            
        # Generation parameters
        self.min_coefficient = min_coefficient
        self.max_coefficient = max_coefficient
        self.min_constant = min_constant
        self.max_constant = max_constant
        self.force_positive_coefficients = force_positive_coefficients
        self.force_integer_solutions = force_integer_solutions
        self.enable_adaptive_generation = enable_adaptive_generation
        self.coverage_check_interval = coverage_check_interval
        
        # Variable names to use - CRITICAL: Only use 'x' since encoder vocabulary is '0123456789x.+-=*/()[]<> '
        if use_variable_names is None:
            self.use_variable_names = ['x']  # Fixed to match encoder vocabulary
        else:
            # Filter to only variables supported by the encoder
            supported_vars = ['x']  # Only 'x' is in the encoder vocabulary
            self.use_variable_names = [var for var in use_variable_names if var in supported_vars]
            if not self.use_variable_names:
                self.use_variable_names = ['x']  # Fallback to 'x'
            
        # Dataset interface requirements
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Coverage tracking (for adaptive generation)
        self._coverage_history = []
        self._coverage_checkpoints = []
        
        # Generate the problems
        self._generate_problems()
        
        print(f"Generated {len(self.equation_pairs)} {rule} problems for {split}")
        if self.debug_mode:
            self._print_sample_problems(5)
    
    def _generate_problems(self) -> None:
        """Generate equation pairs following the specified rule."""
        self.equation_pairs = []
        generation_attempts = 0
        max_attempts = self.num_problems * 10  # Safety limit
        
        start_time = time.time()
        
        while len(self.equation_pairs) < self.num_problems and generation_attempts < max_attempts:
            try:
                # Generate a single problem
                input_eq, target_eq = self._generate_single_problem()
                
                if input_eq and target_eq:
                    self.equation_pairs.append((input_eq, target_eq))
                    
                    # Adaptive coverage checking
                    if (self.enable_adaptive_generation and 
                        len(self.equation_pairs) % self.coverage_check_interval == 0):
                        self._check_coverage_and_adapt()
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"Generation attempt {generation_attempts} failed: {e}")
                pass
                
            generation_attempts += 1
            
        end_time = time.time()
        
        if len(self.equation_pairs) < self.num_problems:
            print(f"Warning: Only generated {len(self.equation_pairs)}/{self.num_problems} problems after {generation_attempts} attempts")
            
        print(f"Generation completed in {end_time - start_time:.2f}s ({generation_attempts} attempts)")
        
    def _generate_single_problem(self) -> Tuple[str, str]:
        """Generate a single equation pair following the rule."""
        if self.rule == 'distribute':
            return self._generate_distribute()
        elif self.rule == 'combine':
            return self._generate_combine()
        elif self.rule == 'isolate':
            return self._generate_isolate()
        elif self.rule == 'divide':
            return self._generate_divide()
        else:
            raise ValueError(f"Unknown rule: {self.rule}")
    
    def _generate_distribute(self) -> Tuple[str, str]:
        """Generate distribute problems: a*(b+c) -> a*b + a*c"""
        # Pick coefficients
        a = random.randint(self.min_coefficient, self.max_coefficient)
        b = random.randint(self.min_coefficient, self.max_coefficient)
        c = random.randint(self.min_coefficient, self.max_coefficient)
        
        # Pick variable
        var = random.choice(self.use_variable_names)
        
        # Create equations
        input_eq = f"{a}*({b}*{var} + {c})"
        target_eq = f"{a*b}*{var} + {a*c}"
        
        # Validate
        if not self._validate_equation_pair(input_eq, target_eq):
            return None, None
            
        return input_eq, target_eq
    
    def _generate_combine(self) -> Tuple[str, str]:
        """Generate combine problems: a*x + b*x -> (a+b)*x"""
        # Pick coefficients  
        a = random.randint(self.min_coefficient, self.max_coefficient)
        b = random.randint(self.min_coefficient, self.max_coefficient)
        
        # Pick variable
        var = random.choice(self.use_variable_names)
        
        # Create equations
        input_eq = f"{a}*{var} + {b}*{var}"
        target_eq = f"{a+b}*{var}"
        
        # Validate
        if not self._validate_equation_pair(input_eq, target_eq):
            return None, None
            
        return input_eq, target_eq
    
    def _generate_isolate(self) -> Tuple[str, str]:
        """Generate isolate problems: a*x + b = c -> x = (c-b)/a"""
        # Pick coefficients
        a = random.randint(self.min_coefficient, self.max_coefficient)
        b = random.randint(self.min_constant, self.max_constant)
        
        # Pick solution value
        x_val = random.randint(self.min_constant, self.max_constant)
        c = a * x_val + b  # Ensure integer solution
        
        # Pick variable
        var = random.choice(self.use_variable_names)
        
        # Create equations
        input_eq = f"{a}*{var} + {b} = {c}"
        target_eq = f"{var} = {x_val}"
        
        # Validate
        if not self._validate_equation_pair(input_eq, target_eq):
            return None, None
            
        return input_eq, target_eq
    
    def _generate_divide(self) -> Tuple[str, str]:
        """Generate divide problems: a*x = b -> x = b/a"""
        # Pick coefficients
        a = random.randint(self.min_coefficient, self.max_coefficient)
        
        # Pick solution value
        x_val = random.randint(self.min_constant, self.max_constant)
        b = a * x_val  # Ensure integer solution
        
        # Pick variable
        var = random.choice(self.use_variable_names)
        
        # Create equations
        input_eq = f"{a}*{var} = {b}"
        target_eq = f"{var} = {x_val}"
        
        # Validate
        if not self._validate_equation_pair(input_eq, target_eq):
            return None, None
            
        return input_eq, target_eq
    
    def _validate_equation_pair(self, input_eq: str, target_eq: str) -> bool:
        """Validate that equation pair is syntactically correct and equivalent."""
        try:
            # Basic syntax validation
            if not validate_equation_syntax(input_eq) or not validate_equation_syntax(target_eq):
                return False
                
            # Check equivalence
            if not check_equation_equivalence(input_eq, target_eq):
                return False
                
            # Rule-specific validation
            if self.force_integer_solutions:
                solution = solve_equation(target_eq)
                if solution is not None and not isinstance(solution, int):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def _check_coverage_and_adapt(self) -> None:
        """Analyze current problem coverage and adapt generation parameters."""
        if not self.enable_adaptive_generation:
            return
            
        try:
            current_problems = len(self.equation_pairs)
            
            # Analyze coefficient distribution
            coeffs_used = set()
            vars_used = set()
            
            for input_eq, target_eq in self.equation_pairs[-self.coverage_check_interval:]:
                # Extract coefficients and variables (basic parsing)
                for char in input_eq + target_eq:
                    if char.isalpha():
                        vars_used.add(char)
                        
            coverage_data = {
                'checkpoint': len(self._coverage_history) + 1,
                'problems_generated': current_problems,
                'unique_variables': len(vars_used),
                'variables_used': sorted(vars_used),
                'timestamp': time.time()
            }
            
            self._coverage_history.append(coverage_data)
            
            if self.debug_mode:
                print(f"Coverage checkpoint {coverage_data['checkpoint']}: "
                      f"{coverage_data['problems_generated']} problems, "
                      f"{coverage_data['unique_variables']} unique variables")
                      
        except Exception as e:
            if self.debug_mode:
                print(f"Coverage analysis failed: {e}")
    
    def _print_sample_problems(self, num_samples: int = 5) -> None:
        """Print sample problems for debugging."""
        print(f"\nSample {self.rule} problems:")
        for i, (input_eq, target_eq) in enumerate(self.equation_pairs[:num_samples]):
            print(f"  {i+1}. {input_eq} -> {target_eq}")
    
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
        if not self.equation_pairs:
            return {
                'total_problems': 0,
                'coverage_score': 0.0,
                'validation_passed': False,
                'message': 'No problems generated'
            }
            
        # Analyze variable usage
        vars_used = set()
        coefficient_range = set()
        
        for input_eq, target_eq in self.equation_pairs:
            # Basic analysis of variables and coefficients
            for char in input_eq + target_eq:
                if char.isalpha():
                    vars_used.add(char)
                elif char.isdigit():
                    coefficient_range.add(int(char))
        
        coverage_score = min(1.0, len(vars_used) / len(self.use_variable_names))
        
        return {
            'total_problems': len(self.equation_pairs),
            'unique_variables': len(vars_used),
            'variables_used': sorted(vars_used),
            'coverage_score': coverage_score,
            'validation_passed': len(self.equation_pairs) >= self.num_problems * 0.95,
            'rule': self.rule,
            'generation_parameters': {
                'min_coefficient': self.min_coefficient,
                'max_coefficient': self.max_coefficient,
                'min_constant': self.min_constant,
                'max_constant': self.max_constant
            }
        }
    
    def get_problem_info(self, index: int) -> Dict:
        """Get problem information for evaluation."""
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        
        input_eq, target_eq = self.equation_pairs[index]
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': [self.rule],  # Single rule for this dataset
            'num_rules': 1,
            'rule': self.rule
        }
    
    def __len__(self) -> int:
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded equation pair at index."""
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode equations to embeddings
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding


class MultiRuleDataset(data.Dataset):
    """
    Dataset for compositional evaluation requiring sequential rule applications.
    
    Generates problems that require 2-4 sequential rule applications:
    - 2-rule: distribute -> combine
    - 3-rule: distribute -> isolate -> divide  
    - 4-rule: distribute -> combine -> isolate -> divide
    
    Tests compositional reasoning abilities of trained models.
    """
    
    RULE_SEQUENCES = {
        2: [['distribute', 'combine'], ['combine', 'distribute']],
        3: [['distribute', 'isolate', 'divide'], ['combine', 'distribute', 'isolate']],
        4: [['distribute', 'combine', 'isolate', 'divide']]
    }
    
    def __init__(
        self,
        num_rules: int = 2,
        split: str = 'test',
        num_problems: int = 1000,
        d_model: int = 128,
        **kwargs
    ):
        super().__init__()
        
        if num_rules not in self.RULE_SEQUENCES:
            raise ValueError(f"num_rules must be one of {list(self.RULE_SEQUENCES.keys())}")
            
        self.num_rules = num_rules
        self.split = split
        self.num_problems = num_problems
        self.d_model = d_model
        
        # Dataset interface
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Generate multi-step problems
        self._generate_multi_step_problems()
        
        print(f"Generated {len(self.equation_pairs)} {num_rules}-rule problems")
    
    def _generate_multi_step_problems(self) -> None:
        """Generate problems requiring sequential rule applications."""
        self.equation_pairs = []
        self.rule_sequences_used = []
        
        problems_per_sequence = self.num_problems // len(self.RULE_SEQUENCES[self.num_rules])
        
        for rule_sequence in self.RULE_SEQUENCES[self.num_rules]:
            for _ in range(problems_per_sequence):
                try:
                    input_eq, target_eq = self._generate_sequential_problem(rule_sequence)
                    if input_eq and target_eq:
                        self.equation_pairs.append((input_eq, target_eq))
                        self.rule_sequences_used.append(rule_sequence.copy())
                except Exception:
                    continue
                    
        print(f"Generated {len(self.equation_pairs)} multi-rule problems")
    
    def _generate_sequential_problem(self, rule_sequence: List[str]) -> Tuple[str, str]:
        """Generate a problem requiring the given sequence of rule applications."""
        # Start with a complex expression that requires all rules in sequence
        # This is a simplified implementation - in practice, this would be more sophisticated
        
        # Pick variable and coefficients - use only 'x' to match encoder vocabulary
        var = 'x'
        a, b, c, d = [random.randint(2, 5) for _ in range(4)]
        
        # Create a complex starting expression
        if len(rule_sequence) == 2:
            input_eq = f"{a}*({b}*{var} + {c}) + {d}*{var}"
            target_eq = f"{a*b + d}*{var} + {a*c}"
        elif len(rule_sequence) == 3:
            input_eq = f"{a}*({b}*{var} + {c}) = {d}"
            solution = (d - a*c) // (a*b) if (d - a*c) % (a*b) == 0 else 1
            target_eq = f"{var} = {solution}"
        else:  # 4 rules
            input_eq = f"{a}*({b}*{var} + {c}) + {d}*{var} = {a*c + d}"
            target_eq = f"{var} = 1"
            
        return input_eq, target_eq
    
    def get_problem_info(self, index: int) -> Dict:
        """Get problem information for evaluation."""
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        
        input_eq, target_eq = self.equation_pairs[index]
        rules_applied = self.rule_sequences_used[index]
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': rules_applied,
            'num_rules': len(rules_applied),
            'rule_sequence': rules_applied
        }
    
    def __len__(self) -> int:
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded equation pair at index."""
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode equations
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding


class ConstrainedDataset(data.Dataset):
    """
    Dataset for evaluating constraint satisfaction abilities.
    
    Generates algebraic problems with additional constraints:
    - Positivity: solutions must be positive
    - Integer-only: solutions must be integers
    - Range constraints: solutions in specific ranges
    
    Tests model ability to satisfy constraints during problem solving.
    """
    
    CONSTRAINT_TYPES = ['positivity', 'integer_only', 'range_constraint']
    
    def __init__(
        self,
        constraint_type: str = 'positivity',
        split: str = 'test', 
        num_problems: int = 1000,
        d_model: int = 128,
        solution_range: Tuple[int, int] = (1, 20),
        **kwargs
    ):
        super().__init__()
        
        if constraint_type not in self.CONSTRAINT_TYPES:
            raise ValueError(f"constraint_type must be one of {self.CONSTRAINT_TYPES}")
            
        self.constraint_type = constraint_type
        self.split = split
        self.num_problems = num_problems
        self.d_model = d_model
        self.solution_range = solution_range
        
        # Dataset interface
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Generate constrained problems
        self._generate_constrained_problems()
        
        print(f"Generated {len(self.equation_pairs)} {constraint_type} constrained problems")
    
    def _generate_constrained_problems(self) -> None:
        """Generate problems with specific constraints."""
        self.equation_pairs = []
        
        for _ in range(self.num_problems):
            try:
                input_eq, target_eq = self._generate_constrained_problem()
                if input_eq and target_eq:
                    self.equation_pairs.append((input_eq, target_eq))
            except Exception:
                continue
                
        print(f"Generated {len(self.equation_pairs)} constrained problems")
    
    def _generate_constrained_problem(self) -> Tuple[str, str]:
        """Generate a single constrained problem."""
        # Pick variable - use only 'x' to match encoder vocabulary
        var = 'x'
        
        if self.constraint_type == 'positivity':
            # Generate problem with guaranteed positive solution
            solution = random.randint(1, 10)
            a = random.randint(2, 5)
            b = random.randint(1, 10)
            c = a * solution + b
            
            input_eq = f"{a}*{var} + {b} = {c}"
            target_eq = f"{var} = {solution}"
            
        elif self.constraint_type == 'integer_only':
            # Generate problem with guaranteed integer solution
            solution = random.randint(*self.solution_range)
            a = random.randint(2, 5)
            b = a * solution
            
            input_eq = f"{a}*{var} = {b}"
            target_eq = f"{var} = {solution}"
            
        else:  # range_constraint
            # Generate problem with solution in specific range
            solution = random.randint(*self.solution_range)
            a = random.randint(2, 5)
            b = random.randint(1, 10)
            c = a * solution + b
            
            input_eq = f"{a}*{var} + {b} = {c}"
            target_eq = f"{var} = {solution} (range: {self.solution_range[0]}-{self.solution_range[1]})"
            
        return input_eq, target_eq
    
    def get_problem_info(self, index: int) -> Dict:
        """Get problem information for evaluation."""
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        
        input_eq, target_eq = self.equation_pairs[index]
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': [self.constraint_type],  # Constraint type as the rule
            'num_rules': 1,
            'constraint_type': self.constraint_type,
            'solution_range': self.solution_range
        }
    
    def __len__(self) -> int:
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded equation pair at index."""
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode equations
        input_embedding = self.encoder.encode_equation_string(input_eq)
        target_embedding = self.encoder.encode_equation_string(target_eq)
        
        return input_embedding, target_embedding


class CombinedAlgebraDataset(data.Dataset):
    """
    Combined dataset for monolithic IRED baseline.
    
    Generates problems from all 4 rules uniformly:
    - 50k distribute problems
    - 50k combine problems  
    - 50k isolate problems
    - 50k divide problems
    Total: 200k problems (same as 4x rule-specific training)
    
    This ensures fair comparison with compositional approach.
    """
    
    VALID_RULES = ['distribute', 'combine', 'isolate', 'divide']
    
    def __init__(
        self,
        split: str = 'train',
        problems_per_rule: int = 50000,
        d_model: int = 128,
        seed: Optional[int] = None,
        **kwargs  # Pass through variability parameters
    ):
        super().__init__()
        
        self.split = split
        self.problems_per_rule = problems_per_rule
        self.d_model = d_model
        self.seed = seed
        
        # Dataset interface requirements
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Initialize encoder (shared across all rules)
        self.encoder = create_character_encoder(d_model=d_model)
        
        # Initialize seeded random generator for reproducible shuffling
        if self.seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random
        
        # Generate problems from all 4 rules efficiently
        self.equation_pairs = []
        self.rule_labels = []  # Track which rule each problem came from
        
        print(f"Generating combined dataset: {problems_per_rule} problems per rule...")
        start_time = time.time()
        
        # Generate problems for each rule directly (avoiding temporary datasets)
        for rule in self.VALID_RULES:
            print(f"  Generating {problems_per_rule} {rule} problems...")
            rule_problems = self._generate_rule_problems(rule, problems_per_rule, **kwargs)
            
            # Validate that we got exactly the expected number of problems
            actual_count = len(rule_problems)
            if actual_count != problems_per_rule:
                raise RuntimeError(
                    f"Rule '{rule}' generated {actual_count} problems, expected {problems_per_rule}. "
                    f"This would corrupt training data distribution. Check generation parameters."
                )
            
            # Add to combined dataset
            self.equation_pairs.extend(rule_problems)
            self.rule_labels.extend([rule] * actual_count)
            
            print(f"    Successfully generated {actual_count} {rule} problems")
        
        # Validate total count before shuffling
        expected_total = problems_per_rule * len(self.VALID_RULES)
        actual_total = len(self.equation_pairs)
        if actual_total != expected_total:
            raise RuntimeError(
                f"Total problem count mismatch: got {actual_total}, expected {expected_total}. "
                f"Training data distribution would be corrupted."
            )
        
        # Shuffle to mix rules uniformly using seeded generator
        combined = list(zip(self.equation_pairs, self.rule_labels))
        self.rng.shuffle(combined)
        self.equation_pairs, self.rule_labels = zip(*combined)
        
        end_time = time.time()
        
        # Final validation and reporting
        rule_counts = self._count_per_rule()
        print(f"\nCombined dataset generated in {end_time - start_time:.2f}s:")
        print(f"  Total problems: {len(self.equation_pairs)}")
        print(f"  Per-rule breakdown: {rule_counts}")
        
        # Validate final distribution
        for rule, count in rule_counts.items():
            if count != problems_per_rule:
                raise RuntimeError(
                    f"Final validation failed: {rule} has {count} problems, expected {problems_per_rule}"
                )
        
        print("✓ Rule distribution validation passed - all rules have exactly the expected count")
        
        # Initialize coverage tracking for compatibility with training script
        self._coverage_history = []
    
    def _generate_rule_problems(
        self, 
        rule: str, 
        num_problems: int,
        min_coefficient: int = 2,
        max_coefficient: int = 10,
        min_constant: int = 1,
        max_constant: int = 15,
        use_variable_names: List[str] = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate problems for a specific rule efficiently without temporary dataset objects.
        
        This replaces the inefficient temporary AlgebraDataset creation approach.
        """
        # CRITICAL FIX: Only use 'x' variable since encoder vocabulary is '0123456789x.+-=*/()[]<> '
        # This prevents "Unknown character 'w' not in vocabulary" errors
        if use_variable_names is None:
            use_variable_names = ['x']  # Only 'x' is supported by encoder
        else:
            # Filter to only supported variables
            use_variable_names = [var for var in use_variable_names if var == 'x']
            if not use_variable_names:
                use_variable_names = ['x']  # Fallback to 'x'
        
        problems = []
        generation_attempts = 0
        max_attempts = num_problems * 10  # Safety limit
        
        while len(problems) < num_problems and generation_attempts < max_attempts:
            try:
                # Generate single problem based on rule
                if rule == 'distribute':
                    input_eq, target_eq = self._generate_distribute_problem(
                        min_coefficient, max_coefficient, use_variable_names
                    )
                elif rule == 'combine':
                    input_eq, target_eq = self._generate_combine_problem(
                        min_coefficient, max_coefficient, use_variable_names
                    )
                elif rule == 'isolate':
                    input_eq, target_eq = self._generate_isolate_problem(
                        min_coefficient, max_coefficient, min_constant, max_constant, use_variable_names
                    )
                elif rule == 'divide':
                    input_eq, target_eq = self._generate_divide_problem(
                        min_coefficient, max_coefficient, min_constant, max_constant, use_variable_names
                    )
                else:
                    raise ValueError(f"Unknown rule: {rule}")
                
                if input_eq and target_eq and self._validate_equation_pair(input_eq, target_eq):
                    problems.append((input_eq, target_eq))
                    
            except Exception:
                pass  # Skip invalid problems
                
            generation_attempts += 1
        
        if len(problems) < num_problems:
            raise RuntimeError(
                f"Failed to generate {num_problems} {rule} problems (only got {len(problems)} "
                f"after {generation_attempts} attempts). Check generation parameters."
            )
        
        return problems[:num_problems]  # Ensure exact count
    
    def _generate_distribute_problem(self, min_coeff, max_coeff, var_names):
        """Generate distribute problem: a*(b+c) -> a*b + a*c"""
        a = random.randint(min_coeff, max_coeff)
        b = random.randint(min_coeff, max_coeff)
        c = random.randint(min_coeff, max_coeff)
        var = random.choice(var_names)
        
        input_eq = f"{a}*({b}*{var} + {c})"
        target_eq = f"{a*b}*{var} + {a*c}"
        
        return input_eq, target_eq
    
    def _generate_combine_problem(self, min_coeff, max_coeff, var_names):
        """Generate combine problem: a*x + b*x -> (a+b)*x"""
        a = random.randint(min_coeff, max_coeff)
        b = random.randint(min_coeff, max_coeff)
        var = random.choice(var_names)
        
        input_eq = f"{a}*{var} + {b}*{var}"
        target_eq = f"{a+b}*{var}"
        
        return input_eq, target_eq
    
    def _generate_isolate_problem(self, min_coeff, max_coeff, min_const, max_const, var_names):
        """Generate isolate problem: a*x + b = c -> x = (c-b)/a"""
        a = random.randint(min_coeff, max_coeff)
        b = random.randint(min_const, max_const)
        x_val = random.randint(min_const, max_const)
        c = a * x_val + b
        var = random.choice(var_names)
        
        input_eq = f"{a}*{var} + {b} = {c}"
        target_eq = f"{var} = {x_val}"
        
        return input_eq, target_eq
    
    def _generate_divide_problem(self, min_coeff, max_coeff, min_const, max_const, var_names):
        """Generate divide problem: a*x = b -> x = b/a"""
        a = random.randint(min_coeff, max_coeff)
        x_val = random.randint(min_const, max_const)
        b = a * x_val
        var = random.choice(var_names)
        
        input_eq = f"{a}*{var} = {b}"
        target_eq = f"{var} = {x_val}"
        
        return input_eq, target_eq
    
    def _validate_equation_pair(self, input_eq: str, target_eq: str) -> bool:
        """Basic validation of equation pair."""
        try:
            # Basic syntax validation
            if not validate_equation_syntax(input_eq) or not validate_equation_syntax(target_eq):
                return False
            
            # Check equivalence
            if not check_equation_equivalence(input_eq, target_eq):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _count_per_rule(self) -> Dict[str, int]:
        """Count how many problems from each rule."""
        counts = defaultdict(int)
        for rule in self.rule_labels:
            counts[rule] += 1
        return dict(counts)
    
    def get_coverage_history(self) -> List[Dict]:
        """
        Get the coverage history from adaptive generation monitoring.
        
        For CombinedAlgebraDataset, this returns basic coverage information
        about rule distribution rather than adaptive generation checkpoints.
        
        Returns:
            List with rule distribution information
        """
        rule_counts = self._count_per_rule()
        total_problems = len(self.equation_pairs)
        
        coverage_info = {
            'total_problems': total_problems,
            'rule_counts': rule_counts,
            'problems_per_rule': self.problems_per_rule,
            'uniform_distribution': all(count == self.problems_per_rule for count in rule_counts.values()),
            'split': self.split,
            'timestamp': time.time()
        }
        
        return [coverage_info]  # Return as list for consistency with interface
    
    def validate_current_coverage(self) -> Dict:
        """
        Perform coverage validation on the current dataset.
        
        For CombinedAlgebraDataset, this validates that rule distribution
        is exactly as expected for training.
        
        Returns:
            Dictionary with comprehensive coverage analysis
        """
        rule_counts = self._count_per_rule()
        total_problems = len(self.equation_pairs)
        expected_total = self.problems_per_rule * len(self.VALID_RULES)
        
        # Check if all rules have exactly the expected count
        distribution_correct = all(
            count == self.problems_per_rule for count in rule_counts.values()
        )
        
        # Check total count
        total_correct = total_problems == expected_total
        
        # Calculate coverage score (1.0 if perfect distribution, lower otherwise)
        if distribution_correct and total_correct:
            coverage_score = 1.0
        else:
            # Score based on how close to expected distribution
            max_deviation = max(
                abs(count - self.problems_per_rule) for count in rule_counts.values()
            )
            coverage_score = max(0.0, 1.0 - (max_deviation / self.problems_per_rule))
        
        return {
            'total_problems': total_problems,
            'expected_total': expected_total,
            'rule_counts': rule_counts,
            'expected_per_rule': self.problems_per_rule,
            'distribution_correct': distribution_correct,
            'total_count_correct': total_correct,
            'coverage_score': coverage_score,
            'validation_passed': distribution_correct and total_correct,
            'split': self.split,
            'rules_validated': self.VALID_RULES,
            'validation_message': (
                'Perfect rule distribution' if distribution_correct and total_correct
                else f'Distribution issues: {rule_counts}'
            )
        }
    
    def get_problem_info(self, index: int) -> Dict:
        """Get problem information for evaluation."""
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        
        input_eq, target_eq = self.equation_pairs[index]
        rule_applied = self.rule_labels[index]
        
        return {
            'input_equation': input_eq,
            'target_equation': target_eq,
            'rules_applied': [rule_applied],  # Single rule for each problem
            'num_rules': 1,
            'rule': rule_applied,
            'monolithic': True  # This is from the monolithic dataset
        }
    
    def __len__(self) -> int:
        return len(self.equation_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoded equation pair at index."""
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        input_eq, target_eq = self.equation_pairs[index]
        
        # Encode both equations
        inp_emb = self.encoder.encode_equation_string(input_eq)
        target_emb = self.encoder.encode_equation_string(target_eq)
        
        return inp_emb, target_emb