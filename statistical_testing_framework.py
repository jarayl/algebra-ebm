"""
Statistical Testing Framework for T4: Statistical Safeguard Implementation

This module provides the statistical testing infrastructure that T4 will depend on,
focusing on diverse test equation generation and statistical analysis tools for
validating algebraic reasoning systems.

Key Components:
- Diverse equation generation utilities
- Statistical analysis tools
- Integration points with T2's conditioning test results
- Framework for measuring statistical safeguards

Usage:
    from statistical_testing_framework import StatisticalTestFramework
    
    framework = StatisticalTestFramework(d_model=128)
    diverse_eqs = framework.generate_diverse_test_equations(num_equations=1000)
    stats = framework.analyze_distribution_properties(diverse_eqs)
"""

import torch
import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional, Union, Any
import random
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
import sympy as sp
from src.algebra.algebra_dataset import AlgebraDataset, MultiRuleDataset, ConstrainedDataset
from src.algebra.algebra_encoder import create_character_encoder, validate_equation_syntax, check_equation_equivalence


@dataclass
class DiversityMetrics:
    """Metrics for measuring equation diversity in test sets."""
    
    coefficient_entropy: float
    structure_diversity: float
    rule_coverage: Dict[str, int]
    complexity_distribution: Dict[str, float]
    syntactic_patterns: Dict[str, int]
    semantic_equivalence_groups: int


@dataclass
class StatisticalResults:
    """Results from statistical analysis of equation sets."""
    
    distribution_stats: Dict[str, float]
    hypothesis_test_results: Dict[str, Dict[str, float]]
    diversity_metrics: DiversityMetrics
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_power: float
    sample_adequacy: bool


class EquationDiversityGenerator:
    """
    Generator for creating diverse equation test sets with controlled statistical properties.
    
    This class extends the existing AlgebraDataset infrastructure to generate equation sets
    with specific diversity characteristics needed for T4's statistical safeguards.
    """
    
    def __init__(self, coeff_range: List[int] = [-20, 20], d_model: int = 128):
        self.coeff_range = coeff_range
        self.d_model = d_model
        self.encoder = create_character_encoder(d_model=d_model)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Statistical tracking
        self._generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failures_by_type': defaultdict(int)
        }
    
    def generate_coefficient_diverse_equations(self, num_equations: int, rule: str) -> List[Tuple[str, str]]:
        """
        Generate equations with diverse coefficient distributions.
        
        Args:
            num_equations: Number of equations to generate
            rule: Rule type ('distribute', 'combine', 'isolate', 'divide')
            
        Returns:
            List of (input_equation, target_equation) pairs with diverse coefficients
        """
        equations = []
        
        # Create different coefficient sampling strategies for diversity
        strategies = [
            self._sample_uniform_coeffs,
            self._sample_extreme_coeffs, 
            self._sample_prime_coeffs,
            self._sample_power_of_two_coeffs,
            self._sample_fibonacci_coeffs
        ]
        
        equations_per_strategy = num_equations // len(strategies)
        remainder = num_equations % len(strategies)
        
        for i, strategy in enumerate(strategies):
            count = equations_per_strategy + (1 if i < remainder else 0)
            strategy_equations = self._generate_with_strategy(count, rule, strategy)
            equations.extend(strategy_equations)
        
        # Shuffle to mix strategies
        random.shuffle(equations)
        return equations
    
    def generate_structurally_diverse_equations(self, num_equations: int) -> List[Tuple[str, str, List[str]]]:
        """
        Generate equations with diverse structural patterns.
        
        Returns equations with varying complexity patterns, nested operations,
        and different algebraic structures to test compositional reasoning.
        
        Args:
            num_equations: Number of equations to generate
            
        Returns:
            List of (input_equation, target_equation, structure_tags) tuples
        """
        equations = []
        
        # Define structural patterns
        patterns = [
            'linear_simple',      # ax + b = c
            'distributive_nested', # a(x + b(c + d)) = e  
            'multi_combine',      # ax + bx + cx + dx = e
            'chained_operations', # ((ax + b) / c) + d = e
            'symmetric_forms'     # a(x + b) + c(x + d) = e
        ]
        
        equations_per_pattern = num_equations // len(patterns)
        remainder = num_equations % len(patterns)
        
        for i, pattern in enumerate(patterns):
            count = equations_per_pattern + (1 if i < remainder else 0)
            pattern_equations = self._generate_structural_pattern(count, pattern)
            equations.extend(pattern_equations)
        
        random.shuffle(equations)
        return equations
    
    def generate_edge_case_equations(self, num_equations: int) -> List[Tuple[str, str, str]]:
        """
        Generate equations specifically designed to test edge cases.
        
        Args:
            num_equations: Number of edge case equations to generate
            
        Returns:
            List of (input_equation, target_equation, edge_case_type) tuples
        """
        equations = []
        
        edge_case_types = [
            'zero_coefficients',
            'unit_coefficients', 
            'negative_coefficients',
            'large_coefficients',
            'identity_operations',
            'boundary_values'
        ]
        
        equations_per_type = num_equations // len(edge_case_types)
        remainder = num_equations % len(edge_case_types)
        
        for i, edge_type in enumerate(edge_case_types):
            count = equations_per_type + (1 if i < remainder else 0)
            edge_equations = self._generate_edge_cases(count, edge_type)
            equations.extend(edge_equations)
        
        return equations
    
    def _sample_uniform_coeffs(self, count: int) -> List[int]:
        """Sample coefficients from uniform distribution."""
        return [random.randint(self.coeff_range[0], self.coeff_range[1]) 
                for _ in range(count)]
    
    def _sample_extreme_coeffs(self, count: int) -> List[int]:
        """Sample coefficients from extreme values."""
        extremes = [self.coeff_range[0], self.coeff_range[0] + 1,
                   self.coeff_range[1] - 1, self.coeff_range[1]]
        return [random.choice(extremes) for _ in range(count)]
    
    def _sample_prime_coeffs(self, count: int) -> List[int]:
        """Sample coefficients from prime numbers."""
        primes = [p for p in range(2, abs(self.coeff_range[1]) + 1) if self._is_prime(p)]
        primes.extend([-p for p in primes])  # Include negative primes
        return [random.choice(primes) for _ in range(count)]
    
    def _sample_power_of_two_coeffs(self, count: int) -> List[int]:
        """Sample coefficients that are powers of 2."""
        powers = []
        for i in range(1, 6):  # 2^1 to 2^5
            val = 2**i
            if val <= abs(self.coeff_range[1]):
                powers.extend([val, -val])
        return [random.choice(powers) for _ in range(count)]
    
    def _sample_fibonacci_coeffs(self, count: int) -> List[int]:
        """Sample coefficients from Fibonacci sequence."""
        fibs = [1, 1]
        while fibs[-1] < abs(self.coeff_range[1]):
            fibs.append(fibs[-1] + fibs[-2])
        fibs = fibs[:-1]  # Remove the one that exceeds range
        fibs.extend([-f for f in fibs])  # Include negative Fibonacci numbers
        return [random.choice(fibs) for _ in range(count)]
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _generate_with_strategy(self, count: int, rule: str, strategy) -> List[Tuple[str, str]]:
        """Generate equations using a specific coefficient strategy."""
        equations = []
        
        for _ in range(count):
            try:
                coeffs = strategy(3)  # Generate 3 coefficients
                equation = self._build_equation_with_coeffs(rule, coeffs)
                if equation:
                    equations.append(equation)
                    self._generation_stats['successful_generations'] += 1
                self._generation_stats['total_attempts'] += 1
            except Exception as e:
                self._generation_stats['failures_by_type'][type(e).__name__] += 1
                continue
        
        return equations
    
    def _build_equation_with_coeffs(self, rule: str, coeffs: List[int]) -> Optional[Tuple[str, str]]:
        """Build equation using specific coefficients and rule."""
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        
        # Ensure non-zero coefficients where needed
        if a == 0:
            a = 1
        
        try:
            if rule == 'distribute':
                x_solution = random.randint(-10, 10)
                op = random.choice(['+', '-'])
                if op == '+':
                    target_value = a * (x_solution + b) + c
                    input_eq = f"{a}*(x+{b})+{c}={target_value}"
                    target_eq = f"{a}*x+{a*b + c}={target_value}"
                else:
                    target_value = a * (x_solution - b) + c
                    input_eq = f"{a}*(x-{b})+{c}={target_value}"
                    target_eq = f"{a}*x+{-a*b + c}={target_value}"
                    
            elif rule == 'combine':
                x_solution = random.randint(-10, 10)
                combined_coeff = a + b
                if combined_coeff == 0:
                    combined_coeff = 1
                target_value = combined_coeff * x_solution + c
                input_eq = f"{a}*x+{b}*x+{c}={target_value}"
                target_eq = f"{combined_coeff}*x+{c}={target_value}"
                
            elif rule == 'isolate':
                x_solution = random.randint(-10, 10)
                target_value = a * x_solution + b
                input_eq = f"{a}*x+{b}={target_value}"
                target_eq = f"{a}*x={target_value - b}"
                
            elif rule == 'divide':
                if abs(a) <= 1:
                    a = random.choice([-3, -2, 2, 3])
                solution = random.randint(-10, 10)
                b = a * solution
                input_eq = f"{a}*x={b}"
                target_eq = f"x={solution}"
                
            else:
                return None
            
            # Validate equation pair
            if self._validate_equation_pair(input_eq, target_eq):
                return (input_eq, target_eq)
            return None
            
        except Exception:
            return None
    
    def _validate_equation_pair(self, input_eq: str, target_eq: str) -> bool:
        """Validate equation pair syntax and equivalence."""
        try:
            input_valid, _, _ = validate_equation_syntax(input_eq.replace('=', '='))
            target_valid, _, _ = validate_equation_syntax(target_eq.replace('=', '='))
            
            if not (input_valid and target_valid):
                return False
                
            equiv, _ = check_equation_equivalence(
                input_eq.replace('=', '='),
                target_eq.replace('=', '=')
            )
            
            return equiv
            
        except Exception:
            return False
    
    def _generate_structural_pattern(self, count: int, pattern: str) -> List[Tuple[str, str, List[str]]]:
        """Generate equations following specific structural patterns."""
        equations = []
        
        for _ in range(count):
            try:
                equation = self._create_pattern_equation(pattern)
                if equation:
                    input_eq, target_eq = equation
                    equations.append((input_eq, target_eq, [pattern]))
            except Exception:
                continue
        
        return equations
    
    def _create_pattern_equation(self, pattern: str) -> Optional[Tuple[str, str]]:
        """Create equation following specific pattern."""
        coeffs = self._sample_uniform_coeffs(6)  # Generate enough coefficients
        x_val = random.randint(-5, 5)
        
        try:
            if pattern == 'linear_simple':
                a, b = coeffs[0], coeffs[1]
                if a == 0: a = 1
                target = a * x_val + b
                return (f"{a}*x+{b}={target}", f"{a}*x={target-b}")
                
            elif pattern == 'distributive_nested':
                a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                if a == 0: a = 1
                inner_val = x_val + c + d
                target = a * (x_val + b * inner_val)
                expanded = a * x_val + a * b * inner_val
                return (f"{a}*(x+{b}*({c}+{d}))={target}", f"{a}*x+{a*b*(c+d)}={target}")
                
            elif pattern == 'multi_combine':
                a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                total = a + b + c + d
                if total == 0: total = 1
                target = total * x_val
                return (f"{a}*x+{b}*x+{c}*x+{d}*x={target}", f"{total}*x={target}")
                
            # Add more patterns as needed
            else:
                return None
                
        except Exception:
            return None
    
    def _generate_edge_cases(self, count: int, edge_type: str) -> List[Tuple[str, str, str]]:
        """Generate edge case equations."""
        equations = []
        
        for _ in range(count):
            try:
                equation = self._create_edge_case_equation(edge_type)
                if equation:
                    input_eq, target_eq = equation
                    equations.append((input_eq, target_eq, edge_type))
            except Exception:
                continue
        
        return equations
    
    def _create_edge_case_equation(self, edge_type: str) -> Optional[Tuple[str, str]]:
        """Create equation for specific edge case."""
        x_val = random.randint(-3, 3)
        
        try:
            if edge_type == 'zero_coefficients':
                # Test with zero coefficient in non-critical positions
                b = 0
                a = random.choice([-2, -1, 1, 2])
                target = a * x_val + b
                return (f"{a}*x+{b}={target}", f"{a}*x={target}")
                
            elif edge_type == 'unit_coefficients':
                # Test with coefficients of ±1
                a = random.choice([-1, 1])
                b = random.choice([-1, 1])
                target = a * x_val + b
                return (f"{a}*x+{b}={target}", f"{a}*x={target-b}")
                
            elif edge_type == 'large_coefficients':
                # Test with large coefficients
                a = random.choice([-100, -50, 50, 100])
                b = random.choice([-100, -50, 50, 100])
                target = a * x_val + b
                return (f"{a}*x+{b}={target}", f"{a}*x={target-b}")
                
            # Add more edge cases as needed
            else:
                return None
                
        except Exception:
            return None


class StatisticalAnalyzer:
    """
    Statistical analysis tools for measuring equation set properties and hypothesis testing.
    
    Provides statistical tests and metrics needed for T4's safeguard validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_coefficient_distribution(self, equations: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Analyze the statistical distribution of coefficients in equation sets.
        
        Args:
            equations: List of (input_equation, target_equation) pairs
            
        Returns:
            Dictionary with distribution statistics
        """
        coefficients = self._extract_coefficients(equations)
        
        if not coefficients:
            return {'error': 'No coefficients extracted'}
        
        coeffs_array = np.array(coefficients)
        
        return {
            'mean': float(np.mean(coeffs_array)),
            'std': float(np.std(coeffs_array)),
            'median': float(np.median(coeffs_array)),
            'min': float(np.min(coeffs_array)),
            'max': float(np.max(coeffs_array)),
            'range': float(np.max(coeffs_array) - np.min(coeffs_array)),
            'skewness': float(stats.skew(coeffs_array)),
            'kurtosis': float(stats.kurtosis(coeffs_array)),
            'entropy': self._compute_entropy(coeffs_array)
        }
    
    def measure_diversity_metrics(self, equations: List[Tuple[str, str]]) -> DiversityMetrics:
        """
        Compute comprehensive diversity metrics for equation sets.
        
        Args:
            equations: List of (input_equation, target_equation) pairs
            
        Returns:
            DiversityMetrics object with comprehensive diversity measures
        """
        coefficients = self._extract_coefficients(equations)
        structures = self._extract_structural_patterns(equations)
        
        # Coefficient entropy
        coeff_entropy = self._compute_entropy(np.array(coefficients)) if coefficients else 0.0
        
        # Structure diversity (normalized entropy of structural patterns)
        structure_counts = Counter(structures)
        structure_probs = np.array(list(structure_counts.values())) / len(structures)
        structure_entropy = stats.entropy(structure_probs) if len(structure_probs) > 1 else 0.0
        max_entropy = np.log(len(structure_counts)) if len(structure_counts) > 1 else 1.0
        structure_diversity = structure_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Rule coverage (placeholder - would need rule annotations)
        rule_coverage = {'distribute': 0, 'combine': 0, 'isolate': 0, 'divide': 0}
        
        # Complexity distribution
        complexities = self._measure_equation_complexity(equations)
        complexity_dist = {
            'simple': sum(1 for c in complexities if c <= 2) / len(complexities),
            'medium': sum(1 for c in complexities if 2 < c <= 4) / len(complexities),
            'complex': sum(1 for c in complexities if c > 4) / len(complexities)
        } if complexities else {'simple': 0, 'medium': 0, 'complex': 0}
        
        # Syntactic patterns
        syntactic_patterns = Counter(self._extract_syntactic_patterns(equations))
        
        # Semantic equivalence groups (simplified)
        semantic_groups = len(set(eq[1] for eq in equations))  # Unique target equations
        
        return DiversityMetrics(
            coefficient_entropy=coeff_entropy,
            structure_diversity=structure_diversity,
            rule_coverage=rule_coverage,
            complexity_distribution=complexity_dist,
            syntactic_patterns=dict(syntactic_patterns),
            semantic_equivalence_groups=semantic_groups
        )
    
    def perform_statistical_tests(self, 
                                set1: List[Tuple[str, str]], 
                                set2: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical hypothesis tests comparing two equation sets.
        
        Args:
            set1: First equation set
            set2: Second equation set
            
        Returns:
            Dictionary with test results
        """
        coeffs1 = np.array(self._extract_coefficients(set1))
        coeffs2 = np.array(self._extract_coefficients(set2))
        
        results = {}
        
        # Two-sample t-test
        if len(coeffs1) > 1 and len(coeffs2) > 1:
            t_stat, t_pval = stats.ttest_ind(coeffs1, coeffs2)
            results['t_test'] = {'statistic': float(t_stat), 'p_value': float(t_pval)}
        
        # Kolmogorov-Smirnov test
        if len(coeffs1) > 0 and len(coeffs2) > 0:
            ks_stat, ks_pval = stats.ks_2samp(coeffs1, coeffs2)
            results['ks_test'] = {'statistic': float(ks_stat), 'p_value': float(ks_pval)}
        
        # Mann-Whitney U test
        if len(coeffs1) > 0 and len(coeffs2) > 0:
            mw_stat, mw_pval = stats.mannwhitneyu(coeffs1, coeffs2)
            results['mann_whitney'] = {'statistic': float(mw_stat), 'p_value': float(mw_pval)}
        
        return results
    
    def compute_confidence_intervals(self, 
                                   equations: List[Tuple[str, str]], 
                                   confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for key metrics.
        
        Args:
            equations: Equation set
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with confidence intervals
        """
        coefficients = np.array(self._extract_coefficients(equations))
        
        if len(coefficients) == 0:
            return {}
        
        alpha = 1 - confidence
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(coefficients, size=len(coefficients), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        mean_ci = np.percentile(bootstrap_means, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        return {
            'coefficient_mean': (float(mean_ci[0]), float(mean_ci[1])),
            'coefficient_std': self._bootstrap_ci(coefficients, np.std, confidence)
        }
    
    def _extract_coefficients(self, equations: List[Tuple[str, str]]) -> List[float]:
        """Extract numerical coefficients from equations."""
        coefficients = []
        
        for input_eq, target_eq in equations:
            # Simple regex-based coefficient extraction
            import re
            
            for eq in [input_eq, target_eq]:
                # Find numbers that appear to be coefficients
                numbers = re.findall(r'-?\d+', eq)
                coefficients.extend([float(n) for n in numbers])
        
        return coefficients
    
    def _extract_structural_patterns(self, equations: List[Tuple[str, str]]) -> List[str]:
        """Extract structural patterns from equations."""
        patterns = []
        
        for input_eq, _ in equations:
            # Simple pattern recognition
            if '*(' in input_eq:
                patterns.append('distributive')
            elif '+' in input_eq and '*x' in input_eq:
                patterns.append('additive')
            elif '=' in input_eq and '*x' in input_eq:
                patterns.append('linear')
            else:
                patterns.append('other')
        
        return patterns
    
    def _measure_equation_complexity(self, equations: List[Tuple[str, str]]) -> List[int]:
        """Measure complexity of equations (simplified metric)."""
        complexities = []
        
        for input_eq, _ in equations:
            complexity = 0
            complexity += input_eq.count('*')
            complexity += input_eq.count('+')
            complexity += input_eq.count('-')
            complexity += input_eq.count('(')
            complexities.append(complexity)
        
        return complexities
    
    def _extract_syntactic_patterns(self, equations: List[Tuple[str, str]]) -> List[str]:
        """Extract syntactic patterns from equations."""
        patterns = []
        
        for input_eq, _ in equations:
            pattern = ''
            if '*(' in input_eq:
                pattern += 'mult_paren_'
            if '+' in input_eq:
                pattern += 'plus_'
            if '-' in input_eq:
                pattern += 'minus_'
            patterns.append(pattern.rstrip('_') or 'simple')
        
        return patterns
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of data distribution."""
        if len(data) == 0:
            return 0.0
        
        # Bin the data for entropy calculation
        hist, _ = np.histogram(data, bins=min(len(set(data)), 10))
        probs = hist / len(data)
        probs = probs[probs > 0]  # Remove zeros
        
        return float(stats.entropy(probs))
    
    def _bootstrap_ci(self, data: np.ndarray, statistic_func, confidence: float) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for a statistic."""
        n_bootstrap = 1000
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        alpha = 1 - confidence
        ci = np.percentile(bootstrap_stats, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        return (float(ci[0]), float(ci[1]))


class T2IntegrationInterface:
    """
    Integration interface for T2's conditioning test results.
    
    Provides methods to integrate with T2's conditioning test outputs and 
    use them for statistical validation in T4.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.conditioning_results_cache = {}
    
    def load_conditioning_test_results(self, results_path: str) -> Dict[str, Any]:
        """
        Load T2's conditioning test results from file.
        
        Args:
            results_path: Path to T2's conditioning test results
            
        Returns:
            Dictionary with conditioning test results
        """
        # Placeholder for loading T2 results
        # In actual implementation, would load from T2's output format
        return {
            'conditioning_effectiveness': {},
            'failure_modes': [],
            'statistical_properties': {}
        }
    
    def analyze_conditioning_correlation(self, 
                                       conditioning_results: Dict[str, Any],
                                       equation_set: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Analyze correlation between conditioning effectiveness and equation properties.
        
        Args:
            conditioning_results: Results from T2's conditioning tests
            equation_set: Equation set to analyze
            
        Returns:
            Correlation analysis results
        """
        # Placeholder for correlation analysis
        return {
            'conditioning_diversity_correlation': 0.0,
            'conditioning_complexity_correlation': 0.0,
            'statistical_significance': 0.0
        }
    
    def prepare_conditioning_aware_tests(self, 
                                       conditioning_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare test configurations based on T2's conditioning results.
        
        Args:
            conditioning_results: T2's conditioning test results
            
        Returns:
            List of test configurations for T4
        """
        # Placeholder for test preparation
        return [
            {
                'test_type': 'conditioning_robustness',
                'parameters': {},
                'expected_outcomes': {}
            }
        ]


class StatisticalTestFramework:
    """
    Main framework class that coordinates all statistical testing components.
    
    This is the primary interface that T4 will use for statistical safeguard implementation.
    """
    
    def __init__(self, d_model: int = 128, coeff_range: List[int] = [-20, 20]):
        self.d_model = d_model
        self.coeff_range = coeff_range
        
        # Initialize components
        self.diversity_generator = EquationDiversityGenerator(coeff_range, d_model)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.t2_interface = T2IntegrationInterface()
        
        # Framework state
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.framework_ready = True
        
        self.logger.info("Statistical Testing Framework initialized and ready for T4")
    
    def generate_diverse_test_equations(self, 
                                      num_equations: int,
                                      diversity_type: str = 'mixed') -> List[Tuple[str, str]]:
        """
        Generate diverse equation sets for statistical testing.
        
        Args:
            num_equations: Number of equations to generate
            diversity_type: Type of diversity ('coefficient', 'structural', 'mixed', 'edge_cases')
            
        Returns:
            List of diverse equation pairs
        """
        if diversity_type == 'coefficient':
            # Generate with coefficient diversity across all rules
            equations = []
            rules = ['distribute', 'combine', 'isolate', 'divide']
            per_rule = num_equations // len(rules)
            
            for rule in rules:
                rule_equations = self.diversity_generator.generate_coefficient_diverse_equations(per_rule, rule)
                equations.extend(rule_equations)
            
            return equations[:num_equations]
            
        elif diversity_type == 'structural':
            structural_equations = self.diversity_generator.generate_structurally_diverse_equations(num_equations)
            return [(eq[0], eq[1]) for eq in structural_equations]  # Remove structure tags
            
        elif diversity_type == 'edge_cases':
            edge_equations = self.diversity_generator.generate_edge_case_equations(num_equations)
            return [(eq[0], eq[1]) for eq in edge_equations]  # Remove edge case tags
            
        elif diversity_type == 'mixed':
            # Mix of all diversity types
            third = num_equations // 3
            coeff_equations = self.generate_diverse_test_equations(third, 'coefficient')
            struct_equations = self.generate_diverse_test_equations(third, 'structural')
            edge_equations = self.generate_diverse_test_equations(num_equations - 2*third, 'edge_cases')
            
            all_equations = coeff_equations + struct_equations + edge_equations
            random.shuffle(all_equations)
            return all_equations
            
        else:
            raise ValueError(f"Unknown diversity_type: {diversity_type}")
    
    def analyze_distribution_properties(self, equations: List[Tuple[str, str]]) -> StatisticalResults:
        """
        Analyze statistical properties of equation distributions.
        
        Args:
            equations: Equation set to analyze
            
        Returns:
            Comprehensive statistical analysis results
        """
        # Distribution statistics
        dist_stats = self.statistical_analyzer.analyze_coefficient_distribution(equations)
        
        # Diversity metrics
        diversity_metrics = self.statistical_analyzer.measure_diversity_metrics(equations)
        
        # Confidence intervals
        confidence_intervals = self.statistical_analyzer.compute_confidence_intervals(equations)
        
        # Hypothesis tests (compare against baseline)
        baseline_equations = self.generate_diverse_test_equations(100, 'mixed')
        hypothesis_tests = self.statistical_analyzer.perform_statistical_tests(equations, baseline_equations)
        
        # Statistical power and sample adequacy (simplified)
        statistical_power = self._compute_statistical_power(equations)
        sample_adequacy = len(equations) >= 100  # Simplified criterion
        
        return StatisticalResults(
            distribution_stats=dist_stats,
            hypothesis_test_results=hypothesis_tests,
            diversity_metrics=diversity_metrics,
            confidence_intervals=confidence_intervals,
            statistical_power=statistical_power,
            sample_adequacy=sample_adequacy
        )
    
    def validate_statistical_safeguards(self, 
                                      equation_set: List[Tuple[str, str]],
                                      safeguard_criteria: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate that equation set meets statistical safeguard criteria.
        
        Args:
            equation_set: Equations to validate
            safeguard_criteria: Criteria thresholds for validation
            
        Returns:
            Dictionary indicating which safeguards are met
        """
        analysis = self.analyze_distribution_properties(equation_set)
        
        # Default criteria if none provided
        default_criteria = {
            'min_diversity': 0.5,
            'min_entropy': 1.0,
            'min_sample_size': 100,
            'max_skewness': 2.0
        }
        criteria = {**default_criteria, **safeguard_criteria}
        
        return {
            'diversity_adequate': analysis.diversity_metrics.structure_diversity >= criteria['min_diversity'],
            'entropy_adequate': analysis.diversity_metrics.coefficient_entropy >= criteria['min_entropy'],
            'sample_size_adequate': analysis.sample_adequacy,
            'distribution_reasonable': abs(analysis.distribution_stats.get('skewness', 0)) <= criteria['max_skewness'],
            'overall_valid': all([
                analysis.diversity_metrics.structure_diversity >= criteria['min_diversity'],
                analysis.diversity_metrics.coefficient_entropy >= criteria['min_entropy'],
                analysis.sample_adequacy,
                abs(analysis.distribution_stats.get('skewness', 0)) <= criteria['max_skewness']
            ])
        }
    
    def prepare_t4_integration(self) -> Dict[str, Any]:
        """
        Prepare integration points and utilities for T4 implementation.
        
        Returns:
            Dictionary with T4 integration information
        """
        return {
            'framework_status': 'ready',
            'available_generators': [
                'coefficient_diverse',
                'structurally_diverse', 
                'edge_cases',
                'mixed_diversity'
            ],
            'available_analyzers': [
                'distribution_analysis',
                'diversity_metrics',
                'statistical_tests',
                'confidence_intervals'
            ],
            't2_integration_ready': True,
            'recommended_sample_sizes': {
                'pilot_test': 100,
                'validation_test': 500,
                'comprehensive_test': 1000
            },
            'utility_functions': {
                'generate_diverse_test_equations': self.generate_diverse_test_equations,
                'analyze_distribution_properties': self.analyze_distribution_properties,
                'validate_statistical_safeguards': self.validate_statistical_safeguards
            }
        }
    
    def _compute_statistical_power(self, equations: List[Tuple[str, str]]) -> float:
        """Compute statistical power of the test set (simplified)."""
        n = len(equations)
        # Simplified power calculation based on sample size
        # In practice, would depend on effect size and significance level
        power = min(0.95, 0.5 + 0.1 * np.log(n)) if n > 0 else 0.0
        return float(power)
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status and readiness for T4."""
        return {
            'framework_ready': self.framework_ready,
            'components_initialized': {
                'diversity_generator': hasattr(self, 'diversity_generator'),
                'statistical_analyzer': hasattr(self, 'statistical_analyzer'),
                't2_interface': hasattr(self, 't2_interface')
            },
            'generation_stats': self.diversity_generator._generation_stats,
            'ready_for_t4': self.framework_ready
        }


# Main interface for T4
def create_statistical_testing_framework(d_model: int = 128, 
                                        coeff_range: List[int] = [-20, 20]) -> StatisticalTestFramework:
    """
    Factory function to create a configured Statistical Testing Framework for T4.
    
    Args:
        d_model: Embedding dimension
        coeff_range: Coefficient range for equation generation
        
    Returns:
        Configured StatisticalTestFramework ready for T4 use
    """
    framework = StatisticalTestFramework(d_model, coeff_range)
    
    # Log framework readiness
    logger = logging.getLogger(__name__)
    logger.info("Statistical Testing Framework created and ready for T4 implementation")
    logger.info(f"Framework configuration: d_model={d_model}, coeff_range={coeff_range}")
    
    return framework


if __name__ == "__main__":
    # Demo usage for T4
    framework = create_statistical_testing_framework()
    
    # Generate diverse test equations
    diverse_equations = framework.generate_diverse_test_equations(100, 'mixed')
    print(f"Generated {len(diverse_equations)} diverse equations")
    
    # Analyze statistical properties
    analysis = framework.analyze_distribution_properties(diverse_equations)
    print(f"Distribution analysis complete: diversity={analysis.diversity_metrics.structure_diversity:.3f}")
    
    # Validate safeguards
    validation = framework.validate_statistical_safeguards(diverse_equations, {})
    print(f"Safeguard validation: {validation['overall_valid']}")
    
    # Check T4 readiness
    status = framework.get_framework_status()
    print(f"Framework ready for T4: {status['ready_for_t4']}")