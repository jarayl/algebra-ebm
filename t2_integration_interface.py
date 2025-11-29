"""
T2 Integration Interface for Statistical Testing Framework

This module provides the integration interface between T2's conditioning test results
and T4's statistical safeguard implementation. Handles data exchange, analysis
correlation, and coordinated testing workflows.

Key Components:
- ConditioningResultsParser: Parses T2 conditioning test outputs
- ConditioningCorrelationAnalyzer: Analyzes correlations between conditioning and statistics
- IntegratedTestCoordinator: Coordinates testing across T2 and T4
- SafeguardValidationBridge: Bridges conditioning tests with statistical safeguards

Usage:
    from t2_integration_interface import create_t2_integration_interface
    
    interface = create_t2_integration_interface()
    conditioning_data = interface.load_t2_results("path/to/t2/results")
    correlation_analysis = interface.analyze_conditioning_correlation(conditioning_data, equations)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import warnings

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ConditioningTestResult:
    """Single conditioning test result from T2."""
    
    test_id: str
    rule_type: str
    equation_input: str
    equation_target: str
    conditioning_method: str
    effectiveness_score: float
    failure_mode: Optional[str]
    computational_cost: float
    convergence_achieved: bool
    metadata: Dict[str, Any]


@dataclass
class ConditioningBatchResults:
    """Batch of conditioning test results from T2."""
    
    batch_id: str
    test_configuration: Dict[str, Any]
    individual_results: List[ConditioningTestResult]
    overall_statistics: Dict[str, float]
    timestamp: str
    t2_version: str


@dataclass
class CorrelationAnalysisResult:
    """Result of correlation analysis between conditioning and statistical properties."""
    
    conditioning_diversity_correlation: float
    conditioning_complexity_correlation: float
    conditioning_coefficient_correlation: float
    effectiveness_distribution_correlation: float
    failure_pattern_analysis: Dict[str, Any]
    statistical_significance: Dict[str, float]
    recommendations: List[str]


@dataclass
class IntegratedTestPlan:
    """Integrated test plan combining T2 conditioning tests with T4 statistical tests."""
    
    test_plan_id: str
    conditioning_test_configs: List[Dict[str, Any]]
    statistical_test_configs: List[Dict[str, Any]]
    coordination_parameters: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    validation_criteria: Dict[str, float]


class ConditioningResultsParser:
    """
    Parser for T2 conditioning test result files.
    
    Handles multiple file formats and provides standardized access to conditioning data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.supported_formats = ['json', 'pickle', 'yaml', 'csv']
    
    def load_conditioning_results(self, file_path: Union[str, Path]) -> ConditioningBatchResults:
        """
        Load conditioning test results from file.
        
        Args:
            file_path: Path to T2 conditioning results file
            
        Returns:
            ConditioningBatchResults object with parsed data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Conditioning results file not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.json':
                return self._load_from_json(file_path)
            elif file_extension == '.pkl':
                return self._load_from_pickle(file_path)
            elif file_extension in ['.yml', '.yaml']:
                return self._load_from_yaml(file_path)
            elif file_extension == '.csv':
                return self._load_from_csv(file_path)
            else:
                # Try to auto-detect format
                return self._auto_detect_and_load(file_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load conditioning results from {file_path}: {str(e)}")
            raise
    
    def parse_conditioning_result_dict(self, result_dict: Dict[str, Any]) -> ConditioningTestResult:
        """
        Parse a single conditioning result from dictionary format.
        
        Args:
            result_dict: Dictionary containing conditioning test result
            
        Returns:
            ConditioningTestResult object
        """
        # Extract required fields with defaults
        test_id = result_dict.get('test_id', 'unknown')
        rule_type = result_dict.get('rule_type', 'unknown')
        equation_input = result_dict.get('equation_input', '')
        equation_target = result_dict.get('equation_target', '')
        conditioning_method = result_dict.get('conditioning_method', 'unknown')
        effectiveness_score = float(result_dict.get('effectiveness_score', 0.0))
        failure_mode = result_dict.get('failure_mode')
        computational_cost = float(result_dict.get('computational_cost', 0.0))
        convergence_achieved = bool(result_dict.get('convergence_achieved', False))
        metadata = result_dict.get('metadata', {})
        
        return ConditioningTestResult(
            test_id=test_id,
            rule_type=rule_type,
            equation_input=equation_input,
            equation_target=equation_target,
            conditioning_method=conditioning_method,
            effectiveness_score=effectiveness_score,
            failure_mode=failure_mode,
            computational_cost=computational_cost,
            convergence_achieved=convergence_achieved,
            metadata=metadata
        )
    
    def validate_conditioning_results(self, results: ConditioningBatchResults) -> Dict[str, bool]:
        """
        Validate the completeness and consistency of conditioning results.
        
        Args:
            results: ConditioningBatchResults to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_results': len(results.individual_results) > 0,
            'all_results_valid': True,
            'effectiveness_scores_valid': True,
            'equations_present': True,
            'metadata_consistent': True
        }
        
        # Check individual results
        for result in results.individual_results:
            if not result.equation_input or not result.equation_target:
                validation['equations_present'] = False
            
            if not (0.0 <= result.effectiveness_score <= 1.0):
                validation['effectiveness_scores_valid'] = False
            
            if not result.test_id or not result.rule_type:
                validation['all_results_valid'] = False
        
        # Check metadata consistency
        rule_types = set(r.rule_type for r in results.individual_results)
        if len(rule_types) == 0:
            validation['metadata_consistent'] = False
        
        return validation
    
    def _load_from_json(self, file_path: Path) -> ConditioningBatchResults:
        """Load conditioning results from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return self._parse_batch_dict(data)
    
    def _load_from_pickle(self, file_path: Path) -> ConditioningBatchResults:
        """Load conditioning results from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, ConditioningBatchResults):
            return data
        elif isinstance(data, dict):
            return self._parse_batch_dict(data)
        else:
            raise ValueError(f"Unsupported pickle data type: {type(data)}")
    
    def _load_from_yaml(self, file_path: Path) -> ConditioningBatchResults:
        """Load conditioning results from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML files")
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return self._parse_batch_dict(data)
    
    def _load_from_csv(self, file_path: Path) -> ConditioningBatchResults:
        """Load conditioning results from CSV file."""
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to batch results
        individual_results = []
        for _, row in df.iterrows():
            result_dict = row.to_dict()
            individual_results.append(self.parse_conditioning_result_dict(result_dict))
        
        # Create batch results
        overall_stats = {
            'mean_effectiveness': df['effectiveness_score'].mean() if 'effectiveness_score' in df.columns else 0.0,
            'total_tests': len(df)
        }
        
        return ConditioningBatchResults(
            batch_id=f"csv_batch_{file_path.stem}",
            test_configuration={},
            individual_results=individual_results,
            overall_statistics=overall_stats,
            timestamp="unknown",
            t2_version="unknown"
        )
    
    def _auto_detect_and_load(self, file_path: Path) -> ConditioningBatchResults:
        """Auto-detect file format and load."""
        # Try JSON first
        try:
            return self._load_from_json(file_path)
        except Exception:
            pass
        
        # Try pickle
        try:
            return self._load_from_pickle(file_path)
        except Exception:
            pass
        
        # Try YAML if available
        if yaml is not None:
            try:
                return self._load_from_yaml(file_path)
            except Exception:
                pass
        
        raise ValueError(f"Could not auto-detect format for file: {file_path}")
    
    def _parse_batch_dict(self, data: Dict[str, Any]) -> ConditioningBatchResults:
        """Parse batch results dictionary."""
        # Extract batch-level information
        batch_id = data.get('batch_id', 'unknown')
        test_configuration = data.get('test_configuration', {})
        overall_statistics = data.get('overall_statistics', {})
        timestamp = data.get('timestamp', 'unknown')
        t2_version = data.get('t2_version', 'unknown')
        
        # Parse individual results
        individual_results = []
        results_data = data.get('individual_results', data.get('results', []))
        
        for result_dict in results_data:
            individual_results.append(self.parse_conditioning_result_dict(result_dict))
        
        return ConditioningBatchResults(
            batch_id=batch_id,
            test_configuration=test_configuration,
            individual_results=individual_results,
            overall_statistics=overall_statistics,
            timestamp=timestamp,
            t2_version=t2_version
        )


class ConditioningCorrelationAnalyzer:
    """
    Analyzer for correlations between conditioning effectiveness and statistical properties.
    
    Provides methods to analyze how conditioning test results correlate with
    equation diversity, complexity, and other statistical properties.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_conditioning_correlation(self,
                                       conditioning_results: ConditioningBatchResults,
                                       equation_properties: Dict[str, Dict[str, Any]]) -> CorrelationAnalysisResult:
        """
        Analyze correlations between conditioning effectiveness and equation properties.
        
        Args:
            conditioning_results: T2 conditioning test results
            equation_properties: Statistical properties of equations
            
        Returns:
            CorrelationAnalysisResult with comprehensive correlation analysis
        """
        # Extract effectiveness scores and equation properties
        effectiveness_scores = [r.effectiveness_score for r in conditioning_results.individual_results]
        
        if not effectiveness_scores:
            return self._empty_correlation_result()
        
        # Calculate correlations
        diversity_corr = self._calculate_diversity_correlation(conditioning_results, equation_properties)
        complexity_corr = self._calculate_complexity_correlation(conditioning_results, equation_properties)
        coefficient_corr = self._calculate_coefficient_correlation(conditioning_results, equation_properties)
        distribution_corr = self._calculate_distribution_correlation(conditioning_results, equation_properties)
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure_patterns(conditioning_results)
        
        # Calculate statistical significance
        significance = self._calculate_correlation_significance(conditioning_results, equation_properties)
        
        # Generate recommendations
        recommendations = self._generate_correlation_recommendations(
            diversity_corr, complexity_corr, coefficient_corr, failure_analysis
        )
        
        return CorrelationAnalysisResult(
            conditioning_diversity_correlation=diversity_corr,
            conditioning_complexity_correlation=complexity_corr,
            conditioning_coefficient_correlation=coefficient_corr,
            effectiveness_distribution_correlation=distribution_corr,
            failure_pattern_analysis=failure_analysis,
            statistical_significance=significance,
            recommendations=recommendations
        )
    
    def analyze_conditioning_robustness(self,
                                      conditioning_results: ConditioningBatchResults) -> Dict[str, float]:
        """
        Analyze robustness metrics of conditioning tests.
        
        Args:
            conditioning_results: T2 conditioning test results
            
        Returns:
            Dictionary with robustness metrics
        """
        if not conditioning_results.individual_results:
            return {}
        
        effectiveness_scores = [r.effectiveness_score for r in conditioning_results.individual_results]
        convergence_rates = [r.convergence_achieved for r in conditioning_results.individual_results]
        computational_costs = [r.computational_cost for r in conditioning_results.individual_results]
        
        return {
            'effectiveness_mean': float(np.mean(effectiveness_scores)),
            'effectiveness_std': float(np.std(effectiveness_scores)),
            'effectiveness_cv': float(np.std(effectiveness_scores) / np.mean(effectiveness_scores)) if np.mean(effectiveness_scores) > 0 else 0.0,
            'convergence_rate': float(np.mean(convergence_rates)),
            'cost_efficiency': float(np.mean(effectiveness_scores) / np.mean(computational_costs)) if np.mean(computational_costs) > 0 else 0.0,
            'robustness_score': self._calculate_robustness_score(effectiveness_scores, convergence_rates)
        }
    
    def identify_conditioning_patterns(self,
                                     conditioning_results: ConditioningBatchResults) -> Dict[str, Any]:
        """
        Identify patterns in conditioning test results.
        
        Args:
            conditioning_results: T2 conditioning test results
            
        Returns:
            Dictionary with identified patterns
        """
        if not conditioning_results.individual_results:
            return {}
        
        # Group by rule type
        rule_groups = defaultdict(list)
        for result in conditioning_results.individual_results:
            rule_groups[result.rule_type].append(result)
        
        # Analyze patterns by rule type
        rule_patterns = {}
        for rule_type, results in rule_groups.items():
            effectiveness = [r.effectiveness_score for r in results]
            convergence = [r.convergence_achieved for r in results]
            
            rule_patterns[rule_type] = {
                'count': len(results),
                'mean_effectiveness': float(np.mean(effectiveness)),
                'effectiveness_range': float(np.max(effectiveness) - np.min(effectiveness)),
                'convergence_rate': float(np.mean(convergence)),
                'stability': 1.0 - (np.std(effectiveness) / np.mean(effectiveness)) if np.mean(effectiveness) > 0 else 0.0
            }
        
        # Overall pattern analysis
        overall_effectiveness = [r.effectiveness_score for r in conditioning_results.individual_results]
        
        return {
            'rule_type_patterns': rule_patterns,
            'overall_patterns': {
                'effectiveness_distribution': self._analyze_distribution_shape(overall_effectiveness),
                'bimodal_tendency': self._detect_bimodal_tendency(overall_effectiveness),
                'outlier_percentage': self._calculate_outlier_percentage(overall_effectiveness)
            }
        }
    
    def _calculate_diversity_correlation(self,
                                       conditioning_results: ConditioningBatchResults,
                                       equation_properties: Dict[str, Dict[str, Any]]) -> float:
        """Calculate correlation between conditioning effectiveness and equation diversity."""
        try:
            # Extract diversity metrics if available
            diversity_scores = []
            effectiveness_scores = []
            
            for result in conditioning_results.individual_results:
                eq_id = result.test_id
                if eq_id in equation_properties:
                    props = equation_properties[eq_id]
                    diversity_score = props.get('diversity_metrics', {}).get('structure_diversity', 0.0)
                    diversity_scores.append(diversity_score)
                    effectiveness_scores.append(result.effectiveness_score)
            
            if len(diversity_scores) >= 3:
                return float(np.corrcoef(diversity_scores, effectiveness_scores)[0, 1])
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_complexity_correlation(self,
                                        conditioning_results: ConditioningBatchResults,
                                        equation_properties: Dict[str, Dict[str, Any]]) -> float:
        """Calculate correlation between conditioning effectiveness and equation complexity."""
        try:
            complexity_scores = []
            effectiveness_scores = []
            
            for result in conditioning_results.individual_results:
                # Simple complexity measure based on equation structure
                complexity = self._estimate_equation_complexity(result.equation_input)
                complexity_scores.append(complexity)
                effectiveness_scores.append(result.effectiveness_score)
            
            if len(complexity_scores) >= 3:
                return float(np.corrcoef(complexity_scores, effectiveness_scores)[0, 1])
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_coefficient_correlation(self,
                                         conditioning_results: ConditioningBatchResults,
                                         equation_properties: Dict[str, Dict[str, Any]]) -> float:
        """Calculate correlation between conditioning effectiveness and coefficient properties."""
        try:
            coefficient_ranges = []
            effectiveness_scores = []
            
            for result in conditioning_results.individual_results:
                coeff_range = self._estimate_coefficient_range(result.equation_input)
                coefficient_ranges.append(coeff_range)
                effectiveness_scores.append(result.effectiveness_score)
            
            if len(coefficient_ranges) >= 3:
                return float(np.corrcoef(coefficient_ranges, effectiveness_scores)[0, 1])
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_distribution_correlation(self,
                                          conditioning_results: ConditioningBatchResults,
                                          equation_properties: Dict[str, Dict[str, Any]]) -> float:
        """Calculate correlation between conditioning effectiveness distribution and equation properties."""
        try:
            effectiveness_scores = [r.effectiveness_score for r in conditioning_results.individual_results]
            if len(effectiveness_scores) < 3:
                return 0.0
            
            # Analyze effectiveness distribution shape
            distribution_skew = float(self._calculate_skewness(effectiveness_scores))
            
            # Correlate with equation property distributions
            if equation_properties:
                prop_values = []
                for eq_props in equation_properties.values():
                    if 'distribution_stats' in eq_props:
                        prop_values.append(eq_props['distribution_stats'].get('mean', 0.0))
                
                if prop_values and len(prop_values) >= 3:
                    prop_skew = float(self._calculate_skewness(prop_values))
                    return abs(distribution_skew - prop_skew)  # Similarity measure
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_failure_patterns(self, conditioning_results: ConditioningBatchResults) -> Dict[str, Any]:
        """Analyze patterns in conditioning test failures."""
        failures = [r for r in conditioning_results.individual_results if r.failure_mode is not None]
        
        if not failures:
            return {'no_failures': True, 'failure_rate': 0.0}
        
        failure_modes = defaultdict(int)
        for failure in failures:
            failure_modes[failure.failure_mode] += 1
        
        total_tests = len(conditioning_results.individual_results)
        failure_rate = len(failures) / total_tests if total_tests > 0 else 0.0
        
        return {
            'failure_rate': float(failure_rate),
            'failure_modes': dict(failure_modes),
            'most_common_failure': max(failure_modes.keys(), key=failure_modes.get) if failure_modes else None,
            'failure_concentration': max(failure_modes.values()) / len(failures) if failures else 0.0
        }
    
    def _calculate_correlation_significance(self,
                                          conditioning_results: ConditioningBatchResults,
                                          equation_properties: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistical significance of correlations."""
        try:
            from scipy.stats import pearsonr
            
            effectiveness_scores = [r.effectiveness_score for r in conditioning_results.individual_results]
            
            if len(effectiveness_scores) < 3:
                return {'insufficient_data': True}
            
            # Test correlation with computational cost
            costs = [r.computational_cost for r in conditioning_results.individual_results]
            if costs and len(costs) == len(effectiveness_scores):
                corr, p_value = pearsonr(effectiveness_scores, costs)
                return {
                    'cost_effectiveness_correlation_pvalue': float(p_value),
                    'cost_effectiveness_correlation': float(corr)
                }
            
            return {}
            
        except Exception:
            return {}
    
    def _generate_correlation_recommendations(self,
                                            diversity_corr: float,
                                            complexity_corr: float,
                                            coefficient_corr: float,
                                            failure_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []
        
        # Diversity correlation recommendations
        if abs(diversity_corr) > 0.3:
            if diversity_corr > 0:
                recommendations.append("Conditioning effectiveness increases with equation diversity - prioritize diverse test sets")
            else:
                recommendations.append("Conditioning effectiveness decreases with diversity - investigate robustness issues")
        
        # Complexity correlation recommendations
        if abs(complexity_corr) > 0.3:
            if complexity_corr < -0.3:
                recommendations.append("Conditioning struggles with complex equations - enhance conditioning methods")
            elif complexity_corr > 0.3:
                recommendations.append("Conditioning works well with complex equations - leverage this strength")
        
        # Failure pattern recommendations
        failure_rate = failure_analysis.get('failure_rate', 0.0)
        if failure_rate > 0.1:
            recommendations.append(f"High failure rate ({failure_rate:.1%}) - investigate failure modes")
        
        if not recommendations:
            recommendations.append("Correlations within acceptable ranges - current conditioning approach adequate")
        
        return recommendations
    
    def _empty_correlation_result(self) -> CorrelationAnalysisResult:
        """Return empty correlation result for edge cases."""
        return CorrelationAnalysisResult(
            conditioning_diversity_correlation=0.0,
            conditioning_complexity_correlation=0.0,
            conditioning_coefficient_correlation=0.0,
            effectiveness_distribution_correlation=0.0,
            failure_pattern_analysis={},
            statistical_significance={},
            recommendations=["Insufficient data for correlation analysis"]
        )
    
    def _estimate_equation_complexity(self, equation: str) -> int:
        """Estimate equation complexity based on structure."""
        complexity = 0
        complexity += equation.count('*')
        complexity += equation.count('+')
        complexity += equation.count('-')
        complexity += equation.count('(')
        complexity += equation.count('/')
        return complexity
    
    def _estimate_coefficient_range(self, equation: str) -> float:
        """Estimate the range of coefficients in an equation."""
        import re
        numbers = re.findall(r'-?\d+', equation)
        if numbers:
            int_numbers = [int(n) for n in numbers]
            return float(max(int_numbers) - min(int_numbers))
        return 0.0
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        data_array = np.array(data)
        return float((np.mean(data_array)**3) / (np.std(data_array)**3)) if np.std(data_array) > 0 else 0.0
    
    def _calculate_robustness_score(self, effectiveness: List[float], convergence: List[bool]) -> float:
        """Calculate overall robustness score."""
        if not effectiveness:
            return 0.0
        
        eff_stability = 1.0 - (np.std(effectiveness) / np.mean(effectiveness)) if np.mean(effectiveness) > 0 else 0.0
        conv_rate = np.mean(convergence) if convergence else 0.0
        
        return float(0.6 * eff_stability + 0.4 * conv_rate)
    
    def _analyze_distribution_shape(self, data: List[float]) -> str:
        """Analyze the shape of a distribution."""
        if len(data) < 3:
            return 'insufficient_data'
        
        data_array = np.array(data)
        skewness = self._calculate_skewness(data)
        
        if abs(skewness) < 0.5:
            return 'symmetric'
        elif skewness > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'
    
    def _detect_bimodal_tendency(self, data: List[float]) -> bool:
        """Detect if data has bimodal tendency."""
        if len(data) < 10:
            return False
        
        try:
            # Simple bimodal detection using histogram
            hist, _ = np.histogram(data, bins=5)
            peaks = np.sum(hist[1:-1] < hist[:-2]) + np.sum(hist[1:-1] < hist[2:])
            return peaks >= 2
        except Exception:
            return False
    
    def _calculate_outlier_percentage(self, data: List[float]) -> float:
        """Calculate percentage of outliers using IQR method."""
        if len(data) < 4:
            return 0.0
        
        try:
            data_array = np.array(data)
            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((data_array < lower_bound) | (data_array > upper_bound))
            return float(outliers) / len(data_array) * 100
        except Exception:
            return 0.0


class IntegratedTestCoordinator:
    """
    Coordinates testing between T2's conditioning tests and T4's statistical tests.
    
    Manages test sequencing, data sharing, and coordinated validation workflows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.active_test_plans = {}
        
    def create_integrated_test_plan(self,
                                  conditioning_config: Dict[str, Any],
                                  statistical_config: Dict[str, Any],
                                  coordination_params: Dict[str, Any]) -> IntegratedTestPlan:
        """
        Create an integrated test plan combining T2 and T4 testing.
        
        Args:
            conditioning_config: Configuration for T2 conditioning tests
            statistical_config: Configuration for T4 statistical tests  
            coordination_params: Parameters for coordinating the tests
            
        Returns:
            IntegratedTestPlan for coordinated execution
        """
        plan_id = f"integrated_plan_{len(self.active_test_plans)}"
        
        # Default coordination parameters
        default_coordination = {
            'sequential_execution': True,
            'share_equation_sets': True,
            'cross_validate_results': True,
            'failure_propagation': True
        }
        coordination_params = {**default_coordination, **coordination_params}
        
        # Create test configurations
        conditioning_tests = self._create_conditioning_test_configs(conditioning_config, coordination_params)
        statistical_tests = self._create_statistical_test_configs(statistical_config, coordination_params)
        
        # Define expected outcomes
        expected_outcomes = {
            'conditioning_effectiveness_threshold': coordination_params.get('effectiveness_threshold', 0.7),
            'statistical_diversity_threshold': coordination_params.get('diversity_threshold', 0.5),
            'correlation_threshold': coordination_params.get('correlation_threshold', 0.3)
        }
        
        # Validation criteria
        validation_criteria = {
            'min_conditioning_success_rate': 0.8,
            'min_statistical_adequacy': 0.8,
            'max_failure_correlation': 0.5
        }
        
        plan = IntegratedTestPlan(
            test_plan_id=plan_id,
            conditioning_test_configs=conditioning_tests,
            statistical_test_configs=statistical_tests,
            coordination_parameters=coordination_params,
            expected_outcomes=expected_outcomes,
            validation_criteria=validation_criteria
        )
        
        self.active_test_plans[plan_id] = plan
        return plan
    
    def execute_integrated_test_plan(self, test_plan: IntegratedTestPlan) -> Dict[str, Any]:
        """
        Execute an integrated test plan.
        
        Args:
            test_plan: IntegratedTestPlan to execute
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing integrated test plan: {test_plan.test_plan_id}")
        
        results = {
            'plan_id': test_plan.test_plan_id,
            'conditioning_results': {},
            'statistical_results': {},
            'integration_results': {},
            'validation_results': {}
        }
        
        try:
            # Execute conditioning tests (T2)
            conditioning_results = self._execute_conditioning_phase(test_plan)
            results['conditioning_results'] = conditioning_results
            
            # Execute statistical tests (T4) 
            statistical_results = self._execute_statistical_phase(test_plan, conditioning_results)
            results['statistical_results'] = statistical_results
            
            # Perform integration analysis
            integration_results = self._perform_integration_analysis(conditioning_results, statistical_results)
            results['integration_results'] = integration_results
            
            # Validate against criteria
            validation_results = self._validate_integrated_results(test_plan, results)
            results['validation_results'] = validation_results
            
            self.logger.info(f"Integrated test plan {test_plan.test_plan_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error executing test plan {test_plan.test_plan_id}: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'failed'
        
        return results
    
    def _create_conditioning_test_configs(self, 
                                        base_config: Dict[str, Any],
                                        coordination_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create conditioning test configurations."""
        # Base configuration for conditioning tests
        base_conditioning = {
            'method': 'gradient_descent',
            'max_iterations': 100,
            'convergence_threshold': 1e-6,
            'learning_rate': 0.01
        }
        
        # Merge with provided configuration
        config = {**base_conditioning, **base_config}
        
        # Create multiple test configurations
        test_configs = []
        
        # Standard configuration
        test_configs.append({
            **config,
            'test_type': 'standard_conditioning',
            'equation_count': coordination_params.get('equation_count', 100)
        })
        
        # Robustness test configuration
        test_configs.append({
            **config,
            'test_type': 'robustness_conditioning',
            'equation_count': coordination_params.get('equation_count', 50),
            'noise_level': 0.1
        })
        
        return test_configs
    
    def _create_statistical_test_configs(self,
                                       base_config: Dict[str, Any],
                                       coordination_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create statistical test configurations."""
        # Base configuration for statistical tests
        base_statistical = {
            'diversity_type': 'mixed',
            'sample_size': 200,
            'confidence_level': 0.95
        }
        
        # Merge with provided configuration
        config = {**base_statistical, **base_config}
        
        # Create multiple test configurations
        test_configs = []
        
        # Diversity analysis
        test_configs.append({
            **config,
            'test_type': 'diversity_analysis',
            'focus': 'equation_diversity'
        })
        
        # Distribution analysis
        test_configs.append({
            **config,
            'test_type': 'distribution_analysis', 
            'focus': 'coefficient_distribution'
        })
        
        return test_configs
    
    def _execute_conditioning_phase(self, test_plan: IntegratedTestPlan) -> Dict[str, Any]:
        """Execute the conditioning test phase."""
        # Placeholder for T2 conditioning test execution
        # In actual implementation, would interface with T2's testing system
        
        conditioning_results = {
            'phase_status': 'completed',
            'test_results': [],
            'overall_effectiveness': 0.75,
            'convergence_rate': 0.85,
            'computational_cost': 150.0
        }
        
        # Simulate some test results
        for i, config in enumerate(test_plan.conditioning_test_configs):
            result = {
                'config_id': i,
                'test_type': config.get('test_type', 'unknown'),
                'effectiveness_score': 0.7 + 0.2 * np.random.random(),
                'convergence_achieved': np.random.random() > 0.2,
                'equation_count': config.get('equation_count', 100)
            }
            conditioning_results['test_results'].append(result)
        
        return conditioning_results
    
    def _execute_statistical_phase(self, 
                                 test_plan: IntegratedTestPlan,
                                 conditioning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the statistical test phase."""
        # Placeholder for T4 statistical test execution
        # In actual implementation, would use the statistical testing framework
        
        statistical_results = {
            'phase_status': 'completed',
            'diversity_analysis': {
                'structure_diversity': 0.68,
                'coefficient_entropy': 2.1,
                'sample_adequacy': True
            },
            'distribution_analysis': {
                'normality_pvalue': 0.15,
                'skewness': 0.3,
                'outlier_percentage': 5.2
            },
            'validation_results': {
                'overall_valid': True,
                'diversity_adequate': True,
                'distribution_reasonable': True
            }
        }
        
        return statistical_results
    
    def _perform_integration_analysis(self,
                                    conditioning_results: Dict[str, Any],
                                    statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integration analysis between conditioning and statistical results."""
        integration_analysis = {
            'correlation_analysis': {},
            'consistency_check': {},
            'combined_validation': {}
        }
        
        # Analyze correlations
        conditioning_effectiveness = conditioning_results.get('overall_effectiveness', 0.0)
        statistical_diversity = statistical_results.get('diversity_analysis', {}).get('structure_diversity', 0.0)
        
        integration_analysis['correlation_analysis'] = {
            'effectiveness_diversity_correlation': float(np.corrcoef([conditioning_effectiveness], [statistical_diversity])[0, 1]) if conditioning_effectiveness > 0 and statistical_diversity > 0 else 0.0,
            'consistency_score': 0.8 if abs(conditioning_effectiveness - statistical_diversity) < 0.3 else 0.5
        }
        
        # Consistency check
        conditioning_success = conditioning_results.get('convergence_rate', 0.0) > 0.7
        statistical_success = statistical_results.get('validation_results', {}).get('overall_valid', False)
        
        integration_analysis['consistency_check'] = {
            'both_phases_successful': conditioning_success and statistical_success,
            'phase_agreement': conditioning_success == statistical_success,
            'integration_score': 0.9 if conditioning_success and statistical_success else 0.6
        }
        
        return integration_analysis
    
    def _validate_integrated_results(self, 
                                   test_plan: IntegratedTestPlan,
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integrated test results against criteria."""
        validation = {
            'criteria_met': {},
            'overall_validation': False,
            'recommendations': []
        }
        
        criteria = test_plan.validation_criteria
        
        # Check conditioning success rate
        conditioning_success_rate = results['conditioning_results'].get('convergence_rate', 0.0)
        validation['criteria_met']['conditioning_success_rate'] = conditioning_success_rate >= criteria['min_conditioning_success_rate']
        
        # Check statistical adequacy
        statistical_adequacy = results['statistical_results'].get('validation_results', {}).get('overall_valid', False)
        validation['criteria_met']['statistical_adequacy'] = statistical_adequacy
        
        # Overall validation
        validation['overall_validation'] = all(validation['criteria_met'].values())
        
        # Generate recommendations
        if not validation['overall_validation']:
            if not validation['criteria_met']['conditioning_success_rate']:
                validation['recommendations'].append("Improve conditioning method parameters")
            if not validation['criteria_met']['statistical_adequacy']:
                validation['recommendations'].append("Increase statistical sample size or diversity")
        else:
            validation['recommendations'].append("Integrated testing successful - criteria met")
        
        return validation


class SafeguardValidationBridge:
    """
    Bridge between T2's conditioning tests and T4's statistical safeguards.
    
    Provides unified validation framework that combines conditioning effectiveness
    with statistical safeguard validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validation_history = []
    
    def validate_combined_safeguards(self,
                                   conditioning_results: ConditioningBatchResults,
                                   statistical_validation: Dict[str, bool],
                                   safeguard_criteria: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate combined safeguards from T2 conditioning and T4 statistical tests.
        
        Args:
            conditioning_results: T2 conditioning test results
            statistical_validation: T4 statistical validation results
            safeguard_criteria: Combined safeguard criteria
            
        Returns:
            Combined validation results
        """
        combined_validation = {
            'conditioning_safeguards': {},
            'statistical_safeguards': {},
            'integrated_safeguards': {},
            'overall_assessment': {}
        }
        
        # Validate conditioning safeguards
        combined_validation['conditioning_safeguards'] = self._validate_conditioning_safeguards(
            conditioning_results, safeguard_criteria
        )
        
        # Include statistical safeguards
        combined_validation['statistical_safeguards'] = statistical_validation
        
        # Integrated safeguard validation
        combined_validation['integrated_safeguards'] = self._validate_integrated_safeguards(
            conditioning_results, statistical_validation, safeguard_criteria
        )
        
        # Overall assessment
        combined_validation['overall_assessment'] = self._assess_overall_safeguards(combined_validation)
        
        # Store in history
        self.validation_history.append({
            'timestamp': str(pd.Timestamp.now()),
            'validation_result': combined_validation
        })
        
        return combined_validation
    
    def _validate_conditioning_safeguards(self,
                                        conditioning_results: ConditioningBatchResults,
                                        criteria: Dict[str, float]) -> Dict[str, bool]:
        """Validate conditioning-specific safeguards."""
        if not conditioning_results.individual_results:
            return {'no_conditioning_data': False}
        
        effectiveness_scores = [r.effectiveness_score for r in conditioning_results.individual_results]
        convergence_rates = [r.convergence_achieved for r in conditioning_results.individual_results]
        
        min_effectiveness = criteria.get('min_conditioning_effectiveness', 0.6)
        min_convergence_rate = criteria.get('min_convergence_rate', 0.7)
        max_failure_rate = criteria.get('max_conditioning_failure_rate', 0.3)
        
        failure_rate = 1.0 - np.mean(convergence_rates)
        
        return {
            'effectiveness_adequate': float(np.mean(effectiveness_scores)) >= min_effectiveness,
            'convergence_adequate': float(np.mean(convergence_rates)) >= min_convergence_rate,
            'failure_rate_acceptable': failure_rate <= max_failure_rate,
            'effectiveness_consistent': float(np.std(effectiveness_scores)) <= criteria.get('max_effectiveness_std', 0.3)
        }
    
    def _validate_integrated_safeguards(self,
                                      conditioning_results: ConditioningBatchResults,
                                      statistical_validation: Dict[str, bool],
                                      criteria: Dict[str, float]) -> Dict[str, bool]:
        """Validate integrated safeguards across both T2 and T4."""
        if not conditioning_results.individual_results:
            return {'no_data_for_integration': False}
        
        # Cross-validation between conditioning and statistical results
        conditioning_success = np.mean([r.convergence_achieved for r in conditioning_results.individual_results])
        statistical_success = statistical_validation.get('overall_valid', False)
        
        # Consistency check
        consistency_threshold = criteria.get('consistency_threshold', 0.2)
        effectiveness_diversity_consistency = abs(conditioning_success - (1.0 if statistical_success else 0.0)) <= consistency_threshold
        
        return {
            'conditioning_statistical_consistency': effectiveness_diversity_consistency,
            'both_systems_functional': conditioning_success > 0.7 and statistical_success,
            'complementary_validation': conditioning_success > 0.6 or statistical_success,
            'integrated_robustness': conditioning_success > 0.8 and statistical_success
        }
    
    def _assess_overall_safeguards(self, combined_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall safeguard status."""
        conditioning_safeguards = combined_validation.get('conditioning_safeguards', {})
        statistical_safeguards = combined_validation.get('statistical_safeguards', {})
        integrated_safeguards = combined_validation.get('integrated_safeguards', {})
        
        # Count passed safeguards
        conditioning_passed = sum(1 for v in conditioning_safeguards.values() if v)
        statistical_passed = sum(1 for v in statistical_safeguards.values() if v)
        integrated_passed = sum(1 for v in integrated_safeguards.values() if v)
        
        total_safeguards = len(conditioning_safeguards) + len(statistical_safeguards) + len(integrated_safeguards)
        total_passed = conditioning_passed + statistical_passed + integrated_passed
        
        safeguard_score = total_passed / total_safeguards if total_safeguards > 0 else 0.0
        
        return {
            'safeguard_score': float(safeguard_score),
            'safeguards_adequate': safeguard_score >= 0.8,
            'conditioning_contribution': conditioning_passed,
            'statistical_contribution': statistical_passed,
            'integration_contribution': integrated_passed,
            'recommendation': self._generate_safeguard_recommendation(safeguard_score, combined_validation)
        }
    
    def _generate_safeguard_recommendation(self, 
                                         score: float,
                                         validation_details: Dict[str, Any]) -> str:
        """Generate recommendation based on safeguard validation."""
        if score >= 0.9:
            return "Excellent safeguard coverage - system ready for deployment"
        elif score >= 0.8:
            return "Good safeguard coverage - minor improvements recommended"
        elif score >= 0.6:
            return "Moderate safeguard coverage - address failing safeguards before deployment"
        else:
            return "Insufficient safeguard coverage - significant improvements required"


def create_t2_integration_interface() -> Dict[str, Any]:
    """
    Factory function to create a complete T2 integration interface.
    
    Returns:
        Dictionary with all T2 integration components
    """
    return {
        'results_parser': ConditioningResultsParser(),
        'correlation_analyzer': ConditioningCorrelationAnalyzer(), 
        'test_coordinator': IntegratedTestCoordinator(),
        'safeguard_bridge': SafeguardValidationBridge(),
        'interface_ready': True
    }


if __name__ == "__main__":
    # Demo usage
    interface = create_t2_integration_interface()
    
    # Create sample conditioning results
    sample_result = ConditioningTestResult(
        test_id="test_001",
        rule_type="distribute",
        equation_input="2*(x+3)=10", 
        equation_target="2*x+6=10",
        conditioning_method="gradient_descent",
        effectiveness_score=0.85,
        failure_mode=None,
        computational_cost=25.5,
        convergence_achieved=True,
        metadata={}
    )
    
    sample_batch = ConditioningBatchResults(
        batch_id="demo_batch",
        test_configuration={},
        individual_results=[sample_result],
        overall_statistics={'mean_effectiveness': 0.85},
        timestamp="2024-01-01",
        t2_version="1.0"
    )
    
    # Analyze correlation
    analyzer = interface['correlation_analyzer']
    correlation_result = analyzer.analyze_conditioning_correlation(sample_batch, {})
    print(f"Correlation analysis complete: diversity_correlation={correlation_result.conditioning_diversity_correlation:.3f}")
    
    print("T2 Integration Interface ready for T4 implementation")