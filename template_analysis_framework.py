#!/usr/bin/env python3
"""
Template Energy Comparison Framework - Task T5 Implementation

This module provides infrastructure for systematically analyzing and comparing
energy values between ground truth solutions and problematic template patterns.
Designed to support the implementation of Task T5: Template Energy Comparison.

Key Capabilities:
1. Template Pattern Identification: Find common solution templates in evaluation logs
2. Energy Comparison: Compare energy values between templates and correct solutions
3. Template Analysis: Statistical analysis of template occurrence patterns
4. Integration: Seamless integration with existing energy computation functions

Example Usage:
    # Initialize framework
    framework = TemplateAnalysisFramework()
    
    # Load models and identify templates
    framework.load_models(rule_models)
    framework.identify_problematic_templates(evaluation_logs)
    
    # Analyze template energies
    results = framework.compare_template_energies(test_cases)
    print(f"Template analysis: {results['summary']}")
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import json
import time
import re

# Import existing components
from src.algebra.algebra_inference import AlgebraInference, load_rule_models, InferenceConfig
from src.algebra.algebra_encoder import EquationDecoder, CharacterLevelEncoder
from src.algebra.algebra_evaluation import compute_embedding_distances

logger = logging.getLogger(__name__)


@dataclass
class TemplatePattern:
    """Represents a problematic template solution pattern."""
    template: str
    frequency: int
    contexts: List[str]  # Equations where this template appeared
    energy_stats: Dict[str, float] = field(default_factory=dict)
    distance_stats: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed statistics."""
        if not self.energy_stats:
            self.energy_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            }
        if not self.distance_stats:
            self.distance_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            }


@dataclass
class TemplateComparisonResult:
    """Results from comparing template energy vs ground truth energy."""
    equation: str
    ground_truth: str
    template: str
    gt_energy: float
    template_energy: float
    energy_difference: float  # template_energy - gt_energy
    gt_distance: float
    template_distance: float
    template_has_advantage: bool  # True if template energy < ground truth energy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateAnalysisReport:
    """Comprehensive template analysis results."""
    total_comparisons: int
    template_patterns: List[TemplatePattern]
    comparison_results: List[TemplateComparisonResult]
    summary_stats: Dict[str, float]
    problematic_cases: List[TemplateComparisonResult]
    recommendations: List[str] = field(default_factory=list)


class TemplateAnalysisFramework:
    """
    Framework for analyzing problematic template patterns and energy comparisons.
    
    This class provides the infrastructure needed for Task T5: Template Energy
    Comparison by identifying common failure patterns and systematically comparing
    energy values between ground truth solutions and template patterns.
    """
    
    def __init__(self, 
                 encoder: Optional[CharacterLevelEncoder] = None,
                 decoder: Optional[EquationDecoder] = None,
                 device: str = 'cpu'):
        """
        Initialize the template analysis framework.
        
        Args:
            encoder: Equation encoder for embedding computation
            decoder: Equation decoder for distance calculation  
            device: Device for tensor operations
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rule_models = {}
        self.inference_engine = None
        
        # Known problematic templates from documentation analysis
        self.known_problematic_templates = [
            "x=4",
            "2*x+x=6", 
            "2*x+3*x+1=11",
            "x=0",
            "x=-1",
            "x=10"
        ]
        
        # Template detection patterns
        self.template_patterns = [
            r"x=\d+",           # Simple numeric assignments
            r"x=-?\d+",         # Numeric assignments with negatives  
            r"\d+\*x\+\d*\*?x=\d+",  # Linear combinations
            r"x\+\d+=\d+",      # Simple addition
            r"\d+\*x=\d+"       # Simple multiplication
        ]
        
        logger.info(f"Initialized TemplateAnalysisFramework on {device}")
    
    def load_models(self, rule_models: Dict[str, Any]) -> None:
        """
        Load rule models for energy computation.
        
        Args:
            rule_models: Dictionary of trained rule-specific EBM models
        """
        self.rule_models = rule_models
        
        # Create inference engine if we have encoder
        if self.encoder is not None and len(rule_models) > 0:
            self.inference_engine = AlgebraInference(
                rule_models=rule_models,
                encoder=self.encoder,
                device=self.device
            )
            logger.info(f"Created inference engine with {len(rule_models)} rule models")
        else:
            logger.warning("Cannot create inference engine - missing encoder or models")
    
    def identify_template_patterns(self, 
                                 prediction_data: List[Tuple[str, str]],
                                 frequency_threshold: int = 3) -> List[TemplatePattern]:
        """
        Identify problematic template patterns from evaluation data.
        
        Args:
            prediction_data: List of (equation, prediction) pairs
            frequency_threshold: Minimum frequency to consider a pattern problematic
            
        Returns:
            List of identified template patterns
        """
        # Count prediction frequencies
        prediction_counter = Counter()
        equation_context = defaultdict(list)
        
        for equation, prediction in prediction_data:
            if prediction is not None:
                prediction_counter[prediction] += 1
                equation_context[prediction].append(equation)
        
        # Identify patterns that appear frequently
        template_patterns = []
        
        for prediction, frequency in prediction_counter.most_common():
            if frequency >= frequency_threshold:
                # Check if it matches known template patterns
                is_template = (
                    prediction in self.known_problematic_templates or
                    any(re.match(pattern, prediction) for pattern in self.template_patterns)
                )
                
                if is_template:
                    template_pattern = TemplatePattern(
                        template=prediction,
                        frequency=frequency,
                        contexts=equation_context[prediction]
                    )
                    template_patterns.append(template_pattern)
                    logger.info(f"Identified template pattern: '{prediction}' (freq={frequency})")
        
        return template_patterns
    
    def compute_energy_for_pair(self, 
                               equation: str, 
                               solution: str) -> Tuple[float, Optional[str]]:
        """
        Compute energy for an equation-solution pair.
        
        Args:
            equation: Input equation string
            solution: Solution string
            
        Returns:
            Tuple of (energy_value, error_message)
        """
        if self.inference_engine is None:
            return float('inf'), "No inference engine available"
        
        try:
            # Encode equation
            equation_embedding = self.encoder.encode([equation])
            solution_embedding = self.encoder.encode([solution])
            
            if equation_embedding is None or solution_embedding is None:
                return float('inf'), "Failed to encode equation or solution"
            
            # Move to device
            equation_embedding = equation_embedding.to(self.device)
            solution_embedding = solution_embedding.to(self.device)
            
            # Compute energy using inference engine
            # Use landscape index k=0 for consistency
            energy = self.inference_engine.compose_energies(
                equation_embedding, 
                solution_embedding, 
                k=0
            )
            
            return energy.item(), None
            
        except Exception as e:
            error_msg = f"Energy computation failed: {str(e)}"
            logger.warning(error_msg)
            return float('inf'), error_msg
    
    def compute_distance_for_pair(self, 
                                equation: str, 
                                solution: str) -> Tuple[float, Optional[str]]:
        """
        Compute embedding distance for an equation-solution pair.
        
        Args:
            equation: Input equation string
            solution: Solution string
            
        Returns:
            Tuple of (distance_value, error_message)
        """
        if self.encoder is None:
            return float('inf'), "No encoder available"
        
        try:
            # Encode both strings
            equation_embedding = self.encoder.encode([equation])
            solution_embedding = self.encoder.encode([solution])
            
            if equation_embedding is None or solution_embedding is None:
                return float('inf'), "Failed to encode equation or solution"
            
            # Compute L2 distance
            distance_tensor = torch.norm(equation_embedding - solution_embedding, dim=1)
            distance = distance_tensor.item()
            
            return distance, None
            
        except Exception as e:
            error_msg = f"Distance computation failed: {str(e)}"
            logger.warning(error_msg)
            return float('inf'), error_msg
    
    def compare_template_energies(self, 
                                test_cases: List[Tuple[str, str]],
                                template_patterns: Optional[List[TemplatePattern]] = None) -> TemplateAnalysisReport:
        """
        Compare energy values between ground truth solutions and template patterns.
        
        Args:
            test_cases: List of (equation, ground_truth_solution) pairs
            template_patterns: Optional list of template patterns to test against
            
        Returns:
            Comprehensive template analysis report
        """
        if template_patterns is None:
            template_patterns = [
                TemplatePattern(template=t, frequency=0, contexts=[]) 
                for t in self.known_problematic_templates
            ]
        
        comparison_results = []
        problematic_cases = []
        
        logger.info(f"Starting template energy comparison with {len(test_cases)} test cases")
        
        for equation, ground_truth in test_cases:
            # Compute ground truth energy and distance
            gt_energy, gt_error = self.compute_energy_for_pair(equation, ground_truth)
            gt_distance, gt_dist_error = self.compute_distance_for_pair(equation, ground_truth)
            
            if gt_error is not None:
                logger.warning(f"Failed to compute ground truth energy for '{equation}': {gt_error}")
                continue
            
            # Test each template pattern
            for template_pattern in template_patterns:
                template = template_pattern.template
                
                # Compute template energy and distance
                template_energy, template_error = self.compute_energy_for_pair(equation, template)
                template_distance, template_dist_error = self.compute_distance_for_pair(equation, template)
                
                if template_error is not None:
                    logger.warning(f"Failed to compute template energy for '{template}': {template_error}")
                    continue
                
                # Create comparison result
                energy_diff = template_energy - gt_energy
                template_has_advantage = template_energy < gt_energy
                
                result = TemplateComparisonResult(
                    equation=equation,
                    ground_truth=ground_truth,
                    template=template,
                    gt_energy=gt_energy,
                    template_energy=template_energy,
                    energy_difference=energy_diff,
                    gt_distance=gt_distance if gt_dist_error is None else float('inf'),
                    template_distance=template_distance if template_dist_error is None else float('inf'),
                    template_has_advantage=template_has_advantage,
                    metadata={
                        'gt_error': gt_error,
                        'template_error': template_error,
                        'gt_dist_error': gt_dist_error,
                        'template_dist_error': template_dist_error
                    }
                )
                
                comparison_results.append(result)
                
                # Track problematic cases where template has lower energy
                if template_has_advantage:
                    problematic_cases.append(result)
                    logger.warning(
                        f"Template '{template}' has lower energy than ground truth for '{equation}': "
                        f"template={template_energy:.3f} vs gt={gt_energy:.3f}"
                    )
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(comparison_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(comparison_results, problematic_cases)
        
        # Create comprehensive report
        report = TemplateAnalysisReport(
            total_comparisons=len(comparison_results),
            template_patterns=template_patterns,
            comparison_results=comparison_results,
            summary_stats=summary_stats,
            problematic_cases=problematic_cases,
            recommendations=recommendations
        )
        
        logger.info(f"Template analysis complete: {len(problematic_cases)} problematic cases out of {len(comparison_results)}")
        
        return report
    
    def _compute_summary_statistics(self, 
                                  comparison_results: List[TemplateComparisonResult]) -> Dict[str, float]:
        """Compute summary statistics for template comparison results."""
        if not comparison_results:
            return {}
        
        energy_diffs = [r.energy_difference for r in comparison_results]
        template_advantages = [r.template_has_advantage for r in comparison_results]
        gt_energies = [r.gt_energy for r in comparison_results if r.gt_energy != float('inf')]
        template_energies = [r.template_energy for r in comparison_results if r.template_energy != float('inf')]
        
        return {
            'total_comparisons': len(comparison_results),
            'template_advantage_rate': sum(template_advantages) / len(template_advantages),
            'mean_energy_difference': np.mean(energy_diffs),
            'std_energy_difference': np.std(energy_diffs),
            'median_energy_difference': np.median(energy_diffs),
            'min_energy_difference': np.min(energy_diffs),
            'max_energy_difference': np.max(energy_diffs),
            'mean_gt_energy': np.mean(gt_energies) if gt_energies else 0.0,
            'mean_template_energy': np.mean(template_energies) if template_energies else 0.0,
            'problematic_case_rate': len([r for r in comparison_results if r.template_has_advantage]) / len(comparison_results)
        }
    
    def _generate_recommendations(self, 
                                comparison_results: List[TemplateComparisonResult],
                                problematic_cases: List[TemplateComparisonResult]) -> List[str]:
        """Generate actionable recommendations based on template analysis results."""
        recommendations = []
        
        if not comparison_results:
            recommendations.append("No comparison results available - check model loading and encoding")
            return recommendations
        
        # Check overall template advantage rate
        advantage_rate = len(problematic_cases) / len(comparison_results)
        
        if advantage_rate > 0.8:
            recommendations.append(
                "CRITICAL: Templates have lower energy than ground truth in >80% of cases. "
                "This indicates a fundamental conditioning failure."
            )
        elif advantage_rate > 0.2:
            recommendations.append(
                "WARNING: Templates have lower energy than ground truth in >20% of cases. "
                "Consider investigating energy function calibration."
            )
        else:
            recommendations.append(
                "Energy landscape appears properly calibrated - templates rarely have advantage."
            )
        
        # Check for specific problematic templates
        template_problem_counts = Counter([case.template for case in problematic_cases])
        for template, count in template_problem_counts.most_common(3):
            recommendations.append(
                f"Template '{template}' frequently has energy advantage ({count} cases) - "
                f"investigate if this template appears in training data."
            )
        
        # Check energy magnitudes
        energy_diffs = [r.energy_difference for r in comparison_results if r.energy_difference != float('inf')]
        if energy_diffs:
            mean_diff = np.mean(energy_diffs)
            if abs(mean_diff) < 0.1:
                recommendations.append(
                    "Energy differences are very small (< 0.1) - may indicate insufficient "
                    "energy separation between correct and incorrect solutions."
                )
        
        return recommendations
    
    def save_analysis_report(self, 
                           report: TemplateAnalysisReport, 
                           output_path: Union[str, Path]) -> None:
        """
        Save template analysis report to JSON file.
        
        Args:
            report: Template analysis report to save
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        
        # Convert report to serializable format
        report_dict = {
            'total_comparisons': report.total_comparisons,
            'template_patterns': [
                {
                    'template': tp.template,
                    'frequency': tp.frequency,
                    'contexts': tp.contexts[:10],  # Limit to first 10 for file size
                    'energy_stats': tp.energy_stats,
                    'distance_stats': tp.distance_stats
                }
                for tp in report.template_patterns
            ],
            'summary_stats': report.summary_stats,
            'problematic_cases': [
                {
                    'equation': case.equation,
                    'ground_truth': case.ground_truth,
                    'template': case.template,
                    'gt_energy': case.gt_energy,
                    'template_energy': case.template_energy,
                    'energy_difference': case.energy_difference,
                    'template_has_advantage': case.template_has_advantage
                }
                for case in report.problematic_cases
            ],
            'recommendations': report.recommendations,
            'analysis_timestamp': time.time()
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Template analysis report saved to {output_path}")
    
    def create_test_cases_from_dataset(self, 
                                     dataset: Any, 
                                     max_cases: int = 50) -> List[Tuple[str, str]]:
        """
        Create test cases from a dataset for template analysis.
        
        Args:
            dataset: Dataset object with equation/solution pairs
            max_cases: Maximum number of test cases to extract
            
        Returns:
            List of (equation, solution) test case pairs
        """
        test_cases = []
        
        try:
            dataset_size = min(len(dataset), max_cases)
            
            for i in range(dataset_size):
                try:
                    example = dataset[i]
                    if isinstance(example, dict):
                        equation = example.get('equation', example.get('input'))
                        solution = example.get('solution', example.get('target', example.get('output')))
                    else:
                        # Assume tuple format
                        equation, solution = example[0], example[1]
                    
                    if equation is not None and solution is not None:
                        test_cases.append((str(equation), str(solution)))
                        
                except (IndexError, KeyError, AttributeError) as e:
                    logger.warning(f"Failed to extract test case {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to create test cases from dataset: {e}")
        
        logger.info(f"Created {len(test_cases)} test cases from dataset")
        return test_cases


# Convenience functions for Task T5 integration

def create_template_analysis_framework(encoder=None, decoder=None, device='cpu') -> TemplateAnalysisFramework:
    """
    Factory function to create a template analysis framework with defaults.
    
    Args:
        encoder: Optional equation encoder
        decoder: Optional equation decoder  
        device: Device for computations
        
    Returns:
        Configured TemplateAnalysisFramework instance
    """
    return TemplateAnalysisFramework(encoder=encoder, decoder=decoder, device=device)


def run_template_energy_analysis(rule_models: Dict[str, Any],
                                encoder: CharacterLevelEncoder,
                                test_cases: List[Tuple[str, str]],
                                output_path: Optional[str] = None,
                                device: str = 'cpu') -> TemplateAnalysisReport:
    """
    High-level function to run complete template energy analysis.
    
    Args:
        rule_models: Trained rule-specific EBM models
        encoder: Equation encoder
        test_cases: List of (equation, ground_truth) test cases
        output_path: Optional path to save analysis report
        device: Device for computations
        
    Returns:
        Complete template analysis report
    """
    # Create framework
    framework = create_template_analysis_framework(encoder=encoder, device=device)
    
    # Load models
    framework.load_models(rule_models)
    
    # Run analysis
    logger.info("Starting template energy analysis...")
    report = framework.compare_template_energies(test_cases)
    
    # Save report if path provided
    if output_path:
        framework.save_analysis_report(report, output_path)
    
    return report


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create mock test cases
    test_cases = [
        ("2*x=10", "x=5"),
        ("3*x+6=21", "x=5"),
        ("x-4=7", "x=11"),
        ("-2*x=14", "x=-7")
    ]
    
    # Create framework
    framework = create_template_analysis_framework()
    
    # Test template pattern identification
    mock_predictions = [
        ("2*x=10", "x=4"),
        ("3*x+6=21", "x=4"), 
        ("x-4=7", "x=4"),
        ("5*x=15", "2*x+x=6"),
        ("4*x=8", "2*x+x=6")
    ]
    
    patterns = framework.identify_template_patterns(mock_predictions, frequency_threshold=2)
    
    print(f"Identified {len(patterns)} template patterns:")
    for pattern in patterns:
        print(f"  Template: '{pattern.template}' (frequency: {pattern.frequency})")
    
    print("\nTemplate analysis framework initialized successfully!")
    print("Note: Full energy analysis requires trained models and proper encoder setup.")