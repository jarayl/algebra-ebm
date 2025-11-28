#!/usr/bin/env python3
"""
Distance Threshold Optimizer - Phase 2

Empirical optimization of distance thresholds using statistical analysis of 
distance distributions from actual sampling attempts. Replaces emergency
threshold (6.0) with data-driven recommendations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ThresholdAnalysis:
    """Results from distance threshold analysis."""
    recommended_threshold: float
    confidence_interval: Tuple[float, float]
    safety_margin: float
    final_threshold: float  # recommended + safety margin
    sample_size: int
    success_rate_at_threshold: float
    statistical_justification: str
    complexity_category: str


class DistanceThresholdOptimizer:
    """
    Optimizer for distance thresholds using empirical data analysis.
    
    Collects distance distributions from actual inference attempts and uses
    statistical analysis to recommend optimal thresholds with safety margins.
    """
    
    def __init__(
        self,
        target_success_rate: float = 0.8,
        confidence_level: float = 0.95,
        safety_margin_factor: float = 1.2,
        min_sample_size: int = 100
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            target_success_rate: Target success rate for threshold optimization (default: 0.8)
            confidence_level: Statistical confidence level (default: 0.95)
            safety_margin_factor: Safety margin multiplier (default: 1.2 = 20% margin)
            min_sample_size: Minimum samples needed for analysis (default: 100)
        """
        self.target_success_rate = target_success_rate
        self.confidence_level = confidence_level
        self.safety_margin_factor = safety_margin_factor
        self.min_sample_size = min_sample_size
        
        # Storage for collected data
        self.distance_data = []
        self.complexity_stratified_data = defaultdict(list)
        
        logger.info(f"Initialized DistanceThresholdOptimizer: target_rate={target_success_rate}, "
                   f"confidence={confidence_level}, safety_margin={safety_margin_factor}")
    
    def collect_distance_data(self, distance_data: Dict[str, Any]) -> None:
        """
        Collect distance data from inference attempts.
        
        Args:
            distance_data: Distance data dict from solve_equation with collect_distance_data=True
        """
        if not isinstance(distance_data, dict):
            logger.warning(f"Invalid distance_data format: {type(distance_data)}")
            return
        
        required_fields = ['distance', 'success', 'equation_complexity']
        if not all(field in distance_data for field in required_fields):
            logger.warning(f"Missing required fields in distance_data: {distance_data.keys()}")
            return
        
        # Only collect data with finite, non-negative distances (decoder available and valid)
        distance_val = distance_data.get('distance')
        if distance_val == float('inf') or distance_val < 0:
            logger.debug(f"Skipping invalid distance: {distance_val}")
            return
            
        self.distance_data.append(distance_data)
        complexity = distance_data['equation_complexity']
        self.complexity_stratified_data[complexity].append(distance_data)
        
        logger.debug(f"Collected distance data: {distance_data['distance']:.3f} "
                    f"(complexity: {complexity}, success: {distance_data['success']})")
    
    def analyze_distance_distribution(
        self, 
        complexity: Optional[str] = None
    ) -> Optional[ThresholdAnalysis]:
        """
        Analyze distance distribution and recommend optimal threshold.
        
        Args:
            complexity: If specified, analyze only this complexity category
                       If None, analyze all data combined
                       
        Returns:
            ThresholdAnalysis with recommendations or None if insufficient data
        """
        # Select data for analysis
        if complexity is None:
            data = self.distance_data
            category = "all_combined"
        else:
            data = self.complexity_stratified_data.get(complexity, [])
            category = complexity
        
        if len(data) < self.min_sample_size:
            logger.warning(f"Insufficient data for {category}: {len(data)} < {self.min_sample_size}")
            return None
        
        logger.info(f"Analyzing {len(data)} samples for complexity: {category}")
        
        # Extract distances and success status
        distances = np.array([d['distance'] for d in data])
        successes = np.array([d['success'] for d in data])
        
        # Find threshold using percentile analysis
        recommended_threshold = self._find_optimal_threshold(distances, successes)
        
        # Calculate confidence interval using bootstrap
        ci_lower, ci_upper = self._bootstrap_confidence_interval(distances, successes)
        
        # Apply safety margin (conservative buffer for statistical uncertainty)
        safety_margin = recommended_threshold * (self.safety_margin_factor - 1.0)
        final_threshold = recommended_threshold + safety_margin
        
        # Calculate success rate at recommended threshold
        success_rate = self._calculate_success_rate_at_threshold(distances, successes, recommended_threshold)
        
        # Generate statistical justification
        justification = self._generate_statistical_justification(
            distances, successes, recommended_threshold, len(data)
        )
        
        analysis = ThresholdAnalysis(
            recommended_threshold=recommended_threshold,
            confidence_interval=(ci_lower, ci_upper),
            safety_margin=safety_margin,
            final_threshold=final_threshold,
            sample_size=len(data),
            success_rate_at_threshold=success_rate,
            statistical_justification=justification,
            complexity_category=category
        )
        
        logger.info(f"Analysis complete for {category}: threshold={final_threshold:.3f} "
                   f"(recommended={recommended_threshold:.3f} + margin={safety_margin:.3f})")
        
        return analysis
    
    def _find_optimal_threshold(self, distances: np.ndarray, successes: np.ndarray) -> float:
        """
        Find optimal threshold using success rate targeting.
        
        Strategy: Find the distance value where target_success_rate fraction of 
        successful equations have distance <= threshold.
        """
        # Get distances of successful decodings only
        successful_distances = distances[successes]
        
        if len(successful_distances) == 0:
            logger.warning("No successful decodings found, using 95th percentile of all distances")
            return float(np.percentile(distances, 95))
        
        # Find threshold where target_success_rate of successful equations are included
        percentile = self.target_success_rate * 100
        threshold = float(np.percentile(successful_distances, percentile))
        
        logger.debug(f"Optimal threshold: {threshold:.3f} (includes {self.target_success_rate} of successful decodings)")
        return threshold
    
    def _bootstrap_confidence_interval(
        self, 
        distances: np.ndarray, 
        successes: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap resampling."""
        bootstrap_thresholds = []
        
        # Set random seed for reproducibility in testing
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(distances), size=len(distances), replace=True)
            boot_distances = distances[indices]
            boot_successes = successes[indices]
            
            # Calculate threshold for this sample - handle potential issues
            try:
                boot_threshold = self._find_optimal_threshold(boot_distances, boot_successes)
                bootstrap_thresholds.append(boot_threshold)
            except Exception as e:
                logger.debug(f"Bootstrap iteration failed: {e}")
                continue
        
        # Calculate confidence interval if we have enough bootstrap samples
        if len(bootstrap_thresholds) < n_bootstrap * 0.5:  # At least 50% success rate
            logger.warning(f"Insufficient bootstrap samples: {len(bootstrap_thresholds)}/{n_bootstrap}")
            # Return wide confidence interval as fallback
            return 0.0, float('inf')
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = float(np.percentile(bootstrap_thresholds, lower_percentile))
        ci_upper = float(np.percentile(bootstrap_thresholds, upper_percentile))
        
        return ci_lower, ci_upper
    
    def _calculate_success_rate_at_threshold(
        self, 
        distances: np.ndarray, 
        successes: np.ndarray, 
        threshold: float
    ) -> float:
        """Calculate what the success rate would be at given threshold."""
        # Simulate applying this threshold
        would_be_successful = distances <= threshold
        
        # Calculate success rate
        total_attempts = len(distances)
        successful_attempts = np.sum(would_be_successful)
        
        return successful_attempts / total_attempts if total_attempts > 0 else 0.0
    
    def _generate_statistical_justification(
        self, 
        distances: np.ndarray, 
        successes: np.ndarray,
        threshold: float,
        sample_size: int
    ) -> str:
        """Generate human-readable statistical justification."""
        successful_distances = distances[successes]
        total_success_rate = np.mean(successes)
        
        # Calculate statistics
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        
        if len(successful_distances) > 0:
            mean_successful_distance = np.mean(successful_distances)
            coverage = np.mean(successful_distances <= threshold)
        else:
            mean_successful_distance = float('inf')
            coverage = 0.0
        
        justification = (
            f"Analysis of {sample_size} samples: "
            f"Overall success rate {total_success_rate:.1%}. "
            f"Distance distribution: mean={mean_distance:.2f}, median={median_distance:.2f}, std={std_distance:.2f}. "
            f"Successful equations have mean distance {mean_successful_distance:.2f}. "
            f"Threshold {threshold:.2f} covers {coverage:.1%} of successful decodings."
        )
        
        return justification
    
    def analyze_all_complexities(self) -> Dict[str, ThresholdAnalysis]:
        """
        Analyze all complexity categories and return comprehensive recommendations.
        
        Returns:
            Dictionary mapping complexity categories to their threshold analyses
        """
        results = {}
        
        # Analyze overall combined data
        overall_analysis = self.analyze_distance_distribution(complexity=None)
        if overall_analysis is not None:
            results['overall'] = overall_analysis
        
        # Analyze each complexity category
        for complexity in self.complexity_stratified_data.keys():
            analysis = self.analyze_distance_distribution(complexity=complexity)
            if analysis is not None:
                results[complexity] = analysis
        
        # Log summary
        logger.info(f"Analysis complete for {len(results)} categories:")
        for category, analysis in results.items():
            logger.info(f"  {category}: threshold={analysis.final_threshold:.3f} "
                       f"(n={analysis.sample_size})")
        
        return results
    
    def save_analysis_results(self, results: Dict[str, ThresholdAnalysis], output_path: str) -> None:
        """Save analysis results to JSON file."""
        # Convert to serializable format
        serializable_results = {}
        for category, analysis in results.items():
            serializable_results[category] = {
                'recommended_threshold': analysis.recommended_threshold,
                'confidence_interval': analysis.confidence_interval,
                'safety_margin': analysis.safety_margin,
                'final_threshold': analysis.final_threshold,
                'sample_size': analysis.sample_size,
                'success_rate_at_threshold': analysis.success_rate_at_threshold,
                'statistical_justification': analysis.statistical_justification,
                'complexity_category': analysis.complexity_category
            }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    def get_recommendation_summary(self, results: Dict[str, ThresholdAnalysis]) -> str:
        """Generate human-readable recommendation summary."""
        if not results:
            return "No analysis results available."
        
        summary = ["=== DISTANCE THRESHOLD OPTIMIZATION RESULTS ===\n"]
        
        # Overall recommendation
        if 'overall' in results:
            overall = results['overall']
            summary.append(f"OVERALL RECOMMENDATION:")
            summary.append(f"  Replace emergency threshold 6.0 → {overall.final_threshold:.2f}")
            summary.append(f"  Based on {overall.sample_size} samples")
            summary.append(f"  Expected success rate: {overall.success_rate_at_threshold:.1%}")
            summary.append(f"  Statistical confidence: {self.confidence_level:.0%}")
            summary.append("")
        
        # Per-complexity recommendations
        complexity_results = {k: v for k, v in results.items() if k != 'overall'}
        if complexity_results:
            summary.append("PER-COMPLEXITY RECOMMENDATIONS:")
            for complexity, analysis in complexity_results.items():
                summary.append(f"  {complexity}: {analysis.final_threshold:.2f} "
                             f"(n={analysis.sample_size}, success={analysis.success_rate_at_threshold:.1%})")
            summary.append("")
        
        # Implementation guidance
        if 'overall' in results:
            overall = results['overall']
            summary.append("IMPLEMENTATION GUIDANCE:")
            summary.append(f"1. Update default distance_threshold: 6.0 → {overall.final_threshold:.2f}")
            summary.append(f"2. Safety margin included: {overall.safety_margin:.2f}")
            summary.append(f"3. Statistical basis: {overall.statistical_justification}")
            summary.append("")
            summary.append("VALIDATION:")
            summary.append(f"- Test on held-out equations to verify {overall.success_rate_at_threshold:.1%} success rate")
            summary.append(f"- Monitor for regression vs emergency threshold")
        
        return "\n".join(summary)


def collect_distance_data_from_equations(
    equations: List[str],
    inference_engine,
    optimizer: DistanceThresholdOptimizer,
    current_threshold: float = 6.0
) -> None:
    """
    Helper function to collect distance data from a list of equations.
    
    Args:
        equations: List of equation strings to test
        inference_engine: AlgebraInference instance
        optimizer: DistanceThresholdOptimizer to collect data
        current_threshold: Current threshold to use for collection
    """
    logger.info(f"Collecting distance data from {len(equations)} equations...")
    
    successful_collections = 0
    for i, equation in enumerate(equations):
        try:
            result = inference_engine.solve_equation(
                equation,
                distance_threshold=current_threshold,
                collect_distance_data=True
            )
            
            if 'distance_data' in result:
                optimizer.collect_distance_data(result['distance_data'])
                successful_collections += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(equations)} equations "
                           f"({successful_collections} successful collections)")
                
        except Exception as e:
            logger.warning(f"Error processing equation '{equation}': {e}")
            # Continue processing other equations rather than failing completely
            continue
    
    logger.info(f"Data collection complete: {successful_collections}/{len(equations)} successful")


if __name__ == "__main__":
    # Example usage
    print("Distance Threshold Optimizer - Phase 2")
    print("Use collect_distance_data_from_equations() to gather data")
    print("Then call analyze_all_complexities() for recommendations")