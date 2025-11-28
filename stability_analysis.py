#!/usr/bin/env python3
"""
Numerical Stability Analysis - Phase 2

Analyzes numerical stability of the inference pipeline across different step sizes
to optimize step size parameter while maintaining convergence reliability.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
import copy

logger = logging.getLogger(__name__)


@dataclass
class StabilityTestResult:
    """Results from testing a single equation with specific step size."""
    equation: str
    step_size: float
    success: bool
    converged: bool
    final_energy: float
    iterations_taken: int
    acceptance_rate: float
    distance_achieved: float
    execution_time: float
    energy_trajectory: List[float]
    error_message: Optional[str] = None
    test_metadata: Optional[Dict[str, Any]] = None


@dataclass
class StepSizeAnalysis:
    """Analysis results for a specific step size."""
    step_size: float
    total_tests: int
    successful_tests: int
    converged_tests: int
    success_rate: float
    convergence_rate: float
    average_final_energy: float
    median_final_energy: float
    average_iterations: float
    average_acceptance_rate: float
    average_execution_time: float
    energy_stability_score: float  # Lower is more stable
    numerical_issues_count: int
    test_results: List[StabilityTestResult]


@dataclass
class StabilityReport:
    """Complete numerical stability analysis report."""
    step_size_analyses: Dict[float, StepSizeAnalysis]
    optimal_step_size: float
    stability_recommendation: str
    numerical_issues_detected: List[str]
    performance_summary: Dict[str, Any]


class NumericalStabilityAnalyzer:
    """
    Analyzes numerical stability across different step sizes for MCMC sampling.
    
    Tests stability by:
    1. Running identical equations with different step sizes
    2. Monitoring energy trajectories for oscillation/divergence
    3. Tracking acceptance rates for healthiness indicators
    4. Detecting numerical issues (NaN, inf, convergence failures)
    5. Measuring performance impact of step size changes
    """
    
    def __init__(
        self,
        base_step_size: float = 0.05,
        test_step_sizes: Optional[List[float]] = None,
        min_convergence_rate: float = 0.8,
        max_energy_variance_threshold: float = 100.0,
        target_acceptance_rate: Tuple[float, float] = (0.2, 0.6)
    ):
        """
        Initialize stability analyzer.
        
        Args:
            base_step_size: Current step size to use as baseline
            test_step_sizes: List of step sizes to test (default: around base_step_size)
            min_convergence_rate: Minimum acceptable convergence rate
            max_energy_variance_threshold: Maximum acceptable energy variance
            target_acceptance_rate: Tuple of (min, max) acceptable acceptance rates
        """
        self.base_step_size = base_step_size
        self.min_convergence_rate = min_convergence_rate
        self.max_energy_variance_threshold = max_energy_variance_threshold
        self.target_acceptance_rate = target_acceptance_rate
        
        if test_step_sizes is None:
            # Test range around base step size
            self.test_step_sizes = [
                base_step_size * 0.5,   # Conservative
                base_step_size * 0.75,  # Slightly conservative
                base_step_size,         # Current
                base_step_size * 1.25,  # Slightly aggressive
                base_step_size * 1.5,   # More aggressive
                base_step_size * 2.0,   # Very aggressive
                base_step_size * 3.0    # Test boundary
            ]
        else:
            self.test_step_sizes = test_step_sizes
            
        logger.info(f"Initialized NumericalStabilityAnalyzer: "
                   f"base_step_size={base_step_size}, "
                   f"test_step_sizes={self.test_step_sizes}, "
                   f"target_acceptance_rate={target_acceptance_rate}")
    
    def create_stability_test_equations(self, count: int = 25) -> List[str]:
        """
        Create test equations for stability analysis.
        
        Focuses on equations that are challenging for MCMC but not too complex,
        to properly test numerical stability without confounding factors.
        
        Args:
            count: Number of equations to generate
            
        Returns:
            List of test equation strings
        """
        equations = []
        
        # Simple equations that should be stable
        simple_equations = [
            "x=1", "x=2", "x=3", "x=-1", "x=-2",
            "2*x=4", "3*x=6", "4*x=8", "5*x=10",
            "x+1=3", "x+2=5", "x-1=2", "x-2=1"
        ]
        equations.extend(simple_equations[:count//3])
        
        # Medium complexity equations
        medium_equations = [
            "2*x+3=7", "3*x-1=8", "4*x+2=14", "5*x-3=12",
            "2*(x+1)=6", "3*(x-1)=9", "4*(x+2)=16", 
            "x**2=4", "x**2=9", "x**2=16", "x**2+1=5"
        ]
        equations.extend(medium_equations[:count//3])
        
        # More challenging but still reasonable equations
        challenging_equations = [
            "x**2-2*x=3", "x**2+x-2=0", "2*x**2-x=1",
            "x**3=8", "x**3=27", "x**3-1=7",
            "2*x**3+x=9", "x**3+x**2=6"
        ]
        equations.extend(challenging_equations[:count - len(equations)])
        
        # Fill to exact count if needed
        while len(equations) < count:
            equations.append(f"x={len(equations) % 10}")
        
        logger.info(f"Created {len(equations)} stability test equations")
        return equations[:count]
    
    def test_single_equation_stability(
        self,
        equation: str,
        step_size: float,
        inference_engine,
        max_iterations: int = 1000
    ) -> StabilityTestResult:
        """
        Test stability of a single equation with specific step size.
        
        Args:
            equation: Equation string to test
            step_size: Step size to use for this test
            inference_engine: AlgebraInference instance
            max_iterations: Maximum MCMC iterations
            
        Returns:
            StabilityTestResult with detailed stability metrics
        """
        # Create inference config with the test step size
        from algebra_inference import InferenceConfig
        
        config = InferenceConfig(
            num_samples=max_iterations,
            step_size=step_size,
            temperature=1.0,
            burn_in_samples=max_iterations // 10,
            collect_stats=True  # Enable detailed statistics collection
        )
        
        start_time = time.time()
        energy_trajectory = []
        success = False
        converged = False
        final_energy = float('inf')
        iterations_taken = 0
        acceptance_rate = 0.0
        distance_achieved = float('inf')
        error_message = None
        
        try:
            # Run inference with detailed monitoring
            result = inference_engine.solve_equation(
                equation,
                config=config,
                collect_distance_data=True
            )
            
            execution_time = time.time() - start_time
            
            # Extract results
            success = result.get('success', False)
            final_energy = result.get('final_energy', float('inf'))
            iterations_taken = result.get('iterations_used', 0)
            
            # Get detailed statistics if available
            stats = result.get('sampling_stats', {})
            acceptance_rate = stats.get('acceptance_rate', 0.0)
            energy_trajectory = stats.get('energy_trajectory', [])
            
            # Distance information
            distance_data = result.get('distance_data', {})
            distance_achieved = distance_data.get('distance', float('inf'))
            
            # Analyze convergence from energy trajectory
            if len(energy_trajectory) > 10:
                # Check if energy stabilized in last 20% of iterations
                stable_region = energy_trajectory[int(len(energy_trajectory) * 0.8):]
                energy_variance = np.var(stable_region) if len(stable_region) > 5 else float('inf')
                converged = energy_variance < self.max_energy_variance_threshold
            else:
                converged = success  # Fallback if no trajectory available
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            logger.warning(f"Stability test failed for equation '{equation}' with step_size={step_size}: {e}")
        
        # Detect numerical issues
        numerical_issues = []
        if np.isnan(final_energy) or np.isinf(final_energy):
            numerical_issues.append("infinite_or_nan_energy")
        if acceptance_rate > 0.99:
            numerical_issues.append("pathological_acceptance_rate") 
        if len(energy_trajectory) > 0 and any(np.isnan(e) or np.isinf(e) for e in energy_trajectory):
            numerical_issues.append("trajectory_numerical_issues")
        if execution_time > 60.0:  # More than 1 minute per equation
            numerical_issues.append("excessive_execution_time")
        
        test_result = StabilityTestResult(
            equation=equation,
            step_size=step_size,
            success=success,
            converged=converged,
            final_energy=final_energy,
            iterations_taken=iterations_taken,
            acceptance_rate=acceptance_rate,
            distance_achieved=distance_achieved,
            execution_time=execution_time,
            energy_trajectory=energy_trajectory[-100:],  # Keep last 100 points for analysis
            error_message=error_message,
            test_metadata={
                'numerical_issues': numerical_issues,
                'max_iterations': max_iterations,
                'target_acceptance_range': self.target_acceptance_rate
            }
        )
        
        return test_result
    
    def analyze_step_size_stability(
        self,
        step_size: float,
        test_equations: List[str],
        inference_engine
    ) -> StepSizeAnalysis:
        """
        Analyze stability for a specific step size across multiple equations.
        
        Args:
            step_size: Step size to analyze
            test_equations: List of equations to test
            inference_engine: AlgebraInference instance
            
        Returns:
            StepSizeAnalysis with aggregated stability metrics
        """
        logger.info(f"Analyzing stability for step_size={step_size} across {len(test_equations)} equations")
        
        test_results = []
        for i, equation in enumerate(test_equations):
            result = self.test_single_equation_stability(
                equation, step_size, inference_engine
            )
            test_results.append(result)
            
            if (i + 1) % 5 == 0:
                current_success_rate = sum(1 for r in test_results if r.success) / len(test_results)
                logger.info(f"Progress step_size={step_size}: {i+1}/{len(test_equations)} "
                           f"(success: {current_success_rate:.1%})")
        
        # Calculate aggregate statistics
        successful_tests = sum(1 for result in test_results if result.success)
        converged_tests = sum(1 for result in test_results if result.converged)
        
        success_rate = successful_tests / len(test_equations) if test_equations else 0.0
        convergence_rate = converged_tests / len(test_equations) if test_equations else 0.0
        
        # Energy statistics (only from finite energies)
        finite_energies = [r.final_energy for r in test_results 
                          if r.final_energy != float('inf') and not np.isnan(r.final_energy)]
        avg_final_energy = np.mean(finite_energies) if finite_energies else float('inf')
        median_final_energy = np.median(finite_energies) if finite_energies else float('inf')
        
        # Other metrics
        finite_iterations = [r.iterations_taken for r in test_results if r.iterations_taken > 0]
        avg_iterations = np.mean(finite_iterations) if finite_iterations else 0.0
        
        finite_acceptance_rates = [r.acceptance_rate for r in test_results 
                                  if 0.0 <= r.acceptance_rate <= 1.0]
        avg_acceptance_rate = np.mean(finite_acceptance_rates) if finite_acceptance_rates else 0.0
        
        execution_times = [r.execution_time for r in test_results]
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        # Stability score calculation (lower is better)
        energy_stability_score = 0.0
        valid_trajectories = [r.energy_trajectory for r in test_results if len(r.energy_trajectory) > 10]
        
        if valid_trajectories:
            trajectory_variances = []
            for trajectory in valid_trajectories:
                # Calculate variance in stable region (last 30%)
                stable_region = trajectory[int(len(trajectory) * 0.7):]
                if len(stable_region) > 3:
                    variance = np.var(stable_region)
                    if not (np.isnan(variance) or np.isinf(variance)):
                        trajectory_variances.append(variance)
            
            energy_stability_score = np.mean(trajectory_variances) if trajectory_variances else float('inf')
        else:
            energy_stability_score = float('inf')
        
        # Count numerical issues
        numerical_issues_count = 0
        for result in test_results:
            if result.test_metadata and result.test_metadata.get('numerical_issues'):
                numerical_issues_count += len(result.test_metadata['numerical_issues'])
        
        analysis = StepSizeAnalysis(
            step_size=step_size,
            total_tests=len(test_equations),
            successful_tests=successful_tests,
            converged_tests=converged_tests,
            success_rate=success_rate,
            convergence_rate=convergence_rate,
            average_final_energy=avg_final_energy,
            median_final_energy=median_final_energy,
            average_iterations=avg_iterations,
            average_acceptance_rate=avg_acceptance_rate,
            average_execution_time=avg_execution_time,
            energy_stability_score=energy_stability_score,
            numerical_issues_count=numerical_issues_count,
            test_results=test_results
        )
        
        logger.info(f"Step size {step_size} analysis complete: "
                   f"success={success_rate:.1%}, convergence={convergence_rate:.1%}, "
                   f"stability_score={energy_stability_score:.2f}")
        
        return analysis
    
    def find_optimal_step_size(
        self,
        analyses: Dict[float, StepSizeAnalysis]
    ) -> Tuple[float, str]:
        """
        Find optimal step size based on stability and performance metrics.
        
        Args:
            analyses: Dictionary mapping step sizes to their analyses
            
        Returns:
            Tuple of (optimal_step_size, recommendation_explanation)
        """
        if not analyses:
            return self.base_step_size, "No analyses available, using base step size"
        
        # Filter out step sizes with unacceptable performance
        acceptable_step_sizes = {}
        
        for step_size, analysis in analyses.items():
            # Must meet minimum convergence rate
            if analysis.convergence_rate < self.min_convergence_rate:
                continue
                
            # Must have acceptable acceptance rate
            if not (self.target_acceptance_rate[0] <= analysis.average_acceptance_rate <= self.target_acceptance_rate[1]):
                continue
                
            # Must not have excessive numerical issues
            issue_rate = analysis.numerical_issues_count / analysis.total_tests
            if issue_rate > 0.2:  # More than 20% of tests had issues
                continue
            
            acceptable_step_sizes[step_size] = analysis
        
        if not acceptable_step_sizes:
            return self.base_step_size, "No step sizes meet stability criteria, using base step size"
        
        # Score remaining step sizes (lower score is better)
        step_size_scores = {}
        
        for step_size, analysis in acceptable_step_sizes.items():
            score = 0.0
            
            # Penalty for low success rate
            score += (1.0 - analysis.success_rate) * 100
            
            # Penalty for low convergence rate  
            score += (1.0 - analysis.convergence_rate) * 50
            
            # Penalty for poor stability
            if analysis.energy_stability_score != float('inf'):
                score += min(analysis.energy_stability_score, 1000) / 10
            else:
                score += 100  # Heavy penalty for infinite stability score
            
            # Penalty for acceptance rate far from target center
            target_center = (self.target_acceptance_rate[0] + self.target_acceptance_rate[1]) / 2
            acceptance_deviation = abs(analysis.average_acceptance_rate - target_center)
            score += acceptance_deviation * 200
            
            # Penalty for execution time (prefer faster)
            score += analysis.average_execution_time * 2
            
            # Small penalty for being far from base step size (prefer minimal changes)
            step_size_deviation = abs(step_size - self.base_step_size) / self.base_step_size
            score += step_size_deviation * 10
            
            step_size_scores[step_size] = score
        
        # Find step size with lowest score
        optimal_step_size = min(step_size_scores.keys(), key=lambda s: step_size_scores[s])
        optimal_analysis = acceptable_step_sizes[optimal_step_size]
        
        # Generate recommendation explanation
        if optimal_step_size == self.base_step_size:
            explanation = f"Current step size {self.base_step_size} is optimal"
        else:
            change_factor = optimal_step_size / self.base_step_size
            if change_factor > 1.0:
                direction = "increase"
                factor = change_factor
            else:
                direction = "decrease" 
                factor = 1.0 / change_factor
                
            explanation = (f"Recommend {direction} step size by {factor:.2f}x "
                         f"({self.base_step_size} → {optimal_step_size}) for "
                         f"better stability (convergence: {optimal_analysis.convergence_rate:.1%}, "
                         f"acceptance: {optimal_analysis.average_acceptance_rate:.1%})")
        
        return optimal_step_size, explanation
    
    def run_stability_analysis(
        self,
        inference_engine,
        equation_count: int = 25
    ) -> StabilityReport:
        """
        Run complete numerical stability analysis across all test step sizes.
        
        Args:
            inference_engine: AlgebraInference instance
            equation_count: Number of test equations to use
            
        Returns:
            StabilityReport with complete analysis and recommendations
        """
        logger.info(f"Starting numerical stability analysis across {len(self.test_step_sizes)} step sizes")
        
        # Generate test equations
        test_equations = self.create_stability_test_equations(equation_count)
        
        # Test each step size
        step_size_analyses = {}
        for step_size in self.test_step_sizes:
            logger.info(f"Testing step size {step_size}...")
            analysis = self.analyze_step_size_stability(
                step_size, test_equations, inference_engine
            )
            step_size_analyses[step_size] = analysis
        
        # Find optimal step size
        optimal_step_size, recommendation = self.find_optimal_step_size(step_size_analyses)
        
        # Detect numerical issues across all tests
        numerical_issues_detected = []
        for step_size, analysis in step_size_analyses.items():
            if analysis.numerical_issues_count > 0:
                issue_rate = analysis.numerical_issues_count / analysis.total_tests
                if issue_rate > 0.1:  # More than 10% issue rate
                    numerical_issues_detected.append(
                        f"Step size {step_size}: {analysis.numerical_issues_count} issues "
                        f"({issue_rate:.1%} of tests)"
                    )
        
        # Performance summary
        performance_summary = {
            'total_step_sizes_tested': len(self.test_step_sizes),
            'total_equations_per_step_size': equation_count,
            'base_step_size': self.base_step_size,
            'optimal_step_size': optimal_step_size,
            'improvement_factor': optimal_step_size / self.base_step_size,
            'numerical_issues_detected': len(numerical_issues_detected) > 0,
            'analysis_timestamp': time.time()
        }
        
        report = StabilityReport(
            step_size_analyses=step_size_analyses,
            optimal_step_size=optimal_step_size,
            stability_recommendation=recommendation,
            numerical_issues_detected=numerical_issues_detected,
            performance_summary=performance_summary
        )
        
        logger.info(f"Stability analysis complete: optimal_step_size={optimal_step_size} "
                   f"(improvement factor: {optimal_step_size/self.base_step_size:.2f}x)")
        
        return report
    
    def save_results(self, report: StabilityReport, output_path: str) -> None:
        """Save stability analysis results to JSON file."""
        # Convert to serializable format
        serializable_analyses = {}
        
        for step_size, analysis in report.step_size_analyses.items():
            serializable_test_results = []
            for result in analysis.test_results:
                serializable_test_results.append({
                    'equation': result.equation,
                    'step_size': result.step_size,
                    'success': result.success,
                    'converged': result.converged,
                    'final_energy': result.final_energy,
                    'iterations_taken': result.iterations_taken,
                    'acceptance_rate': result.acceptance_rate,
                    'distance_achieved': result.distance_achieved,
                    'execution_time': result.execution_time,
                    'energy_trajectory': result.energy_trajectory,
                    'error_message': result.error_message,
                    'test_metadata': result.test_metadata
                })
            
            serializable_analyses[str(step_size)] = {
                'step_size': analysis.step_size,
                'total_tests': analysis.total_tests,
                'successful_tests': analysis.successful_tests,
                'converged_tests': analysis.converged_tests,
                'success_rate': analysis.success_rate,
                'convergence_rate': analysis.convergence_rate,
                'average_final_energy': analysis.average_final_energy,
                'median_final_energy': analysis.median_final_energy,
                'average_iterations': analysis.average_iterations,
                'average_acceptance_rate': analysis.average_acceptance_rate,
                'average_execution_time': analysis.average_execution_time,
                'energy_stability_score': analysis.energy_stability_score,
                'numerical_issues_count': analysis.numerical_issues_count,
                'test_results': serializable_test_results
            }
        
        serializable_report = {
            'step_size_analyses': serializable_analyses,
            'optimal_step_size': report.optimal_step_size,
            'stability_recommendation': report.stability_recommendation,
            'numerical_issues_detected': report.numerical_issues_detected,
            'performance_summary': report.performance_summary,
            'test_parameters': {
                'base_step_size': self.base_step_size,
                'test_step_sizes': self.test_step_sizes,
                'min_convergence_rate': self.min_convergence_rate,
                'max_energy_variance_threshold': self.max_energy_variance_threshold,
                'target_acceptance_rate': self.target_acceptance_rate
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Stability analysis results saved to {output_file}")
    
    def get_stability_report(self, report: StabilityReport) -> str:
        """Generate human-readable stability analysis report."""
        lines = ["=== NUMERICAL STABILITY ANALYSIS REPORT ===\n"]
        
        # Overall summary
        lines.append("STABILITY ANALYSIS SUMMARY:")
        summary = report.performance_summary
        lines.append(f"  Base Step Size: {summary['base_step_size']}")
        lines.append(f"  Optimal Step Size: {summary['optimal_step_size']}")
        lines.append(f"  Improvement Factor: {summary['improvement_factor']:.2f}x")
        lines.append(f"  Step Sizes Tested: {summary['total_step_sizes_tested']}")
        lines.append(f"  Equations per Step Size: {summary['total_equations_per_step_size']}")
        lines.append("")
        
        # Recommendation
        lines.append("RECOMMENDATION:")
        lines.append(f"  {report.stability_recommendation}")
        lines.append("")
        
        # Numerical issues
        if report.numerical_issues_detected:
            lines.append("⚠️  NUMERICAL ISSUES DETECTED:")
            for issue in report.numerical_issues_detected:
                lines.append(f"  - {issue}")
            lines.append("")
        else:
            lines.append("✅ No significant numerical issues detected across tested step sizes.")
            lines.append("")
        
        # Step size breakdown
        lines.append("STEP SIZE ANALYSIS BREAKDOWN:")
        sorted_step_sizes = sorted(report.step_size_analyses.keys())
        
        for step_size in sorted_step_sizes:
            analysis = report.step_size_analyses[step_size]
            lines.append(f"  Step Size {step_size}:")
            lines.append(f"    Success Rate: {analysis.success_rate:.1%}")
            lines.append(f"    Convergence Rate: {analysis.convergence_rate:.1%}")
            lines.append(f"    Avg Acceptance Rate: {analysis.average_acceptance_rate:.1%}")
            lines.append(f"    Stability Score: {analysis.energy_stability_score:.2f}")
            lines.append(f"    Avg Execution Time: {analysis.average_execution_time:.3f}s")
            lines.append(f"    Numerical Issues: {analysis.numerical_issues_count}")
            
            # Highlight optimal
            if step_size == report.optimal_step_size:
                lines.append("    ★ OPTIMAL STEP SIZE ★")
            lines.append("")
        
        # Implementation guidance
        lines.append("IMPLEMENTATION GUIDANCE:")
        if report.optimal_step_size != summary['base_step_size']:
            lines.append(f"1. Update inference config: step_size = {report.optimal_step_size}")
            lines.append(f"2. Test on validation set before production deployment")
            lines.append(f"3. Monitor acceptance rates in production (target: {self.target_acceptance_rate})")
            lines.append(f"4. Watch for numerical stability issues with new step size")
        else:
            lines.append("✅ Current step size is already optimal")
            lines.append("✅ No parameter changes needed")
        
        return "\n".join(lines)


def run_stability_analysis_with_inference_engine(
    inference_engine,
    base_step_size: float = 0.05,
    output_dir: str = "./stability_results"
) -> StabilityReport:
    """
    Convenience function to run stability analysis with an inference engine.
    
    Args:
        inference_engine: AlgebraInference instance
        base_step_size: Current step size to analyze around
        output_dir: Directory to save results
        
    Returns:
        StabilityReport
    """
    analyzer = NumericalStabilityAnalyzer(
        base_step_size=base_step_size,
        min_convergence_rate=0.8,
        target_acceptance_rate=(0.2, 0.6)
    )
    
    report = analyzer.run_stability_analysis(inference_engine, equation_count=25)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_results(report, output_path / "stability_analysis_results.json")
    
    # Save human-readable report
    report_text = analyzer.get_stability_report(report)
    with open(output_path / "stability_analysis_report.txt", 'w') as f:
        f.write(report_text)
    
    logger.info(f"Stability analysis complete: optimal step size = {report.optimal_step_size}")
    return report


if __name__ == "__main__":
    # Example usage
    print("Numerical Stability Analysis - Phase 2")
    print("Run run_stability_analysis_with_inference_engine() to test with inference engine")
    print("Or create NumericalStabilityAnalyzer() for custom testing")