#!/usr/bin/env python3
"""
Template Energy Comparison - Task T5 Implementation

Implements the exact template analysis as specified in the implementation plan
for Task T5: Template Energy Comparison. This script compares energies of 
ground truth vs common templates to identify systematic failures.

Dependencies: T1 (checkpoint loading), T2 (energy computation), T4 (statistical framework)
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import existing components
from src.algebra.algebra_inference import load_rule_models, AlgebraInference
from src.algebra.algebra_encoder import CharacterLevelEncoder, create_character_encoder
from template_analysis_framework import (
    TemplateAnalysisFramework, 
    TemplatePattern, 
    TemplateComparisonResult,
    run_template_energy_analysis
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_energy_and_gradient(models: Dict[str, Any], equation: str, candidate: str) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Compute energy and gradient for equation-candidate pair using T2's exact implementation.
    
    This function provides the exact interface expected by the implementation plan
    and ensures consistency with T2 conditioning test framework.
    
    Args:
        models: Dictionary of rule models (from load_rule_models)
        equation: Input equation string
        candidate: Candidate solution string
        
    Returns:
        Tuple of (energy_value, gradient_tensor)
    """
    # Use T2's exact implementation for consistency
    try:
        from debug_conditioning_test import compute_energy_and_gradient as t2_compute_energy
        return t2_compute_energy(models, equation, candidate)
    except ImportError:
        logger.warning("T2 implementation not available, using fallback")
        
    # Fallback implementation if T2 module not available
    encoder = create_character_encoder(d_model=128)
    
    # Create inference engine
    inference = AlgebraInference(
        rule_models=models,
        encoder=encoder,
        device='cpu'
    )
    
    try:
        # Encode equation and candidate
        equation_embedding = encoder.encode([equation])
        candidate_embedding = encoder.encode([candidate])
        
        if equation_embedding is None or candidate_embedding is None:
            logger.warning(f"Failed to encode equation '{equation}' or candidate '{candidate}'")
            return float('inf'), None
        
        # Move to device
        device = next(iter(models.values())).device if models else 'cpu'
        equation_embedding = equation_embedding.to(device)
        candidate_embedding = candidate_embedding.to(device)
        
        # Compute energy and gradient using inference engine
        energy, gradient = inference.compute_energy_and_gradient(
            equation_embedding, 
            candidate_embedding, 
            k=0  # Use landscape index 0 for consistency
        )
        
        return energy.item(), gradient
        
    except Exception as e:
        logger.error(f"Error computing energy for '{equation}', '{candidate}': {e}")
        return float('inf'), None


def compute_embedding_distance(equation: str, candidate: str) -> float:
    """
    Compute embedding distance between equation and candidate using T3's implementation.
    
    Args:
        equation: Input equation string
        candidate: Candidate solution string
        
    Returns:
        L2 distance in embedding space
    """
    # Use T3's exact implementation for consistency
    try:
        from debug_distance_validation import test_distance_function
        # T3 doesn't expose a direct function, so use existing algebra_evaluation
        from algebra_evaluation import compute_embedding_distance as eval_distance
        return eval_distance(equation, candidate)
    except ImportError:
        logger.warning("T3/evaluation implementation not available, using fallback")
    
    # Fallback implementation
    try:
        # Create encoder
        encoder = create_character_encoder(d_model=128)
        
        # Encode both strings
        equation_embedding = encoder.encode([equation])
        candidate_embedding = encoder.encode([candidate])
        
        if equation_embedding is None or candidate_embedding is None:
            logger.warning(f"Failed to encode for distance calculation")
            return float('inf')
        
        # Compute L2 distance
        distance = torch.norm(equation_embedding - candidate_embedding, dim=1).item()
        return distance
        
    except Exception as e:
        logger.error(f"Error computing distance for '{equation}', '{candidate}': {e}")
        return float('inf')


def analyze_template_energies() -> bool:
    """
    Main template energy analysis function implementing Task T5 requirements.
    
    Compares energies of ground truth vs common templates as specified in the
    implementation plan. This is the core function that Task T5 will call.
    
    Returns:
        True if ground truth consistently has lower energy than templates,
        False if templates frequently have lower energy (indicating problems)
    """
    logger.info("=== TEMPLATE ENERGY ANALYSIS (Task T5) ===")
    
    try:
        # Load rule models (from Task T1 checkpoint verification)
        logger.info("Loading rule models...")
        try:
            from debug_conditioning_test import load_rule_models_wrapper
            models = load_rule_models_wrapper()
        except ImportError:
            logger.warning("T2 wrapper not available, using direct load_rule_models")
            models = load_rule_models(['distribute', 'combine', 'isolate', 'divide'])
        
        if not models:
            logger.error("No rule models loaded - cannot perform energy analysis")
            return False
        
        logger.info(f"Loaded {len(models)} rule models")
        
        # Test cases: equation -> correct solution (from implementation plan)
        test_cases = [
            ("2*x=10", "x=5"),
            ("3*x+6=21", "x=5"), 
            ("x-4=7", "x=11"),
            ("-2*x=14", "x=-7")
        ]
        
        # Common problematic templates from logs (from implementation plan)
        problem_templates = ["x=4", "2*x+x=6", "2*x+3*x+1=11"]
        
        results = []
        
        for equation, true_solution in test_cases:
            logger.info(f"\nAnalyzing equation: {equation}")
            
            # Compute ground truth energy and distance
            true_energy, _ = compute_energy_and_gradient(models, equation, true_solution)
            true_distance = compute_embedding_distance(equation, true_solution)
            
            logger.info(f"True solution '{true_solution}': E={true_energy:.3f}, dist={true_distance:.3f}")
            
            for template in problem_templates:
                # Compute template energy and distance
                template_energy, _ = compute_energy_and_gradient(models, equation, template)
                template_distance = compute_embedding_distance(equation, template)
                
                energy_diff = template_energy - true_energy
                logger.info(f"Template '{template}': E={template_energy:.3f} (Δ={energy_diff:+.3f}), dist={template_distance:.3f}")
                
                if energy_diff < 0:  # Template has lower energy than truth
                    logger.error(f"❌ CRITICAL: Template '{template}' has lower energy than ground truth!")
                    results.append(False)
                else:
                    logger.info(f"✅ Ground truth has lower energy than template")
                    results.append(True)
        
        # Calculate success rate
        success_rate = sum(results) / len(results) if results else 0.0
        logger.info(f"\nGround truth energy advantage: {success_rate:.1%}")
        
        # Determine overall result based on implementation plan criteria
        if success_rate < 0.8:
            logger.error("❌ CRITICAL: Templates frequently have lower energy than ground truth")
            logger.error("This indicates a systematic conditioning failure in the energy function")
            return False
        
        logger.info("✅ Ground truth consistently has lower energy than templates")
        logger.info("Template energy comparison PASSED - energy landscape properly calibrated")
        return True
        
    except Exception as e:
        logger.error(f"Template energy analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_extended_template_analysis(output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run extended template analysis using the full framework capabilities.
    
    Args:
        output_file: Optional path to save detailed analysis results
        
    Returns:
        Dictionary with detailed analysis results
    """
    logger.info("=== EXTENDED TEMPLATE ANALYSIS ===")
    
    try:
        # Load models and encoder
        models = load_rule_models(['distribute', 'combine', 'isolate', 'divide'])
        encoder = create_character_encoder(d_model=128)
        
        if not models:
            logger.error("No rule models loaded")
            return {'success': False, 'error': 'No models available'}
        
        # Extended test cases for comprehensive analysis
        extended_test_cases = [
            ("2*x=10", "x=5"),
            ("3*x+6=21", "x=5"), 
            ("x-4=7", "x=11"),
            ("-2*x=14", "x=-7"),
            ("4*x+8=20", "x=3"),
            ("x+10=15", "x=5"),
            ("5*x-10=0", "x=2"),
            ("-3*x+9=0", "x=3")
        ]
        
        # Run comprehensive analysis
        report = run_template_energy_analysis(
            rule_models=models,
            encoder=encoder,
            test_cases=extended_test_cases,
            output_path=output_file,
            device='cpu'
        )
        
        # Extract key metrics
        analysis_results = {
            'success': True,
            'total_comparisons': report.total_comparisons,
            'template_advantage_rate': report.summary_stats.get('template_advantage_rate', 0.0),
            'problematic_cases': len(report.problematic_cases),
            'energy_separation_quality': 'good' if report.summary_stats.get('template_advantage_rate', 0.0) < 0.2 else 'poor',
            'recommendations': report.recommendations,
            'detailed_report': report
        }
        
        logger.info(f"Extended analysis complete: {analysis_results['problematic_cases']} problematic cases")
        logger.info(f"Template advantage rate: {analysis_results['template_advantage_rate']:.1%}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Extended template analysis failed: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    """
    Main entry point for Task T5: Template Energy Comparison
    
    This script can be run standalone or imported by other debugging scripts.
    It implements the exact specification from the implementation plan.
    """
    
    print("=== Task T5: Template Energy Comparison ===")
    print("Dependencies: T1 (checkpoint loading), T2 (energy computation), T4 (statistical framework)")
    print()
    
    # Run basic template analysis (as specified in implementation plan)
    basic_success = analyze_template_energies()
    
    print()
    print("=== BASIC TEMPLATE ANALYSIS RESULT ===")
    print(f"Result: {'PASS' if basic_success else 'FAIL'}")
    
    # Run extended analysis if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--extended':
        print("\n=== Running Extended Analysis ===")
        
        output_file = 'template_analysis_results.json' if len(sys.argv) > 2 else None
        extended_results = run_extended_template_analysis(output_file)
        
        if extended_results['success']:
            print(f"Extended analysis complete:")
            print(f"  Total comparisons: {extended_results['total_comparisons']}")
            print(f"  Template advantage rate: {extended_results['template_advantage_rate']:.1%}")
            print(f"  Energy separation quality: {extended_results['energy_separation_quality']}")
            
            if output_file:
                print(f"  Detailed results saved to: {output_file}")
        else:
            print(f"Extended analysis failed: {extended_results['error']}")
    
    print()
    print("Task T5 Template Energy Comparison - Complete")
    
    # Exit with appropriate code for shell integration
    sys.exit(0 if basic_success else 1)