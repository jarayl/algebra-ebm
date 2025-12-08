#!/usr/bin/env python3
"""
Distance Function Validation - Phase 2

Validates the L2 distance calculation in the equation decoder against known 
ground truth solutions to ensure mathematical correctness and expected behavior.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DistanceValidationResult:
    """Results from distance function validation tests."""
    test_name: str
    passed: bool
    expected_distance: float
    actual_distance: float
    tolerance: float
    error_message: Optional[str] = None
    test_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationSummary:
    """Summary of all distance validation tests."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    critical_failures: List[str]
    test_results: List[DistanceValidationResult]


class DistanceFunctionValidator:
    """
    Validates distance function correctness against mathematical ground truth.
    
    Tests the L2 distance calculation used in equation decoding to ensure:
    1. Mathematical correctness of distance computation
    2. Proper handling of identical embeddings (distance = 0)
    3. Triangle inequality preservation
    4. Scaling behavior matches expectations
    5. Numerical stability across different magnitudes
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize distance validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.test_results = []
        
        logger.info(f"Initialized DistanceFunctionValidator with tolerance={tolerance}")
    
    def create_test_embeddings(self, d_model: int = 128) -> Dict[str, torch.Tensor]:
        """
        Create test embeddings with known mathematical properties.
        
        Args:
            d_model: Embedding dimension
            
        Returns:
            Dictionary of named test embeddings
        """
        embeddings = {}
        
        # Zero embedding
        embeddings['zero'] = torch.zeros(d_model)
        
        # Unit embeddings (single 1.0 at different positions)
        embeddings['unit_first'] = torch.zeros(d_model)
        embeddings['unit_first'][0] = 1.0
        
        embeddings['unit_last'] = torch.zeros(d_model)
        embeddings['unit_last'][-1] = 1.0
        
        embeddings['unit_middle'] = torch.zeros(d_model)
        embeddings['unit_middle'][d_model // 2] = 1.0
        
        # Uniform embeddings
        embeddings['uniform_ones'] = torch.ones(d_model)
        embeddings['uniform_twos'] = torch.ones(d_model) * 2.0
        embeddings['uniform_halves'] = torch.ones(d_model) * 0.5
        
        # Random normalized embeddings (reproducible)
        torch.manual_seed(42)
        embeddings['random_norm'] = torch.randn(d_model)
        embeddings['random_norm'] = embeddings['random_norm'] / torch.norm(embeddings['random_norm'])
        
        torch.manual_seed(123)
        embeddings['random_scaled'] = torch.randn(d_model) * 5.0
        
        # Specific geometric cases - orthogonal normalized vectors
        embeddings['orthogonal_1'] = torch.zeros(d_model)
        embeddings['orthogonal_1'][:d_model//2] = 1.0 / np.sqrt(d_model//2)
        
        embeddings['orthogonal_2'] = torch.zeros(d_model)
        embeddings['orthogonal_2'][d_model//2:] = 1.0 / np.sqrt(d_model - d_model//2)
        
        logger.info(f"Created {len(embeddings)} test embeddings with d_model={d_model}")
        return embeddings
    
    def compute_l2_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Reference implementation of L2 distance calculation.
        
        This matches the sklearn NearestNeighbors euclidean distance used in
        the decoder's decode_embedding() method (algebra_encoder.py:537).
        
        Args:
            emb1: First embedding tensor
            emb2: Second embedding tensor
            
        Returns:
            L2 (Euclidean) distance as float
        """
        if emb1.shape != emb2.shape:
            raise ValueError(f"Embedding shape mismatch: {emb1.shape} vs {emb2.shape}")
        
        # Convert to numpy arrays exactly like the decoder does
        emb1_np = emb1.detach().cpu().numpy().reshape(1, -1)
        emb2_np = emb2.detach().cpu().numpy().reshape(1, -1)
        
        # Calculate L2 distance: sqrt(sum((a - b)^2))
        # This matches sklearn.neighbors.NearestNeighbors euclidean metric
        diff = emb1_np - emb2_np
        distance = np.sqrt(np.sum(diff ** 2))
        
        return float(distance)
    
    def test_identical_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> List[DistanceValidationResult]:
        """Test that distance between identical embeddings is zero."""
        results = []
        
        for name, embedding in embeddings.items():
            distance = self.compute_l2_distance(embedding, embedding)
            
            passed = abs(distance) < self.tolerance
            result = DistanceValidationResult(
                test_name=f"identical_{name}",
                passed=passed,
                expected_distance=0.0,
                actual_distance=distance,
                tolerance=self.tolerance,
                error_message=None if passed else f"Distance should be 0.0, got {distance}",
                test_metadata={'embedding_name': name, 'test_type': 'identical'}
            )
            results.append(result)
            
            if not passed:
                logger.warning(f"FAILED: Identical embedding distance test for {name}: {distance}")
        
        logger.info(f"Completed {len(results)} identical embedding tests")
        return results
    
    def test_known_distances(self, embeddings: Dict[str, torch.Tensor]) -> List[DistanceValidationResult]:
        """Test specific embedding pairs with known mathematical distances."""
        results = []
        
        # Test cases with known expected distances
        test_cases = [
            # Zero to unit vectors should have distance 1.0
            ('zero', 'unit_first', 1.0, "Zero to unit vector"),
            ('zero', 'unit_last', 1.0, "Zero to unit vector (different position)"),
            ('zero', 'unit_middle', 1.0, "Zero to unit vector (middle position)"),
            
            # Between unit vectors (should be sqrt(2))
            ('unit_first', 'unit_last', np.sqrt(2), "Between orthogonal unit vectors"),
            ('unit_first', 'unit_middle', np.sqrt(2), "Between orthogonal unit vectors (different)"),
            
            # Uniform scaling tests
            ('uniform_ones', 'uniform_twos', np.sqrt(embeddings['uniform_ones'].shape[0]), 
             "Uniform scaling test"),
            ('zero', 'uniform_ones', np.sqrt(embeddings['uniform_ones'].shape[0]), 
             "Zero to uniform ones"),
            
            # Orthogonal normalized vectors test (should have distance sqrt(2))
            ('orthogonal_1', 'orthogonal_2', np.sqrt(2.0), 
             "Orthogonal normalized halves"),
        ]
        
        for emb1_name, emb2_name, expected_dist, description in test_cases:
            if emb1_name not in embeddings or emb2_name not in embeddings:
                logger.warning(f"Skipping test: missing embedding {emb1_name} or {emb2_name}")
                continue
                
            actual_dist = self.compute_l2_distance(embeddings[emb1_name], embeddings[emb2_name])
            
            # Use relative tolerance for larger distances
            rel_tolerance = max(self.tolerance, abs(expected_dist) * 1e-6)
            passed = abs(actual_dist - expected_dist) < rel_tolerance
            
            result = DistanceValidationResult(
                test_name=f"known_distance_{emb1_name}_to_{emb2_name}",
                passed=passed,
                expected_distance=expected_dist,
                actual_distance=actual_dist,
                tolerance=rel_tolerance,
                error_message=None if passed else f"{description}: expected {expected_dist:.6f}, got {actual_dist:.6f}",
                test_metadata={
                    'emb1_name': emb1_name, 
                    'emb2_name': emb2_name, 
                    'description': description,
                    'test_type': 'known_distance'
                }
            )
            results.append(result)
            
            if not passed:
                logger.warning(f"FAILED: {description}: expected {expected_dist:.6f}, got {actual_dist:.6f}")
        
        logger.info(f"Completed {len(results)} known distance tests")
        return results
    
    def test_triangle_inequality(self, embeddings: Dict[str, torch.Tensor]) -> List[DistanceValidationResult]:
        """Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        results = []
        
        # Test triangle inequality on key embedding triples
        test_triples = [
            ('zero', 'unit_first', 'unit_last'),
            ('zero', 'uniform_ones', 'uniform_twos'),
            ('orthogonal_1', 'orthogonal_2', 'uniform_ones'),
            ('unit_first', 'unit_middle', 'unit_last'),
            ('random_norm', 'uniform_halves', 'zero')
        ]
        
        for a_name, b_name, c_name in test_triples:
            if not all(name in embeddings for name in [a_name, b_name, c_name]):
                continue
                
            # Calculate distances
            d_ac = self.compute_l2_distance(embeddings[a_name], embeddings[c_name])
            d_ab = self.compute_l2_distance(embeddings[a_name], embeddings[b_name])
            d_bc = self.compute_l2_distance(embeddings[b_name], embeddings[c_name])
            
            # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            triangle_sum = d_ab + d_bc
            passed = d_ac <= triangle_sum + self.tolerance
            
            result = DistanceValidationResult(
                test_name=f"triangle_inequality_{a_name}_{b_name}_{c_name}",
                passed=passed,
                expected_distance=triangle_sum,  # Upper bound
                actual_distance=d_ac,
                tolerance=self.tolerance,
                error_message=None if passed else f"Triangle inequality violated: {d_ac:.6f} > {triangle_sum:.6f}",
                test_metadata={
                    'a_name': a_name, 'b_name': b_name, 'c_name': c_name,
                    'd_ab': d_ab, 'd_bc': d_bc, 'd_ac': d_ac,
                    'test_type': 'triangle_inequality'
                }
            )
            results.append(result)
            
            if not passed:
                logger.warning(f"FAILED: Triangle inequality {a_name}-{b_name}-{c_name}: "
                             f"{d_ac:.6f} > {triangle_sum:.6f}")
        
        logger.info(f"Completed {len(results)} triangle inequality tests")
        return results
    
    def test_symmetry(self, embeddings: Dict[str, torch.Tensor]) -> List[DistanceValidationResult]:
        """Test distance symmetry: d(a,b) = d(b,a)."""
        results = []
        
        # Test pairs for symmetry
        test_pairs = [
            ('zero', 'unit_first'),
            ('uniform_ones', 'uniform_twos'),
            ('orthogonal_1', 'orthogonal_2'),
            ('random_norm', 'random_scaled'),
            ('unit_middle', 'uniform_halves')
        ]
        
        for a_name, b_name in test_pairs:
            if a_name not in embeddings or b_name not in embeddings:
                continue
                
            d_ab = self.compute_l2_distance(embeddings[a_name], embeddings[b_name])
            d_ba = self.compute_l2_distance(embeddings[b_name], embeddings[a_name])
            
            passed = abs(d_ab - d_ba) < self.tolerance
            
            result = DistanceValidationResult(
                test_name=f"symmetry_{a_name}_{b_name}",
                passed=passed,
                expected_distance=d_ab,
                actual_distance=d_ba,
                tolerance=self.tolerance,
                error_message=None if passed else f"Symmetry violated: d({a_name},{b_name})={d_ab:.6f} != d({b_name},{a_name})={d_ba:.6f}",
                test_metadata={
                    'a_name': a_name, 'b_name': b_name,
                    'd_ab': d_ab, 'd_ba': d_ba,
                    'test_type': 'symmetry'
                }
            )
            results.append(result)
            
            if not passed:
                logger.warning(f"FAILED: Symmetry test {a_name}-{b_name}: {d_ab:.6f} != {d_ba:.6f}")
        
        logger.info(f"Completed {len(results)} symmetry tests")
        return results
    
    def test_sklearn_consistency(self, embeddings: Dict[str, torch.Tensor]) -> List[DistanceValidationResult]:
        """Test that our distance calculation matches sklearn's implementation."""
        results = []
        
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            result = DistanceValidationResult(
                test_name="sklearn_consistency_import",
                passed=False,
                expected_distance=0.0,
                actual_distance=float('inf'),
                tolerance=self.tolerance,
                error_message="sklearn not available for validation",
                test_metadata={'test_type': 'sklearn_consistency'}
            )
            results.append(result)
            return results
        
        # Test pairs for consistency
        test_pairs = [
            ('zero', 'unit_first'),
            ('uniform_ones', 'random_norm'),
            ('orthogonal_1', 'orthogonal_2')
        ]
        
        for emb1_name, emb2_name in test_pairs:
            if emb1_name not in embeddings or emb2_name not in embeddings:
                continue
            
            emb1, emb2 = embeddings[emb1_name], embeddings[emb2_name]
            
            # Our implementation
            our_distance = self.compute_l2_distance(emb1, emb2)
            
            # sklearn implementation (matching decoder usage)
            emb1_np = emb1.detach().cpu().numpy().reshape(1, -1)
            emb2_np = emb2.detach().cpu().numpy().reshape(1, -1)
            
            nn_search = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_search.fit(emb2_np)
            distances, _ = nn_search.kneighbors(emb1_np)
            sklearn_distance = distances[0][0]
            
            passed = abs(our_distance - sklearn_distance) < self.tolerance
            
            result = DistanceValidationResult(
                test_name=f"sklearn_consistency_{emb1_name}_{emb2_name}",
                passed=passed,
                expected_distance=sklearn_distance,
                actual_distance=our_distance,
                tolerance=self.tolerance,
                error_message=None if passed else f"Distance mismatch: ours={our_distance:.6f}, sklearn={sklearn_distance:.6f}",
                test_metadata={
                    'emb1_name': emb1_name, 'emb2_name': emb2_name,
                    'test_type': 'sklearn_consistency'
                }
            )
            results.append(result)
        
        logger.info(f"Completed {len(results)} sklearn consistency tests")
        return results
    
    def test_decoder_integration(self, encoder, decoder) -> List[DistanceValidationResult]:
        """
        Test distance calculation integration with actual encoder/decoder.
        
        Args:
            encoder: Encoder instance (CharacterLevelEncoder or ASTEncoder)
            decoder: EquationDecoder instance with built candidate set
            
        Returns:
            List of validation results
        """
        results = []
        
        if decoder.nn_search is None:
            result = DistanceValidationResult(
                test_name="decoder_integration_setup",
                passed=False,
                expected_distance=0.0,
                actual_distance=float('inf'),
                tolerance=self.tolerance,
                error_message="Decoder candidate set not built - call build_candidate_set() first",
                test_metadata={'test_type': 'integration_setup'}
            )
            results.append(result)
            return results
        
        # Test equations with known candidates - covering different complexities
        test_equations = [
            "x=0",  # Should be in default candidate set
            "x=1",  # Should be in default candidate set
            "x=2",  # Should be in default candidate set  
            "2*x=4",  # Should be in default candidate set
            "x+1=2",  # Should be in default candidate set
            "3*x=6"   # Should be in default candidate set
        ]
        
        for eq in test_equations:
            try:
                # Encode equation
                embedding = encoder.encode_equation_string(eq)
                
                # Decode and get distance
                decoded_eq, distance = decoder.decode_embedding(embedding)
                
                # Validate distance is non-negative and finite
                distance_valid = (distance >= 0 and 
                                distance != float('inf') and 
                                distance == distance)  # Not NaN
                
                # If equation should match exactly, distance should be very small
                exact_match_expected = decoded_eq == eq
                if exact_match_expected:
                    distance_reasonable = distance < 1e-8  # Very small for exact match
                elif decoded_eq is not None:
                    distance_reasonable = distance < 10.0  # Reasonable upper bound for valid matches
                else:
                    distance_reasonable = True  # No match found is acceptable
                
                passed = distance_valid and distance_reasonable
                
                result = DistanceValidationResult(
                    test_name=f"decoder_integration_{eq.replace('*', 'star').replace('+', 'plus').replace('=', 'eq')}",
                    passed=passed,
                    expected_distance=0.0 if exact_match_expected else float('nan'),
                    actual_distance=distance,
                    tolerance=self.tolerance,
                    error_message=None if passed else f"Invalid distance for equation {eq}: {distance}",
                    test_metadata={
                        'equation': eq,
                        'decoded': decoded_eq,
                        'exact_match': exact_match_expected,
                        'test_type': 'decoder_integration'
                    }
                )
                results.append(result)
                
            except Exception as e:
                result = DistanceValidationResult(
                    test_name=f"decoder_integration_error_{eq.replace('*', 'star').replace('+', 'plus').replace('=', 'eq')}",
                    passed=False,
                    expected_distance=0.0,
                    actual_distance=float('inf'),
                    tolerance=self.tolerance,
                    error_message=f"Integration test failed for {eq}: {e}",
                    test_metadata={
                        'equation': eq,
                        'error': str(e),
                        'test_type': 'decoder_integration_error'
                    }
                )
                results.append(result)
        
        logger.info(f"Completed {len(results)} decoder integration tests")
        return results
    
    def run_all_tests(
        self, 
        d_model: int = 128,
        encoder=None, 
        decoder=None
    ) -> ValidationSummary:
        """
        Run complete distance function validation test suite.
        
        Args:
            d_model: Embedding dimension for test embeddings
            encoder: Optional encoder for integration testing
            decoder: Optional decoder for integration testing
            
        Returns:
            ValidationSummary with complete test results
        """
        logger.info(f"Starting complete distance validation test suite with d_model={d_model}")
        
        # Create test embeddings
        embeddings = self.create_test_embeddings(d_model)
        
        # Run all test categories
        all_results = []
        
        # Mathematical correctness tests
        all_results.extend(self.test_identical_embeddings(embeddings))
        all_results.extend(self.test_known_distances(embeddings))
        all_results.extend(self.test_triangle_inequality(embeddings))
        all_results.extend(self.test_symmetry(embeddings))
        
        # Implementation consistency tests
        all_results.extend(self.test_sklearn_consistency(embeddings))
        
        # Integration tests if components provided
        if encoder is not None and decoder is not None:
            all_results.extend(self.test_decoder_integration(encoder, decoder))
        
        # Store results
        self.test_results = all_results
        
        # Generate summary
        passed_count = sum(1 for result in all_results if result.passed)
        failed_count = len(all_results) - passed_count
        pass_rate = passed_count / len(all_results) if all_results else 0.0
        
        # Identify critical failures (mathematical property violations)
        critical_test_types = ['identical', 'known_distance', 'triangle_inequality', 'symmetry', 'sklearn_consistency']
        critical_failures = [
            result.test_name for result in all_results 
            if not result.passed and any(ct in result.test_name for ct in critical_test_types)
        ]
        
        summary = ValidationSummary(
            total_tests=len(all_results),
            passed_tests=passed_count,
            failed_tests=failed_count,
            pass_rate=pass_rate,
            critical_failures=critical_failures,
            test_results=all_results
        )
        
        logger.info(f"Distance validation complete: {passed_count}/{len(all_results)} tests passed "
                   f"({pass_rate:.1%}), {len(critical_failures)} critical failures")
        
        return summary
    
    def save_results(self, summary: ValidationSummary, output_path: str) -> None:
        """Save validation results to JSON file."""
        # Convert to serializable format
        serializable_results = []
        for result in summary.test_results:
            serializable_results.append({
                'test_name': result.test_name,
                'passed': result.passed,
                'expected_distance': result.expected_distance,
                'actual_distance': result.actual_distance,
                'tolerance': result.tolerance,
                'error_message': result.error_message,
                'test_metadata': result.test_metadata
            })
        
        serializable_summary = {
            'summary': {
                'total_tests': summary.total_tests,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'pass_rate': summary.pass_rate,
                'critical_failures': summary.critical_failures
            },
            'test_results': serializable_results,
            'validation_timestamp': str(Path().cwd()),
            'tolerance_used': self.tolerance
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        logger.info(f"Validation results saved to {output_file}")
    
    def get_validation_report(self, summary: ValidationSummary) -> str:
        """Generate human-readable validation report."""
        report = ["=== DISTANCE FUNCTION VALIDATION REPORT ===\n"]
        
        # Overall summary
        report.append(f"OVERALL RESULTS:")
        report.append(f"  Total Tests: {summary.total_tests}")
        report.append(f"  Passed: {summary.passed_tests}")
        report.append(f"  Failed: {summary.failed_tests}")
        report.append(f"  Pass Rate: {summary.pass_rate:.1%}")
        report.append("")
        
        # Critical failures
        if summary.critical_failures:
            report.append("⚠️  CRITICAL FAILURES (Mathematical Property Violations):")
            for failure in summary.critical_failures:
                report.append(f"  - {failure}")
            report.append("")
        else:
            report.append("✅ No critical mathematical property violations detected.")
            report.append("")
        
        # Test category breakdown
        test_categories = {}
        for result in summary.test_results:
            category = result.test_metadata.get('test_type', 'unknown') if result.test_metadata else 'unknown'
            if category not in test_categories:
                test_categories[category] = {'passed': 0, 'failed': 0}
            
            if result.passed:
                test_categories[category]['passed'] += 1
            else:
                test_categories[category]['failed'] += 1
        
        report.append("TEST CATEGORY BREAKDOWN:")
        for category, counts in test_categories.items():
            total = counts['passed'] + counts['failed']
            rate = counts['passed'] / total if total > 0 else 0.0
            report.append(f"  {category}: {counts['passed']}/{total} passed ({rate:.1%})")
        report.append("")
        
        # Detailed failures
        failures = [r for r in summary.test_results if not r.passed]
        if failures:
            report.append("DETAILED FAILURE ANALYSIS:")
            for failure in failures:
                report.append(f"  Test: {failure.test_name}")
                if failure.error_message:
                    report.append(f"    Error: {failure.error_message}")
                report.append(f"    Expected: {failure.expected_distance:.6f}")
                report.append(f"    Actual: {failure.actual_distance:.6f}")
                report.append(f"    Tolerance: {failure.tolerance:.6f}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if not summary.critical_failures:
            report.append("✅ Distance function appears mathematically correct")
            report.append("✅ Safe to proceed with threshold optimization")
        else:
            report.append("❌ Critical mathematical issues detected")
            report.append("❌ Fix distance calculation before using in production")
            
        if summary.pass_rate < 0.95:
            report.append("⚠️  Consider investigating failed tests")
        
        return "\n".join(report)


def validate_distance_function_with_encoder(
    encoder_type: str = "character",
    d_model: int = 128,
    output_dir: str = "./validation_results"
) -> ValidationSummary:
    """
    Convenience function to validate distance function with a specific encoder.
    
    Args:
        encoder_type: Either "character" or "ast"
        d_model: Embedding dimension
        output_dir: Directory to save results
        
    Returns:
        ValidationSummary
    """
    from src.algebra.algebra_encoder import create_character_encoder, create_ast_encoder, create_decoder_with_default_candidates
    
    # Create encoder
    if encoder_type == "character":
        encoder = create_character_encoder(d_model=d_model)
    elif encoder_type == "ast":
        encoder = create_ast_encoder(d_model=d_model)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create decoder with default candidates
    decoder = create_decoder_with_default_candidates(encoder, distance_threshold=10.0)
    
    # Run validation
    validator = DistanceFunctionValidator(tolerance=1e-6)
    summary = validator.run_all_tests(d_model=d_model, encoder=encoder, decoder=decoder)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    validator.save_results(summary, output_path / f"distance_validation_{encoder_type}.json")
    
    # Save human-readable report
    report = validator.get_validation_report(summary)
    with open(output_path / f"distance_validation_report_{encoder_type}.txt", 'w') as f:
        f.write(report)
    
    logger.info(f"Validation complete for {encoder_type} encoder: {summary.pass_rate:.1%} pass rate")
    return summary


if __name__ == "__main__":
    # Example usage
    print("Distance Function Validation - Phase 2")
    print("Run validate_distance_function_with_encoder() to test with specific encoder")
    print("Or create DistanceFunctionValidator() for custom testing")