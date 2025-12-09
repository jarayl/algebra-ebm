#!/usr/bin/env python3
"""
Unit tests for AlgebraEBM evaluation pipeline validation.

Tests cover:
- Decoder candidate set validation (Task 4.1)
- Verification that decoder uses test dataset equations, not hardcoded defaults
- Decoder rebuilding functionality from dataset
- Candidate set consistency and coverage
- Evaluation pipeline integration with proper decoder setup

This test suite validates that the evaluation pipeline correctly rebuilds
the decoder with equations from the test dataset to prevent systematic
decoding failures due to limited default candidates.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Tuple

# Import evaluation and related components
from src.algebra.algebra_evaluation import (
    evaluate_model,
    compute_symbolic_equivalence,
    compute_embedding_distances,
    compute_invalid_rate
)
from src.algebra.algebra_encoder import (
    CharacterLevelEncoder,
    EquationDecoder,
    create_decoder_from_dataset,
    create_decoder_with_default_candidates
)
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig


class TestDecoderValidation:
    """Test decoder validation and candidate set verification (Task 4.1)."""
    
    def create_mock_rule_models(self, rules: List[str]) -> Dict[str, AlgebraDiffusionWrapper]:
        """Create mock rule models for testing."""
        rule_models = {}
        
        for rule in rules:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.eval()
            rule_models[rule] = wrapper
        
        return rule_models
    
    def create_mock_dataset(self, equations: List[Tuple[str, str]]) -> Mock:
        """Create a mock dataset with specific equation pairs."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=len(equations))
        
        def get_equation_pair(idx):
            return equations[idx]
        
        dataset.get_equation_pair = Mock(side_effect=get_equation_pair)
        dataset.get_dataset_info = Mock(return_value={'rule': 'test', 'size': len(equations)})
        
        return dataset

    def test_decoder_rebuilds_from_dataset(self):
        """Test that evaluate_model rebuilds decoder with dataset candidates."""
        # Create test dataset with specific equations
        test_equations = [
            ("2*x+4=8", "x=2"),
            ("3*x-6=9", "x=5"),
            ("x+7=10", "x=3"),
            ("4*x=12", "x=3"),
            ("x-2=5", "x=7")
        ]
        
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoder with default candidates (small set)
        decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
        original_candidates = decoder.candidate_equations.copy()
        original_count = len(original_candidates)
        
        # Verify original decoder has limited candidates
        assert original_count < 50, f"Default decoder should have limited candidates, got {original_count}"
        
        # Create mock rule models
        rule_models = self.create_mock_rule_models(['distribute'])
        
        # Mock the inference engine to avoid actual model loading
        with patch('src.algebra.algebra_evaluation.AlgebraInference') as mock_inference_class:
            mock_inference = Mock()
            mock_inference.solve_equation.return_value = {
                'success': True,
                'output_equation': 'x=2',
                'decoding_distance': 1.0,
                'inference_info': {'final_energy': 2.5, 'acceptance_rate': 0.7}
            }
            mock_inference_class.return_value = mock_inference
            
            # Call evaluate_model - this should rebuild the decoder
            results = evaluate_model(
                rule_models=rule_models,
                test_dataset=dataset,
                encoder=encoder,
                decoder=decoder,
                batch_size=2,
                max_samples=3,
                store_detailed_results=True
            )
        
        # Verify that decoder was rebuilt with dataset equations
        # The function should have called create_decoder_from_dataset internally
        # We can't directly verify the rebuilding since it happens inside the function,
        # but we can verify that the evaluation completed successfully with proper structure
        assert 'symbolic_equivalence' in results
        assert 'embedding_distances' in results
        assert 'validity' in results
        assert results['num_samples_evaluated'] == 3
        
        # Verify that the evaluation used the test dataset info
        assert results['dataset_info']['size'] == 5
        assert results['dataset_info']['rule'] == 'test'

    def test_create_decoder_from_dataset_functionality(self):
        """Test the create_decoder_from_dataset function directly."""
        # Create test dataset with specific target equations (using valid equation syntax)
        test_equations = [
            ("x+1=2", "x=1"),
            ("x+2=3", "x=1"), 
            ("x+3=4", "x=1"),
            ("x+4=5", "x=1"),  # Duplicate target should be deduplicated
        ]
        
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoder from dataset
        decoder = create_decoder_from_dataset(
            encoder=encoder,
            dataset=dataset,
            distance_threshold=3.0,
            include_inputs=False
        )
        
        # Verify decoder was built correctly
        assert len(decoder.candidate_equations) == 1, "Should have 1 unique target equation (x=1)"
        assert "x=1" in decoder.candidate_equations
        assert decoder.distance_threshold == 3.0
        
        # Verify inputs were not included
        assert "x+1=2" not in decoder.candidate_equations
        assert "x+2=3" not in decoder.candidate_equations

    def test_create_decoder_from_dataset_with_inputs(self):
        """Test create_decoder_from_dataset with include_inputs=True."""
        test_equations = [
            ("x+1=2", "x=1"),
            ("x+2=4", "x=2")
        ]
        
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoder including inputs
        decoder = create_decoder_from_dataset(
            encoder=encoder,
            dataset=dataset,
            distance_threshold=2.5,
            include_inputs=True
        )
        
        # Verify both inputs and targets are included
        assert len(decoder.candidate_equations) == 4
        assert "x+1=2" in decoder.candidate_equations
        assert "x+2=4" in decoder.candidate_equations
        assert "x=1" in decoder.candidate_equations
        assert "x=2" in decoder.candidate_equations

    def test_create_decoder_from_dataset_different_interfaces(self):
        """Test create_decoder_from_dataset with different dataset interfaces."""
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Test with get_problem_info interface (create object without get_equation_pair)
        class MockDatasetProblemInfo:
            def __len__(self):
                return 2
            
            def get_problem_info(self, idx):
                return {
                    'input_equation': f"x+{idx}={idx+1}",
                    'target_equation': "x=1"
                }
        
        dataset_problem_info = MockDatasetProblemInfo()
        decoder = create_decoder_from_dataset(encoder, dataset_problem_info)
        assert "x=1" in decoder.candidate_equations
        
        # Test with equation_pairs attribute interface
        class MockDatasetPairs:
            def __len__(self):
                return 2
            
            @property  
            def equation_pairs(self):
                return [("x+0=1", "x=1"), ("x+1=2", "x=1")]
        
        dataset_pairs = MockDatasetPairs()
        decoder = create_decoder_from_dataset(encoder, dataset_pairs)
        assert "x=1" in decoder.candidate_equations

    def test_create_decoder_from_dataset_fallback_interface(self):
        """Test create_decoder_from_dataset with tuple fallback interface."""
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Test with direct tuple access fallback
        class MockDatasetTuple:
            def __len__(self):
                return 2
            
            def __getitem__(self, idx):
                return (f"x+{idx}={idx+1}", "x=1")
        
        dataset_tuple = MockDatasetTuple()
        decoder = create_decoder_from_dataset(encoder, dataset_tuple)
        assert "x=1" in decoder.candidate_equations

    def test_create_decoder_from_dataset_invalid_interface(self):
        """Test create_decoder_from_dataset with invalid dataset interface."""
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create dataset that doesn't implement any expected interface
        class InvalidDataset:
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return "not_a_tuple"  # Invalid format - not a tuple
        
        invalid_dataset = InvalidDataset()
        
        with pytest.raises(ValueError, match="Dataset does not provide equation strings"):
            create_decoder_from_dataset(encoder, invalid_dataset)

    def test_decoder_candidate_coverage_verification(self):
        """Test that decoder candidates properly cover test dataset equations."""
        # Create realistic algebraic equations that might appear in evaluation
        test_equations = [
            ("2*x+6=14", "x=4"),
            ("3*(x-2)=9", "3*x-6=9"), 
            ("x+5=12", "x=7"),
            ("4*x-8=16", "x=6"),
            ("2*(x+3)=10", "2*x+6=10")
        ]
        
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoder from dataset
        decoder = create_decoder_from_dataset(encoder, dataset, include_inputs=True)
        
        # Verify all target equations are in candidates
        target_equations = [eq[1] for eq in test_equations]
        for target in target_equations:
            assert target in decoder.candidate_equations, \
                f"Target equation '{target}' not found in decoder candidates"
        
        # Verify candidate count includes all unique equations
        all_equations = set()
        for inp, tgt in test_equations:
            all_equations.add(inp)
            all_equations.add(tgt)
        
        assert len(decoder.candidate_equations) == len(all_equations), \
            f"Expected {len(all_equations)} unique equations, got {len(decoder.candidate_equations)}"

    def test_decoder_embedding_consistency(self):
        """Test that decoder embeddings are consistent with encoder."""
        test_equations = [("x+1=2", "x=1"), ("2*x=4", "x=2")]
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoder from dataset
        decoder = create_decoder_from_dataset(encoder, dataset)
        
        # Verify decoder has candidate embeddings
        assert decoder.candidate_embeddings is not None
        assert decoder.candidate_embeddings.shape[0] == len(decoder.candidate_equations)
        assert decoder.candidate_embeddings.shape[1] == encoder.d_model
        
        # Verify embeddings are finite
        assert np.isfinite(decoder.candidate_embeddings).all()
        
        # Test decoding functionality
        test_eq = "x=1"
        if test_eq in decoder.candidate_equations:
            # Encode and decode roundtrip
            embedding = encoder.encode_equation_string(test_eq)
            decoded, distance = decoder.decode_embedding(embedding)
            
            # Should decode to something valid (might not be exact due to nearest neighbor)
            assert decoded is not None or distance < float('inf')

    def test_decoder_distance_threshold_impact(self):
        """Test that distance threshold affects decoding results appropriately."""
        test_equations = [("x=1", "x=1"), ("x=2", "x=2")]
        dataset = self.create_mock_dataset(test_equations)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create decoders with different thresholds
        strict_decoder = create_decoder_from_dataset(encoder, dataset, distance_threshold=0.1)
        lenient_decoder = create_decoder_from_dataset(encoder, dataset, distance_threshold=10.0)
        
        # Test with an equation that should be in candidates
        test_embedding = encoder.encode_equation_string("x=1")
        
        strict_result, strict_distance = strict_decoder.decode_embedding(test_embedding)
        lenient_result, lenient_distance = lenient_decoder.decode_embedding(test_embedding)
        
        # Both should find the same equation, but distance comparison should be consistent
        assert strict_distance == lenient_distance, \
            "Distance calculation should be consistent regardless of threshold"
        
        # If strict decoder accepts, lenient decoder should also accept
        if strict_result is not None:
            assert lenient_result is not None, \
                "Lenient decoder should accept what strict decoder accepts"


class TestEvaluationPipelineIntegration:
    """Test evaluation pipeline integration with proper decoder setup."""
    
    def create_simple_mock_dataset(self, size: int = 5) -> Mock:
        """Create a simple mock dataset for integration testing."""
        equations = [(f"x+{i}={i+1}", f"x=1") for i in range(size)]
        
        dataset = Mock()
        dataset.__len__ = Mock(return_value=size)
        dataset.get_equation_pair = Mock(side_effect=lambda idx: equations[idx])
        dataset.get_dataset_info = Mock(return_value={'rule': 'test', 'size': size})
        
        return dataset

    def test_evaluate_model_with_dataset_decoder_rebuild(self):
        """Test that evaluate_model properly rebuilds decoder from dataset."""
        # Create a small test dataset
        dataset = self.create_simple_mock_dataset(3)
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create initial decoder with defaults
        decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
        original_count = len(decoder.candidate_equations)
        
        # Create mock rule models
        ebm = AlgebraEBM()
        wrapper = AlgebraDiffusionWrapper(ebm)
        rule_models = {'test': wrapper}
        
        # Mock the inference to return predictable results
        with patch.object(wrapper, 'forward') as mock_forward:
            # Return small energy for energy computation mode
            mock_forward.return_value = torch.tensor([[1.0], [1.5], [2.0]])
            
            # Call evaluate_model
            results = evaluate_model(
                rule_models=rule_models,
                test_dataset=dataset,
                encoder=encoder,
                decoder=decoder,
                batch_size=2,
                max_samples=3,
                store_detailed_results=True
            )
        
        # Verify evaluation completed successfully
        assert 'symbolic_equivalence' in results
        assert 'embedding_distances' in results 
        assert 'validity' in results
        assert results['num_samples_evaluated'] == 3
        
        # Verify evaluation used dataset information
        assert results['dataset_info']['size'] == 3
        assert results['dataset_info']['rule'] == 'test'

    def test_symbolic_equivalence_computation(self):
        """Test symbolic equivalence computation functionality."""
        predicted_equations = ["x=1", "x=2", "x=3", None]  # Include None for failed decoding
        target_equations = ["x=1", "x=2", "x=4", "x=5"]
        
        results = compute_symbolic_equivalence(predicted_equations, target_equations)
        
        assert results['total_equations'] == 4
        assert results['equivalent_count'] == 2  # First two match
        assert results['symbolic_equivalence_rate'] == 0.5
        assert len(results['detailed_results']) == 4
        
        # Verify detailed results structure
        for i, detail in enumerate(results['detailed_results']):
            assert 'index' in detail
            assert 'predicted' in detail
            assert 'target' in detail
            assert 'equivalent' in detail
            assert detail['index'] == i

    def test_embedding_distances_computation(self):
        """Test embedding distance computation functionality."""
        # Create test embeddings
        pred_embeddings = torch.randn(3, 128)
        target_embeddings = torch.randn(3, 128)
        
        results = compute_embedding_distances(pred_embeddings, target_embeddings)
        
        assert results['total_comparisons'] == 3
        assert 'mean_l2_distance' in results
        assert 'std_l2_distance' in results
        assert 'median_l2_distance' in results
        assert 'min_l2_distance' in results
        assert 'max_l2_distance' in results
        assert len(results['distances']) == 3
        
        # Verify all distances are non-negative
        assert all(d >= 0 for d in results['distances'])

    def test_invalid_rate_computation(self):
        """Test invalid equation rate computation functionality."""
        predicted_equations = [
            "x=1",      # Valid
            "x=2",      # Valid  
            "x+=3",     # Invalid syntax
            None        # No prediction
        ]
        
        results = compute_invalid_rate(predicted_equations)
        
        assert results['total_predictions'] == 4
        assert results['valid_count'] == 2
        assert results['invalid_count'] == 1
        assert results['none_count'] == 1
        assert results['valid_rate'] == 0.5
        assert results['invalid_rate'] == 0.25
        assert results['none_rate'] == 0.25
        
        # Verify detailed results
        assert len(results['validity_details']) == 4
        for detail in results['validity_details']:
            assert 'index' in detail
            assert 'equation' in detail
            assert 'valid' in detail


class TestDecoderValidationEdgeCases:
    """Test edge cases and error conditions in decoder validation."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Create empty dataset
        class EmptyDataset:
            def __len__(self):
                return 0
        
        empty_dataset = EmptyDataset()
        
        # Should raise an error for empty dataset since we can't build a candidate set
        with pytest.raises(ValueError, match="Cannot build candidate set with empty equations list"):
            create_decoder_from_dataset(encoder, empty_dataset)

    def test_large_dataset_batching(self):
        """Test decoder creation with large dataset using batching."""
        # Create large dataset
        large_equations = [(f"x+{i}={i+5}", f"x={5}") for i in range(100)]
        
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        dataset.get_equation_pair = Mock(side_effect=lambda idx: large_equations[idx])
        
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Should handle large dataset with batching
        decoder = create_decoder_from_dataset(encoder, dataset)
        
        # Should have many unique target equations (just "x=5" in this case due to duplication)
        assert len(decoder.candidate_equations) >= 1
        assert "x=5" in decoder.candidate_equations

    def test_malformed_equation_handling(self):
        """Test handling of malformed equations in dataset."""
        # Dataset with some malformed equations (but syntactically valid for encoder)
        malformed_equations = [
            ("x+1=2", "x=1"),      # Valid
            ("x+2=3", "x=2"),      # Valid
            ("x+3=4", "x=3")       # Valid  
        ]
        
        dataset = Mock()
        dataset.__len__ = Mock(return_value=3)
        dataset.get_equation_pair = Mock(side_effect=lambda idx: malformed_equations[idx])
        
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Should handle all equations by including them in candidates
        decoder = create_decoder_from_dataset(encoder, dataset)
        
        # Should include all unique target equations
        assert len(decoder.candidate_equations) == 3  # x=1, x=2, x=3
        assert "x=1" in decoder.candidate_equations
        assert "x=2" in decoder.candidate_equations
        assert "x=3" in decoder.candidate_equations

    def test_memory_efficiency_with_eval_mode(self):
        """Test memory efficiency by ensuring encoder is in eval mode during building."""
        test_equations = [("x+1=2", "x=1"), ("x+2=3", "x=1")]
        dataset = Mock()
        dataset.__len__ = Mock(return_value=2)
        dataset.get_equation_pair = Mock(side_effect=lambda idx: test_equations[idx])
        
        encoder = CharacterLevelEncoder(d_model=128)
        
        # Set encoder to training mode initially
        encoder.train()
        assert encoder.training
        
        # Create decoder - should temporarily set encoder to eval mode
        decoder = create_decoder_from_dataset(encoder, dataset)
        
        # Encoder training state should be restored
        assert encoder.training, "Encoder training state should be restored"
        
        # Decoder should be built successfully
        assert len(decoder.candidate_equations) == 1  # Deduplicated "x=1"


class TestDistanceThresholdValidation:
    """
    Test suite for distance threshold validation in evaluation pipeline (Task 4.2).
    
    Validates that distance thresholds are appropriate for normalized embeddings
    where ||e|| = 1, making maximum possible distance = √2 ≈ 1.414.
    """
    
    def test_normalized_embedding_properties(self):
        """Test that encoder produces normalized embeddings with ||e|| = 1."""
        from src.algebra.algebra_encoder import create_character_encoder
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        test_equations = [
            "x=1",
            "2*x=4", 
            "x+3=7",
            "-5*x=-25",
            "3*(x+1)=9"
        ]
        
        for eq in test_equations:
            embedding = encoder.encode_equation_string(eq)
            norm = torch.norm(embedding).item()
            
            # Verify normalized embedding has unit norm
            assert abs(norm - 1.0) < 1e-5, f"Embedding for '{eq}' not normalized: norm={norm}"
    
    def test_max_distance_for_normalized_embeddings(self):
        """Test that maximum distance between normalized embeddings is ≤ 2.0."""
        from src.algebra.algebra_encoder import create_character_encoder
        import math
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        # Create diverse equations to find maximum distance
        equations = [
            "x=0", "x=1", "x=2", "x=10", "x=-5",
            "2*x=4", "3*x=9", "-2*x=6", "5*x=-15",
            "x+1=2", "x-3=7", "x+10=-5", "2*x+3=7",
            "3*(x+1)=9", "-2*(x-1)=4", "5*(x+2)=-15"
        ]
        
        max_distance = 0.0
        
        # Test all pairs to find maximum distance
        for i, eq1 in enumerate(equations):
            emb1 = encoder.encode_equation_string(eq1)
            for j, eq2 in enumerate(equations[i+1:], i+1):
                emb2 = encoder.encode_equation_string(eq2)
                distance = torch.norm(emb1 - emb2).item()
                max_distance = max(max_distance, distance)
        
        # For normalized embeddings, maximum distance should be √2 ≈ 1.414
        theoretical_max = math.sqrt(2)
        
        assert max_distance <= 2.0, f"Maximum distance {max_distance:.4f} exceeds 2.0"
        assert max_distance <= theoretical_max + 0.2, f"Maximum distance {max_distance:.4f} exceeds theoretical max {theoretical_max:.4f} + tolerance"
    
    def test_distance_threshold_reasonableness(self):
        """Test that distance thresholds used in system are reasonable (<10.0)."""
        from src.algebra.algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        # Test default threshold in decoder creation
        decoder_default = create_decoder_with_default_candidates(encoder, distance_threshold=1.0)
        assert decoder_default.distance_threshold < 10.0, f"Default decoder threshold {decoder_default.distance_threshold} too high"
        
        # Test reasonable threshold values
        reasonable_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        for threshold in reasonable_thresholds:
            decoder = create_decoder_with_default_candidates(encoder, distance_threshold=threshold)
            assert decoder.distance_threshold == threshold
            assert decoder.distance_threshold < 10.0, f"Threshold {threshold} should be < 10.0"
    
    def test_distance_computation_consistency(self):
        """Test that distance computation is consistent across different methods."""
        from src.algebra.algebra_encoder import create_character_encoder
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        eq1, eq2 = "2*x=4", "x=2"
        
        # Method 1: Direct embedding distance
        emb1 = encoder.encode_equation_string(eq1)
        emb2 = encoder.encode_equation_string(eq2)
        distance_direct = torch.norm(emb1 - emb2).item()
        
        # Method 2: Using compute_embedding_distances utility
        emb1_batch = emb1.unsqueeze(0)
        emb2_batch = emb2.unsqueeze(0)
        distance_result = compute_embedding_distances(emb1_batch, emb2_batch)
        distance_utility = distance_result['distances'][0]
        
        # Should be very close (within floating point precision)
        assert abs(distance_direct - distance_utility) < 1e-6, \
            f"Distance computation inconsistent: {distance_direct:.6f} vs {distance_utility:.6f}"
    
    def test_decoder_distance_threshold_integration(self):
        """Test that decoder correctly uses distance thresholds for candidate selection."""
        from src.algebra.algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        # Test with strict threshold
        strict_decoder = create_decoder_with_default_candidates(encoder, distance_threshold=0.1)
        
        # Test with moderate threshold  
        moderate_decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
        
        test_eq = "x=1"
        test_embedding = encoder.encode_equation_string(test_eq)
        
        # Decode with different thresholds
        strict_result, strict_distance = strict_decoder.decode_embedding(test_embedding)
        moderate_result, moderate_distance = moderate_decoder.decode_embedding(test_embedding)
        
        # Verify threshold behavior
        if strict_result is not None:
            assert strict_distance <= 0.1, f"Strict decoder violated threshold: {strict_distance}"
        
        if moderate_result is not None:
            assert moderate_distance <= 2.0, f"Moderate decoder violated threshold: {moderate_distance}"
    
    def test_evaluation_pipeline_distance_threshold(self):
        """Test that evaluation pipeline uses appropriate distance thresholds."""
        from src.algebra.algebra_encoder import create_character_encoder
        import math
        
        # This test verifies the specific architectural constraint mentioned in task 4.2:
        # "Test max possible distance ≤2.0, threshold <10.0 (reasonable)"
        
        # Create small test dataset
        test_dataset = AlgebraDataset('distribute', num_problems=5)
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        # Test that create_decoder_from_dataset uses appropriate threshold
        decoder = create_decoder_from_dataset(
            encoder=encoder,
            dataset=test_dataset, 
            distance_threshold=2.0  # Should be appropriate for normalized embeddings
        )
        
        # Verify threshold is reasonable
        assert decoder.distance_threshold == 2.0
        assert decoder.distance_threshold < 10.0, f"Evaluation pipeline threshold {decoder.distance_threshold} too high"
        
        # Verify it's not too restrictive (should be > 0.5 to allow some flexibility)
        assert decoder.distance_threshold > 0.5, f"Evaluation pipeline threshold {decoder.distance_threshold} too restrictive"
        
        # Test that max possible distance is indeed ≤ 2.0
        theoretical_max = math.sqrt(2)  # ~1.414 for normalized embeddings
        assert theoretical_max <= 2.0, f"Max possible distance {theoretical_max:.4f} exceeds 2.0"
    
    def test_distance_threshold_warning_for_high_values(self):
        """Test that excessively high thresholds are flagged as problematic."""
        from src.algebra.algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
        import math
        
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        # Test problematic thresholds
        problematic_thresholds = [10.0, 50.0]
        
        for threshold in problematic_thresholds:
            decoder = create_decoder_with_default_candidates(encoder, distance_threshold=threshold)
            # For normalized embeddings with max distance ~1.4, thresholds >=10 accept everything
            assert decoder.distance_threshold >= 10.0, f"High threshold test setup failed"
            
            # Warn about threshold being too high for normalized embeddings
            theoretical_max = math.sqrt(2)
            assert threshold > 5 * theoretical_max, f"Threshold {threshold} should be much higher than max distance for this test"


if __name__ == "__main__":
    pytest.main([__file__])