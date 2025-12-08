#!/usr/bin/env python3
"""
Unit tests for algebra constraint energy functions.

Tests cover:
- PositivityEnergy constraint behavior and interface
- IntegernessEnergy constraint behavior and interface  
- ConstraintComposition functionality
- extract_solution_value helper function
- Gradient computation and differentiability
- Integration with existing EBM infrastructure
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

# Import constraint classes to test
from src.algebra.algebra_constraints import (
    PositivityEnergy,
    IntegernessEnergy,
    ConstraintComposition,
    ConstraintDiffusionWrapper,
    extract_solution_value,
    _extract_numerical_value_from_equation
)

# Import related classes for integration testing
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper


class TestExtractSolutionValue:
    """Test the extract_solution_value helper function."""
    
    def test_extract_numerical_value_from_equation(self):
        """Test the text parsing helper function."""
        test_cases = [
            ("x = 3.5", 3.5),
            ("x=7", 7.0),
            ("x = -2.1", -2.1),
            ("3.5 = x", 3.5),
            ("-1.5 = x", -1.5),
            ("x = +4.2", 4.2),
            ("5", 5.0),  # Just a number
            ("no number here", 0.0),  # Default fallback
            ("", 0.0),  # Empty string
        ]
        
        for equation_text, expected_value in test_cases:
            result = _extract_numerical_value_from_equation(equation_text)
            assert abs(result - expected_value) < 1e-6, f"Failed for '{equation_text}': got {result}, expected {expected_value}"
    
    def test_extract_solution_value_heuristic_mode(self):
        """Test extract_solution_value without decoder (heuristic mode)."""
        batch_size = 4
        embed_dim = 128
        
        # Create test embeddings with different magnitudes
        embeddings = torch.randn(batch_size, embed_dim)
        embeddings[0] *= 0.1  # Small magnitude
        embeddings[1] *= 1.0  # Medium magnitude
        embeddings[2] *= 2.0  # Large magnitude
        embeddings[3] *= 5.0  # Very large magnitude
        
        # Extract solution values
        solution_values = extract_solution_value(embeddings, decoder=None)
        
        # Verify output properties
        assert solution_values.shape == (batch_size,)
        assert solution_values.dtype == torch.float32
        assert torch.all(torch.abs(solution_values) <= 10.0)  # Should be bounded by tanh
        
        # Verify monotonicity (larger embeddings -> larger solution values in magnitude)
        embedding_magnitudes = torch.norm(embeddings, dim=-1)
        assert solution_values[0] < solution_values[1] < solution_values[2] < solution_values[3]
    
    def test_extract_solution_value_with_mock_decoder(self):
        """Test extract_solution_value with mocked decoder."""
        batch_size = 3
        embed_dim = 128
        embeddings = torch.randn(batch_size, embed_dim)
        
        # Mock decoder that returns known equations
        mock_decoder = Mock()
        mock_decoder.decode_batch.return_value = ["x = 2.5", "x = -1.0", "x = 0.0"]
        
        solution_values = extract_solution_value(embeddings, decoder=mock_decoder)
        
        expected_values = torch.tensor([2.5, -1.0, 0.0], dtype=torch.float32, device=embeddings.device)
        torch.testing.assert_close(solution_values, expected_values, rtol=1e-5, atol=1e-6)
    
    def test_extract_solution_value_decoder_fallback(self):
        """Test fallback to heuristic mode when decoder fails."""
        batch_size = 2
        embed_dim = 128
        embeddings = torch.randn(batch_size, embed_dim)
        
        # Mock decoder that raises exception
        mock_decoder = Mock()
        mock_decoder.decode_batch.side_effect = Exception("Decoder error")
        
        # Should fall back to heuristic mode without crashing
        solution_values = extract_solution_value(embeddings, decoder=mock_decoder)
        
        assert solution_values.shape == (batch_size,)
        assert torch.all(torch.isfinite(solution_values))


class TestPositivityEnergy:
    """Test PositivityEnergy constraint class."""
    
    def test_initialization(self):
        """Test PositivityEnergy initialization and validation."""
        # Valid initialization
        pos_energy = PositivityEnergy(beta=0.5, inp_dim=64, out_dim=64)
        assert pos_energy.beta == 0.5
        assert pos_energy.inp_dim == 64
        assert pos_energy.out_dim == 64
        assert pos_energy.constraint_type == "positivity"
        assert "positivity_constraint" in pos_energy.rule_name
        
        # Test beta validation
        with pytest.raises(ValueError, match="beta must be in range"):
            PositivityEnergy(beta=0.05)  # Too small
        
        with pytest.raises(ValueError, match="beta must be in range"):
            PositivityEnergy(beta=1.5)  # Too large
    
    def test_forward_basic_functionality(self):
        """Test PositivityEnergy forward pass with known solution values."""
        pos_energy = PositivityEnergy(beta=0.5)
        batch_size = 4
        
        # Create dummy input/output embeddings
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Mock extract_solution_value to return known values
        mock_values = torch.tensor([2.0, -1.0, 0.0, -3.0])  # Mix of positive, negative, zero
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = pos_energy(inp, out, t)
        
        # Verify output shape
        assert energy.shape == (batch_size, 1)
        
        # Verify energy values: E = beta * max(0, -x)^2
        expected_energies = torch.tensor([
            0.5 * max(0, -2.0)**2,   # Positive -> 0 energy
            0.5 * max(0, -(-1.0))**2, # Negative -> 0.5 * 1^2 = 0.5
            0.5 * max(0, -0.0)**2,   # Zero -> 0 energy  
            0.5 * max(0, -(-3.0))**2  # Negative -> 0.5 * 3^2 = 4.5
        ]).unsqueeze(-1)
        
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_forward_all_positive_solutions(self):
        """Test PositivityEnergy with all positive solutions (should give zero energy)."""
        pos_energy = PositivityEnergy(beta=0.3)
        batch_size = 3
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # All positive solution values
        mock_values = torch.tensor([1.5, 3.0, 0.1])
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = pos_energy(inp, out, t)
        
        # All energies should be zero for positive values
        expected_energies = torch.zeros(batch_size, 1)
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_forward_all_negative_solutions(self):
        """Test PositivityEnergy with all negative solutions."""
        pos_energy = PositivityEnergy(beta=0.8)
        batch_size = 3
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # All negative solution values
        mock_values = torch.tensor([-1.0, -2.5, -0.5])
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = pos_energy(inp, out, t)
        
        # Compute expected energies: E = beta * max(0, -x)^2
        expected_energies = torch.tensor([
            0.8 * 1.0**2,   # -(-1.0) = 1.0, energy = 0.8
            0.8 * 2.5**2,   # -(-2.5) = 2.5, energy = 0.8 * 6.25 = 5.0
            0.8 * 0.5**2    # -(-0.5) = 0.5, energy = 0.8 * 0.25 = 0.2
        ]).unsqueeze(-1)
        
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_gradient_computation(self):
        """Test that PositivityEnergy produces valid gradients."""
        pos_energy = PositivityEnergy(beta=0.5)
        batch_size = 2
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128, requires_grad=True)
        t = torch.randint(0, 10, (batch_size,))
        
        energy = pos_energy(inp, out, t)
        
        # Compute gradients
        loss = energy.sum()
        loss.backward()
        
        # Verify gradients exist and are finite
        assert out.grad is not None
        assert torch.all(torch.isfinite(out.grad))


class TestIntegernessEnergy:
    """Test IntegernessEnergy constraint class."""
    
    def test_initialization(self):
        """Test IntegernessEnergy initialization and validation."""
        # Valid initialization
        int_energy = IntegernessEnergy(beta=0.3, inp_dim=64, out_dim=64)
        assert int_energy.beta == 0.3
        assert int_energy.inp_dim == 64
        assert int_energy.out_dim == 64
        assert int_energy.constraint_type == "integerness"
        assert "integerness_constraint" in int_energy.rule_name
        
        # Test beta validation
        with pytest.raises(ValueError, match="beta must be in range"):
            IntegernessEnergy(beta=0.01)  # Too small
        
        with pytest.raises(ValueError, match="beta must be in range"):
            IntegernessEnergy(beta=2.0)  # Too large
    
    def test_forward_basic_functionality(self):
        """Test IntegernessEnergy forward pass with known solution values."""
        int_energy = IntegernessEnergy(beta=0.4)
        batch_size = 5
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Mix of integer and non-integer values
        mock_values = torch.tensor([2.0, 2.3, -1.0, 1.7, 0.0])
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = int_energy(inp, out, t)
        
        # Verify output shape
        assert energy.shape == (batch_size, 1)
        
        # Verify energy values: E = beta * (x - round(x))^2
        rounded_values = torch.round(mock_values)  # [2.0, 2.0, -1.0, 2.0, 0.0]
        distances = torch.abs(mock_values - rounded_values)  # [0.0, 0.3, 0.0, 0.3, 0.0]
        expected_energies = (0.4 * distances**2).unsqueeze(-1)
        
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_forward_all_integer_solutions(self):
        """Test IntegernessEnergy with all integer solutions (should give zero energy)."""
        int_energy = IntegernessEnergy(beta=0.6)
        batch_size = 4
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # All integer solution values
        mock_values = torch.tensor([0.0, 1.0, -2.0, 5.0])
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = int_energy(inp, out, t)
        
        # All energies should be zero for integer values
        expected_energies = torch.zeros(batch_size, 1)
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_forward_fractional_solutions(self):
        """Test IntegernessEnergy with fractional solutions."""
        int_energy = IntegernessEnergy(beta=0.5)
        batch_size = 3
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Fractional solution values
        mock_values = torch.tensor([1.5, -0.3, 2.8])
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            energy = int_energy(inp, out, t)
        
        # Expected energies: E = beta * (x - round(x))^2
        # round(1.5) = 2.0, distance = 0.5, energy = 0.5 * 0.25 = 0.125
        # round(-0.3) = 0.0, distance = 0.3, energy = 0.5 * 0.09 = 0.045
        # round(2.8) = 3.0, distance = 0.2, energy = 0.5 * 0.04 = 0.02
        expected_energies = torch.tensor([
            0.5 * (1.5 - 2.0)**2,  # 0.5 * 0.25 = 0.125
            0.5 * (-0.3 - 0.0)**2, # 0.5 * 0.09 = 0.045
            0.5 * (2.8 - 3.0)**2   # 0.5 * 0.04 = 0.02
        ]).unsqueeze(-1)
        
        torch.testing.assert_close(energy, expected_energies, rtol=1e-5, atol=1e-6)
    
    def test_gradient_computation(self):
        """Test that IntegernessEnergy produces valid gradients."""
        int_energy = IntegernessEnergy(beta=0.4)
        batch_size = 2
        
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128, requires_grad=True)
        t = torch.randint(0, 10, (batch_size,))
        
        energy = int_energy(inp, out, t)
        
        # Compute gradients
        loss = energy.sum()
        loss.backward()
        
        # Verify gradients exist and are finite
        assert out.grad is not None
        assert torch.all(torch.isfinite(out.grad))


class TestConstraintComposition:
    """Test ConstraintComposition utility class."""
    
    def test_initialization(self):
        """Test ConstraintComposition initialization."""
        # Create mock rule energies
        rule_energies = {
            "rule1": Mock(spec=AlgebraEBM),
            "rule2": Mock(spec=AlgebraEBM)
        }
        
        # Create constraint energies
        constraint_energies = [
            PositivityEnergy(beta=0.5),
            IntegernessEnergy(beta=0.3)
        ]
        
        # Test with default weights
        composition = ConstraintComposition(rule_energies, constraint_energies)
        assert len(composition.rule_energies) == 2
        assert len(composition.constraint_energies) == 2
        assert composition.constraint_weights == [1.0, 1.0]
        
        # Test with custom weights
        custom_weights = [0.5, 0.8]
        composition_custom = ConstraintComposition(rule_energies, constraint_energies, custom_weights)
        assert composition_custom.constraint_weights == [0.5, 0.8]
        
        # Test weight length validation
        with pytest.raises(ValueError, match="Number of constraint weights"):
            ConstraintComposition(rule_energies, constraint_energies, [0.5])  # Wrong length
    
    def test_compute_total_energy_basic(self):
        """Test basic total energy computation."""
        batch_size = 2
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Mock rule energies that return known values
        rule_energy_1 = torch.tensor([[1.0], [2.0]])
        rule_energy_2 = torch.tensor([[0.5], [1.5]])
        
        mock_rule1 = Mock()
        mock_rule1.return_value = rule_energy_1
        mock_rule2 = Mock()
        mock_rule2.return_value = rule_energy_2
        
        rule_energies = {"rule1": mock_rule1, "rule2": mock_rule2}
        
        # Create constraint energies with mocked solution extraction
        pos_energy = PositivityEnergy(beta=0.5)
        int_energy = IntegernessEnergy(beta=0.3)
        constraint_energies = [pos_energy, int_energy]
        constraint_weights = [2.0, 1.5]  # Custom weights
        
        composition = ConstraintComposition(rule_energies, constraint_energies, constraint_weights)
        
        # Mock solution values for constraints
        mock_values = torch.tensor([-1.0, 2.3])  # Negative and non-integer
        
        with patch('src.algebra.algebra_constraints.extract_solution_value', return_value=mock_values):
            total_energy, energy_breakdown = composition.compute_total_energy(inp, out, t)
        
        # Verify total energy shape
        assert total_energy.shape == (batch_size, 1)
        
        # Verify energy breakdown structure
        expected_keys = [
            "rule_rule1", "rule_rule2", 
            "constraint_positivity", "constraint_positivity_weighted",
            "constraint_integerness", "constraint_integerness_weighted"
        ]
        for key in expected_keys:
            assert key in energy_breakdown
            assert energy_breakdown[key].shape == (batch_size, 1)
        
        # Verify mock calls
        mock_rule1.assert_called_once_with(inp, out, t)
        mock_rule2.assert_called_once_with(inp, out, t)
    
    def test_compute_total_energy_active_rules_subset(self):
        """Test computing energy with only a subset of active rules."""
        batch_size = 1
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        mock_rule1 = Mock(return_value=torch.tensor([[1.0]]))
        mock_rule2 = Mock(return_value=torch.tensor([[2.0]]))
        mock_rule3 = Mock(return_value=torch.tensor([[3.0]]))
        
        rule_energies = {"rule1": mock_rule1, "rule2": mock_rule2, "rule3": mock_rule3}
        constraint_energies = []  # No constraints for simplicity
        
        composition = ConstraintComposition(rule_energies, constraint_energies)
        
        # Only activate rule1 and rule3
        total_energy, breakdown = composition.compute_total_energy(
            inp, out, t, active_rules=["rule1", "rule3"]
        )
        
        # Only rule1 and rule3 should be called
        mock_rule1.assert_called_once()
        mock_rule3.assert_called_once()
        mock_rule2.assert_not_called()
        
        # Breakdown should only contain active rules
        assert "rule_rule1" in breakdown
        assert "rule_rule3" in breakdown
        assert "rule_rule2" not in breakdown
    
    def test_get_constraint_summary(self):
        """Test constraint summary generation."""
        constraint_energies = [
            PositivityEnergy(beta=0.5),
            IntegernessEnergy(beta=0.3)
        ]
        constraint_weights = [1.2, 0.8]
        
        composition = ConstraintComposition({}, constraint_energies, constraint_weights)
        summary = composition.get_constraint_summary()
        
        assert "constraint_0" in summary
        assert "constraint_1" in summary
        
        # Check first constraint summary
        constraint_0 = summary["constraint_0"]
        assert constraint_0["type"] == "positivity"
        assert constraint_0["weight"] == 1.2
        assert constraint_0["beta"] == 0.5
        assert "positivity_constraint" in constraint_0["rule_name"]
        
        # Check second constraint summary
        constraint_1 = summary["constraint_1"]
        assert constraint_1["type"] == "integerness"
        assert constraint_1["weight"] == 0.8
        assert constraint_1["beta"] == 0.3
        assert "integerness_constraint" in constraint_1["rule_name"]


class TestConstraintDiffusionWrapper:
    """Test ConstraintDiffusionWrapper class."""
    
    def test_initialization(self):
        """Test ConstraintDiffusionWrapper initialization."""
        pos_energy = PositivityEnergy(beta=0.4)
        wrapper = ConstraintDiffusionWrapper(pos_energy)
        
        assert wrapper.constraint_energy is pos_energy
        assert wrapper.inp_dim == pos_energy.inp_dim
        assert wrapper.out_dim == pos_energy.out_dim
        assert wrapper.rule_name == pos_energy.rule_name
    
    def test_forward_return_energy(self):
        """Test forward pass returning energy only."""
        pos_energy = PositivityEnergy(beta=0.3)
        wrapper = ConstraintDiffusionWrapper(pos_energy)
        
        batch_size = 2
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        energy = wrapper(inp, out, t, return_energy=True)
        
        assert energy.shape == (batch_size, 1)
        assert torch.all(energy >= 0)  # Constraint energies should be non-negative
    
    def test_forward_return_gradient(self):
        """Test forward pass returning gradient."""
        int_energy = IntegernessEnergy(beta=0.5)
        wrapper = ConstraintDiffusionWrapper(int_energy)
        
        batch_size = 2
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        grad = wrapper(inp, out, t, return_energy=False)
        
        assert grad.shape == (batch_size, 128)
        assert torch.all(torch.isfinite(grad))
    
    def test_forward_return_both(self):
        """Test forward pass returning both energy and gradient."""
        pos_energy = PositivityEnergy(beta=0.6)
        wrapper = ConstraintDiffusionWrapper(pos_energy)
        
        batch_size = 1
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        energy, grad = wrapper(inp, out, t, return_both=True)
        
        assert energy.shape == (batch_size, 1)
        assert grad.shape == (batch_size, 128)
        assert torch.all(energy >= 0)
        assert torch.all(torch.isfinite(grad))
    
    def test_gradient_computation_finite(self):
        """Test that gradient computation produces finite results."""
        int_energy = IntegernessEnergy(beta=0.4)
        wrapper = ConstraintDiffusionWrapper(int_energy)
        
        batch_size = 3
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Test multiple calls to ensure stability
        for _ in range(5):
            grad = wrapper(inp, out, t)
            assert torch.all(torch.isfinite(grad))
            assert grad.requires_grad  # Should support higher-order gradients


class TestIntegrationWithExistingEBM:
    """Test integration with existing AlgebraEBM infrastructure."""
    
    def test_interface_compatibility(self):
        """Test that constraint energies have compatible interfaces with AlgebraEBM."""
        # Create instances
        algebra_ebm = AlgebraEBM(inp_dim=128, out_dim=128)
        pos_energy = PositivityEnergy(beta=0.5)
        int_energy = IntegernessEnergy(beta=0.3)
        
        batch_size = 2
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # All should accept the same inputs and return similar outputs
        ebm_energy = algebra_ebm(inp, out, t)
        pos_constraint_energy = pos_energy(inp, out, t)
        int_constraint_energy = int_energy(inp, out, t)
        
        # Check output shapes are compatible
        assert ebm_energy.shape == pos_constraint_energy.shape == int_constraint_energy.shape == (batch_size, 1)
        
        # All should be differentiable
        assert ebm_energy.requires_grad
        # Constraint energies will depend on extract_solution_value behavior
        assert torch.all(torch.isfinite(pos_constraint_energy))
        assert torch.all(torch.isfinite(int_constraint_energy))
    
    def test_diffusion_wrapper_compatibility(self):
        """Test that ConstraintDiffusionWrapper has compatible interface with AlgebraDiffusionWrapper."""
        # Create wrappers
        algebra_ebm = AlgebraEBM(inp_dim=64, out_dim=64)
        algebra_wrapper = AlgebraDiffusionWrapper(algebra_ebm)
        
        pos_energy = PositivityEnergy(beta=0.4, inp_dim=64, out_dim=64)
        constraint_wrapper = ConstraintDiffusionWrapper(pos_energy)
        
        batch_size = 2
        inp = torch.randn(batch_size, 64)
        out = torch.randn(batch_size, 64)
        t = torch.randint(0, 10, (batch_size,))
        
        # Both should support the same interface
        for return_energy in [True, False]:
            for return_both in [True, False]:
                if return_energy and return_both:
                    continue  # Invalid combination
                
                ebm_result = algebra_wrapper(inp, out, t, return_energy=return_energy, return_both=return_both)
                constraint_result = constraint_wrapper(inp, out, t, return_energy=return_energy, return_both=return_both)
                
                if return_both:
                    assert isinstance(ebm_result, tuple) and len(ebm_result) == 2
                    assert isinstance(constraint_result, tuple) and len(constraint_result) == 2
                    assert ebm_result[0].shape == constraint_result[0].shape  # Energy shapes
                    assert ebm_result[1].shape == constraint_result[1].shape  # Gradient shapes
                elif return_energy:
                    assert ebm_result.shape == constraint_result.shape  # Energy shapes
                else:
                    assert ebm_result.shape == constraint_result.shape  # Gradient shapes


# Test edge cases and error handling
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_beta_boundary(self):
        """Test behavior at beta boundaries."""
        # Just above minimum
        pos_energy = PositivityEnergy(beta=0.1)
        int_energy = IntegernessEnergy(beta=0.1)
        
        assert pos_energy.beta == 0.1
        assert int_energy.beta == 0.1
        
        # Just at maximum
        pos_energy_max = PositivityEnergy(beta=1.0)
        int_energy_max = IntegernessEnergy(beta=1.0)
        
        assert pos_energy_max.beta == 1.0
        assert int_energy_max.beta == 1.0
    
    def test_empty_constraint_composition(self):
        """Test ConstraintComposition with no constraints."""
        rule_energies = {"rule1": Mock(return_value=torch.tensor([[1.0]]))}
        constraint_energies = []
        
        composition = ConstraintComposition(rule_energies, constraint_energies)
        assert len(composition.constraint_energies) == 0
        assert len(composition.constraint_weights) == 0
        
        # Should still work for rule energies only
        inp = torch.randn(1, 128)
        out = torch.randn(1, 128)
        t = torch.randint(0, 10, (1,))
        
        total_energy, breakdown = composition.compute_total_energy(inp, out, t)
        assert total_energy.shape == (1, 1)
        assert "rule_rule1" in breakdown
    
    def test_dimension_mismatch_error_handling(self):
        """Test error handling for dimension mismatches."""
        pos_energy = PositivityEnergy(inp_dim=64, out_dim=32)
        
        # Correct dimensions should work
        inp = torch.randn(2, 64)
        out = torch.randn(2, 32)
        t = torch.randint(0, 10, (2,))
        
        energy = pos_energy(inp, out, t)
        assert energy.shape == (2, 1)
        
        # Wrong output dimension should raise error
        wrong_out = torch.randn(2, 64)  # Should be 32
        with pytest.raises(AssertionError, match="Expected out_dim"):
            pos_energy(inp, wrong_out, t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])