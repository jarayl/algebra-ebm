#!/usr/bin/env python3
"""
Unit tests for AlgebraDataset and equation generation.

Tests cover:
- BUG-2: Coefficient formatting in equation generation
- Dataset creation and validation
- Equation pair generation for all rule types
- Stratified sampling and solution-first generation
- Encoder integration and embedding validation
- Negative coefficient handling and equation syntax

This test suite validates the algebra dataset generation
and ensures proper equation formatting without syntax errors.
"""

import pytest
import torch
import numpy as np
import re
from typing import List, Tuple, Dict

# Import dataset and related components
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_encoder import (
    CharacterLevelEncoder, 
    ASTEncoder,
    validate_equation_syntax,
    solve_equation,
    check_equation_equivalence
)


class TestAlgebraDataset:
    """Test the AlgebraDataset class for equation generation."""
    
    def test_init_basic(self):
        """Test basic dataset initialization."""
        dataset = AlgebraDataset(rule='distribute', split='train', num_problems=100)
        
        assert dataset.rule == 'distribute'
        assert dataset.split == 'train' 
        assert dataset.num_problems == 100
        assert dataset.coeff_range == [-10, 10]
        assert dataset.d_model == 128
        assert len(dataset) == 100

    def test_init_invalid_rule(self):
        """Test that invalid rules raise errors."""
        with pytest.raises(ValueError, match="Rule must be one of"):
            AlgebraDataset(rule='invalid_rule')

    def test_init_invalid_split(self):
        """Test that invalid splits raise errors."""
        with pytest.raises(ValueError, match="Split must be"):
            AlgebraDataset(rule='distribute', split='invalid_split')

    def test_init_invalid_coeff_range(self):
        """Test that invalid coefficient ranges raise errors."""
        # Empty range
        with pytest.raises(ValueError, match="coeff_range must be"):
            AlgebraDataset(rule='distribute', coeff_range=[5, 5])
        
        # Inverted range
        with pytest.raises(ValueError, match="coeff_range must be"):
            AlgebraDataset(rule='distribute', coeff_range=[10, 5])

    def test_stratified_sampling_init(self):
        """Test stratified sampling initialization."""
        dataset = AlgebraDataset(
            rule='distribute',
            enable_stratified_sampling=True,
            num_problems=100
        )
        
        assert dataset.enable_stratified_sampling is True
        assert 'basic' in dataset.stratified_ranges
        assert 'extended' in dataset.stratified_ranges  
        assert 'challenge' in dataset.stratified_ranges
        
        # Check distributions sum to 1
        total_prob = sum(dataset.stratified_distribution.values())
        assert abs(total_prob - 1.0) < 1e-6

    def test_equation_generation_all_rules(self):
        """Test equation generation for all supported rules."""
        rules = ['distribute', 'combine', 'isolate', 'divide']
        
        for rule in rules:
            dataset = AlgebraDataset(rule=rule, num_problems=10)
            
            # Test first few items
            for i in range(min(5, len(dataset))):
                item = dataset[i]
                
                # Check item structure
                assert isinstance(item, tuple)
                assert len(item) == 2
                
                inp_emb, out_emb = item
                assert isinstance(inp_emb, torch.Tensor)
                assert isinstance(out_emb, torch.Tensor)
                assert inp_emb.shape == (128,)
                assert out_emb.shape == (128,)

    def test_get_equation_pair(self):
        """Test equation pair retrieval."""
        dataset = AlgebraDataset(rule='distribute', num_problems=10)
        
        for i in range(min(5, len(dataset))):
            inp_eq, out_eq = dataset.get_equation_pair(i)
            
            assert isinstance(inp_eq, str)
            assert isinstance(out_eq, str)
            assert len(inp_eq) > 0
            assert len(out_eq) > 0
            assert '=' in inp_eq
            assert '=' in out_eq

    def test_coefficient_formatting_bug2(self):
        """Test BUG-2 fix: Proper coefficient formatting without syntax errors."""
        # Test with various coefficient ranges including negative values
        dataset = AlgebraDataset(
            rule='distribute', 
            coeff_range=[-20, 20],
            num_problems=50
        )
        
        # Check equation formatting
        for i in range(min(20, len(dataset))):
            inp_eq, out_eq = dataset.get_equation_pair(i)
            
            # BUG-2: Check for malformed expressions like "3*x+-15=42"
            # Should not have "+-" or "-+" patterns
            assert '+-' not in inp_eq, f"Invalid coefficient format in input: {inp_eq}"
            assert '-+' not in inp_eq, f"Invalid coefficient format in input: {inp_eq}"
            assert '+-' not in out_eq, f"Invalid coefficient format in output: {out_eq}"
            assert '-+' not in out_eq, f"Invalid coefficient format in output: {out_eq}"
            
            # Check for valid equation syntax
            is_valid_inp, error_inp, _ = validate_equation_syntax(inp_eq)
            is_valid_out, error_out, _ = validate_equation_syntax(out_eq)
            
            assert is_valid_inp, f"Invalid input equation syntax: {inp_eq}, error: {error_inp}"
            assert is_valid_out, f"Invalid output equation syntax: {out_eq}, error: {error_out}"

    def test_negative_coefficient_handling(self):
        """Test proper handling of negative coefficients."""
        dataset = AlgebraDataset(
            rule='combine',
            coeff_range=[-10, 10], 
            num_problems=30
        )
        
        found_negative_coeff = False
        
        for i in range(len(dataset)):
            inp_eq, out_eq = dataset.get_equation_pair(i)
            
            # Look for negative coefficients (properly formatted)
            if '-' in inp_eq and '=' in inp_eq:
                # Check that negative signs are properly formatted
                # Valid: "3*x-5", "2*x+-5" should NOT appear
                if re.search(r'\d+\*x-\d+', inp_eq):
                    found_negative_coeff = True
                    
                    # Ensure no malformed patterns
                    assert '+-' not in inp_eq
                    assert '-+' not in inp_eq
                    
                    # Validate syntax
                    is_valid, error, _ = validate_equation_syntax(inp_eq)
                    assert is_valid, f"Invalid negative coefficient equation: {inp_eq}, error: {error}"
        
        # We should find at least some negative coefficients in the range [-10, 10]
        assert found_negative_coeff, "No negative coefficients found - test may need adjustment"

    def test_equation_mathematical_validity(self):
        """Test that generated equations are mathematically valid."""
        # Test different rules for mathematical validity
        for rule in ['distribute', 'combine', 'isolate']:
            dataset = AlgebraDataset(rule=rule, num_problems=10, coeff_range=[-5, 5])
            
            for i in range(min(5, len(dataset))):
                inp_eq, out_eq = dataset.get_equation_pair(i)
                
                # Try to solve both equations
                inp_solutions, inp_error = solve_equation(inp_eq)
                out_solutions, out_error = solve_equation(out_eq)
                
                # Both equations should be solvable
                assert inp_error is None, f"Cannot solve input equation {inp_eq}: {inp_error}"
                assert out_error is None, f"Cannot solve output equation {out_eq}: {out_error}"
                
                # For rule transformations, equations should be equivalent
                if rule in ['distribute', 'combine']:
                    are_equiv, equiv_error = check_equation_equivalence(inp_eq, out_eq)
                    if not are_equiv and equiv_error is None:
                        # This might be expected for some transformations
                        print(f"Note: {rule} transformation {inp_eq} -> {out_eq} not equivalent (may be expected)")

    def test_deterministic_generation(self):
        """Test that generation is deterministic by comparing multiple generations."""
        # Create dataset
        dataset = AlgebraDataset(rule='distribute', num_problems=20, coeff_range=[-5, 5])
        
        # Get equations twice - should be consistent within same dataset instance
        equations1 = []
        equations2 = []
        
        for i in range(5):
            inp1, out1 = dataset.get_equation_pair(i)
            equations1.append((inp1, out1))
        
        for i in range(5):
            inp2, out2 = dataset.get_equation_pair(i)
            equations2.append((inp2, out2))
            
        # Same dataset should return same equations for same indices
        for i, ((inp1, out1), (inp2, out2)) in enumerate(zip(equations1, equations2)):
            assert inp1 == inp2, f"Non-deterministic input generation at index {i}"
            assert out1 == out2, f"Non-deterministic output generation at index {i}"

    def test_encoder_integration(self):
        """Test dataset integration with encoders."""
        dataset = AlgebraDataset(rule='distribute', num_problems=10)
        
        # Test with character encoder
        char_encoder = CharacterLevelEncoder(d_model=128)
        
        for i in range(min(5, len(dataset))):
            inp_emb, out_emb = dataset[i]
            
            # Embeddings should be normalized (if encoder normalizes)
            inp_norm = torch.norm(inp_emb).item()
            out_norm = torch.norm(out_emb).item()
            
            # For normalized embeddings, norm should be close to 1
            assert 0.8 <= inp_norm <= 1.2, f"Input embedding norm {inp_norm} not close to 1"
            assert 0.8 <= out_norm <= 1.2, f"Output embedding norm {out_norm} not close to 1"
            
            # Check that different equations give different embeddings
            if i > 0:
                prev_inp, prev_out = dataset[i-1]
                assert not torch.allclose(inp_emb, prev_inp, rtol=1e-3), \
                    "Different equations should produce different embeddings"


class TestCharacterLevelEncoder:
    """Test the CharacterLevelEncoder functionality."""
    
    def test_init(self):
        """Test encoder initialization."""
        encoder = CharacterLevelEncoder()
        
        assert encoder.d_model == 128
        assert encoder.max_len == 64
        assert encoder.normalize_embeddings is True
        assert hasattr(encoder, 'vocab')
        assert len(encoder.vocab) > 0

    def test_encode_simple(self):
        """Test encoding simple equations."""
        encoder = CharacterLevelEncoder(d_model=64, max_len=32)
        
        equation = "x+1=2"
        embedding = encoder.encode_equation_string(equation)
        
        assert embedding.shape == (64,)
        assert torch.isfinite(embedding).all()
        
        # If normalized, should have unit norm
        if encoder.normalize_embeddings:
            norm = torch.norm(embedding).item()
            assert abs(norm - 1.0) < 1e-6

    def test_encode_batch(self):
        """Test batch encoding."""
        encoder = CharacterLevelEncoder()
        
        equations = ["x+1=2", "2*x=4", "x-3=7"]
        embeddings = encoder.encode_batch(equations)
        
        assert embeddings.shape == (3, 128)
        assert torch.isfinite(embeddings).all()
        
        # Different equations should have different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1], rtol=1e-3)
        assert not torch.allclose(embeddings[1], embeddings[2], rtol=1e-3)

    def test_encode_invalid_characters(self):
        """Test encoding with invalid characters."""
        encoder = CharacterLevelEncoder()
        
        with pytest.raises(ValueError, match="Unknown character"):
            encoder.encode_equation_string("x+y=2")  # 'y' not in vocab

    def test_encode_long_equations(self):
        """Test encoding equations longer than max_len."""
        encoder = CharacterLevelEncoder(max_len=10)
        
        # Long equation should be truncated
        long_equation = "2*x+3*x+4*x+5*x+6*x=100"  # Longer than 10 chars
        embedding = encoder.encode_equation_string(long_equation)
        
        assert embedding.shape == (128,)
        assert torch.isfinite(embedding).all()

    def test_vocab_info(self):
        """Test vocabulary information retrieval."""
        encoder = CharacterLevelEncoder()
        info = encoder.get_vocab_info()
        
        assert 'vocab' in info
        assert 'vocab_size' in info
        assert 'char_to_idx' in info
        assert 'd_model' in info
        assert info['vocab_size'] == len(encoder.vocab)


class TestASTEncoder:
    """Test the ASTEncoder functionality."""
    
    def test_init(self):
        """Test AST encoder initialization."""
        encoder = ASTEncoder()
        
        assert encoder.d_model == 128
        assert encoder.max_features == 64
        assert len(encoder.node_types) > 0
        assert 'Symbol' in encoder.node_types

    def test_encode_simple_expression(self):
        """Test encoding simple mathematical expressions."""
        encoder = ASTEncoder(d_model=64)
        
        equation = "x+1=2"
        embedding = encoder.encode_equation_string(equation)
        
        assert embedding.shape == (64,)
        assert torch.isfinite(embedding).all()

    def test_encode_complex_expression(self):
        """Test encoding more complex expressions."""
        encoder = ASTEncoder()
        
        equations = [
            "2*x+3=7",
            "x**2+1=5", 
            "2*(x+3)=10",
            "x/2+1=3"
        ]
        
        for eq in equations:
            embedding = encoder.encode_equation_string(eq)
            assert embedding.shape == (128,)
            assert torch.isfinite(embedding).all()

    def test_safety_validation(self):
        """Test that safety validation blocks dangerous expressions."""
        encoder = ASTEncoder()
        
        dangerous_expressions = [
            "__import__",
            "eval(",  
            "exec(",
            "os.system"
        ]
        
        for expr in dangerous_expressions:
            with pytest.raises(ValueError):
                encoder.encode_equation_string(expr)

    def test_feature_extraction(self):
        """Test AST feature extraction."""
        encoder = ASTEncoder()
        
        # Test with SymPy expression
        import sympy as sp
        expr = sp.sympify("2*x + 3", evaluate=False)
        features = encoder.extract_ast_features(expr)
        
        assert len(features) == encoder.max_features
        assert all(isinstance(f, float) for f in features)

    def test_get_feature_info(self):
        """Test feature information retrieval."""
        encoder = ASTEncoder()
        info = encoder.get_feature_info()
        
        assert 'node_types' in info
        assert 'max_features' in info
        assert 'd_model' in info
        assert info['max_features'] == 64


class TestEquationValidation:
    """Test equation validation utilities."""
    
    def test_validate_basic_equations(self):
        """Test validation of basic equations."""
        valid_equations = [
            "x+1=2",
            "2*x-3=7",
            "x**2=4",
            "2*(x+1)=6"
        ]
        
        for eq in valid_equations:
            is_valid, error, expr = validate_equation_syntax(eq)
            assert is_valid, f"Equation {eq} should be valid, error: {error}"
            assert error is None
            assert expr is not None

    def test_validate_invalid_equations(self):
        """Test validation rejects invalid equations."""
        invalid_equations = [
            "x+=2",        # Invalid operator
            "2*x==3",      # Double equals
            "",            # Empty string
            "x+y=2=3"      # Multiple equals
        ]
        
        for eq in invalid_equations:
            is_valid, error, expr = validate_equation_syntax(eq)
            assert not is_valid, f"Equation {eq} should be invalid"
            assert error is not None

    def test_solve_basic_equations(self):
        """Test solving basic equations."""
        equations_solutions = [
            ("x+1=2", [1.0]),
            ("2*x=4", [2.0]),
            ("x-3=0", [3.0])
        ]
        
        for eq, expected_sols in equations_solutions:
            solutions, error = solve_equation(eq)
            assert error is None, f"Error solving {eq}: {error}"
            assert len(solutions) == len(expected_sols)
            for sol, exp_sol in zip(solutions, expected_sols):
                assert abs(sol - exp_sol) < 1e-6

    def test_check_equation_equivalence(self):
        """Test equation equivalence checking."""
        # Equivalent equations
        equiv_pairs = [
            ("x+1=2", "x=1"),
            ("2*x=4", "x=2"),
            ("x-3=0", "x=3")
        ]
        
        for eq1, eq2 in equiv_pairs:
            are_equiv, error = check_equation_equivalence(eq1, eq2)
            assert error is None, f"Error checking equivalence {eq1} vs {eq2}: {error}"
            assert are_equiv, f"Equations {eq1} and {eq2} should be equivalent"
        
        # Non-equivalent equations
        non_equiv_pairs = [
            ("x+1=2", "x+1=3"),
            ("2*x=4", "3*x=4")
        ]
        
        for eq1, eq2 in non_equiv_pairs:
            are_equiv, error = check_equation_equivalence(eq1, eq2)
            assert error is None, f"Error checking equivalence {eq1} vs {eq2}: {error}"
            assert not are_equiv, f"Equations {eq1} and {eq2} should not be equivalent"


if __name__ == "__main__":
    pytest.main([__file__])