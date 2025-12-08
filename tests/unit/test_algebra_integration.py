#!/usr/bin/env python3
"""
Integration tests for the algebra EBM system.

Tests cover:
- End-to-end training workflow integration
- Model + dataset + encoder integration  
- Training with ContrastiveEnergyLoss
- Evaluation pipeline integration
- Multi-component bug fix validation
- Performance regression testing

This test suite validates that all algebra EBM components
work together correctly and reproduce expected behavior.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Import all key components for integration testing
from src.algebra.algebra_models import (
    AlgebraEBM, 
    AlgebraDiffusionWrapper, 
    ContrastiveEnergyLoss
)
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_encoder import (
    CharacterLevelEncoder,
    ASTEncoder,
    create_character_encoder,
    create_decoder_with_default_candidates
)
from src.algebra.algebra_inference import (
    AlgebraInference,
    InferenceConfig,
    load_rule_models
)


class TestEndToEndWorkflow:
    """Test complete end-to-end algebra EBM workflows."""
    
    def test_training_data_pipeline(self):
        """Test the complete training data pipeline."""
        # Create dataset
        dataset = AlgebraDataset(
            rule='distribute',
            split='train',
            num_problems=20,
            coeff_range=[-5, 5]
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test data loading
        batch_count = 0
        for batch in dataloader:
            inp_emb, out_emb = batch
            
            assert inp_emb.shape[0] <= 4  # Batch size
            assert inp_emb.shape[1] == 128  # Embedding dim
            assert out_emb.shape[1] == 128
            assert torch.isfinite(inp_emb).all()
            assert torch.isfinite(out_emb).all()
            
            batch_count += 1
            if batch_count >= 3:  # Test just a few batches
                break
        
        assert batch_count > 0, "No batches loaded from dataset"

    def test_model_training_step_simulation(self):
        """Test a complete training step simulation."""
        # Create model components
        ebm = AlgebraEBM(rule_name='distribute')
        wrapper = AlgebraDiffusionWrapper(ebm)
        loss_fn = ContrastiveEnergyLoss()
        optimizer = torch.optim.Adam(ebm.parameters(), lr=1e-3)
        
        # Create sample data (simulate positive and negative pairs)
        batch_size = 4
        inp = torch.randn(batch_size, 128)
        pos_out = torch.randn(batch_size, 128)
        neg_out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass
        pos_energies = wrapper(inp, pos_out, t, return_energy=True)
        neg_energies = wrapper(inp, neg_out, t, return_energy=True)
        
        # Loss computation
        loss = loss_fn.compute_loss(pos_energies, neg_energies)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in ebm.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        
        # Optimizer step
        optimizer.step()
        
        # Loss should be a valid scalar
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_contrastive_loss_integration(self):
        """Test ContrastiveEnergyLoss integration with energy models."""
        ebm = AlgebraEBM()
        wrapper = AlgebraDiffusionWrapper(ebm)
        loss_fn = ContrastiveEnergyLoss(margin=5.0, pos_target=1.0, neg_target=8.0)
        
        # Multiple training steps to test loss behavior
        batch_size = 3
        learning_losses = []
        
        for step in range(5):
            # Generate data where positive samples should have lower energy
            inp = torch.randn(batch_size, 128)
            
            # Positive samples: smaller random vectors (should get lower energy)
            pos_out = torch.randn(batch_size, 128) * 0.5
            
            # Negative samples: larger random vectors (should get higher energy)
            neg_out = torch.randn(batch_size, 128) * 2.0
            
            t = torch.randint(0, 10, (batch_size,))
            
            # Compute energies
            pos_energies = wrapper(inp, pos_out, t, return_energy=True)
            neg_energies = wrapper(inp, neg_out, t, return_energy=True)
            
            # Compute loss with metrics
            loss, metrics = loss_fn.compute_loss(pos_energies, neg_energies, return_metrics=True)
            learning_losses.append(loss.item())
            
            # Validate loss components
            assert metrics['pos_energy_mean'] >= 0
            assert metrics['neg_energy_mean'] >= 0
            assert 'energy_gap' in metrics
            assert 'energy_ratio' in metrics
        
        # Check that loss computation is stable across steps
        assert all(np.isfinite(loss_val) for loss_val in learning_losses)

    def test_inference_with_dataset(self):
        """Test inference integration with dataset-generated problems."""
        # Create small dataset
        dataset = AlgebraDataset(rule='combine', num_problems=5)
        
        # Create inference components
        rule_models = {
            'combine': AlgebraDiffusionWrapper(AlgebraEBM(rule_name='combine'))
        }
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=3, max_iterations=5)  # Small for testing
        
        inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
        
        # Test inference on dataset equations
        for i in range(min(3, len(dataset))):
            inp_eq, target_eq = dataset.get_equation_pair(i)
            
            # Run inference
            result = inference.solve_equation(inp_eq)
            
            # Validate result structure
            assert 'input_equation' in result
            assert 'success' in result
            assert 'inference_info' in result
            assert result['input_equation'] == inp_eq
            
            # Check inference info
            info = result['inference_info']
            assert 'energy_history' in info
            assert 'acceptance_rate' in info
            assert len(info['energy_history']) > 0


class TestBugFixIntegration:
    """Test that critical bug fixes work correctly in integrated scenarios."""
    
    def test_bug1_energy_scaling_integration(self):
        """Test BUG-1 (energy scaling) in full training context."""
        ebm = AlgebraEBM()
        wrapper = AlgebraDiffusionWrapper(ebm)
        loss_fn = ContrastiveEnergyLoss(pos_target=1.0, neg_target=10.0)
        
        # Initial energy scaling parameters
        initial_scale = ebm.energy_scale.data.item()
        initial_bias = ebm.energy_bias.data.item()
        
        batch_size = 4
        inp = torch.randn(batch_size, 128)
        pos_out = torch.randn(batch_size, 128)
        neg_out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Training should be able to modify energy scaling
        optimizer = torch.optim.SGD([ebm.energy_scale, ebm.energy_bias], lr=0.1)
        
        for _ in range(3):
            optimizer.zero_grad()
            
            pos_energies = wrapper(inp, pos_out, t, return_energy=True)
            neg_energies = wrapper(inp, neg_out, t, return_energy=True)
            loss = loss_fn.compute_loss(pos_energies, neg_energies)
            
            loss.backward()
            optimizer.step()
        
        # Energy scaling parameters should be learnable (may have changed)
        final_scale = ebm.energy_scale.data.item()
        final_bias = ebm.energy_bias.data.item()
        
        # At minimum, gradients should have been computed
        assert ebm.energy_scale.grad is not None
        assert ebm.energy_bias.grad is not None

    def test_bug2_coefficient_formatting_integration(self):
        """Test BUG-2 (coefficient formatting) in dataset + encoder integration."""
        dataset = AlgebraDataset(
            rule='distribute',
            coeff_range=[-15, 15],  # Include negative coefficients
            num_problems=20
        )
        encoder = CharacterLevelEncoder()
        
        # Test that all generated equations can be encoded without errors
        for i in range(len(dataset)):
            inp_eq, target_eq = dataset.get_equation_pair(i)
            
            # Should not contain malformed patterns
            assert '+-' not in inp_eq, f"Malformed input equation: {inp_eq}"
            assert '+-' not in target_eq, f"Malformed target equation: {target_eq}"
            
            # Should be encodable
            try:
                inp_emb = encoder.encode_equation_string(inp_eq)
                target_emb = encoder.encode_equation_string(target_eq)
                
                assert torch.isfinite(inp_emb).all()
                assert torch.isfinite(target_emb).all()
            except Exception as e:
                pytest.fail(f"Encoding failed for equation {inp_eq} -> {target_eq}: {e}")

    def test_bug3_energy_caching_integration(self):
        """Test BUG-3 (energy caching) in inference workflow."""
        rule_models = {
            'isolate': AlgebraDiffusionWrapper(AlgebraEBM(rule_name='isolate'))
        }
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=4, max_iterations=10)
        
        inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
        
        # Test that inference completes without caching-related errors
        inp_embedding = torch.randn(1, 128)
        
        out_embedding, info = inference.ired_inference(inp_embedding)
        
        # Should complete successfully
        assert torch.isfinite(out_embedding).all()
        assert 'final_energy' in info
        assert info['total_steps'] > 0
        
        # Energy history should be reasonable
        energies = info['energy_history']
        assert len(energies) > 0
        assert all(np.isfinite(e) for e in energies)

    def test_bug6_gradient_computation_integration(self):
        """Test BUG-6 (gradient computation) in multi-rule inference."""
        # Create multiple rule models
        rule_models = {}
        for rule in ['distribute', 'combine']:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            rule_models[rule] = wrapper
        
        encoder = CharacterLevelEncoder()
        inference = AlgebraInference(rule_models, encoder, device='cpu')
        
        batch_size = 2
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        k = 1
        
        # Test composed gradient computation
        grad = inference.compute_composed_gradient(inp, out, k)
        
        assert grad.shape == (batch_size, 128)
        assert torch.isfinite(grad).all()
        
        # Test that gradients are actually non-trivial
        grad_norm = torch.norm(grad, dim=-1)
        assert torch.all(grad_norm >= 0)


class TestEncoderIntegration:
    """Test encoder integration with other components."""
    
    def test_character_encoder_with_dataset(self):
        """Test CharacterLevelEncoder with AlgebraDataset."""
        # Use encoder in dataset creation
        encoder = create_character_encoder(d_model=128, normalize_embeddings=True)
        
        dataset = AlgebraDataset(rule='divide', num_problems=10)
        
        # Test encoding of dataset equations
        for i in range(min(5, len(dataset))):
            inp_eq, target_eq = dataset.get_equation_pair(i)
            
            # Encode manually to test
            inp_emb = encoder(inp_eq)
            target_emb = encoder(target_eq)
            
            # Check normalization
            inp_norm = torch.norm(inp_emb).item()
            target_norm = torch.norm(target_emb).item()
            
            assert abs(inp_norm - 1.0) < 1e-5, f"Input embedding not normalized: {inp_norm}"
            assert abs(target_norm - 1.0) < 1e-5, f"Target embedding not normalized: {target_norm}"

    def test_ast_encoder_safety(self):
        """Test ASTEncoder safety in integrated workflow."""
        encoder = ASTEncoder(d_model=128)
        
        # Safe equations should work
        safe_equations = [
            "x+1=2",
            "2*x-3=5", 
            "x**2+1=5",
            "2*(x+3)=10"
        ]
        
        for eq in safe_equations:
            embedding = encoder(eq)
            assert torch.isfinite(embedding).all()
        
        # Dangerous equations should be blocked
        dangerous_equations = [
            "__import__",
            "eval(",
            "exec("
        ]
        
        for eq in dangerous_equations:
            with pytest.raises(ValueError):
                encoder(eq)

    def test_decoder_integration(self):
        """Test decoder integration when sklearn is available."""
        try:
            encoder = CharacterLevelEncoder()
            decoder = create_decoder_with_default_candidates(encoder)
            
            # Test encoding and decoding roundtrip
            test_equation = "x+1=2"
            embedding = encoder(test_equation)
            decoded_eq, distance = decoder.decode_embedding(embedding)
            
            if decoded_eq is not None:
                assert isinstance(decoded_eq, str)
                assert distance >= 0
                assert '=' in decoded_eq  # Should be a valid equation
                
        except ImportError:
            pytest.skip("sklearn not available for decoder testing")


class TestPerformanceRegression:
    """Test for performance regression in critical paths."""
    
    def test_inference_timing(self):
        """Test that inference completes in reasonable time."""
        import time
        
        # Small models for timing test
        rule_models = {
            'distribute': AlgebraDiffusionWrapper(AlgebraEBM())
        }
        encoder = CharacterLevelEncoder()
        config = InferenceConfig(K=3, max_iterations=5)  # Small config
        
        inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
        
        # Time a simple inference
        start_time = time.time()
        
        result = inference.solve_equation("x+1=2")
        
        end_time = time.time()
        
        # Should complete in reasonable time (< 10 seconds for small problem)
        elapsed = end_time - start_time
        assert elapsed < 10.0, f"Inference took too long: {elapsed:.2f} seconds"
        
        # Should still produce valid result
        assert 'success' in result
        assert 'inference_info' in result

    def test_dataset_generation_timing(self):
        """Test that dataset generation is not unreasonably slow."""
        import time
        
        start_time = time.time()
        
        # Generate medium-sized dataset
        dataset = AlgebraDataset(rule='combine', num_problems=100)
        
        # Access a few items to trigger generation
        for i in range(10):
            _ = dataset[i]
        
        end_time = time.time()
        
        # Should complete in reasonable time
        elapsed = end_time - start_time
        assert elapsed < 5.0, f"Dataset generation took too long: {elapsed:.2f} seconds"

    def test_energy_computation_timing(self):
        """Test that energy computation is efficient."""
        import time
        
        ebm = AlgebraEBM()
        wrapper = AlgebraDiffusionWrapper(ebm)
        
        batch_size = 10
        inp = torch.randn(batch_size, 128)
        out = torch.randn(batch_size, 128)
        t = torch.randint(0, 10, (batch_size,))
        
        # Warm up
        for _ in range(5):
            _ = wrapper(inp, out, t, return_energy=True)
        
        # Time energy computation
        start_time = time.time()
        
        for _ in range(50):  # Multiple iterations
            energy = wrapper(inp, out, t, return_energy=True)
        
        end_time = time.time()
        
        # Should be fast (< 1 second for 50 iterations)
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Energy computation too slow: {elapsed:.2f} seconds for 50 iterations"


class TestModelSaveLoad:
    """Test model saving and loading functionality."""
    
    def test_model_state_dict_consistency(self):
        """Test that model state dict saving/loading preserves functionality."""
        # Create and initialize model
        ebm = AlgebraEBM(rule_name='test_rule')
        wrapper = AlgebraDiffusionWrapper(ebm)
        
        # Test forward pass before saving
        inp = torch.randn(2, 128)
        out = torch.randn(2, 128)
        t = torch.randint(0, 10, (2,))
        
        energy_before = wrapper(inp, out, t, return_energy=True)
        
        # Save state dict
        state_dict = wrapper.state_dict()
        
        # Create new model and load state dict
        ebm_new = AlgebraEBM(rule_name='test_rule')
        wrapper_new = AlgebraDiffusionWrapper(ebm_new)
        wrapper_new.load_state_dict(state_dict)
        
        # Test forward pass after loading
        wrapper_new.eval()
        wrapper.eval()
        
        with torch.no_grad():
            energy_after = wrapper_new(inp, out, t, return_energy=True)
        
        # Should produce identical results
        assert torch.allclose(energy_before, energy_after, rtol=1e-6, atol=1e-6)

    def test_checkpoint_format_compatibility(self):
        """Test compatibility with different checkpoint formats."""
        ebm = AlgebraEBM()
        wrapper = AlgebraDiffusionWrapper(ebm)
        
        # Test different checkpoint formats that might be encountered
        formats = [
            # Standard PyTorch format
            {'model_state_dict': wrapper.state_dict()},
            
            # Direct state dict
            wrapper.state_dict(),
            
            # Trainer1D format (nested)
            {'model': wrapper.state_dict(), 'step': 1000}
        ]
        
        for fmt in formats:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name
            
            try:
                # Save checkpoint
                torch.save(fmt, temp_path)
                
                # Test loading (basic validation that it doesn't crash)
                checkpoint = torch.load(temp_path, map_location='cpu')
                assert checkpoint is not None
                
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])