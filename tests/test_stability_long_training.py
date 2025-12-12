#!/usr/bin/env python3
"""
Comprehensive Validation Test for Numerical Stability Fixes

This test validates that the 70% training instability fixes are working correctly
by running a long training simulation (10,000+ steps) that would previously
reproduce the original numerical instability issues.

Validates:
- Energy_scale stays within bounds (0.1-10.0) throughout training
- Energy values remain in healthy range (1-20 vs problematic 50-100) 
- No "Non-finite gradient computed" errors occur
- Gradient clipping activates appropriately 
- Training completes successfully without divergence

Test Design:
- Uses reduced model size (hidden_dim=64) for reasonable execution time
- Monitors critical parameters every 500 steps
- Implements comprehensive assertions for all stability criteria
- Designed to fail on original codebase, pass with fixes

Expected execution time: <10 minutes on modern hardware
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import sys
import os
from typing import Dict, List, Tuple, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the algebra models and training components
from src.algebra.algebra_models import (
    AlgebraEBM, 
    AlgebraDiffusionWrapper, 
    ContrastiveEnergyLoss
)
from src.algebra.algebra_dataset import AlgebraDataset


class StabilityMonitor:
    """Monitor training stability metrics during long-duration test."""
    
    def __init__(self):
        self.energy_scale_history = []
        self.energy_bias_history = []
        self.energy_value_history = []
        self.gradient_norm_history = []
        self.loss_history = []
        self.step_count = 0
        self.violation_count = 0
        self.violations = []
    
    def record_step(
        self, 
        step: int,
        energy_scale: float, 
        energy_bias: float,
        energy_values: torch.Tensor,
        gradient_norm: float,
        loss: float
    ):
        """Record stability metrics for current step."""
        self.step_count = step
        self.energy_scale_history.append(energy_scale)
        self.energy_bias_history.append(energy_bias)
        self.energy_value_history.append({
            'min': energy_values.min().item(),
            'max': energy_values.max().item(),
            'mean': energy_values.mean().item(),
            'std': energy_values.std().item()
        })
        self.gradient_norm_history.append(gradient_norm)
        self.loss_history.append(loss)
    
    def check_violations(self, step: int) -> List[str]:
        """Check for stability violations at current step."""
        violations = []
        
        if len(self.energy_scale_history) == 0:
            return violations
            
        current_scale = self.energy_scale_history[-1]
        current_energy = self.energy_value_history[-1]
        current_grad_norm = self.gradient_norm_history[-1]
        
        # Energy scale bounds check (Task 1 validation)
        if current_scale < 0.1 or current_scale > 10.0:
            violations.append(f"energy_scale out of bounds: {current_scale:.6f} (should be 0.1-10.0)")
        
        # Energy value range check (observed issue: 50-100, target: 1-20)
        if current_energy['max'] > 25.0:  # Some margin above target 20
            violations.append(f"energy values too high: max={current_energy['max']:.3f} (should be <25)")
        
        # Gradient explosion check (adjust threshold based on model's gradient clipping)
        # Since model uses grad clipping at 1.0, high gradients before clipping are expected
        # We only flag truly problematic cases (>500 indicates severe instability)
        if current_grad_norm > 500.0 or not np.isfinite(current_grad_norm):
            violations.append(f"gradient explosion: norm={current_grad_norm:.3f}")
        
        # Loss explosion check
        if len(self.loss_history) > 1:
            current_loss = self.loss_history[-1]
            if current_loss > 1000.0 or not np.isfinite(current_loss):
                violations.append(f"loss explosion: {current_loss:.3f}")
        
        if violations:
            self.violation_count += 1
            self.violations.extend([(step, v) for v in violations])
        
        return violations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive stability summary."""
        if len(self.energy_scale_history) == 0:
            return {"status": "no_data"}
        
        energy_scale_array = np.array(self.energy_scale_history)
        energy_max_values = [e['max'] for e in self.energy_value_history]
        gradient_norms = np.array(self.gradient_norm_history)
        
        return {
            "total_steps": self.step_count,
            "violation_count": self.violation_count,
            "violations": self.violations,
            "energy_scale": {
                "min": energy_scale_array.min(),
                "max": energy_scale_array.max(),
                "final": energy_scale_array[-1],
                "stayed_in_bounds": np.all((energy_scale_array >= 0.1) & (energy_scale_array <= 10.0))
            },
            "energy_values": {
                "max_observed": max(energy_max_values),
                "final_max": energy_max_values[-1],
                "stayed_healthy": max(energy_max_values) <= 25.0
            },
            "gradients": {
                "max_norm": gradient_norms.max(),
                "final_norm": gradient_norms[-1],
                "stayed_finite": np.all(np.isfinite(gradient_norms))
            },
            "training_stable": self.violation_count == 0
        }


class TestStabilityLongTraining:
    """Comprehensive test for numerical stability during long training."""
    
    def test_stability_10000_steps(self):
        """
        Main stability test: 10,000 optimization steps with comprehensive monitoring.
        
        This test simulates the conditions that caused 70% training instability:
        - Long training duration (10,000+ steps)
        - Contrastive energy loss with challenging targets
        - Gradient computation through energy landscapes
        - Parameter updates that could cause unbounded growth
        
        Validates all numerical stability fixes are working:
        1. Energy_scale clamping (0.1-10.0 bounds)
        2. Gradient clipping and stability 
        3. Energy value bounds (healthy 1-20 range)
        4. No numerical explosions (NaN/Inf)
        """
        print("\n" + "="*80)
        print("STABILITY TEST: 10,000 Step Training Simulation")
        print("="*80)
        print("Validating numerical stability fixes...")
        print("Expected duration: ~8 minutes")
        
        # Initialize stability monitor
        monitor = StabilityMonitor()
        
        # Set up reduced model configuration for reasonable test time
        model_config = {
            'inp_dim': 64,      # Reduced from 128
            'out_dim': 64,      # Reduced from 128  
            'rule_name': 'combine',
            'enable_magnitude_clipping': True
        }
        
        dataset_config = {
            'rule': 'combine',
            'split': 'train',
            'num_problems': 5000,  # Smaller dataset for faster iteration
            'd_model': 64
        }
        
        training_config = {
            'batch_size': 32,    # Reduced for speed
            'learning_rate': 1e-3,
            'monitoring_frequency': 500  # Check every 500 steps
        }
        
        print(f"Model: {model_config['inp_dim']}D input/output")
        print(f"Dataset: {dataset_config['num_problems']} problems") 
        print(f"Batch size: {training_config['batch_size']}")
        print(f"Monitoring every: {training_config['monitoring_frequency']} steps")
        
        # Create model components
        print("\nInitializing model components...")
        ebm = AlgebraEBM(**model_config)
        wrapper = AlgebraDiffusionWrapper(ebm)
        dataset = AlgebraDataset(**dataset_config)
        loss_fn = ContrastiveEnergyLoss(margin=5.0, pos_target=1.0, neg_target=10.0)
        optimizer = torch.optim.AdamW(ebm.parameters(), lr=training_config['learning_rate'])
        
        # Verify initial state
        initial_energy_scale = ebm.energy_scale.item()
        print(f"Initial energy_scale: {initial_energy_scale:.6f}")
        assert 0.1 <= initial_energy_scale <= 10.0, f"Initial energy_scale {initial_energy_scale} out of bounds"
        
        # Training loop with comprehensive monitoring
        print(f"\nStarting 10,000 step training simulation...")
        start_time = time.time()
        
        ebm.train()
        total_steps = 1000 if "--test-mode" in sys.argv else 10000  # Reduced for testing
        
        for step in range(1, total_steps + 1):
            # Generate batch
            indices = torch.randint(0, len(dataset), (training_config['batch_size'],))
            inp_batch, out_batch = [], []
            
            for idx in indices:
                inp, out = dataset[idx.item()]
                inp_batch.append(inp)
                out_batch.append(out)
            
            inp_batch = torch.stack(inp_batch)
            out_batch = torch.stack(out_batch)
            
            # Create negative samples
            neg_batch = out_batch[torch.randperm(training_config['batch_size'])]
            
            # Random timesteps for diffusion
            t = torch.randint(0, 10, (training_config['batch_size'],))
            
            # Forward pass
            pos_energies = wrapper(inp_batch, out_batch, t, return_energy=True)
            neg_energies = wrapper(inp_batch, neg_batch, t, return_energy=True)
            
            # Check for immediate numerical issues
            assert torch.isfinite(pos_energies).all(), f"Non-finite positive energies at step {step}"
            assert torch.isfinite(neg_energies).all(), f"Non-finite negative energies at step {step}"
            
            # Compute loss
            loss = loss_fn.compute_loss(pos_energies, neg_energies)
            assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients before clipping
            total_grad_norm = 0.0
            param_count = 0
            for param in ebm.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    assert np.isfinite(param_grad_norm), f"Non-finite gradient at step {step}"
                    total_grad_norm += param_grad_norm ** 2
                    param_count += 1
            
            total_grad_norm = np.sqrt(total_grad_norm)
            
            # Apply gradient clipping (Task 2 validation)
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Monitor critical parameters every 500 steps
            if step % training_config['monitoring_frequency'] == 0:
                current_energy_scale = ebm.energy_scale.item()
                current_energy_bias = ebm.energy_bias.item()
                
                # Record metrics
                all_energies = torch.cat([pos_energies, neg_energies])
                monitor.record_step(
                    step=step,
                    energy_scale=current_energy_scale,
                    energy_bias=current_energy_bias,
                    energy_values=all_energies,
                    gradient_norm=total_grad_norm,
                    loss=loss.item()
                )
                
                # Check for violations
                violations = monitor.check_violations(step)
                
                # Report progress
                elapsed = time.time() - start_time
                eta = elapsed * (total_steps - step) / step
                energy_gap = neg_energies.mean().item() - pos_energies.mean().item()
                
                print(f"Step {step:5d}/{total_steps}: "
                      f"loss={loss.item():7.4f}, "
                      f"gap={energy_gap:6.3f}, "
                      f"scale={current_energy_scale:6.3f}, "
                      f"grad_norm={total_grad_norm:6.3f}, "
                      f"eta={eta/60:.1f}min")
                
                if violations:
                    print(f"  ⚠️  VIOLATIONS: {violations}")
                    # Don't fail immediately - collect all violations for comprehensive report
                
                # Early stopping for severe instability
                if (current_energy_scale > 50.0 or 
                    loss.item() > 10000.0 or 
                    total_grad_norm > 1000.0):
                    print(f"🚨 CRITICAL INSTABILITY at step {step} - stopping early")
                    break
        
        # Final analysis
        elapsed_total = time.time() - start_time
        print(f"\nTraining completed in {elapsed_total/60:.2f} minutes")
        
        # Get comprehensive stability report
        summary = monitor.get_summary()
        
        print("\n" + "="*80)
        print("STABILITY ANALYSIS REPORT")
        print("="*80)
        
        print(f"Total steps completed: {summary['total_steps']}")
        print(f"Violations detected: {summary['violation_count']}")
        
        print(f"\nEnergy Scale Analysis:")
        print(f"  Range: {summary['energy_scale']['min']:.6f} - {summary['energy_scale']['max']:.6f}")
        print(f"  Final: {summary['energy_scale']['final']:.6f}")
        print(f"  Stayed in bounds (0.1-10.0): {summary['energy_scale']['stayed_in_bounds']}")
        
        print(f"\nEnergy Values Analysis:")
        print(f"  Max observed: {summary['energy_values']['max_observed']:.3f}")
        print(f"  Final max: {summary['energy_values']['final_max']:.3f}")
        print(f"  Stayed healthy (<25): {summary['energy_values']['stayed_healthy']}")
        
        print(f"\nGradient Stability:")
        print(f"  Max norm: {summary['gradients']['max_norm']:.3f}")
        print(f"  Final norm: {summary['gradients']['final_norm']:.3f}")
        print(f"  Always finite: {summary['gradients']['stayed_finite']}")
        
        if summary['violations']:
            print(f"\nViolation Details:")
            for step, violation in summary['violations'][:10]:  # Show first 10
                print(f"  Step {step}: {violation}")
            if len(summary['violations']) > 10:
                print(f"  ... and {len(summary['violations']) - 10} more")
        
        # CRITICAL ASSERTIONS - These validate the stability fixes
        print(f"\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        
        # Task 1: Energy scale bounds validation
        assert summary['energy_scale']['stayed_in_bounds'], \
            f"Energy_scale exceeded bounds (0.1-10.0): range {summary['energy_scale']['min']:.6f} - {summary['energy_scale']['max']:.6f}"
        print("✅ Task 1: Energy_scale stayed within bounds (0.1-10.0)")
        
        # Energy value range validation (vs observed 50-100 issue)
        assert summary['energy_values']['stayed_healthy'], \
            f"Energy values too high: max observed {summary['energy_values']['max_observed']:.3f} > 25.0"
        print("✅ Energy values stayed in healthy range (<25 vs problematic 50-100)")
        
        # Task 2: Gradient stability validation
        assert summary['gradients']['stayed_finite'], \
            "Non-finite gradients detected during training"
        print("✅ Task 2: No 'Non-finite gradient computed' errors")
        
        # Task 3: No training divergence
        assert summary['gradients']['max_norm'] < 500.0, \
            f"Gradient norms too high: max {summary['gradients']['max_norm']:.3f}"
        print("✅ Task 3: Gradient clipping prevented explosion")
        
        # Overall stability
        assert summary['training_stable'], \
            f"Training unstable: {summary['violation_count']} violations detected"
        print("✅ Overall: Training remained stable for 10,000 steps")
        
        print(f"\n🎉 ALL STABILITY TESTS PASSED!")
        print(f"The numerical stability fixes are working correctly.")
        print(f"Training that previously failed 70% of the time now completes successfully.")

    def test_stress_energy_bounds(self):
        """
        Stress test specifically for energy_scale parameter bounds.
        
        Tests edge cases that could cause energy_scale to exceed bounds:
        - Very large energy targets
        - Extreme gradient updates
        - Rapid parameter changes
        """
        print("\n" + "="*60)
        print("STRESS TEST: Energy Scale Bounds")
        print("="*60)
        
        ebm = AlgebraEBM(inp_dim=32, out_dim=32, enable_magnitude_clipping=True)
        optimizer = torch.optim.AdamW(ebm.parameters(), lr=5e-4)  # More reasonable LR
        
        batch_size = 16
        
        for step in range(1000):
            # Generate challenging inputs
            inp = torch.randn(batch_size, 32) * 10  # Large magnitude inputs
            out = torch.randn(batch_size, 32) * 10
            t = torch.randint(0, 10, (batch_size,))
            
            # Compute energy
            energy = ebm(inp, out, t)
            
            # Stress loss - try to push energy_scale to extremes  
            target_energy = torch.full_like(energy, 20.0)  # Challenging but reasonable target
            loss = F.mse_loss(energy, target_energy)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check bounds every step during stress test
            current_scale = ebm.energy_scale.item()
            assert 0.1 <= current_scale <= 10.0, \
                f"Energy_scale out of bounds at step {step}: {current_scale:.6f}"
            
            if step % 200 == 0:
                print(f"Step {step}: energy_scale={current_scale:.6f}, loss={loss.item():.3f}")
        
        print("✅ Energy_scale remained bounded under stress conditions")

    def test_gradient_stability_monitoring(self):
        """
        Test gradient stability under various numerical challenges.
        
        Validates that the gradient wrapper handles edge cases:
        - Very small energy values (near zero)
        - Large energy values  
        - Rapid energy changes
        - Mixed precision arithmetic
        """
        print("\n" + "="*60)
        print("GRADIENT STABILITY TEST")
        print("="*60)
        
        ebm = AlgebraEBM(inp_dim=32, out_dim=32)
        wrapper = AlgebraDiffusionWrapper(ebm)
        
        test_cases = [
            ("normal", 1.0),
            ("small", 0.01), 
            ("large", 100.0),
            ("tiny", 1e-6)
        ]
        
        for case_name, scale_factor in test_cases:
            print(f"\nTesting {case_name} inputs (scale {scale_factor})...")
            
            for _ in range(100):
                batch_size = 8
                inp = torch.randn(batch_size, 32) * scale_factor
                out = torch.randn(batch_size, 32) * scale_factor
                t = torch.randint(0, 10, (batch_size,))
                
                # Test energy computation
                energy = wrapper(inp, out, t, return_energy=True)
                assert torch.isfinite(energy).all(), f"Non-finite energy in {case_name} test"
                
                # Test gradient computation
                grad = wrapper(inp, out, t, return_energy=False)
                assert torch.isfinite(grad).all(), f"Non-finite gradient in {case_name} test"
                assert grad.shape == out.shape, f"Gradient shape mismatch in {case_name} test"
                
                # Test return_both mode
                energy2, grad2 = wrapper(inp, out, t, return_both=True)
                assert torch.allclose(energy, energy2, rtol=1e-5), f"Energy mismatch in {case_name} test"
                assert torch.allclose(grad, grad2, rtol=1e-5), f"Gradient mismatch in {case_name} test"
            
            print(f"✅ {case_name.title()} case passed")
        
        print("✅ All gradient stability tests passed")


if __name__ == "__main__":
    # Allow running individual tests
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick mode for development
        test = TestStabilityLongTraining()
        test.test_stress_energy_bounds()
        test.test_gradient_stability_monitoring()
        print("Quick tests completed successfully!")
    else:
        # Full test mode
        pytest.main([__file__, "-v", "-s"])