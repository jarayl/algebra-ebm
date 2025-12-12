#!/usr/bin/env python3
"""
Test script for the enhanced calibration method with interface validation and stratified sampling.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from algebra.algebra_inference import InferenceEngine
from algebra.algebra_encoder import CharacterLevelEncoder
from algebra.algebra_dataset import AlgebraDataset


def test_calibration_improvements():
    """Test the calibration improvements."""
    print("Testing calibration improvements...")
    
    # Create a simple dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, rule_name):
            super().__init__()
            self.rule_name = rule_name
            # Simple linear layer to simulate energy computation
            self.fc = torch.nn.Linear(256, 1)  # 128 + 128 = 256 for input+output embeddings
            
        def forward(self, inp_emb, out_emb, t, return_energy=False):
            # Combine embeddings and compute energy
            combined = torch.cat([inp_emb, out_emb], dim=-1)
            energy = self.fc(combined).squeeze(-1)
            
            # Add some rule-specific bias for testing different scales
            if self.rule_name == 'distribute':
                energy = energy * 2.0  # Larger scale
            elif self.rule_name == 'combine':
                energy = energy * 0.5  # Smaller scale
            
            return energy
    
    # Create dummy rule models
    rule_models = {
        'distribute': DummyModel('distribute'),
        'combine': DummyModel('combine'),
        'isolate': DummyModel('isolate'),
    }
    
    # Create encoder
    encoder = CharacterLevelEncoder(d_model=128)
    
    # Create inference engine
    engine = InferenceEngine(rule_models, encoder)
    
    # Create test dataset with isolate rule (which has '=' signs)
    print("Creating test dataset...")
    test_dataset = AlgebraDataset(rule='isolate', num_problems=50)
    
    # Test interface validation
    print("Testing dataset interface validation...")
    try:
        interface_info = engine._validate_dataset_interface(test_dataset)
        print(f"Interface validation passed: {interface_info}")
    except Exception as e:
        print(f"Interface validation failed: {e}")
        return
    
    # Test stratified sampling
    print("Testing stratified sampling...")
    try:
        sample_indices = engine._stratified_sample_indices(test_dataset, interface_info, 20)
        print(f"Generated {len(sample_indices)} stratified sample indices")
        
        # Check complexity distribution
        complexities = []
        for idx in sample_indices:
            input_eq, _ = engine._extract_equation_pair(test_dataset, idx, interface_info)
            complexity = engine._estimate_equation_complexity(input_eq)
            complexities.append(complexity)
        
        from collections import Counter
        complexity_counts = Counter(complexities)
        print(f"Complexity distribution: {dict(complexity_counts)}")
        
    except Exception as e:
        print(f"Stratified sampling failed: {e}")
        return
    
    # Test calibration
    print("Testing enhanced calibration...")
    try:
        calibration_scales = engine.calibrate_energy_scales(test_dataset, num_samples=30)
        print(f"Calibration completed successfully!")
        print(f"Scaling factors: {calibration_scales}")
        
        # Verify that scales are reasonable
        for rule, scale in calibration_scales.items():
            assert 0.01 <= scale <= 100.0, f"Scale {scale} for {rule} outside reasonable range"
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nCalibration improvements working correctly!")


if __name__ == "__main__":
    test_calibration_improvements()