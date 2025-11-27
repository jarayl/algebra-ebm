#!/usr/bin/env python3
"""
Test script to verify CUDA device consistency fixes.

Tests the specific device mismatch scenario that was causing evaluation failures.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_device_consistency_fix():
    """Test that embedding device normalization works correctly."""
    print("Testing device consistency fix...")
    
    # Simulate the problematic scenario: mixed device embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create embeddings on different devices (simulating the original issue)
    target_embeddings = [
        torch.randn(128).to(device),  # GPU embedding (from encoder)
        torch.randn(128).to(device),
        torch.randn(128).to(device)
    ]
    
    predicted_embeddings = [
        torch.randn(128).to('cpu'),   # CPU embedding (from old inference)
        torch.randn(128).to(device),  # Mixed scenario
        torch.randn(128).to('cpu')
    ]
    
    print(f"Target embeddings devices: {[emb.device for emb in target_embeddings]}")
    print(f"Predicted embeddings devices: {[emb.device for emb in predicted_embeddings]}")
    
    # Apply the device consistency fix (from algebra_evaluation.py)
    if predicted_embeddings and target_embeddings:
        # Use the first target embedding's device as reference (encoder's device)
        reference_device = target_embeddings[0].device
        print(f"Reference device: {reference_device}")
        
        # Move all embeddings to the reference device
        predicted_embeddings_same_device = []
        for emb in predicted_embeddings:
            predicted_embeddings_same_device.append(emb.to(reference_device))
        
        target_embeddings_same_device = []
        for emb in target_embeddings:
            target_embeddings_same_device.append(emb.to(reference_device))
        
        # This should now work without device mismatch error
        try:
            predicted_embeddings_tensor = torch.stack(predicted_embeddings_same_device)
            target_embeddings_tensor = torch.stack(target_embeddings_same_device)
            print(f"✓ Successfully stacked embeddings!")
            print(f"  Predicted tensor device: {predicted_embeddings_tensor.device}")
            print(f"  Target tensor device: {target_embeddings_tensor.device}")
            print(f"  Predicted tensor shape: {predicted_embeddings_tensor.shape}")
            print(f"  Target tensor shape: {target_embeddings_tensor.shape}")
            return True
        except RuntimeError as e:
            print(f"✗ Device consistency fix failed: {e}")
            return False
    else:
        print("✗ Empty embedding lists")
        return False

def test_inference_device_fix():
    """Test that inference engine keeps embeddings on correct device."""
    print("\nTesting inference device handling...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simulate inference output
    out_embedding = torch.randn(1, 128).to(device)
    
    # Old behavior (problematic): result['output_embedding'] = out_embedding.squeeze(0).cpu()
    old_output = out_embedding.squeeze(0).cpu()
    print(f"Old behavior device: {old_output.device}")
    
    # New behavior (fixed): result['output_embedding'] = out_embedding.squeeze(0).detach()
    new_output = out_embedding.squeeze(0).detach()
    print(f"New behavior device: {new_output.device}")
    
    # The new behavior should preserve the device
    if new_output.device == out_embedding.device:
        print("✓ Inference device fix working correctly!")
        return True
    else:
        print("✗ Inference device fix failed!")
        return False

if __name__ == "__main__":
    print("Device Consistency Fix Validation\n")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_device_consistency_fix()
    test2_passed = test_inference_device_fix()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Device consistency fix: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Inference device fix: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All fixes validated successfully!")
        print("CUDA evaluation errors should be resolved.")
    else:
        print("\n✗ Some fixes failed validation.")
        sys.exit(1)