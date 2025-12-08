#!/usr/bin/env python3
"""
Debug script to reproduce the evaluation error
"""

import sys
sys.path.append('.')

import torch
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Import the components
    from algebra_evaluation import evaluate_model_suite
    from algebra_dataset import ConstrainedDataset
    from algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
    from algebra_inference import load_rule_models
    
    # Create a small constrained dataset
    print("Creating constrained dataset...")
    dataset = ConstrainedDataset(
        num_rules=2,
        constraints=['positive'],
        split='test', 
        num_problems=3,  # Very small for debugging
        d_model=128,
        seed=42
    )
    
    # Create encoder
    print("Creating encoder...")
    encoder = create_character_encoder(d_model=128)
    
    # Create decoder  
    print("Creating decoder...")
    try:
        decoder = create_decoder_with_default_candidates(encoder, distance_threshold=1.5)
    except ImportError:
        print("sklearn not available - decoder disabled")
        decoder = None
    
    # Mock rule models (since we don't have trained models)
    print("Creating mock rule models...")
    from algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
    
    mock_models = {}
    for rule in ['distribute', 'combine', 'isolate', 'divide']:
        ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule)
        wrapper = AlgebraDiffusionWrapper(ebm)
        mock_models[rule] = wrapper
        
    # Try evaluation
    print("Running evaluation...")
    test_datasets = {'constrained_positive': dataset}
    
    results = evaluate_model_suite(
        rule_models=mock_models,
        test_datasets=test_datasets,
        encoder=encoder,
        decoder=decoder,
        max_samples=3  # Very small
    )
    
    print(f"Results: {results}")
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()