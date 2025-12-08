#!/usr/bin/env python3
"""
Debug script to find the exact location of the tuple unpacking error
"""

import sys
sys.path.append('.')

import torch
import logging
import traceback

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Import the components
    from algebra_evaluation import compute_invalid_rate
    from algebra_encoder import validate_equation_syntax
    
    # Test the specific function that's likely causing the error
    print("Testing validate_equation_syntax...")
    
    test_equations = [
        "x=2",
        "-10*(x-5)+3-2=11", 
        "-7*x=-14",
        "-9*x+4=-23",
        None,  # This could cause issues
        "",    # Empty string
    ]
    
    for eq in test_equations:
        try:
            print(f"\nTesting equation: {eq}")
            if eq is None:
                print("Skipping None equation")
                continue
                
            result = validate_equation_syntax(eq)
            print(f"validate_equation_syntax returned: {result} (length: {len(result)})")
            
            # Try unpacking
            is_valid, error_msg, parsed_expr = result
            print(f"Successfully unpacked: is_valid={is_valid}")
            
        except Exception as e:
            print(f"Error with equation '{eq}': {e}")
            traceback.print_exc()
    
    # Test compute_invalid_rate function
    print("\n" + "="*50)
    print("Testing compute_invalid_rate...")
    
    try:
        test_pred_equations = ["x=2", None, "-7*x=-14", "invalid syntax}"]
        result = compute_invalid_rate(test_pred_equations)
        print(f"compute_invalid_rate succeeded: {result}")
    except Exception as e:
        print(f"Error in compute_invalid_rate: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"Top-level error: {e}")
    traceback.print_exc()