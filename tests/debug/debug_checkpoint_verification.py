#!/usr/bin/env python3
"""
Checkpoint Verification Script for Algebra EBM System

This script validates model checkpoint integrity, file paths, and structure
according to the implementation requirements in T1.

Usage:
    python debug_checkpoint_verification.py
"""

import hashlib
import os
import os.path
import time
from datetime import datetime
from pathlib import Path
import torch
from typing import Dict, List, Optional, Tuple

def get_expected_checkpoint_paths() -> Dict[str, List[str]]:
    """
    Get expected checkpoint paths based on the inference patterns.
    Returns dictionary mapping rule types to potential checkpoint paths.
    """
    model_dir = './results'
    rule_names = ['combine', 'distribute', 'divide', 'isolate']
    
    checkpoint_paths = {}
    
    for rule_name in rule_names:
        # Based on algebra_inference.py:759-765, try multiple possible checkpoint paths
        potential_paths = [
            f'{model_dir}/{rule_name}/model.pt',
            f'{model_dir}/{rule_name}/checkpoint.pt', 
            f'{model_dir}/{rule_name}/model-1.pt',
            f'{model_dir}/{rule_name}/model-final.pt'
        ]
        checkpoint_paths[rule_name] = potential_paths
    
    return checkpoint_paths

def verify_checkpoint_file(checkpoint_path: str) -> Dict[str, any]:
    """
    Verify a single checkpoint file's integrity and structure.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with verification results
    """
    result = {
        'path': checkpoint_path,
        'exists': False,
        'readable': False,
        'modified_time': None,
        'modified_readable': None,
        'file_size': None,
        'hash': None,
        'torch_loadable': False,
        'checkpoint_keys': [],
        'checkpoint_type': 'unknown',
        'epoch': 'unknown',
        'structure_valid': False,
        'errors': []
    }
    
    try:
        # Check if file exists
        result['exists'] = os.path.exists(checkpoint_path)
        if not result['exists']:
            result['errors'].append("File does not exist")
            return result
        
        # Check if file is readable
        result['readable'] = os.access(checkpoint_path, os.R_OK)
        if not result['readable']:
            result['errors'].append("File exists but is not readable")
            return result
            
        # Get file metadata
        stat_info = os.stat(checkpoint_path)
        result['modified_time'] = stat_info.st_mtime
        result['modified_readable'] = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        result['file_size'] = stat_info.st_size
        
        # Calculate file hash
        with open(checkpoint_path, 'rb') as f:
            file_contents = f.read()
            result['hash'] = hashlib.sha256(file_contents).hexdigest()
        
        # Try to load with PyTorch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        result['torch_loadable'] = True
        
        # Analyze checkpoint structure
        if isinstance(checkpoint, dict):
            result['checkpoint_keys'] = list(checkpoint.keys())
            
            # Detect checkpoint type based on keys (patterns from algebra_inference.py)
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                result['checkpoint_type'] = 'Trainer1D/GaussianDiffusion1D'
                # Check if nested model contains ebm keys
                model_keys = list(checkpoint['model'].keys())
                if any(key.startswith('ebm.') for key in model_keys):
                    result['checkpoint_type'] += ' (with ebm keys)'
            elif 'model_state_dict' in checkpoint:
                result['checkpoint_type'] = 'Standard PyTorch'
            elif any(key.startswith('ebm.') for key in checkpoint.keys()):
                result['checkpoint_type'] = 'Direct EBM state dict'
            
            # Extract epoch information
            if 'epoch' in checkpoint:
                result['epoch'] = checkpoint['epoch']
            elif 'step' in checkpoint:
                result['epoch'] = f"step_{checkpoint['step']}"
            
            # Check structure validity
            # Standard PyTorch checkpoint should have these keys
            standard_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            # Trainer1D checkpoint should have these keys
            trainer1d_keys = ['model', 'step']
            
            has_standard = all(key in checkpoint for key in standard_keys)
            has_trainer1d = all(key in checkpoint for key in trainer1d_keys)
            has_direct_ebm = any(key.startswith('ebm.') for key in checkpoint.keys())
            
            result['structure_valid'] = has_standard or has_trainer1d or has_direct_ebm
            
            if not result['structure_valid']:
                result['errors'].append(f"Checkpoint structure invalid. Keys: {result['checkpoint_keys']}")
        
        else:
            result['errors'].append(f"Checkpoint is not a dict, type: {type(checkpoint)}")
            
    except Exception as e:
        result['errors'].append(f"Error loading checkpoint: {str(e)}")
    
    return result

def verify_checkpoint_integrity() -> bool:
    """
    Main checkpoint verification function.
    Verifies all expected checkpoints in the system.
    
    Returns:
        True if all verification criteria are met, False otherwise
    """
    print("=== CHECKPOINT VERIFICATION SCRIPT ===")
    print(f"Starting verification at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checkpoint_paths = get_expected_checkpoint_paths()
    
    all_results = {}
    found_checkpoints = 0
    valid_checkpoints = 0
    
    for rule_type, paths in checkpoint_paths.items():
        print(f"Rule '{rule_type}':")
        
        rule_found = False
        rule_valid = False
        
        for checkpoint_path in paths:
            result = verify_checkpoint_file(checkpoint_path)
            all_results[checkpoint_path] = result
            
            print(f"  Path: {checkpoint_path}")
            print(f"    Exists: {result['exists']}")
            
            if result['exists']:
                rule_found = True
                found_checkpoints += 1
                
                print(f"    Readable: {result['readable']}")
                print(f"    Modified: {result['modified_readable']}")
                print(f"    Size: {result['file_size']:,} bytes")
                print(f"    Hash: {result['hash'][:16]}...")
                print(f"    PyTorch loadable: {result['torch_loadable']}")
                
                if result['torch_loadable']:
                    print(f"    Checkpoint type: {result['checkpoint_type']}")
                    print(f"    Epoch/Step: {result['epoch']}")
                    print(f"    Keys: {result['checkpoint_keys']}")
                    print(f"    Structure valid: {result['structure_valid']}")
                    
                    if result['structure_valid']:
                        rule_valid = True
                        valid_checkpoints += 1
                
                if result['errors']:
                    print(f"    Errors: {result['errors']}")
                
                # Only check first existing checkpoint per rule
                break
            
        if not rule_found:
            print(f"    ❌ No checkpoint found for rule '{rule_type}'")
        elif rule_valid:
            print(f"    ✅ Valid checkpoint found for rule '{rule_type}'")
        else:
            print(f"    ⚠️  Checkpoint found but validation failed for rule '{rule_type}'")
        
        print()
    
    # Summary
    total_rules = len(checkpoint_paths)
    print("=== VERIFICATION SUMMARY ===")
    print(f"Total rule types checked: {total_rules}")
    print(f"Checkpoints found: {found_checkpoints}")
    print(f"Valid checkpoints: {valid_checkpoints}")
    
    # Check success criteria from T1
    success_criteria = []
    
    # Criterion 1: At least some checkpoint paths exist and are readable
    if found_checkpoints > 0:
        success_criteria.append("✅ Checkpoint files found and readable")
        
        # Check modification timestamps are reasonable (within last year)
        current_time = time.time()
        one_year_ago = current_time - (365 * 24 * 60 * 60)
        
        reasonable_timestamps = 0
        for path, result in all_results.items():
            if result['exists'] and result['modified_time']:
                if result['modified_time'] > one_year_ago:
                    reasonable_timestamps += 1
        
        if reasonable_timestamps > 0:
            success_criteria.append("✅ File modification timestamps are reasonable")
        else:
            success_criteria.append("❌ No recent checkpoint modifications found")
            
        # Criterion 2: Checkpoint structure contains required keys
        if valid_checkpoints > 0:
            success_criteria.append("✅ Checkpoint structure contains required keys")
        else:
            success_criteria.append("❌ No checkpoints have valid structure")
    else:
        success_criteria.append("❌ No checkpoint paths exist")
        success_criteria.append("❌ Cannot verify timestamps (no files found)")
        success_criteria.append("❌ Cannot verify structure (no files found)")
    
    print()
    print("=== SUCCESS CRITERIA CHECK ===")
    for criterion in success_criteria:
        print(criterion)
    
    # Overall success if we have at least one valid checkpoint
    overall_success = valid_checkpoints > 0
    
    print(f"\n{'✅ OVERALL SUCCESS' if overall_success else '❌ OVERALL FAILURE'}")
    
    if not overall_success:
        print("\n=== RECOMMENDATIONS ===")
        if found_checkpoints == 0:
            print("- No checkpoints found. Train models first using train_algebra.py")
            print("- Check if model training completed successfully")
            print("- Verify results directory path and permissions")
        elif valid_checkpoints == 0:
            print("- Checkpoints found but structure is invalid")
            print("- Check checkpoint format compatibility")
            print("- Consider retraining with correct checkpoint format")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = verify_checkpoint_integrity()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(2)