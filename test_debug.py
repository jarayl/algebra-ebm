#!/usr/bin/env python3
"""
Test the enhanced debugging in eval script
"""

import subprocess
import sys

# Test with a quick evaluation that should show any errors with full tracebacks
cmd = [
    sys.executable, "eval_algebra.py",
    "--model_dir", "./results",  # This directory likely doesn't exist
    "--eval_type", "constrained",
    "--constrained_problems", "10",
    "--quick_test",
    "--verbose"
]

print("Testing enhanced error reporting...")
print(f"Running command: {' '.join(cmd)}")
print("=" * 60)

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
except subprocess.TimeoutExpired:
    print("Command timed out")
except Exception as e:
    print(f"Error running command: {e}")