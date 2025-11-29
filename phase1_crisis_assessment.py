#!/usr/bin/env python3
"""
Phase 1 Integration and Crisis Assessment - Task T6 Implementation

This script runs all Phase 1 diagnostic tests and determines the next steps
for systematic model failure diagnosis. Implements the exact specification
from the T6 implementation plan.

Dependencies: All Phase 1 tasks completed (T1, T2, T3, T4, T5)
"""

import sys
import os
sys.path.append('.')

import logging
from typing import Dict, Any, List
from pathlib import Path

# Import Phase 1 diagnostic components
from debug_checkpoint_verification import verify_checkpoint_integrity
from debug_conditioning_test import test_equation_conditioning
from debug_distance_validation import test_distance_function  
from debug_statistical_test import statistical_conditioning_test
from debug_template_analysis import analyze_template_energies

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def phase1_assessment() -> str:
    """
    Run all Phase 1 tests and determine next steps.
    
    This implements the exact specification from the T6 implementation plan,
    including the decision tree for determining root causes and recommendations.
    
    Returns:
        String indicating assessment result: "resolved", "wrong_checkpoint", 
        "broken_conditioning", "broken_distance", or "complex_failure"
    """
    print("=== PHASE 1 CRISIS ASSESSMENT ===")
    print("Running comprehensive diagnostic tests to identify root cause")
    print("=" * 60)
    
    # Define tests with exact function mappings from implementation plan
    tests = [
        ("Checkpoint Integrity", verify_checkpoint_integrity),
        ("Equation Conditioning", test_equation_conditioning),
        ("Distance Function", test_distance_function),
        ("Statistical Safeguard", lambda: statistical_conditioning_test(20)),
        ("Template Energy Analysis", analyze_template_energies)
    ]
    
    results = {}
    
    # Execute all Phase 1 diagnostic tests
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print(f"\n" + "=" * 60)
    print("PHASE 1 DIAGNOSTIC RESULTS SUMMARY")
    print("=" * 60)
    
    # Display summary
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25s}: {status}")
    
    # Determine root cause using implementation plan decision tree
    failed_tests = [name for name, result in results.items() if not result]
    
    print(f"\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    if len(failed_tests) == 0:
        print("✅ CRISIS RESOLVED: All Phase 1 tests passed")
        print("🎉 All diagnostic systems are functioning correctly!")
        print("📋 RECOMMENDATION: Proceed to Phase 3 infrastructure (skip Phase 2)")
        print("   The systematic failure may have been resolved or")
        print("   was a transient issue that has self-corrected.")
        return "resolved"
    
    elif "Checkpoint Integrity" in failed_tests:
        print("❌ ROOT CAUSE: Wrong checkpoint loaded")
        print("🔍 ANALYSIS: Model checkpoint files are missing, corrupted, or incorrectly located")
        print("📋 RECOMMENDATION: Fix checkpoint paths and reload model")
        print("   ACTIONS:")
        print("   1. Check if models were trained successfully")
        print("   2. Verify checkpoint file paths in results/ directories")  
        print("   3. Re-run training if checkpoints are missing")
        print("   4. Check disk space and file permissions")
        return "wrong_checkpoint"
    
    elif "Equation Conditioning" in failed_tests or "Statistical Safeguard" in failed_tests:
        print("❌ ROOT CAUSE: Broken conditioning mechanism")
        print("🔍 ANALYSIS: Energy function does not properly condition on input equations")
        print("📋 RECOMMENDATION: Debug energy function input processing")
        print("   INDICATORS:")
        if "Equation Conditioning" in failed_tests:
            print("   - Basic conditioning test failed: energy std deviation < 0.1")
        if "Statistical Safeguard" in failed_tests:
            print("   - Statistical safeguard failed: insufficient energy variation across diverse equations")
        print("   ACTIONS:")
        print("   1. Check equation encoding pipeline")
        print("   2. Verify model architecture and input processing") 
        print("   3. Review training data and conditioning mechanisms")
        return "broken_conditioning"
    
    elif "Distance Function" in failed_tests:
        print("❌ ROOT CAUSE: Distance function misconfiguration")
        print("🔍 ANALYSIS: Embedding distance calculation is not properly calibrated")
        print("📋 RECOMMENDATION: Debug canonicalization and distance metric")
        print("   INDICATORS:")
        print("   - Self-distances too large or different-equation distances too small")
        print("   ACTIONS:")
        print("   1. Check equation canonicalization process")
        print("   2. Verify embedding encoder consistency")
        print("   3. Review distance threshold calibration")
        return "broken_distance"
    
    else:
        print("⚠️  COMPLEX FAILURE: Multiple issues detected")
        print(f"🔍 FAILED SYSTEMS: {', '.join(failed_tests)}")
        print("📋 RECOMMENDATION: Proceed to Phase 2 systematic validation")
        print("   ANALYSIS: Multiple diagnostic systems failing suggests")
        print("   deep architectural or training issues requiring comprehensive")
        print("   validation and debugging.")
        print("   ACTIONS:")
        print("   1. Run Phase 2 extended statistical conditioning verification")
        print("   2. Perform training data integrity audit")
        print("   3. Implement independent validation components")
        print("   4. Consider architectural review and debugging")
        return "complex_failure"


def generate_crisis_report(assessment_result: str, output_file: str = "phase1_crisis_report.txt") -> None:
    """
    Generate a comprehensive crisis assessment report.
    
    Args:
        assessment_result: Result from phase1_assessment()
        output_file: Path to save the report
    """
    print(f"\n📄 Generating detailed crisis report: {output_file}")
    
    try:
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 1 CRISIS ASSESSMENT REPORT\n") 
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Assessment Result: {assessment_result}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            if assessment_result == "resolved":
                f.write("✅ STATUS: CRISIS RESOLVED\n")
                f.write("All diagnostic systems are functioning correctly.\n")
                f.write("No immediate action required.\n\n")
                
            elif assessment_result == "wrong_checkpoint":
                f.write("❌ STATUS: CHECKPOINT FAILURE\n")
                f.write("Model checkpoints are missing or corrupted.\n")
                f.write("PRIORITY: HIGH - Immediate training/checkpoint recovery needed.\n\n")
                
            elif assessment_result == "broken_conditioning":
                f.write("❌ STATUS: CONDITIONING FAILURE\n")
                f.write("Energy function conditioning mechanism is broken.\n")
                f.write("PRIORITY: CRITICAL - Core model functionality compromised.\n\n")
                
            elif assessment_result == "broken_distance":
                f.write("❌ STATUS: DISTANCE CALIBRATION FAILURE\n") 
                f.write("Embedding distance function is misconfigured.\n")
                f.write("PRIORITY: MEDIUM - Affects evaluation but not core inference.\n\n")
                
            else:
                f.write("⚠️  STATUS: COMPLEX SYSTEM FAILURE\n")
                f.write("Multiple diagnostic systems failing simultaneously.\n")
                f.write("PRIORITY: CRITICAL - Requires comprehensive investigation.\n\n")
            
            f.write("RECOMMENDED NEXT STEPS\n")
            f.write("-" * 40 + "\n")
            
            if assessment_result == "resolved":
                f.write("1. Proceed to Phase 3 infrastructure deployment\n")
                f.write("2. Begin preventive monitoring setup\n")
                f.write("3. Document resolution for future reference\n\n")
                
            elif assessment_result == "wrong_checkpoint":
                f.write("1. Verify checkpoint file locations and permissions\n")
                f.write("2. Check training logs for completion status\n") 
                f.write("3. Re-run training if checkpoints missing\n")
                f.write("4. Implement checkpoint backup procedures\n\n")
                
            elif assessment_result == "broken_conditioning":
                f.write("1. Debug equation encoding and model input processing\n")
                f.write("2. Review training data and conditioning mechanisms\n")
                f.write("3. Run Phase 2 extended conditioning verification\n")
                f.write("4. Consider model architecture review\n\n")
                
            elif assessment_result == "broken_distance":
                f.write("1. Audit equation canonicalization process\n")
                f.write("2. Verify embedding encoder consistency\n")
                f.write("3. Recalibrate distance thresholds\n")
                f.write("4. Cross-validate with independent distance implementation\n\n")
                
            else:
                f.write("1. Execute Phase 2 systematic validation immediately\n")
                f.write("2. Perform comprehensive training data audit\n")
                f.write("3. Implement independent validation frameworks\n") 
                f.write("4. Consider emergency model rollback procedures\n\n")
            
            f.write("ESCALATION CRITERIA\n")
            f.write("-" * 40 + "\n")
            f.write("- If Phase 1 resolution fails: Escalate to Phase 2\n")
            f.write("- If multiple critical systems fail: Consider emergency protocols\n")
            f.write("- If training infrastructure issues: Involve DevOps/Infrastructure team\n")
            f.write("- If architectural problems detected: Engage senior engineering review\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated by Phase 1 Crisis Assessment (T6)\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Crisis report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate crisis report: {e}")


def main():
    """
    Main entry point for Phase 1 Crisis Assessment (T6).
    """
    print("Phase 1 Crisis Assessment - Systematic Model Failure Diagnosis")
    print("=" * 70)
    print("Dependencies: T1 (Checkpoint), T2 (Conditioning), T3 (Distance),")
    print("              T4 (Statistical), T5 (Template Analysis)")
    print("=" * 70)
    
    # Run comprehensive Phase 1 assessment
    assessment_result = phase1_assessment()
    
    # Generate detailed crisis report
    generate_crisis_report(assessment_result)
    
    print(f"\n" + "=" * 70)
    print("PHASE 1 CRISIS ASSESSMENT COMPLETE")
    print("=" * 70)
    print(f"Assessment Result: {assessment_result.upper()}")
    
    # Provide clear next step guidance
    if assessment_result == "resolved":
        print("🎉 SUCCESS: Crisis resolved - ready for Phase 3 infrastructure")
        exit_code = 0
    elif assessment_result == "wrong_checkpoint":
        print("⚠️  ACTION REQUIRED: Fix checkpoint issues before continuing")
        exit_code = 1
    elif assessment_result in ["broken_conditioning", "complex_failure"]:
        print("🚨 CRITICAL: Escalate to Phase 2 systematic validation")
        exit_code = 2
    elif assessment_result == "broken_distance":
        print("⚠️  MODERATE: Fix distance calibration issues")
        exit_code = 1
    else:
        print("❓ UNKNOWN: Unexpected assessment result")
        exit_code = 3
    
    print("Detailed analysis available in: phase1_crisis_report.txt")
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    """
    Command-line interface for Phase 1 Crisis Assessment.
    
    Usage:
        python phase1_crisis_assessment.py
    """
    exit_code = main()
    sys.exit(exit_code)