#!/usr/bin/env python3
"""
Simple integration test for the multi-seed statistical framework.
Tests the basic functionality without running full experiments.
"""

import sys
import tempfile
import json
from pathlib import Path
import numpy as np

# Test imports
try:
    from scripts.statistical_comparison_evaluation import StatisticalComparisonFramework
    print("✅ StatisticalComparisonFramework import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test pandas import and data handling
try:
    import pandas as pd
    import scipy.stats as stats
    print("✅ Statistical dependencies available")
except Exception as e:
    print(f"❌ Statistical dependencies failed: {e}")
    sys.exit(1)

def test_framework_creation():
    """Test creating the statistical framework."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            framework = StatisticalComparisonFramework(tmp_dir, num_samples=100)
            print(f"✅ Framework created successfully in {tmp_dir}")
            return True
    except Exception as e:
        print(f"❌ Framework creation failed: {e}")
        return False

def test_data_processing():
    """Test the data processing functionality."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            framework = StatisticalComparisonFramework(tmp_dir, num_samples=100)
            
            # Create mock results data
            mock_results = {
                42: {
                    'seed': 42,
                    'monolithic_results': {
                        'single_rule_accuracy': 28.5,
                        'multi_rule_accuracy': 8.7,
                        '2_rule_accuracy': 1.5,
                        '3_rule_accuracy': 7.2,
                        '4_rule_accuracy': 17.3
                    },
                    'compositional_results': {
                        'single_rule_accuracy': 28.6,
                        'multi_rule_accuracy': 12.8,
                        '2_rule_accuracy': 1.3,
                        '3_rule_accuracy': 12.1,
                        '4_rule_accuracy': 24.7
                    }
                },
                123: {
                    'seed': 123,
                    'monolithic_results': {
                        'single_rule_accuracy': 28.7,
                        'multi_rule_accuracy': 8.9,
                        '2_rule_accuracy': 1.7,
                        '3_rule_accuracy': 7.4,
                        '4_rule_accuracy': 17.5
                    },
                    'compositional_results': {
                        'single_rule_accuracy': 28.4,
                        'multi_rule_accuracy': 12.4,
                        '2_rule_accuracy': 1.5,
                        '3_rule_accuracy': 11.9,
                        '4_rule_accuracy': 24.3
                    }
                }
            }
            
            # Test data extraction
            df = framework.extract_performance_metrics(mock_results)
            assert len(df) == 4, f"Expected 4 rows, got {len(df)}"
            assert 'approach' in df.columns, "Missing 'approach' column"
            assert 'multi_rule_acc' in df.columns, "Missing 'multi_rule_acc' column"
            print("✅ Data extraction successful")
            
            # Test statistical analysis
            stats_results = framework.compute_statistical_tests(df)
            assert 'multi_rule_acc' in stats_results, "Missing multi_rule_acc in stats"
            assert 'p_value' in stats_results['multi_rule_acc'], "Missing p_value"
            print("✅ Statistical analysis successful")
            
            # Test report generation
            summary = framework.generate_summary_report(df, stats_results)
            assert len(summary) > 100, "Summary report too short"
            assert "Statistical Comparison Analysis Report" in summary, "Missing title"
            print("✅ Report generation successful")
            
            # Test LaTeX table generation
            latex = framework.generate_paper_tables(df, stats_results)
            assert "\\begin{table}" in latex, "Missing LaTeX table"
            assert "\\toprule" in latex, "Missing table formatting"
            print("✅ LaTeX table generation successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eval_algebra_seed_parameter():
    """Test that eval_algebra.py accepts seed parameter."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "eval_algebra.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "--seed SEED" in result.stdout:
            print("✅ eval_algebra.py has seed parameter")
            return True
        else:
            print("❌ eval_algebra.py missing seed parameter")
            return False
            
    except Exception as e:
        print(f"❌ eval_algebra.py test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("="*60)
    print("MULTI-SEED STATISTICAL FRAMEWORK INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Framework Creation", test_framework_creation),
        ("Data Processing", test_data_processing),  
        ("eval_algebra.py Integration", test_eval_algebra_seed_parameter)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Framework ready for real experiments!")
        print("\nNext steps:")
        print("1. Run: sbatch run_train_monolithic.sh")
        print("2. Run: sbatch run_train_algebra.sh")  
        print("3. Run: sbatch run_comparison_eval.sh")
        print("4. Check statistical_comparison_results/ for paper-ready tables")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before running experiments")
        return 1

if __name__ == '__main__':
    sys.exit(main())