#!/usr/bin/env python3
"""
Analyze Energy Landscape Quality from Diagnostic Evaluation

Purpose: Determine what percentage of problems have correct vs inverted energy landscapes
Baseline: 54% correct (with normalization)
Target: >80% correct (without normalization)

Usage:
    python scripts/analyze_energy_landscapes.py results/diagnostic_no_norm/evaluation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_trajectory(trajectory_file: Path) -> Dict:
    """Load trajectory JSON file."""
    with open(trajectory_file, 'r') as f:
        return json.load(f)

def analyze_single_trajectory(traj: Dict) -> Tuple[bool, float, float]:
    """
    Analyze a single problem trajectory.

    Returns:
        correct: True if E(inp→target) < E(inp→input) [correct landscape]
        energy_to_target: Final energy to target
        energy_to_input: Energy to input (identity transformation)
    """
    # Get final energies from trajectory
    if 'final_energy' in traj:
        energy_to_target = traj['final_energy']
    elif 'energies' in traj and len(traj['energies']) > 0:
        energy_to_target = traj['energies'][-1]
    else:
        return None, None, None

    # Get energy to input (if available)
    # This would be computed by evaluating E(inp, inp)
    if 'energy_to_input' in traj:
        energy_to_input = traj['energy_to_input']
    else:
        # If not available, we can't determine correctness
        return None, energy_to_target, None

    # Correct landscape: target has LOWER energy than input
    correct = energy_to_target < energy_to_input

    return correct, energy_to_target, energy_to_input

def analyze_all_trajectories(diagnostics_dir: Path) -> Dict:
    """Analyze all trajectory files in diagnostics directory."""
    trajectory_files = list(diagnostics_dir.glob("problem_*_trajectory.json"))

    if not trajectory_files:
        print(f"✗ No trajectory files found in {diagnostics_dir}")
        return None

    print(f"Found {len(trajectory_files)} trajectory files")

    results = {
        'total': len(trajectory_files),
        'correct': 0,
        'inverted': 0,
        'unknown': 0,
        'energy_gaps': [],
        'problems': []
    }

    for traj_file in trajectory_files:
        traj = load_trajectory(traj_file)
        correct, e_target, e_input = analyze_single_trajectory(traj)

        if correct is None:
            results['unknown'] += 1
        else:
            if correct:
                results['correct'] += 1
            else:
                results['inverted'] += 1

            energy_gap = e_input - e_target  # Positive = correct direction
            results['energy_gaps'].append(energy_gap)
            results['problems'].append({
                'file': traj_file.name,
                'correct': correct,
                'e_target': e_target,
                'e_input': e_input,
                'gap': energy_gap
            })

    return results

def print_analysis(results: Dict):
    """Print analysis results with decision criteria."""
    total = results['total']
    correct = results['correct']
    inverted = results['inverted']
    unknown = results['unknown']

    if total == 0:
        print("No data to analyze")
        return

    # Calculate percentages
    pct_correct = (correct / total) * 100 if total > 0 else 0
    pct_inverted = (inverted / total) * 100 if total > 0 else 0
    pct_unknown = (unknown / total) * 100 if total > 0 else 0

    print("\n" + "="*70)
    print("ENERGY LANDSCAPE QUALITY ANALYSIS")
    print("="*70)
    print(f"\nTotal Problems: {total}")
    print(f"  Correct Landscapes:  {correct:3d} ({pct_correct:5.1f}%) - E(inp→target) < E(inp→input) ✓")
    print(f"  Inverted Landscapes: {inverted:3d} ({pct_inverted:5.1f}%) - E(inp→target) > E(inp→input) ✗")
    print(f"  Unknown:             {unknown:3d} ({pct_unknown:5.1f}%)")

    if results['energy_gaps']:
        gaps = np.array(results['energy_gaps'])
        print(f"\nEnergy Gap Statistics (E_input - E_target):")
        print(f"  Mean:   {gaps.mean():8.3f}")
        print(f"  Median: {np.median(gaps):8.3f}")
        print(f"  Std:    {gaps.std():8.3f}")
        print(f"  Min:    {gaps.min():8.3f}")
        print(f"  Max:    {gaps.max():8.3f}")

    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    baseline_correct = 54.0
    improvement = pct_correct - baseline_correct
    print(f"\nBaseline (with normalization):    {baseline_correct:5.1f}% correct")
    print(f"Current  (without normalization): {pct_correct:5.1f}% correct")
    print(f"Improvement:                      {improvement:+5.1f} percentage points")

    print("\n" + "="*70)
    print("DECISION CRITERIA")
    print("="*70)

    if pct_correct >= 80:
        print("\n✓ SUCCESS: Energy landscapes >80% correct")
        print("\n  → ROOT CAUSE CONFIRMED")
        print("    Normalization was breaking energy learning")
        print("\n  → NEXT STEP: Full Retraining (T0b)")
        print("    Retrain all 5 models without normalization")
        print("    bash scripts/full_retrain_no_norm.sh")
        print("\n  → EXPECTED OUTCOMES:")
        print("    - Single-rule accuracy: 50-85% (vs current 6.3%)")
        print("    - Multi-rule accuracy: 10-30% (vs current 0%)")

    elif pct_correct >= 60:
        print("\n⚠ PARTIAL SUCCESS: Energy landscapes 60-80% correct")
        print("\n  → PARTIAL FIX")
        print("    Normalization contributed but not sole cause")
        print("\n  → NEXT STEP: Investigate Issue #2")
        print("    Add gradient logging to verify energy_scale/energy_bias")
        print("    are actually being optimized during training")
        print("\n  → ACTIONS:")
        print("    1. Check training logs for energy_scale values")
        print("    2. Verify parameters are in optimizer")
        print("    3. Consider full retraining anyway (may still help)")

    else:
        print("\n✗ HYPOTHESIS REJECTED: Energy landscapes <60% correct")
        print("\n  → ROOT CAUSE INCORRECT")
        print("    Normalization is not the main issue")
        print("\n  → NEXT STEPS:")
        print("    1. Investigate Issue #2 (energy scale parameter learning)")
        print("    2. Investigate Issue #3 (insufficient inference iterations)")
        print("    3. Re-examine training logs for other anomalies")
        print("\n  → ALTERNATIVE HYPOTHESES:")
        print("    - Energy scale/bias parameters stuck (not optimizing)")
        print("    - Energy function architecture fundamentally flawed")
        print("    - Decoder introducing errors (not energy landscape)")

    print("\n" + "="*70)
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_energy_landscapes.py <diagnostics_dir>")
        print("Example: python scripts/analyze_energy_landscapes.py results/diagnostic_no_norm/evaluation/diagnostics")
        sys.exit(1)

    diagnostics_dir = Path(sys.argv[1])

    if not diagnostics_dir.exists():
        # Try as evaluation dir (look for diagnostics subdir)
        eval_dir = diagnostics_dir
        diagnostics_dir = eval_dir / "diagnostics"

        if not diagnostics_dir.exists():
            print(f"✗ Directory not found: {diagnostics_dir}")
            sys.exit(1)

    print(f"Analyzing energy landscapes from: {diagnostics_dir}")

    results = analyze_all_trajectories(diagnostics_dir)

    if results is None:
        print("\n✗ Analysis failed: No trajectory data found")
        print("\nPossible causes:")
        print("  1. Evaluation didn't run with --enable_diagnostics")
        print("  2. Diagnostics directory is empty")
        print("  3. Wrong directory path")
        sys.exit(1)

    print_analysis(results)

    # Save detailed results
    output_file = diagnostics_dir.parent / "energy_landscape_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()
