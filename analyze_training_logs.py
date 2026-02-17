#!/usr/bin/env python3
"""
Analyze training logs for loss balance issues and energy statistics patterns.
"""
import re
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log and extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract energy monitoring lines
    energy_lines = re.findall(r'\[EnergyMonitor\] Average energy gap.*?: ([\d.]+), PosE=([\d.]+), NegE=([\d.]+)', content)
    
    # Extract loss balance lines
    balance_lines = re.findall(r'\[LossBalance\] Step (\d+): MSE=([\d.]+)%, Energy=([\d.]+)%, EnergyScale=([\d.]+), EnergyGap=([\d.]+)', content)
    
    return energy_lines, balance_lines

def main():
    print("="*80)
    print("TRAINING LOG ANALYSIS")
    print("="*80)
    
    # Find the distribute training log
    log_dir = Path('slurm/logs')
    log_files = list(log_dir.glob('*train*distribute*.out'))
    
    if not log_files:
        print("ERROR: No training logs found!")
        return
    
    log_file = sorted(log_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\nAnalyzing: {log_file.name}")
    
    energy_lines, balance_lines = parse_log_file(log_file)
    
    if not energy_lines:
        print("ERROR: No energy monitoring data found in log!")
        return
    
    print(f"\nFound {len(energy_lines)} energy monitoring points")
    print(f"Found {len(balance_lines)} loss balance points")
    
    # Analyze energy gap evolution
    print(f"\n{'='*80}")
    print("ENERGY GAP EVOLUTION")
    print("="*80)
    print(f"{'Point':<10} {'Gap':<10} {'PosE':<10} {'NegE':<10} {'Ratio':<10}")
    print("-"*80)
    
    gaps = []
    pos_energies = []
    neg_energies = []
    
    for i, (gap, pos_e, neg_e) in enumerate(energy_lines):
        gap_val = float(gap)
        pos_val = float(pos_e)
        neg_val = float(neg_e)
        
        gaps.append(gap_val)
        pos_energies.append(pos_val)
        neg_energies.append(neg_val)
        
        if i < 10 or i >= len(energy_lines) - 5:  # First 10 and last 5
            ratio = neg_val / pos_val if pos_val > 0 else 0
            print(f"{i:<10} {gap_val:<10.2f} {pos_val:<10.2f} {neg_val:<10.2f} {ratio:<10.2f}")
        elif i == 10:
            print("...")
    
    # Statistics
    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)
    
    avg_pos = sum(pos_energies) / len(pos_energies)
    avg_neg = sum(neg_energies) / len(neg_energies)
    
    print(f"\n{'='*80}")
    print("STATISTICS")
    print("="*80)
    print(f"Energy gap:  avg={avg_gap:.2f}, min={min_gap:.2f}, max={max_gap:.2f}")
    print(f"Pos energy:  avg={avg_pos:.2f}")
    print(f"Neg energy:  avg={avg_neg:.2f}")
    print(f"Expected ratio (neg/pos): {avg_neg/avg_pos:.2f} (target: ~15.0)")
    
    # Check for concerning patterns
    print(f"\n{'='*80}")
    print("DIAGNOSTICS")
    print("="*80)
    
    if avg_gap < 5.0:
        print("⚠️  Low energy gap - model may not be learning strong separation")
    else:
        print(f"✓ Energy gap looks reasonable ({avg_gap:.2f})")
    
    if avg_neg / avg_pos < 2.0:
        print("⚠️  Neg/Pos ratio too low - contrastive loss may not be working")
    else:
        print(f"✓ Neg/Pos ratio looks reasonable ({avg_neg/avg_pos:.2f})")
    
    # Check variance - high variance suggests instability
    gap_variance = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
    gap_std = gap_variance ** 0.5
    
    print(f"\nEnergy gap variance: std={gap_std:.2f}")
    if gap_std > avg_gap * 0.5:
        print("⚠️  HIGH VARIANCE - training may be unstable!")
    else:
        print("✓ Variance looks stable")

if __name__ == "__main__":
    main()
