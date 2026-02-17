#!/usr/bin/env python3
"""
Analyze what characteristics distinguish problems with correct vs inverted energy landscapes.
"""
import torch
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).parent))

from src.algebra.algebra_encoder import create_character_encoder
from src.algebra.algebra_dataset import AlgebraDataset
from src.algebra.algebra_inference import load_rule_models

def extract_coefficients(eq_str):
    """Extract coefficient magnitudes from equation."""
    # Find all numbers in the equation
    nums = [int(x) for x in re.findall(r'\d+', eq_str)]
    if not nums:
        return 0, 0, 0
    return min(nums), max(nums), sum(nums) / len(nums)

def main():
    print("="*80)
    print("PROBLEM PATTERN ANALYSIS")
    print("="*80)
    
    device = 'cpu'  # Use CPU for faster loading
    models = load_rule_models(['distribute'], './results', device)
    ebm_model = models['distribute'].model.ebm if hasattr(models['distribute'], 'model') else models['distribute'].ebm
    
    encoder = create_character_encoder(d_model=128)
    dataset = AlgebraDataset(rule='distribute', split='test', num_problems=200, d_model=128)
    
    timestep = torch.tensor([0.0], device=device)
    
    correct_problems = []
    inverted_problems = []
    
    print(f"\nTesting 200 problems...\n")
    
    for i in range(min(200, len(dataset))):
        input_eq, target_eq = dataset.get_equation_pair(i)
        input_emb = encoder(input_eq).to(device)
        target_emb = encoder(target_eq).to(device)
        
        with torch.no_grad():
            e_inp_inp = ebm_model(input_emb.unsqueeze(0), input_emb.unsqueeze(0), timestep).item()
            e_inp_tgt = ebm_model(input_emb.unsqueeze(0), target_emb.unsqueeze(0), timestep).item()
        
        gap = e_inp_tgt - e_inp_inp
        is_correct = gap < 0
        
        # Extract features
        min_coef, max_coef, avg_coef = extract_coefficients(input_eq)
        inp_len = len(input_eq)
        tgt_len = len(target_eq)
        
        problem_data = {
            'input': input_eq,
            'target': target_eq,
            'gap': gap,
            'e_inp_inp': e_inp_inp,
            'e_inp_tgt': e_inp_tgt,
            'min_coef': min_coef,
            'max_coef': max_coef,
            'avg_coef': avg_coef,
            'inp_len': inp_len,
            'tgt_len': tgt_len
        }
        
        if is_correct:
            correct_problems.append(problem_data)
        else:
            inverted_problems.append(problem_data)
    
    print(f"Results: {len(correct_problems)} correct, {len(inverted_problems)} inverted")
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Compare statistics
    def calc_stats(problems, key):
        values = [p[key] for p in problems]
        if not values:
            return 0, 0
        return sum(values) / len(values), max(values) - min(values)
    
    print(f"\n{'Metric':<20} {'Correct (avg)':<15} {'Inverted (avg)':<15} {'Difference':<15}")
    print("-"*80)
    
    for key in ['min_coef', 'max_coef', 'avg_coef', 'inp_len', 'tgt_len', 'e_inp_inp', 'e_inp_tgt']:
        correct_avg, _ = calc_stats(correct_problems, key)
        inverted_avg, _ = calc_stats(inverted_problems, key)
        diff = inverted_avg - correct_avg
        print(f"{key:<20} {correct_avg:<15.2f} {inverted_avg:<15.2f} {diff:+15.2f}")
    
    # Show example problems from each category
    print(f"\n{'='*80}")
    print("SAMPLE CORRECT PROBLEMS (first 5):")
    print("="*80)
    for i, p in enumerate(correct_problems[:5]):
        print(f"{i+1}. {p['input']} → {p['target']}")
        print(f"   Gap: {p['gap']:.4f}, Coefs: {p['min_coef']}-{p['max_coef']}")
    
    print(f"\n{'='*80}")
    print("SAMPLE INVERTED PROBLEMS (first 5):")
    print("="*80)
    for i, p in enumerate(inverted_problems[:5]):
        print(f"{i+1}. {p['input']} → {p['target']}")
        print(f"   Gap: {p['gap']:.4f}, Coefs: {p['min_coef']}-{p['max_coef']}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("="*80)
    print("Look for systematic differences in coefficient ranges, equation lengths,")
    print("or energy magnitudes that distinguish correct vs inverted problems.")

if __name__ == "__main__":
    main()
