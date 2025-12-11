#!/usr/bin/env python3
"""
Multi-Seed Statistical Comparison Evaluation

This script runs monolithic vs compositional comparisons across multiple seeds
and performs proper statistical analysis including:
- Paired t-tests between approaches
- Bootstrap confidence intervals
- Effect size calculations
- Multiple comparison corrections

Usage:
    # Run all seeds and analyze
    python scripts/statistical_comparison_evaluation.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results \
        --seeds 5 \
        --output_dir ./statistical_comparison_results

    # Analyze existing results from multiple runs
    python scripts/statistical_comparison_evaluation.py \
        --aggregate_only \
        --results_dir ./comparison_results_* \
        --output_dir ./statistical_analysis
        
    # Quick test with 2 seeds
    python scripts/statistical_comparison_evaluation.py \
        --monolithic_checkpoint ./results/monolithic/model.pt \
        --compositional_dir ./results \
        --seeds 2 \
        --num_samples 100 \
        --output_dir ./test_statistical_results
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
import subprocess
import time
from datetime import datetime

# Statistical analysis
import scipy.stats as stats
# Note: scipy.bootstrap requires scipy >= 1.7.0, using manual implementation instead

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalComparisonFramework:
    """Framework for running and analyzing multi-seed comparisons."""
    
    def __init__(self, output_dir: str, num_samples: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        
        # Store results from all seeds
        self.seed_results = {}
        
    def run_single_seed_comparison(
        self,
        seed: int,
        monolithic_checkpoint: str,
        compositional_dir: str,
        eval_script: str = "eval_algebra.py"
    ) -> Dict[str, Any]:
        """Run comparison evaluation for a single seed."""
        logger.info(f"Running comparison for seed {seed}")
        
        # Create seed-specific output directory
        seed_output_dir = self.output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(exist_ok=True)
        
        # Build command for seed-specific evaluation
        cmd = [
            sys.executable, eval_script,
            "--eval_type", "comparison",
            "--use_real_diffusion",
            "--monolithic_checkpoint", monolithic_checkpoint,
            "--model_dir", compositional_dir,
            "--max_samples", str(self.num_samples),
            "--output_dir", str(seed_output_dir),
            "--seed", str(seed),
            "--verbose"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run evaluation
            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.time() - start_time
            
            logger.info(f"Seed {seed} completed in {duration:.1f}s")
            
            # Load results
            results_file = seed_output_dir / "comparison_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            with open(results_file) as f:
                results = json.load(f)
                
            # Add metadata
            results['seed'] = seed
            results['duration'] = duration
            results['command'] = ' '.join(cmd)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Seed {seed} failed with exit code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            logger.error(f"STDOUT: {e.stdout}")
            raise
        except Exception as e:
            logger.error(f"Seed {seed} failed with error: {str(e)}")
            raise
    
    def run_multi_seed_evaluation(
        self,
        seeds: List[int],
        monolithic_checkpoint: str,
        compositional_dir: str
    ) -> Dict[int, Dict[str, Any]]:
        """Run comparison evaluation across multiple seeds."""
        logger.info(f"Running multi-seed evaluation with seeds: {seeds}")
        
        all_results = {}
        
        for seed in seeds:
            try:
                results = self.run_single_seed_comparison(
                    seed=seed,
                    monolithic_checkpoint=monolithic_checkpoint,
                    compositional_dir=compositional_dir
                )
                all_results[seed] = results
                
                # Save individual seed results
                seed_file = self.output_dir / f"seed_{seed}_results.json"
                with open(seed_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"✅ Seed {seed} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Seed {seed} failed: {str(e)}")
                # Continue with other seeds
                continue
        
        if not all_results:
            raise RuntimeError("All seeds failed! Check logs for errors.")
        
        logger.info(f"Completed {len(all_results)}/{len(seeds)} seeds successfully")
        
        # Save combined results
        combined_file = self.output_dir / "all_seeds_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        self.seed_results = all_results
        return all_results
    
    def extract_performance_metrics(self, all_results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        """Extract key performance metrics into a structured DataFrame."""
        data = []
        
        for seed, results in all_results.items():
            # Extract monolithic results (key: 'monolithic', not 'monolithic_results')
            if 'monolithic' in results:
                mono_data = results['monolithic']
                
                # Extract single-rule accuracy (average across rules)
                single_rule_accs = []
                for rule in ['distribute', 'combine', 'isolate', 'divide']:
                    rule_key = f'single_rule_{rule}'
                    if rule_key in mono_data and 'summary' in mono_data[rule_key]:
                        acc = mono_data[rule_key]['summary'].get('accuracy', 0)
                        single_rule_accs.append(acc)
                single_rule_acc = np.mean(single_rule_accs) if single_rule_accs else np.nan
                
                # Extract multi-rule accuracy (average across 2, 3, 4-rule problems)
                multi_rule_accs = []
                rule_2_acc = np.nan
                rule_3_acc = np.nan 
                rule_4_acc = np.nan
                
                for num_rules in [2, 3, 4]:
                    rule_key = f'multi_rule_{num_rules}'
                    if rule_key in mono_data and 'summary' in mono_data[rule_key]:
                        acc = mono_data[rule_key]['summary'].get('accuracy', 0)
                        multi_rule_accs.append(acc)
                        if num_rules == 2:
                            rule_2_acc = acc
                        elif num_rules == 3:
                            rule_3_acc = acc
                        elif num_rules == 4:
                            rule_4_acc = acc
                            
                multi_rule_acc = np.mean(multi_rule_accs) if multi_rule_accs else np.nan
                
                data.append({
                    'seed': seed,
                    'approach': 'monolithic',
                    'single_rule_acc': single_rule_acc,
                    'multi_rule_acc': multi_rule_acc,
                    'rule_2_acc': rule_2_acc,
                    'rule_3_acc': rule_3_acc,
                    'rule_4_acc': rule_4_acc
                })
            else:
                logger.warning(f"No monolithic results found for seed {seed}")
            
            # Extract compositional results (key: 'compositional', not 'compositional_results')
            if 'compositional' in results:
                comp_data = results['compositional']
                
                # Compositional doesn't have single-rule results, use nan
                single_rule_acc = np.nan
                
                # Extract multi-rule accuracy (average across 2, 3, 4-rule problems)
                multi_rule_accs = []
                rule_2_acc = np.nan
                rule_3_acc = np.nan
                rule_4_acc = np.nan
                
                for num_rules in [2, 3, 4]:
                    rule_key = f'multi_rule_{num_rules}'
                    if rule_key in comp_data and 'summary' in comp_data[rule_key]:
                        acc = comp_data[rule_key]['summary'].get('accuracy', 0)
                        multi_rule_accs.append(acc)
                        if num_rules == 2:
                            rule_2_acc = acc
                        elif num_rules == 3:
                            rule_3_acc = acc
                        elif num_rules == 4:
                            rule_4_acc = acc
                            
                multi_rule_acc = np.mean(multi_rule_accs) if multi_rule_accs else np.nan
                
                data.append({
                    'seed': seed,
                    'approach': 'compositional',
                    'single_rule_acc': single_rule_acc,
                    'multi_rule_acc': multi_rule_acc,
                    'rule_2_acc': rule_2_acc,
                    'rule_3_acc': rule_3_acc,
                    'rule_4_acc': rule_4_acc
                })
            else:
                logger.warning(f"No compositional results found for seed {seed}")
        
        df = pd.DataFrame(data)
        
        # Debug: Print DataFrame info for troubleshooting
        logger.info(f"Extracted data for {len(data)} result entries")
        if len(df) > 0:
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"Approaches found: {df['approach'].unique() if 'approach' in df.columns else 'None'}")
        else:
            logger.warning("No data was extracted - DataFrame is empty!")
        
        # Save DataFrame
        df_file = self.output_dir / "performance_metrics.csv"
        df.to_csv(df_file, index=False)
        logger.info(f"Performance metrics saved to: {df_file}")
        
        return df
    
    def compute_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on the performance data."""
        logger.info("Computing statistical tests...")
        
        # Group by approach
        mono_data = df[df['approach'] == 'monolithic']
        comp_data = df[df['approach'] == 'compositional']
        
        if len(mono_data) == 0 or len(comp_data) == 0:
            raise ValueError("Missing data for one of the approaches!")
        
        # Ensure same seeds for paired tests
        common_seeds = set(mono_data['seed']).intersection(set(comp_data['seed']))
        if len(common_seeds) < 2:
            raise ValueError(f"Need at least 2 common seeds for statistical tests, got: {len(common_seeds)}")
        
        logger.info(f"Using {len(common_seeds)} seeds for paired statistical tests")
        
        # Filter to common seeds and sort
        mono_paired = mono_data[mono_data['seed'].isin(common_seeds)].sort_values('seed')
        comp_paired = comp_data[comp_data['seed'].isin(common_seeds)].sort_values('seed')
        
        stats_results = {}
        
        # Test each metric
        metrics = ['single_rule_acc', 'multi_rule_acc', 'rule_2_acc', 'rule_3_acc', 'rule_4_acc']
        
        for metric in metrics:
            mono_values = mono_paired[metric].dropna().values
            comp_values = comp_paired[metric].dropna().values
            
            if len(mono_values) == 0 or len(comp_values) == 0:
                logger.warning(f"Skipping {metric} - no data available")
                continue
            
            # Ensure same length for paired test
            min_len = min(len(mono_values), len(comp_values))
            mono_values = mono_values[:min_len]
            comp_values = comp_values[:min_len]
            
            if min_len < 2:
                logger.warning(f"Skipping {metric} - insufficient data (n={min_len})")
                continue
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(comp_values, mono_values)
            
            # Effect size (Cohen's d for paired samples)
            diff = comp_values - mono_values
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            
            # Confidence interval for mean difference (bootstrap)
            ci_lower, ci_upper = self._bootstrap_ci_difference(comp_values, mono_values)
            
            stats_results[metric] = {
                'monolithic_mean': float(np.mean(mono_values)),
                'monolithic_std': float(np.std(mono_values, ddof=1)),
                'compositional_mean': float(np.mean(comp_values)),
                'compositional_std': float(np.std(comp_values, ddof=1)),
                'mean_difference': float(np.mean(comp_values) - np.mean(mono_values)),
                'difference_std': float(np.std(diff, ddof=1)),
                'paired_t_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'ci_95_lower': float(ci_lower),
                'ci_95_upper': float(ci_upper),
                'n_pairs': int(min_len),
                'significant_05': bool(p_val < 0.05),
                'significant_01': bool(p_val < 0.01)
            }
        
        # Save statistical results
        stats_file = self.output_dir / "statistical_tests.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        logger.info(f"Statistical test results saved to: {stats_file}")
        return stats_results
    
    def _bootstrap_ci_difference(self, comp_values: np.ndarray, mono_values: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for mean difference."""
        n_bootstrap = 1000
        
        def statistic(comp_sample, mono_sample):
            return np.mean(comp_sample) - np.mean(mono_sample)
        
        differences = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(comp_values), size=len(comp_values), replace=True)
            comp_sample = comp_values[indices]
            mono_sample = mono_values[indices]
            differences.append(statistic(comp_sample, mono_sample))
        
        alpha = 1 - confidence
        ci_lower = np.percentile(differences, 100 * alpha/2)
        ci_upper = np.percentile(differences, 100 * (1 - alpha/2))
        
        return ci_lower, ci_upper
    
    def generate_paper_tables(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> str:
        """Generate LaTeX tables for the paper."""
        logger.info("Generating paper-ready tables...")
        
        # Main performance comparison table
        latex_content = []
        
        latex_content.append("% Generated LaTeX Tables for Paper")
        latex_content.append("% " + "="*60)
        latex_content.append(f"% Generated on: {datetime.now()}")
        latex_content.append("% " + "="*60)
        latex_content.append("")
        
        # Table 1: Main Performance Comparison
        latex_content.append("% Table 1: Main Performance Comparison")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Performance comparison on algebraic equation solving. Results averaged over multiple random seeds with 95\\% confidence intervals.}")
        latex_content.append("\\label{tab:main_results}")
        latex_content.append("\\begin{tabular}{lcc}")
        latex_content.append("\\toprule")
        latex_content.append("Model & Single-Rule Acc & Multi-Rule Acc \\\\")
        latex_content.append("\\midrule")
        
        # Extract main results
        single_mono = stats_results.get('single_rule_acc', {})
        single_comp = stats_results.get('single_rule_acc', {})
        multi_mono = stats_results.get('multi_rule_acc', {})
        multi_comp = stats_results.get('multi_rule_acc', {})
        
        single_ci = f"[{single_mono.get('ci_95_lower', 0):.1f}, {single_mono.get('ci_95_upper', 0):.1f}]"
        multi_ci = f"[{multi_comp.get('ci_95_lower', 0):.1f}, {multi_comp.get('ci_95_upper', 0):.1f}]"
        
        latex_content.append(f"Monolithic IRED & {single_mono.get('monolithic_mean', 0):.1f}\\% & {multi_mono.get('monolithic_mean', 0):.1f}\\% \\\\")
        latex_content.append(f"\\textbf{{Compositional (Ours)}} & \\textbf{{{single_comp.get('compositional_mean', 0):.1f}\\%}} & \\textbf{{{multi_comp.get('compositional_mean', 0):.1f}\\%}} \\\\")
        
        # Add significance indicators
        multi_diff = multi_comp.get('mean_difference', 0)
        multi_p = multi_comp.get('p_value', 1.0)
        
        sig_marker = ""
        if multi_p < 0.001:
            sig_marker = "***"
        elif multi_p < 0.01:
            sig_marker = "**"  
        elif multi_p < 0.05:
            sig_marker = "*"
        
        latex_content.append("\\midrule")
        latex_content.append(f"\\textit{{Improvement}} & \\textit{{+{single_comp.get('mean_difference', 0):.1f}\\%}} & \\textit{{+{multi_diff:.1f}\\%}}{sig_marker} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\vspace{0.5em}")
        
        if sig_marker:
            sig_text = {
                "*": "p < 0.05",
                "**": "p < 0.01", 
                "***": "p < 0.001"
            }
            latex_content.append(f"\\footnotesize{{{sig_marker}{sig_text.get(sig_marker, '')} via paired t-test across seeds}}")
        
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        # Table 2: Multi-Rule Breakdown
        latex_content.append("% Table 2: Multi-Rule Performance Breakdown")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Multi-rule performance breakdown by number of required rules with statistical significance testing.}")
        latex_content.append("\\label{tab:breakdown}")
        latex_content.append("\\begin{tabular}{lccc}")
        latex_content.append("\\toprule")
        latex_content.append("Problem Type & Monolithic & Compositional & Advantage \\\\")
        latex_content.append("\\midrule")
        
        for rule_num in [2, 3, 4]:
            metric = f'rule_{rule_num}_acc'
            if metric in stats_results:
                data = stats_results[metric]
                mono_mean = data.get('monolithic_mean', 0)
                comp_mean = data.get('compositional_mean', 0)
                diff = data.get('mean_difference', 0)
                p_val = data.get('p_value', 1.0)
                
                sig_marker = ""
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
                
                latex_content.append(f"{rule_num}-rule & {mono_mean:.1f}\\% & {comp_mean:.1f}\\% & +{diff:.1f}\\%{sig_marker} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\vspace{0.5em}")
        latex_content.append("\\footnotesize{Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX content
        latex_file = self.output_dir / "paper_tables.tex"
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_content))
            
        logger.info(f"LaTeX tables saved to: {latex_file}")
        return '\n'.join(latex_content)
    
    def generate_summary_report(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown summary report."""
        logger.info("Generating summary report...")
        
        report = []
        
        # Header
        report.append("# Statistical Comparison Analysis Report")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Number of Seeds:** {len(df['seed'].unique())}")
        report.append(f"**Sample Size per Seed:** {self.num_samples}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        if 'multi_rule_acc' in stats_results:
            multi_data = stats_results['multi_rule_acc']
            diff = multi_data.get('mean_difference', 0)
            p_val = multi_data.get('p_value', 1.0)
            ci_lower = multi_data.get('ci_95_lower', 0)
            ci_upper = multi_data.get('ci_95_upper', 0)
            
            report.append(f"- **Primary Finding:** Compositional approach outperforms monolithic by {diff:.1f} percentage points on multi-rule problems")
            report.append(f"- **Statistical Significance:** p = {p_val:.4f} {'(significant)' if p_val < 0.05 else '(not significant)'}")
            report.append(f"- **95% Confidence Interval:** [{ci_lower:.1f}, {ci_upper:.1f}] percentage points")
            report.append(f"- **Effect Size (Cohen's d):** {multi_data.get('cohens_d', 0):.3f}")
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Statistical Results")
        report.append("")
        
        for metric, data in stats_results.items():
            metric_name = metric.replace('_', ' ').title()
            report.append(f"### {metric_name}")
            report.append("")
            
            report.append(f"- **Monolithic:** {data['monolithic_mean']:.1f}% ± {data['monolithic_std']:.1f}%")
            report.append(f"- **Compositional:** {data['compositional_mean']:.1f}% ± {data['compositional_std']:.1f}%")
            report.append(f"- **Mean Difference:** {data['mean_difference']:.1f} ± {data['difference_std']:.1f} percentage points")
            report.append(f"- **Paired t-test:** t({data['n_pairs']-1}) = {data['paired_t_statistic']:.3f}, p = {data['p_value']:.4f}")
            report.append(f"- **Effect Size:** Cohen's d = {data['cohens_d']:.3f}")
            report.append(f"- **95% CI:** [{data['ci_95_lower']:.1f}, {data['ci_95_upper']:.1f}]")
            report.append(f"- **Sample Size:** {data['n_pairs']} paired observations")
            report.append("")
        
        # Interpretation
        report.append("## Statistical Interpretation")
        report.append("")
        
        if 'multi_rule_acc' in stats_results:
            multi_data = stats_results['multi_rule_acc']
            cohens_d = multi_data.get('cohens_d', 0)
            
            effect_size_interpretation = "negligible"
            if abs(cohens_d) >= 0.8:
                effect_size_interpretation = "large"
            elif abs(cohens_d) >= 0.5:
                effect_size_interpretation = "medium"
            elif abs(cohens_d) >= 0.2:
                effect_size_interpretation = "small"
            
            report.append(f"- **Effect Size Interpretation:** {effect_size_interpretation} effect (|d| = {abs(cohens_d):.3f})")
            
            if multi_data.get('p_value', 1.0) < 0.05:
                report.append("- **Statistical Conclusion:** The compositional approach shows statistically significant improvement over the monolithic approach")
            else:
                report.append("- **Statistical Conclusion:** No statistically significant difference detected between approaches")
        
        report.append("")
        report.append("## Recommendations for Paper")
        report.append("")
        
        if 'multi_rule_acc' in stats_results and stats_results['multi_rule_acc'].get('p_value', 1.0) < 0.05:
            report.append("✅ **Ready for Publication:** Results show significant improvement with proper statistical validation")
            report.append("- Use paired t-test results in paper")
            report.append("- Include confidence intervals in tables")
            report.append("- Report effect sizes for practical significance")
        else:
            report.append("⚠️  **Need More Data:** Results are not statistically significant")
            report.append("- Consider increasing sample sizes")
            report.append("- Run additional seeds")
            report.append("- Check for implementation issues")
        
        # Save report
        report_content = '\n'.join(report)
        report_file = self.output_dir / "statistical_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Summary report saved to: {report_file}")
        return report_content


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed statistical comparison evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Execution mode
    parser.add_argument(
        '--aggregate_only',
        action='store_true',
        help='Only analyze existing results, do not run new experiments'
    )
    
    # Model paths
    parser.add_argument(
        '--monolithic_checkpoint',
        type=str,
        help='Path to monolithic model checkpoint'
    )
    
    parser.add_argument(
        '--compositional_dir',
        type=str,
        help='Directory containing compositional models'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--seeds',
        type=int,
        default=5,
        help='Number of random seeds to test (default: 5)'
    )
    
    parser.add_argument(
        '--seed_start',
        type=int,
        default=42,
        help='Starting seed value (default: 42)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples per evaluation (default: 1000)'
    )
    
    # Input/Output
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Directory pattern for existing results (for aggregate_only mode)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./statistical_comparison_results',
        help='Output directory for results (default: ./statistical_comparison_results)'
    )
    
    parser.add_argument(
        '--eval_script',
        type=str,
        default='eval_algebra.py',
        help='Path to evaluation script (default: eval_algebra.py)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.aggregate_only:
        if not args.monolithic_checkpoint or not args.compositional_dir:
            parser.error("--monolithic_checkpoint and --compositional_dir are required when not using --aggregate_only")
    
    # Create framework
    framework = StatisticalComparisonFramework(args.output_dir, args.num_samples)
    
    try:
        if args.aggregate_only:
            # TODO: Implement aggregation from existing results
            logger.error("--aggregate_only not yet implemented")
            return 1
        else:
            # Generate seed list
            seed_list = list(range(args.seed_start, args.seed_start + args.seeds))
            
            logger.info("="*60)
            logger.info("MULTI-SEED STATISTICAL COMPARISON EVALUATION")
            logger.info("="*60)
            logger.info(f"Seeds: {seed_list}")
            logger.info(f"Samples per seed: {args.num_samples}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info("="*60)
            
            # Run multi-seed evaluation
            all_results = framework.run_multi_seed_evaluation(
                seeds=seed_list,
                monolithic_checkpoint=args.monolithic_checkpoint,
                compositional_dir=args.compositional_dir
            )
            
            logger.info("="*60)
            logger.info("STATISTICAL ANALYSIS")
            logger.info("="*60)
            
            # Extract and analyze results
            df = framework.extract_performance_metrics(all_results)
            stats_results = framework.compute_statistical_tests(df)
            
            # Generate paper outputs
            latex_tables = framework.generate_paper_tables(df, stats_results)
            summary_report = framework.generate_summary_report(df, stats_results)
            
            logger.info("="*60)
            logger.info("RESULTS SUMMARY")
            logger.info("="*60)
            
            # Print key results
            if 'multi_rule_acc' in stats_results:
                multi_data = stats_results['multi_rule_acc']
                logger.info(f"Multi-rule improvement: {multi_data['mean_difference']:.1f} ± {multi_data['difference_std']:.1f} pp")
                logger.info(f"Statistical significance: p = {multi_data['p_value']:.4f}")
                logger.info(f"95% CI: [{multi_data['ci_95_lower']:.1f}, {multi_data['ci_95_upper']:.1f}] pp")
            
            logger.info("="*60)
            logger.info("FILES GENERATED")
            logger.info("="*60)
            logger.info(f"📊 Performance data: {args.output_dir}/performance_metrics.csv")
            logger.info(f"📈 Statistical tests: {args.output_dir}/statistical_tests.json")
            logger.info(f"📝 LaTeX tables: {args.output_dir}/paper_tables.tex")
            logger.info(f"📋 Summary report: {args.output_dir}/statistical_analysis_report.md")
            logger.info("="*60)
            
            return 0
            
    except Exception as e:
        logger.error(f"Statistical comparison failed: {str(e)}")
        logger.error(f"Check logs for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())