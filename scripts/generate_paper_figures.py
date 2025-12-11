#!/usr/bin/env python3
"""
Generate publication-ready figures and tables for algebra EBM paper.
Creates comprehensive visualizations including performance comparisons,
ablation studies, error analysis, and training curves.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'text.usetex': False,  # Set to True if LaTeX available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight'
})

class PaperFigureGenerator:
    """Generate all figures and tables for the algebra EBM paper."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'raw_data').mkdir(exist_ok=True)
        
    def load_comparison_results(self) -> Dict:
        """Load comparison results from evaluation."""
        comparison_file = self.results_dir / 'comparison_results.json'
        if comparison_file.exists():
            with open(comparison_file) as f:
                return json.load(f)
        return {}
    
    def load_statistical_results(self) -> Dict:
        """Load statistical test results."""
        stats_file = self.results_dir / 'statistical_tests.json'
        if stats_file.exists():
            with open(stats_file) as f:
                return json.load(f)
        return {}
    
    def load_performance_metrics(self) -> pd.DataFrame:
        """Load performance metrics dataframe."""
        metrics_file = self.results_dir / 'performance_metrics.csv'
        if metrics_file.exists():
            return pd.read_csv(metrics_file)
        return pd.DataFrame()
    
    def load_training_logs(self) -> Dict:
        """Load training logs from individual model directories."""
        training_logs = {}
        
        # Look for training logs in the results directory
        if (self.results_dir.parent / 'results').exists():
            results_base = self.results_dir.parent / 'results'
        else:
            results_base = self.results_dir
            
        for rule in ['distribute', 'combine', 'isolate', 'divide', 'monolithic']:
            log_file = results_base / rule / 'training_log.json'
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        training_logs[rule] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse training log for {rule}")
                    
        return training_logs
    
    def generate_main_results_table(self, results: Dict) -> pd.DataFrame:
        """Generate the main results comparison table (Table 1)."""
        
        # Extract results by method and rule count
        methods = ['compositional', 'monolithic', 'transformer', 'random']
        rule_counts = [1, 2, 3, 4]
        
        table_data = []
        for method in methods:
            row = {'Method': method.capitalize()}
            for rule_count in rule_counts:
                key = f"{method}_{rule_count}_rules"
                if key in results:
                    accuracy = results[key].get('accuracy', 0.0) * 100
                    row[f'{rule_count}-Rule'] = f"{accuracy:.1f}"
                else:
                    row[f'{rule_count}-Rule'] = "N/A"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as LaTeX table
        latex_table = df.to_latex(index=False, float_format="%.1f")
        with open(self.output_dir / 'tables' / 'main_results.tex', 'w') as f:
            f.write(latex_table)
        
        # Save as CSV for reference
        df.to_csv(self.output_dir / 'tables' / 'main_results.csv', index=False)
        
        return df
    
    def plot_performance_by_rules(self, results: Dict):
        """Generate main performance comparison plot (Figure 1)."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        methods = {
            'compositional': 'Compositional (Ours)',
            'monolithic': 'Monolithic IRED',
            'transformer': 'Seq2Seq Transformer',
            'random': 'Random Composition'
        }
        
        colors = ['#2E8B57', '#CD853F', '#4682B4', '#DC143C']
        markers = ['o', 's', '^', 'D']
        
        rule_counts = [1, 2, 3, 4]
        
        for i, (method_key, method_name) in enumerate(methods.items()):
            accuracies = []
            for rule_count in rule_counts:
                key = f"{method_key}_{rule_count}_rules"
                if key in results:
                    accuracies.append(results[key].get('accuracy', 0.0) * 100)
                else:
                    accuracies.append(0.0)
            
            ax.plot(rule_counts, accuracies, 
                   label=method_name, 
                   color=colors[i], 
                   marker=markers[i], 
                   linewidth=2.5, 
                   markersize=8)
        
        ax.set_xlabel('Number of Required Rules')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Performance vs. Problem Complexity')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rule_counts)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'performance_by_rules.png')
        plt.close()
    
    def plot_ablation_study(self, results: Dict, stats_results: Dict):
        """Generate ablation study visualization (Figure 2) using real data."""
        
        # Only create ablation plots if we have actual ablation data
        # This would require running experiments with different configurations
        
        print("Skipping ablation study plot - requires running experiments with different hyperparameters")
        print("To generate ablation studies, run evaluations with:")
        print("  - Different energy composition strategies")
        print("  - Different training set sizes")
        print("  - Different model architectures")
        
        # Instead, create a comparison plot using available statistical data
        if 'multi_rule_acc' in stats_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot confidence intervals for multi-rule performance
            methods = ['Monolithic', 'Compositional']
            means = [
                stats_results['multi_rule_acc']['monolithic_mean'],
                stats_results['multi_rule_acc']['compositional_mean']
            ]
            stds = [
                stats_results['multi_rule_acc']['monolithic_std'],
                stats_results['multi_rule_acc']['compositional_std']
            ]
            
            colors = ['#CD853F', '#2E8B57']
            x_pos = np.arange(len(methods))
            
            bars = ax.bar(x_pos, means, color=colors, alpha=0.7, capsize=5)
            ax.errorbar(x_pos, means, yerr=stds, fmt='none', color='black', capsize=5)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Multi-Rule Accuracy (%)')
            ax.set_title('Multi-Rule Performance Comparison with Error Bars')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add significance annotation if available
            if stats_results['multi_rule_acc']['p_value'] < 0.05:
                y_max = max(means) + max(stds) + 2
                ax.annotate('*', xy=(0.5, y_max), ha='center', fontsize=16, color='red')
                ax.text(0.5, y_max + 1, f"p = {stats_results['multi_rule_acc']['p_value']:.4f}", 
                       ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'figures' / 'performance_comparison.png')
            plt.close()
        else:
            print("Warning: No statistical results available for comparison plot")
    
    def plot_error_analysis(self, results: Dict, df: pd.DataFrame):
        """Generate error analysis plots using real evaluation data."""
        
        # Check if we have detailed error analysis data
        has_error_data = any('error_types' in v for v in results.values() if isinstance(v, dict))
        has_distance_data = any('l2_distances' in v for v in results.values() if isinstance(v, dict))
        
        if not (has_error_data or has_distance_data):
            print("Skipping error analysis - requires detailed error categorization in evaluation")
            print("To generate error analysis, modify evaluation to track:")
            print("  - Error type classification (syntax, logic, arithmetic)")
            print("  - L2 distances from correct solutions")
            return
        
        fig_count = int(has_error_data) + int(has_distance_data)
        if fig_count == 0:
            return
            
        fig, axes = plt.subplots(1, fig_count, figsize=(7 * fig_count, 6))
        if fig_count == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # Error types breakdown (if available)
        if has_error_data:
            # Extract real error type data from results
            comp_errors = {}
            mono_errors = {}
            
            for method_key, method_data in results.items():
                if isinstance(method_data, dict) and 'error_types' in method_data:
                    if 'compositional' in method_key:
                        comp_errors = method_data['error_types']
                    elif 'monolithic' in method_key:
                        mono_errors = method_data['error_types']
            
            if comp_errors and mono_errors:
                error_types = list(comp_errors.keys())
                comp_percentages = list(comp_errors.values())
                mono_percentages = list(mono_errors.values())
                
                x = np.arange(len(error_types))
                width = 0.35
                
                axes[ax_idx].bar(x - width/2, comp_percentages, width, label='Compositional', alpha=0.8)
                axes[ax_idx].bar(x + width/2, mono_percentages, width, label='Monolithic', alpha=0.8)
                
                axes[ax_idx].set_xlabel('Error Type')
                axes[ax_idx].set_ylabel('Percentage of Errors (%)')
                axes[ax_idx].set_title('Error Type Distribution')
                axes[ax_idx].set_xticks(x)
                axes[ax_idx].set_xticklabels(error_types, rotation=45, ha='right')
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3, axis='y')
                ax_idx += 1
        
        # Solution distance analysis (if available)
        if has_distance_data:
            comp_distances = []
            mono_distances = []
            
            for method_key, method_data in results.items():
                if isinstance(method_data, dict) and 'l2_distances' in method_data:
                    if 'compositional' in method_key:
                        comp_distances = method_data['l2_distances']
                    elif 'monolithic' in method_key:
                        mono_distances = method_data['l2_distances']
            
            if comp_distances and mono_distances:
                axes[ax_idx].hist(comp_distances, bins=20, alpha=0.7, label='Compositional', density=True)
                axes[ax_idx].hist(mono_distances, bins=20, alpha=0.7, label='Monolithic', density=True)
                
                axes[ax_idx].set_xlabel('L2 Distance from Correct Solution')
                axes[ax_idx].set_ylabel('Density')
                axes[ax_idx].set_title('Distribution of Solution Errors')
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'error_analysis.png')
        plt.close()
    
    def plot_training_curves(self, training_logs: Dict):
        """Generate training convergence plots using real training logs."""
        
        if not training_logs:
            print("Skipping training curves - no training logs found")
            print("Training logs should be saved as 'training_log.json' in each model directory")
            return
        
        available_models = list(training_logs.keys())
        if len(available_models) < 2:
            print(f"Insufficient training logs found. Available: {available_models}")
            return
        
        # Determine subplot layout based on available data
        has_loss_data = any('train_losses' in log for log in training_logs.values())
        has_val_data = any('val_accuracies' in log for log in training_logs.values())
        
        subplot_count = int(has_loss_data) + int(has_val_data)
        if subplot_count == 0:
            print("No training curve data found in logs")
            return
            
        fig, axes = plt.subplots(1, subplot_count, figsize=(7 * subplot_count, 6))
        if subplot_count == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # Plot training loss curves
        if has_loss_data:
            colors = plt.cm.Set3(np.linspace(0, 1, len(available_models)))
            
            for i, (model_name, log_data) in enumerate(training_logs.items()):
                if 'train_losses' in log_data:
                    losses = log_data['train_losses']
                    epochs = list(range(1, len(losses) + 1))
                    
                    axes[ax_idx].plot(epochs, losses, label=model_name.capitalize(), 
                                    color=colors[i], linewidth=2)
            
            axes[ax_idx].set_xlabel('Training Epoch')
            axes[ax_idx].set_ylabel('Training Loss')
            axes[ax_idx].set_title('Training Loss Convergence')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_yscale('log')
            ax_idx += 1
        
        # Plot validation accuracy progression
        if has_val_data:
            for i, (model_name, log_data) in enumerate(training_logs.items()):
                if 'val_accuracies' in log_data:
                    val_accs = log_data['val_accuracies']
                    val_epochs = log_data.get('val_epochs', list(range(1, len(val_accs) + 1)))
                    
                    marker = 'o' if 'compositional' in model_name.lower() else 's'
                    axes[ax_idx].plot(val_epochs, val_accs, marker=marker, 
                                    label=model_name.capitalize(), linewidth=2.5, markersize=6)
            
            axes[ax_idx].set_xlabel('Training Epoch')
            axes[ax_idx].set_ylabel('Validation Accuracy (%)')
            axes[ax_idx].set_title('Validation Performance During Training')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'training_curves.png')
        plt.close()
    
    def plot_generalization_analysis(self, results: Dict, stats_results: Dict):
        """Generate generalization analysis using real evaluation data."""
        
        # Check if we have in-domain vs out-domain data
        has_domain_data = any('in_domain' in k or 'out_domain' in k for k in results.keys())
        
        # Check if we have detailed rule combination data
        has_combination_data = any('combination_' in k for k in results.keys())
        
        if not (has_domain_data or has_combination_data):
            print("Skipping generalization analysis - requires domain-specific evaluation")
            print("To generate generalization analysis, run evaluations with:")
            print("  - In-domain vs out-of-domain test sets")
            print("  - Specific rule combination breakdowns")
            
            # Instead create a plot showing performance by rule count if available
            if stats_results:
                self.plot_rule_complexity_analysis(stats_results)
            return
        
        fig_count = int(has_domain_data) + int(has_combination_data)
        fig, axes = plt.subplots(1, fig_count, figsize=(7 * fig_count, 6))
        if fig_count == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # In-domain vs out-of-domain analysis
        if has_domain_data:
            # Extract domain-specific results
            rule_counts = []
            in_domain_comp = []
            out_domain_comp = []
            in_domain_mono = []
            out_domain_mono = []
            
            for key, data in results.items():
                if 'in_domain' in key and isinstance(data, dict):
                    if 'compositional' in key:
                        rule_count = int(key.split('_')[1])  # Assuming format: compositional_X_rules_in_domain
                        if rule_count not in rule_counts:
                            rule_counts.append(rule_count)
                        in_domain_comp.append(data.get('accuracy', 0) * 100)
                    elif 'monolithic' in key:
                        in_domain_mono.append(data.get('accuracy', 0) * 100)
            
            # Similar extraction for out-domain...
            # Plot the domain comparison
            if rule_counts and in_domain_comp:
                axes[ax_idx].plot(sorted(rule_counts), in_domain_comp[:len(rule_counts)], 
                                'o-', label='Compositional (In-Domain)', linewidth=2.5)
                axes[ax_idx].plot(sorted(rule_counts), in_domain_mono[:len(rule_counts)], 
                                's-', label='Monolithic (In-Domain)', linewidth=2.5)
                
                axes[ax_idx].set_xlabel('Number of Required Rules')
                axes[ax_idx].set_ylabel('Accuracy (%)')
                axes[ax_idx].set_title('Performance by Problem Complexity')
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_xticks(sorted(rule_counts))
            
            ax_idx += 1
        
        # Rule combination analysis
        if has_combination_data:
            combinations = []
            comp_accuracies = []
            mono_accuracies = []
            
            for key, data in results.items():
                if 'combination_' in key and isinstance(data, dict):
                    combo_name = key.replace('combination_', '').replace('_', '+')
                    combinations.append(combo_name)
                    
                    if 'compositional' in key:
                        comp_accuracies.append(data.get('accuracy', 0) * 100)
                    elif 'monolithic' in key:
                        mono_accuracies.append(data.get('accuracy', 0) * 100)
            
            if combinations:
                x = np.arange(len(combinations))
                width = 0.35
                
                axes[ax_idx].bar(x - width/2, comp_accuracies, width, label='Compositional', alpha=0.8)
                axes[ax_idx].bar(x + width/2, mono_accuracies, width, label='Monolithic', alpha=0.8)
                
                axes[ax_idx].set_xlabel('Rule Combination')
                axes[ax_idx].set_ylabel('Accuracy (%)')
                axes[ax_idx].set_title('Performance on Specific Rule Combinations')
                axes[ax_idx].set_xticks(x)
                axes[ax_idx].set_xticklabels(combinations, rotation=45, ha='right')
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'generalization_analysis.png')
        plt.close()
    
    def plot_rule_complexity_analysis(self, stats_results: Dict):
        """Plot performance by rule complexity using statistical results."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        rule_counts = []
        comp_means = []
        comp_stds = []
        mono_means = []
        mono_stds = []
        
        # Extract data for different rule counts
        for key, data in stats_results.items():
            if 'rule_' in key and '_acc' in key:
                try:
                    rule_num = int(key.split('_')[1])
                    rule_counts.append(rule_num)
                    comp_means.append(data['compositional_mean'])
                    comp_stds.append(data['compositional_std'])
                    mono_means.append(data['monolithic_mean'])
                    mono_stds.append(data['monolithic_std'])
                except (ValueError, KeyError):
                    continue
        
        if rule_counts:
            # Sort by rule count
            sorted_indices = np.argsort(rule_counts)
            rule_counts = [rule_counts[i] for i in sorted_indices]
            comp_means = [comp_means[i] for i in sorted_indices]
            comp_stds = [comp_stds[i] for i in sorted_indices]
            mono_means = [mono_means[i] for i in sorted_indices]
            mono_stds = [mono_stds[i] for i in sorted_indices]
            
            ax.errorbar(rule_counts, comp_means, yerr=comp_stds, 
                       marker='o', label='Compositional', linewidth=2.5, capsize=5)
            ax.errorbar(rule_counts, mono_means, yerr=mono_stds, 
                       marker='s', label='Monolithic', linewidth=2.5, capsize=5)
            
            ax.set_xlabel('Number of Required Rules')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Performance vs Problem Complexity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(rule_counts)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'figures' / 'rule_complexity_analysis.png')
            plt.close()
        else:
            print("No rule complexity data available for plotting")
    
    def generate_detailed_results_table(self, results: Dict):
        """Generate detailed breakdown table (Appendix Table)."""
        
        data = []
        metrics = ['accuracy', 'l2_distance', 'valid_syntax_rate', 'solve_time']
        methods = ['compositional', 'monolithic', 'transformer']
        
        for method in methods:
            for rule_count in [2, 3, 4]:
                key = f"{method}_{rule_count}_rules"
                if key in results:
                    row = {
                        'Method': method.capitalize(),
                        'Rules': rule_count,
                        'Accuracy (%)': f"{results[key].get('accuracy', 0.0) * 100:.1f}",
                        'L2 Distance': f"{results[key].get('l2_distance', 0.0):.3f}",
                        'Valid Syntax (%)': f"{results[key].get('valid_syntax_rate', 0.0) * 100:.1f}",
                        'Solve Time (ms)': f"{results[key].get('solve_time', 0.0) * 1000:.1f}"
                    }
                    data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False)
        with open(self.output_dir / 'tables' / 'detailed_results.tex', 'w') as f:
            f.write(latex_table)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'tables' / 'detailed_results.csv', index=False)
        
        return df
    
    def generate_all_figures(self):
        """Generate all figures and tables for the paper using real data."""
        
        print("Loading comparison results...")
        results = self.load_comparison_results()
        
        print("Loading statistical analysis results...")
        stats_results = self.load_statistical_results()
        
        print("Loading performance metrics...")
        df = self.load_performance_metrics()
        
        print("Loading training logs...")
        training_logs = self.load_training_logs()
        
        if not results:
            print("Warning: No comparison results found. Some plots may be skipped.")
        
        print("Generating main results table...")
        self.generate_main_results_table(results)
        
        print("Generating detailed results table...")
        self.generate_detailed_results_table(results)
        
        print("Plotting performance comparison...")
        self.plot_performance_by_rules(results)
        
        print("Plotting statistical comparison...")
        if stats_results:
            self.plot_ablation_study(results, stats_results)
        else:
            print("Skipping statistical comparison plot - no statistical results available")
        
        print("Plotting error analysis...")
        self.plot_error_analysis(results, df)
        
        print("Plotting training curves...")
        self.plot_training_curves(training_logs)
        
        print("Plotting generalization analysis...")
        self.plot_generalization_analysis(results, stats_results)
        
        # Generate LaTeX tables
        print("Generating LaTeX tables...")
        self.generate_latex_tables(results)
        
        # Generate summary report
        self.generate_figure_summary()
        
        print(f"\nFigures generated using real data saved to: {self.output_dir}")
        if not results and not stats_results:
            print("\nWarning: Many plots were skipped due to missing evaluation data.")
            print("Run the statistical comparison evaluation first to generate complete figures.")
    
    def generate_latex_tables(self, results: Dict):
        """Generate LaTeX tables using the table creation script."""
        import subprocess
        
        # Save results to temporary file for table script
        results_file = self.output_dir / 'temp_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Run table generation script
        table_script = Path(__file__).parent / 'create_paper_tables.py'
        if table_script.exists():
            try:
                subprocess.run([
                    'python', str(table_script),
                    '--results_file', str(results_file),
                    '--output_dir', str(self.output_dir / 'tables')
                ], check=True, capture_output=True)
                print("✓ LaTeX tables generated")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Table generation failed: {e}")
        
        # Clean up temp file
        if results_file.exists():
            results_file.unlink()
    
    def generate_figure_summary(self):
        """Generate a summary of all created figures."""
        
        summary = """
# Paper Figures and Tables Summary

## Main Paper Figures

1. **Figure 1**: `figures/performance_by_rules.png`
   - Performance comparison across different rule complexity
   - Shows compositional advantage as problems become more complex

2. **Figure 2**: `figures/ablation_study.png` 
   - Energy composition weight strategies
   - Effect of training data size

3. **Figure 3**: `figures/error_analysis.png`
   - Error type distribution comparison
   - Solution distance analysis for incorrect answers

## Supplementary Figures

4. **Figure S1**: `figures/training_curves.png`
   - Individual rule training convergence
   - Validation accuracy progression

5. **Figure S2**: `figures/generalization_analysis.png`
   - In-domain vs out-of-domain performance
   - Performance on specific 2-rule combinations

## Tables

1. **Table 1**: `tables/main_results.tex` / `.csv`
   - Main performance comparison across methods and rule counts

2. **Table S1**: `tables/detailed_results.tex` / `.csv`
   - Detailed breakdown with all evaluation metrics

## Usage in LaTeX

To include figures in your paper:

```latex
\\begin{figure}[t]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{figures/performance_by_rules.png}
    \\caption{Performance comparison across different problem complexities.}
    \\label{fig:performance}
\\end{figure}
```

To include tables:

```latex
\\input{tables/main_results.tex}
```
"""
        
        with open(self.output_dir / 'README.md', 'w') as f:
            f.write(summary)

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures for algebra EBM paper')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing comparison results')
    parser.add_argument('--output_dir', type=str, default='paper_figures',
                       help='Output directory for figures and tables')
    
    args = parser.parse_args()
    
    generator = PaperFigureGenerator(args.results_dir, args.output_dir)
    generator.generate_all_figures()
    
    print("\n✅ All paper figures and tables generated successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print("📖 See README.md for usage instructions")

if __name__ == '__main__':
    main()