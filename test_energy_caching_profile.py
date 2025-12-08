#!/usr/bin/env python3
"""
Profile energy caching optimization in IRED inference.

Tests the existing energy caching implementation and identifies potential
improvements for achieving 30-50% inference speedup.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import logging

# Import inference components
from src.algebra.algebra_inference import AlgebraInference, InferenceConfig
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper
from src.algebra.algebra_encoder import CharacterLevelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyCallCounter:
    """Wrapper to count energy computation calls for profiling."""
    
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model
        self.energy_call_count = 0
        self.forward_call_count = 0
        
    def __call__(self, inp, out, t, return_energy=True):
        """Track calls and delegate to wrapped model."""
        if return_energy:
            self.energy_call_count += 1
        self.forward_call_count += 1
        return self.wrapped_model(inp, out, t, return_energy=return_energy)
    
    def to(self, device):
        """Delegate device movement."""
        self.wrapped_model.to(device)
        return self
        
    def eval(self):
        """Delegate eval mode."""
        self.wrapped_model.eval()
        return self
        
    def reset_counters(self):
        """Reset call counters."""
        self.energy_call_count = 0
        self.forward_call_count = 0


class InferenceProfiler:
    """Profile inference performance and energy caching effectiveness."""
    
    def __init__(self):
        self.results = []
        
    def create_test_models(self, rule_names: List[str]) -> Dict[str, EnergyCallCounter]:
        """Create mock rule models with call counting."""
        rule_models = {}
        
        for rule in rule_names:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            wrapper.eval()
            
            # Wrap with call counter
            counter_wrapper = EnergyCallCounter(wrapper)
            rule_models[rule] = counter_wrapper
            
        return rule_models
    
    def profile_inference(
        self, 
        equation: str,
        rule_names: List[str],
        config: InferenceConfig,
        iterations: int = 5
    ) -> Dict:
        """Profile inference performance for a given equation and config."""
        
        # Create models and encoder
        rule_models = self.create_test_models(rule_names)
        encoder = CharacterLevelEncoder()
        inference = AlgebraInference(rule_models, encoder, config=config, device='cpu')
        
        # Warm up
        try:
            with torch.no_grad():
                inp_embedding = encoder(equation).unsqueeze(0)
            _, _ = inference.ired_inference(inp_embedding)
        except Exception as e:
            logger.warning(f"Warmup failed for {equation}: {e}")
        
        # Reset counters
        for model in rule_models.values():
            model.reset_counters()
        
        # Profile multiple iterations
        times = []
        energy_calls_list = []
        forward_calls_list = []
        
        for i in range(iterations):
            try:
                # Encode equation
                with torch.no_grad():
                    inp_embedding = encoder(equation).unsqueeze(0)
                
                # Reset counters for this iteration
                for model in rule_models.values():
                    model.reset_counters()
                
                # Time inference
                start_time = time.perf_counter()
                out_embedding, info = inference.ired_inference(inp_embedding)
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
                # Collect call counts
                total_energy_calls = sum(m.energy_call_count for m in rule_models.values())
                total_forward_calls = sum(m.forward_call_count for m in rule_models.values())
                
                energy_calls_list.append(total_energy_calls)
                forward_calls_list.append(total_forward_calls)
                
                logger.debug(f"Iteration {i}: {inference_time:.4f}s, {total_energy_calls} energy calls")
                
            except Exception as e:
                logger.error(f"Profile iteration {i} failed for {equation}: {e}")
                continue
        
        if not times:
            return {
                'equation': equation,
                'error': 'All iterations failed',
                'config': config.__dict__
            }
        
        # Compute statistics
        result = {
            'equation': equation,
            'rule_names': rule_names,
            'config': {
                'K': config.K,
                'max_iterations': config.max_iterations,
                'step_size': config.step_size,
                'use_adaptive_step': config.use_adaptive_step
            },
            'iterations': len(times),
            'times': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'raw': times
            },
            'energy_calls': {
                'mean': np.mean(energy_calls_list) if energy_calls_list else 0,
                'std': np.std(energy_calls_list) if energy_calls_list else 0,
                'min': np.min(energy_calls_list) if energy_calls_list else 0,
                'max': np.max(energy_calls_list) if energy_calls_list else 0,
                'raw': energy_calls_list
            },
            'forward_calls': {
                'mean': np.mean(forward_calls_list) if forward_calls_list else 0,
                'std': np.std(forward_calls_list) if forward_calls_list else 0,
                'min': np.min(forward_calls_list) if forward_calls_list else 0,
                'max': np.max(forward_calls_list) if forward_calls_list else 0,
                'raw': forward_calls_list
            }
        }
        
        self.results.append(result)
        return result
    
    def profile_caching_effectiveness(self) -> Dict:
        """Profile the effectiveness of current energy caching implementation."""
        
        logger.info("Profiling energy caching effectiveness...")
        
        # Test equations of varying complexity
        test_equations = [
            "x+1=2",
            "2*x+3=7", 
            "x*x=4",
            "2*(x+3)=10",
            "x**2+2*x=8"
        ]
        
        # Test different configurations
        configs = [
            InferenceConfig(K=5, max_iterations=10, step_size=0.01),
            InferenceConfig(K=10, max_iterations=20, step_size=0.01),
            InferenceConfig(K=5, max_iterations=30, step_size=0.005),
        ]
        
        rule_names = ['distribute', 'combine', 'isolate']
        
        all_results = []
        
        for config in configs:
            for equation in test_equations:
                logger.info(f"Profiling: {equation} with K={config.K}, iter={config.max_iterations}")
                
                result = self.profile_inference(equation, rule_names, config, iterations=3)
                all_results.append(result)
                
                # Log summary
                if 'times' in result:
                    mean_time = result['times']['mean']
                    mean_energy_calls = result['energy_calls']['mean']
                    logger.info(f"  Time: {mean_time:.4f}s, Energy calls: {mean_energy_calls:.1f}")
        
        # Analyze caching effectiveness
        analysis = self.analyze_caching_results(all_results)
        
        return {
            'profile_results': all_results,
            'caching_analysis': analysis,
            'recommendations': self.generate_optimization_recommendations(analysis)
        }
    
    def analyze_caching_results(self, results: List[Dict]) -> Dict:
        """Analyze caching effectiveness from profile results."""
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Extract key metrics
        times = []
        energy_calls = []
        forward_calls = []
        total_steps = []
        
        for result in results:
            if 'times' in result and 'energy_calls' in result:
                times.append(result['times']['mean'])
                energy_calls.append(result['energy_calls']['mean'])
                forward_calls.append(result['forward_calls']['mean'])
                
                # Estimate total steps from config
                config = result['config']
                estimated_steps = config['K'] * config['max_iterations']
                total_steps.append(estimated_steps)
        
        if not times:
            return {'error': 'No valid timing data'}
        
        # Calculate caching effectiveness metrics
        avg_time = np.mean(times)
        avg_energy_calls = np.mean(energy_calls)
        avg_forward_calls = np.mean(forward_calls)
        avg_total_steps = np.mean(total_steps)
        
        # Estimate potential speedup from better caching
        # Current implementation already has some caching
        # Ideal case: 1 energy call per optimization step
        theoretical_min_calls = avg_total_steps
        current_calls = avg_energy_calls
        
        potential_reduction = max(0, (current_calls - theoretical_min_calls) / current_calls)
        estimated_speedup = 1.0 / (1.0 - potential_reduction * 0.7)  # 70% of reduction translates to speedup
        
        analysis = {
            'avg_inference_time': avg_time,
            'avg_energy_calls_per_inference': avg_energy_calls,
            'avg_forward_calls_per_inference': avg_forward_calls,
            'avg_total_optimization_steps': avg_total_steps,
            'theoretical_min_energy_calls': theoretical_min_calls,
            'potential_call_reduction_pct': potential_reduction * 100,
            'estimated_speedup_factor': estimated_speedup,
            'estimated_speedup_pct': (estimated_speedup - 1.0) * 100,
            'caching_efficiency': 1.0 - (current_calls / max(avg_total_steps * 2, current_calls))  # Rough efficiency estimate
        }
        
        return analysis
    
    def generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate specific optimization recommendations."""
        
        recommendations = []
        
        if 'estimated_speedup_pct' in analysis:
            speedup_pct = analysis['estimated_speedup_pct']
            
            if speedup_pct >= 20:
                recommendations.append(
                    f"HIGH PRIORITY: Current caching can be improved for {speedup_pct:.1f}% speedup. "
                    f"Focus on reducing redundant energy computations."
                )
            elif speedup_pct >= 10:
                recommendations.append(
                    f"MEDIUM PRIORITY: Moderate speedup potential ({speedup_pct:.1f}%). "
                    f"Consider advanced caching strategies."
                )
            else:
                recommendations.append(
                    f"LOW PRIORITY: Limited speedup potential ({speedup_pct:.1f}%). "
                    f"Current caching is relatively effective."
                )
        
        if 'caching_efficiency' in analysis:
            efficiency = analysis['caching_efficiency']
            
            if efficiency < 0.5:
                recommendations.append(
                    "Improve cache hit rate by extending cache lifetime across landscapes."
                )
            
            if efficiency < 0.7:
                recommendations.append(
                    "Consider implementing gradient caching to pair with energy caching."
                )
        
        # Always include general recommendations
        recommendations.extend([
            "Profile individual energy computation time to identify bottlenecks.",
            "Consider batching energy computations for multiple rules.",
            "Investigate tensor reuse opportunities in gradient computation."
        ])
        
        return recommendations


def main():
    """Main profiling entry point."""
    
    logger.info("Starting energy caching optimization profiling...")
    
    profiler = InferenceProfiler()
    results = profiler.profile_caching_effectiveness()
    
    # Print results
    print("\n" + "="*60)
    print("ENERGY CACHING OPTIMIZATION PROFILE RESULTS")
    print("="*60)
    
    analysis = results['caching_analysis']
    
    print(f"\nCURRENT PERFORMANCE:")
    print(f"  Average inference time: {analysis.get('avg_inference_time', 0):.4f}s")
    print(f"  Average energy calls per inference: {analysis.get('avg_energy_calls_per_inference', 0):.1f}")
    print(f"  Average total optimization steps: {analysis.get('avg_total_optimization_steps', 0):.1f}")
    print(f"  Caching efficiency: {analysis.get('caching_efficiency', 0)*100:.1f}%")
    
    print(f"\nOPTIMIZATION POTENTIAL:")
    print(f"  Theoretical min energy calls: {analysis.get('theoretical_min_energy_calls', 0):.1f}")
    print(f"  Potential call reduction: {analysis.get('potential_call_reduction_pct', 0):.1f}%")
    print(f"  Estimated speedup factor: {analysis.get('estimated_speedup_factor', 1):.2f}x")
    print(f"  Estimated speedup: {analysis.get('estimated_speedup_pct', 0):.1f}%")
    
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save detailed results
    output_file = "energy_caching_profile_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Determine if optimization target is achievable
    target_speedup = 30  # 30% minimum target
    estimated_speedup = analysis.get('estimated_speedup_pct', 0)
    
    if estimated_speedup >= target_speedup:
        print(f"\n✅ TARGET ACHIEVABLE: {target_speedup}% speedup target can be met with current optimization potential.")
        status = "achievable"
    else:
        print(f"\n⚠️  TARGET CHALLENGING: Only {estimated_speedup:.1f}% speedup estimated, below {target_speedup}% target.")
        status = "challenging"
    
    return {
        'status': status,
        'estimated_speedup_pct': estimated_speedup,
        'target_speedup_pct': target_speedup,
        'detailed_results': results
    }


if __name__ == "__main__":
    main()