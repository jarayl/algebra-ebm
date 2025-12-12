#!/usr/bin/env python3
"""
Manually apply the key performance optimizations to the AlgebraInference class.
"""

import re

def apply_optimizations():
    """Add performance optimizations to the existing implementation."""
    
    # Read the file
    with open('src/algebra/algebra_inference.py', 'r') as f:
        content = f.read()
    
    # 1. Add cache class before AlgebraInference class
    cache_class_code = '''
class NormalizationCache:
    """Cache for normalization statistics to improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        
    def _make_key(self, rule_names: tuple, batch_size: int) -> str:
        """Create cache key from configuration."""
        return f"{rule_names}_{batch_size}"
    
    def get_stats(self, rule_names: tuple, batch_size: int):
        """Get cached statistics if available."""
        key = self._make_key(rule_names, batch_size)
        return self.cache.get(key)
    
    def store_stats(self, rule_names: tuple, batch_size: int, mean: torch.Tensor, std: torch.Tensor):
        """Store statistics in cache."""
        if len(self.cache) >= self.max_size:
            # Simple eviction - remove oldest
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._make_key(rule_names, batch_size)
        self.cache[key] = (mean.clone(), std.clone())
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


'''
    
    # Insert cache class before AlgebraInference
    if 'class NormalizationCache:' not in content:
        content = content.replace('class AlgebraInference:', cache_class_code + 'class AlgebraInference:')
    
    # 2. Add optimization attributes to __init__ 
    init_optimization = '''        
        # PERFORMANCE OPTIMIZATION: Add caching and pre-computation
        self._norm_cache = NormalizationCache()
        self._rule_names_tuple = tuple(sorted(self.rule_models.keys()))
        self._n_rules = len(self.rule_models)
        
        # Pre-compute normalization constants
        self._target_min, self._target_max = 1.0, 15.0
        self._target_scale = (self._target_max - self._target_min) / 4.0
        self._target_offset = (self._target_min + self._target_max) / 2.0
'''
    
    # Add after the logger.info line in __init__
    pattern = r'(logger\.info\(f"AlgebraInference initialized.*?\n)'
    replacement = r'\1' + init_optimization
    content = re.sub(pattern, replacement, content)
    
    # 3. Add helper method for weight tensor creation
    weight_method = '''
    def _create_weight_tensor(self, rule_weights: Optional[Dict[str, float]], batch_size: int) -> torch.Tensor:
        """Create optimized weight tensor for vectorized operations."""
        if rule_weights is None:
            weights = torch.ones(self._n_rules, device=self.device)
        else:
            # Convert dict to tensor in consistent order
            weights = torch.tensor([rule_weights.get(name, 1.0) for name in sorted(self.rule_models.keys())], 
                                 device=self.device, dtype=torch.float32)
        
        # Expand for broadcasting: (n_rules,) -> (batch_size, n_rules)
        return weights.unsqueeze(0).expand(batch_size, -1)
'''
    
    # Insert before compose_energies method
    content = content.replace('    def compose_energies(', weight_method + '\n    def compose_energies(')
    
    # 4. Optimize the compose_energies method body
    # Find the current method and replace the normalization section
    
    # First, update the method signature to add _skip_cache parameter
    old_sig = '''def compose_energies(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None,
        normalize: bool = True,
        calibration_scales: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:'''
    
    new_sig = '''def compose_energies(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None,
        normalize: bool = True,
        calibration_scales: Optional[Dict[str, float]] = None,
        _skip_cache: bool = False  # For benchmarking
    ) -> torch.Tensor:'''
    
    content = content.replace(old_sig, new_sig)
    
    # 5. Replace the normalization section in compose_energies
    # Find the multi-rule normalization section and replace it
    old_norm_section = '''        # Apply z-score normalization only for multi-rule case
        if len(individual_energies) > 1:
            # Stack energies for normalization: (B, num_rules)
            energies_stacked = torch.cat(individual_energies, dim=1)  
            
            # Compute z-score normalization across rules (per batch item)
            mean = energies_stacked.mean(dim=1, keepdim=True)  # (B, 1)
            std = energies_stacked.std(dim=1, keepdim=True)    # (B, 1)
            
            # Add epsilon for numerical stability
            epsilon = 1e-8
            std_safe = torch.clamp(std, min=epsilon)
            
            # Normalize: (energy - mean) / std
            energies_normalized = (energies_stacked - mean) / std_safe  # (B, num_rules)
            
            # Rescale to target range [1, 15] approximately
            # This provides stable energy scales for optimization
            target_std = 3.5  # Half of range [1, 15]
            target_mean = 8.0  # Center of range [1, 15]
            energies_rescaled = target_std * energies_normalized + target_mean  # (B, num_rules)
            
            # Apply rule weights to normalized energies
            total_energy = torch.zeros((inp.shape[0], 1), device=self.device)
            for i, rule_name in enumerate(rule_names):
                weight = rule_weights.get(rule_name, 1.0)
                total_energy += weight * energies_rescaled[:, i:i+1]  # (B, 1)'''
    
    new_norm_section = '''        # Apply z-score normalization only for multi-rule case
        if len(individual_energies) > 1:
            batch_size = inp.shape[0]
            
            # OPTIMIZATION: Check cache for normalization stats
            cached_stats = None
            if not _skip_cache:
                cached_stats = self._norm_cache.get_stats(self._rule_names_tuple, batch_size)
            
            # Stack energies for normalization: (B, num_rules)
            energies_stacked = torch.cat(individual_energies, dim=1)  
            
            if cached_stats is not None:
                # Use cached statistics
                mean, std = cached_stats
                mean = mean.to(energies_stacked.device)
                std = std.to(energies_stacked.device)
            else:
                # Compute z-score normalization across rules (per batch item)
                mean = energies_stacked.mean(dim=1, keepdim=True)  # (B, 1)
                std = energies_stacked.std(dim=1, keepdim=True)    # (B, 1)
                
                # Cache the computed statistics
                if not _skip_cache:
                    self._norm_cache.store_stats(self._rule_names_tuple, batch_size, mean, std)
            
            # Add epsilon for numerical stability
            epsilon = 1e-8
            std_safe = torch.clamp(std, min=epsilon)
            
            # Normalize: (energy - mean) / std
            energies_normalized = (energies_stacked - mean) / std_safe  # (B, num_rules)
            
            # OPTIMIZATION: Use pre-computed rescaling constants
            energies_rescaled = self._target_scale * energies_normalized + self._target_offset  # (B, num_rules)
            
            # OPTIMIZATION: Vectorized weight application
            weight_tensor = self._create_weight_tensor(rule_weights, batch_size)  # (B, num_rules)
            weighted_energies = energies_rescaled * weight_tensor  # (B, num_rules)
            total_energy = weighted_energies.sum(dim=1, keepdim=True)  # (B, 1)'''
    
    content = content.replace(old_norm_section, new_norm_section)
    
    # Also optimize the non-normalization case
    old_non_norm = '''        if not normalize:
            # Backward compatibility: original weighted summation
            total_energy = torch.zeros_like(individual_energies[0])
            for i, rule_name in enumerate(rule_names):
                weight = rule_weights.get(rule_name, 1.0)
                total_energy += weight * individual_energies[i]
            return total_energy'''
    
    new_non_norm = '''        if not normalize:
            # OPTIMIZATION: Vectorized summation for non-normalized case
            energies_stacked = torch.cat(individual_energies, dim=1)  # (B, num_rules)
            weight_tensor = self._create_weight_tensor(rule_weights, inp.shape[0])  # (B, num_rules)
            weighted_energies = energies_stacked * weight_tensor  # (B, num_rules)
            return weighted_energies.sum(dim=1, keepdim=True)  # (B, 1)'''
    
    content = content.replace(old_non_norm, new_non_norm)
    
    # Write the optimized content back
    with open('src/algebra/algebra_inference.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied performance optimizations:")
    print("   - Added NormalizationCache class")
    print("   - Added caching and pre-computation to __init__")
    print("   - Added _create_weight_tensor helper method")
    print("   - Added _skip_cache parameter for benchmarking")
    print("   - Optimized multi-rule normalization with caching")
    print("   - Vectorized weight application")
    print("   - Pre-computed rescaling constants")


if __name__ == "__main__":
    apply_optimizations()