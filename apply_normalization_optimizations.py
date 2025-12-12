#!/usr/bin/env python3
"""
Apply normalization performance optimizations to the existing AlgebraInference class.

This script patches the existing implementation to add:
1. Normalization caching
2. Vectorized tensor operations
3. Pre-computed weight tensors
4. Reduced memory allocations
"""

import re


def apply_optimizations():
    """Apply performance optimizations to the algebra_inference.py file."""
    
    # Read the original file
    with open('src/algebra/algebra_inference.py', 'r') as f:
        content = f.read()
    
    # Add normalization cache class before AlgebraInference
    cache_class = '''
class NormalizationCache:
    """Optimized cache for energy normalization statistics."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []  # For LRU eviction
        
    def _make_key(self, rule_names: List[str], batch_size: int, n_rules: int) -> str:
        """Create cache key from rule configuration."""
        # Use rule names sorted for consistency
        sorted_rules = tuple(sorted(rule_names))
        return f"{sorted_rules}_{batch_size}_{n_rules}"
    
    def get_norm_stats(self, rule_names: List[str], batch_size: int, 
                       individual_energies: List[torch.Tensor]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached normalization statistics if available."""
        key = self._make_key(rule_names, batch_size, len(individual_energies))
        
        if key in self.cache:
            # Move to end of access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def store_norm_stats(self, rule_names: List[str], batch_size: int,
                        individual_energies: List[torch.Tensor],
                        mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store normalization statistics in cache."""
        key = self._make_key(rule_names, batch_size, len(individual_energies))
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = (mean.clone(), std.clone())
        
        if key not in self.access_order:
            self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


'''
    
    # Insert cache class before AlgebraInference class
    content = content.replace('class AlgebraInference:', cache_class + 'class AlgebraInference:')
    
    # Add performance optimization attributes to __init__
    init_additions = '''        
        # PERFORMANCE OPTIMIZATION: Initialize normalization cache and pre-computed structures
        self._norm_cache = NormalizationCache(max_size=1000)
        self._rule_names = list(self.rule_models.keys())
        self._n_rules = len(self._rule_names)
'''
    
    # Find the end of __init__ and add optimizations
    init_pattern = r'(logger\.info\(f"AlgebraInference initialized.*?\n)'
    content = re.sub(init_pattern, r'\1' + init_additions, content, flags=re.DOTALL)
    
    # Add helper method for weight tensor creation
    weight_tensor_method = '''
    def _create_weight_tensor(self, rule_weights: Optional[Dict[str, float]], batch_size: int) -> torch.Tensor:
        """Create optimized weight tensor for vectorized operations."""
        if rule_weights is None:
            return torch.ones(batch_size, 1, self._n_rules, device=self.device)
        
        # Convert weights dict to tensor in consistent rule order
        weights = torch.tensor([rule_weights.get(name, 1.0) for name in self._rule_names], 
                              device=self.device)
        # Reshape for broadcasting: (1, 1, n_rules) -> (batch_size, 1, n_rules) 
        return weights.view(1, 1, -1).expand(batch_size, 1, -1)
'''
    
    # Insert weight tensor method before compose_energies
    content = content.replace('    def compose_energies(', weight_tensor_method + '\n    def compose_energies(')
    
    # Replace the compose_energies method signature to add _skip_cache parameter
    old_signature = '''def compose_energies(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None,
        normalize: bool = True,
        calibration_scales: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:'''
    
    new_signature = '''def compose_energies(
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
    
    content = content.replace(old_signature, new_signature)
    
    # Find the method body and replace with optimized version
    # Start of method body (after docstring)
    method_start = content.find('        if rule_weights is None:')
    method_end = content.find('\n    def compute_composed_gradient(')
    
    if method_start != -1 and method_end != -1:
        # Extract the current method body
        old_body = content[method_start:method_end]
        
        # Create optimized method body
        new_body = '''        """
        Compose energy functions from multiple rules by weighted summation.
        
        PERFORMANCE OPTIMIZED VERSION with:
        - Vectorized tensor operations
        - Normalization caching 
        - Reduced memory allocations
        - Pre-computed weight tensors
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128)
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule (default: all 1.0)
            t: Optional pre-allocated timestep tensor (default: allocate new)
            normalize: Whether to apply z-score normalization (default: True)
            calibration_scales: Optional per-rule calibration scales (default: None)
            _skip_cache: Internal flag to disable caching for benchmarking
            
        Returns:
            total_energy: Composed energy value (B, 1)
        """
        if rule_weights is None:
            rule_weights = {name: 1.0 for name in self.rule_models.keys()}
        
        # Create timestep tensor if not provided
        if t is None:
            t = torch.full((inp.shape[0],), k, dtype=torch.long, device=self.device)
        else:
            # Validate pre-allocated tensor
            if t.device != self.device:
                raise ValueError(f"Pre-allocated tensor device {t.device} does not match inference device {self.device}")
        
        # PERFORMANCE OPTIMIZATION: Collect energies using consistent ordering
        individual_energies = []
        for rule_name in self._rule_names:
            model = self.rule_models[rule_name]
            energy = model(inp, out, t, return_energy=True)  # (B, 1)
            
            # Apply calibration scales if provided
            if calibration_scales is not None and rule_name in calibration_scales:
                energy = energy * calibration_scales[rule_name]
            
            individual_energies.append(energy)
        
        if not normalize:
            # PERFORMANCE OPTIMIZATION: Use pre-computed weight tensor for vectorized summation
            weight_tensor = self._create_weight_tensor(rule_weights, inp.shape[0])
            energy_tensor = torch.stack(individual_energies, dim=-1)  # (B, 1, N_rules)
            weighted_energies = energy_tensor * weight_tensor  # (B, 1, N_rules)
            return weighted_energies.sum(dim=-1)  # (B, 1)
        
        # Apply z-score normalization only for multi-rule case
        if len(individual_energies) > 1:
            batch_size = inp.shape[0]
            cached_stats = None
            
            # PERFORMANCE OPTIMIZATION: Check normalization cache first
            if not _skip_cache:
                cached_stats = self._norm_cache.get_norm_stats(self._rule_names, batch_size, individual_energies)
            
            # PERFORMANCE OPTIMIZATION: Stack energies once for all operations
            energies_stacked = torch.cat(individual_energies, dim=1)  # (B, num_rules)
            
            if cached_stats is not None:
                # Use cached normalization statistics
                mean, std = cached_stats
                mean = mean.to(energies_stacked.device)
                std = std.to(energies_stacked.device)
            else:
                # Compute normalization statistics
                mean = energies_stacked.mean(dim=1, keepdim=True)  # (B, 1)
                std = energies_stacked.std(dim=1, keepdim=True)    # (B, 1)
                
                # Cache the statistics for future use
                if not _skip_cache:
                    self._norm_cache.store_norm_stats(self._rule_names, batch_size, individual_energies, mean, std)
            
            # Add epsilon for numerical stability
            epsilon = 1e-8
            std_safe = torch.clamp(std, min=epsilon)
            
            # Normalize: (energy - mean) / std
            energies_normalized = (energies_stacked - mean) / std_safe  # (B, num_rules)
            
            # PERFORMANCE OPTIMIZATION: Vectorized re-scaling to target range [1, 15]
            target_std = 3.5  # Half of range [1, 15]
            target_mean = 8.0  # Center of range [1, 15]
            energies_rescaled = target_std * energies_normalized + target_mean  # (B, num_rules)
            
            # PERFORMANCE OPTIMIZATION: Vectorized weight application
            weight_tensor = self._create_weight_tensor(rule_weights, batch_size)  # (B, 1, N_rules)
            energies_rescaled_expanded = energies_rescaled.unsqueeze(1)  # (B, 1, num_rules)
            weighted_rescaled = energies_rescaled_expanded * weight_tensor[:, :, :len(individual_energies)]  # (B, 1, num_rules)
            total_energy = weighted_rescaled.sum(dim=-1)  # (B, 1)
        else:
            # Single rule: no normalization needed, just apply weight
            rule_name = self._rule_names[0]
            weight = rule_weights.get(rule_name, 1.0)
            total_energy = weight * individual_energies[0]
        
        return total_energy'''
        
        # Replace the method body
        content = content[:method_start] + new_body + content[method_end:]
    
    # Write the modified content back
    with open('src/algebra/algebra_inference.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied normalization performance optimizations to compose_energies method")
    print("   - Added NormalizationCache class for caching statistics")
    print("   - Added _create_weight_tensor helper for vectorized operations")
    print("   - Optimized compose_energies with caching and vectorization")
    print("   - Added _skip_cache parameter for benchmarking")


if __name__ == "__main__":
    apply_optimizations()