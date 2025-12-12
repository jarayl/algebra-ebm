#!/usr/bin/env python3
"""
Fix the rule weight ordering issue in the optimized implementation.
"""

def fix_optimized_demo():
    """Fix the weight ordering in the optimized demo."""
    
    with open('optimized_compose_energies_demo.py', 'r') as f:
        content = f.read()
    
    # Fix the _create_weight_tensor method to use the same ordering as baseline
    old_method = '''    def _create_weight_tensor(self, rule_weights: Optional[Dict[str, float]], batch_size: int) -> torch.Tensor:
        """OPTIMIZATION 3: Create vectorized weight tensor."""
        if rule_weights is None:
            weights = torch.ones(self._n_rules, device=self.device)
        else:
            weights = torch.tensor([rule_weights.get(name, 1.0) for name in sorted(self.rule_models.keys())], 
                                 device=self.device, dtype=torch.float32)
        
        return weights.unsqueeze(0).expand(batch_size, -1)'''
    
    new_method = '''    def _create_weight_tensor(self, rule_weights: Optional[Dict[str, float]], rule_names: List[str], batch_size: int) -> torch.Tensor:
        """OPTIMIZATION 3: Create vectorized weight tensor."""
        if rule_weights is None:
            weights = torch.ones(len(rule_names), device=self.device)
        else:
            weights = torch.tensor([rule_weights.get(name, 1.0) for name in rule_names], 
                                 device=self.device, dtype=torch.float32)
        
        return weights.unsqueeze(0).expand(batch_size, -1)'''
    
    content = content.replace(old_method, new_method)
    
    # Update the calls to _create_weight_tensor to pass rule_names
    old_call1 = 'weight_tensor = self._create_weight_tensor(rule_weights, inp.shape[0])  # (B, num_rules)'
    new_call1 = 'weight_tensor = self._create_weight_tensor(rule_weights, rule_names, inp.shape[0])  # (B, num_rules)'
    
    old_call2 = 'weight_tensor = self._create_weight_tensor(rule_weights, batch_size)  # (B, num_rules)'
    new_call2 = 'weight_tensor = self._create_weight_tensor(rule_weights, rule_names, batch_size)  # (B, num_rules)'
    
    content = content.replace(old_call1, new_call1)
    content = content.replace(old_call2, new_call2)
    
    # Also need to add the import for List at the top
    import_line = 'from typing import Dict, List, Optional, Tuple'
    if import_line not in content:
        content = content.replace('from typing import Dict, List, Optional, Tuple', import_line)
    
    with open('optimized_compose_energies_demo.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed rule weight ordering in optimized implementation")


if __name__ == "__main__":
    fix_optimized_demo()