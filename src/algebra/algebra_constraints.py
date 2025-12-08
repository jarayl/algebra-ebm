"""
Algebra Constraint Energy Functions for Test-Time Injection

Implements hand-designed constraint energies that can be composed with 
rule-specific energy functions during inference for guided problem solving.

Classes:
- PositivityEnergy: Penalizes negative solution values
- IntegernessEnergy: Penalizes non-integer solution values  
- ConstraintComposition: Utility for composing multiple constraint energies

Functions:
- extract_solution_value: Extract numerical solution from equation embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, List
import logging
import re
import math

# Set up logging
logger = logging.getLogger(__name__)


def extract_solution_value(equation_embedding: torch.Tensor, decoder=None) -> torch.Tensor:
    """
    Extract numerical solution value from equation embedding.
    
    This function attempts to decode the equation embedding back to text
    and extract the numerical value of the variable (typically 'x').
    
    Args:
        equation_embedding: Equation embedding tensor (B, embed_dim)
        decoder: Optional decoder for embedding-to-text conversion
        
    Returns:
        solution_values: Extracted numerical values (B,)
        
    Note:
        For demonstration purposes, this implementation uses a simplified
        approach. In practice, this would require the full decoder pipeline.
    """
    batch_size = equation_embedding.shape[0]
    device = equation_embedding.device
    
    if decoder is not None:
        # Full decoding approach (when decoder is available)
        try:
            # Decode embeddings to text equations
            equations = decoder.decode_batch(equation_embedding)
            
            # Extract solution values from equations
            solution_values = []
            for eq in equations:
                value = _extract_numerical_value_from_equation(eq)
                solution_values.append(value)
            
            return torch.tensor(solution_values, device=device, dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Decoder-based extraction failed: {e}. Falling back to heuristic method.")
    
    # Heuristic approach: Use embedding statistics as proxy for solution value
    # This is a simplified demonstration - assumes embedding magnitude correlates with solution
    embedding_magnitude = torch.norm(equation_embedding, dim=-1)  # (B,)
    
    # Map embedding magnitude to reasonable solution range [-10, 10]
    # Using tanh to bound the output and provide smooth gradients
    solution_values = 10.0 * torch.tanh(embedding_magnitude / 5.0)
    
    return solution_values


def _extract_numerical_value_from_equation(equation_text: str) -> float:
    """
    Extract numerical value from equation text (e.g., "x = 3.5" -> 3.5).
    
    Args:
        equation_text: String representation of equation
        
    Returns:
        Extracted numerical value, or 0.0 if extraction fails
    """
    # Look for patterns like "x = number" or "x=number"
    patterns = [
        r'x\s*=\s*([+-]?\d*\.?\d+)',  # x = 3.5, x=-2, etc.
        r'([+-]?\d*\.?\d+)\s*=\s*x',  # 3.5 = x
        r'([+-]?\d*\.?\d+)$'          # Just a number at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, equation_text.lower())
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Default fallback
    logger.debug(f"Could not extract numerical value from: {equation_text}")
    return 0.0


class PositivityEnergy(nn.Module):
    """
    Constraint energy that penalizes negative solution values.
    
    This constraint encourages solutions to be positive (x > 0) by applying
    energy penalties proportional to how negative the solution is.
    
    Energy Function:
        E(x) = beta * max(0, -x)^2
        
    Where:
    - x is the extracted solution value  
    - beta is the constraint strength parameter
    - Energy is 0 for positive solutions, increases quadratically for negative values
    
    Args:
        beta: Constraint strength parameter in range [0.1, 1.0] (default: 0.5)
        inp_dim: Input embedding dimension (default: 128) 
        out_dim: Output embedding dimension (default: 128)
    """
    
    def __init__(
        self, 
        beta: float = 0.5,
        inp_dim: int = 128,
        out_dim: int = 128
    ):
        super(PositivityEnergy, self).__init__()
        
        # Validate beta parameter
        if not (0.1 <= beta <= 1.0):
            raise ValueError(f"beta must be in range [0.1, 1.0], got {beta}")
        
        self.beta = beta
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.constraint_type = "positivity"
        
        # Store for compatibility with AlgebraEBM interface
        self.rule_name = f"positivity_constraint_beta_{beta}"
        
        logger.info(f"Initialized PositivityEnergy with beta={beta}")
    
    def forward(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute positivity constraint energy.
        
        Args:
            inp: Input equation embedding (B, inp_dim) - unused for constraint
            out: Output equation embedding (B, out_dim) 
            t: Timestep (B,) - unused for constraint but kept for interface compatibility
            
        Returns:
            energy: Constraint energy (B, 1)
        """
        # Input validation
        assert out.shape[-1] == self.out_dim, f"Expected out_dim={self.out_dim}, got {out.shape[-1]}"
        batch_size = out.shape[0]
        
        # Extract solution values from output embeddings
        solution_values = extract_solution_value(out)  # (B,)
        
        # Compute positivity constraint energy: E = beta * max(0, -x)^2
        # This penalizes negative values quadratically while leaving positive values unpenalized
        negative_part = torch.clamp(-solution_values, min=0.0)  # max(0, -x)
        constraint_energy = self.beta * negative_part.pow(2)  # beta * max(0, -x)^2
        
        # Reshape to match expected output format (B, 1)
        energy = constraint_energy.unsqueeze(-1)
        
        # Log constraint violations for monitoring
        num_violations = (solution_values < 0).sum().item()
        if num_violations > 0:
            avg_violation = negative_part[negative_part > 0].mean().item() if num_violations > 0 else 0.0
            logger.debug(f"PositivityEnergy: {num_violations}/{batch_size} violations, "
                        f"avg_violation={avg_violation:.4f}, avg_energy={energy.mean().item():.6f}")
        
        return energy


class IntegernessEnergy(nn.Module):
    """
    Constraint energy that penalizes non-integer solution values.
    
    This constraint encourages solutions to be integers by applying energy
    penalties based on distance to the nearest integer.
    
    Energy Function:
        E(x) = beta * min(|x - floor(x)|, |x - ceil(x)|)^2
              = beta * (x - round(x))^2
        
    Where:
    - x is the extracted solution value
    - beta is the constraint strength parameter  
    - Energy is 0 for integer solutions, increases with distance from integers
    
    Args:
        beta: Constraint strength parameter in range [0.1, 1.0] (default: 0.3)
        inp_dim: Input embedding dimension (default: 128)
        out_dim: Output embedding dimension (default: 128)
    """
    
    def __init__(
        self, 
        beta: float = 0.3,
        inp_dim: int = 128,
        out_dim: int = 128
    ):
        super(IntegernessEnergy, self).__init__()
        
        # Validate beta parameter
        if not (0.1 <= beta <= 1.0):
            raise ValueError(f"beta must be in range [0.1, 1.0], got {beta}")
        
        self.beta = beta
        self.inp_dim = inp_dim  
        self.out_dim = out_dim
        self.constraint_type = "integerness"
        
        # Store for compatibility with AlgebraEBM interface
        self.rule_name = f"integerness_constraint_beta_{beta}"
        
        logger.info(f"Initialized IntegernessEnergy with beta={beta}")
    
    def forward(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute integerness constraint energy.
        
        Args:
            inp: Input equation embedding (B, inp_dim) - unused for constraint
            out: Output equation embedding (B, out_dim)
            t: Timestep (B,) - unused for constraint but kept for interface compatibility
            
        Returns:
            energy: Constraint energy (B, 1)
        """
        # Input validation
        assert out.shape[-1] == self.out_dim, f"Expected out_dim={self.out_dim}, got {out.shape[-1]}"
        batch_size = out.shape[0]
        
        # Extract solution values from output embeddings
        solution_values = extract_solution_value(out)  # (B,)
        
        # Compute integerness constraint energy: E = beta * (x - round(x))^2
        # This penalizes distance from the nearest integer
        nearest_integers = torch.round(solution_values)
        distance_to_integer = torch.abs(solution_values - nearest_integers)
        constraint_energy = self.beta * distance_to_integer.pow(2)
        
        # Reshape to match expected output format (B, 1)
        energy = constraint_energy.unsqueeze(-1)
        
        # Log constraint violations for monitoring 
        violation_threshold = 0.1  # Consider solutions within 0.1 of an integer as "close enough"
        num_violations = (distance_to_integer > violation_threshold).sum().item()
        if num_violations > 0:
            avg_violation = distance_to_integer[distance_to_integer > violation_threshold].mean().item()
            logger.debug(f"IntegernessEnergy: {num_violations}/{batch_size} violations, "
                        f"avg_violation={avg_violation:.4f}, avg_energy={energy.mean().item():.6f}")
        
        return energy


class ConstraintComposition:
    """
    Utility for composing multiple constraint energies with rule energies.
    
    This class facilitates combining rule-specific energy functions with
    constraint energy functions during inference, allowing for guided
    problem solving with multiple objectives.
    
    Args:
        rule_energies: Dictionary of rule-specific energy functions
        constraint_energies: List of constraint energy functions
        constraint_weights: Optional weights for constraint energies (default: all 1.0)
    """
    
    def __init__(
        self,
        rule_energies: Dict[str, nn.Module],
        constraint_energies: List[nn.Module],
        constraint_weights: Optional[List[float]] = None
    ):
        self.rule_energies = rule_energies
        self.constraint_energies = constraint_energies
        
        # Set default weights if not provided
        if constraint_weights is None:
            constraint_weights = [1.0] * len(constraint_energies)
        
        if len(constraint_weights) != len(constraint_energies):
            raise ValueError(
                f"Number of constraint weights ({len(constraint_weights)}) "
                f"must match number of constraint energies ({len(constraint_energies)})"
            )
        
        self.constraint_weights = constraint_weights
        
        logger.info(f"ConstraintComposition initialized with {len(rule_energies)} rule energies "
                   f"and {len(constraint_energies)} constraint energies")
    
    def compute_total_energy(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        t: torch.Tensor,
        active_rules: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total energy by composing rule and constraint energies.
        
        Args:
            inp: Input equation embedding (B, inp_dim)
            out: Output equation embedding (B, out_dim) 
            t: Diffusion timestep (B,)
            active_rules: List of rule names to include (default: all rules)
            
        Returns:
            total_energy: Combined energy (B, 1)
            energy_breakdown: Dict with individual energy contributions
        """
        batch_size = inp.shape[0]
        device = inp.device
        
        # Initialize energy accumulator
        total_energy = torch.zeros(batch_size, 1, device=device, dtype=inp.dtype)
        energy_breakdown = {}
        
        # Add rule energies
        if active_rules is None:
            active_rules = list(self.rule_energies.keys())
        
        for rule_name in active_rules:
            if rule_name in self.rule_energies:
                rule_energy = self.rule_energies[rule_name](inp, out, t)
                total_energy += rule_energy
                energy_breakdown[f"rule_{rule_name}"] = rule_energy
            else:
                logger.warning(f"Rule '{rule_name}' not found in available rules: {list(self.rule_energies.keys())}")
        
        # Add constraint energies with weights
        for i, (constraint, weight) in enumerate(zip(self.constraint_energies, self.constraint_weights)):
            constraint_energy = constraint(inp, out, t)
            weighted_constraint_energy = weight * constraint_energy
            total_energy += weighted_constraint_energy
            
            constraint_name = f"constraint_{constraint.constraint_type}"
            energy_breakdown[constraint_name] = constraint_energy
            energy_breakdown[f"{constraint_name}_weighted"] = weighted_constraint_energy
        
        return total_energy, energy_breakdown
    
    def get_constraint_summary(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Get summary information about active constraints.
        
        Returns:
            Dictionary mapping constraint indices to their properties
        """
        summary = {}
        for i, (constraint, weight) in enumerate(zip(self.constraint_energies, self.constraint_weights)):
            summary[f"constraint_{i}"] = {
                "type": constraint.constraint_type,
                "weight": weight,
                "beta": constraint.beta,
                "rule_name": constraint.rule_name
            }
        return summary


# Compatibility wrapper for diffusion interface
class ConstraintDiffusionWrapper(nn.Module):
    """
    Diffusion wrapper for constraint energies to enable gradient computation.
    
    This wrapper provides the same interface as AlgebraDiffusionWrapper
    for constraint energy functions, enabling their use in gradient-based inference.
    
    Args:
        constraint_energy: PositivityEnergy or IntegernessEnergy instance
    """
    
    def __init__(self, constraint_energy: Union[PositivityEnergy, IntegernessEnergy]):
        super(ConstraintDiffusionWrapper, self).__init__()
        self.constraint_energy = constraint_energy
        self.inp_dim = constraint_energy.inp_dim
        self.out_dim = constraint_energy.out_dim
        self.rule_name = constraint_energy.rule_name
    
    def forward(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        t: torch.Tensor,
        return_energy: bool = False,
        return_both: bool = False
    ) -> torch.Tensor:
        """
        Compute constraint energy gradients for diffusion training/inference.
        
        Args:
            inp: Input equation embedding (B, inp_dim)
            out: Output equation embedding (B, out_dim)
            t: Diffusion timestep (B,)
            return_energy: If True, return energy instead of gradient
            return_both: If True, return tuple (energy, gradient)
            
        Returns:
            grad: Energy gradient dE/dout (B, out_dim) [default]
            energy: Energy value (B, 1) [if return_energy=True]
            (energy, grad): Tuple [if return_both=True]
        """
        # Input validation
        assert inp.shape[-1] == self.inp_dim, f"Expected inp_dim={self.inp_dim}, got {inp.shape[-1]}"
        assert out.shape[-1] == self.out_dim, f"Expected out_dim={self.out_dim}, got {out.shape[-1]}"
        assert inp.shape[0] == out.shape[0] == t.shape[0], "Batch sizes must match"
        
        # Enable gradient computation for output
        if not out.requires_grad:
            out = out.detach().clone().requires_grad_(True)
        else:
            out = out.clone().requires_grad_(True)
        
        # Compute constraint energy
        energy = self.constraint_energy(inp, out, t)  # (B, 1)
        
        if return_energy:
            return energy
        
        # Compute gradient dE/dout using autograd
        if not torch.isfinite(energy).all():
            logger.warning("ConstraintDiffusionWrapper: Non-finite energy detected, returning zero gradient")
            grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        else:
            try:
                grad = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=out,
                    create_graph=True
                )[0]  # (B, out_dim)
                
                # Verify gradient is finite
                if not torch.isfinite(grad).all():
                    logger.warning("ConstraintDiffusionWrapper: Non-finite gradient computed, using zero gradient")
                    grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
                    
            except RuntimeError as e:
                logger.error(f"ConstraintDiffusionWrapper: Gradient computation failed: {e}")
                grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        
        if return_both:
            return energy, grad
        else:
            return grad