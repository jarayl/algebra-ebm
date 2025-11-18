"""
Algebra Energy-Based Models (AlgebraEBM)

Implements energy models for algebraic reasoning using the IRED framework.
Core models for learning rule-specific energy functions that can be composed
at inference time for zero-shot multi-rule generalization.

Classes:
- AlgebraEBM: Energy function for algebraic rule validity (matches IRED Table 8)
- AlgebraDiffusionWrapper: Diffusion wrapper for gradient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Import utilities from existing models
from models import SinusoidalPosEmb, swish


class AlgebraEBM(nn.Module):
    """
    Energy-Based Model for algebraic rule validity.
    
    Architecture matches IRED Table 8 specification:
    - Time MLP: SinusoidalPosEmb(128) → Linear(128) → GELU → Linear(128)
    - FC1: Linear(inp_dim + out_dim → 512) + Swish  
    - FC2: Linear(512 → 512) + FiLM(time_emb) + Swish
    - FC3: Linear(512 → 512) + FiLM(time_emb) + Swish
    - Output: Linear(512 → out_dim), energy = ||output_vector||^2
    
    Args:
        inp_dim: Input equation embedding dimension (default: 128)
        out_dim: Output equation embedding dimension (default: 128)  
        rule_name: Name of algebraic rule ('distribute', 'combine', 'isolate', 'divide')
    """
    
    def __init__(
        self, 
        inp_dim: int = 128, 
        out_dim: int = 128,
        rule_name: Optional[str] = None
    ):
        super(AlgebraEBM, self).__init__()
        
        # Store dimensions and rule info
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.rule_name = rule_name
        
        # Architecture parameters matching IRED Table 8
        fourier_dim = 128  # For SinusoidalPosEmb
        time_dim = 128     # Time embedding dimension
        hidden_dim = 512   # Hidden layer width
        
        # Time MLP: SinusoidalPosEmb(128) → Linear(128) → GELU → Linear(128)
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main architecture layers
        self.fc1 = nn.Linear(inp_dim + out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        
        # FiLM conditioning layers for timestep injection
        self.t_map_fc2 = nn.Linear(time_dim, 2 * hidden_dim)  # scale + bias for FC2
        self.t_map_fc3 = nn.Linear(time_dim, 2 * hidden_dim)  # scale + bias for FC3
        
    def forward(self, inp: torch.Tensor, out: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing energy for (inp, out) pair at timestep t.
        
        Args:
            inp: Input equation embedding (B, inp_dim)
            out: Output equation embedding (B, out_dim)  
            t: Diffusion timestep (B,) in range [0, 9]
            
        Returns:
            energy: Non-negative energy scalar (B, 1)
        """
        # Input validation
        assert inp.shape[-1] == self.inp_dim, f"Expected inp_dim={self.inp_dim}, got {inp.shape[-1]}"
        assert out.shape[-1] == self.out_dim, f"Expected out_dim={self.out_dim}, got {out.shape[-1]}"
        assert inp.shape[0] == out.shape[0] == t.shape[0], "Batch sizes must match"
        
        # Concatenate input and output embeddings
        x = torch.cat([inp, out], dim=-1)  # (B, inp_dim + out_dim)
        
        # Compute time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)
        
        # Extract FiLM parameters for conditioning
        fc2_params = self.t_map_fc2(t_emb)  # (B, 2 * hidden_dim)
        fc2_gain, fc2_bias = torch.chunk(fc2_params, 2, dim=-1)  # Each (B, hidden_dim)
        
        fc3_params = self.t_map_fc3(t_emb)  # (B, 2 * hidden_dim)
        fc3_gain, fc3_bias = torch.chunk(fc3_params, 2, dim=-1)  # Each (B, hidden_dim)
        
        # Forward through layers with FiLM conditioning
        h = swish(self.fc1(x))  # FC1: Linear + Swish
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)  # FC2: Linear + FiLM + Swish
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)  # FC3: Linear + FiLM + Swish
        
        # Output layer and energy computation
        output = self.fc4(h)  # (B, out_dim)
        
        # Energy = ||output_vector||^2 (L2 norm squared for non-negative energy)
        energy = output.pow(2).sum(dim=-1, keepdim=True)  # (B, 1)
        
        return energy


class AlgebraDiffusionWrapper(nn.Module):
    """
    Diffusion wrapper for AlgebraEBM that computes energy gradients.
    
    Interfaces AlgebraEBM with IRED's GaussianDiffusion1D by computing
    gradients dE/dout needed for score matching and inference optimization.
    
    Args:
        ebm: AlgebraEBM instance to wrap
    """
    
    def __init__(self, ebm: AlgebraEBM):
        super(AlgebraDiffusionWrapper, self).__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim
        
        # Store rule name for identification
        if hasattr(ebm, 'rule_name'):
            self.rule_name = ebm.rule_name
    
    def forward(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        t: torch.Tensor,
        return_energy: bool = False,
        return_both: bool = False
    ) -> torch.Tensor:
        """
        Compute energy gradients for diffusion training/inference.
        
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
        out = out.requires_grad_(True)
        
        # Compute energy E(inp, out, t)
        energy = self.ebm(inp, out, t)  # (B, 1)
        
        if return_energy:
            return energy
        
        # Compute gradient dE/dout using autograd
        # create_graph=True enables backpropagation through gradients
        grad = torch.autograd.grad(
            outputs=energy.sum(),     # Sum for scalar loss
            inputs=out,               # Gradient w.r.t. output
            create_graph=True         # Enable higher-order gradients
        )[0]  # (B, out_dim)
        
        if return_both:
            return energy, grad
        else:
            return grad