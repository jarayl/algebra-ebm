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
from typing import Tuple, Optional, Union, Dict
from collections import deque
import logging

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
        
        # Optimized numerical stability with fast path for common case
        # Fast path: check energy range first to avoid expensive element-wise operations
        energy_max = energy.max().item()
        energy_min = energy.min().item()
        
        # Fast path: if values are in normal range, skip detailed checks (5-10x faster)  
        # Note: energy from ||x||^2 is always >= 0, and max/min are NaN if any element is NaN
        if energy_min >= 0.0 and energy_max <= 1e6 and not (torch.isnan(energy_max) or torch.isnan(energy_min)):
            # Normal case - no intervention needed, values are finite and in acceptable range
            pass
        else:
            # Detailed intervention needed - use original expensive checks
            logger = logging.getLogger(__name__)
            
            # Handle NaN/Inf cases (should be rare in normal operation)
            if not torch.isfinite(energy).all():
                logger.warning("AlgebraEBM detected non-finite energy values, applying numerical stabilization")
                # Use median of valid energies if available, else safe fallback
                finite_mask = torch.isfinite(energy)
                if finite_mask.any():
                    fallback_value = energy[finite_mask].median()
                else:
                    fallback_value = torch.tensor(100.0, device=energy.device)
                energy = torch.where(finite_mask, energy, fallback_value)
            
            # Apply log-based soft limiting for extreme values
            if energy_max > 1e6:
                logger.debug(f"AlgebraEBM applying soft limiting for extreme energy max: {energy_max:.2e}")
                extreme_mask = energy > 1e6
                # Soft log-based clamping: log(1 + (x - 1e6)) + 1e6
                # Preserves gradients and maintains energy ordering
                energy = torch.where(
                    extreme_mask,
                    torch.log1p(energy - 1e6) + 1e6,
                    energy
                )
        
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


class ContrastiveEnergyLoss:
    """
    Contrastive energy loss for training algebraic EBMs.
    
    Implements proper energy supervision where:
    - E_pos (valid transformations) should have LOW energy (< margin/2)
    - E_neg (invalid transformations) should have HIGH energy (> margin) 
    - Energy gap E_neg - E_pos should be >= margin for good separation
    
    Args:
        margin: Energy separation margin (default: 10.0)
        pos_target: Target energy for positive samples (default: 1.0)
        neg_target: Target energy for negative samples (default: 15.0)
    """
    
    def __init__(self, margin: float = 10.0, pos_target: float = 1.0, neg_target: float = 15.0):
        self.margin = margin
        self.pos_target = pos_target
        self.neg_target = neg_target
        
        # Track energy gap statistics for monitoring (bounded history to prevent memory leaks)
        self.energy_gap_history = deque(maxlen=1000)
        self.pos_energy_history = deque(maxlen=1000)
        self.neg_energy_history = deque(maxlen=1000)
    
    def compute_loss(
        self,
        pos_energies: torch.Tensor,
        neg_energies: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute contrastive energy loss.
        
        Args:
            pos_energies: Energies for positive (valid) samples (B_pos, 1)
            neg_energies: Energies for negative (invalid) samples (B_neg, 1)
            return_metrics: If True, return additional metrics dict
            
        Returns:
            loss: Contrastive loss value
            metrics: Dict with energy statistics (if return_metrics=True)
        """
        # Input validation - cannot compute contrastive loss without both positive and negative samples
        if pos_energies.numel() == 0 or neg_energies.numel() == 0:
            # Fail fast for empty batches to prevent silent training failure
            error_msg = (
                f"ContrastiveEnergyLoss received empty batch: "
                f"pos_energies={pos_energies.numel()}, neg_energies={neg_energies.numel()}. "
                f"This indicates a data pipeline issue and prevents gradient computation. "
                f"Training cannot proceed with empty batches."
            )
            raise ValueError(error_msg)
        
        # L2 loss pushing positive energies toward low target
        pos_loss = F.mse_loss(pos_energies, torch.full_like(pos_energies, self.pos_target))
        
        # L2 loss pushing negative energies toward high target  
        neg_loss = F.mse_loss(neg_energies, torch.full_like(neg_energies, self.neg_target))
        
        # Margin loss ensuring separation E_neg > E_pos + margin
        pos_mean = pos_energies.mean()
        neg_mean = neg_energies.mean()
        energy_gap = neg_mean - pos_mean
        margin_loss = F.relu(self.margin - energy_gap)
        
        # Combined loss with equal weighting
        total_loss = pos_loss + neg_loss + margin_loss
        
        # Update monitoring statistics (deque automatically maintains size limit)
        self.energy_gap_history.append(energy_gap.item())
        self.pos_energy_history.append(pos_mean.item())
        self.neg_energy_history.append(neg_mean.item())
        
        if return_metrics:
            metrics = {
                'energy_gap': energy_gap.item(),
                'pos_energy_mean': pos_mean.item(),
                'neg_energy_mean': neg_mean.item(),
                'pos_loss': pos_loss.item(),
                'neg_loss': neg_loss.item(),
                'margin_loss': margin_loss.item(),
                'energy_ratio': (neg_mean / torch.clamp(pos_mean, min=1e-6)).item()
            }
            return total_loss, metrics
        else:
            return total_loss
    
    def get_energy_gap_stats(self) -> Dict[str, float]:
        """Get recent energy gap statistics for monitoring."""
        if not self.energy_gap_history:
            return {'gap_mean': 0.0, 'gap_std': 0.0, 'ratio_mean': 1.0}
        
        gaps = torch.tensor(self.energy_gap_history)
        pos_energies = torch.tensor(self.pos_energy_history)
        neg_energies = torch.tensor(self.neg_energy_history)
        
        # Compute ratios with numerical safety
        ratios = neg_energies / torch.clamp(pos_energies, min=1e-6)
        
        return {
            'gap_mean': gaps.mean().item(),
            'gap_std': gaps.std().item(),
            'ratio_mean': ratios.mean().item(),
            'ratio_std': ratios.std().item(),
            'success_rate': (gaps >= self.margin).float().mean().item()
        }
    
    def is_well_separated(self, threshold_ratio: float = 5.0) -> bool:
        """
        Check if energy gap indicates good contrastive learning.
        
        Args:
            threshold_ratio: Required E_neg/E_pos ratio for success
            
        Returns:
            True if recent energy gaps indicate good separation
        """
        stats = self.get_energy_gap_stats()
        return stats['ratio_mean'] >= threshold_ratio and stats['success_rate'] >= 0.8