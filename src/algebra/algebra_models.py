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
from src.models import SinusoidalPosEmb, swish


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
        enable_magnitude_clipping: Whether to apply magnitude clipping to prevent extreme energy values (default: True)
    """
    
    def __init__(
        self, 
        inp_dim: int = 128, 
        out_dim: int = 128,
        rule_name: Optional[str] = None,
        enable_magnitude_clipping: bool = True
    ):
        super(AlgebraEBM, self).__init__()
        
        # Store dimensions and rule info
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.rule_name = rule_name
        self.enable_magnitude_clipping = enable_magnitude_clipping
        
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
        
        # Learnable energy scaling to match contrastive loss targets
        # This allows the model to output energies in the target range (pos~1, neg~10)
        # With xavier init and normalized inputs, raw energies will be ~0.5-2.0
        # We start with small scale and let learning adjust
        self.energy_scale = nn.Parameter(torch.tensor(1.0))   # Start at 1.0, learn the scale
        self.energy_bias = nn.Parameter(torch.tensor(0.0))    # Start at 0, learn the bias
        
        # Apply proper weight initialization to prevent flat energy landscapes
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for good energy landscape formation.
        
        Key insights:
        1. With normalized inputs (||x||=1), we need controlled scaling
        2. FiLM layers must start near-identity to not overwhelm input signal
        3. fc4 uses smaller init so raw energy starts in reasonable range
        
        Critical Fix: FiLM bias initialization was dominating input signal.
        When FiLM bias std (0.24) >> hidden state std (0.04), the time
        conditioning overwhelms the actual input-dependent information.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == 'fc4':
                    # Smaller init for output layer to keep raw energies bounded
                    # Want ||fc4(h)||^2 ~ 1-5 initially, so ||fc4(h)|| ~ 1-2
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif name in ['fc1', 'fc2', 'fc3']:
                    # Standard xavier for hidden layers
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif name in ['t_map_fc2', 't_map_fc3']:
                    # CRITICAL: Initialize FiLM layers to near-identity
                    # Output is [gain, bias] where applied as: h * (gain + 1) + bias
                    # We want gain ≈ 0 and bias ≈ 0 initially so FiLM is near-identity
                    # Use very small init so FiLM doesn't dominate input signal
                    nn.init.normal_(module.weight, std=0.01)  # Very small weights
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)  # Zero bias means gain≈0, shift≈0
                else:
                    # Default init for time MLP layers
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
    def forward(
        self, 
        inp: torch.Tensor, 
        out: torch.Tensor, 
        t: torch.Tensor,
        return_energy: bool = False,
        return_both: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Forward pass computing energy for (inp, out) pair at timestep t.
        
        Args:
            inp: Input equation embedding (B, inp_dim)
            out: Output equation embedding (B, out_dim)  
            t: Diffusion timestep (B,) in range [0, 9]
            return_energy: If True, return energy (default: True for compatibility)
            return_both: If True, return (energy, None) for interface compatibility
            
        Returns:
            energy: Non-negative energy scalar (B, 1) [default and return_energy=True]
            (energy, None): Tuple for interface compatibility [if return_both=True]
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
        
        # Output layer with proper numerical conditioning
        output = self.fc4(h)  # (B, out_dim)
        
        # Apply numerical conditioning to prevent problematic values before energy computation
        # This addresses root causes rather than patching symptoms after energy computation
        if not torch.isfinite(output).all():
            logger = logging.getLogger(__name__)
            logger.warning("AlgebraEBM detected non-finite output values, applying input conditioning")
            # Apply conditioning to problematic outputs while preserving valid ones
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
        # Apply gradient-preserving bounds to prevent extreme energy values (if enabled)
        if self.enable_magnitude_clipping:
            # Use soft clipping that maintains differentiability and only affects large values
            output_magnitude = torch.norm(output, dim=-1, keepdim=True)
            max_magnitude = 1000.0  # Corresponds to energy ~1e6 (1000^2)
            if output_magnitude.max() > max_magnitude:
                # Soft conditioning: only scale down values that exceed the threshold
                # Use smooth transition that preserves gradients and maintains relative ordering
                # For magnitude > max_magnitude: smoothly scale down to max_magnitude
                # For magnitude <= max_magnitude: leave unchanged (scale_factor = 1)
                excess_ratio = (output_magnitude - max_magnitude) / max_magnitude
                scale_factor = torch.where(
                    output_magnitude > max_magnitude,
                    max_magnitude / (output_magnitude + 1e-8),  # Scale large values down to threshold
                    torch.ones_like(output_magnitude)  # Leave normal values unchanged
                )
                output = output * scale_factor
        
        # Energy = scale * ||output_vector||^2 + bias (learnable to match contrastive targets)
        # Base energy from L2 norm squared
        raw_energy = output.pow(2).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # Apply learnable scaling to match contrastive loss target range (~1 to ~15)
        # This is critical: without scaling, energies are stuck at ~0.2 with normalized inputs
        energy = self.energy_scale * raw_energy + self.energy_bias
        
        # Energy statistics monitoring for debugging and analysis (only in DEBUG mode)
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            energy_stats = {
                'min': energy.min().item(),
                'max': energy.max().item(),
                'mean': energy.mean().item(),
                'std': energy.std().item()
            }
            logger.debug(f"AlgebraEBM energy stats: min={energy_stats['min']:.6e}, "
                        f"max={energy_stats['max']:.6e}, mean={energy_stats['mean']:.6e}, "
                        f"std={energy_stats['std']:.6e}, clipping={'enabled' if self.enable_magnitude_clipping else 'disabled'}")
            
            # Detect flat energy landscape - but only warn if batch has varied outputs
            # (identical inputs in a batch legitimately produce identical energies)
            if energy.shape[0] > 1 and energy_stats['std'] < 1e-6:
                # Check if outputs are also identical (expected) vs model ignoring output (bug)
                output_std = output.std().item()
                if output_std > 1e-4:  # Outputs vary but energies don't - this is the real bug
                    logger.warning(f"FLAT ENERGY LANDSCAPE: outputs vary (std={output_std:.6f}) but energies identical")
        
        # Numerical stability monitoring - detect Inf/NaN values in energy
        if not torch.isfinite(energy).all():
            logger = logging.getLogger(__name__)
            inf_count = torch.isinf(energy).sum().item()
            nan_count = torch.isnan(energy).sum().item()
            finite_count = torch.isfinite(energy).sum().item()
            batch_size = energy.shape[0]
            
            logger.warning(f"AlgebraEBM numerical instability detected: "
                          f"batch_size={batch_size}, finite={finite_count}, "
                          f"inf={inf_count}, nan={nan_count}, "
                          f"clipping={'enabled' if self.enable_magnitude_clipping else 'disabled'}")
            
            # Additional debug info for troubleshooting
            if logger.isEnabledFor(logging.DEBUG):
                output_magnitude = torch.norm(output, dim=-1, keepdim=True)
                logger.debug(f"Output magnitude stats: min={output_magnitude.min().item():.6e}, "
                           f"max={output_magnitude.max().item():.6e}, "
                           f"output_finite={torch.isfinite(output).all().item()}")
        
        # Performance optimization: fast path for normal cases (preserving existing optimization)
        energy_max = energy.max().item()
        import math
        if energy_max <= 1e6 and not math.isnan(energy_max):
            # Normal case - properly conditioned input leads to stable energy
            pass
        else:
            # This should rarely occur with proper conditioning above
            logger = logging.getLogger(__name__)
            logger.warning(f"AlgebraEBM: Unexpected large energy after conditioning: {energy_max:.2e}")
        
        # Handle return format based on arguments
        if return_both:
            # Return (energy, None) for interface compatibility with wrapper
            return energy, None
        else:
            # Default behavior: always return energy for AlgebraEBM
            # (return_energy parameter is ignored since this is an energy model)
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
        
        # CRITICAL FIX: Properly enable gradient computation for output
        # The input tensor from dataloader is detached, so we must clone it first
        # before requiring gradients. Otherwise autograd.grad() fails silently.
        if not out.requires_grad:
            out = out.detach().clone().requires_grad_(True)
        else:
            # If already requires grad, ensure we have a fresh computation graph
            out = out.clone().requires_grad_(True)
        
        # Compute energy E(inp, out, t)
        energy = self.ebm(inp, out, t)  # (B, 1)
        
        if return_energy:
            return energy
        
        # Compute gradient dE/dout using autograd with numerical stability protection
        # Early detection of problematic energy values before expensive gradient computation
        if not torch.isfinite(energy).all():
            logger = logging.getLogger(__name__)
            logger.warning("AlgebraDiffusionWrapper: Non-finite energy detected, returning zero gradient")
            grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        else:
            try:
                # create_graph=True enables backpropagation through gradients
                grad = torch.autograd.grad(
                    outputs=energy.sum(),     # Sum for scalar loss
                    inputs=out,               # Gradient w.r.t. output
                    create_graph=True         # Enable higher-order gradients
                )[0]  # (B, out_dim)
                
                # Verify gradient is finite (additional safety check)
                if not torch.isfinite(grad).all():
                    logger = logging.getLogger(__name__)
                    logger.warning("AlgebraDiffusionWrapper: Non-finite gradient computed, using zero gradient")
                    grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
                    
            except RuntimeError as e:
                # Handle gradient computation failures gracefully
                logger = logging.getLogger(__name__)
                logger.error(f"AlgebraDiffusionWrapper: Gradient computation failed: {e}")
                logger.error(f"Energy stats: min={energy.min().item():.6e}, max={energy.max().item():.6e}")
                grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        
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
    
    def __init__(self, margin: float = 5.0, pos_target: float = 1.0, neg_target: float = 10.0):
        # Adjusted defaults for better convergence:
        # - pos_target=1.0: Valid transformations should have low energy
        # - neg_target=10.0: Invalid transformations should have higher energy (reduced from 15)
        # - margin=5.0: Required gap between neg and pos (reduced from 10 for faster learning)
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
        """Get recent energy gap statistics for monitoring.
        
        Returns:
            Dictionary with keys:
            - gap_mean: mean energy gap
            - gap_std: standard deviation (NaN if insufficient data)
            - ratio_mean: mean energy ratio  
            - ratio_std: standard deviation (NaN if insufficient data)
            - sample_count: number of samples used
            
        Note: std returns NaN for single-element case (mathematically undefined).
        Consumers should check math.isnan() or sample_count < 2 before using std values.
        """
        if not self.energy_gap_history:
            return {
                'gap_mean': 0.0, 
                'gap_std': float('nan'), 
                'ratio_mean': 1.0,
                'ratio_std': float('nan'),
                'sample_count': 0
            }
        
        gaps = torch.tensor(self.energy_gap_history, dtype=torch.float32)
        pos_energies = torch.tensor(self.pos_energy_history, dtype=torch.float32)
        neg_energies = torch.tensor(self.neg_energy_history, dtype=torch.float32)
        
        # Compute ratios with numerical safety
        ratios = neg_energies / torch.clamp(pos_energies, min=1e-6)
        
        # Return NaN for insufficient data (mathematically correct)
        # This distinguishes 'no variance' (std=0.0) from 'insufficient data' (std=NaN)
        gap_std = gaps.std().item() if len(gaps) > 1 else float('nan')
        ratio_std = ratios.std().item() if len(ratios) > 1 else float('nan')
        
        return {
            'gap_mean': gaps.mean().item(),
            'gap_std': gap_std,
            'ratio_mean': ratios.mean().item(),
            'ratio_std': ratio_std,
            'sample_count': len(gaps),
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