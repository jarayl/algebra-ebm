"""
Algebraic IRED-Style Inference

Implements the IRED annealed gradient descent inference procedure for solving
algebraic equations through energy landscape optimization. Supports composition
of multiple rule-specific energy functions for zero-shot multi-rule generalization.

Core Algorithm:
1. Initialize from noise: out = torch.randn(128)  
2. Iterate through K=10 landscapes with cosine schedule
3. For each landscape, do T=20 gradient steps
4. Compose multiple rule energies by summing
5. Energy-based acceptance criteria for stability
6. Proper landscape scaling: out *= (sigma_k_next / sigma_k)

Example Usage:
    # Load trained rule EBMs
    rule_models = load_rule_models(['distribute', 'combine', 'isolate', 'divide'])
    
    # Create inference engine
    inference = AlgebraInference(rule_models, encoder, decoder)
    
    # Solve equation
    result = inference.solve_equation("2*(x+3)+4=10", max_steps=20)
"""

import torch
import math
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import deque

# Import existing components
from src.algebra.algebra_encoder import CharacterLevelEncoder, ASTEncoder, EquationDecoder
from src.algebra.algebra_models import AlgebraEBM, AlgebraDiffusionWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """
    Configuration for IRED algebra inference parameters.
    
    These parameters control the annealed gradient descent process used
    for solving algebraic equations through energy landscape optimization.
    """
    
    # Core inference parameters  
    step_size: float = 0.01  # Gradient descent step size (reduced from 0.1 for stability)
    max_iterations: int = 50  # Gradient steps per landscape (increased from 20 for convergence)
    K: int = 10  # Number of energy landscapes to traverse
    
    # Advanced parameters
    use_adaptive_step: bool = True  # Whether to adapt step size per landscape
    energy_threshold: float = 1e-6  # Early stopping threshold for very low energy
    
    # Adaptive step size parameters
    step_size_decay_rate: float = 0.7  # Exponential decay rate for adaptive step sizing
    step_size_decay_interval: int = 3  # Apply decay every N landscapes
    
    # TODO: Future safety features (currently not implemented to avoid "security theater")
    # 
    # Previously had unused parameters: max_gradient_norm, energy_bounds, convergence_threshold, min_improvement_steps
    # These were validated but never enforced in the inference logic, creating false sense of security.
    # 
    # If implementing these features in the future:
    # - energy_bounds: Should be optional and disabled by default for EBMs (need unbounded energy differences)
    # - max_gradient_norm: Would require modification of optimization loop with gradient clipping
    # - convergence_threshold/min_improvement_steps: Would require tracking energy changes across iterations
    # - All should have clear enforcement mechanisms, not just validation
    #
    # Design principle: Better to have no safety feature than one that doesn't work
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.K > 10000:
            raise ValueError(
                f"K exceeds maximum of 10000 (got {self.K}). "
                f"Large K causes memory exhaustion during step size precomputation. "
                f"Typical use: K=5-20 landscapes, maximum supported: K=10000."
            )
        if self.energy_threshold < 0:
            raise ValueError(f"energy_threshold must be non-negative, got {self.energy_threshold}")
        if self.step_size_decay_rate <= 0 or self.step_size_decay_rate >= 1:
            raise ValueError(
                f"step_size_decay_rate must be in (0, 1), got {self.step_size_decay_rate}. "
                f"Value of 1.0 disables decay (no-op). Use use_adaptive_step=False instead."
            )
        if self.step_size_decay_interval <= 0:
            raise ValueError(f"step_size_decay_interval must be positive, got {self.step_size_decay_interval}")
        
        # Precompute step sizes for performance optimization
        if self.use_adaptive_step:
            self._step_sizes = [
                self.step_size * (self.step_size_decay_rate ** (k // self.step_size_decay_interval)) 
                for k in range(self.K)
            ]
        else:
            self._step_sizes = [self.step_size] * self.K
        
        logger.debug(f"InferenceConfig validated: step_size={self.step_size}, max_iterations={self.max_iterations}")
    
    def get_adaptive_step_size(self, landscape_idx: int) -> float:
        """Get step size for a specific landscape, optionally with adaptation.
        
        Uses exponential decay: step_size * (decay_rate ** (landscape_idx // interval))
        This gradually reduces step size to enable fine-grained optimization in later landscapes.
        
        Args:
            landscape_idx: Index of current landscape (0 to K-1)
            
        Returns:
            Step size for this landscape
        """
        if landscape_idx < 0 or landscape_idx >= len(self._step_sizes):
            raise IndexError(f"landscape_idx {landscape_idx} out of bounds [0, {len(self._step_sizes)})")
        return self._step_sizes[landscape_idx]
    
    def should_early_stop(self, energy: float) -> bool:
        """Check if energy is low enough for early stopping."""
        return energy < self.energy_threshold
    


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for diffusion noise as used in IRED.
    
    Args:
        timesteps: Number of timesteps (K landscapes)
        s: Small offset for numerical stability
        
    Returns:
        betas: Beta schedule tensor of shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def compute_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Compute cumulative alpha values from cosine schedule.
    
    Args:
        timesteps: Number of timesteps 
        s: Small offset parameter
        
    Returns:
        alphas_cumprod: Cumulative alpha values of shape (timesteps,)
    """
    betas = cosine_beta_schedule(timesteps, s)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


class AlgebraInference:
    """
    IRED-style inference engine for algebraic equation solving.
    
    Implements annealed gradient descent through K energy landscapes,
    with support for compositional energy summation across multiple rules.
    
    Args:
        rule_models: Dict mapping rule names to trained EBM models
        encoder: Equation encoder (CharacterLevelEncoder or ASTEncoder) 
        decoder: Equation decoder for converting embeddings back to strings
        config: InferenceConfig with optimization parameters (uses defaults if None)
        device: Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        rule_models: Dict[str, AlgebraDiffusionWrapper],
        encoder: Union[CharacterLevelEncoder, ASTEncoder],
        decoder: Optional[EquationDecoder] = None,
        config: Optional[InferenceConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.rule_models = rule_models
        self.encoder = encoder
        self.decoder = decoder
        self.config = config if config is not None else InferenceConfig()
        self.K = self.config.K  # For backward compatibility
        self.device = device
        
        # Move models and encoder to device
        for model in self.rule_models.values():
            model.to(device)
        try:
            self.encoder.to(device)
        except Exception as e:
            logger.warning(f"Could not move encoder to device {device}: {e}")
            # Encoder might not support .to() method, that's ok
        
        # Compute cosine schedule for landscape scaling
        self.alphas_cumprod = compute_alphas_cumprod(self.config.K).to(device)
        
        # Set models to evaluation mode
        for model in self.rule_models.values():
            model.eval()
        self.encoder.eval()
        
        logger.info(f"Initialized AlgebraInference with {len(rule_models)} rule models on {device}")
    
    def compose_energies(
        self,
        inp: torch.Tensor,
        out: torch.Tensor, 
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compose energy functions from multiple rules by weighted summation.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128)
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule (default: all 1.0)
            t: Optional pre-allocated timestep tensor (default: allocate new)
            
        Returns:
            total_energy: Composed energy value (B, 1)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        total_energy = 0.0
        # Use pre-allocated tensor if provided, otherwise allocate new one
        if t is None:
            t = torch.full((inp.shape[0],), k, dtype=torch.long, device=inp.device)
        else:
            # Validate pre-allocated tensor is on correct device
            if t.device != inp.device:
                raise ValueError(f"Pre-allocated tensor device {t.device} does not match input device {inp.device}")
        
        for rule_name, model in self.rule_models.items():
            weight = rule_weights.get(rule_name, 1.0)
            energy = model(inp, out, t, return_energy=True)  # (B, 1)
            total_energy += weight * energy

        # Normalize by number of rules to keep gradient magnitudes stable
        # regardless of composition size (prevents gradient explosions in multi-rule)
        num_rules = len(self.rule_models)
        if num_rules > 1:
            total_energy = total_energy / num_rules

        return total_energy
    
    def compute_composed_gradient(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient of composed energy w.r.t. output embedding.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128) 
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule
            t: Optional pre-allocated timestep tensor (default: allocate new)
            
        Returns:
            grad: Energy gradient dE/dout (B, 128)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        # Enable gradient computation
        out = out.requires_grad_(True)
        
        # Compute composed energy
        total_energy = self.compose_energies(inp, out, k, rule_weights, t)
        
        # Compute gradient with numerical stability protection
        # Early detection of problematic energy values before expensive gradient computation
        if not torch.isfinite(total_energy).all():
            logger.warning("AlgebraInference: Non-finite energy detected in composed gradient, returning zero gradient")
            grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        else:
            try:
                grad = torch.autograd.grad(
                    outputs=total_energy.sum(),
                    inputs=out,
                    create_graph=False
                )[0]

                # Verify gradient is finite (additional safety check)
                if not torch.isfinite(grad).all():
                    logger.warning("AlgebraInference: Non-finite gradient computed in composed gradient, using zero gradient")
                    grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)

            except RuntimeError as e:
                # Handle gradient computation failures gracefully
                logger.error(f"AlgebraInference: Composed gradient computation failed: {e}")
                logger.error(f"Energy stats: min={total_energy.min().item():.6e}, max={total_energy.max().item():.6e}")
                grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)

        return grad
    
    def compute_energy_and_gradient(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights: Optional[Dict[str, float]] = None,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both composed energy and gradient in a single forward pass.
        
        This optimizes performance by avoiding redundant energy computation.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128) 
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule
            t: Optional pre-allocated timestep tensor (default: allocate new)
            
        Returns:
            energy: Composed energy value (B, 1)
            grad: Energy gradient dE/dout (B, 128)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        # Enable gradient computation
        out = out.requires_grad_(True)
        
        # Compute composed energy
        total_energy = self.compose_energies(inp, out, k, rule_weights, t)
        
        # Compute gradient with numerical stability protection
        # Early detection of problematic energy values before expensive gradient computation
        if not torch.isfinite(total_energy).all():
            logger.warning("AlgebraInference: Non-finite energy detected in energy+gradient computation, returning zero gradient")
            grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)
        else:
            try:
                grad = torch.autograd.grad(
                    outputs=total_energy.sum(),
                    inputs=out,
                    create_graph=False
                )[0]

                # Verify gradient is finite (additional safety check)
                if not torch.isfinite(grad).all():
                    logger.warning("AlgebraInference: Non-finite gradient computed in energy+gradient computation, using zero gradient")
                    grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)

            except RuntimeError as e:
                # Handle gradient computation failures gracefully
                logger.error(f"AlgebraInference: Energy+gradient computation failed: {e}")
                logger.error(f"Energy stats: min={total_energy.min().item():.6e}, max={total_energy.max().item():.6e}")
                grad = torch.zeros_like(out, device=out.device, dtype=out.dtype)

        return total_energy, grad
    
    def ired_inference(
        self,
        inp_embedding: torch.Tensor,
        config: Optional[InferenceConfig] = None,
        rule_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Core IRED inference algorithm with annealed gradient descent.
        
        Args:
            inp_embedding: Input equation embedding (B, 128)  
            config: InferenceConfig with optimization parameters (uses self.config if None)
            rule_weights: Optional weights for rule composition
            
        Returns:
            out_embedding: Final optimized embedding (B, 128)
            info: Dictionary with optimization statistics
        """
        # Use provided config or fall back to instance config
        if config is None:
            config = self.config
        else:
            # Validate that runtime config.K matches precomputed alphas_cumprod
            if config.K != len(self.alphas_cumprod):
                raise ValueError(
                    f"config.K={config.K} does not match precomputed alphas_cumprod length={len(self.alphas_cumprod)}. "
                    f"Either use instance config (K={self.config.K}) or reinitialize AlgebraInference with new config."
                )
            
        # Input validation
        if len(inp_embedding.shape) != 2 or inp_embedding.shape[1] != 128:
            raise ValueError(f"inp_embedding must have shape (B, 128), got {inp_embedding.shape}")
        if len(self.rule_models) == 0:
            raise ValueError("No rule models loaded - cannot perform inference")
        
        # Enhanced validation for production robustness
        if not torch.is_tensor(inp_embedding):
            raise TypeError(f"inp_embedding must be torch.Tensor, got {type(inp_embedding)}")
        if torch.isnan(inp_embedding).any():
            raise ValueError("inp_embedding contains NaN values")
        if torch.isinf(inp_embedding).any():
            raise ValueError("inp_embedding contains Inf values")
        
        batch_size = inp_embedding.shape[0]
        
        # Initialize from noise  
        out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
        
        # Track optimization statistics (bounded by config parameters)
        info = {
            'energy_history': [],
            'step_sizes': [],
            'landscape_transitions': [],
            'gradient_norms': [],
            'accepted_steps': 0,
            'total_steps': 0
            # config_used removed to eliminate unnecessary overhead
            # If needed for debugging, can be added behind a config.include_debug_info flag
        }
        
        # OPTIMIZATION: Initialize energy caching variables
        have_cached_energy = False
        cached_energy_val = None
        
        # OPTIMIZATION: Pre-allocate all timestep tensors to reduce memory allocation overhead
        timestep_tensors = {}
        for k_idx in range(config.K):
            timestep_tensors[k_idx] = torch.full((batch_size,), k_idx, dtype=torch.long, device=inp_embedding.device)
        
        # OPTIMIZATION: Track caching effectiveness
        cache_hits = 0
        cache_misses = 0
        
        # Scale step size for multi-rule composition to account for gradient variance growth
        num_rules = len(self.rule_models)
        composition_scale = 1.0 / math.sqrt(num_rules) if num_rules > 1 else 1.0

        # Iterate through K landscapes
        for k in range(config.K):
            sigma_k = torch.sqrt(1 - self.alphas_cumprod[k]).item()

            # Adaptive step size using config method, scaled for composition
            current_step_size = config.get_adaptive_step_size(k) * composition_scale
            info['step_sizes'].append(current_step_size)
            
            logger.debug(f"Landscape {k}, sigma_k={sigma_k:.4f}, step_size={current_step_size:.4f}")
            
            # Use pre-allocated timestep tensor (tensor pre-allocation optimization)
            timestep_tensor = timestep_tensors[k]
            
            # OPTIMIZATION: Reset energy cache when starting new landscape
            have_cached_energy = False
            cached_energy_val = None
            
            # max_iterations gradient descent steps in this landscape
            for t in range(config.max_iterations):
                # OPTIMIZATION: Energy caching to avoid redundant forward passes
                # Key insight: if we have cached energy from previous iteration, reuse it
                if have_cached_energy:
                    # Use cached energy from previous iteration
                    energy_before_val = cached_energy_val
                    grad = self.compute_composed_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                    cache_hits += 1
                else:
                    # No cached energy - compute both energy and gradient atomically
                    energy_current, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights, timestep_tensor)
                    # Handle batched energy tensors: take mean across batch for tracking
                    energy_before_val = energy_current.mean().item()
                    cache_misses += 1
                grad_norm = torch.norm(grad).item()
                info['energy_history'].append(energy_before_val)
                info['gradient_norms'].append(grad_norm)

                # Gradient norm clipping: clip large gradients instead of stopping
                max_grad_norm = 10.0
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)
                    logger.debug(f"Gradient clipped at landscape {k}, step {t}: "
                                 f"{grad_norm:.2e} -> {max_grad_norm:.2e}")
                
                # Energy stagnation detection (check last few steps)
                if len(info['energy_history']) >= 10:
                    recent_energies = info['energy_history'][-10:]
                    energy_std = torch.tensor(recent_energies).std().item()
                    if energy_std < 1e-6 and grad_norm < 1e-4:
                        logger.info(f"Convergence detected at landscape {k}, step {t}: "
                                   f"energy_std={energy_std:.2e}, grad_norm={grad_norm:.2e}")
                        info['convergence_reason'] = f'converged_k{k}_t{t}'
                        break
                
                # Gradient descent step
                out_new = out - current_step_size * grad
                
                # Metropolis acceptance criteria with temperature schedule
                energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights, timestep_tensor)
                # Handle batched energy tensors: take mean across batch for comparison
                energy_after_val = energy_after.mean().item()
                delta_E = energy_after_val - energy_before_val
                
                # Temperature schedule constants (extracted per maintainability requirements)
                # Controls annealing: high temp early (exploration), low temp later (exploitation)
                # Values empirically chosen to maintain acceptance rates 0.2-0.6
                # NOTE: May need theoretical validation per simulated annealing convergence requirements
                LANDSCAPE_DECAY = -0.05  # Decay across K landscapes (gentler than original -0.1)
                ITERATION_DECAY = -0.02  # Decay within landscape (gentler than original -0.05)
                MIN_TEMPERATURE = 0.1    # Floor to ensure continued exploration
                
                # Energy clipping constant (extracted per maintainability requirements)
                # When T=MIN_TEMPERATURE (0.1), clips at 5.0; when T=1.0, clips at 50.0
                # For clipped_delta_E=50*T: exp(-50) ≈ 1.9e-22 (effectively zero acceptance probability)
                MAX_ENERGY_DELTA_MULTIPLIER = 50.0
                
                # Safe division: max_iterations validated >0 in InferenceConfig.__post_init__
                # If max_iterations=0 possible, add validation there (not defensive max() here)
                temperature = 1.0 * math.exp(LANDSCAPE_DECAY * k) * math.exp(ITERATION_DECAY * t / config.max_iterations)
                temperature = max(temperature, MIN_TEMPERATURE)
                
                # Metropolis acceptance probability: P(accept) = min(1, exp(-delta_E / T))
                # For delta_E <= 0: always accept (energy decrease or floating-point zero)
                # Prevents accept_prob > 1.0 for negative delta_E
                if delta_E <= 0:
                    accept_prob = 1.0
                else:
                    # Numerical stability: clip large energy differences to prevent exp() overflow
                    # Asymmetric behavior at low T: clips at 5.0 when T=0.1, at 50.0 when T=1.0
                    clipped_delta_E = min(delta_E, MAX_ENERGY_DELTA_MULTIPLIER * temperature)
                    accept_prob = math.exp(-clipped_delta_E / temperature)
                
                # Probabilistic acceptance decision
                # Use Python random.random() for CPU-based generation (avoids GPU-CPU sync overhead)
                # For reproducibility: call random.seed(seed) before inference
                # NOTE: This is scientific computing (Metropolis sampling), not cryptographic context
                import random
                random_sample = random.random()
                accepted = random_sample < accept_prob
                
                # Debug logging removed from hot loop for performance and security
                # Summary statistics logged after loop completion
                
                if accepted:
                    # Update with gradient tracking preserved (detach to avoid graph accumulation)
                    out = out_new.detach().requires_grad_(True)
                    
                    # OPTIMIZATION: Cache energy for next iteration - KEY OPTIMIZATION
                    # Since out = out_new, the energy of 'out' in next iteration is energy_after_val
                    have_cached_energy = True
                    cached_energy_val = energy_after_val  # Cache the scalar value
                    info['accepted_steps'] += 1
                    
                    # Early stopping using config threshold
                    if config.should_early_stop(energy_after_val):
                        logger.debug(f"Early stopping at landscape {k}, step {t}, energy={energy_after_val:.6f}")
                        break
                else:
                    # OPTIMIZATION: If step rejected, 'out' stays the same
                    # The current energy (energy_before_val) is still valid for next iteration
                    have_cached_energy = True
                    cached_energy_val = energy_before_val
                
                info['total_steps'] += 1
            
            info['landscape_transitions'].append(k)
            
            # Check for convergence between landscapes - overall convergence monitoring
            # Only check after completing at least a few landscapes
            if k >= 2 and len(info['gradient_norms']) >= 20:
                # Check if recent gradients are consistently small across landscapes
                recent_grads = info['gradient_norms'][-20:]
                avg_grad_norm = sum(recent_grads) / len(recent_grads)
                max_grad_norm = max(recent_grads)
                
                # Check if recent energies are stable across landscapes 
                recent_energies = info['energy_history'][-20:]
                energy_range = max(recent_energies) - min(recent_energies)
                
                # Overall convergence criteria: small gradients and stable energy
                if avg_grad_norm < 1e-3 and max_grad_norm < 1e-2 and energy_range < 0.01:
                    logger.info(f"Overall convergence achieved after landscape {k}: "
                               f"avg_grad={avg_grad_norm:.2e}, max_grad={max_grad_norm:.2e}, "
                               f"energy_range={energy_range:.2e}")
                    info['convergence_reason'] = f'overall_convergence_k{k}'
                    break
                    
            # Check if we broke out of inner loop due to convergence
            if 'convergence_reason' in info:
                logger.info(f"Early exit from outer loop due to: {info['convergence_reason']}")
                break
            
            # Scale for next landscape (except for last)
            if k < config.K - 1:
                sigma_k_next = torch.sqrt(1 - self.alphas_cumprod[k + 1]).item()
                
                # Handle numerical edge cases in scaling
                if sigma_k > 1e-8:  # Avoid division by zero/very small numbers
                    scale_factor = sigma_k_next / sigma_k
                else:
                    scale_factor = 1.0
                    logger.warning(f"Small sigma_k={sigma_k:.8f} at landscape {k}, using scale_factor=1.0")
                
                out = out.detach() * scale_factor
                out = out.requires_grad_(True)
        
        # Final statistics
        final_k = k  # k is the last completed landscape
        final_timestep_tensor = torch.full((batch_size,), final_k, dtype=torch.long, device=inp_embedding.device)
        final_energy_tensor = self.compose_energies(inp_embedding, out, final_k, rule_weights, final_timestep_tensor)
        info['final_energy'] = final_energy_tensor.mean().item()  # Handle batched tensors safely
        info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
        
        # OPTIMIZATION: Add caching effectiveness statistics
        total_cache_operations = cache_hits + cache_misses
        info['cache_hits'] = cache_hits
        info['cache_misses'] = cache_misses
        info['cache_hit_rate'] = cache_hits / max(total_cache_operations, 1)
        
        # Add convergence status to final reporting
        convergence_status = info.get('convergence_reason', 'completed_all_landscapes')
        logger.info(f"Inference completed. Final energy: {info['final_energy']:.6f}, "
                   f"Acceptance rate: {info['acceptance_rate']:.3f}, "
                   f"Cache hit rate: {info['cache_hit_rate']:.3f}, "
                   f"Convergence: {convergence_status}")
        
        return out.detach(), info
    
    def _estimate_equation_complexity(self, equation: str) -> str:
        """
        Estimate equation complexity for stratified distance analysis.
        
        Args:
            equation: Input equation string
            
        Returns:
            Complexity category: 'linear', 'quadratic', 'cubic', 'unknown'
        """
        # Improved pattern matching using regex with word boundaries
        # NOTE: This heuristic identifies highest polynomial degree, NOT true algorithmic complexity
        # For production, consider using sympy or proper expression parser
        import re
        
        eq = equation.lower().replace(' ', '')
        
        # Check polynomial degree indicators - safer approach avoiding complex regex
        # Highest degree wins (polynomial degree, not computational complexity) 
        # First check for obvious function names to avoid false positives
        if any(func in eq for func in ['max', 'min', 'sin', 'cos', 'exp', 'log']):
            # For equations with functions, be more conservative - just check for x presence
            if 'x' in eq:
                return 'linear'  # Conservative classification
            else:
                return 'unknown'
        
        # For regular polynomial expressions, check degree
        if 'x**3' in eq or 'x^3' in eq or 'x*x*x' in eq:
            return 'cubic'
        elif 'x**2' in eq or 'x^2' in eq or 'x*x' in eq:
            return 'quadratic'
        elif 'x' in eq:
            return 'linear'
        else:
            return 'unknown'  # No variable found
    
    def solve_equation(
        self,
        input_equation: str,
        config: Optional[InferenceConfig] = None,
        rule_weights: Optional[Dict[str, float]] = None,
        distance_threshold: float = 6.0,  # Standard distance threshold for valid decoding
        collect_distance_data: bool = False  # Phase 2: Enable distance data collection for optimization
    ) -> Dict[str, Any]:
        """
        Solve an algebraic equation using IRED inference.
        
        Args:
            input_equation: Input equation string (e.g., "2*(x+3)+4=10")
            config: InferenceConfig with optimization parameters (uses self.config if None)
            rule_weights: Optional weights for rule composition  
            distance_threshold: Maximum distance for valid decoding
                              Standard value: 2.0 provides good balance between precision and recall.
            collect_distance_data: If True, collect distance data for statistical analysis (Phase 2)
            
        Returns:
            result: Dictionary containing solution and metadata
                   If collect_distance_data=True, includes 'distance_data' field with analysis info
        """
        # Input validation
        if not isinstance(input_equation, str):
            raise TypeError(f"input_equation must be string, got {type(input_equation)}")
        if not input_equation.strip():
            raise ValueError("input_equation cannot be empty")
        if len(input_equation) > 1000:  # Reasonable length limit
            raise ValueError(f"input_equation too long ({len(input_equation)} chars), max 1000")
        
        # Security validation - character whitelist and injection prevention
        # Allow only safe mathematical characters to prevent injection attacks
        import re
        if not re.match(r'^[a-zA-Z0-9_\s\+\-\*/\^\(\)\=\.]+$', input_equation):
            raise ValueError(
                "input_equation contains invalid characters. "
                "Allowed: alphanumeric, _, +, -, *, /, ^, (, ), =, ., spaces"
            )
        
        # Detect obvious injection patterns (defense in depth)
        dangerous_patterns = ['__', 'import', 'exec', 'eval', 'system', 'os.', 'subprocess']
        eq_lower = input_equation.lower()
        for pattern in dangerous_patterns:
            if pattern in eq_lower:
                raise ValueError(f"input_equation contains potentially dangerous pattern: '{pattern}'")
        
        logger.info(f"Solving equation: '{input_equation[:100]}{'...' if len(input_equation) > 100 else ''}'")
        
        
        try:
            # Encode input equation
            with torch.no_grad():
                inp_embedding = self.encoder(input_equation).unsqueeze(0).to(self.device)  # (1, 128)
            
            # Run IRED inference
            out_embedding, info = self.ired_inference(
                inp_embedding, config=config, rule_weights=rule_weights
            )
            
            # Decode output embedding
            result = {
                'input_equation': input_equation,
                'success': False,
                'output_equation': None,
                'decoding_distance': float('inf'),
                'inference_info': info
            }
            
            if self.decoder is not None:
                decoded_eq, distance = self.decoder.decode_embedding(out_embedding.squeeze(0))
                
                if decoded_eq is not None and distance <= distance_threshold:
                    result['success'] = True
                    result['output_equation'] = decoded_eq
                    result['decoding_distance'] = distance
                    logger.info(f"Solution found: '{decoded_eq}' (distance: {distance:.4f})")
                else:
                    logger.warning(f"No valid decoding found. Best distance: {distance:.4f}")
                    result['output_equation'] = decoded_eq  # May be None
                    result['decoding_distance'] = distance
                
                # Phase 2: Collect distance data for statistical analysis if requested
                if collect_distance_data:
                    # Estimate equation complexity for stratified analysis
                    equation_complexity = self._estimate_equation_complexity(input_equation)
                    
                    distance_data = {
                        'input_equation': input_equation,
                        'distance': distance,
                        'threshold_used': distance_threshold,
                        'success': result['success'],
                        'final_energy': info.get('final_energy', float('inf')),
                        'acceptance_rate': info.get('acceptance_rate', 0.0),
                        'equation_length': len(input_equation),
                        'equation_complexity': equation_complexity,
                        'config_params': {
                            'step_size': config.step_size if config else self.config.step_size,
                            'max_iterations': config.max_iterations if config else self.config.max_iterations,
                            'K': config.K if config else self.config.K
                        }
                    }
                    result['distance_data'] = distance_data
                    logger.debug(f"Distance data collected: {distance:.4f} (threshold: {distance_threshold})")
            else:
                logger.warning("No decoder provided - returning raw embedding")
                result['output_embedding'] = out_embedding.squeeze(0).detach()
                
                # Phase 2: Handle distance data collection when no decoder available
                if collect_distance_data:
                    equation_complexity = self._estimate_equation_complexity(input_equation)
                    result['distance_data'] = {
                        'input_equation': input_equation,
                        'distance': float('inf'),  # No distance available without decoder
                        'threshold_used': distance_threshold,
                        'success': False,
                        'final_energy': info.get('final_energy', float('inf')),
                        'acceptance_rate': info.get('acceptance_rate', 0.0),
                        'equation_length': len(input_equation),
                        'equation_complexity': equation_complexity,
                        'decoder_available': False,
                        'config_params': {
                            'step_size': config.step_size if config else self.config.step_size,
                            'max_iterations': config.max_iterations if config else self.config.max_iterations,
                            'K': config.K if config else self.config.K
                        }
                    }
            
            return result
            
        except (ValueError, TypeError) as e:
            # Input validation or tensor operation errors
            logger.error(f"Invalid input for equation '{input_equation}': {str(e)}")
            return {
                'input_equation': input_equation,
                'success': False,
                'error': f"Input error: {str(e)}",
                'inference_info': {}
            }
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # PyTorch/CUDA runtime errors
            logger.error(f"Runtime error solving equation '{input_equation}': {str(e)}")
            return {
                'input_equation': input_equation,
                'success': False,
                'error': f"Runtime error: {str(e)}",
                'inference_info': {}
            }
        except Exception as e:
            # Unexpected errors - log with more detail for debugging
            logger.error(f"Unexpected error solving equation '{input_equation}': {type(e).__name__}: {str(e)}", exc_info=True)
            return {
                'input_equation': input_equation,
                'success': False,
                'error': f"Unexpected error: {type(e).__name__}",
                'inference_info': {}
            }
    
    def solve_batch(
        self,
        input_equations: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Solve a batch of equations (currently processes one at a time).
        
        Args:
            input_equations: List of input equation strings
            **kwargs: Arguments passed to solve_equation
            
        Returns:
            results: List of solution dictionaries
        """
        results = []
        for eq in input_equations:
            result = self.solve_equation(eq, **kwargs)
            results.append(result)
        return results


def load_rule_models(
    rule_names: List[str],
    model_dir: str = './results',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, AlgebraDiffusionWrapper]:
    """
    Load trained rule-specific EBM models from disk.
    
    Args:
        rule_names: List of rule names to load ('distribute', 'combine', 'isolate', 'divide')
        model_dir: Directory containing saved models (default: './results')
        device: Device to load models on
        
    Returns:
        rule_models: Dictionary mapping rule names to loaded models
    """
    rule_models = {}
    
    for rule_name in rule_names:
        # Try multiple possible checkpoint paths
        possible_paths = [
            Path(model_dir) / rule_name / 'model.pt',
            Path(model_dir) / rule_name / 'checkpoint.pt',
            Path(model_dir) / rule_name / 'model-1.pt',
            Path(model_dir) / rule_name / 'model-final.pt'
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            logger.warning(f"Model not found for {rule_name}. Tried paths: {[str(p) for p in possible_paths]}")
            continue
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Debug logging to understand checkpoint structure
            logger.info(f"Loading checkpoint for {rule_name} from {model_path}")
            if isinstance(checkpoint, dict):
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
                if 'model' in checkpoint:
                    if isinstance(checkpoint['model'], dict):
                        logger.info(f"Found nested 'model' dict with {len(checkpoint['model'])} keys")
                        logger.info(f"Sample model keys: {list(checkpoint['model'].keys())[:5]}...")
                    else:
                        logger.info(f"'model' key exists but value is type: {type(checkpoint['model'])}")
            else:
                logger.info(f"Checkpoint is not a dict, type: {type(checkpoint)}")
            
            # Create model architecture 
            ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
            wrapper = AlgebraDiffusionWrapper(ebm)
            
            # Handle different checkpoint formats with more robust detection
            if isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                # Trainer1D / GaussianDiffusion1D checkpoint
                logger.info(f"Loading from Trainer1D checkpoint format for {rule_name}")
                full_state = checkpoint['model']
                logger.info(f"Model state has {len(full_state)} parameters")

                # Detect EBM key patterns and check for mixed formats
                has_orig_mod_keys = any(k.startswith('_orig_mod.model.ebm.') for k in full_state.keys())
                has_regular_keys = any(k.startswith('model.ebm.') for k in full_state.keys())
                
                # Warn about mixed formats which could indicate checkpoint corruption
                if has_orig_mod_keys and has_regular_keys:
                    logger.warning(f"Mixed key formats detected in {rule_name} checkpoint: both '_orig_mod.model.ebm.' and 'model.ebm.' prefixes found. This may indicate checkpoint corruption.")
                
                # Case 1: diffusion-style keys with _orig_mod prefix, e.g. '_orig_mod.model.ebm.fc1.weight'
                ebm_state = None
                if has_orig_mod_keys:
                    logger.info("Detected diffusion-style state dict with '_orig_mod.model.ebm.' keys; extracting EBM params")

                    # Keep only the EBM parameters and strip the leading '_orig_mod.model.' prefix
                    ebm_state = {
                        k.replace('_orig_mod.model.', '', 1): v
                        for k, v in full_state.items()
                        if k.startswith('_orig_mod.model.ebm.')
                    }

                # Case 2: diffusion-style keys without _orig_mod prefix, e.g. 'model.ebm.fc1.weight'
                elif has_regular_keys:
                    logger.info("Detected diffusion-style state dict with nested 'model.ebm.' keys; extracting EBM params")

                    # Keep only the EBM parameters and strip the leading 'model.' prefix
                    ebm_state = {
                        k.replace('model.', '', 1): v
                        for k, v in full_state.items()
                        if k.startswith('model.ebm.')
                    }

                # If we have an ebm_state from either case above, load it
                if ebm_state is not None:
                    # Optional debug: see how this lines up with the wrapper's expected keys
                    expected_keys = set(wrapper.state_dict().keys())
                    got_keys = set(ebm_state.keys())
                    missing = expected_keys - got_keys
                    extra = got_keys - expected_keys
                    logger.info(f"EBM state has {len(ebm_state)} parameters "
                                f"(missing: {len(missing)}, extra: {len(extra)})")

                    # Load state dict with strict=False to handle missing parameters
                    missing_keys = wrapper.load_state_dict(ebm_state, strict=False)
                    
                    # Initialize missing energy_scale and energy_bias parameters if they weren't in the checkpoint
                    if missing_keys.missing_keys:
                        logger.info(f"Missing parameters in {rule_name} checkpoint: {missing_keys.missing_keys}")
                        
                        # Initialize energy_scale and energy_bias with their default values if missing
                        if 'ebm.energy_scale' in missing_keys.missing_keys:
                            wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
                            logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")
                        
                        if 'ebm.energy_bias' in missing_keys.missing_keys:
                            wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
                            logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
                        
                        # Check if there are any other missing keys we don't know how to handle
                        unhandled_missing = [k for k in missing_keys.missing_keys 
                                           if k not in ['ebm.energy_scale', 'ebm.energy_bias']]
                        if unhandled_missing:
                            logger.warning(f"Unhandled missing parameters in {rule_name}: {unhandled_missing}")
                    
                    if missing_keys.unexpected_keys:
                        logger.info(f"Unexpected parameters in {rule_name} checkpoint (will be ignored): {missing_keys.unexpected_keys}")

                # Case 3: older / simpler checkpoint where 'model' is already just the wrapper state dict
                else:
                    logger.info("No 'model.ebm.' keys detected; treating checkpoint['model'] as direct wrapper state_dict")
                    missing_keys = wrapper.load_state_dict(full_state, strict=False)
                    
                    # Initialize missing energy_scale and energy_bias parameters if they weren't in the checkpoint
                    if missing_keys.missing_keys:
                        logger.info(f"Missing parameters in {rule_name} checkpoint: {missing_keys.missing_keys}")
                        
                        # Initialize energy_scale and energy_bias with their default values if missing
                        if 'ebm.energy_scale' in missing_keys.missing_keys:
                            wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
                            logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")
                        
                        if 'ebm.energy_bias' in missing_keys.missing_keys:
                            wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
                            logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
                        
                        # Check if there are any other missing keys we don't know how to handle
                        unhandled_missing = [k for k in missing_keys.missing_keys 
                                           if k not in ['ebm.energy_scale', 'ebm.energy_bias']]
                        if unhandled_missing:
                            logger.warning(f"Unhandled missing parameters in {rule_name}: {unhandled_missing}")
                    
                    if missing_keys.unexpected_keys:
                        logger.info(f"Unexpected parameters in {rule_name} checkpoint (will be ignored): {missing_keys.unexpected_keys}")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Standard PyTorch format
                logger.info(f"Loading from standard PyTorch checkpoint format for {rule_name}")
                missing_keys = wrapper.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
                # Initialize missing energy_scale and energy_bias parameters if they weren't in the checkpoint
                if missing_keys.missing_keys:
                    logger.info(f"Missing parameters in {rule_name} checkpoint: {missing_keys.missing_keys}")
                    
                    # Initialize energy_scale and energy_bias with their default values if missing
                    if 'ebm.energy_scale' in missing_keys.missing_keys:
                        wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")
                    
                    if 'ebm.energy_bias' in missing_keys.missing_keys:
                        wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
                    
                    # Check if there are any other missing keys we don't know how to handle
                    unhandled_missing = [k for k in missing_keys.missing_keys 
                                       if k not in ['ebm.energy_scale', 'ebm.energy_bias']]
                    if unhandled_missing:
                        logger.warning(f"Unhandled missing parameters in {rule_name}: {unhandled_missing}")
                
                if missing_keys.unexpected_keys:
                    logger.info(f"Unexpected parameters in {rule_name} checkpoint (will be ignored): {missing_keys.unexpected_keys}")
            elif isinstance(checkpoint, dict) and any(key.startswith('ebm.') for key in checkpoint.keys()):
                # Direct state dict format - check for EBM-specific keys
                logger.info(f"Loading from direct state dict format for {rule_name}")
                missing_keys = wrapper.load_state_dict(checkpoint, strict=False)
                
                # Initialize missing energy_scale and energy_bias parameters if they weren't in the checkpoint
                if missing_keys.missing_keys:
                    logger.info(f"Missing parameters in {rule_name} checkpoint: {missing_keys.missing_keys}")
                    
                    # Initialize energy_scale and energy_bias with their default values if missing
                    if 'ebm.energy_scale' in missing_keys.missing_keys:
                        wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")
                    
                    if 'ebm.energy_bias' in missing_keys.missing_keys:
                        wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
                    
                    # Check if there are any other missing keys we don't know how to handle
                    unhandled_missing = [k for k in missing_keys.missing_keys 
                                       if k not in ['ebm.energy_scale', 'ebm.energy_bias']]
                    if unhandled_missing:
                        logger.warning(f"Unhandled missing parameters in {rule_name}: {unhandled_missing}")
                
                if missing_keys.unexpected_keys:
                    logger.info(f"Unexpected parameters in {rule_name} checkpoint (will be ignored): {missing_keys.unexpected_keys}")
            else:
                # Last resort - try loading as direct state dict with detailed error info
                logger.warning(f"Unknown checkpoint format for {rule_name}")
                logger.warning(f"Checkpoint type: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    logger.warning(f"Available keys: {list(checkpoint.keys())}")
                    # Check if this looks like a Trainer1D checkpoint that we missed
                    if 'step' in checkpoint and 'model' in checkpoint:
                        logger.error(f"This looks like a Trainer1D checkpoint but 'model' key handling failed!")
                        logger.error(f"Type of checkpoint['model']: {type(checkpoint.get('model', 'missing'))}")
                        if 'model' in checkpoint:
                            logger.error(f"checkpoint['model'] = {checkpoint['model']}")
                logger.warning("Attempting direct load anyway...")
                missing_keys = wrapper.load_state_dict(checkpoint, strict=False)
                
                # Initialize missing energy_scale and energy_bias parameters if they weren't in the checkpoint
                if missing_keys.missing_keys:
                    logger.info(f"Missing parameters in {rule_name} checkpoint: {missing_keys.missing_keys}")
                    
                    # Initialize energy_scale and energy_bias with their default values if missing
                    if 'ebm.energy_scale' in missing_keys.missing_keys:
                        wrapper.ebm.energy_scale.data.fill_(1.0)  # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_scale to 1.0 for {rule_name}")
                    
                    if 'ebm.energy_bias' in missing_keys.missing_keys:
                        wrapper.ebm.energy_bias.data.fill_(0.0)   # Default value from AlgebraEBM.__init__
                        logger.info(f"Initialized missing ebm.energy_bias to 0.0 for {rule_name}")
                    
                    # Check if there are any other missing keys we don't know how to handle
                    unhandled_missing = [k for k in missing_keys.missing_keys 
                                       if k not in ['ebm.energy_scale', 'ebm.energy_bias']]
                    if unhandled_missing:
                        logger.warning(f"Unhandled missing parameters in {rule_name}: {unhandled_missing}")
                
                if missing_keys.unexpected_keys:
                    logger.info(f"Unexpected parameters in {rule_name} checkpoint (will be ignored): {missing_keys.unexpected_keys}")
            
            wrapper.to(device)
            wrapper.eval()
            
            rule_models[rule_name] = wrapper
            logger.info(f"Successfully loaded model for rule: {rule_name} from {model_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found for {rule_name}: {str(e)}")
            continue
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"PyTorch error loading model for {rule_name}: {str(e)}")
            continue
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid model format for {rule_name}: {str(e)}")
            logger.error(f"Available checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading model for {rule_name}: {type(e).__name__}: {str(e)}", exc_info=True)
            continue
    
    return rule_models


def create_inference_engine(
    rule_names: List[str] = ['distribute', 'combine', 'isolate', 'divide'],
    encoder_type: str = 'character',
    model_dir: str = './results',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> AlgebraInference:
    """
    Factory function to create a complete inference engine.
    
    Args:
        rule_names: List of rule names to load models for
        encoder_type: Type of encoder ('character' or 'ast')
        model_dir: Directory containing saved models
        device: Device to run on
        
    Returns:
        inference: Configured AlgebraInference instance
    """
    # Load rule models
    rule_models = load_rule_models(rule_names, model_dir, device)
    
    if not rule_models:
        raise ValueError(f"No models loaded from {model_dir}")
    
    # Create encoder
    if encoder_type == 'character':
        encoder = CharacterLevelEncoder(d_model=128)
    elif encoder_type == 'ast':
        encoder = ASTEncoder(d_model=128) 
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create decoder
    try:
        from algebra_encoder import create_decoder_with_default_candidates
        decoder = create_decoder_with_default_candidates(encoder)
    except ImportError:
        logger.warning("sklearn not available - decoder disabled")
        decoder = None
    
    # Create inference engine
    inference = AlgebraInference(rule_models, encoder, decoder, device=device)
    
    return inference


# Example usage and testing functions

def test_inference_simple():
    """Test inference on simple equations."""
    logger.info("Testing simple inference...")
    
    try:
        # Create mock models for testing (replace with actual loading)
        rule_models = {}
        for rule in ['distribute', 'combine', 'isolate', 'divide']:
            ebm = AlgebraEBM(rule_name=rule)
            wrapper = AlgebraDiffusionWrapper(ebm)
            rule_models[rule] = wrapper
        
        # Create encoder
        encoder = CharacterLevelEncoder()
        
        # Create inference engine
        inference = AlgebraInference(rule_models, encoder)
        
        # Test equation
        result = inference.solve_equation("2*x+4=8")
        
        logger.info(f"Test result: {result}")
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"Test failed due to invalid configuration: {str(e)}")
        return None
    except RuntimeError as e:
        logger.error(f"Test failed due to runtime error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Test failed unexpectedly: {type(e).__name__}: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    # Run simple test
    test_inference_simple()