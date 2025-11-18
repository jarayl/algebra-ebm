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
import torch.nn as nn
import math
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path
import logging

# Import existing components
from algebra_encoder import CharacterLevelEncoder, ASTEncoder, EquationDecoder
from algebra_models import AlgebraEBM, AlgebraDiffusionWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        K: Number of energy landscapes (default: 10)
        device: Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        rule_models: Dict[str, AlgebraDiffusionWrapper],
        encoder: Union[CharacterLevelEncoder, ASTEncoder],
        decoder: Optional[EquationDecoder] = None,
        K: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.rule_models = rule_models
        self.encoder = encoder
        self.decoder = decoder
        self.K = K
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
        self.alphas_cumprod = compute_alphas_cumprod(K).to(device)
        
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
        rule_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compose energy functions from multiple rules by weighted summation.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128)
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule (default: all 1.0)
            
        Returns:
            total_energy: Composed energy value (B, 1)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        total_energy = 0.0
        t = torch.full((inp.shape[0],), k, dtype=torch.long, device=self.device)
        
        for rule_name, model in self.rule_models.items():
            weight = rule_weights.get(rule_name, 1.0)
            energy = model(inp, out, t, return_energy=True)  # (B, 1)
            total_energy += weight * energy
        
        return total_energy
    
    def compute_composed_gradient(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute gradient of composed energy w.r.t. output embedding.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128) 
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule
            
        Returns:
            grad: Energy gradient dE/dout (B, 128)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        # Enable gradient computation
        out = out.requires_grad_(True)
        
        # Compute composed energy
        total_energy = self.compose_energies(inp, out, k, rule_weights)
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=total_energy.sum(),
            inputs=out,
            create_graph=True
        )[0]
        
        return grad
    
    def compute_energy_and_gradient(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        k: int,
        rule_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both composed energy and gradient in a single forward pass.
        
        This optimizes performance by avoiding redundant energy computation.
        
        Args:
            inp: Input equation embedding (B, 128)
            out: Output equation embedding (B, 128) 
            k: Landscape index [0, K-1]
            rule_weights: Optional weights for each rule
            
        Returns:
            energy: Composed energy value (B, 1)
            grad: Energy gradient dE/dout (B, 128)
        """
        if rule_weights is None:
            rule_weights = {rule: 1.0 for rule in self.rule_models.keys()}
        
        # Enable gradient computation
        out = out.requires_grad_(True)
        
        # Compute composed energy
        total_energy = self.compose_energies(inp, out, k, rule_weights)
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=total_energy.sum(),
            inputs=out,
            create_graph=True
        )[0]
        
        return total_energy, grad
    
    def ired_inference(
        self,
        inp_embedding: torch.Tensor,
        T: int = 20,
        step_size: float = 0.1,
        rule_weights: Optional[Dict[str, float]] = None,
        use_adaptive_step: bool = True,
        energy_threshold: float = 1e-6
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Core IRED inference algorithm with annealed gradient descent.
        
        Args:
            inp_embedding: Input equation embedding (B, 128)  
            T: Number of gradient steps per landscape (default: 20)
            step_size: Initial step size for gradient descent (default: 0.1)
            rule_weights: Optional weights for rule composition
            use_adaptive_step: Whether to adapt step size per landscape
            energy_threshold: Early stopping threshold for energy
            
        Returns:
            out_embedding: Final optimized embedding (B, 128)
            info: Dictionary with optimization statistics
        """
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
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if energy_threshold < 0:
            raise ValueError(f"energy_threshold must be non-negative, got {energy_threshold}")
        
        batch_size = inp_embedding.shape[0]
        
        # Initialize from noise  
        out = torch.randn(batch_size, 128, device=self.device, requires_grad=True)
        
        # Track optimization statistics
        info = {
            'energy_history': [],
            'step_sizes': [],
            'landscape_transitions': [],
            'gradient_norms': [],
            'accepted_steps': 0,
            'total_steps': 0
        }
        
        # Iterate through K landscapes
        for k in range(self.K):
            sigma_k = torch.sqrt(1 - self.alphas_cumprod[k]).item()
            
            # Adaptive step size (decrease for later landscapes)
            current_step_size = step_size * (0.5 ** (k // 3)) if use_adaptive_step else step_size
            info['step_sizes'].append(current_step_size)
            
            logger.debug(f"Landscape {k}, sigma_k={sigma_k:.4f}, step_size={current_step_size:.4f}")
            
            # T gradient descent steps in this landscape
            for t in range(T):
                # Compute composed energy and gradient in single pass (performance optimization)
                energy_before, grad = self.compute_energy_and_gradient(inp_embedding, out, k, rule_weights)
                
                info['energy_history'].append(energy_before.item())
                info['gradient_norms'].append(torch.norm(grad).item())
                
                # Gradient descent step
                out_new = out - current_step_size * grad
                
                # Energy-based acceptance criteria with numerical tolerance
                energy_after = self.compose_energies(inp_embedding, out_new, k, rule_weights)
                energy_diff = energy_after.item() - energy_before.item()
                
                # Accept if energy decreases or stays roughly the same (within tolerance)
                if energy_diff < 1e-8:  # Small tolerance for numerical precision
                    out = out_new.detach().requires_grad_(True)  # Detach to avoid graph accumulation
                    info['accepted_steps'] += 1
                    
                    # Early stopping if energy is very low
                    if energy_after.item() < energy_threshold:
                        logger.debug(f"Early stopping at landscape {k}, step {t}, energy={energy_after.item():.6f}")
                        break
                
                info['total_steps'] += 1
            
            info['landscape_transitions'].append(k)
            
            # Scale for next landscape (except for last)
            if k < self.K - 1:
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
        info['final_energy'] = self.compose_energies(inp_embedding, out, self.K-1, rule_weights).item()
        info['acceptance_rate'] = info['accepted_steps'] / max(info['total_steps'], 1)
        
        logger.info(f"Inference completed. Final energy: {info['final_energy']:.6f}, "
                   f"Acceptance rate: {info['acceptance_rate']:.3f}")
        
        return out.detach(), info
    
    def solve_equation(
        self,
        input_equation: str,
        T: int = 20,
        step_size: float = 0.1,
        rule_weights: Optional[Dict[str, float]] = None,
        max_candidates: int = 1000,
        distance_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Solve an algebraic equation using IRED inference.
        
        Args:
            input_equation: Input equation string (e.g., "2*(x+3)+4=10")
            T: Number of gradient steps per landscape
            step_size: Step size for gradient descent
            rule_weights: Optional weights for rule composition  
            max_candidates: Maximum candidates for decoder
            distance_threshold: Maximum distance for valid decoding
            
        Returns:
            result: Dictionary containing solution and metadata
        """
        # Input validation
        if not isinstance(input_equation, str):
            raise TypeError(f"input_equation must be string, got {type(input_equation)}")
        if not input_equation.strip():
            raise ValueError("input_equation cannot be empty")
        if len(input_equation) > 1000:  # Reasonable length limit
            raise ValueError(f"input_equation too long ({len(input_equation)} chars), max 1000")
        
        logger.info(f"Solving equation: '{input_equation[:100]}{'...' if len(input_equation) > 100 else ''}'")
        
        try:
            # Encode input equation
            with torch.no_grad():
                inp_embedding = self.encoder(input_equation).unsqueeze(0).to(self.device)  # (1, 128)
            
            # Run IRED inference
            out_embedding, info = self.ired_inference(
                inp_embedding, T=T, step_size=step_size, rule_weights=rule_weights
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
            else:
                logger.warning("No decoder provided - returning raw embedding")
                result['output_embedding'] = out_embedding.squeeze(0).cpu()
            
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
        model_path = Path(model_dir) / rule_name / 'model.pt'
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        try:
            # Load model state dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model architecture 
            ebm = AlgebraEBM(inp_dim=128, out_dim=128, rule_name=rule_name)
            wrapper = AlgebraDiffusionWrapper(ebm)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                wrapper.load_state_dict(checkpoint['model_state_dict'])
            else:
                wrapper.load_state_dict(checkpoint)
            
            wrapper.to(device)
            wrapper.eval()
            
            rule_models[rule_name] = wrapper
            logger.info(f"Loaded model for rule: {rule_name}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found for {rule_name}: {str(e)}")
            continue
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"PyTorch error loading model for {rule_name}: {str(e)}")
            continue
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid model format for {rule_name}: {str(e)}")
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