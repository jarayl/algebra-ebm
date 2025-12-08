# Diffusion module - lazy imports to avoid circular dependencies
# Individual modules can be imported directly: from src.diffusion.denoising_diffusion_pytorch_1d import GaussianDiffusion1D

__all__ = [
    'GaussianDiffusion1D',
    'Trainer1D',
    'LogicMachine',
    'GPT',
]
