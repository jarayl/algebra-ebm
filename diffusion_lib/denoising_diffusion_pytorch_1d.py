import math
import sys
import collections
from collections import deque
from multiprocessing import cpu_count
from pathlib import Path
from functools import partial
from collections import namedtuple
from tabulate import tabulate

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn
import torch.nn.functional as F

from einops import reduce
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import os.path as osp
import time

# Import ContrastiveEnergyLoss for enhanced energy supervision
try:
    from algebra_models import ContrastiveEnergyLoss
    CONTRASTIVE_LOSS_AVAILABLE = True
except ImportError:
    CONTRASTIVE_LOSS_AVAILABLE = False
    print("Warning: ContrastiveEnergyLoss not available - falling back to cross-entropy")


class LossBalanceMonitor:
    """
    Monitors the balance between MSE and energy loss components during training.
    
    Alerts when MSE loss dominates energy loss by too large a factor, indicating
    potential energy landscape formation issues.
    
    Args:
        alert_threshold: Ratio threshold for MSE:Energy dominance alerts (default: 100.0)
        alert_frequency: Steps between potential alerts (default: 500) 
        history_size: Number of balance ratios to track (default: 1000)
    """
    
    def __init__(self, alert_threshold: float = 100.0, alert_frequency: int = 500, history_size: int = 1000):
        self.alert_threshold = alert_threshold
        self.alert_frequency = alert_frequency
        self.history_size = history_size
        
        # Bounded history to prevent memory leaks (using deque for O(1) operations)
        self.balance_history = deque(maxlen=history_size)
        self.step_count = 0
        
        # Statistics tracking
        self.total_alerts_sent = 0
        self.max_imbalance_seen = 0.0
        
    def check_balance(self, loss_mse: torch.Tensor, loss_energy: torch.Tensor, 
                     current_scale: torch.Tensor) -> float:
        """
        Check loss balance and alert if MSE dominates too much.
        
        Args:
            loss_mse: MSE loss magnitude
            loss_energy: Energy loss magnitude  
            current_scale: Current adaptive scale being applied
            
        Returns:
            balance_ratio: Ratio of MSE to scaled energy contributions
        """
        self.step_count += 1
        
        # Calculate effective contributions to total loss
        mse_contribution = loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse
        scaled_energy_contribution = (current_scale * loss_energy).item() if torch.is_tensor(loss_energy) else current_scale * loss_energy
        
        # Compute balance ratio (MSE dominance factor)
        balance_ratio = mse_contribution / (scaled_energy_contribution + 1e-8)
        
        # Track history (deque automatically maintains max size)
        self.balance_history.append(balance_ratio)
            
        # Update statistics
        self.max_imbalance_seen = max(self.max_imbalance_seen, balance_ratio)
        
        # Alert if threshold exceeded and it's time to alert
        if (balance_ratio > self.alert_threshold and 
            self.step_count % self.alert_frequency == 0):
            
            self._send_imbalance_alert(balance_ratio, mse_contribution, scaled_energy_contribution)
        
        return balance_ratio
    
    def _send_imbalance_alert(self, ratio: float, mse_contrib: float, energy_contrib: float):
        """Send imbalance alert with actionable information."""
        self.total_alerts_sent += 1
        
        # Calculate recent trend
        recent_window = min(100, len(self.balance_history))
        if recent_window > 10:
            recent_avg = sum(self.balance_history[-recent_window:]) / recent_window
            trend = "increasing" if recent_avg > ratio * 0.9 else "decreasing"
        else:
            trend = "unknown"
            recent_avg = ratio
            
        print(f"⚠️  [LossBalanceAlert] Step {self.step_count}: MSE dominates by {ratio:.1f}x!")
        print(f"    MSE contribution: {mse_contrib:.3f}")
        print(f"    Energy contribution: {energy_contrib:.6f}")
        print(f"    Recent trend: {trend} (avg last {recent_window}: {recent_avg:.1f}x)")
        print(f"    Recommendation: Energy gradients are too weak for landscape formation")
        
    def get_statistics(self) -> dict:
        """Get monitoring statistics for analysis."""
        if not self.balance_history:
            return {"status": "no_data"}
            
        recent_window = min(100, len(self.balance_history))
        recent_ratios = self.balance_history[-recent_window:]
        
        return {
            "status": "active",
            "steps_monitored": self.step_count,
            "total_alerts": self.total_alerts_sent, 
            "max_imbalance": self.max_imbalance_seen,
            "current_ratio": self.balance_history[-1],
            "recent_avg_ratio": sum(recent_ratios) / len(recent_ratios),
            "alert_threshold": self.alert_threshold,
            "samples_tracked": len(self.balance_history)
        }


def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        # Debug hooks guarded for production deployment
        # If needed for development, use: ENABLE_IPDB_HOOKS=1 python script.py
        try:
            import ipdb
            import os
            if os.getenv('ENABLE_IPDB_HOOKS') == '1':
                # we are NOT in interactive mode, print the exception...
                traceback.print_exception(type, value, tb)
                # ...then start the debugger in post-mortem mode.
                ipdb.post_mortem(tb)
            else:
                # Debug hooks disabled, fall back to default behavior
                sys.__excepthook__(type, value, tb)
        except ImportError:
            # ipdb not available, continue without debugging hooks
            traceback.print_exception(type, value, tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    import os
    try:
        if os.getenv('ENABLE_IPDB_HOOKS') == '1':
            import ipdb
            if not hasattr(_custom_exception_hook, 'origin_hook'):
                _custom_exception_hook.origin_hook = sys.excepthook
                sys.excepthook = _custom_exception_hook
    except ImportError:
        pass  # ipdb not available, continue without hooks


def unhook_exception_ipdb():
    """Remove the hook to ipdb when an exception is raised."""
    if hasattr(_custom_exception_hook, 'origin_hook'):
        sys.excepthook = _custom_exception_hook.origin_hook

# Only hook exceptions if explicitly enabled
hook_exception_ipdb()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float = 0
    avg: float = 0
    sum: float = 0
    sum2: float = 0
    std: float = 0
    count: float = 0
    tot_count: float = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val * val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count
        self.std = (self.sum2 / self.count - self.avg * self.avg) ** 0.5

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        use_contrastive_energy_loss = False,
        enable_loss_balance_monitoring = True,
        step_size_multiplier = 0.1,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
        enable_semantic_corruption = False,
        corruption_strategy_probs = None,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.self_condition = False
        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = use_innerloop_opt
        self.use_contrastive_energy_loss = use_contrastive_energy_loss
        self.enable_loss_balance_monitoring = enable_loss_balance_monitoring
        self.step_size_multiplier = step_size_multiplier
        
        # Initialize ContrastiveEnergyLoss if requested and available
        self.contrastive_loss_fn = None
        if self.use_contrastive_energy_loss:
            if CONTRASTIVE_LOSS_AVAILABLE:
                self.contrastive_loss_fn = ContrastiveEnergyLoss(
                    margin=10.0,      # Target energy gap (same as research report)
                    pos_target=1.0,   # Correct solutions should have low energy
                    neg_target=15.0   # Incorrect solutions should have high energy
                )
                print("[ContrastiveLoss] Initialized with margin=10.0, pos_target=1.0, neg_target=15.0")
            else:
                print("Warning: ContrastiveEnergyLoss requested but not available, falling back to cross-entropy")
                self.use_contrastive_energy_loss = False
        
        # Initialize LossBalanceMonitor for detecting training issues
        self.loss_balance_monitor = None
        if self.enable_loss_balance_monitoring:
            self.loss_balance_monitor = LossBalanceMonitor(
                alert_threshold=50.0,    # Alert when MSE dominates by 50x+ (indicates weak energy gradients)
                alert_frequency=500,     # Check every 500 steps (balance thoroughness vs spam)  
                history_size=1000        # Track last 1000 measurements for trend analysis
            )
            print("[LossBalanceMonitor] Initialized with threshold=50.0x dominance detection")

        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path
        self.enable_semantic_corruption = enable_semantic_corruption
        # Validate and cache corruption strategy probabilities
        if corruption_strategy_probs is not None:
            strategy_names = ['heavy_gaussian', 'extreme_gaussian', 'pure_random']
            if self.enable_semantic_corruption:
                strategy_names.append('semantic')
            
            # Validate keys
            invalid_keys = set(corruption_strategy_probs.keys()) - set(strategy_names)
            if invalid_keys:
                raise ValueError(f"Invalid strategy keys: {invalid_keys}. Valid: {strategy_names}. Check enable_semantic_corruption setting.")
            
            # Validate values
            for name, prob in corruption_strategy_probs.items():
                if not isinstance(prob, (int, float)) or prob < 0:
                    raise ValueError(f"Invalid probability for {name}: {prob}. Must be non-negative number.")
            
            # Validate sum > 0
            probs = [corruption_strategy_probs.get(name, 0.0) for name in strategy_names]
            total_prob = sum(probs)
            if total_prob <= 0:
                raise ValueError(f"Probabilities must sum to positive value, got {total_prob}")
            
            # Cache as torch tensor for performance
            normalized = [p / total_prob for p in probs]
            self._cached_probs_tensor = torch.tensor(normalized, dtype=torch.float32)
            self._strategy_names = strategy_names
        else:
            self._cached_probs_tensor = None
            self._strategy_names = ['heavy_gaussian', 'extreme_gaussian', 'pure_random']
            if self.enable_semantic_corruption:
                self._strategy_names.append('semantic')

        self.corruption_strategy_probs = corruption_strategy_probs or {}
        
        # Initialize corruption strategy counters for logging
        self.corruption_strategy_counts = {}
        self.total_corruption_samples = 0

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        base_step_sizes = betas * torch.sqrt(1 / (1 - alphas_cumprod))
        register_buffer('opt_step_size', base_step_sizes * self.step_size_multiplier)
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))
        
        # Log step size statistics for monitoring optimization stability
        step_size_min = self.opt_step_size.min().item()
        step_size_mean = self.opt_step_size.mean().item()
        step_size_max = self.opt_step_size.max().item()
        print(f"[StepSizeInit] multiplier={self.step_size_multiplier:.3f}, "
              f"range=[{step_size_min:.6f}, {step_size_mean:.6f}, {step_size_max:.6f}] (min/mean/max)")

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)
        # whether to autonormalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, inp, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        with torch.enable_grad():
            model_output = self.model(inp, x, t)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(cond, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, cond, x, t, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False):
        b, *_, device = *x.shape, x.device

        if type(t) == int:
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        else:
            batched_times = t
            noise = torch.randn_like(x)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start

        if with_noise:
            pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t, return_both=True)
                img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf  # / (i + 1) ** 0.5

                if mask is not None:
                    img_new = img_new * (1 - mask) + mask * data_cond

                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
                img_new = torch.clamp(img_new, -max_val, max_val)

                energy_new = self.model(inp, img_new, t, return_energy=True)
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

            cond_val = None
            if mask is not None:
                cond_val = self.q_sample(x_start = inp, t = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask

            img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)

            if mask is not None:
                img = img * (1 - mask) + cond_val * mask

            # if t < 50:

            if self.sudoku:
                step = 20
            else:
                step = 5

            if self.use_innerloop_opt:
                if t < 1:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                img = img.detach()

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf

            img = torch.clamp(img, -max_val, max_val)

            # Correctly scale output
            img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
            preds.append(img_unscaled)

            batched_times_prev = batched_times - 1

            if t != 0:
                img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj=False):
        # seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def permute_equations(self, x_start):
        """
        Apply semantic corruption to algebraic equation embeddings.
        
        Creates "hard negative" samples by applying algebraic-specific corruptions
        that break mathematical relationships while preserving tensor shapes.
        
        Note: Only works correctly when embedding_dim is divisible by 4.
        
        Args:
            x_start: Input equation embeddings with shape [batch, seq_length]
            
        Returns:
            Corrupted equation embeddings with same shape as input
        """
        batch_size, embedding_dim = x_start.shape
        device = x_start.device
        
        # Check divisibility by 4 - required for quarter-based operations
        if embedding_dim % 4 != 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Skipping semantic corruption: embedding_dim={embedding_dim} not divisible by 4.")
            return x_start.clone()
        
        corrupted = x_start.clone()
        
        # Strategy 1: Operand position shuffling (40% probability)
        # Shuffle quarters of embedding to simulate swapping operands/operators  
        if torch.rand(1).item() < 0.4:
            # Vectorized shuffling for efficiency (no padding needed since divisible by 4)
            quarter_size = embedding_dim // 4
            quarters = corrupted.view(batch_size, 4, quarter_size)
            
            # Create random permutation for each batch element
            perm_indices = torch.stack([torch.randperm(4, device=device) for _ in range(batch_size)])
            shuffled_quarters = quarters[torch.arange(batch_size)[:, None], perm_indices]
            corrupted = shuffled_quarters.view(batch_size, -1)
        
        # Strategy 2: Coefficient corruption (30% probability)  
        # Randomly negate parts to simulate incorrect signs/coefficients
        if torch.rand(1).item() < 0.3:
            # Select random 25% of dimensions to negate
            flip_mask = torch.rand_like(corrupted) < 0.25
            corrupted = torch.where(flip_mask, -corrupted, corrupted)
        
        # Strategy 3: Structural noise injection (30% probability)
        # Add controlled noise to break algebraic relationships
        if torch.rand(1).item() < 0.3:
            # Add structured noise with magnitude scaled to input
            # Use torch.clamp with tensor min for torch.compile compatibility (fixes zuf0 undefined error)
            noise_scale = torch.clamp(0.5 * corrupted.std(), min=torch.tensor(0.1, device=corrupted.device, dtype=corrupted.dtype))
            structural_noise = noise_scale * torch.randn_like(corrupted)
            corrupted = corrupted + structural_noise
            
        return corrupted

    def p_losses(self, inp, x_start, mask, t, noise = None):
        b, *c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        if mask is not None:
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t = t, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond

        # predict and take gradient step

        model_out = self.model(inp, x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None:
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target

        loss = F.mse_loss(model_out, target, reduction = 'none')

        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_mse = loss

        if self.supervise_energy_landscape:
            noise = torch.randn_like(x_start)
            data_sample = self.q_sample(x_start = x_start, t = t, noise = noise)

            if mask is not None:
                data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond

            # Add a noise contrastive estimation term with samples drawn from the data distribution
            #noise = torch.randn_like(x_start)

            # Multi-strategy negative sampling for enhanced energy contrast
            # Use pre-validated and cached strategy configuration
            if self._cached_probs_tensor is not None:
                strategy_idx = torch.multinomial(self._cached_probs_tensor.to(x_start.device), 1).item()
            else:
                strategy_idx = torch.randint(0, len(self._strategy_names), (1,)).item()
            
            # Apply selected corruption strategy (robust against ordering changes)
            strategy_name = self._strategy_names[strategy_idx]
            if strategy_name == 'heavy_gaussian':
                xmin_noise = self.q_sample(x_start=x_start, t=t, noise=noise * 3.0)
            elif strategy_name == 'extreme_gaussian':
                xmin_noise = self.q_sample(x_start=x_start, t=t, noise=noise * 5.0)
            elif strategy_name == 'pure_random':
                xmin_noise = torch.randn_like(x_start)
            elif strategy_name == 'semantic' and self.enable_semantic_corruption:
                xmin_noise = self.permute_equations(x_start)
            else:
                # Fallback to original strategy for safety
                xmin_noise = self.q_sample(x_start=x_start, t=t, noise=noise * 3.0)
            
            # Efficient logging with minimal overhead
            selected_strategy = strategy_name 
            self.corruption_strategy_counts[selected_strategy] = self.corruption_strategy_counts.get(selected_strategy, 0) + 1
            self.total_corruption_samples += 1
            
            # Periodic logging (reduced frequency for performance)
            if self.total_corruption_samples % 1000 == 0:
                usage_percentages = {s: 100.0 * c / self.total_corruption_samples 
                                   for s, c in self.corruption_strategy_counts.items()}
                print(f"[CorruptionMonitor] Strategy usage after {self.total_corruption_samples} samples: {usage_percentages}")

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            else:
                data_cond = None

            if self.sudoku:
                s = x_start.size()
                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im.view(-1, 729)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.connectivity:
                s = x_start.size()
                x_start_im = x_start.view(-1, 12, 12)
                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.shortest_path:
                x_start_list = x_start.argmax(dim=2)
                classes = x_start.size(2)
                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

                loss_opt = torch.ones(1)

                # loss_scale will be computed adaptively below
            else:

                xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
                xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

                xmin_noise = xmin_noise.detach()
                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)

                # loss_opt = torch.ones(1)


                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()

                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise

                # nrep = 1


                # loss_scale will be computed adaptively below

            xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0)
            x_concat = torch.cat([data_sample, xmin_noise], dim=0)
            t_concat = torch.cat([t, t], dim=0)
            
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)
            energy_real, energy_fake_opt = torch.chunk(energy, 2, 0)
            energy_stack = torch.cat([energy_real, energy_fake_opt], dim=-1)

            # Choose energy loss computation method based on configuration
            energy_metrics = None  # Initialize for scope clarity
            if self.use_contrastive_energy_loss and self.contrastive_loss_fn is not None:
                # Use ContrastiveEnergyLoss for enhanced energy supervision with explicit targets
                loss_energy, energy_metrics = self.contrastive_loss_fn.compute_loss(
                    pos_energies=energy_real,  # Valid transformations should have low energy
                    neg_energies=energy_fake_opt,  # Invalid transformations should have high energy  
                    return_metrics=True
                )
                
                # Validate and reshape ContrastiveEnergyLoss output
                if loss_energy.dim() == 0:  # Scalar
                    loss_energy = loss_energy.unsqueeze(0).expand(energy_real.size(0), 1)
                elif loss_energy.shape == (energy_real.size(0), 1):  # Already correct shape
                    pass  # No reshaping needed
                elif loss_energy.dim() == 1 and loss_energy.size(0) == energy_real.size(0):  # (B,) shape
                    loss_energy = loss_energy.unsqueeze(1)
                else:
                    raise ValueError(
                        f"ContrastiveEnergyLoss returned unexpected shape {loss_energy.shape}. "
                        f"Expected scalar, ({energy_real.size(0)}, 1), or ({energy_real.size(0)},). "
                        f"Check your ContrastiveEnergyLoss implementation.")
            else:
                # Use cross-entropy energy loss (original approach)
                target = torch.zeros(energy_real.size(0), device=energy_real.device).long()
                loss_energy = F.cross_entropy(-1 * energy_stack, target, reduction='none')[:, None]

            # Energy gap monitoring
            energy_gap = energy_fake_opt.mean() - energy_real.mean()

            if not hasattr(self, 'energy_gap_history'):
                self.energy_gap_history = deque(maxlen=1000)

            self.energy_gap_history.append(energy_gap.item())

            if len(self.energy_gap_history) % 100 == 0:
                # Get last 100 entries (deque doesn't support slicing)
                recent_count = min(100, len(self.energy_gap_history))
                recent_gaps = list(self.energy_gap_history)[-recent_count:]
                avg_gap = sum(recent_gaps) / recent_count
                gap_msg = f"[EnergyMonitor] Average energy gap (last 100 steps): {avg_gap:.3f}"
                
                # Add ContrastiveEnergyLoss metrics if available  
                if self.use_contrastive_energy_loss and energy_metrics is not None:
                    gap_msg += (f", PosE={energy_metrics['pos_energy_mean']:.2f}, "
                               f"NegE={energy_metrics['neg_energy_mean']:.2f}, "
                               f"Margin={energy_metrics['margin_loss']:.4f}")
                
                print(gap_msg)

            # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()

            # CRITICAL FIX: Mathematically correct adaptive loss weighting for true 50:50 balance
            # Replaces incorrect scaling that allowed up to 500x imbalance
            
            # Initialize EMA state for reproducibility across runs
            if not hasattr(self, '_ema_mse_mag'):
                self._ema_mse_mag = None
                self._ema_energy_mag = None
                self._adaptive_loss_step_counter = 0

            mse_magnitude = loss_mse.mean().detach()
            energy_magnitude = torch.clamp(loss_energy.mean().detach(), min=1e-6)
            
            # EMA smoothing for reproducibility (prevents step-to-step noise)
            ema_decay = 0.99
            if self._ema_mse_mag is None:
                self._ema_mse_mag = mse_magnitude
                self._ema_energy_mag = energy_magnitude
            else:
                self._ema_mse_mag = ema_decay * self._ema_mse_mag + (1 - ema_decay) * mse_magnitude
                self._ema_energy_mag = ema_decay * self._ema_energy_mag + (1 - ema_decay) * energy_magnitude

            # Mathematically correct 50:50 weighting: energy_scale = mse / energy  
            # For equal contributions: mse_contrib = energy_contrib -> mse * 1.0 = energy * scale -> scale = mse / energy
            energy_loss_scale_factor = self._ema_mse_mag / (self._ema_energy_mag + 1e-8)
            
            # Conservative bounds to prevent training instability while maintaining energy gradients
            energy_loss_scale_factor = torch.clamp(energy_loss_scale_factor, min=0.1, max=10.0)
            
            self._adaptive_loss_step_counter += 1
            
            # Log adaptive scaling progress every 1000 steps for monitoring
            if self._adaptive_loss_step_counter % 1000 == 0:
                print(f"[AdaptiveScale] Step {self._adaptive_loss_step_counter}: "
                      f"EMA_MSE={self._ema_mse_mag:.3f}, EMA_Energy={self._ema_energy_mag:.6f}, "
                      f"EnergyWeight={energy_loss_scale_factor:.3f} (target: ~0.5 for balance)")
            
            # Monitor loss balance for potential training issues
            if self.loss_balance_monitor is not None:
                balance_ratio = self.loss_balance_monitor.check_balance(
                    loss_mse=mse_magnitude,
                    loss_energy=energy_magnitude, 
                    current_scale=energy_loss_scale_factor
                )

            loss = loss_mse + energy_loss_scale_factor * loss_energy # + 0.001 * loss_opt
            return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean())
        else:
            loss = loss_mse
            return loss.mean(), (loss_mse.mean(), -1, -1)

    def forward(self, inp, target, mask, *args, **kwargs):
        b, *c = target.shape
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            self.out_dim = c[-1]
            self.out_shape = c

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(inp, target, mask, t, *args, **kwargs)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        validation_batch_size = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        data_workers = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        pin_memory = False,
        persistent_workers = False,
        split_batches = True,
        metric = 'mse',
        cond_mask = False,
        validation_dataset = None,
        extra_validation_datasets = None,
        extra_validation_every_mul = 10,
        evaluate_first = False,
        latent = False,
        autoencode_model = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        # Conditioning on mask

        self.cond_mask = cond_mask

        # Whether to do reasoning in the latent space

        self.latent = latent

        if autoencode_model is not None:
            self.autoencode_model = autoencode_model.cuda()

        # sampling and training hyperparameters
        self.out_dim = self.model.out_dim

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.extra_validation_every_mul = extra_validation_every_mul

        self.batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size if validation_batch_size is not None else train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # Evaluation metric.
        self.metric = metric
        self.data_workers = data_workers

        if self.data_workers is None:
            self.data_workers = min(cpu_count(), 16)

        # dataset and dataloader
        
        # Store optimization parameters
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Calculate optimal dataloader kwargs
        dataloader_kwargs = {
            'batch_size': train_batch_size,
            'shuffle': True, 
            'pin_memory': self.pin_memory,
            'num_workers': self.data_workers
        }
        
        # Add persistent workers if supported and beneficial
        if self.persistent_workers and self.data_workers and self.data_workers > 0:
            dataloader_kwargs['persistent_workers'] = True

        dl = DataLoader(dataset, **dataloader_kwargs)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.validation_dataset = validation_dataset

        if self.validation_dataset is not None:
            val_kwargs = {
                'batch_size': validation_batch_size,
                'shuffle': False,
                'pin_memory': self.pin_memory,
                'num_workers': self.data_workers
            }
            if self.persistent_workers and self.data_workers and self.data_workers > 0:
                val_kwargs['persistent_workers'] = True
                
            dl = DataLoader(self.validation_dataset, **val_kwargs)
            dl = self.accelerator.prepare(dl)
            self.validation_dl = dl
        else:
            self.validation_dl = None

        self.extra_validation_datasets = extra_validation_datasets

        if self.extra_validation_datasets is not None:
            self.extra_validation_dls = dict()
            for key, dataset in self.extra_validation_datasets.items():
                extra_kwargs = {
                    'batch_size': validation_batch_size,
                    'shuffle': False,
                    'pin_memory': self.pin_memory,
                    'num_workers': self.data_workers
                }
                if self.persistent_workers and self.data_workers and self.data_workers > 0:
                    extra_kwargs['persistent_workers'] = True
                    
                dl = DataLoader(dataset, **extra_kwargs)
                dl = self.accelerator.prepare(dl)
                self.extra_validation_dls[key] = dl
        else:
            self.extra_validation_dls = None

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.evaluate_first = evaluate_first

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if osp.isfile(milestone):
            milestone_file = milestone
        else:
            milestone_file = str(self.results_folder / f'model-{milestone}.pt')
        data = torch.load(milestone_file)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.evaluate_first:
            milestone = self.step // self.save_and_sample_every
            self.evaluate(device, milestone)
            self.evaluate_first = False  # hack: later we will use this flag as a bypass signal to determine whether we want to run extra validation.

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                end_tiem = time.time()
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.cond_mask:
                        inp, label, mask = data
                        inp, label, mask = inp.float().to(device), label.float().to(device), mask.float().to(device)
                    elif self.latent:
                        inp, label, label_gt, mask_latent = data
                        mask_latent = mask_latent.float().to(device)
                        inp, label, label_gt = inp.float().to(device), label.float().to(device), label_gt.float().to(device)
                        mask = None
                    else:
                        inp, label = data
                        inp, label = inp.float().to(device), label.float().to(device)
                        mask = None

                    data_time = time.time() - end_time; end_time = time.time()

                    with self.accelerator.autocast():
                        loss, (loss_denoise, loss_energy, loss_opt) = self.model(inp, label, mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy:.4f} loss_opt: {loss_opt:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # if True:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)

                        if self.latent:
                            self.evaluate(device, milestone, inp=inp, label=label_gt, mask=mask_latent)
                        else:
                            self.evaluate(device, milestone, inp=inp, label=label, mask=mask)


                pbar.update(1)

        accelerator.print('training complete')
        
        # Always save final model at end of training
        if self.accelerator.is_local_main_process:
            import os
            
            # Calculate what the final milestone would be  
            final_milestone = max(1, self.step // self.save_and_sample_every)
            
            # Always save the final model state
            self.save(final_milestone)
            
            # Create model.pt symlink for compatibility
            final_model_path = self.results_folder / f'model-{final_milestone}.pt'
            model_pt_path = self.results_folder / 'model.pt'
            
            if final_model_path.exists():
                # Remove existing symlink if it exists
                if model_pt_path.exists() or model_pt_path.is_symlink():
                    model_pt_path.unlink()
                # Create new symlink
                try:
                    os.symlink(f'model-{final_milestone}.pt', str(model_pt_path))
                    print(f"Created model.pt -> model-{final_milestone}.pt")
                except OSError as e:
                    print(f"Warning: Could not create model.pt symlink: {e}")
            else:
                print(f"Warning: Final model file not found at {final_model_path}")

    def evaluate(self, device, milestone, inp=None, label=None, mask=None):
        print('Running Evaluation...')
        self.ema.ema_model.eval()

        if inp is not None and label is not None:
            with torch.no_grad():
                # batches = num_to_groups(self.num_samples, self.batch_size)

                if self.latent:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0)), range(1)))
                else:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                    # all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0), return_traj=True), range(1)))
                # all_samples_list = list(map(lambda n: self.model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                # all_samples_list = [self.model.sample(inp, batch_size=inp.size(0))]

                all_samples = torch.cat(all_samples_list, dim = 0)

                print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (Train)')
                if self.metric == 'mse':
                    all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (all_samples - label).pow(2).mean()
                    rows = [('mse_error', mse_error)]
                    print(tabulate(rows))
                elif self.metric == 'bce':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sudoku':
                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sort':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(sort_accuracy(all_samples_list[0], label, mask))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sort-2':
                    assert len(all_samples_list) == 1
                    summary = sort_accuracy_2(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'shortest-path-1d':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(shortest_path_1d_accuracy(all_samples_list[0], label, mask, inp))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sudoku_latent':
                    sample = all_samples_list[0].view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)

                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(prediction, label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                else:
                    raise NotImplementedError()

        if self.validation_dl is not None:
            self._run_validation(self.validation_dl, device, milestone, prefix = 'Validation')

        if (self.step % (self.save_and_sample_every * self.extra_validation_every_mul) == 0 and self.extra_validation_dls is not None) or self.evaluate_first:
            for key, extra_dl in self.extra_validation_dls.items():
                self._run_validation(extra_dl, device, milestone, prefix = key)

    def _run_validation(self, dl, device, milestone, prefix='Validation'):
        meters = collections.defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'running on the validation dataset (ID: {prefix})')):
                if self.cond_mask:
                    inp, label, mask = map(lambda x: x.float().to(device), data)
                elif self.latent:
                    inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
                else:
                    inp, label = map(lambda x: x.float().to(device), data)
                    mask = None

                if self.latent:
                    # Masking doesn't make sense in the latent space
                    # samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                else:
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))

                # np.savez("sudoku.npz", inp=inp.detach().cpu().numpy(), label=label.detach().cpu().numpy(), mask=mask.detach().cpu().numpy(), samples=samples.detach().cpu().numpy())
                # import pdb
                # pdb.set_trace()
                # print("here")
                if self.metric == 'sudoku':
                    # samples_traj = samples
                    summary = sudoku_accuracy(samples[-1], label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sudoku_latent':
                    sample = samples.view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)
                    summary = sudoku_accuracy(prediction, label_gt, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sort':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(sort_accuracy(samples, label, mask))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'sort-2':
                    summary = sort_accuracy_2(samples, label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'shortest-path-1d':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    # summary.update(shortest_path_1d_accuracy_closed_loop(samples, label, mask, inp, self.ema.ema_model.sample))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'mse':
                    # all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (samples - label).pow(2).mean()
                    meters['mse'].update(mse_error, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'bce':
                    summary = binary_classification_accuracy_4(samples, label)
                    for k, v in summary.items():
                        meters[k].update(v, n=samples.shape[0])
                    if i > 20:
                        break
                else:
                    raise NotImplementedError()

            rows = [[k, v.avg] for k, v in meters.items()]
            print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
            print(tabulate(rows))


as_float = lambda x: float(x.item())


@torch.no_grad()
def binary_classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '', saturation: bool = True) -> dict[str, float]:
    r"""Compute the accuracy of binary classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.
        saturation: whether to check the saturation of the prediction. Saturation
            is defined as :math:`1 - \min(pred, 1 - pred)`

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}


@torch.no_grad()
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    # table = list()
    # table.append(('pred', pred[0].squeeze()))
    # table.append(('label', label[0].squeeze()))
    # print(tabulate(table))

    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.0) & (pred < 0.0)).sum() / numel
    gt_0_pred_1 = ((label < 0.0) & (pred >= 0.0)).sum() / numel
    gt_1_pred_0 = ((label > 0.0) & (pred < 0.0)).sum() / numel
    gt_1_pred_1 = ((label > 0.0) & (pred >= 0.0)).sum() / numel

    accuracy = gt_0_pred_0 + gt_1_pred_1
    balanced_accuracy = sum([
        gt_0_pred_0 / ((label < 0.0).float().sum() / numel),
        gt_1_pred_1 / ((label >= 0.0).float().sum() / numel),
    ]) / 2

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
        prefix + '/accuracy': as_float(accuracy),
        prefix + '/balance_accuracy': as_float(balanced_accuracy),
    }


@torch.no_grad()
def sudoku_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    pred = pred.view(-1, 9, 9, 9).argmax(dim=-1)
    label = label.view(-1, 9, 9, 9).argmax(dim=-1)

    correct = (pred == label).float()
    mask = mask.view(-1, 9, 9, 9)[:, :, :, 0]
    mask_inverse = 1 - mask

    accuracy = (correct * mask_inverse).sum() / mask_inverse.sum()

    return {
        'accuracy': as_float(accuracy),
        'consistency': as_float(sudoku_consistency(pred)),
        'board_accuracy': as_float(sudoku_score(pred))
    }


def sudoku_consistency(pred: torch.Tensor) -> bool:
    pred_onehot = F.one_hot(pred, num_classes=9)

    all_row_correct = (pred_onehot.sum(dim=1) == 1).all(dim=-1).all(dim=-1)
    all_col_correct = (pred_onehot.sum(dim=2) == 1).all(dim=-1).all(dim=-1)

    blocked = pred_onehot.view(-1, 3, 3, 3, 3, 9)
    all_block_correct = (blocked.sum(dim=(2, 4)) == 1).all(dim=-1).all(dim=-1).all(dim=-1)

    return (all_row_correct & all_col_correct & all_block_correct).float().mean()


def sudoku_score(pred: torch.Tensor) -> bool:
    valid_mask = torch.ones_like(pred)

    pred_sum_axis_1 = pred.sum(dim=1, keepdim=True)
    pred_sum_axis_2 = pred.sum(dim=2, keepdim=True)

    # Use the sum criteria from the SAT-Net paper
    axis_1_mask = (pred_sum_axis_1 == 36)
    axis_2_mask = (pred_sum_axis_2 == 36)

    valid_mask = valid_mask * axis_1_mask.float() * axis_2_mask.float()

    valid_mask = valid_mask.view(-1, 3, 3, 3, 3)
    grid_mask = pred.view(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36

    valid_mask = valid_mask * grid_mask.float()

    return valid_mask.mean()


def sort_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = (label[:, 0, ..., 2] * 0.5 + 0.5).sum(dim=-1).cpu()
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_xy = pred[:, t, ..., -1].reshape(pred.shape[0], -1).argmax(dim=-1)
        pred_x = torch.div(pred_xy, pred.shape[2], rounding_mode='floor')
        pred_y = pred_xy % pred.shape[2]
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    ground_truth = torch.arange(pred.shape[2] - 1, -1, -1, device=array.device).unsqueeze(0).repeat(pred.shape[0], 1)
    elem_close = (array - ground_truth).abs() < 0.1
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()
    return {
        'element_correct': as_float(element_correct),
        'array_correct': as_float(array_correct),
    }


def sort_accuracy_2(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = label[:, 0, :, 0].clone().cpu()  # B x N
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_x = pred[:, t, :, 1].argmax(dim=-1)  # B x N
        pred_y = pred[:, t, :, 2].argmax(dim=-1)  # B x N
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    # stupid_impl_array = label[:, 0, :, 0].clone()  # B x N
    # for b in range(pred.shape[0]):
    #     for t in range(pred.shape[1]):
    #         pred_x = pred[b, t, :, 1].argmax(dim=-1)2
    #         pred_y = pred[b, t, :, 2].argmax(dim=-1)
    #         # swap x and y
    #         u, v = stupid_impl_array[b, pred_y].clone(), stupid_impl_array[b, pred_x].clone()
    #         stupid_impl_array[b, pred_x], stupid_impl_array[b, pred_y] = u, v

    # assert (array == stupid_impl_array).all(), 'Inconsistent implementation'
    # print('Consistent implementation!!')

    elem_close = torch.abs(array - label[:, -1, :, 0].cpu()) < 1e-5
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()

    pred_first_action = pred[:, 0, :, 1:3].argmax(dim=-2).cpu()
    label_first_action = label[:, 0, :, 1:3].argmax(dim=-2).cpu()
    first_action_correct = (pred_first_action == label_first_action).all(dim=-1).float().mean()

    return {
        'element_accuracy' + name: as_float(element_correct),
        'array_accuracy' + name: as_float(array_correct),
        'first_action_accuracy' + name: as_float(first_action_correct)
    }


def shortest_path_1d_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    pred_argmax = pred[:, :, :, -1].argmax(-1)
    label_argmax = label[:, :, :, -1].argmax(-1)

    argmax_accuracy = (pred_argmax == label_argmax).float().mean()

    # vis_array = torch.stack([pred_argmax, label_argmax], dim=1)
    # table = list()
    # for i in range(len(vis_array)):
    #     table.append((vis_array[i, 0].cpu().tolist(), vis_array[i, 1].cpu().tolist()))
    # print(tabulate(table))

    pred_argmax_first = pred_argmax[:, 0]
    label_argmax_first = label_argmax[:, 0]

    first_action_accuracy = (pred_argmax_first == label_argmax_first).float().mean()

    first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
    first_action_t = pred_argmax_first
    first_action_feasibility = (inp[
        torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
        first_action_s,
        first_action_t,
        0
    ] > 0).float().cpu()

    final_t = label_argmax[:, -1]
    first_action_accuracy_2 = first_action_distance_accuracy(inp[..., 0], first_action_s, final_t, first_action_t).float().cpu()
    first_action_accuracy_2 = first_action_accuracy_2 * first_action_feasibility

    return {
        'argmax_accuracy' + name: as_float(argmax_accuracy),
        'first_action_accuracy' + name: as_float(first_action_accuracy),
        'first_action_feasibility' + name: as_float(first_action_feasibility.mean()),
        'first_action_accuracy_2' + name: as_float(first_action_accuracy_2.mean()),
    }


def get_shortest_batch(edges: torch.Tensor) -> torch.Tensor:
    """ Return the length of shortest path between nodes. """
    b = edges.shape[0]
    n = edges.shape[1]

    # n + 1 indicates unreachable.
    shortest = torch.ones((b, n, n), dtype=torch.float32, device=edges.device) * (n + 1)
    shortest[torch.where(edges == 1)] = 1
    # Make sure that shortest[x, x] = 0
    shortest -= shortest * torch.eye(n).unsqueeze(0).to(shortest.device)
    shortest = shortest

    # Floyd Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    shortest[:, i, j] = torch.min(shortest[:, i, j], shortest[:, i, k] + shortest[:, k, j])
    return shortest


def first_action_distance_accuracy(edge: torch.Tensor, s: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
    shortest = get_shortest_batch(edge.detach().cpu())
    b = edge.shape[0]
    b_arrange = torch.arange(b, dtype=torch.int64, device=edge.device)
    return shortest[b_arrange, pred, t] < shortest[b_arrange, s, t]


def shortest_path_1d_accuracy_closed_loop(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, sample_fn, name: str = '', execution_steps: int = 1):
    assert execution_steps in (1, 2), 'Only 1-step and 2-step execution is supported'
    b, t, n, _ = pred.shape
    failed = torch.zeros(b, dtype=torch.bool, device='cpu')
    succ = torch.zeros(b, dtype=torch.bool, device='cpu')

    for i in range(8 // execution_steps):
        pred_argmax = pred[:, :, :, -1].argmax(-1)
        pred_argmax_first = pred_argmax[:, 0]
        pred_argmax_second = pred_argmax[:, 1]
        target_argmax = inp[:, :, 0, 3].argmax(dim=-1)

        first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
        first_action_t = pred_argmax_first
        first_action_feasibility = (inp[
            torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
            first_action_s,
            first_action_t,
            0
        ] > 0).cpu()
        last_t = first_action_t

        failed |= ~(first_action_feasibility.to(torch.bool))
        succ |= (first_action_t == target_argmax).cpu() & ~failed

        print(f'Step {i} (F) s={first_action_s[0].item()}, t={first_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={first_action_feasibility[0].item()}')

        if execution_steps >= 2:
            second_action_s = first_action_t
            second_action_t = pred_argmax_second
            second_action_feasibility = (inp[
                torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
                second_action_s,
                second_action_t,
                0
            ] > 0).cpu()
            failed |= ~(second_action_feasibility.to(torch.bool))
            succ |= (second_action_t == target_argmax).cpu() & ~failed
            last_t = second_action_t

            print(f'Step {i} (S) s={second_action_s[0].item()}, t={second_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={second_action_feasibility[0].item()}')

        inp_clone = inp.clone()
        inp_clone[:, :, :, 1] = 0
        inp_clone[torch.arange(b, dtype=torch.int64, device=inp.device), last_t, :, 1] = 1
        inp = inp_clone
        pred = sample_fn(inp, label, mask, batch_size=inp.size(0))

    return {
        'closed_loop_success_rate' + name: as_float(succ.float().mean()),
    }

