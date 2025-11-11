# Operator-Level Compositional Energy-Based Reasoning for Symbolic Algebra

*(A 3–4 Month Proof-of-Concept Plan Written as a Paper)*

---

## Abstract

We study whether modular energy-based reasoning can support **zero-shot algebraic problem solving** by composing independently learned rule-level energy functions at inference time. Our method trains a separate energy function for each algebraic manipulation rule (e.g., distribute, combine like terms, isolate variable, divide coefficient) using only single-step equations. At test time, we **sum** these energies to construct a joint landscape and then optimize in that landscape to solve multi-step, multi-rule equations the model never saw during training.

This approach builds upon Iterative Reasoning through Energy Diffusion (IRED), which learns a sequence of annealed energy landscapes and performs inference by descending those landscapes with gradient-based optimization, achieving strong out-of-distribution generalization on Sudoku, path planning, and matrix reasoning. ([arXiv][1])

**Key distinction from IRED:** While IRED trains a single energy function per task, we propose training separate energies per algebraic rule and composing them at test time. IRED mentions this compositional possibility but does not implement or evaluate it. Our work makes this concrete for symbolic algebra.

We evaluate along three axes:

1. **Zero-shot compositional generalization:** Can we solve problems requiring 2–4 algebraic rules even though training only saw single-rule problems?
2. **Constraint control at test time:** Can we inject additional constraint energies (e.g., "solution must be positive") without retraining? 
3. **Inference quality:** How does proper IRED-style annealed optimization perform on composed algebraic landscapes?

---

## 1. Introduction

A long-standing goal in machine reasoning is **systematic generalization**: solve problems that are *compositions* of skills seen during training, even if the exact composition never appeared in the data. This is especially natural in algebra, where human solvers chain modular rules ("distribute," "combine like terms," "isolate," "divide the coefficient") in different orders to reach a canonical solved form.

Mainstream neural models struggle with this kind of compositionality unless they are heavily finetuned on long multi-step traces. Neuro-symbolic systems (e.g., Neural Logic Machines) explicitly build modular structures for logical rules and have demonstrated impressive length generalization in relational reasoning and program-like tasks. ([arXiv][4])

Energy-based approaches offer a different angle. Iterative Reasoning through Energy Diffusion (IRED) showed that we can learn energy functions — along with an annealed sequence of auxiliary "landscapes" — and then perform iterative optimization at inference to solve puzzles and planning tasks beyond the training distribution. The key is that inference is *not* a single forward pass, but learned search. ([arXiv][1])

### Problem we address

* Can we learn one energy per algebraic rule (e.g., distribute)?
* Then, only at test time, sum those rule energies to solve equations that require multiple rules in sequence, in an order and depth never seen during training?

### Core hypothesis

If each rule-energy correctly scores "this next step is algebraically valid," then the sum of several rule-energies should prefer solution states reachable by *some* legal rule sequence, without us ever training on that entire sequence.

### Contributions

1. **Algebra as a compositional energy benchmark.**
   We construct a dataset of linear equation problems where each step corresponds to a standard algebraic manipulation. We train only on *single-rule* instances and test on *multi-rule* compositions that require chaining 2–4 rules. We provide symbolic ground truth via SymPy (exact equality checking).

2. **Operator-level energy functions.**
   We define one energy function per algebraic operator. During inference we form a composed landscape by summing these energies and optimize that sum using IRED's annealed gradient descent.

3. **Proper IRED implementation for algebra.**
   We adapt IRED's training (denoising + contrastive supervision) and inference (annealed landscape traversal with proper scaling) to symbolic equation embeddings.

4. **Constraint-aware inference.**
   We show how to bolt on additional constraint energies (e.g., "x must be positive"), similar to compositional EBMs in vision. ([energy-based-model.github.io][3])

---

## 2. Background

### 2.1 Energy-Based Reasoning and IRED

IRED (Iterative Reasoning through Energy Diffusion) treats reasoning as searching for a configuration $y$ (candidate solution) that minimizes an energy $E_\theta(x, y, k)$, where $x$ is the problem instance and $k$ indexes an annealed "landscape." The method learns:

* A *sequence* of energy landscapes with increasing sharpness,
* A score model that predicts gradients of the energy via denoising supervision, and
* A contrastive term that shapes each landscape.

At inference, IRED repeatedly updates a candidate solution by following gradients of these learned energy landscapes across annealed noise levels, allowing it to solve tasks (e.g., more difficult Sudoku boards, longer paths) that exceed training difficulty. ([arXiv][1])

**IRED training has two components:**

1. **Score matching loss (denoising)**: Given a corrupted sample $\tilde{y} = \sqrt{1-\sigma_k^2} y^* + \sigma_k \epsilon$, train the energy gradient to predict the noise: $||\nabla_y E_\theta(x, \tilde{y}, k) - \epsilon||^2$

2. **Energy contrastive loss**: Force valid solutions to have lower energy than corrupted negatives: $-\log\frac{e^{-E^+}}{e^{-E^+} + e^{-E^-}}$

**IRED inference (Algorithm 2):**
1. Initialize $y$ from Gaussian noise
2. For each landscape $k=1$ to $K$:
   - Run $T$ gradient descent steps: $y \leftarrow y - \lambda \nabla_y E(x,y,k)$
   - Scale for next landscape: $y \leftarrow \frac{\sqrt{1-\sigma_k^2}}{\sqrt{1-\sigma_{k-1}^2}} y$
3. Return final $y$

**Critical note:** IRED uses a single trajectory with landscape scaling, not multi-particle optimization. The annealing happens through the sequence of progressively sharper landscapes.

### 2.2 What IRED Does Not Do

IRED trains one energy function per task (e.g., one for Sudoku, one for matrix inverse). While the paper mentions that "one may also add additional inference-time constraints (e.g., by composing the learned IRED energy function with other energy functions)," **this compositional capability is not implemented or evaluated in the original work**.

Our proposal makes this concrete: we train separate energy functions for different algebraic rules and compose them at test time.

### 2.3 Compositional Energy Minimization (From Other Work)

Recent work on compositional energy minimization proposes training separate energy functions for subproblems, then composing them *without retraining* by summing the energies at test time. ([arXiv][2]) This provides conceptual motivation for our rule-level composition, though we use IRED's training and inference framework rather than multi-particle methods.

### 2.4 Neuro-Symbolic Modularity

Neural Logic Machines learn differentiable logic modules that extrapolate to longer reasoning chains. ([arXiv][4]) Our setup is philosophically similar — modular components that combine at test time — but differs in mechanism. We define independent *energies* over candidate algebraic states and *sum* them to produce preferences over whole solutions.

---

## 3. Problem Setup

### 3.1 Algebraic States and Rules

We consider single-variable linear equations of the general form
$$a(x + b) + c = d,$$
and their equivalent rearrangements, where solving means producing a state equivalent to
$$x = \alpha$$
for some scalar $\alpha$.

We focus on a small set of algebra rules:

1. **Distribute**: $a(x + b) \rightarrow ax + ab$
2. **Combine like terms**: $ax + bx \rightarrow (a{+}b)x$
3. **Isolate (move constants)**: $x + c = d \rightarrow x = d - c$
4. **Divide coefficient**: $kx = m \rightarrow x = m/k$

Each rule maps one algebraic state to another in a single step.

### 3.2 Rule-Level Energies

For each rule $r \in \{\text{distribute, combine, isolate, divide}\}$, we train an energy function
$$E_r(x, y, k) \in \mathbb{R}$$
that scores how valid it is to have state $y$ given input state $x$ at landscape index $k$.

* Low energy = "$y$ is a correct configuration reachable from $x$ using rule $r$"
* High energy = "invalid or algebraically inconsistent"

Following IRED, $k \in \{1, \dots, K\}$ indexes progressively sharper energy landscapes. In our experiments we set $K = 10$.

### 3.3 Composed Energy for Multi-Rule Problems

At test time, multi-step algebra problems require multiple rules in sequence. Instead of training on entire sequences, we define a **composed energy**:
$$E_{\text{total}}(x, y, k) = \sum_{r \in \mathcal{R}} \lambda_r \, E_r(x, y, k)$$
where $\mathcal{R}$ is the set of all rules we want to allow and $\lambda_r \ge 0$ are scalar weights (default $\lambda_r = 1$).

**This is our key novelty:** IRED trains one energy per task. We train one energy per *rule* and compose them at inference.

### 3.4 Constraint Energies

We can inject additional constraints at inference:

* Positivity: enforce $x > 0$
* Integerness: enforce $x \in \mathbb{Z}$

We define constraint energies $E_{\text{pos}}$, $E_{\text{int}}$ and extend:
$$E_{\text{total}}^{\text{constrained}}(x,y,k) = E_{\text{total}}(x,y,k) + \beta_{\text{pos}} E_{\text{pos}}(y) + \beta_{\text{int}} E_{\text{int}}(y)$$

---

## 4. Method

### 4.1 Algebraic State Encoding

We need to embed algebraic expressions (discrete strings) into a continuous vector space. We use a simple character-level encoder for the baseline:

```python
import torch
import numpy as np
import sympy as sp

VOCAB = '0123456789x+-=*/() '
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}

def encode_equation_char(eq_str, d_model=128, max_len=64):
    """
    Character-level encoder: one-hot each char, flatten, project.
    """
    one_hot = np.zeros((max_len, len(VOCAB)), dtype=np.float32)
    for i, c in enumerate(eq_str[:max_len]):
        if c in CHAR_TO_IDX:
            one_hot[i, CHAR_TO_IDX[c]] = 1.0
    
    flat = one_hot.flatten()  # shape: (max_len * |Vocab|)
    vec = np.zeros((d_model,), dtype=np.float32)
    vec[: min(d_model, flat.shape[0])] = flat[: d_model]
    return vec  # (d_model,)

def sympy_solution(eq_str):
    """Extract ground-truth solution for x using SymPy."""
    lhs_str, rhs_str = eq_str.split("=")
    x = sp.Symbol('x')
    lhs = sp.sympify(lhs_str)
    rhs = sp.sympify(rhs_str)
    sol = sp.solve(sp.Eq(lhs, rhs), x)
    return sol
```

### 4.2 Dataset Generation

We generate supervised pairs for training:

* **Single-rule pairs:** Input state → Output state after one rule application
* **Negative samples:** Input state paired with incorrect output

```python
from torch.utils.data import Dataset

class AlgebraDataset(Dataset):
    """
    IRED-compatible dataset for algebraic reasoning.
    Modes:
      - problem_type='single_rule' for training rule-specific energies
      - problem_type='multi_rule' for held-out compositional test
    """
    def __init__(self,
                 num_problems=50000,
                 rules=('distribute', 'combine', 'isolate', 'divide'),
                 problem_type='single_rule',
                 seed=0):
        self.rules = list(rules)
        self.problem_type = problem_type
        self.rng = np.random.default_rng(seed)
        self.examples = self._generate(num_problems)
        
        # IRED expects these attributes
        self.inp_dim = 128
        self.out_dim = 128
    
    def _rand_int(self, lo, hi):
        return int(self.rng.integers(lo, hi))
    
    def _single_rule_instance(self, rule):
        """Return (x_str, y_pos_str, y_neg_str) for one rule application."""
        
        if rule == 'distribute':
            # a*(x + b) = rhs  --> ax + ab = rhs
            a = self._rand_int(2, 10)
            b = self._rand_int(1, 10)
            xval = self._rand_int(1, 15)
            rhs = a * (xval + b)
            
            x_str = f"{a}*(x+{b})={rhs}"
            y_pos = f"{a}*x+{a*b}={rhs}"
            y_neg = f"{a}*x+{a*(b+1)}={rhs}"  # incorrect
        
        elif rule == 'combine':
            # ax + bx = rhs  --> (a+b)x = rhs
            a = self._rand_int(1, 6)
            b = self._rand_int(1, 6)
            xval = self._rand_int(1, 10)
            rhs = (a + b) * xval
            
            x_str = f"{a}*x+{b}*x={rhs}"
            y_pos = f"{a+b}*x={rhs}"
            y_neg = f"{a+b+1}*x={rhs}"  # incorrect
        
        elif rule == 'isolate':
            # x + c = rhs  --> x = rhs - c
            c = self._rand_int(1, 10)
            xval = self._rand_int(1, 20)
            rhs = xval + c
            
            x_str = f"x+{c}={rhs}"
            y_pos = f"x={xval}"
            y_neg = f"x={xval+1}"  # incorrect
        
        elif rule == 'divide':
            # kx = rhs  --> x = rhs/k
            k = self._rand_int(2, 10)
            xval = self._rand_int(1, 20)
            rhs = k * xval
            
            x_str = f"{k}*x={rhs}"
            y_pos = f"x={xval}"
            y_neg = f"x={xval+1}"  # incorrect
        
        return {
            'x_str': x_str,
            'y_pos_str': y_pos,
            'y_neg_str': y_neg,
            'rule': rule
        }
    
    def _multi_rule_instance(self):
        """
        Build a multi-step equation by applying 2-4 random rules *backwards*
        from a solved state, so we know it's solvable.
        """
        xval = self._rand_int(1, 15)
        
        # Start from solved form
        curr = f"x={xval}"
        steps = []
        
        for _ in range(self._rand_int(2, 5)):
            r = self.rng.choice(self.rules)
            # Invert the rule to make problem harder
            if r == 'divide':
                k = self._rand_int(2, 10)
                curr = f"{k}*x={k*xval}"
            elif r == 'isolate':
                c = self._rand_int(1, 10)
                curr = f"x+{c}={xval+c}"
            elif r == 'combine':
                a = self._rand_int(1, 6)
                b = self._rand_int(1, 6)
                curr = f"{a}*x+{b}*x={(a+b)*xval}"
            elif r == 'distribute':
                a = self._rand_int(2, 6)
                b = self._rand_int(1, 6)
                curr = f"{a}*(x+{b})={a*(xval+b)}"
            
            steps.append((curr, r))
        
        return {
            'x_str': steps[-1][0],  # hardest state
            'y_pos_str': f"x={xval}",  # target
            'y_neg_str': None,  # no negative for test
            'rule': 'multi'
        }
    
    def _generate(self, N):
        items = []
        if self.problem_type == 'single_rule':
            n_per_rule = max(1, N // len(self.rules))
            for r in self.rules:
                for _ in range(n_per_rule):
                    items.append(self._single_rule_instance(r))
        else:
            for _ in range(N):
                items.append(self._multi_rule_instance())
        return items
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Encode to vectors
        x_vec = encode_equation_char(ex['x_str'], d_model=128)
        y_pos_vec = encode_equation_char(ex['y_pos_str'], d_model=128)
        
        x_tensor = torch.tensor(x_vec, dtype=torch.float32)
        y_pos_tensor = torch.tensor(y_pos_vec, dtype=torch.float32)
        
        if ex['y_neg_str'] is not None:
            y_neg_vec = encode_equation_char(ex['y_neg_str'], d_model=128)
            y_neg_tensor = torch.tensor(y_neg_vec, dtype=torch.float32)
            # IRED format: return (inp, out_pos, out_neg)
            return x_tensor, y_pos_tensor, y_neg_tensor
        else:
            # Test format: return (inp, out_pos)
            return x_tensor, y_pos_tensor
```

### 4.3 Energy Model Architecture

Following IRED's actual architecture for continuous tasks (Table 8 in paper), we use a simple 3-layer MLP:

```python
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding from diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class AlgebraEBM(nn.Module):
    """
    Energy-Based Model for algebraic rule validity.
    Architecture matches IRED's continuous task models.
    
    Input:
      inp: embedding of current equation (B, inp_dim)
      out: embedding of candidate next equation (B, out_dim)
      t: landscape / diffusion timestep (B,)
    Output:
      energy scalar (B, 1)
    """
    def __init__(self, inp_dim=128, out_dim=128, hidden=512):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        # Time embedding
        fourier_dim, time_dim = 128, 128
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main network: 3-layer MLP matching IRED Table 8
        self.fc1 = nn.Linear(inp_dim + out_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        
        # Output: IRED uses L2 norm of output vector as energy
        self.fc_out = nn.Linear(hidden, out_dim)
        
        # Time modulation (FiLM-style)
        self.t_map_fc2 = nn.Linear(time_dim, 2 * hidden)
        self.t_map_fc3 = nn.Linear(time_dim, 2 * hidden)
    
    def forward(self, inp, out, t):
        """
        inp, out: (B, dim)
        t: (B,) timestep
        returns: (B, 1) energy
        """
        t_emb = self.time_mlp(t)
        
        # Concatenate input and output
        h = torch.cat([inp, out], dim=-1)
        h = F.gelu(self.fc1(h))
        
        # Layer 2 with time modulation
        t_scale, t_shift = self.t_map_fc2(t_emb).chunk(2, dim=-1)
        h = F.gelu(self.fc2(h) * (1 + t_scale) + t_shift)
        
        # Layer 3 with time modulation
        t_scale, t_shift = self.t_map_fc3(t_emb).chunk(2, dim=-1)
        h = F.gelu(self.fc3(h) * (1 + t_scale) + t_shift)
        
        # Output vector
        out_vec = self.fc_out(h)  # (B, out_dim)
        
        # Energy is L2 norm squared (always non-negative)
        energy = (out_vec ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
        return energy
```

For compatibility with IRED's `GaussianDiffusion1D` training code, we wrap this to return gradients:

```python
class AlgebraDiffusionWrapper(nn.Module):
    """
    Wrapper that returns dE/dout for IRED's denoising supervision.
    This matches the interface expected by GaussianDiffusion1D.
    """
    def __init__(self, ebm_model):
        super().__init__()
        self.ebm = ebm_model
        self.inp_dim = ebm_model.inp_dim
        self.out_dim = ebm_model.out_dim
    
    def forward(self, inp, out, t):
        """Return gradient of energy wrt out."""
        out = out.requires_grad_(True)
        energy = self.ebm(inp, out, t)
        
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=out,
            create_graph=True
        )[0]
        
        return grad  # (B, out_dim)
```

### 4.4 Training Objective

Following IRED Algorithm 1, we train with two losses:

**1. Score matching (denoising) loss:**

Given ground truth pair $(x, y^*)$, we corrupt $y^*$ at noise level $\sigma_k$:
$$\tilde{y} = \sqrt{1-\sigma_k^2} y^* + \sigma_k \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Train the energy gradient to predict the noise:
$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{k,\epsilon} ||\nabla_y E_r(x, \tilde{y}, k) - \epsilon||^2$$

**2. Contrastive landscape loss:**

For landscape $k$, ensure positive examples have lower energy than negatives:
$$\mathcal{L}_{\text{contrast}} = -\log \frac{e^{-E_r(x, \tilde{y}^+, k)}}{e^{-E_r(x, \tilde{y}^+, k)} + e^{-E_r(x, \tilde{y}^-, k)}}$$

where $\tilde{y}^+$ is corrupted positive and $\tilde{y}^-$ is corrupted negative (both with same noise).

**Total loss:**
$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \alpha \mathcal{L}_{\text{contrast}}$$

We train one `AlgebraEBM` per rule $r$ on only single-rule data.

This plugs directly into IRED's `GaussianDiffusion1D` trainer from the public codebase.

### 4.5 Inference

Following IRED Algorithm 2 exactly:

```python
def ired_inference(energy_fns, inp, K=10, T=20, step_size=0.1, device='cuda'):
    """
    IRED-style inference with annealed landscape traversal.
    
    Args:
        energy_fns: dict {rule_name: AlgebraEBM model}
        inp: (1, inp_dim) input equation encoding
        K: number of landscapes
        T: optimization steps per landscape
        step_size: gradient descent learning rate
    
    Returns:
        out: (out_dim,) final predicted equation encoding
    """
    inp = inp.to(device)
    dim = inp.shape[-1]
    
    # Noise schedule (cosine, matching IRED)
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    betas = cosine_beta_schedule(K).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Initialize from Gaussian noise
    out = torch.randn(1, dim, device=device)
    
    # Traverse landscapes k=1 to K
    for k in range(K):
        sigma_k = torch.sqrt(1 - alphas_cumprod[k])
        t_tensor = torch.tensor([float(k)], device=device)
        
        # T optimization steps on landscape k
        for _ in range(T):
            out.requires_grad_(True)
            
            # Composed energy: sum all rule energies
            E_total = 0
            for rule_name, ebm in energy_fns.items():
                E_total = E_total + ebm(inp, out, t_tensor)
            
            # Gradient descent
            grad = torch.autograd.grad(
                outputs=E_total,
                inputs=out,
                create_graph=False
            )[0]
            
            with torch.no_grad():
                out_new = out - step_size * grad
                
                # Only accept if energy decreases (IRED Algorithm 2)
                E_old = sum(ebm(inp, out, t_tensor) for ebm in energy_fns.values())
                E_new = sum(ebm(inp, out_new, t_tensor) for ebm in energy_fns.values())
                
                if E_new < E_old:
                    out = out_new
        
        # Scale for next landscape (critical IRED step!)
        if k < K - 1:
            with torch.no_grad():
                sigma_k_next = torch.sqrt(1 - alphas_cumprod[k + 1])
                scale = sigma_k_next / sigma_k
                out = out * scale
    
    return out.detach().cpu().numpy()[0]  # (dim,)
```

**Key differences from original proposal:**
1. Single trajectory, not multi-particle
2. Proper landscape scaling between k steps
3. Energy decrease check before accepting updates
4. Cosine noise schedule matching IRED

To decode back to equation string, we use nearest-neighbor in embedding space to a candidate set of syntactically valid equations, then verify with SymPy.

---

## 5. Evaluation Protocol

### 5.1 Train / Test Splits for Compositional Generalization

* **Training:** Only *single-rule* problems. Each rule-specific EBM sees inputs solvable by exactly one application of that rule.
* **Testing:** Multi-step equations requiring 2–4 rules in sequence. These compositions never appear in training.

### 5.2 Metrics

1. **Symbolic equivalence (primary):** Use SymPy to check if predicted equation solves to same $x$ value as ground truth.

2. **Embedding L2 distance (auxiliary):** Measure $||y_{\text{pred}} - y_{\text{gold}}||_2$ in encoding space.

3. **Invalid-step rate:** How often does decoding produce syntactically invalid equations?

### 5.3 Baselines

* **Monolithic IRED:** Train a single IRED-style EBM on single-rule data (no modular separation). At test, optimize this single energy on multi-rule problems.

* **Modular Sum (Ours):** Train four rule-level EBMs, sum their energies at test time, use IRED inference.

* **Neural Logic Machines:** Adapt NLM-style architecture to sequence algebraic transformations for comparison. ([arXiv][4])

---

## 6. Expected Results

We expect to observe:

| Model               | Single-Rule Accuracy | Multi-Rule Accuracy |
| ------------------- | -------------------- | ------------------- |
| Monolithic IRED     | ~90%                 | ~20–30%             |
| Modular Sum (ours)  | ~85%                 | **~50–60%**         |
| NLM-style baseline  | ~90%                 | ~70%+               |

The gap between "Monolithic IRED" and "Modular Sum" on multi-rule test sets would directly support the claim that *operator-level modular energies improve compositional generalization*.

---

## 7. Implementation Details

### 7.1 Integration with IRED Codebase

We use the existing IRED training infrastructure:

```python
# Pseudocode for training one rule's energy
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D

# Create dataset for one rule
dataset_distribute = AlgebraDataset(
    num_problems=50000,
    rules=('distribute',),
    problem_type='single_rule'
)

# Create EBM and wrap for diffusion training
ebm = AlgebraEBM(inp_dim=128, out_dim=128)
model = AlgebraDiffusionWrapper(ebm)

# IRED's diffusion wrapper
diffusion = GaussianDiffusion1D(
    model,
    seq_length=128,
    timesteps=10,  # K landscapes
    sampling_timesteps=10,
    supervise_energy_landscape=True,  # Enable contrastive loss
    use_innerloop_opt=True  # Enable T-step optimization per landscape
)

# IRED's trainer
trainer = Trainer1D(
    diffusion,
    dataset_distribute,
    train_batch_size=2048,
    train_lr=1e-4,
    train_num_steps=50000,
    results_folder='./results/distribute'
)

trainer.train()
```

Repeat for each rule. At test time, load all four EBMs and compose energies.

### 7.2 Ablations

**Encoder choice:**
* Char-level baseline
* AST encoder (SymPy tree embeddings)

**Number of modular energies:**
* 1 energy (monolithic)
* 4 energies (one per rule) ← our main approach
* 8 energies (finer-grained splits)

**Constraint energies:**
Define positivity constraint:
$$E_{\text{pos}}(y) = \max(0, -\text{solution\_value}(y))$$

Add at inference: $E_{\text{total}} + \beta E_{\text{pos}}$

---

## 8. Limitations

1. **Decoding from continuous space:** The model reasons in continuous embeddings, then we decode back to text. This can yield syntactically invalid expressions. Unlike images (where small errors are tolerable), algebra requires exact symbols.

2. **Single-variable linear focus:** We start with single-variable linear equations. Extending to quadratics, systems, or symbolic simplification requires more rule types.

3. **Comparison to program induction:** NLM-style systems that explicitly learn symbolic transformations may achieve higher accuracy. Our contribution is showing EBMs can realize similar modularity *without* executing a learned program.

4. **IRED's known limitations:** IRED inference requires many gradient steps and can be slow. For tasks with known specifications, dedicated algorithms are faster. However, IRED learns from data without task specifications.

---

## 9. Conclusion

We propose a concrete path to compositional algebraic reasoning with energy-based models:

* Train one IRED-style energy model per algebraic rule using only single-step data.
* At inference, **compose** these rule energies by summing them and **optimize** this composed landscape using IRED's annealed gradient descent.
* Use SymPy for symbolic verification.
* Demonstrate controllable reasoning by adding constraint energies at test time.

**Key clarification:** While IRED mentions compositional possibilities, it does not implement or evaluate them. Our work makes this concrete for symbolic algebra and provides the first empirical test of rule-level energy composition in a reasoning domain.

If experiments confirm:
1. Significant improvement in multi-rule accuracy over monolithic baseline (20–30 percentage points),
2. Successful constraint injection via added energies, and
3. Proper functioning of IRED's annealed optimization on composed landscapes,

then this becomes a self-contained workshop paper demonstrating compositional energy-based reasoning on a new symbolic domain.

---

## Appendix A. SymPy-Based Correctness Check

```python
def check_equivalence(eq_pred_str, eq_true_str):
    """
    Return True if both equations imply the same solution for x.
    """
    def solve_x(eq_str):
        lhs_s, rhs_s = eq_str.split("=")
        x = sp.Symbol('x')
        lhs = sp.sympify(lhs_s)
        rhs = sp.sympify(rhs_s)
        sol = sp.solve(sp.Eq(lhs, rhs), x)
        return sol
    
    pred_sol = solve_x(eq_pred_str)
    true_sol = solve_x(eq_true_str)
    
    if len(pred_sol) != len(true_sol):
        return False
    
    for a, b in zip(pred_sol, true_sol):
        if not sp.simplify(a - b) == 0:
            return False
    
    return True
```

---

## Appendix B. Training Loop with IRED's Actual Implementation

```python
def train_rule_energy_ired_style(rule_name, dataset, device='cuda'):
    """
    Train one rule's energy using actual IRED infrastructure.
    """
    # Create model
    ebm = AlgebraEBM(inp_dim=128, out_dim=128).to(device)
    model = AlgebraDiffusionWrapper(ebm)
    
    # IRED's GaussianDiffusion1D handles:
    # - Noise schedule (cosine)
    # - Denoising loss computation
    # - Contrastive loss computation
    # - Annealed landscape management
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=128,  # output dimension
        timesteps=10,  # K=10 landscapes
        sampling_timesteps=10,
        supervise_energy_landscape=True,  # Enables contrastive loss
        use_innerloop_opt=True  # Enables multi-step optimization
    )
    
    # IRED's Trainer1D handles:
    # - Data loading
    # - Training loop
    # - Checkpointing
    # - Validation
    trainer = Trainer1D(
        diffusion,
        dataset,
        train_batch_size=2048,
        train_lr=1e-4,
        train_num_steps=50000,
        results_folder=f'./results/{rule_name}',
        metric='mse'  # Evaluation metric
    )
    
    # Train
    trainer.train()
    
    return ebm
```

The actual loss computation happens inside `GaussianDiffusion1D.forward()`:

1. Sample timestep $k$
2. Corrupt output: $\tilde{y} = \sqrt{1-\sigma_k^2} y^+ + \sigma_k \epsilon$
3. Get model prediction: $\nabla_y E(x, \tilde{y}, k)$
4. MSE loss: $||\nabla_y E - \epsilon||^2$
5. If negatives provided, compute contrastive loss
6. Return combined loss

This matches IRED Algorithm 1 exactly.

---

### References

* **[1]** Iterative Reasoning through Energy Diffusion (IRED): Learning annealed energy landscapes with denoising + contrastive supervision; inference via gradient descent across landscapes with proper scaling. https://arxiv.org/abs/2406.11179

* **[2]** Compositional Energy Minimization: Proposes composing separately trained energies at inference; provides conceptual motivation for our approach. https://arxiv.org/html/2510.20607v1

* **[3]** Compositional Visual Generation with Energy Based Models: Shows attribute composition by summing energies in vision domain. https://energy-based-model.github.io/compositional-generation-inference/

* **[4]** Neural Logic Machines: Learns differentiable logic modules that extrapolate to longer reasoning chains. https://arxiv.org/abs/1904.11694