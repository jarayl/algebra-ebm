# Full Algebra EBM Implementation Plan
## Step-by-Step Guide for IRED Codebase Integration

---

## Phase 1: Data Infrastructure

### Step 1: Create Algebraic Equation Encoder
**File to Create:** `algebra_encoder.py`
**Reference Files:** None (new module)

**Description:** Implement equation encoding to convert symbolic algebra strings into continuous embeddings. The proposal specifies two encoder variants: character-level baseline and AST-based encoder using SymPy.

**Key Components:**
- Character-level encoder with vocabulary: `'0123456789x+-=*/() '`
- One-hot encoding per character, flattened and projected to `d_model=128`
- AST encoder using SymPy's symbolic expression trees
- Reversible decoding function using nearest-neighbor search

**Technical Details:**
- Input: equation string (e.g., `"2(x+3)+4=10"`)
- Output: `(128,)` continuous embedding vector
- Must handle variable-length inputs with padding to `max_len=64`

---

### Step 2: Create Algebra Dataset Classes
**File to Create:** `algebra_dataset.py`
**Reference Files:** `dataset.py` (Addition, LowRankDataset classes lines 227-400)

**Description:** Implement PyTorch Dataset classes for generating and loading algebraic equation problems. Create separate datasets for each rule type and for compositional evaluation.

**Key Components:**
1. **`AlgebraDataset`** - Base class for single-rule problems
   - Rules: `distribute`, `combine`, `isolate`, `divide`
   - Generate pairs: (input_equation, target_equation)
   - Use SymPy to verify correctness
   - `num_problems=50000` per rule for training

2. **`MultiRuleDataset`** - For compositional testing
   - Generate equations requiring 2-4 sequential rule applications
   - Never seen during training (zero-shot evaluation)
   
3. **`ConstrainedDataset`** - For constraint evaluation
   - Add positivity/integerness requirements to test problems

**Technical Details:**
- Inherit from `torch.utils.data.Dataset`
- Implement `__getitem__` to return encoded (inp, target) pairs
- Set `self.inp_dim = 128` and `self.out_dim = 128`
- Generate synthetic equations with random coefficients in range `[-10, 10]`

---

### Step 3: Create Noisy Dataset Wrapper
**File to Modify:** Use existing `NoisyWrapper` from `dataset.py` (lines 144-198)

**Description:** Wrap algebra datasets with IRED's noise corruption for denoising training. No modifications needed - the existing wrapper already implements the correct cosine noise schedule.

**Key Details:**
- Uses `cosine_beta_schedule(timesteps=10)` matching K=10 landscapes
- Corrupts targets: `y_tilde = sqrt(1-sigma_k^2) * y + sigma_k * epsilon`
- Returns: `(inp, corrupted_target, next_corrupted_target)` for contrastive loss

---

## Phase 2: Model Architecture

### Step 4: Implement AlgebraEBM Energy Model
**File to Create:** `algebra_models.py`
**Reference Files:** `models.py` (EBM class lines 1-200), Table 8 from IRED paper

**Description:** Create the core energy function architecture for algebraic rules. This is the heart of the system - one model per rule that learns to score algebraic validity.

**Architecture Specification (matching IRED Table 8):**
```python
class AlgebraEBM(nn.Module):
    Input: 
        - inp: (B, 128) - encoded input equation
        - out: (B, 128) - encoded candidate output
        - t: (B,) - landscape timestep [0-9]
    
    Layers:
        - Time MLP: SinusoidalPosEmb(128) → Linear(128) → GELU → Linear(128)
        - FC1: Linear(inp_dim + out_dim → 512) + Swish
        - FC2: Linear(512 → 512) + FiLM(time_emb) + Swish
        - FC3: Linear(512 → 512) + FiLM(time_emb) + Swish
        - Out: Linear(512 → out_dim)
    
    Output: energy = ||output_vector||^2 (L2 norm squared)
```

**Key Implementation Points:**
- FiLM conditioning: `h = fc(h) * (1 + scale) + shift` where scale/shift from time embedding
- Energy must be non-negative (ensured by L2 norm)
- Separate instances for each of 4 rules

---

### Step 5: Implement Diffusion Wrapper
**File to Modify:** `algebra_models.py` (same as Step 4)
**Reference Files:** `models.py` (DiffusionWrapper lines 201-250)

**Description:** Wrap AlgebraEBM to interface with IRED's GaussianDiffusion1D trainer. The wrapper computes energy gradients needed for score matching.

**Key Components:**
```python
class AlgebraDiffusionWrapper(nn.Module):
    def forward(self, inp, out, t):
        # Enable gradient computation
        out = out.requires_grad_(True)
        
        # Get energy value
        energy = self.ebm(inp, out, t)
        
        # Compute gradient dE/dout
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=out,
            create_graph=True
        )[0]
        
        return grad  # Shape: (B, 128)
```

**Critical:** Gradients must be computed with `create_graph=True` for backpropagation through the energy function.

---

## Phase 3: Training Infrastructure

### Step 6: Integrate with GaussianDiffusion1D
**File to Use:** Existing `diffusion_lib/denoising_diffusion_pytorch_1d.py`
**No modifications needed**

**Description:** IRED's diffusion module already implements:
- Cosine noise schedule
- Denoising loss: `||∇_y E - epsilon||^2`
- Contrastive landscape loss: `-log(exp(-E+) / (exp(-E+) + exp(-E-)))`
- Multi-step optimization during training (`use_innerloop_opt=True`)

**Usage:**
```python
diffusion = GaussianDiffusion1D(
    model=AlgebraDiffusionWrapper(ebm),
    seq_length=128,
    timesteps=10,  # K landscapes
    sampling_timesteps=10,
    supervise_energy_landscape=True,  # Enable contrastive
    use_innerloop_opt=True  # Enable T-step opt
)
```

---

### Step 7: Set Up Training Loop with Trainer1D
**File to Use:** Existing `diffusion_lib/denoising_diffusion_pytorch_1d.py`
**No modifications needed**

**Description:** IRED's Trainer1D handles all training logistics. Already implements data loading, checkpointing, validation, and metric tracking.

**Usage:**
```python
trainer = Trainer1D(
    diffusion,
    dataset_wrapped,  # NoisyWrapper(AlgebraDataset)
    train_batch_size=2048,
    train_lr=1e-4,
    train_num_steps=50000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    results_folder=f'./results/{rule_name}',
    save_and_sample_every=1000,
    metric='mse'
)
```

---

### Step 8: Create Main Training Script
**File to Create:** `train_algebra.py`
**Reference Files:** `train.py` (lines 1-200)

**Description:** Main entry point for training individual rule energies. This orchestrates the training of all 4 rule-specific EBMs.

**Key Components:**
1. Parse command-line arguments (rule name, hyperparameters)
2. Load appropriate AlgebraDataset for selected rule
3. Wrap in NoisyWrapper
4. Initialize AlgebraEBM + AlgebraDiffusionWrapper
5. Create GaussianDiffusion1D
6. Create Trainer1D
7. Run training loop

**Example Command:**
```bash
python train_algebra.py --rule distribute --batch_size 2048 \
    --timesteps 10 --train_steps 50000 \
    --supervise-energy-landscape True --use-innerloop-opt True
```

**Must train 4 times (once per rule):**
- `--rule distribute`
- `--rule combine`
- `--rule isolate`
- `--rule divide`

---

## Phase 4: Inference Implementation

### Step 9: Implement IRED-Style Inference
**File to Create:** `algebra_inference.py`
**Reference Files:** Proposal Section 4.5, IRED Algorithm 2

**Description:** Implement the annealed gradient descent inference procedure that optimizes through K=10 energy landscapes. This is the test-time optimization that solves equations.

**Algorithm (from proposal):**
```python
def ired_inference(energy_fns, inp, K=10, T=20, step_size=0.1):
    """
    energy_fns: dict of {rule_name: EBM_model} for composition
    inp: encoded input equation (128,)
    K: number of landscapes
    T: gradient steps per landscape
    """
    # 1. Initialize from noise
    out = torch.randn(128)
    
    # 2. Cosine schedule for landscape scaling
    alphas_cumprod = cosine_schedule(K)
    
    # 3. Iterate through landscapes
    for k in range(K):
        sigma_k = sqrt(1 - alphas_cumprod[k])
        
        # T gradient descent steps
        for t in range(T):
            # Compute composed energy gradient
            grad = 0
            for rule_name, ebm in energy_fns.items():
                out_grad = compute_energy_gradient(ebm, inp, out, k)
                grad += out_grad
            
            # Update with gradient descent
            out_new = out - step_size * grad
            
            # Accept only if energy decreases
            energy_old = sum([ebm(inp, out, k) for ebm in energy_fns.values()])
            energy_new = sum([ebm(inp, out_new, k) for ebm in energy_fns.values()])
            if energy_new < energy_old:
                out = out_new
        
        # Scale for next landscape
        if k < K-1:
            sigma_k_next = sqrt(1 - alphas_cumprod[k+1])
            out = out * (sigma_k_next / sigma_k)
    
    return out
```

**Critical:** Must use proper landscape scaling between k values, and energy check before accepting updates.

---

### Step 10: Implement Equation Decoding
**File to Modify:** `algebra_inference.py`

**Description:** Convert continuous embeddings back to symbolic equation strings. Uses nearest-neighbor search over candidate valid equations.

**Key Components:**
1. Generate candidate set of syntactically valid equations
2. Encode all candidates to embedding space
3. Find nearest neighbor to predicted embedding
4. Verify with SymPy

**Implementation:**
```python
def decode_equation(embedding, candidate_pool, encoder):
    """
    embedding: (128,) continuous vector
    candidate_pool: list of valid equation strings
    encoder: AlgebraEncoder instance
    """
    # Encode all candidates
    candidate_embeddings = [encoder(eq) for eq in candidate_pool]
    
    # Find nearest neighbor (L2 distance)
    distances = [torch.norm(embedding - emb) for emb in candidate_embeddings]
    best_idx = torch.argmin(distances)
    
    return candidate_pool[best_idx]
```

---

### Step 11: Implement Compositional Energy Summation
**File to Modify:** `algebra_inference.py`

**Description:** Core novelty - compose multiple rule energies at inference time by summing. This enables zero-shot multi-rule generalization.

**Key Components:**
```python
def compose_energies(rule_ebms, inp, out, k, lambdas=None):
    """
    rule_ebms: dict {rule_name: EBM_model}
    lambdas: optional weights (default all 1.0)
    """
    if lambdas is None:
        lambdas = {rule: 1.0 for rule in rule_ebms.keys()}
    
    total_energy = 0
    for rule_name, ebm in rule_ebms.items():
        energy = ebm(inp, out, k)
        total_energy += lambdas[rule_name] * energy
    
    return total_energy
```

**Usage:** Load 4 trained rule EBMs, compose at test time to solve multi-rule problems.

---

## Phase 5: Evaluation Framework

### Step 12: Implement SymPy Correctness Checker
**File to Create:** `algebra_evaluation.py`
**Reference Files:** Proposal Appendix A

**Description:** Use SymPy to verify symbolic equivalence of predicted solutions. This is the primary evaluation metric.

**Key Components:**
```python
def check_equivalence(pred_eq_str, true_eq_str):
    """
    Returns True if both equations have same solution for x.
    """
    def solve_x(eq_str):
        lhs_str, rhs_str = eq_str.split("=")
        x = sp.Symbol('x')
        lhs = sp.sympify(lhs_str)
        rhs = sp.sympify(rhs_str)
        sol = sp.solve(sp.Eq(lhs, rhs), x)
        return sol
    
    pred_sol = solve_x(pred_eq_str)
    true_sol = solve_x(true_eq_str)
    
    if len(pred_sol) != len(true_sol):
        return False
    
    for a, b in zip(pred_sol, true_sol):
        if not sp.simplify(a - b) == 0:
            return False
    
    return True
```

---

### Step 13: Implement Evaluation Metrics
**File to Modify:** `algebra_evaluation.py`

**Description:** Compute evaluation metrics across test sets. Track multiple metrics to diagnose failure modes.

**Metrics to Implement:**
1. **Symbolic Equivalence (Primary):** % of predictions that solve to correct x value
2. **Embedding L2 Distance (Auxiliary):** `||y_pred - y_true||_2` in embedding space
3. **Invalid Step Rate:** % of decoded equations that are syntactically invalid
4. **Per-Rule Breakdown:** Accuracy split by which rules are required

**Implementation:**
```python
def evaluate_model(model_dict, test_dataset, decoder):
    """
    model_dict: {rule_name: trained_EBM}
    test_dataset: MultiRuleDataset instance
    """
    correct = 0
    total = 0
    l2_distances = []
    invalid_count = 0
    
    for inp_str, target_str in test_dataset:
        # Encode
        inp_emb = encoder(inp_str)
        
        # Inference
        pred_emb = ired_inference(model_dict, inp_emb)
        
        # Decode
        pred_str = decode_equation(pred_emb, candidate_pool, encoder)
        
        # Check validity
        try:
            is_valid = is_syntactically_valid(pred_str)
        except:
            is_valid = False
            
        if not is_valid:
            invalid_count += 1
            continue
        
        # Check equivalence
        if check_equivalence(pred_str, target_str):
            correct += 1
        
        # L2 distance
        target_emb = encoder(target_str)
        l2_distances.append(torch.norm(pred_emb - target_emb))
        
        total += 1
    
    return {
        'accuracy': correct / total,
        'invalid_rate': invalid_count / total,
        'mean_l2': torch.mean(torch.stack(l2_distances))
    }
```

---

### Step 14: Create Evaluation Script
**File to Create:** `eval_algebra.py`
**Reference Files:** None (new script)

**Description:** Main script for running all evaluations. Tests single-rule, multi-rule, and constrained variants.

**Evaluation Sets:**
1. **Single-Rule Test:** Held-out problems from each rule's distribution
2. **Multi-Rule Test (2 rules):** Equations requiring 2 sequential rules
3. **Multi-Rule Test (3 rules):** Equations requiring 3 sequential rules
4. **Multi-Rule Test (4 rules):** Equations requiring all 4 rules
5. **Constrained Test:** Multi-rule + positivity/integerness constraints

**Expected Results (from proposal Section 6):**
- Single-Rule Accuracy: ~85%
- Multi-Rule Accuracy: ~50-60%
- Monolithic Baseline Multi-Rule: ~20-30%

---

## Phase 6: Constraint Energies

### Step 15: Implement Constraint Energy Functions
**File to Create:** `algebra_constraints.py`
**Reference Files:** Proposal Section 3.4

**Description:** Additional energy functions for test-time constraints. These are NOT learned, just hand-designed functions that can be added to the composed energy.

**Key Constraints:**

1. **Positivity Constraint:**
```python
class PositivityEnergy(nn.Module):
    def forward(self, inp, out, k):
        """
        Penalize if solution x < 0
        """
        # Decode to get x value
        x_value = extract_solution_value(out)
        
        # Energy penalty if negative
        energy = torch.max(torch.zeros_like(x_value), -x_value)
        
        return energy
```

2. **Integerness Constraint:**
```python
class IntegernessEnergy(nn.Module):
    def forward(self, inp, out, k):
        """
        Penalize if solution x is not an integer
        """
        x_value = extract_solution_value(out)
        
        # Distance to nearest integer
        nearest_int = torch.round(x_value)
        energy = (x_value - nearest_int) ** 2
        
        return energy
```

**Usage:**
```python
# At inference time, add to composed energy
total_energy = (
    sum_rule_energies(...) + 
    beta_pos * positivity_energy(...) +
    beta_int * integerness_energy(...)
)
```

---

### Step 16: Test Constraint Injection
**File to Modify:** `eval_algebra.py`

**Description:** Validate that constraint energies successfully bias solutions toward desired properties without retraining.

**Test Cases:**
1. Same multi-rule problem with/without positivity constraint
2. Verify solution changes from negative to positive
3. Same for integerness constraint
4. Measure constraint satisfaction rate

---

## Phase 7: Baseline Implementations

### Step 17: Implement Monolithic IRED Baseline
**File to Create:** `train_algebra_monolithic.py`
**Reference Files:** `train_algebra.py` (Step 8)

**Description:** Train single unified EBM on all single-rule data combined (not separated by rule). This is the main baseline to beat.

**Key Difference:**
- Same architecture as AlgebraEBM
- Dataset combines all 4 rule types: 200k problems total
- No compositional structure - one energy for everything
- Expected to fail on multi-rule problems (~20-30% vs ~50-60% for modular)

---

### Step 18: Implement NLM Baseline (Optional)
**File to Create:** `train_algebra_nlm.py`
**Reference Files:** `diffusion_lib/nlm.py`, `diffusion_lib/nlm_utils.py`

**Description:** Adapt Neural Logic Machines architecture as comparative baseline. This represents the neuro-symbolic alternative.

**Key Components:**
- Use existing NLM modules from IRED codebase
- Learn discrete transformation operators for each rule
- Execute symbolic transformations directly
- Expected to achieve ~70%+ on multi-rule problems

**Note:** This is lower priority - focus on demonstrating modular EBM improvement over monolithic first.

---

## Phase 8: Ablation Studies

### Step 19: Implement Encoder Ablations
**File to Modify:** `algebra_encoder.py`

**Description:** Compare character-level vs AST-based encoders as specified in proposal Section 7.2.

**Variants:**
1. **Char-level (baseline):** Simple one-hot per character
2. **AST encoder:** Use SymPy to parse expressions into trees, embed tree structure

**Expected:** AST encoder should improve accuracy by ~5-10% by leveraging algebraic structure.

---

### Step 20: Implement Energy Granularity Ablations
**File to Create:** `train_algebra_ablations.py`

**Description:** Test different numbers of modular energies as specified in proposal Section 7.2.

**Variants:**
1. **1 energy (monolithic):** Baseline - single EBM
2. **4 energies:** Main approach - one per rule
3. **8 energies:** Finer-grained - split rules into sub-operations

**Example 8-energy split:**
- Distribute multiply
- Distribute addition
- Combine like terms (same coefficient)
- Combine like terms (different coefficient)
- Isolate variable (addition)
- Isolate variable (subtraction)
- Divide coefficient (positive)
- Divide coefficient (negative)

**Expected:** 4 energies should be sweet spot. 8 may overfit.

---

## Phase 9: Analysis and Visualization

### Step 21: Implement Energy Landscape Visualization
**File to Create:** `visualize_landscapes.py`
**Reference Files:** IRED paper Figure 4

**Description:** Visualize learned energy landscapes to understand what the model learns. Plot energy values across the solution space for different landscape indices k.

**Key Visualizations:**
1. **Energy vs Solution Distance:** Plot E(x,y,k) as function of ||y - y*||
2. **Per-Landscape Maps:** Show how landscapes sharpen from k=1 to k=10
3. **Composed Energy:** Visualize sum of rule energies vs individual energies

---

### Step 22: Implement Inference Trajectory Visualization
**File to Create:** `visualize_inference.py`

**Description:** Track and plot the optimization trajectory during inference. Show how the solution evolves across landscapes.

**Key Plots:**
1. **Energy Decrease:** Plot total energy over optimization steps
2. **Per-Rule Energy:** Show individual rule energy contributions
3. **Solution Convergence:** Plot ||y_t - y*|| over time
4. **Landscape Transitions:** Mark when k increases

---

## Phase 10: Integration and Testing

### Step 23: Create End-to-End Pipeline
**File to Create:** `run_full_experiment.py`

**Description:** Master script that runs entire experimental pipeline from training to evaluation.

**Workflow:**
1. Train 4 rule-specific EBMs (parallel or sequential)
2. Train monolithic baseline
3. Evaluate on all test sets
4. Generate comparison tables
5. Create visualizations
6. Save results to JSON/CSV

---

### Step 24: Implement Unit Tests
**File to Create:** `test_algebra.py`

**Description:** Comprehensive unit tests for all components.

**Test Coverage:**
1. Encoder/decoder round-trip
2. Dataset generation correctness (verify with SymPy)
3. Energy function output shapes
4. Gradient computation correctness
5. Inference convergence on toy problems
6. Constraint energy behavior

---

### Step 25: Create Results Analysis Notebook
**File to Create:** `analysis.ipynb`

**Description:** Jupyter notebook for analyzing results and generating paper figures.

**Analyses:**
1. Comparison table (Section 6 of proposal)
2. Accuracy by number of rules required
3. Constraint satisfaction rates
4. Ablation study results
5. Energy landscape visualizations
6. Failure case analysis

---

## Summary of Key Files to Create/Modify

### New Files to Create:
1. `algebra_encoder.py` - Equation encoding/decoding
2. `algebra_dataset.py` - Dataset classes
3. `algebra_models.py` - AlgebraEBM + wrapper
4. `train_algebra.py` - Main training script
5. `algebra_inference.py` - IRED inference for algebra
6. `algebra_evaluation.py` - Metrics and evaluation
7. `eval_algebra.py` - Evaluation script
8. `algebra_constraints.py` - Constraint energies
9. `train_algebra_monolithic.py` - Baseline training
10. `train_algebra_nlm.py` - NLM baseline (optional)
11. `train_algebra_ablations.py` - Ablation variants
12. `visualize_landscapes.py` - Energy visualization
13. `visualize_inference.py` - Trajectory visualization
14. `run_full_experiment.py` - Master pipeline
15. `test_algebra.py` - Unit tests
16. `analysis.ipynb` - Results notebook

### Existing Files to Use (No Modifications):
1. `diffusion_lib/denoising_diffusion_pytorch_1d.py` - GaussianDiffusion1D, Trainer1D
2. `dataset.py` - NoisyWrapper, cosine_beta_schedule
3. `models.py` - Reference for architecture patterns

### Dependencies to Install:
- PyTorch
- SymPy (for symbolic algebra)
- NumPy
- tqdm
- matplotlib (for visualization)
- jupyter (for analysis)

---

## Estimated Timeline

- **Phase 1-2 (Data + Models):** 1 week
- **Phase 3 (Training Infrastructure):** 3 days
- **Phase 4 (Inference):** 1 week
- **Phase 5 (Evaluation):** 3 days
- **Phase 6 (Constraints):** 2 days
- **Phase 7 (Baselines):** 1 week
- **Phase 8 (Ablations):** 3 days
- **Phase 9 (Visualization):** 3 days
- **Phase 10 (Integration):** 3 days

**Total Estimated Time:** 6-8 weeks

---

## Critical Implementation Notes

1. **Energy Function Sign:** Energy MUST be non-negative. Use L2 norm squared as in IRED.

2. **Landscape Scaling:** Proper scaling between landscapes k is critical: `y *= (sigma_{k+1} / sigma_k)`

3. **Gradient Computation:** Must use `create_graph=True` for backprop through energy gradients.

4. **Contrastive Loss:** Requires both positive and negative examples. Use corrupted versions of ground truth.

5. **SymPy Verification:** Always verify generated equations are solvable before adding to dataset.

6. **Decoding Strategy:** Nearest-neighbor search requires large candidate pool (10k+ equations) for good coverage.

7. **Constraint Weights:** Beta values for constraint energies need tuning - start with 0.1-1.0 range.

8. **Training Stability:** Use EMA (exponential moving average) of model weights as in original IRED.

9. **Inference Step Size:** May need adaptive step size per landscape. Start with 0.1, decrease for later k.

10. **GPU Memory:** Batch size 2048 requires ~16GB GPU. Reduce if needed.

---

## Expected Experimental Results

From Proposal Section 6:

| Model               | Single-Rule Acc | Multi-Rule Acc |
|---------------------|----------------|----------------|
| Monolithic IRED     | ~90%           | ~20–30%        |
| **Modular Sum**     | **~85%**       | **~50–60%**    |
| NLM Baseline        | ~90%           | ~70%+          |

**Success Criteria:**
- Modular approach achieves 20-30 percentage point improvement over monolithic on multi-rule
- Successfully injects constraints without retraining
- Demonstrates proper IRED landscape optimization on algebraic domain

---

This completes the comprehensive step-by-step implementation plan!