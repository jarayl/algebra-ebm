# Algebra EBM Implementation Todo List

This document tracks our progress through implementing the full Algebra EBM system based on the IRED framework.

## 📊 Overall Progress: 6/25 Steps Completed (24%)

**Completed Infrastructure:**
- ✅ Algebraic equation encoding/decoding system (Step 1)  
- ✅ Noisy dataset wrapper (Step 3)
- ✅ GaussianDiffusion1D integration (Step 6)
- ✅ Trainer1D setup (Step 7) 
- ✅ Equation decoding via nearest-neighbor (Step 10)
- ✅ SymPy correctness checking (Step 12)

**Ready to Implement Next:**
- 📋 Algebra dataset classes (Step 2)
- 📋 AlgebraEBM energy model (Step 4)
- 📋 Diffusion wrapper (Step 5)

---

## Phase 1: Data Infrastructure ✅ 2/3 Complete

### ✅ **Step 1: Create Algebraic Equation Encoder** ✅ COMPLETED
- [x] File: `algebra_encoder.py` 
- [x] Character-level encoder with vocabulary: `'0123456789x+-=*/() '`
- [x] One-hot encoding per character, flattened and projected to `d_model=128`
- [x] AST encoder using SymPy's symbolic expression trees
- [x] Reversible decoding function using nearest-neighbor search
- [x] Handle variable-length inputs with padding to `max_len=64`
- [x] SymPy validation and equivalence checking functions included

### ⏳ **Step 2: Create Algebra Dataset Classes**
- [ ] File: `algebra_dataset.py`
- [ ] `AlgebraDataset` - Base class for single-rule problems (distribute, combine, isolate, divide)
- [ ] `MultiRuleDataset` - For compositional testing (2-4 sequential rule applications)
- [ ] `ConstrainedDataset` - For constraint evaluation (positivity/integerness)
- [ ] Generate 50,000 problems per rule for training
- [ ] Inherit from `torch.utils.data.Dataset`
- [ ] Set `self.inp_dim = 128` and `self.out_dim = 128`

### ✅ **Step 3: Set Up Noisy Dataset Wrapper** ✅ COMPLETED
- [x] Use existing `NoisyWrapper` from `dataset.py`
- [x] Verify cosine noise schedule with `timesteps=10`
- [x] Ensure corruption: `y_tilde = sqrt(1-sigma_k^2) * y + sigma_k * epsilon`

---

## Phase 2: Model Architecture ⏳

### ✅ **Step 4: Implement AlgebraEBM Energy Model**
- [ ] File: `algebra_models.py`
- [ ] Time MLP: SinusoidalPosEmb(128) → Linear(128) → GELU → Linear(128)
- [ ] FC1: Linear(inp_dim + out_dim → 512) + GELU
- [ ] FC2: Linear(512 → 512) + FiLM(time_emb) + GELU
- [ ] FC3: Linear(512 → 512) + FiLM(time_emb) + GELU
- [ ] Output: Linear(512 → out_dim), energy = ||output_vector||^2
- [ ] FiLM conditioning: `h = fc(h) * (1 + scale) + shift`

### ✅ **Step 5: Implement Diffusion Wrapper**
- [ ] File: `algebra_models.py` (same file as Step 4)
- [ ] `AlgebraDiffusionWrapper` class
- [ ] Enable gradient computation with `out.requires_grad_(True)`
- [ ] Compute energy gradients with `create_graph=True`
- [ ] Return gradient shape: `(B, 128)`

---

## Phase 3: Training Infrastructure ✅ 2/3 Complete

### ✅ **Step 6: Integrate with GaussianDiffusion1D** ✅ COMPLETED
- [x] Use existing `diffusion_lib/denoising_diffusion_pytorch_1d.py`
- [x] Configure: `seq_length=128`, `timesteps=10`, `supervise_energy_landscape=True`
- [x] Set `use_innerloop_opt=True` for T-step optimization

### ✅ **Step 7: Set Up Training Loop with Trainer1D** ✅ COMPLETED
- [x] Use existing Trainer1D from diffusion lib
- [x] Configure: `train_batch_size=2048`, `train_lr=1e-4`, `train_num_steps=50000`
- [x] Set `ema_decay=0.995`, `gradient_accumulate_every=1`

### ✅ **Step 8: Create Main Training Script**
- [ ] File: `train_algebra.py`
- [ ] Parse command-line arguments for rule name and hyperparameters
- [ ] Train 4 separate models: distribute, combine, isolate, divide
- [ ] Save models to `./results/{rule_name}/`

---

## Phase 4: Inference Implementation ✅ 1/3 Complete

### ✅ **Step 9: Implement IRED-Style Inference**
- [ ] File: `algebra_inference.py`
- [ ] `ired_inference()` function with K=10 landscapes, T=20 gradient steps
- [ ] Initialize from noise: `out = torch.randn(128)`
- [ ] Cosine schedule for landscape scaling
- [ ] Energy-based acceptance criteria for gradient updates
- [ ] Proper landscape scaling: `out *= (sigma_k_next / sigma_k)`

### ✅ **Step 10: Implement Equation Decoding** ✅ COMPLETED
- [x] File: `algebra_encoder.py` (implemented as EquationDecoder class)
- [x] `decode_equation()` function using nearest-neighbor search
- [x] Generate candidate pool of syntactically valid equations
- [x] Find nearest neighbor using L2 distance in embedding space
- [x] Verify with SymPy before returning

### ✅ **Step 11: Implement Compositional Energy Summation**
- [ ] File: `algebra_inference.py` (same file)
- [ ] `compose_energies()` function
- [ ] Sum multiple rule energies with optional lambda weights
- [ ] Load 4 trained rule EBMs for test-time composition

---

## Phase 5: Evaluation Framework ✅ 1/3 Complete

### ✅ **Step 12: Implement SymPy Correctness Checker** ✅ COMPLETED
- [x] File: `algebra_encoder.py` (implemented as helper functions)
- [x] `check_equivalence()` function (`check_equation_equivalence()`)
- [x] Parse equations and solve for x using SymPy (`solve_equation()`)
- [x] Compare solution sets for symbolic equivalence
- [x] Handle multiple solutions and edge cases
- [x] Syntax validation function (`validate_equation_syntax()`) included

### ✅ **Step 13: Implement Evaluation Metrics**
- [ ] File: `algebra_evaluation.py` (same file)
- [ ] Symbolic Equivalence (Primary): % correct x values
- [ ] Embedding L2 Distance (Auxiliary): `||y_pred - y_true||_2`
- [ ] Invalid Step Rate: % syntactically invalid decoded equations
- [ ] Per-Rule Breakdown: Accuracy split by required rules

### ✅ **Step 14: Create Evaluation Script**
- [ ] File: `eval_algebra.py`
- [ ] Single-Rule Test: Held-out problems from each rule
- [ ] Multi-Rule Test: 2, 3, and 4 sequential rule combinations
- [ ] Constrained Test: Multi-rule + positivity/integerness constraints
- [ ] Expected results: Single-Rule ~85%, Multi-Rule ~50-60%

---

## Phase 6: Constraint Energies ⏳

### ✅ **Step 15: Implement Constraint Energy Functions**
- [ ] File: `algebra_constraints.py`
- [ ] `PositivityEnergy`: Penalize if solution x < 0
- [ ] `IntegernessEnergy`: Penalize non-integer solutions
- [ ] Hand-designed functions (not learned)
- [ ] Additive to composed rule energies

### ✅ **Step 16: Test Constraint Injection**
- [ ] Modify `eval_algebra.py`
- [ ] Test same problems with/without constraints
- [ ] Verify solution bias toward desired properties
- [ ] Measure constraint satisfaction rate

---

## Phase 7: Baseline Implementations ⏳

### ✅ **Step 17: Implement Monolithic IRED Baseline**
- [ ] File: `train_algebra_monolithic.py`
- [ ] Same AlgebraEBM architecture
- [ ] Combined dataset: all 4 rules together (200k problems)
- [ ] Expected: ~90% single-rule, ~20-30% multi-rule

### ✅ **Step 18: Implement NLM Baseline (Optional)**
- [ ] File: `train_algebra_nlm.py`
- [ ] Use existing NLM modules from IRED codebase
- [ ] Learn discrete transformation operators
- [ ] Expected: ~70%+ multi-rule accuracy

---

## Phase 8: Ablation Studies ⏳

### ✅ **Step 19: Implement Encoder Ablations**
- [ ] Modify `algebra_encoder.py`
- [ ] Compare character-level vs AST-based encoders
- [ ] Expected: AST ~5-10% improvement

### ✅ **Step 20: Implement Energy Granularity Ablations**
- [ ] File: `train_algebra_ablations.py`
- [ ] Test 1 energy (monolithic), 4 energies (main), 8 energies (fine-grained)
- [ ] Expected: 4 energies as sweet spot

---

## Phase 9: Analysis and Visualization ⏳

### ✅ **Step 21: Implement Energy Landscape Visualization**
- [ ] File: `visualize_landscapes.py`
- [ ] Energy vs Solution Distance plots
- [ ] Per-Landscape maps (k=1 to k=10)
- [ ] Composed vs individual rule energies

### ✅ **Step 22: Implement Inference Trajectory Visualization**
- [ ] File: `visualize_inference.py`
- [ ] Energy decrease over optimization steps
- [ ] Per-rule energy contributions
- [ ] Solution convergence trajectories
- [ ] Landscape transition markers

---

## Phase 10: Integration and Testing ⏳

### ✅ **Step 23: Create End-to-End Pipeline**
- [ ] File: `run_full_experiment.py`
- [ ] Train 4 rule-specific EBMs
- [ ] Train monolithic baseline
- [ ] Run all evaluations
- [ ] Generate comparison tables and visualizations
- [ ] Save results to JSON/CSV

### ✅ **Step 24: Implement Unit Tests**
- [ ] File: `test_algebra.py`
- [ ] Encoder/decoder round-trip tests
- [ ] Dataset generation correctness (SymPy verification)
- [ ] Energy function output shapes
- [ ] Gradient computation correctness
- [ ] Inference convergence on toy problems
- [ ] Constraint energy behavior

### ✅ **Step 25: Create Results Analysis Notebook**
- [ ] File: `analysis.ipynb`
- [ ] Comparison table matching proposal Section 6
- [ ] Accuracy by number of rules required
- [ ] Constraint satisfaction rates
- [ ] Ablation study results
- [ ] Energy landscape visualizations
- [ ] Failure case analysis

---

## Critical Implementation Notes ⚠️

1. **Energy Function Sign**: Energy MUST be non-negative (use L2 norm squared)
2. **Landscape Scaling**: Proper scaling between landscapes: `y *= (sigma_{k+1} / sigma_k)`
3. **Gradient Computation**: Use `create_graph=True` for backprop through energy gradients
4. **Contrastive Loss**: Requires both positive and negative examples
5. **SymPy Verification**: Always verify generated equations are solvable
6. **Decoding Strategy**: Need large candidate pool (10k+ equations) for coverage
7. **Constraint Weights**: Beta values need tuning (start with 0.1-1.0)
8. **Training Stability**: Use EMA of model weights
9. **Inference Step Size**: May need adaptive step size per landscape
10. **GPU Memory**: Batch size 2048 requires ~16GB GPU

---

## Expected Results 🎯

| Model               | Single-Rule Acc | Multi-Rule Acc |
|---------------------|----------------|----------------|
| Monolithic IRED     | ~90%           | ~20–30%        |
| **Modular Sum**     | **~85%**       | **~50–60%**    |
| NLM Baseline        | ~90%           | ~70%+          |

**Success Criteria:**
- ✅ 20-30 percentage point improvement over monolithic on multi-rule
- ✅ Successful constraint injection without retraining  
- ✅ Proper IRED landscape optimization on algebraic domain

---

## Dependencies 📦

- PyTorch
- SymPy (for symbolic algebra)
- NumPy
- tqdm
- matplotlib (for visualization)
- jupyter (for analysis)

---

**Estimated Timeline:** 6-8 weeks total
- Phase 1-2: 1 week
- Phase 3: 3 days  
- Phase 4: 1 week
- Phase 5: 3 days
- Phase 6: 2 days
- Phase 7: 1 week
- Phase 8: 3 days
- Phase 9: 3 days
- Phase 10: 3 days