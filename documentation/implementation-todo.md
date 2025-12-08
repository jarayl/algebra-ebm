# Algebra EBM Implementation TODO List

**Generated:** 2025-12-08  
**Based on:** Comprehensive analysis of implementation plan, current codebase status, and critical bug reports

This todo list provides a detailed roadmap for completing the algebra EBM implementation. Each step includes dependencies, success criteria, implementation details, and concurrency guidelines to enable efficient parallel development.

## <¯ **Executive Summary**

**Current Status:** ~85% Complete - Core infrastructure implemented, but models need retraining and critical fixes  
**Estimated Completion:** 2-3 weeks (assuming parallel execution of independent tasks)  
**Primary Blockers:** Model retraining with bug fixes, baseline comparisons, constraint handling

## =Ê **Priority Classification**

- =¨ **CRITICAL** - Blocks core functionality
- =% **HIGH** - Major features for successful demonstration  
- =á **MEDIUM** - Important but not blocking
- =â **LOW** - Nice-to-have features

---

## Phase 1: Critical Bug Fixes & Model Retraining

### =¨ **CRITICAL: Fix Loss Scale Imbalance**
- [ ] **Task**: Implement adaptive loss scaling in training pipeline
- **File**: `src/diffusion/denoising_diffusion_pytorch_1d.py:1084`
- **Dependencies**: None - can start immediately
- **Parallel**:  Can run concurrently with other fixes
- **Issue**: Energy loss contributes only 0.3% vs 99.7% for MSE, causing flat energy landscapes
- **Success Criteria**:
  - Energy contribution increased to 40-60% of total loss
  - Energy gaps between correct/incorrect solutions: 8-12 units (currently ~1 unit)
  - Training loss shows balanced MSE and energy components
- **Implementation Details**:
  - Replace hardcoded `loss_scale = 0.5` with adaptive scaling
  - Monitor energy component percentage during training
  - Target: energy_loss / (mse_loss + energy_loss)  [0.4, 0.6]
- **Careful Of**: Ensure energy scaling doesn't cause training instability
- **References**: `documentation/reports/energy-landscape-flatness-research-2025-12-06.md`

### =¨ **CRITICAL: Fix Dataset Format Bugs**  
- [ ] **Task**: Resolve coefficient sign handling in equation generation
- **File**: `src/algebra/algebra_dataset.py:289, 294, 418, 423`
- **Dependencies**: None - can start immediately
- **Parallel**:  Can run concurrently with loss scaling fix
- **Issue**: 25-40% of training data contains malformed equations like "3*x+-15=42"
- **Success Criteria**:
  - Zero malformed equations in generated datasets
  - All equations pass SymPy parsing validation
  - Proper sign handling for negative coefficients
- **Implementation Details**:
  - Fix string formatting to handle negative coefficients: `f"{coeff:+d}"` ’ `f"{coeff}"`  
  - Add validation step to reject malformed equations during generation
  - Update coefficient formatting for all rule types
- **Careful Of**: Ensure fix doesn't reduce dataset variety
- **References**: `documentation/reports/algebra-ebm-performance-bugs-2025-12-08.md`

### =¨ **CRITICAL: Fix Energy Caching Bug**
- [ ] **Task**: Optimize energy computation to avoid redundant forward passes
- **File**: `src/algebra/algebra_inference.py:454-530`
- **Dependencies**: None - performance optimization
- **Parallel**:  Can run concurrently with other fixes
- **Issue**: 30-50% performance degradation from redundant neural network calls
- **Success Criteria**:
  - Cache energy values within optimization steps
  - 30-50% inference speedup
  - Maintain numerical accuracy
- **Implementation Details**:
  - Cache energy values from previous iterations when input unchanged
  - Add cache invalidation when models or timesteps change
  - Profile inference time before/after optimization
- **Careful Of**: Ensure cache correctness doesn't introduce bugs
- **References**: Multiple energy caching test files in `/tests/debug/`

### =¨ **CRITICAL: Retrain Models with Bug Fixes**
- [ ] **Task**: Train all 4 rule-specific EBMs using fixed training pipeline
- **File**: `train_algebra.py`
- **Dependencies**: Must complete loss scaling, dataset format, and energy caching fixes
- **Parallel**: L Sequential - depends on bug fixes
- **Issue**: Current models show 0.0% accuracy due to flat energy landscapes
- **Success Criteria**:
  - 4 models trained (distribute, combine, isolate, divide)
  - Energy parameters (energy_scale, energy_bias) properly learned
  - Energy landscape shows 8-12 unit gaps between correct/incorrect
  - Single-rule accuracy e 80%
- **Implementation Details**:
  - Train each rule separately: `python train_algebra.py --rule [distribute|combine|isolate|divide]`
  - Use fixed loss scaling and dataset generation
  - Monitor training metrics for energy landscape sharpness
  - Save checkpoints with proper energy parameters
- **Careful Of**: Verify energy_scale ` 1.0 and energy_bias ` 0.0 after training
- **Estimated Time**: 4-6 hours per model (16-24 hours total)

### =¨ **CRITICAL: Fix Evaluation Configuration**
- [ ] **Task**: Update evaluation script to use correct distance threshold
- **File**: `eval_algebra.py:649`
- **Dependencies**: None - immediate fix
- **Parallel**:  Can fix immediately
- **Issue**: Distance threshold mismatch (35.0 vs 50.0) causes all predictions to be rejected
- **Success Criteria**:
  - Evaluation uses emergency threshold (50.0) matching inference default
  - Predictions accepted for decoding and correctness checking
  - Non-zero accuracy visible in evaluation results
- **Implementation Details**:
  ```python
  # Change line 649
  decoder = create_decoder_with_default_candidates(encoder, distance_threshold=50.0)
  ```
- **Careful Of**: This is a temporary fix until decoder candidate set is rebuilt
- **References**: `documentation/reports/zero-accuracy-evaluation-debug-2025-12-08.md`

---

## Phase 2: Core Implementation Completion

### =% **HIGH: Implement Monolithic IRED Baseline**
- [ ] **Task**: Create monolithic training script for comparison
- **File**: Create `train_algebra_monolithic.py`
- **Dependencies**: Phase 1 bug fixes completed
- **Parallel**:  Can develop while Phase 1 models train
- **Issue**: Need baseline comparison to demonstrate modular improvement
- **Success Criteria**:
  - Single EBM trained on combined data from all rules (200k problems)
  - Same architecture as rule-specific models
  - Expected performance: ~20-30% on multi-rule tasks
- **Implementation Details**:
  - Combine datasets from all 4 rules into single training set
  - Use identical AlgebraEBM architecture  
  - No rule-specific separation in training
  - Save as unified model for multi-rule evaluation
- **Careful Of**: Ensure fair comparison - same architecture and training parameters
- **Estimated Time**: 1 day development + 6 hours training

### =% **HIGH: Implement Constraint Energy Functions**
- [ ] **Task**: Create hand-designed constraint energies for test-time injection
- **File**: Create `src/algebra/algebra_constraints.py`
- **Dependencies**: Phase 1 model training completed
- **Parallel**:  Can develop in parallel with training
- **Issue**: Need demonstration of constraint injection capability
- **Success Criteria**:
  - PositivityEnergy: penalizes negative solutions
  - IntegernessEnergy: penalizes non-integer solutions
  - Composable with rule energies at inference time
- **Implementation Details**:
  ```python
  class PositivityEnergy(nn.Module):
      def forward(self, inp, out, k):
          x_value = extract_solution_value(out)
          return torch.max(torch.zeros_like(x_value), -x_value)
  ```
- **Careful Of**: Constraint weights (beta values) need tuning in range [0.1, 1.0]
- **Estimated Time**: 2-3 days

### =% **HIGH: Test Constraint Injection**
- [ ] **Task**: Validate constraint energies bias solutions correctly
- **File**: Modify evaluation to include constraint testing
- **Dependencies**: Constraint energy functions completed
- **Parallel**: L Sequential - depends on constraint implementation
- **Success Criteria**:
  - Same problem with/without positivity shows solution change (negative ’ positive)
  - Integerness constraint produces integer solutions
  - Constraint satisfaction rate e 80%
- **Implementation Details**:
  - Evaluate same test problems with different constraint combinations
  - Measure constraint violation rates before/after injection
  - Generate comparison reports
- **Careful Of**: Ensure constraints don't break primary task accuracy
- **Estimated Time**: 1-2 days

### =% **HIGH: Create End-to-End Pipeline**
- [ ] **Task**: Master script to orchestrate complete experimental workflow
- **File**: Create `run_full_experiment.py`
- **Dependencies**: All training and evaluation components completed
- **Parallel**: L Sequential - orchestrates everything
- **Issue**: Need automated pipeline for reproducible results
- **Success Criteria**:
  - Train all models (4 modular + 1 monolithic)
  - Evaluate on all test sets (single-rule, multi-rule, constrained)
  - Generate comparison tables and visualizations
  - Save results to structured JSON/CSV
- **Implementation Details**:
  - Command-line interface for experimental configuration
  - Progress tracking and checkpoint management
  - Automatic result compilation and reporting
- **Careful Of**: Handle training failures gracefully with checkpoints
- **Estimated Time**: 2-3 days

---

## Phase 3: Performance Optimization & Analysis

### =á **MEDIUM: Fix Decoder Candidate Set Mismatch**
- [ ] **Task**: Rebuild decoder with comprehensive equation candidates
- **File**: `src/algebra/algebra_evaluation.py:317-331`
- **Dependencies**: Dataset format fixes completed
- **Parallel**:  Can work in parallel with training
- **Issue**: Decoder has only 49 equations vs thousands needed for coverage
- **Success Criteria**:
  - Decoder candidate set covers training distribution
  - Distance threshold reduced back to normal range (1.5-3.0)
  - Improved decoding accuracy and reduced distances
- **Implementation Details**:
  - Generate comprehensive candidate set from dataset templates
  - Include equations for all coefficient ranges and rule combinations
  - Validate candidate coverage against test problems
- **Careful Of**: Balance candidate set size vs computation time
- **Estimated Time**: 2-3 days

### =á **MEDIUM: Implement Energy Granularity Ablations**
- [ ] **Task**: Test different numbers of modular energies (1, 4, 8)
- **File**: Create `train_algebra_ablations.py`
- **Dependencies**: Core training pipeline working
- **Parallel**:  Can run in parallel with main experiments
- **Success Criteria**:
  - Train models with 1, 4, and 8 energy functions
  - Compare performance across granularity levels
  - Demonstrate 4-energy optimality
- **Implementation Details**:
  - 8-energy variant: split rules into sub-operations
    - Distribute: multiply vs addition
    - Combine: same vs different coefficients  
    - Isolate: addition vs subtraction
    - Divide: positive vs negative coefficients
- **Careful Of**: 8-energy may overfit with limited training data
- **Estimated Time**: 3-4 days

### =á **MEDIUM: Encoder Comparison Evaluation**
- [ ] **Task**: Compare character-level vs AST-based encoder performance
- **File**: Use existing encoders in `algebra_encoder.py`
- **Dependencies**: Model training completed
- **Parallel**:  Can evaluate in parallel
- **Success Criteria**:
  - Train models with both encoder types
  - Compare accuracy on same test sets
  - Expected: AST encoder ~5-10% improvement
- **Implementation Details**:
  - Re-train representative models with AST encoder
  - Keep all other parameters identical
  - Generate detailed comparison analysis
- **Careful Of**: Ensure fair comparison with identical training procedures
- **Estimated Time**: 2-3 days

### =á **MEDIUM: Implement Energy Landscape Visualization**
- [ ] **Task**: Create visualization tools for learned energy landscapes
- **File**: Create `visualize_landscapes.py`
- **Dependencies**: Trained models available
- **Parallel**:  Can develop independently
- **Success Criteria**:
  - Energy vs solution distance plots
  - Per-landscape sharpening (k=1 to k=10)
  - Composed energy vs individual rule energies
- **Implementation Details**:
  - Sample solution space around ground truth
  - Plot energy surfaces for different landscape indices
  - Compare individual vs composed energy landscapes
- **Careful Of**: Visualization performance with high-dimensional spaces
- **Estimated Time**: 2-3 days

### =á **MEDIUM: Implement Inference Trajectory Visualization**
- [ ] **Task**: Visualize optimization trajectories during inference
- **File**: Create `visualize_inference.py`
- **Dependencies**: Inference pipeline working
- **Parallel**:  Can develop independently
- **Success Criteria**:
  - Track energy decrease over optimization steps
  - Show per-rule energy contributions
  - Plot solution convergence over time
  - Mark landscape transitions clearly
- **Implementation Details**:
  - Instrument inference with trajectory logging
  - Create interactive plots for optimization paths
  - Compare successful vs failed inference attempts
- **Careful Of**: Logging overhead shouldn't affect inference performance
- **Estimated Time**: 2-3 days

---

## Phase 4: Advanced Features & Analysis

### =á **MEDIUM: Fix Minor Performance Bugs**
- [ ] **Task**: Address remaining encoder and data quality issues
- **Files**: `algebra_encoder.py:69`, `algebra_dataset.py:455-456`
- **Dependencies**: None - independent fixes
- **Parallel**:  Can fix anytime
- **Issues**: 
  - Encoder crashes on unknown characters
  - Silent zero coefficient fallback creates wrong solutions
- **Success Criteria**:
  - Extended encoder vocabulary handles all characters
  - Zero coefficient detection triggers regeneration
  - 100% data quality validation
- **Implementation Details**:
  - Extend VOCAB to include all possible equation characters
  - Replace silent fallback with explicit regeneration
  - Add comprehensive data validation pipeline
- **Careful Of**: Ensure fixes don't reduce data variety
- **Estimated Time**: 1 day

### =â **LOW: Implement NLM Baseline (Optional)**
- [ ] **Task**: Neural Logic Machine baseline for comparison
- **File**: Create `train_algebra_nlm.py`
- **Dependencies**: Existing NLM modules in IRED codebase
- **Parallel**:  Independent baseline implementation
- **Success Criteria**:
  - Adapt existing NLM architecture to algebraic rules
  - Expected performance: ~70%+ on multi-rule tasks
  - Demonstrate EBM competitive advantage
- **Implementation Details**:
  - Use existing NLM modules from diffusion_lib
  - Learn discrete transformation operators
  - Execute symbolic transformations directly
- **Careful Of**: This is optional - focus on modular vs monolithic first
- **Estimated Time**: 1 week

### =â **LOW: Comprehensive Unit Test Coverage**
- [ ] **Task**: Expand unit test coverage for all components
- **File**: Extend tests in `/tests/unit/`
- **Dependencies**: None - continuous improvement
- **Parallel**:  Can develop throughout implementation
- **Current Status**: Partial coverage exists
- **Success Criteria**:
  - Test coverage e 90% for core components
  - Integration tests for full pipelines
  - Regression tests for bug fixes
- **Implementation Details**:
  - Constraint energy testing
  - End-to-end pipeline validation
  - Performance regression detection
- **Careful Of**: Don't let test development slow feature implementation
- **Estimated Time**: Ongoing, 30 minutes per component

### =â **LOW: Create Results Analysis Notebook**
- [ ] **Task**: Jupyter notebook for publication-quality analysis
- **File**: Create `analysis.ipynb`
- **Dependencies**: Complete experimental results
- **Parallel**: L Sequential - requires all results
- **Success Criteria**:
  - Comparison tables matching proposal Section 6
  - Accuracy breakdown by number of rules required
  - Constraint satisfaction analysis
  - Energy landscape visualizations
  - Statistical significance testing
- **Implementation Details**:
  - Publication-ready figures and tables
  - Statistical analysis of experimental results
  - Error analysis and failure case categorization
- **Careful Of**: Ensure statistical rigor in comparisons
- **Estimated Time**: 2-3 days after results available

---

## = **Dependency Tree & Parallel Execution Plan**

### **Week 1: Critical Foundation** 
**Parallel Block 1A** (No dependencies):
- [ ] Fix loss scale imbalance
- [ ] Fix dataset format bugs  
- [ ] Fix energy caching bug
- [ ] Fix evaluation configuration
- [ ] Start constraint energy development

**Sequential Block 1B** (Depends on 1A):
- [ ] Retrain all 4 models with fixes

### **Week 2: Core Implementation**
**Parallel Block 2A** (Depends on Week 1):
- [ ] Train monolithic baseline
- [ ] Complete constraint energies
- [ ] Fix decoder candidate set
- [ ] Start visualization tools

**Parallel Block 2B** (Independent):
- [ ] Fix minor performance bugs
- [ ] Expand unit tests
- [ ] Energy granularity ablations

### **Week 3: Integration & Analysis**  
**Sequential Block 3A** (Depends on Weeks 1-2):
- [ ] Test constraint injection
- [ ] Create end-to-end pipeline
- [ ] Run comprehensive evaluation

**Parallel Block 3B** (Independent):
- [ ] Complete visualization tools
- [ ] Encoder comparison evaluation
- [ ] Results analysis notebook

### **Optional Extensions** (If Time Permits):
- [ ] NLM baseline implementation
- [ ] Advanced constraint types
- [ ] Performance optimization

---

## =Ý **Success Metrics**

### **Phase 1 Complete** 
- Models train successfully with balanced loss (energy 40-60%)
- Energy landscapes show 8-12 unit gaps
- Zero malformed training examples
- Evaluation returns non-zero accuracy

### **Phase 2 Complete**   
- Single-rule accuracy e 80% for all rule-specific models
- Multi-rule accuracy e 50% for modular approach
- Constraint injection successfully biases solutions
- Monolithic baseline shows d 30% multi-rule accuracy

### **Phase 3 Complete** 
- Full experimental pipeline runs end-to-end
- Comprehensive evaluation across all test sets
- Statistical comparison validates modular improvement
- Visualization tools demonstrate energy landscape behavior

### **Success Criteria Validation**
- [ ] Modular approach shows 20-30 percentage point improvement over monolithic
- [ ] Constraint injection works without retraining  
- [ ] IRED-style inference demonstrates proper landscape optimization
- [ ] Results support compositional energy-based reasoning claims

---

##   **Critical Implementation Notes**

### **Energy Function Requirements**
- Energy MUST be non-negative (use L2 norm squared)
- Proper landscape scaling: `y *= (sigma_k_next / sigma_k)` 
- Gradient computation with `create_graph=True`
- Learnable energy_scale and energy_bias parameters

### **Training Stability**
- Use EMA (exponential moving average) of model weights
- Adaptive step size per landscape (start 0.1, decrease for later k)
- Batch size 2048 requires ~16GB GPU memory
- Contrastive loss requires both positive and negative examples

### **Evaluation Rigor**  
- Always verify with SymPy before using generated equations
- Nearest-neighbor decoding requires large candidate pool (10k+ equations)
- Constraint weights (beta values) need tuning in 0.1-1.0 range
- Statistical significance testing for performance comparisons

---

## =Ú **Key References**

- **Implementation Plan**: `documentation/implementation_plan.md` 
- **Bug Reports**: `documentation/reports/` (3 major analysis reports)
- **IRED Paper**: Iterative Reasoning through Energy Diffusion methodology
- **Test Files**: Comprehensive unit and integration test coverage in `/tests/`

**Estimated Total Time**: 2-3 weeks with parallel execution  
**Critical Path**: Model retraining (Week 1) ’ Baseline training (Week 2) ’ Full evaluation (Week 3)  
**Success Probability**: High - most infrastructure complete, mainly execution and validation remaining