# Algebra EBM Research Plan

**Project**: Compositional vs Monolithic Energy-Based Models for Algebraic Equation Solving
**Created**: 2026-02-11
**Last Updated**: 2026-02-16
**Current Phase**: DEBUG
**Git SHA**: 4dc7798

---

## Project Timeline

### Phase 1: Implementation (Completed: 2026-02-11)
- Implemented compositional EBM architecture with 4 rule models (distribute, combine, isolate, divide)
- Implemented monolithic baseline for comparison
- Created evaluation framework with 7 experiments
- Status: COMPLETE

### Phase 2: Initial Training & Evaluation (Completed: 2026-02-13)
- Trained all 5 models (4 rule + 1 monolithic)
- Ran evaluation suite (exp_001 through exp_007)
- Status: COMPLETE
- Results: Poor performance across board (single-rule ~6-29%, multi-rule 0-5%)

### Phase 3: Code Audits & Fixes (Completed: 2026-02-15)
- **AUDIT-001**: Fixed rule selection bug in evaluation
- **AUDIT-002**: Fixed inference hyperparameter mismatch
- **AUDIT-003**: Diagnosed IRED inference local minima problem
- **DATAGEN-001 through 005**: Fixed dataset generation issues
- Status: COMPLETE

### Phase 4: Retraining with Fixes (Completed: 2026-02-16)
- Retrained all 5 models with corrected training data (DATAGEN fixes)
- All models converged successfully (9-10 unit energy gaps)
- Status: COMPLETE

### Phase 5: Post-Fix Evaluation (Completed: 2026-02-16)
- Re-ran all evaluation experiments
- Results analyzed in `documentation/evaluation-analysis.md`
- **CRITICAL FINDING**: Training works (9-10 unit gaps), but inference completely fails (6% single-rule, 0% multi-rule)
- Status: COMPLETE

### Phase 6: Inference Diagnostics & Investigation (Current: 2026-02-16)
- **Goal**: Understand why IRED inference fails despite successful training
- **Status**: PENDING IMPLEMENTATION
- **Next Steps**: Implement diagnostic logging and experiments

---

## Original Hypothesis (Now REJECTED)

### Core Hypothesis
Compositional energy-based models enable:
1. Learning individual transformation rules independently
2. Composing rules at inference time for multi-step problems
3. Better generalization than monolithic approaches

### Why Hypothesis is Rejected

**Evidence from Current Results**:
- **Part 1 (Independent Learning)**: ✅ Partially works - models train successfully with 9-10 unit energy gaps
- **Part 2 (Composition at Inference)**: ❌ **FAILS** - 0% accuracy on all multi-rule problems
- **Part 3 (Better Generalization)**: ❌ **FAILS** - compositional (4.9%) barely better than monolithic (4.5%) in previous evaluation

**Critical Issue**: Training succeeds but inference fails. This is an **inference architecture problem**, not a training or compositional modeling problem.

---

## Current Findings Summary

### What Works
1. **Training Convergence**: All 5 models achieve strong energy discrimination
   - Positive energies (correct): ~5
   - Negative energies (incorrect): ~15
   - Energy gap: 9-10 units (excellent)

2. **Dataset Generation**: DATAGEN-001 through 005 fixes validated
   - Zero mathematical errors via SymPy verification
   - Proper format matching (equations, not bare expressions)
   - Validation functions properly unpacking tuples

3. **Code Infrastructure**: AUDIT-001 and AUDIT-002 fixes working
   - Rule selection properly extracts relevant rules
   - Inference hyperparameters use correct defaults (50 iters, 0.01 step)

### What Fails
1. **Single-Rule Inference**: 6.3% average accuracy (expected 85%)
   - distribute: 6.0%
   - combine: 6.3% (dropped from 100% pre-DATAGEN-002)
   - isolate: 6.1%
   - divide: 6.7%
   - Invalid rate: 0.0% (produces valid syntax, wrong answers)

2. **Multi-Rule Inference**: 0.0% accuracy across all configurations
   - 2-rule: 0.0% (1000 problems)
   - 3-rule: 0.0% (1000 problems)
   - 4-rule: 0.0% (1000 problems)

3. **Constrained Inference**: 0.0% accuracy across all constraint types
   - Positive constraint: 0.0%
   - Integer constraint: 0.0%
   - Both constraints: 0.0%

4. **Compositional Advantage**: Previous evaluation showed 4.9% vs 4.5% monolithic (marginal +0.4 points)

### Root Cause: IRED Inference Local Minima

**IRED Inference Process**:
1. Initialize latent embedding from random noise
2. Iteratively refine via gradient descent on energy landscape
3. Decode final embedding to output equation

**Known Problems** (from AUDIT-003):
- Energy landscapes have many local optima
- Single random initialization per problem (no multi-start)
- Fixed step sizes may overshoot/undershoot
- Fixed iteration count (50 steps)
- Embeddings may not have smooth gradients

**Evidence**:
- Training shows clear energy discrimination (gap=9-10)
- Inference produces valid syntax (0% invalid rate)
- But wrong solutions (6% accuracy suggests random guessing)
- Multi-rule compounds the problem (0% = product of failures)

**Why combine Dropped from 100% to 6.3%**:
- DATAGEN-002 changed combine from bare expressions to full equations
- Previous: `3*x + 5*x` → `8*x` (trivial, similar embeddings)
- Now: `3*x + 5*x = c` → `8*x = c` (more complex, harder embedding space)
- The one working case broke due to increased complexity

---

## Next Experiments (Inference Diagnostics)

### Priority 1: Add Inference Logging Infrastructure
**Goal**: Understand what's happening during IRED inference
**Implementation**:
- Log energy at each iteration
- Log gradient norms
- Log embedding evolution (L2 distance from start)
- Log acceptance rates (if using Langevin)
- Log final decoding attempts
**Save to**: `results/evaluation/{experiment_id}/diagnostics/problem_{i}_trajectory.json`

### Priority 2: Multi-Start Inference
**Goal**: Test if different random initializations find better solutions
**Implementation**:
- Add `--num_starts` parameter (default 10)
- Run inference from N different random seeds
- Keep result with lowest final energy
- Log which start index won
**Hypothesis**: If multi-start improves accuracy >20%, confirms local minima problem

### Priority 3: Configurable Iterations
**Goal**: Test if more iterations help escape local minima
**Implementation**:
- Add `--max_inference_iterations` parameter
- Default 50, allow up to 500
- Log iteration count used per problem
**Hypothesis**: If 500 iters >> 50 iters, need more optimization steps

### Priority 4: Momentum (Optional)
**Goal**: Help gradient descent escape local minima
**Implementation**:
- Add `momentum` parameter (default 0.9)
- Update gradient step: `velocity = momentum * velocity + grad`
**Hypothesis**: Momentum helps traverse flat regions of energy landscape

### Priority 5: Input-Guided Initialization (Optional)
**Goal**: Test if starting from input embedding helps
**Implementation**:
- Encode input equation through model
- Use as starting point instead of random noise
- Compare against random init
**Hypothesis**: Starting near input may be closer to target than random noise

### Priority 6: Manual Dataset Verification
**Goal**: Ensure test datasets are actually correct
**Implementation**:
- Manually inspect 10-20 problems per rule type
- Verify mathematical correctness with independent SymPy
- Check format matches training data
- Verify single rule application achieves target
**Hypothesis**: If datasets wrong, explains everything

### Priority 7: Debug exp_007 Missing Results
**Goal**: Understand monolithic baseline performance
**Implementation**:
- Investigate why evaluation report empty
- Re-run if needed with added logging
**Importance**: Critical for comparing compositional vs monolithic

---

## Decision Points & Pivot Criteria

### Decision Point 1: After Diagnostic Experiments (1 week, 2026-02-23)
**If multi-start and increased iterations improve single-rule accuracy to >50%**:
→ Continue with inference improvements, proceed to multi-rule composition

**If improvements <20%**:
→ Deep dive into energy landscape smoothness, consider architectural changes

**If improvements <5%**:
→ Consider pivoting away from IRED inference entirely

### Decision Point 2: After Single-Rule Fixed (2 weeks, 2026-03-02)
**If single-rule >70% accuracy achieved**:
→ Attack multi-rule composition problem
→ Test if composition works once individual rules work

**If single-rule stuck at 20-50%**:
→ Re-evaluate compositional EBM approach viability
→ Consider alternative architectures

### Decision Point 3: After Multi-Rule Attempts (3 weeks, 2026-03-09)
**If multi-rule >30% accuracy achieved**:
→ Proceed with research paper, document approach

**If multi-rule <10% accuracy**:
→ Abandon compositional EBM approach
→ Pivot to alternative: seq2seq transformers, graph neural nets, or hybrid symbolic-neural

---

## Alternative Approaches to Consider

### If Pivoting Away from Compositional EBMs:

**Option A: Seq2Seq Transformers**
- Encoder-decoder architecture
- Input: equation string
- Output: transformed equation string
- Rule label as conditioning
- Pros: Proven on similar tasks, no energy landscape issues
- Cons: Less interpretable, requires more data

**Option B: Graph Neural Networks**
- Operate on equation parse trees
- Learn graph transformations
- Compositional by design (tree operations)
- Pros: Structured representation, interpretable
- Cons: More complex implementation

**Option C: Hybrid Symbolic-Neural**
- Neural module suggests rule applications
- Symbolic executor applies transformations
- Combine strengths of both approaches
- Pros: Guarantees correctness, compositional
- Cons: Complex system, may need rule templates

**Option D: Monolithic Diffusion (Non-Compositional)**
- Single large model handles all transformations
- Use diffusion for inference (not IRED)
- Pros: Avoids composition complexity
- Cons: Loses compositional benefits, black box

---

## Success Metrics

### Short-Term (1 week)
- [ ] Implement inference logging infrastructure
- [ ] Run diagnostic experiments on 10-50 problems
- [ ] Identify specific failure modes in IRED inference
- [ ] Determine feasibility of fixing via hyperparameters

### Medium-Term (2 weeks)
- [ ] Single-rule accuracy >50% (from current 6%)
- [ ] Understand energy landscape properties
- [ ] Clear path forward for multi-rule composition

### Long-Term (4 weeks)
- [ ] Single-rule accuracy >70%
- [ ] Multi-rule accuracy >30%
- [ ] Compositional advantage over monolithic >10 points
- [ ] OR: Successful pivot to alternative approach with preliminary results

---

## Resources & References

### Key Files
- Implementation: `src/algebra/algebra_models.py`, `src/algebra/algebra_inference.py`
- Evaluation: `src/algebra/algebra_evaluation.py`
- Training: `train_algebra.py`
- Dataset: `src/algebra/algebra_dataset.py`

### Key Documentation
- Evaluation analysis: `documentation/evaluation-analysis.md`
- Debugging log: `documentation/debugging.md`
- Pipeline state: `.state/pipeline.json`

### Key Results
- Training models: `results/{rule}/model.pt`
- Evaluation results: `results/evaluation/exp_{001-007}/`
- Test datasets: `results/test_datasets/*.json`

### Git Commits
- DATAGEN fixes: 4dc7798
- AUDIT fixes: 3e94761
- Option 1 energy fix: ea707cb

---

## Open Questions

1. **Why did combine drop from 100% to 6.3%?**
   - DATAGEN-002 changed format from expressions to equations
   - Increased embedding space complexity?
   - Or exposed existing inference problem that was masked?

2. **Are test datasets actually correct?**
   - DATAGEN fixes validated programmatically
   - But manual inspection not yet done
   - Could subtle errors remain?

3. **Is IRED fundamentally unsuitable for this task?**
   - Works well in some domains (image generation)
   - Algebra may have too complex energy landscapes
   - May need different inference strategy entirely

4. **Can composition work if single-rule works?**
   - Current 0% multi-rule may be due to compounding 6% single-rule
   - Or composition adds new failure modes
   - Need to fix single-rule first to know

5. **What does monolithic baseline show?**
   - exp_007 produced no results output
   - Previous run: 29% single-rule, 4.5% multi-rule
   - Need to re-run for fair comparison

---

## Timeline Estimates

**Best Case** (everything works with hyperparameter tuning):
- Week 1: Diagnostic experiments reveal simple fixes
- Week 2: Single-rule >50% with multi-start + momentum
- Week 3: Multi-rule >30% with composition working
- Week 4: Paper draft, results validated

**Realistic Case** (some pivoting needed):
- Week 1: Diagnostics show fundamental issues
- Week 2: Try architectural changes (embedding space, energy formulation)
- Week 3: Partial improvements (single-rule ~40%, multi-rule ~10%)
- Week 4: Decision to pivot or continue

**Worst Case** (approach is dead end):
- Week 1: Diagnostics show insurmountable problems
- Week 2: Confirm IRED unsuitable for algebra domain
- Week 3: Design alternative approach (transformer baseline)
- Week 4: Begin implementation of new approach

---

## Conclusion

The compositional EBM approach has a **critical inference failure** despite successful training. The next phase focuses on understanding and fixing IRED inference through diagnostic experiments. If inference can be fixed (target: >50% single-rule), the approach remains viable. If not, we pivot to alternative architectures within 2-3 weeks.

**Current Status**: Phase 6 (Inference Diagnostics) - Implementation pending
**Next Action**: Implement diagnostic logging and multi-start experiments
**Critical Path**: Fix single-rule inference first, then tackle multi-rule composition
