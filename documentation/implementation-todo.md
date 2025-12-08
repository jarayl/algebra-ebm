# Implementation To‑Do List (Generated from `implementation-plan.md`)

> **Goal**: Bring the multi‑rule evaluation pipeline to a working state, then improve model compositional generalisation.

---

## ✅ Immediate Fix (< 1 hour)
| # | Task | Success Criteria | Dependencies | Notes / Pitfalls |
|---|------|-------------------|--------------|-------------------|
| 1 | ✅ **Pass `distance_threshold` through the evaluation chain** – modify `src/algebra/algebra_evaluation.py` at line ~406 to call `solve_equation(..., distance_threshold=decoder.distance_threshold)` | Evaluation runs without `ValueError` and uses the decoder's threshold (10.0) instead of the default 6.0 | None | Ensure import of `decoder` is available; keep signature compatible with other callers. |
| 2 | ✅ **Raise default decoder threshold for multi‑rule tests** – edit `eval_algebra.py` line 649 to `create_decoder_with_default_candidates(encoder, distance_threshold=35.0)` | Decoding of multi‑rule problems no longer fails due to a too‑low threshold (should accept distances up to ~35). | Task 1 (optional – can be done concurrently) | Do not commit a value that harms other test suites; 35.0 is a safety‑margin based on 29.6×1.2. |

*These two changes can be made in parallel, but both must be merged before the next step.*

---

## 📋 Short‑Term Fix (1‑3 days)
| # | Task | Success Criteria | Dependencies | Notes / Pitfalls |
|---|------|-------------------|--------------|-------------------|
| 3 | ✅ **Add diagnostic logging for decoder rebuild** – in `src/algebra/algebra_evaluation.py` before/after `create_decoder_from_dataset` log candidate counts and a sample of equations. | Log output shows `Decoder before rebuild: X candidates` and `Decoder after rebuild: Y candidates` with Y > 49. | Tasks 1 & 2 completed and code compiled. | Use `logger.info`; avoid excessive logging in production. |
| 4 | ✅ **Verify rebuild actually occurs** – run a quick evaluation (e.g., `python eval_algebra.py --eval_type multi_rule --num_rules 4 --max_problems 5`) and confirm logs from Task 3 appear. | Confirmation that the rebuild path is exercised for each test dataset. | Task 3 implemented. | Ensure the test run uses the same decoder instance passed to `evaluate_model_suite`. |
| 5 | ✅ **Inspect target equations** – add a small script (or notebook) that loads `MultiRuleDataset(num_rules=4, num_problems=100)` and prints the first 10 target equations. Save output to `logs/multi_rule_targets.log`. | Developer can see concrete equation forms (e.g., `-8x+-50=-130`). | Tasks 3 & 4 completed. | No code changes required beyond a helper script; keep it out of the main package. |

*Tasks 3‑5 can be performed concurrently once the immediate fixes are merged.*

---

## 📈 Medium‑Term Fix (1‑2 weeks)
| # | Task | Success Criteria | Dependencies | Notes / Pitfalls |
|---|------|-------------------|--------------|-------------------|
| 6 | **Investigate compositional generalisation** – run the model on a curriculum of 2‑rule → 3‑rule → 4‑rule problems, record L2 distances and accuracy per rule depth. | A clear report (e.g., `reports/compositional_analysis.md`) showing degradation trends. | Tasks 1‑5 completed and baseline works. | Use existing `eval_algebra.py` with `--num_rules` flag; may need to adjust batch size.
| 7 | **Implement curriculum learning** – modify training script to first train on `num_rules=2`, then fine‑tune on `3`, finally on `4`. | After training, multi‑rule‑4 accuracy improves > 10 % (or distance drops below 15). | Completion of Task 6 (analysis) to justify effort. | Ensure checkpoint saving between stages; avoid overwriting good weights.
| 8 | **Run distance‑threshold optimiser** – use `src/algebra/distance_threshold_optimizer.py` on the new evaluation data to compute an empirically optimal threshold. | Optimiser outputs a threshold value that yields ≥ 95 % success rate on validation set. | Tasks 6‑7 (model trained) and logs available. | Keep the optimiser script configurable; do not hard‑code paths.

*Tasks 6‑8 are sequential: analysis → curriculum → optimisation.*

---

## 🏗️ Long‑Term Fix (1 + month)
| # | Task | Success Criteria | Dependencies | Notes / Pitfalls |
|---|------|-------------------|--------------|-------------------|
| 9 | **Redesign decoding strategy** – prototype a seq2seq transformer decoder (or character‑level autoregressive decoder) that generates equation strings directly. | Prototype can decode at least 80 % of multi‑rule‑4 targets on a held‑out set. | Completion of medium‑term work to confirm infrastructure is stable. | This is a research‑grade change; keep the NN‑based decoder as a fallback.
| 10 | **Integrate end‑to‑end training** – add a decoding loss term to the EBM training loop so the model is penalised for producing embeddings far from any valid equation. | Training loss curve shows decreasing decoding loss; evaluation accuracy improves. | Task 9 prototype ready and API stable. | Requires careful weighting of loss terms to avoid destabilising energy training.
| 11 | **Documentation & CI** – update `README.md`, add unit tests for the new decoder, and ensure CI runs the full evaluation suite with the new thresholds. | CI passes on every push; documentation reflects new workflow. | All functional changes merged. | Keep CI time reasonable; use a subset of problems for quick checks.

*Long‑term tasks can be worked on in parallel by separate contributors once the groundwork (Tasks 1‑8) is solid.*

---

## 📌 General Success Criteria
- **Zero runtime errors** during `python eval_algebra.py` for all `--eval_type` values.
- **Multi‑rule‑4 accuracy > 0 %** (ideally > 10 %).
- **Distance thresholds** are data‑driven (generated by optimiser) and documented.
- **Logs** clearly indicate decoder candidate count before/after rebuild.
- **All new code** passes linting (`flake8`) and has unit‑test coverage ≥ 80 % for critical paths.

---

## 🔧 Things to Be Careful Of
- Do not hard‑code absolute paths; use `os.path.join` or relative imports.
- Preserve backward compatibility for existing single‑rule evaluations.
- When raising thresholds, ensure they are not so high that random embeddings are accepted.
- Logging should be configurable (e.g., respect `LOG_LEVEL` env var) to avoid noisy CI output.
- Any new decoder model must be versioned and stored under `models/` with clear naming.

---

*This to‑do list is deliberately exhaustive to allow developers to pick independent work streams while respecting required dependencies.*
