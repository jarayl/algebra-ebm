# Working CLI Examples for Monolithic Evaluation

## Basic Monolithic Evaluation
```bash
python eval_algebra.py \
    --eval_type monolithic \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --max_samples 100 \
    --output_dir ./eval_test
```

## With Custom Parameters
```bash
python eval_algebra.py \
    --eval_type monolithic \
    --monolithic_checkpoint /path/to/custom/model.pt \
    --max_samples 500 \
    --output_dir ./monolithic_results \
    --device cuda \
    --verbose
```

## Quick Test Mode
```bash
python eval_algebra.py \
    --eval_type monolithic \
    --monolithic_checkpoint ./results/monolithic/model.pt \
    --quick_test \
    --max_samples 50
```

## Expected Output Structure
When the monolithic checkpoint exists, the command will:

1. Load the monolithic model from the checkpoint
2. Evaluate on single-rule datasets (distribute, combine, isolate, divide)
3. Evaluate on multi-rule datasets (2-rule, 3-rule, 4-rule)
4. Save results to `{output_dir}/monolithic_evaluation.json`
5. Print summary statistics

## Integration Point
This infrastructure is ready to use once training completes and produces:
- `./results/monolithic/model.pt` (monolithic model checkpoint)

## Current Status
✅ Function implemented: `run_monolithic_evaluation()`
✅ CLI integration complete: `--eval_type monolithic`
✅ Arguments working: `--monolithic_checkpoint`
✅ Dataset loading tested
✅ Error handling in place
⏳ Awaiting: Monolithic model training completion