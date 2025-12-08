# Curriculum Learning Implementation Walkthrough

## Summary
Implemented the infrastructure for curriculum learning to improve compositional generalization in algebra EBMs. This allows models to be trained on intermediate steps extracted from complex multi-rule problems, rather than just single-rule examples.

## Changes Made

### 1. **Curriculum Dataset** (`src/algebra/algebra_dataset.py`)
- **Execution Tracing**: Modified `_generate_forward_multirule_problem` to return a trace of all intermediate equations (`input`, `step1`, `step2`... `target`).
- **CurriculumDataset Class**: Created a new dataset class that:
    - Generates multi-rule problems (depth 2, 3, or 4).
    - Extracts valid single-step transitions matching a target rule (e.g., `distribute`).
    - Exposes them as standard training pairs.
- **Training Support**: Updated `MultiRuleDataset` to allow `split='train'`.

### 2. **Training Script** (`train_algebra.py`)
- **New Arguments**: Added `--use_curriculum` and `--curriculum_depth`.
- **Integration**: Logic to instantiate `CurriculumDataset` when enabled.

### 3. **Cluster Deployment** (`scp.sh`)
- Updated to recursively copy `src/` and `scripts/` to ensure the new dataset logic and analysis scripts are deployed.

## How to Run

### 1. Deploy to Cluster
```bash
./scp.sh
```

### 2. Run Curriculum Training
To train the `distribute` model on steps extracted from 2-rule problems:
```bash
python train_algebra.py \
    --rule distribute \
    --use_curriculum True \
    --curriculum_depth 2 \
    --num_problems 50000 \
    --train_steps 10000
```
This trains the model on "harder" examples that naturally occur in multi-step chains.

### 3. Run Compositional Analysis
After training, evaluate performance across depths:
```bash
python scripts/compositional_analysis.py
```

## Next Steps
- Run the compositional analysis baseline.
- Train models with curriculum depth 2, then 3, then 4.
- Compare performance to baseline.
