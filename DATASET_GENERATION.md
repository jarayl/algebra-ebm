# Algebra Dataset Generation Guide

This guide shows you how to generate and verify algebra datasets for EBM training.

## Quick Start (Generate All Datasets)

```bash
# Generate all datasets for all rules (12 datasets total)
./generate_all_datasets.sh

# Verify all generated datasets
./verify_all_datasets.sh
```

This generates **50,000 training + 10,000 test + 10,000 val** problems for each of the 4 rules.

---

## Dataset Sizes (Recommended for Training)

| Dataset Type | Train Size | Test Size | Val Size | Purpose |
|--------------|------------|-----------|----------|---------|
| **Single Rule** | 50,000 | 10,000 | 10,000 | Rule-specific EBM training |
| **Multi-Rule** | 50,000 | 10,000 | 10,000 | Compositional evaluation |
| **Constrained** | 50,000 | 5,000 | 5,000 | Constraint testing |

**Why 50,000?**
- Sufficient for EBM convergence (based on similar tasks in codebase)
- Generates quickly (~5-10 minutes per dataset)
- Small enough to fit in memory
- Can increase to 100,000+ if needed

---

## Individual Dataset Generation

### Generate Single Rule Dataset

```bash
# Distribute rule (train split)
python gen_algebra_dataset.py --rule distribute --split train --size 50000

# Combine rule (test split)
python gen_algebra_dataset.py --rule combine --split test --size 10000

# Isolate rule (validation split)
python gen_algebra_dataset.py --rule isolate --split val --size 10000

# Divide rule (all splits)
python gen_algebra_dataset.py --rule divide --split train --size 50000
python gen_algebra_dataset.py --rule divide --split test --size 10000
python gen_algebra_dataset.py --rule divide --split val --size 10000
```

### Generate All Single-Rule Datasets at Once

```bash
python gen_algebra_dataset.py --all
```

This generates train/test/val for all 4 rules (distribute, combine, isolate, divide).

### Generate Multi-Rule Dataset (for compositional testing)

```bash
# 2-rule composition (test only - not for training)
python gen_algebra_dataset.py --multirule --num_rules 2 --split test --size 10000

# 3-rule composition
python gen_algebra_dataset.py --multirule --num_rules 3 --split test --size 10000

# 4-rule composition
python gen_algebra_dataset.py --multirule --num_rules 4 --split test --size 10000
```

### Generate Constrained Dataset (for constraint evaluation)

```bash
# Positive constraint only
python gen_algebra_dataset.py --constrained --num_rules 2 --constraints positive --split test --size 5000

# Integer constraint only
python gen_algebra_dataset.py --constrained --num_rules 2 --constraints integer --split test --size 5000

# Both constraints
python gen_algebra_dataset.py --constrained --num_rules 2 --constraints positive integer --split test --size 5000
```

---

## Verification

### Verify Specific Dataset

```bash
# Verify distribute rule (train split, 1000 samples)
python verify.py --rule distribute --split train --num_problems 1000 --sample_size 1000

# Verify combine rule (full verification)
python verify.py --rule combine --split train --num_problems 50000

# Verify multi-rule dataset
python verify.py --multirule --num_rules 2 --num_problems 10000

# Verify constrained dataset
python verify.py --constrained --num_rules 2 --constraints positive --num_problems 5000
```

### Verify All Datasets

```bash
./verify_all_datasets.sh
```

### Verification Output

Verification checks:
- ✓ **Syntax validity**: Both input and target parse correctly
- ✓ **Mathematical equivalence**: Equations have same solution
- ✓ **Constraint satisfaction**: Solutions meet requirements (for constrained datasets)

Expected success rate: **≥95%** (script exits with error if <95%)

---

## Custom Parameters

### Change Coefficient Range

```bash
# Generate with larger coefficients (-50 to 50)
python gen_algebra_dataset.py --rule distribute --split train --size 50000 --coeff_range -50 50

# Generate with smaller coefficients (-5 to 5)
python gen_algebra_dataset.py --rule combine --split train --size 50000 --coeff_range -5 5
```

### Change Embedding Dimension

```bash
# Generate with d_model=256 (larger embeddings)
python gen_algebra_dataset.py --rule isolate --split train --size 50000 --d_model 256

# Generate with d_model=64 (smaller embeddings, faster training)
python gen_algebra_dataset.py --rule divide --split train --size 50000 --d_model 64
```

### Save Without Compression (for debugging)

```bash
python gen_algebra_dataset.py --rule distribute --split train --size 1000 --no-compress
```

---

## Dataset Files

Generated datasets are saved to: `./data/algebra/`

**Naming convention:**
- Single rule: `algebra_{rule}_{split}_{size}.pkl.gz`
- Multi-rule: `algebra_multirule_{num_rules}_{split}_{size}.pkl.gz`
- Constrained: `algebra_constrained_{num_rules}_{constraints}_{split}_{size}.pkl.gz`

**Examples:**
```
data/algebra/algebra_distribute_train_50000.pkl.gz
data/algebra/algebra_combine_test_10000.pkl.gz
data/algebra/algebra_multirule_2_test_10000.pkl.gz
data/algebra/algebra_constrained_2_positive_test_5000.pkl.gz
```

---

## Checking Dataset Files

```bash
# List all generated datasets
ls -lh data/algebra/

# Check total disk usage
du -sh data/algebra/

# Count number of datasets
ls data/algebra/ | wc -l
```

**Expected disk usage:**
- Single rule dataset (50k): ~5-10 MB compressed
- Total for all rules (12 datasets): ~100-150 MB

---

## Complete Workflow Example

```bash
# 1. Generate all datasets (takes ~10-15 minutes)
./generate_all_datasets.sh

# 2. Verify all datasets (takes ~5 minutes)
./verify_all_datasets.sh

# 3. Check what was created
ls -lh data/algebra/

# 4. Ready to train!
# (See training guide for next steps)
```

---

## Troubleshooting

### Generation is slow
- **Normal**: ~1000 problems/second on modern CPU
- **Expected time**: 50,000 problems ≈ 50 seconds
- If much slower: check CPU usage, close other programs

### Low success rate (<95%)
- Check coefficient range (very large ranges may cause overflow)
- Review failed problems with `--verbose` flag
- This is normal for complex multi-rule datasets (≥90% is acceptable)

### Out of memory
- Reduce dataset size: `--size 10000` for testing
- Reduce embedding dimension: `--d_model 64`
- Generate datasets one at a time instead of using `--all`

### Dataset file not found during training
- Make sure you're in the project root directory
- Check that `data/algebra/` exists
- Verify file permissions: `ls -la data/algebra/`

---

## Recommended Dataset Sizes for Different Scenarios

### Quick Prototyping (Fast iteration)
```bash
python gen_algebra_dataset.py --all --size 1000
# ~12 datasets × 1000 = 12,000 total problems
# Generation time: ~2 minutes
```

### Standard Training (Recommended)
```bash
./generate_all_datasets.sh  # Uses 50k/10k/10k split
# ~12 datasets × 70,000 = ~300,000 total problems
# Generation time: ~10-15 minutes
```

### Large-Scale Training (Maximum performance)
```bash
# Modify generate_all_datasets.sh:
TRAIN_SIZE=200000
TEST_SIZE=20000
VAL_SIZE=20000
# ~12 datasets × 240,000 = ~1,000,000 total problems
# Generation time: ~45-60 minutes
```

---

## Next Steps

After generating and verifying your datasets:

1. **Train single-rule EBMs** (see training documentation)
2. **Evaluate on multi-rule datasets** (compositional testing)
3. **Test constraint injection** (using constrained datasets)

For training instructions, see the main README or training guide.
