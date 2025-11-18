#!/bin/bash
# Generate all algebra datasets for EBM training
# This script generates datasets for all rules and splits

set -e  # Exit on error

echo "========================================="
echo "ALGEBRA DATASET GENERATION"
echo "========================================="
echo ""
echo "This will generate datasets for:"
echo "  - 4 rules: distribute, combine, isolate, divide"
echo "  - 3 splits per rule: train, test, val"
echo "  - Total: 12 single-rule datasets"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create data directory
mkdir -p data/algebra

# Dataset sizes (optimized for training)
TRAIN_SIZE=50000
TEST_SIZE=10000
VAL_SIZE=10000

echo ""
echo "========================================="
echo "Generating Single-Rule Datasets"
echo "========================================="

# Generate all single-rule datasets
RULES=("distribute" "combine" "isolate" "divide")
SPLITS=("train" "test" "val")

for RULE in "${RULES[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        # Determine size based on split
        if [ "$SPLIT" = "train" ]; then
            SIZE=$TRAIN_SIZE
        elif [ "$SPLIT" = "test" ]; then
            SIZE=$TEST_SIZE
        else
            SIZE=$VAL_SIZE
        fi

        echo ""
        echo "Generating: $RULE ($SPLIT) - $SIZE problems"
        python gen_algebra_dataset.py \
            --rule $RULE \
            --split $SPLIT \
            --size $SIZE \
            --d_model 128 \
            --compress

        echo "✓ Complete: $RULE ($SPLIT)"
    done
done

echo ""
echo "========================================="
echo "All Datasets Generated!"
echo "========================================="
echo ""
echo "Generated datasets:"
ls -lh data/algebra/
echo ""
echo "Total disk usage:"
du -sh data/algebra/
echo ""
echo "Next steps:"
echo "  1. Verify datasets: ./verify_all_datasets.sh"
echo "  2. Start training: python train.py --dataset algebra --rule distribute"
echo ""