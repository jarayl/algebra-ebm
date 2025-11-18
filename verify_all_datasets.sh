#!/bin/bash
# Verify all generated algebra datasets
# This script runs verification on all datasets to ensure quality

set -e  # Exit on error

echo "========================================="
echo "ALGEBRA DATASET VERIFICATION"
echo "========================================="
echo ""
echo "This will verify all generated datasets for:"
echo "  - Syntax correctness"
echo "  - Mathematical equivalence"
echo "  - Solution accuracy"
echo ""

# Verification parameters
SAMPLE_SIZE=1000  # Verify 1000 problems per dataset (fast verification)

echo "Verification sample size: $SAMPLE_SIZE problems per dataset"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "========================================="
echo "Verifying Single-Rule Datasets"
echo "========================================="

# Verify all single-rule datasets
RULES=("distribute" "combine" "isolate" "divide")
SPLITS=("train" "test" "val")

# Summary arrays
declare -a PASSED=()
declare -a FAILED=()

for RULE in "${RULES[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        echo ""
        echo "---------------------------------------"
        echo "Verifying: $RULE ($SPLIT)"
        echo "---------------------------------------"

        # Run verification
        if python verify.py \
            --rule $RULE \
            --split $SPLIT \
            --num_problems $SAMPLE_SIZE \
            --sample_size $SAMPLE_SIZE \
            --show_failures; then
            PASSED+=("$RULE-$SPLIT")
            echo "✓ PASSED: $RULE ($SPLIT)"
        else
            FAILED+=("$RULE-$SPLIT")
            echo "✗ FAILED: $RULE ($SPLIT)"
        fi
    done
done

echo ""
echo "========================================="
echo "VERIFICATION SUMMARY"
echo "========================================="
echo ""
echo "Passed: ${#PASSED[@]}"
for dataset in "${PASSED[@]}"; do
    echo "  ✓ $dataset"
done
echo ""
echo "Failed: ${#FAILED[@]}"
for dataset in "${FAILED[@]}"; do
    echo "  ✗ $dataset"
done
echo ""

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "⚠ WARNING: Some datasets failed verification!"
    echo "Review the output above for details."
    exit 1
else
    echo "✓ All datasets passed verification!"
    echo ""
    echo "Datasets are ready for training."
    exit 0
fi
