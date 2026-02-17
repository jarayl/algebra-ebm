#!/bin/bash
#
# Diagnostic Experiment: Train distribute model without encoder normalization
#
# Purpose: Verify that encoder normalization is the root cause of energy landscape failure
# Expected: Energy landscape correctness improves from 54% to >80%
#
# Analysis: documentation/deep-dive-analysis.md
# Summary: documentation/CRITICAL-FINDINGS.md

set -e  # Exit on error

# Configuration
RULE="distribute"
OUTPUT_DIR="results/diagnostic_no_norm"
TRAIN_STEPS=10000
BATCH_SIZE=2048
TIMESTEPS=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "DIAGNOSTIC EXPERIMENT: Training Without Encoder Normalization"
echo "========================================================================"
echo "Rule: $RULE"
echo "Output: $OUTPUT_DIR"
echo "Train Steps: $TRAIN_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Timesteps: $TIMESTEPS"
echo ""
echo "MODIFICATION APPLIED:"
echo "  src/algebra/algebra_encoder.py line 135"
echo "  Changed: if self.normalize_embeddings:"
echo "  To:      if False:  # DISABLED FOR DIAGNOSTIC"
echo ""
echo "HYPOTHESIS:"
echo "  Encoder normalization forces ||embedding|| = 1.0 (unit sphere)"
echo "  Energy E = scale * ||output||^2 + bias cannot discriminate"
echo "  Result: Random energy landscapes (54% correct, 46% inverted)"
echo ""
echo "EXPECTED OUTCOME:"
echo "  - Energy scale parameters learn meaningful values (not stuck at 1.0)"
echo "  - Energy landscapes improve from 54% to >80% correct"
echo "  - Single-rule accuracy improves from 6% to >30%"
echo "========================================================================"
echo ""

# Log the modification status
echo "Checking encoder normalization status..."
if grep -q "if False:.*DISABLED FOR DIAGNOSTIC" src/algebra/algebra_encoder.py; then
    echo "✓ Normalization is DISABLED (diagnostic mode)"
else
    echo "✗ WARNING: Normalization may still be ENABLED"
    echo "  Please verify src/algebra/algebra_encoder.py line 135"
    echo "  Expected: if False:  # DISABLED FOR DIAGNOSTIC"
    exit 1
fi

echo ""
echo "Starting training at $(date)"
echo ""

# Train the model
python train_algebra.py \
    --rule "$RULE" \
    --train_steps "$TRAIN_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --timesteps "$TIMESTEPS" \
    --output_dir "$OUTPUT_DIR" \
    --supervise-energy-landscape True \
    --use-contrastive-energy-loss True \
    --use-innerloop-opt True \
    --step_size_multiplier 0.1 \
    --save-interval 2000 \
    --validation-interval 1000 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training completed at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check training logs for energy scale parameter values"
echo "     grep 'energy_scale' $OUTPUT_DIR/training.log"
echo ""
echo "  2. Run diagnostic evaluation:"
echo "     bash scripts/diagnostic_no_norm_eval.sh"
echo ""
echo "  3. Compare energy landscape quality:"
echo "     Expected: >80% correct (vs baseline 54%)"
echo ""
