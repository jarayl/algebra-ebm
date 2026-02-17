#!/bin/bash
#
# Diagnostic Evaluation: Test model trained without encoder normalization
#
# Purpose: Measure energy landscape quality and compare to baseline
# Baseline: 54% correct energy landscapes, 6.3% single-rule accuracy
# Target: >80% correct energy landscapes, >30% single-rule accuracy

set -e  # Exit on error

# Configuration
MODEL_DIR="results/diagnostic_no_norm"
OUTPUT_DIR="results/diagnostic_no_norm/evaluation"
NUM_PROBLEMS=100  # Small sample for quick iteration
RULE="distribute"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "DIAGNOSTIC EVALUATION: Energy Landscape Quality Test"
echo "========================================================================"
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "Problems: $NUM_PROBLEMS (sample for quick feedback)"
echo "Rule: $RULE"
echo ""
echo "BASELINE METRICS (with normalization enabled):"
echo "  - Energy landscape correctness: 54% (essentially random 50/50)"
echo "  - Single-rule accuracy: 6.3%"
echo "  - Multi-rule accuracy: 0%"
echo ""
echo "TARGET METRICS (with normalization disabled):"
echo "  - Energy landscape correctness: >80%"
echo "  - Single-rule accuracy: >30%"
echo ""
echo "ANALYSIS FOCUS:"
echo "  - Count problems where E(inp→target) < E(inp→inp) [CORRECT]"
echo "  - Count problems where E(inp→target) > E(inp→inp) [INVERTED]"
echo "  - Calculate percentage correct"
echo "========================================================================"
echo ""

# Check if model exists
if [ ! -f "$MODEL_DIR/$RULE/model.pt" ]; then
    echo "✗ ERROR: Model not found at $MODEL_DIR/$RULE/model.pt"
    echo "  Please run training first: bash scripts/diagnostic_no_norm_train.sh"
    exit 1
fi

echo "✓ Model found at $MODEL_DIR/$RULE/model.pt"
echo ""

# Check encoder normalization status
echo "Verifying encoder normalization is disabled..."
if grep -q "if False:.*DISABLED FOR DIAGNOSTIC" src/algebra/algebra_encoder.py; then
    echo "✓ Normalization is DISABLED (diagnostic mode)"
else
    echo "✗ WARNING: Normalization may be ENABLED"
    echo "  Evaluation will use normalized embeddings (not testing the fix)"
    echo "  Please verify src/algebra/algebra_encoder.py line 135"
fi

echo ""
echo "Starting evaluation at $(date)"
echo ""

# Run evaluation with diagnostic logging
python eval_algebra.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type single_rule \
    --rule "$RULE" \
    --single_rule_problems "$NUM_PROBLEMS" \
    --seed 42 \
    --enable_diagnostics \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

echo ""
echo "Evaluation completed at $(date)"
echo "Results: $OUTPUT_DIR"
echo ""

# Analyze results
echo "========================================================================"
echo "RESULTS ANALYSIS"
echo "========================================================================"

# Extract accuracy if available in logs
if grep -q "Accuracy" "$OUTPUT_DIR/evaluation.log"; then
    echo "Accuracy Results:"
    grep "Accuracy" "$OUTPUT_DIR/evaluation.log" | head -5
    echo ""
fi

# Check for energy landscape statistics
if [ -d "$OUTPUT_DIR/diagnostics" ]; then
    echo "Diagnostic data available at: $OUTPUT_DIR/diagnostics"
    echo "Analyzing energy landscape quality..."

    # Count trajectory files
    NUM_TRAJECTORIES=$(find "$OUTPUT_DIR/diagnostics" -name "problem_*_trajectory.json" | wc -l)
    echo "  Found $NUM_TRAJECTORIES trajectory files"

    # TODO: Add Python script to analyze energy landscape correctness
    # For now, manual inspection required
    echo ""
    echo "Manual analysis required:"
    echo "  1. Check if energy trajectories show decreasing energy"
    echo "  2. Compare final energies: E(inp→target) vs E(inp→inp)"
    echo "  3. Calculate percentage where E(inp→target) < E(inp→inp)"
else
    echo "No diagnostic data found. Re-run with --enable_diagnostics"
fi

echo ""
echo "========================================================================"
echo "DECISION CRITERIA"
echo "========================================================================"
echo ""
echo "IF energy landscape correctness >80%:"
echo "  → ROOT CAUSE CONFIRMED: Normalization was breaking energy learning"
echo "  → NEXT STEP: Full retraining (T0b) - retrain all 5 models"
echo "  → EXPECTED: Single-rule 50-85%, Multi-rule 10-30%"
echo ""
echo "IF energy landscape correctness 60-80%:"
echo "  → PARTIAL FIX: Normalization contributed but not sole cause"
echo "  → NEXT STEP: Investigate Issue #2 (energy scale parameter learning)"
echo "  → ACTION: Add gradient logging, verify scale/bias are updating"
echo ""
echo "IF energy landscape correctness <60%:"
echo "  → ROOT CAUSE INCORRECT: Normalization not the main issue"
echo "  → NEXT STEP: Investigate alternative hypotheses"
echo "  → ACTION: Check Issue #2 (parameter optimization) and #3 (iterations)"
echo ""
echo "To proceed with full retraining:"
echo "  bash scripts/full_retrain_no_norm.sh"
echo ""
