#!/bin/bash
#SBATCH --job-name=alg_diag_eval
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_eval_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@fas.harvard.edu

# Diagnostic Evaluation: Test model trained without encoder normalization
# Purpose: Measure energy landscape quality improvement
# Baseline: 54% correct, Target: >80% correct

echo "========================================================================"
echo "DIAGNOSTIC EVALUATION: Energy Landscape Quality Test"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "BASELINE METRICS (with normalization):"
echo "  - Energy landscape correctness: 54%"
echo "  - Single-rule accuracy: 6.3%"
echo ""
echo "TARGET METRICS (without normalization):"
echo "  - Energy landscape correctness: >80%"
echo "  - Single-rule accuracy: >30%"
echo "========================================================================"
echo ""

# Load required modules
module load cuda/11.8.0-fasrc01
module load python/3.10.12-fasrc01

# Verify GPU is available
echo "Checking GPU availability..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: GPU not available"
    exit 1
fi
echo ""

# Clone repository to temporary directory
WORK_DIR="/tmp/algebra-ebm-job-${SLURM_JOB_ID}"
echo "Setting up workspace: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone the repo
GIT_SHA="${GIT_SHA:-HEAD}"
echo "Cloning repository at commit $GIT_SHA..."
git clone "$GIT_URL" .
git checkout "$GIT_SHA"
git submodule update --init --recursive

# Navigate to project directory
cd projects/algebra-ebm

# Verify encoder normalization is disabled
echo ""
echo "Verifying encoder modification..."
if grep -q "if False:.*DISABLED FOR DIAGNOSTIC" src/algebra/algebra_encoder.py; then
    echo "✓ Encoder normalization is DISABLED (diagnostic mode)"
else
    echo "✗ WARNING: Encoder normalization may be ENABLED"
    echo "  Results may not reflect the fix being tested"
fi

# Create Python virtual environment
echo ""
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Evaluation configuration
MODEL_DIR="/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm"
OUTPUT_DIR="/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm/evaluation"
NUM_PROBLEMS=100
RULE="distribute"

# Create output directory on persistent storage
mkdir -p "$OUTPUT_DIR"

# Check if model exists
echo ""
echo "Checking for trained model..."
if [ ! -f "$MODEL_DIR/$RULE/model.pt" ]; then
    echo "✗ ERROR: Model not found at $MODEL_DIR/$RULE/model.pt"
    echo "  Please ensure training job completed successfully"
    exit 1
fi
echo "✓ Model found at $MODEL_DIR/$RULE/model.pt"

echo ""
echo "========================================================================"
echo "EVALUATION CONFIGURATION"
echo "========================================================================"
echo "Model Directory: $MODEL_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Problems: $NUM_PROBLEMS (sample for quick feedback)"
echo "Rule: $RULE"
echo "========================================================================"
echo ""

# Run evaluation with diagnostic logging
echo "Starting evaluation at $(date)"
echo ""

python eval_algebra.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type single_rule \
    --rule "$RULE" \
    --single_rule_problems "$NUM_PROBLEMS" \
    --seed 42 \
    --enable_diagnostics

EVAL_EXIT=$?

echo ""
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT"

if [ $EVAL_EXIT -eq 0 ]; then
    echo "✓ Evaluation successful"

    # Extract accuracy from logs
    echo ""
    echo "Accuracy Results:"
    grep "Accuracy" "$OUTPUT_DIR/evaluation.log" 2>/dev/null | head -5 || echo "  (logs not available)"

    # Check for diagnostic data
    echo ""
    if [ -d "$OUTPUT_DIR/diagnostics" ]; then
        NUM_TRAJECTORIES=$(find "$OUTPUT_DIR/diagnostics" -name "problem_*_trajectory.json" 2>/dev/null | wc -l)
        echo "✓ Diagnostic data generated: $NUM_TRAJECTORIES trajectory files"
        echo "  Location: $OUTPUT_DIR/diagnostics/"
    else
        echo "✗ WARNING: No diagnostic data found"
    fi
else
    echo "✗ Evaluation failed with exit code $EVAL_EXIT"
fi

# Cleanup temporary directory
echo ""
echo "Cleaning up temporary workspace..."
cd /
rm -rf "$WORK_DIR"

echo ""
echo "========================================================================"
echo "DIAGNOSTIC EVALUATION COMPLETE"
echo "========================================================================"
if [ $EVAL_EXIT -eq 0 ]; then
    echo "Next Step: Analyze energy landscape quality"
    echo "  Command: python scripts/analyze_energy_landscapes.py $OUTPUT_DIR"
    echo ""
    echo "Decision Criteria:"
    echo "  - IF >80% correct: ROOT CAUSE CONFIRMED → Full retraining"
    echo "  - IF 60-80% correct: PARTIAL FIX → Investigate Issue #2"
    echo "  - IF <60% correct: HYPOTHESIS REJECTED → Alternative root causes"
fi
echo "========================================================================"

exit $EVAL_EXIT
