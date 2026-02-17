#!/bin/bash
#SBATCH --job-name=alg_diag_train
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@fas.harvard.edu

# Diagnostic Training: distribute model without encoder normalization
# Purpose: Verify that encoder normalization is the root cause (Issue #1)
# Expected: Energy landscapes improve from 54% to >80% correct

echo "========================================================================"
echo "DIAGNOSTIC TRAINING: Encoder Normalization Fix"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "HYPOTHESIS:"
echo "  Encoder normalization (||embedding|| = 1.0) breaks energy learning"
echo "  Energy function E = scale * ||output||^2 + bias cannot discriminate"
echo ""
echo "MODIFICATION:"
echo "  src/algebra/algebra_encoder.py line 135"
echo "  Normalization DISABLED for this diagnostic"
echo ""
echo "EXPECTED OUTCOME:"
echo "  - Energy scale parameters learn meaningful values (not stuck at 1.0)"
echo "  - Energy landscapes improve from 54% to >80% correct"
echo "  - Single-rule accuracy improves from 6% to >30%"
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
git clone /n/home03/mkrasnow/research-repo .
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
    echo "✗ ERROR: Encoder normalization may still be ENABLED"
    echo "  Expected modification not found in src/algebra/algebra_encoder.py"
    exit 1
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

# Training configuration
RULE="distribute"
OUTPUT_DIR="/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm"
TRAIN_STEPS=10000
BATCH_SIZE=2048
TIMESTEPS=10

# Create output directory on persistent storage
mkdir -p "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "TRAINING CONFIGURATION"
echo "========================================================================"
echo "Rule: $RULE"
echo "Output Directory: $OUTPUT_DIR"
echo "Train Steps: $TRAIN_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Timesteps: $TIMESTEPS"
echo "========================================================================"
echo ""

# Run training
echo "Starting training at $(date)"
echo ""

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
    --validation-interval 1000

TRAIN_EXIT=$?

echo ""
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training successful"

    # Check if model was saved
    if [ -f "$OUTPUT_DIR/$RULE/model.pt" ]; then
        echo "✓ Model saved to: $OUTPUT_DIR/$RULE/model.pt"
        MODEL_SIZE=$(du -h "$OUTPUT_DIR/$RULE/model.pt" | cut -f1)
        echo "  Model size: $MODEL_SIZE"
    else
        echo "✗ WARNING: Model file not found at expected location"
    fi

    # Check for energy scale parameter values in logs
    echo ""
    echo "Checking energy scale parameter learning..."
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo "Final energy scale values:"
        grep -i "energy_scale" "$OUTPUT_DIR/training.log" | tail -5 || echo "  (logging not available)"
    fi
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
fi

# Cleanup temporary directory
echo ""
echo "Cleaning up temporary workspace..."
cd /
rm -rf "$WORK_DIR"

echo ""
echo "========================================================================"
echo "DIAGNOSTIC TRAINING COMPLETE"
echo "========================================================================"
echo "Next Step: Submit diagnostic evaluation job"
echo "  sbatch slurm/eval_diagnostic_no_norm.sbatch"
echo ""
echo "Expected: Evaluation will test energy landscape quality"
echo "  Target: >80% correct (vs baseline 54%)"
echo "========================================================================"

exit $TRAIN_EXIT
