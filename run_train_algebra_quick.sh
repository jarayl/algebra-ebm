#!/bin/bash
#SBATCH -J train_algebra_quick               # Job name
#SBATCH -p gpu_test                          # Use GPU partition
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-02:00:00                       # 2 hours
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o train_algebra_quick_%j.out        # STDOUT file
#SBATCH -e train_algebra_quick_%j.err        # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Quick Algebra EBM Training (Testing Mode)"
echo "=============================================="
echo "Date:          $(date)"
echo "Job ID:        $SLURM_JOB_ID"
echo "=============================================="

# Use the existing training script but with minimal parameters
# This is a modified version of run_train_algebra.sh with quick settings

# ------------------------------------------------------------------------------
# Setup (same as original script)
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/algebra_train_quick_${SLURM_JOB_ID}"

mkdir -p "$LAB_SCRATCH_ROOT" && mkdir -p "$JOB_SCRATCH" && cd "$JOB_SCRATCH"

# Clone repository
REPO_URL="https://github.com/mdkrasnow/algebra-ebm.git"
REPO_DIR="$JOB_SCRATCH/algebra-ebm"

git clone "$REPO_URL" "$REPO_DIR"

# Copy files
/bin/cp "$REPO_DIR"/train_algebra.py "$JOB_SCRATCH"/
/bin/cp -r "$REPO_DIR"/src "$JOB_SCRATCH"/

# Setup Python
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${JOB_SCRATCH}:${PYTHONPATH}"

python -m pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy

# ------------------------------------------------------------------------------
# Quick Training Parameters (MINIMAL FOR TESTING)
# ------------------------------------------------------------------------------

BATCH_SIZE=512           # Smaller for safety
TRAIN_STEPS=1000         # Very quick test
NUM_PROBLEMS=1000        # Small dataset
TIMESTEPS=10
GRADIENT_ACCUMULATE=2

echo "QUICK TRAINING PARAMETERS:"
echo "  Batch size: $BATCH_SIZE"
echo "  Training steps: $TRAIN_STEPS (very quick test)"
echo "  Problems per rule: $NUM_PROBLEMS"
echo "  Total time estimate: ~5-10 minutes per rule"

# ------------------------------------------------------------------------------
# Train all 4 rules with quick settings
# ------------------------------------------------------------------------------

RESULTS_DIR="$JOB_SCRATCH/results"
mkdir -p "$RESULTS_DIR"

RULES=("distribute" "combine" "isolate" "divide")
SUCCESSFUL_RULES=()
FAILED_RULES=()

for rule in "${RULES[@]}"; do
    echo "=============================================="
    echo "Training rule: $rule (quick mode)"
    echo "=============================================="
    
    mkdir -p "$RESULTS_DIR/$rule"
    
    start_time=$(date +%s)
    
    python train_algebra.py \
        --rule "$rule" \
        --batch_size $BATCH_SIZE \
        --train_steps $TRAIN_STEPS \
        --num_problems $NUM_PROBLEMS \
        --timesteps $TIMESTEPS \
        --gradient_accumulate_every $GRADIENT_ACCUMULATE \
        --results_folder "$RESULTS_DIR/$rule" \
        --save_and_sample_every 250 \
        --supervise-energy-landscape True \
        --use-contrastive-energy-loss True \
        --use-innerloop-opt True \
        --amp True \
        --fp16 True \
        --compile_model True \
        --compile_backend eager
    
    TRAIN_EXIT=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $TRAIN_EXIT -eq 0 ] && [ -f "$RESULTS_DIR/$rule/model.pt" ]; then
        SUCCESSFUL_RULES+=("$rule")
        model_size=$(du -h "$RESULTS_DIR/$rule/model.pt" | cut -f1)
        echo "✓ $rule completed in ${duration}s, model: $model_size"
    else
        FAILED_RULES+=("$rule")
        echo "✗ $rule failed"
    fi
done

# ------------------------------------------------------------------------------
# Copy results back and summary
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
mkdir -p "$FINAL_RESULTS_DIR"

rsync -av "$RESULTS_DIR/" "$FINAL_RESULTS_DIR/"

echo ""
echo "=============================================="
echo "  QUICK TRAINING SUMMARY"
echo "=============================================="
echo "Successful: ${#SUCCESSFUL_RULES[@]}/4 rules"
echo "Results in: $FINAL_RESULTS_DIR"

for rule in "${RULES[@]}"; do
    if [ -f "$FINAL_RESULTS_DIR/$rule/model.pt" ]; then
        echo "✓ $rule: Ready for evaluation"
    else
        echo "✗ $rule: Failed"
    fi
done

if [ ${#SUCCESSFUL_RULES[@]} -eq 4 ]; then
    echo ""
    echo "🎉 ALL QUICK MODELS READY!"
    echo "Next steps:"
    echo "1. Train monolithic: sbatch run_train_monolithic.sh"
    echo "2. Run comparison:    sbatch run_comparison_eval.sh"
    exit 0
else
    echo ""
    echo "⚠️  Some models failed - check logs"
    exit 1
fi