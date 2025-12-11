#!/bin/bash
#SBATCH -J train_monolithic                   # Job name
#SBATCH -p gpu_test                           # Use GPU partition (change to 'gpu' for real training)
#SBATCH --account=ydu_lab                     # Your lab account
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH -c 16                                 # 16 CPU cores
#SBATCH -t 00-10:00:00                        # 2 hours (adjust for longer training)
#SBATCH --mem=64G                             # 64 GB RAM
#SBATCH -o train_monolithic_%j.out            # STDOUT file
#SBATCH -e train_monolithic_%j.err            # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Monolithic Algebra EBM Training Started"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configure FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/monolithic_train_${SLURM_JOB_ID}"

echo "Lab scratch root: $LAB_SCRATCH_ROOT"
echo "Job scratch dir : $JOB_SCRATCH"

mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

mkdir -p "$JOB_SCRATCH" || {
    echo "ERROR: Cannot create $JOB_SCRATCH"
    exit 1
}

cd "$JOB_SCRATCH" || {
    echo "ERROR: cd to JOB_SCRATCH failed"
    exit 1
}

echo "Now working in scratch: $(pwd)"

# ------------------------------------------------------------------------------
# 2. Clone Git repository
# ------------------------------------------------------------------------------

REPO_URL="https://github.com/mdkrasnow/algebra-ebm.git"
REPO_DIR="$JOB_SCRATCH/algebra-ebm"

echo "Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    rm -rf "$REPO_DIR"
fi

git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository"
    exit 1
}

echo "Repository cloned to: $REPO_DIR"

# ------------------------------------------------------------------------------
# 3. Copy files to scratch
# ------------------------------------------------------------------------------

echo "Copying monolithic training files..."

# Copy training script
/bin/cp "$REPO_DIR"/train_algebra_monolithic.py "$JOB_SCRATCH"/

# Copy entire src directory for proper imports
/bin/cp -r "$REPO_DIR"/src "$JOB_SCRATCH"/

echo "Files copied successfully."

# ------------------------------------------------------------------------------
# 4. Python environment setup
# ------------------------------------------------------------------------------

echo "Loading modules..."
module load python/3.10.9-fasrc01 || {
    echo "ERROR: Failed to load Python module"
    exit 1
}
module load cuda/12.2.0-fasrc01 || {
    echo "ERROR: Failed to load CUDA module"
    exit 1
}

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${JOB_SCRATCH}:${PYTHONPATH}"

echo "Installing dependencies..."
python -m pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy || {
    echo "ERROR: Dependency installation failed"
    exit 1
}

echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')"

# ------------------------------------------------------------------------------
# 5. Training configuration
# ------------------------------------------------------------------------------

# Training parameters for quick testing (adjust for full training)
TRAIN_STEPS=50000         # 50k steps as requested
PROBLEMS_PER_RULE=200000   # Full dataset as requested
BATCH_SIZE=2048           # Conservative for memory
TIMESTEPS=10
GRADIENT_ACCUMULATE=2

# Check GPU memory and adjust if needed
GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>/dev/null || echo "16")
if (( $(echo "$GPU_MEMORY < 15" | bc -l) )); then
    echo "GPU has ${GPU_MEMORY}GB memory, reducing batch size"
    BATCH_SIZE=512
fi

echo "Training parameters:"
echo "  Train steps: $TRAIN_STEPS"
echo "  Problems per rule: $PROBLEMS_PER_RULE"
echo "  Batch size: $BATCH_SIZE"
echo "  Effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATE))"

# ------------------------------------------------------------------------------
# 6. Run monolithic training
# ------------------------------------------------------------------------------

RESULTS_DIR="$JOB_SCRATCH/results/monolithic"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Starting Monolithic Training"
echo "=============================================="

start_time=$(date +%s)

python train_algebra_monolithic.py \
    --train_steps $TRAIN_STEPS \
    --problems_per_rule $PROBLEMS_PER_RULE \
    --batch_size $BATCH_SIZE \
    --timesteps $TIMESTEPS \
    --gradient_accumulate_every $GRADIENT_ACCUMULATE \
    --results_folder "$RESULTS_DIR" \
    --save_and_sample_every 500 \
    --supervise-energy-landscape True \
    --use-contrastive-energy-loss True \
    --use-innerloop-opt True \
    --amp True \
    --fp16 True \
    --pin_memory True \
    --persistent_workers True \
    --compile_model True \
    --compile_backend eager

TRAIN_EXIT=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# ------------------------------------------------------------------------------
# 7. Training results
# ------------------------------------------------------------------------------

echo "=============================================="
echo "  Training Summary"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Monolithic training completed successfully in ${duration}s"
    
    if [ -f "$RESULTS_DIR/model.pt" ]; then
        model_size=$(du -h "$RESULTS_DIR/model.pt" | cut -f1)
        echo "✓ Model saved: $model_size"
    else
        echo "✗ Warning: model.pt not found"
        TRAIN_EXIT=1
    fi
else
    echo "✗ Monolithic training FAILED with exit code: $TRAIN_EXIT"
fi

# ------------------------------------------------------------------------------
# 8. Copy results back
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
mkdir -p "$FINAL_RESULTS_DIR"

echo ""
echo "Copying results to: $FINAL_RESULTS_DIR"
rsync -av "$RESULTS_DIR/" "$FINAL_RESULTS_DIR/monolithic/"

# Copy logs
if [ -f "$JOB_SCRATCH"/*.log ]; then
    /bin/cp "$JOB_SCRATCH"/*.log "$FINAL_RESULTS_DIR"/
fi

echo ""
echo "=============================================="
echo "  Final Results"
echo "=============================================="

if [ -f "$FINAL_RESULTS_DIR/monolithic/model.pt" ]; then
    model_size=$(du -h "$FINAL_RESULTS_DIR/monolithic/model.pt" | cut -f1)
    echo "✓ Monolithic model saved: $model_size"
    echo "✓ Ready for comparison evaluation!"
    echo ""
    echo "Next steps:"
    echo "1. Run comparison evaluation with:"
    echo "   sbatch run_comparison_eval.sh"
else
    echo "✗ No monolithic model found"
    TRAIN_EXIT=1
fi

echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Exit Code: $TRAIN_EXIT"
echo "=============================================="

exit $TRAIN_EXIT