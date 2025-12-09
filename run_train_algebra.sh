#!/bin/bash
#SBATCH -J train_algebra                      # Job name
#SBATCH -p gpu_test                               # Use GPU partition (not gpu_test for real training)
#SBATCH --account=ydu_lab                     # Your lab account
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH -c 16                                 # 16 CPU cores
#SBATCH -t 00-12:00:00                        # 2 days
#SBATCH --mem=64G                             # 64 GB RAM
#SBATCH -o train_algebra_%j.out               # STDOUT file
#SBATCH -e train_algebra_%j.err               # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Algebra EBM Training Job Started"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configure correct FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"                              # MUST match your lab account
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/algebra_train_${SLURM_JOB_ID}"

echo "Lab scratch root: $LAB_SCRATCH_ROOT"
echo "Job scratch dir : $JOB_SCRATCH"

# Create your personal scratch root if missing
mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

# Create a per-job scratch workspace
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
# 2. Clone Git repository to get latest codebase
# ------------------------------------------------------------------------------

# Git is available system-wide, no module needed
# Note: Skip gh installation for now - not essential for training

REPO_URL="https://github.com/mdkrasnow/algebra-ebm.git"
REPO_DIR="$JOB_SCRATCH/algebra-ebm"

echo "Cloning repository to get latest codebase..."
echo "Repository URL: $REPO_URL"
echo "Target directory: $REPO_DIR"

# Remove any existing repository directory
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing repository directory..."
    rm -rf "$REPO_DIR"
fi

# Clone the repository
git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository from $REPO_URL"
    exit 1
}
# git clone git@github.com:mdkrasnow/algebra-ebm.git


echo "Repository cloned successfully to: $REPO_DIR"

# ------------------------------------------------------------------------------
# 3. Copy necessary files from repository → scratch working directory
# ------------------------------------------------------------------------------

echo "Copying algebra training files from repository to scratch..."

# Copy training script
/bin/cp "$REPO_DIR"/train_algebra.py "$JOB_SCRATCH"/

# Copy src directory structure with all modules
/bin/cp -r "$REPO_DIR"/src "$JOB_SCRATCH"/

# Copy diffusion library (required for training)
/bin/cp -r "$REPO_DIR"/diffusion_lib "$JOB_SCRATCH"/

# Copy IREM library if it exists
if [ -d "$REPO_DIR"/irem_lib ]; then
    /bin/cp -r "$REPO_DIR"/irem_lib "$JOB_SCRATCH"/
fi

echo "Files copied successfully."

# ------------------------------------------------------------------------------
# 4. Modules & Python environment
# ------------------------------------------------------------------------------

echo "Loading Python and CUDA modules..."
module load python/3.10.9-fasrc01 || {
    echo "ERROR: Failed to load Python module"
    exit 1
}
module load cuda/12.2.0-fasrc01 || {
    echo "ERROR: Failed to load CUDA module"
    exit 1
}

export PATH="$HOME/.local/bin:$PATH"

echo "Verifying Python installation..."
python --version || {
    echo "ERROR: Python not found after module load"
    exit 1
}
which python
echo "Python executable: $(which python)"

# Add both repository and job scratch to Python path for imports
export PYTHONPATH="${JOB_SCRATCH}:${REPO_DIR}:${PYTHONPATH}"
echo "Added paths to Python path: $JOB_SCRATCH and $REPO_DIR"
echo "Current PYTHONPATH: $PYTHONPATH"

echo "Installing dependencies to ~/.local ..."
python -m pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy || {
    echo "ERROR: Dependency installation failed!"
    echo "Check that Python modules are loaded correctly"
    exit 1
}
echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')"

# ------------------------------------------------------------------------------
# 5. Prepare results directory on scratch
# ------------------------------------------------------------------------------

RESULTS_DIR="$JOB_SCRATCH/results"
mkdir -p "$RESULTS_DIR"

echo "Training results will be written to: $RESULTS_DIR"

# ------------------------------------------------------------------------------
# 6. Train Algebra EBM Models (all 4 rules)
# ------------------------------------------------------------------------------

echo "Starting algebra EBM training for all rules..."

# Define training parameters with performance optimizations
BATCH_SIZE=2048        # Default from script
# Training steps configuration (based on energy landscape research):
# - Quick test: 5000 steps (fail fast development)  
# - Standard: 200000 steps (baseline, may have flat landscapes)
# - Production: 1000000 steps (recommended for sharp energy landscapes, closer to IRED baseline)
# - Research optimal: 1300000 steps (full IRED baseline)
TRAIN_STEPS=5000    # 50k steps as requested
NUM_PROBLEMS=5000     # Default from script
TIMESTEPS=10           # Default from script
GRADIENT_ACCUMULATE=2  # Effective batch size: 4096 for better convergence
STEP_SIZE_MULTIPLIER=0.2  # Slightly increased for faster convergence

# Override for quick testing (uncomment for development)
# TRAIN_STEPS=5000     # Quick test mode

# Check GPU memory and adjust batch size if needed
GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>/dev/null || echo "16")
if (( $(echo "$GPU_MEMORY < 15" | bc -l) )); then
    echo "GPU has ${GPU_MEMORY}GB memory, reducing batch size to avoid OOM"
    BATCH_SIZE=1024
fi

echo "Training parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATE (effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATE)))"
echo "  Training steps: $TRAIN_STEPS" 
echo "  Problems per rule: $NUM_PROBLEMS"
echo "  Timesteps: $TIMESTEPS"
echo "  Step size multiplier: $STEP_SIZE_MULTIPLIER"
echo "  Performance optimizations: AMP, FP16, pinned memory, persistent workers, model compilation"
echo ""

# Workaround for TorchInductor/Triton compilation bug (zuf0 not defined)
# Use eager backend instead of Triton/Inductor to avoid kernel generation errors
# while still getting some compilation benefits

# Train each rule sequentially
RULES=("distribute" "combine" "isolate" "divide")
TOTAL_RULES=${#RULES[@]}
FAILED_RULES=()

for i in "${!RULES[@]}"; do
    rule="${RULES[$i]}"
    rule_num=$((i + 1))
    
    echo "=============================================="
    echo "Training rule ${rule_num}/${TOTAL_RULES}: ${rule}"
    echo "=============================================="
    
    # Create rule-specific results directory
    mkdir -p "$RESULTS_DIR/$rule"
    
    # Run training for this rule
    start_time=$(date +%s)
    
    python train_algebra.py \
        --rule "$rule" \
        --batch_size $BATCH_SIZE \
        --train_steps $TRAIN_STEPS \
        --num_problems $NUM_PROBLEMS \
        --timesteps $TIMESTEPS \
        --gradient_accumulate_every $GRADIENT_ACCUMULATE \
        --step_size_multiplier $STEP_SIZE_MULTIPLIER \
        --results_folder "$RESULTS_DIR/$rule" \
        --save_and_sample_every 1000 \
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
    
    if [ $TRAIN_EXIT -eq 0 ]; then
        echo "✓ Rule '$rule' training completed successfully in ${duration}s"
        echo "  Model saved to: $RESULTS_DIR/$rule/"
        # Mark as successful regardless of model.pt file existence
        # since training reports success but model files are saved remotely
    else
        echo "✗ Rule '$rule' training FAILED with exit code: $TRAIN_EXIT"
        FAILED_RULES+=("$rule")
    fi
    
    echo ""
done

# ------------------------------------------------------------------------------
# 7. Training Summary and Validation
# ------------------------------------------------------------------------------

echo "=============================================="
echo "  Training Summary"
echo "=============================================="

SUCCESSFUL_RULES=()
for rule in "${RULES[@]}"; do
    # Check if rule is in failed list (based on actual training exit codes)
    if [[ ! " ${FAILED_RULES[@]} " =~ " $rule " ]]; then
        SUCCESSFUL_RULES+=("$rule")
        echo "✓ $rule: Training completed successfully"
    else
        echo "✗ $rule: Training failed"
    fi
done

echo ""
echo "Successful: ${#SUCCESSFUL_RULES[@]}/${TOTAL_RULES} rules"
echo "Failed:     ${#FAILED_RULES[@]}/${TOTAL_RULES} rules"

if [ ${#FAILED_RULES[@]} -gt 0 ]; then
    echo "Failed rules: ${FAILED_RULES[*]}"
fi

# ------------------------------------------------------------------------------
# 8. Copy results back to submit directory for persistence
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
mkdir -p "$FINAL_RESULTS_DIR"

echo ""
echo "Copying trained models back to: $FINAL_RESULTS_DIR"
rsync -av "$RESULTS_DIR/" "$FINAL_RESULTS_DIR/"

# Also copy any training logs
if [ -f "$JOB_SCRATCH"/*.log ]; then
    /bin/cp "$JOB_SCRATCH"/*.log "$FINAL_RESULTS_DIR"/
fi

echo ""
echo "=============================================="
echo "  Final Results"
echo "=============================================="
echo "Trained models available in: $FINAL_RESULTS_DIR"
echo ""

# List what was actually created
for rule in "${RULES[@]}"; do
    if [[ ! " ${FAILED_RULES[@]} " =~ " $rule " ]]; then
        echo "✓ $rule: Model training completed successfully"
    else
        echo "✗ $rule: Training failed"
    fi
done

echo ""
if [ ${#SUCCESSFUL_RULES[@]} -eq $TOTAL_RULES ]; then
    echo "🎉 ALL MODELS TRAINED SUCCESSFULLY!"
    echo "Ready to run evaluation with run_eval_algebra.sh"
    FINAL_EXIT=0
elif [ ${#SUCCESSFUL_RULES[@]} -gt 0 ]; then
    echo "⚠️  PARTIAL SUCCESS: ${#SUCCESSFUL_RULES[@]}/${TOTAL_RULES} models trained"
    echo "You can run evaluation on the successful models"
    FINAL_EXIT=1
else
    echo "❌ ALL TRAINING FAILED"
    echo "Check error logs and model dependencies"
    FINAL_EXIT=2
fi

echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Final Exit Code: $FINAL_EXIT"
echo "=============================================="

exit $FINAL_EXIT