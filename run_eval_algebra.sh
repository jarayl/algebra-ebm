#!/bin/bash
#SBATCH -J eval_algebra                      # Job name
#SBATCH -p gpu_test                          # Partition (use gpu for real runs)
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-04:00:00                       # 4 hours
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o eval_algebra_%j.out               # STDOUT file
#SBATCH -e eval_algebra_%j.err               # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Algebra EBM Evaluation Job Started"
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
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/algebra_eval_${SLURM_JOB_ID}"

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
# 2. Copy necessary files from home → scratch
# ------------------------------------------------------------------------------

echo "Copying algebra evaluation files to scratch..."

# Copy all algebra-related Python files
/bin/cp "$SLURM_SUBMIT_DIR"/eval_algebra.py "$JOB_SCRATCH"/
/bin/cp "$SLURM_SUBMIT_DIR"/algebra_*.py "$JOB_SCRATCH"/

# Copy core infrastructure files  
/bin/cp "$SLURM_SUBMIT_DIR"/dataset.py "$JOB_SCRATCH"/
/bin/cp "$SLURM_SUBMIT_DIR"/models.py "$JOB_SCRATCH"/

# Copy diffusion library (required for model loading)
/bin/cp -r "$SLURM_SUBMIT_DIR"/diffusion_lib "$JOB_SCRATCH"/

# Copy IREM library if it exists
if [ -d "$SLURM_SUBMIT_DIR"/irem_lib ]; then
    /bin/cp -r "$SLURM_SUBMIT_DIR"/irem_lib "$JOB_SCRATCH"/
fi

# Copy trained models from results directory
if [ -d "$SLURM_SUBMIT_DIR"/results ]; then
    echo "Copying trained models from results directory..."
    /bin/cp -r "$SLURM_SUBMIT_DIR"/results "$JOB_SCRATCH"/
else
    echo "WARNING: No results directory found. Models must be trained first."
    echo "Expected model directories:"
    echo "  - results/distribute/"
    echo "  - results/combine/"
    echo "  - results/isolate/"
    echo "  - results/divide/"
fi

echo "Files copied successfully."

# ------------------------------------------------------------------------------
# 3. Modules & Python environment
# ------------------------------------------------------------------------------

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies to ~/.local ..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy
echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}');"

# ------------------------------------------------------------------------------
# 4. Prepare results directory on scratch
# ------------------------------------------------------------------------------

EVAL_RESULTS_DIR="$JOB_SCRATCH/evaluation_results"
mkdir -p "$EVAL_RESULTS_DIR"

echo "Evaluation results will be written to: $EVAL_RESULTS_DIR"

# ------------------------------------------------------------------------------
# 5. Run Algebra EBM Evaluation
# ------------------------------------------------------------------------------

echo "Starting algebra EBM evaluation..."

# Set evaluation parameters
MODEL_DIR="$JOB_SCRATCH/results"
OUTPUT_DIR="$EVAL_RESULTS_DIR"

# Run full evaluation suite
echo "Running full evaluation suite..."
python eval_algebra.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type full \
    --single_rule_problems 1000 \
    --multi_rule_problems 1000 \
    --constrained_problems 500 \
    --save_detailed \
    --verbose \
    --device auto \
    --inference_T 20 \
    --inference_step_size 0.1 \
    --seed 42

EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    echo "Full evaluation completed successfully!"
    
    # Also run quick individual evaluations for each rule
    echo "Running individual rule evaluations..."
    
    for rule in distribute combine isolate divide; do
        echo "Evaluating rule: $rule"
        python eval_algebra.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --eval_type single_rule \
            --rule "$rule" \
            --single_rule_problems 1000 \
            --verbose \
            --device auto \
            --seed 42
    done
    
    # Run multi-rule evaluations
    for num_rules in 2 3 4; do
        echo "Evaluating ${num_rules}-rule compositions..."
        python eval_algebra.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --eval_type multi_rule \
            --num_rules "$num_rules" \
            --multi_rule_problems 1000 \
            --verbose \
            --device auto \
            --seed 42
    done
    
    echo "All evaluations completed!"
else
    echo "Full evaluation failed with exit code: $EVAL_EXIT"
fi

# ------------------------------------------------------------------------------
# 6. Copy results back to home directory for safekeeping
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/evaluation_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_RESULTS_DIR"

echo "Syncing evaluation results back to: $FINAL_RESULTS_DIR"
rsync -av "$EVAL_RESULTS_DIR/" "$FINAL_RESULTS_DIR/"

# Also copy any logs or additional outputs
if [ -f "$JOB_SCRATCH"/*.log ]; then
    /bin/cp "$JOB_SCRATCH"/*.log "$FINAL_RESULTS_DIR"/
fi

echo "=============================================="
echo "  Results Summary"
echo "=============================================="
echo "Evaluation results saved to: $FINAL_RESULTS_DIR"
echo ""

# Print summary if main evaluation report exists
MAIN_REPORT="$FINAL_RESULTS_DIR/evaluation_report_full.txt"
if [ -f "$MAIN_REPORT" ]; then
    echo "EVALUATION SUMMARY:"
    echo "==================="
    head -50 "$MAIN_REPORT"
else
    echo "Main evaluation report not found. Check individual result files in:"
    echo "$FINAL_RESULTS_DIR"
fi

echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Final Exit Code: $EVAL_EXIT"
echo "=============================================="

exit $EVAL_EXIT