#!/bin/bash
#SBATCH -J comparison_eval                    # Job name
#SBATCH -p gpu_test                           # Use GPU partition (change to 'gpu' for real runs)
#SBATCH --account=ydu_lab                     # Your lab account
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH -c 16                                 # 16 CPU cores
#SBATCH -t 00-01:00:00                        # 1 hour (adjust as needed)
#SBATCH --mem=64G                             # 64 GB RAM
#SBATCH -o comparison_eval_%j.out             # STDOUT file
#SBATCH -e comparison_eval_%j.err             # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Monolithic vs Compositional Comparison"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configure FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/comparison_${SLURM_JOB_ID}"

echo "Setting up scratch workspace: $JOB_SCRATCH"

mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

mkdir -p "$JOB_SCRATCH" || {
    echo "ERROR: Cannot create $JOB_SCRATCH"
    exit 1
}

cd "$JOB_SCRATCH"

# ------------------------------------------------------------------------------
# 2. Clone repository and setup
# ------------------------------------------------------------------------------

REPO_URL="https://github.com/mdkrasnow/algebra-ebm.git"
REPO_DIR="$JOB_SCRATCH/algebra-ebm"

echo "Cloning repository..."
git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository"
    exit 1
}

# Copy evaluation scripts
/bin/cp "$REPO_DIR"/eval_algebra.py "$JOB_SCRATCH"/
/bin/cp "$REPO_DIR"/scripts/compare_monolithic_vs_compositional.py "$JOB_SCRATCH"/
/bin/cp -r "$REPO_DIR"/src "$JOB_SCRATCH"/

# ------------------------------------------------------------------------------
# 3. Find and copy trained models
# ------------------------------------------------------------------------------

echo "Looking for trained models..."

MODELS_FOUND=false

# Check submit directory first (most likely location)
if [ -d "$SLURM_SUBMIT_DIR/results" ]; then
    echo "Found models in submit directory"
    /bin/cp -r "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/
    MODELS_FOUND=true
elif [ -d "$REPO_DIR/results" ]; then
    echo "Found models in repository"
    /bin/cp -r "$REPO_DIR/results" "$JOB_SCRATCH"/
    MODELS_FOUND=true
fi

if [ "$MODELS_FOUND" = false ]; then
    echo "ERROR: No trained models found!"
    echo "Expected structure:"
    echo "  - results/monolithic/model.pt"
    echo "  - results/distribute/model.pt"
    echo "  - results/combine/model.pt"
    echo "  - results/isolate/model.pt" 
    echo "  - results/divide/model.pt"
    echo ""
    echo "Run training scripts first:"
    echo "  sbatch run_train_monolithic.sh"
    echo "  sbatch run_train_algebra.sh"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Validate model files
# ------------------------------------------------------------------------------

echo "Validating model files..."

# Check monolithic model
if [ ! -f "$JOB_SCRATCH/results/monolithic/model.pt" ]; then
    echo "ERROR: Monolithic model not found at results/monolithic/model.pt"
    echo "Run: sbatch run_train_monolithic.sh"
    exit 1
fi

# Check individual rule models
MISSING_RULES=0
for rule in distribute combine isolate divide; do
    if [ ! -f "$JOB_SCRATCH/results/$rule/model.pt" ]; then
        echo "WARNING: Missing $rule model"
        MISSING_RULES=$((MISSING_RULES + 1))
    fi
done

if [ $MISSING_RULES -eq 4 ]; then
    echo "ERROR: No individual rule models found!"
    echo "Run: sbatch run_train_algebra.sh"
    exit 1
elif [ $MISSING_RULES -gt 0 ]; then
    echo "WARNING: Only $((4 - MISSING_RULES))/4 rule models found"
    echo "Comparison will run with available models"
fi

echo "✓ Found monolithic model: $(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)"
echo "✓ Found $((4 - MISSING_RULES))/4 rule models"

# ------------------------------------------------------------------------------
# 5. Setup Python environment
# ------------------------------------------------------------------------------

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${JOB_SCRATCH}:${PYTHONPATH}"

echo "Installing dependencies..."
python -m pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy

echo "GPU check:"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ------------------------------------------------------------------------------
# 6. Run comparison evaluation
# ------------------------------------------------------------------------------

EVAL_SAMPLES=50  # Quick test (change to 1000 for full evaluation)
OUTPUT_DIR="$JOB_SCRATCH/comparison_results"

echo "=============================================="
echo "  Running Comparison Evaluation"
echo "=============================================="
echo "Monolithic model: $JOB_SCRATCH/results/monolithic/model.pt"
echo "Rule models dir:  $JOB_SCRATCH/results"
echo "Samples per test: $EVAL_SAMPLES"
echo "Output dir:       $OUTPUT_DIR"
echo ""

start_time=$(date +%s)

# Option 2: Direct call (recommended for cluster)
python eval_algebra.py \
    --eval_type comparison \
    --use_real_diffusion \
    --checkpoint "$JOB_SCRATCH/results/monolithic/model.pt" \
    --monolithic_checkpoint "$JOB_SCRATCH/results/monolithic/model.pt" \
    --model_dir "$JOB_SCRATCH/results" \
    --max_samples $EVAL_SAMPLES \
    --output_dir "$OUTPUT_DIR" \
    --verbose

EVAL_EXIT=$?

# Option 1: Use wrapper script (alternative - has path issues on cluster)
# python compare_monolithic_vs_compositional.py \
#     --monolithic_checkpoint "$JOB_SCRATCH/results/monolithic/model.pt" \
#     --compositional_dir "$JOB_SCRATCH/results" \
#     --num_samples $EVAL_SAMPLES \
#     --output_dir "$OUTPUT_DIR" \
#     --verbose

end_time=$(date +%s)
duration=$((end_time - start_time))

# ------------------------------------------------------------------------------
# 7. Results analysis
# ------------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Evaluation Results"
echo "=============================================="

if [ $EVAL_EXIT -eq 0 ]; then
    echo "✓ Comparison evaluation completed in ${duration}s"
    
    # Show comparison report if it exists
    if [ -f "$OUTPUT_DIR/comparison_report.md" ]; then
        echo ""
        echo "COMPARISON SUMMARY:"
        echo "===================="
        head -30 "$OUTPUT_DIR/comparison_report.md"
    else
        echo "⚠️  Comparison report not found"
    fi
else
    echo "✗ Evaluation failed with exit code: $EVAL_EXIT"
fi

# ------------------------------------------------------------------------------
# 8. Copy results back
# ------------------------------------------------------------------------------

FINAL_OUTPUT="$SLURM_SUBMIT_DIR/comparison_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_OUTPUT"

echo ""
echo "Copying results to: $FINAL_OUTPUT"
rsync -av "$OUTPUT_DIR/" "$FINAL_OUTPUT/"

# Copy logs
if [ -f "$JOB_SCRATCH"/*.log ]; then
    /bin/cp "$JOB_SCRATCH"/*.log "$FINAL_OUTPUT"/
fi

echo ""
echo "=============================================="
echo "  Final Summary"
echo "=============================================="

if [ $EVAL_EXIT -eq 0 ]; then
    echo "✅ COMPARISON EVALUATION SUCCESSFUL!"
    echo ""
    echo "Results saved to: $FINAL_OUTPUT"
    echo ""
    echo "Key files:"
    if [ -f "$FINAL_OUTPUT/comparison_report.md" ]; then
        echo "  📊 comparison_report.md (main results)"
    fi
    if [ -f "$FINAL_OUTPUT/comparison_results.json" ]; then
        echo "  📈 comparison_results.json (detailed data)"
    fi
    
    echo ""
    echo "View results with:"
    echo "  cat $FINAL_OUTPUT/comparison_report.md"
else
    echo "❌ Evaluation failed"
    echo "Check error logs in: $FINAL_OUTPUT"
fi

echo ""
echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Exit Code: $EVAL_EXIT"
echo "=============================================="

exit $EVAL_EXIT