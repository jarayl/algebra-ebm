#!/bin/bash
#SBATCH -J eval_algebra                      # Job name
#SBATCH -p gpu_test                          # Partition (use gpu for real runs)
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-07:00:00                       # 2 days
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
# 2. Clone Git repository to get latest codebase
# ------------------------------------------------------------------------------

# Git is available system-wide, no module needed

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


echo "Repository cloned successfully to: $REPO_DIR"

# ------------------------------------------------------------------------------
# 3. Copy necessary files from repository → scratch working directory
# ------------------------------------------------------------------------------

echo "Copying algebra evaluation files from repository to scratch..."

# Copy all algebra-related Python files
/bin/cp "$REPO_DIR"/eval_algebra.py "$JOB_SCRATCH"/
/bin/cp "$REPO_DIR"/algebra_*.py "$JOB_SCRATCH"/

# Copy core infrastructure files  
/bin/cp "$REPO_DIR"/dataset.py "$JOB_SCRATCH"/
/bin/cp "$REPO_DIR"/models.py "$JOB_SCRATCH"/

# Copy diffusion library (required for model loading)
/bin/cp -r "$REPO_DIR"/diffusion_lib "$JOB_SCRATCH"/

# Copy IREM library if it exists
if [ -d "$REPO_DIR"/irem_lib ]; then
    /bin/cp -r "$REPO_DIR"/irem_lib "$JOB_SCRATCH"/
fi

# Copy trained models from results directory
# Check multiple locations for trained models
MODELS_COPIED=false

# Check repository results directory
if [ -d "$REPO_DIR"/results ]; then
    echo "Copying trained models from repository results directory..."
    /bin/cp -r "$REPO_DIR"/results "$JOB_SCRATCH"/
    MODELS_COPIED=true
# Check submit directory 
elif [ -d "$SLURM_SUBMIT_DIR"/results ]; then
    echo "Copying trained models from submit directory..."
    /bin/cp -r "$SLURM_SUBMIT_DIR"/results "$JOB_SCRATCH"/
    MODELS_COPIED=true
# Check home directory
elif [ -d "$HOME"/results ]; then
    echo "Copying trained models from home directory..."
    /bin/cp -r "$HOME"/results "$JOB_SCRATCH"/
    MODELS_COPIED=true
# Check if models are already in scratch (from previous runs)
elif [ -d "$LAB_SCRATCH_ROOT"/results ]; then
    echo "Copying trained models from lab scratch..."
    /bin/cp -r "$LAB_SCRATCH_ROOT"/results "$JOB_SCRATCH"/
    MODELS_COPIED=true
fi

if [ "$MODELS_COPIED" = false ]; then
    echo "ERROR: No trained models found in any expected location!"
    echo "Checked locations:"
    echo "  - $REPO_DIR/results"
    echo "  - $SLURM_SUBMIT_DIR/results"
    echo "  - $HOME/results"
    echo "  - $LAB_SCRATCH_ROOT/results"
    echo ""
    echo "Expected model structure:"
    echo "  - results/distribute/model.pt (or model-*.pt)"
    echo "  - results/combine/model.pt (or model-*.pt)"
    echo "  - results/isolate/model.pt (or model-*.pt)"
    echo "  - results/divide/model.pt (or model-*.pt)"
    echo ""
    echo "Models must be trained first using the training script."
    echo "FAST FAIL: Cannot run evaluation without trained models."
    exit 1
fi

# Verify model files after copying
if [ -d "$JOB_SCRATCH/results" ]; then
    echo "Checking for model files in copied results..."
    find "$JOB_SCRATCH/results" -name "*.pt" -type f | head -10
    
    # Count models found
    MODEL_COUNT=$(find "$JOB_SCRATCH/results" -name "*.pt" -type f | wc -l)
    echo "Found $MODEL_COUNT model files total"
    
    if [ $MODEL_COUNT -eq 0 ]; then
        echo "ERROR: No model files (.pt) found in results directory!"
        echo "FAST FAIL: Cannot run evaluation without model files."
        exit 1
    fi
    
    # Check each rule directory - require at least one model per rule
    MISSING_RULES=0
    for rule in distribute combine isolate divide; do
        RULE_DIR="$JOB_SCRATCH/results/$rule"
        if [ -d "$RULE_DIR" ]; then
            MODEL_FILES=$(find "$RULE_DIR" -name "*.pt" -type f | wc -l)
            echo "Rule $rule: $MODEL_FILES model files"
            if [ $MODEL_FILES -eq 0 ]; then
                echo "  ERROR: No model files found for rule $rule"
                MISSING_RULES=$((MISSING_RULES + 1))
            else
                echo "  Model files for $rule:"
                find "$RULE_DIR" -name "*.pt" -type f | head -3
            fi
        else
            echo "Rule $rule: directory not found"
            MISSING_RULES=$((MISSING_RULES + 1))
        fi
    done
    
    if [ $MISSING_RULES -gt 0 ]; then
        echo "ERROR: $MISSING_RULES rules are missing models!"
        echo "FAST FAIL: Cannot run complete evaluation without all rule models."
        exit 1
    fi
else
    echo "ERROR: No results directory created - evaluation will fail"
    echo "FAST FAIL: Missing results directory."
    exit 1
fi

echo "Files copied successfully."

# ------------------------------------------------------------------------------
# 4. Modules & Python environment
# ------------------------------------------------------------------------------

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

# Add repository to Python path for imports
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
echo "Added repository to Python path: $REPO_DIR"
echo "Current PYTHONPATH: $PYTHONPATH"

echo "Installing dependencies to ~/.local ..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn sympy
echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU availability and fail fast if CUDA required but not available
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); exit(0 if torch.cuda.is_available() else 1)" || {
    echo "WARNING: CUDA not available, falling back to CPU (evaluation will be much slower)"
    echo "If you specifically need GPU evaluation, this job should fail fast."
    # Uncomment the following lines to fail fast when GPU is required:
    # echo "FAST FAIL: GPU required but CUDA not available."
    # exit 1
}

# ------------------------------------------------------------------------------
# 5. Prepare results directory on scratch
# ------------------------------------------------------------------------------

EVAL_RESULTS_DIR="$JOB_SCRATCH/evaluation_results"
mkdir -p "$EVAL_RESULTS_DIR"

echo "Evaluation results will be written to: $EVAL_RESULTS_DIR"

# ------------------------------------------------------------------------------
# 6. Run Algebra EBM Evaluation (using REAL diffusion inference)
# ------------------------------------------------------------------------------

echo "Starting algebra EBM evaluation..."
echo ""
echo "=============================================="
echo "  USING REAL GaussianDiffusion1D.sample()"
echo "  This achieves 87%+ distance improvement"
echo "=============================================="
echo ""

# Set evaluation parameters
MODEL_DIR="$JOB_SCRATCH/results"
OUTPUT_DIR="$EVAL_RESULTS_DIR"

# Run full evaluation suite with REAL diffusion inference
# NOTE: --use_real_diffusion uses GaussianDiffusion1D.sample() which is the
# CORRECT inference method (achieves 87% distance improvement vs 45% with old method)
echo "Running full evaluation suite with real diffusion inference..."
python eval_algebra.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type full \
    --use_real_diffusion \
    --single_rule_problems 100 \
    --multi_rule_problems 100 \
    --constrained_problems 50 \
    --save_detailed \
    --verbose \
    --device auto \
    --seed 42

EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    echo "Full evaluation completed successfully!"
    
    # Also run quick individual evaluations for each rule with real diffusion
    echo "Running individual rule evaluations with real diffusion..."
    
    # Start background jobs for each rule
    declare -A rule_pids
    for rule in distribute combine isolate divide; do
        echo "Evaluating rule: $rule"
        
        # Find the checkpoint for this rule
        RULE_CHECKPOINT=""
        if [ -f "$MODEL_DIR/$rule/model.pt" ]; then
            RULE_CHECKPOINT="$MODEL_DIR/$rule/model.pt"
        elif [ -f "$MODEL_DIR/$rule/model-1.pt" ]; then
            RULE_CHECKPOINT="$MODEL_DIR/$rule/model-1.pt"
        else
            echo "WARNING: No checkpoint found for rule $rule, skipping individual evaluation"
            continue
        fi
        
        python eval_algebra.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --eval_type single_rule \
            --rule "$rule" \
            --use_real_diffusion \
            --checkpoint "$RULE_CHECKPOINT" \
            --single_rule_problems 100 \
            --verbose \
            --device auto \
            --seed 42 &
        
        rule_pids[$rule]=$!
        echo "Rule $rule evaluation started with PID: ${rule_pids[$rule]}"
    done
    
    # Wait for all background jobs and collect exit codes
    echo "Waiting for all rule evaluations to complete..."
    for rule in distribute combine isolate divide; do
        wait ${rule_pids[$rule]}
        RULE_EXIT=$?
        if [ $RULE_EXIT -ne 0 ]; then
            echo "ERROR: Individual evaluation for rule $rule failed with exit code: $RULE_EXIT"
            echo "FAST FAIL: Stopping due to failed rule evaluation."
            exit $RULE_EXIT
        else
            echo "Rule $rule evaluation completed successfully"
        fi
    done
    
    # Multi-rule evaluations NOTE: --use_real_diffusion not yet supported for multi-rule
    # These use the legacy AlgebraInference method (lower accuracy expected)
    echo ""
    echo "=============================================="
    echo "  Multi-rule evaluations (legacy inference)"
    echo "  NOTE: Real diffusion not yet supported"
    echo "=============================================="
    for num_rules in 2 3 4; do
        echo "Evaluating ${num_rules}-rule compositions (legacy method)..."
        python eval_algebra.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --eval_type multi_rule \
            --num_rules "$num_rules" \
            --multi_rule_problems 100 \
            --verbose \
            --device auto \
            --inference_T 20 \
            --inference_step_size 0.1 \
            --seed 42
        
        # Check multi-rule evaluation exit code
        MULTI_EXIT=$?
        if [ $MULTI_EXIT -ne 0 ]; then
            echo "ERROR: Multi-rule evaluation for $num_rules rules failed with exit code: $MULTI_EXIT"
            echo "FAST FAIL: Stopping further multi-rule evaluations."
            exit $MULTI_EXIT
        fi
    done
    
    echo "All evaluations completed!"
else
    echo "Full evaluation failed with exit code: $EVAL_EXIT"
    echo "FAST FAIL: Main evaluation failed, skipping additional evaluations."
    exit $EVAL_EXIT
fi

# ------------------------------------------------------------------------------
# 7. Copy results back to home directory for safekeeping
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