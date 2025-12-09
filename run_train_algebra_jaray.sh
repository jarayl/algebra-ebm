#!/bin/bash
# =============================================================================
# Local Training Script for Algebra EBM Models
# =============================================================================
# Simplified version for local/remote SSH setup where:
# - Repository is already cloned
# - Python environment is already set up
# - No SLURM/FASRC environment needed
#
# Usage:
#   bash run_train_algebra_jaray.sh
#   
# Or to run in background:
#   nohup bash run_train_algebra_jaray.sh > train_output.log 2>&1 &
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Algebra EBM Training (Local Setup)"
echo "=============================================="
echo "Date:          $(date)"
echo "Hostname:      $(hostname)"
echo "Working Dir:   $(pwd)"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"

# Results directory
RESULTS_DIR="$SCRIPT_DIR/train_results"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"

# ------------------------------------------------------------------------------
# 2. Activate virtual environment (if exists)
# ------------------------------------------------------------------------------

if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "No venv found, using system Python"
fi

# Verify Python
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ------------------------------------------------------------------------------
# 3. Check GPU availability
# ------------------------------------------------------------------------------

echo ""
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')
else:
    print('WARNING: No GPU detected, training will be slow!')
"

# ------------------------------------------------------------------------------
# 4. Training Parameters
# ------------------------------------------------------------------------------

# Training steps configuration:
# - Quick test: 5000 steps
# - Standard: 50000 steps  
# - Production: 200000 steps
TRAIN_STEPS=50000

BATCH_SIZE=2048
NUM_PROBLEMS=50000
TIMESTEPS=10
GRADIENT_ACCUMULATE=2
STEP_SIZE_MULTIPLIER=0.2

# Adjust batch size for smaller GPUs
GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
if (( $(echo "$GPU_MEMORY < 15 && $GPU_MEMORY > 0" | bc -l) )); then
    echo "GPU has ${GPU_MEMORY}GB memory, reducing batch size"
    BATCH_SIZE=1024
fi

echo ""
echo "Training parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATE (effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATE)))"
echo "  Training steps: $TRAIN_STEPS" 
echo "  Problems per rule: $NUM_PROBLEMS"
echo "  Timesteps: $TIMESTEPS"
echo "  Step size multiplier: $STEP_SIZE_MULTIPLIER"
echo ""

# ------------------------------------------------------------------------------
# 5. Train All 4 Rules
# ------------------------------------------------------------------------------

RULES=("distribute" "combine" "isolate" "divide")
TOTAL_RULES=${#RULES[@]}
FAILED_RULES=()
SUCCESSFUL_RULES=()

for i in "${!RULES[@]}"; do
    rule="${RULES[$i]}"
    rule_num=$((i + 1))
    
    echo "=============================================="
    echo "Training rule ${rule_num}/${TOTAL_RULES}: ${rule}"
    echo "=============================================="
    
    # Create rule-specific results directory
    mkdir -p "$RESULTS_DIR/$rule"
    
    # Run training
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
        --save_and_sample_every 5000 \
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
        echo "✓ Rule '$rule' training completed in ${duration}s"
        SUCCESSFUL_RULES+=("$rule")
    else
        echo "✗ Rule '$rule' training FAILED (exit code: $TRAIN_EXIT)"
        FAILED_RULES+=("$rule")
    fi
    
    echo ""
done

# ------------------------------------------------------------------------------
# 6. Training Summary
# ------------------------------------------------------------------------------

echo "=============================================="
echo "  Training Summary"
echo "=============================================="

for rule in "${RULES[@]}"; do
    if [ -f "$RESULTS_DIR/$rule/model.pt" ]; then
        model_size=$(du -h "$RESULTS_DIR/$rule/model.pt" | cut -f1)
        echo "✓ $rule: $model_size model saved at $RESULTS_DIR/$rule/model.pt"
    else
        echo "✗ $rule: No model file found"
    fi
done

echo ""
echo "Successful: ${#SUCCESSFUL_RULES[@]}/${TOTAL_RULES} rules"
echo "Failed:     ${#FAILED_RULES[@]}/${TOTAL_RULES} rules"

if [ ${#FAILED_RULES[@]} -gt 0 ]; then
    echo "Failed rules: ${FAILED_RULES[*]}"
fi

echo ""
if [ ${#SUCCESSFUL_RULES[@]} -eq $TOTAL_RULES ]; then
    echo "🎉 ALL MODELS TRAINED SUCCESSFULLY!"
    echo ""
    echo "Next step - run evaluation:"
    echo "  bash run_eval_local.sh"
    FINAL_EXIT=0
elif [ ${#SUCCESSFUL_RULES[@]} -gt 0 ]; then
    echo "⚠️  PARTIAL SUCCESS: ${#SUCCESSFUL_RULES[@]}/${TOTAL_RULES} models trained"
    FINAL_EXIT=1
else
    echo "❌ ALL TRAINING FAILED"
    FINAL_EXIT=2
fi

echo "=============================================="
echo "  Finished at: $(date)"
echo "=============================================="

exit $FINAL_EXIT
