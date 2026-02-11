#!/bin/bash
#SBATCH --job-name=algebra_train_combine_debug
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/debug_combine_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/debug_combine_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=========================================="
echo "DEBUG: Algebra EBM Training: Combine Rule"
echo "=========================================="
echo "[1] Job started at $(date)"
echo "[1] Running on host: $(hostname)"
echo "[1] Job ID: $SLURM_JOB_ID"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

WORK_DIR="${TMPDIR:-/tmp}/algebra-train-${SLURM_JOB_ID}"
echo "[2] Creating work directory: $WORK_DIR"
mkdir -p "$WORK_DIR"
echo "[2] Work directory created"

echo "[3] Loading modules..."
module load python/3.10.13-fasrc01
echo "[3] Python loaded"
module load cuda/11.8.0-fasrc01
echo "[3] CUDA loaded"

echo "[4] GPU info:"
nvidia-smi | head -20

echo "[5] Cloning repository from: $GIT_URL"
git clone "$GIT_URL" "$WORK_DIR" 2>&1 | head -20
echo "[5] Clone completed"

echo "[6] Entering work directory"
cd "$WORK_DIR"
echo "[6] Current directory: $(pwd)"

echo "[7] Checking out git SHA: $GIT_SHA"
git checkout "$GIT_SHA" 2>&1 | head -10
echo "[7] Checkout completed"

echo "[8] Repository info: $(git rev-parse --short HEAD)"
git log -1 --oneline

echo "[9] Listing projects/algebra-ebm directory:"
ls -la projects/algebra-ebm/ | head -20

echo "[10] Listing Python files:"
ls -la projects/algebra-ebm/train*.py

echo "[11] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5
echo "[11] PyTorch installed"
pip install -q -r projects/algebra-ebm/requirements.txt 2>&1 | tail -5
echo "[11] Requirements installed"

echo "[12] Starting combine rule training..."
cd projects/algebra-ebm
echo "[12] Current directory: $(pwd)"
python train_algebra.py \
    --rule combine \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device cuda
TRAIN_EXIT=$?
echo "[12] Training completed with exit code: $TRAIN_EXIT"

echo "[13] Job completed at $(date)"
