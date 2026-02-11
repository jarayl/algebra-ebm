#!/bin/bash
#SBATCH --job-name=algebra_train_isolate
#SBATCH --output=projects/algebra-ebm/slurm/logs/%x_%j.out
#SBATCH --error=projects/algebra-ebm/slurm/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "=========================================="
echo "Algebra EBM Training: Isolate Rule"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

WORK_DIR="${TMPDIR:-/tmp}/algebra-train-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "GPU info:"
nvidia-smi

echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
git log -1 --oneline

echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
pip install -q -r projects/algebra-ebm/requirements.txt 2>/dev/null || true

echo "Starting isolate rule training..."
cd projects/algebra-ebm
python train_algebra.py \
    --rule isolate \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device cuda

echo "Training completed at $(date)"

REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/algebra-ebm/results/isolate" ]; then
    mkdir -p "$REPO_ROOT/projects/algebra-ebm/results"
    rsync -av "$WORK_DIR/projects/algebra-ebm/results/isolate/" "$REPO_ROOT/projects/algebra-ebm/results/isolate/"
    echo "Results copied to $REPO_ROOT/projects/algebra-ebm/results/isolate/"
else
    echo "Warning: No results directory found at $WORK_DIR/projects/algebra-ebm/results/isolate"
fi

echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

echo "Job completed at $(date)"
