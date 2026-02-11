#!/bin/bash
#SBATCH --job-name=algebra_train_distribute
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/%x_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "=========================================="
echo "Algebra EBM Training: Distribute Rule"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Configuration (passed via environment variables)
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Create work directory for this job
WORK_DIR="${TMPDIR:-/tmp}/algebra-train-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU availability
echo "GPU info:"
nvidia-smi

# Clone repository
echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

# Install dependencies if needed
echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
pip install -q -r projects/algebra-ebm/requirements.txt 2>/dev/null || true

# Run training
echo "Starting distribute rule training..."
cd projects/algebra-ebm
python train_algebra.py \
    --rule distribute \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device cuda

echo "Training completed at $(date)"

# Copy results to persistent storage before cleanup
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/algebra-ebm/results/distribute" ]; then
    mkdir -p "$REPO_ROOT/projects/algebra-ebm/results"
    rsync -av "$WORK_DIR/projects/algebra-ebm/results/distribute/" "$REPO_ROOT/projects/algebra-ebm/results/distribute/"
    echo "Results copied to $REPO_ROOT/projects/algebra-ebm/results/distribute/"
else
    echo "Warning: No results directory found at $WORK_DIR/projects/algebra-ebm/results/distribute"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

echo "Job completed at $(date)"
