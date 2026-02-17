#!/bin/bash
#SBATCH --job-name=algebra_diag_longrun
#SBATCH --output=slurm/logs/algebra_diag_longrun_%j.out
#SBATCH --error=slurm/logs/algebra_diag_longrun_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Capture git commit
GIT_SHA=${GIT_SHA:-$(git rev-parse HEAD)}

# Setup environment
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
nvidia-smi

# Clone to temp directory
WORK_DIR="/tmp/algebra-ebm-diag-job-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repo
git clone https://github.com/mdkrasnow/research-repo.git .
git checkout "$GIT_SHA"

# Install dependencies
pip install --user -e .

# Navigate to project
cd projects/algebra-ebm

# Run diagnostic evaluation
python eval_algebra.py \
    --model_dir results \
    --eval_type single_rule \
    --rule distribute \
    --single_rule_problems 100 \
    --inference_T 500 \
    --num_inference_starts 1 \
    --enable_diagnostics \
    --diagnostics_dir results/diagnostics/longrun \
    --output_dir results/diagnostics/longrun \
    --seed 42

# Copy results back
rsync -av results/diagnostics/longrun/ /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostics/longrun/

# Cleanup
cd /
rm -rf "$WORK_DIR"
