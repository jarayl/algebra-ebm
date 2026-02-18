#!/bin/bash
#SBATCH --job-name=algebra_diag_multistart
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diag_multistart_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diag_multistart_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

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

# Clone algebra-ebm repo directly
git clone https://github.com/mdkrasnow/algebra-ebm.git .
git checkout "$GIT_SHA"

# Install dependencies
pip install --user -e .

# Run diagnostic evaluation
python eval_algebra.py \
    --model_dir /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results \
    --eval_type single_rule \
    --rule distribute \
    --single_rule_problems 100 \
    --inference_T 50 \
    --num_inference_starts 10 \
    --enable_diagnostics \
    --diagnostics_dir /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostics/multistart \
    --output_dir /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostics/multistart \
    --seed 42

# Cleanup temp workspace (results already written to home dir)
cd /
rm -rf "$WORK_DIR"
