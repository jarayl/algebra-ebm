#!/bin/bash
#SBATCH --job-name=debug_single_v2
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/%x_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

echo "=========================================="
echo "Debug Single Problem - Inference Failure Analysis"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# GPU info
nvidia-smi

# Navigate to project
cd /n/home03/mkrasnow/research-repo/projects/algebra-ebm

# Run diagnostic
python debug_single_problem.py

echo ""
echo "Job completed at $(date)"
