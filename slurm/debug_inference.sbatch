#!/bin/bash
#SBATCH --job-name=algebra_debug
#SBATCH --output=slurm/logs/algebra_debug_%j.out
#SBATCH --error=slurm/logs/algebra_debug_%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "=========================================="
echo "Algebra EBM Inference Diagnostics"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Work directory: $PWD"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# GPU info
nvidia-smi

# Navigate to project
cd projects/algebra-ebm

# Run diagnostics
echo ""
echo "Running inference diagnostics..."
python debug_inference.py

echo ""
echo "Job completed at $(date)"
