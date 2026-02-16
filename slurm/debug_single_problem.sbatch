#!/bin/bash
#SBATCH --job-name=debug_single
#SBATCH --output=slurm/logs/debug_single_%j.out
#SBATCH --error=slurm/logs/debug_single_%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

echo "Debug Single Problem - Inference Failure Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Run diagnostics
cd projects/algebra-ebm
python debug_single_problem.py

echo ""
echo "Completed: $(date)"
