#!/bin/bash
#SBATCH --job-name=algebra_test
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/test_%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/test_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

echo "Test job started"
echo "Hostname: $(hostname)"
nvidia-smi
echo "Test completed"
