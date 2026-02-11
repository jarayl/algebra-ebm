# Cluster Training Guide: Algebra EBM Models
**Date:** 2026-02-11
**Status:** Ready to submit training jobs

## Overview

All algebra-ebm models are trained and stored **exclusively on the cluster**. This guide covers:
1. Submitting training jobs
2. Monitoring progress
3. Verifying model storage
4. Running evaluation experiments on cluster

## Quick Start

```bash
# 1. Set up cluster SSH (if needed)
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh

# 2. Submit all 5 training jobs
cd projects/algebra-ebm
bash submit_cluster_training.sh

# 3. Monitor progress
squeue -u $USER

# 4. After training completes, run evaluations
python run_experiments.py
```

## Detailed Steps

### Step 1: Ensure Cluster Connection

```bash
# Verify SSH session is active
scripts/cluster/ensure_session.sh

# If not, set up SSH
scripts/cluster/ssh_bootstrap.sh
```

### Step 2: Submit Training Jobs

```bash
cd projects/algebra-ebm
bash submit_cluster_training.sh
```

This script will:
- Submit 4 jobs to `gpu_test` partition (shorter time limit, higher priority)
  - distribute rule model
  - combine rule model
  - isolate rule model
  - divide rule model
- Submit 1 job to `gpu` partition (longer time limit)
  - monolithic baseline model
- Create tracking file at `.state/cluster_training.json` with job IDs
- Display job IDs and monitoring instructions

### Step 3: Monitor Training Progress

**Check job status:**
```bash
squeue -u $USER
squeue -j <job_id>
```

**Watch logs in real-time:**
```bash
# After login to cluster
ssh cluster.local
tail -f /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_train_distribute_*.out
```

**Check final status:**
```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### Step 4: Verify Model Storage

**Check models are on cluster:**
```bash
ssh cluster.local
ls -lh /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/
```

Expected output:
```
distribute/model.pt          # Distribute rule model
combine/model.pt             # Combine rule model
isolate/model.pt             # Isolate rule model
divide/model.pt              # Divide rule model
monolithic/model.pt          # Monolithic baseline
```

### Step 5: Run Evaluation Experiments

After training completes:

```bash
cd projects/algebra-ebm

# Run all evaluations
python run_experiments.py

# Or run specific experiments
python run_experiments.py --experiment exp_001_single_rule_baseline
python run_experiments.py --experiment exp_007_comparison
```

## SLURM Job Details

### Job Configuration

Each training job has:
- **1 GPU** (A100 or equivalent)
- **4 CPU cores** for data loading
- **32GB RAM**
- **Time limit:** 4 hours (distribute/combine/isolate/divide), 6 hours (monolithic)

### Job Workflow

1. Clone repository from git at specified SHA
2. Install dependencies
3. Load Python and CUDA modules
4. Run training script with GPU
5. Sync results to persistent storage `/n/home03/mkrasnow/research-repo/`
6. Clean up temporary directory in `/tmp`

### Estimated Training Times

With GPU acceleration:
- Distribute rule: ~1-2 hours
- Combine rule: ~1-2 hours
- Isolate rule: ~1-2 hours
- Divide rule: ~1-2 hours
- Monolithic baseline: ~2-3 hours
- **Total: ~8-12 hours (parallel execution)**

## Monitoring Commands

### Check all your jobs
```bash
squeue -u $USER
```

### Check specific job details
```bash
squeue -j <job_id>
sacct -j <job_id> --format=All
```

### Get job resource usage
```bash
sacct -j <job_id> --format=JobID,JobName,CPUTime,MaxRSS,Elapsed,State
```

### View job output in real-time (on cluster)
```bash
ssh cluster.local
cd /n/home03/mkrasnow/research-repo
tail -f projects/algebra-ebm/slurm/logs/algebra_train_*.out
```

### View specific job log
```bash
cat projects/algebra-ebm/slurm/logs/algebra_train_distribute_<JOB_ID>.out
cat projects/algebra-ebm/slurm/logs/algebra_train_distribute_<JOB_ID>.err
```

## Troubleshooting

### Job not submitting
**Problem:** Error when running `submit_cluster_training.sh`
**Solution:**
- Check SSH is connected: `scripts/cluster/ensure_session.sh`
- Verify SBATCH files exist: `ls -la projects/algebra-ebm/slurm/train_*.sbatch`

### Job cancelled or timeout
**Problem:** Job killed due to time limit or memory
**Solution:**
- Increase time limit in `.sbatch` file (currently 4-6 hours)
- Reduce batch_size in training script
- Check logs: `cat projects/algebra-ebm/slurm/logs/algebra_train_*.err`

### Models not saved to persistent storage
**Problem:** Models trained but not in `/n/home03/mkrasnow/research-repo/`
**Solution:**
- Check logs for rsync errors
- Manually copy: `rsync -av cluster.local:/tmp/algebra-train-<JOB_ID>/projects/algebra-ebm/results/ projects/algebra-ebm/results/`

### Evaluation can't find models
**Problem:** eval_algebra.py says "No models loaded"
**Solution:**
- Verify models are on cluster: `ssh cluster.local "ls /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/"`
- Check model_dir in eval command matches cluster path
- Verify SSH session is active

## Cluster Storage Paths

**Local (development):**
```
/Users/mkrasnow/Desktop/research-repo/projects/algebra-ebm/
  ├── train_algebra.py
  ├── eval_algebra.py
  └── (code only, no models)
```

**Cluster (persistent models):**
```
/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/
  ├── distribute/model.pt
  ├── combine/model.pt
  ├── isolate/model.pt
  ├── divide/model.pt
  └── monolithic/model.pt
```

## Tracking Job Submissions

Job submission tracking is saved to:
```
projects/algebra-ebm/.state/cluster_training.json
```

This file contains:
- Submission timestamp
- Git SHA used for training
- SLURM Job IDs for each model
- Status and notes

You can view it to remember job IDs:
```bash
cat projects/algebra-ebm/.state/cluster_training.json | python -m json.tool
```

## Full Workflow Example

```bash
# 1. Navigate to repo
cd /Users/mkrasnow/Desktop/research-repo

# 2. Ensure cluster connection
scripts/cluster/ensure_session.sh

# 3. Submit training (5 jobs)
cd projects/algebra-ebm
bash submit_cluster_training.sh

# Output will show job IDs and tracking file location

# 4. Monitor for ~12 hours
squeue -u $USER

# 5. After "STATE" shows COMPLETED for all jobs
sacct -j <job_id1>,<job_id2>,<job_id3>,<job_id4>,<job_id5> --format=JobID,JobName,State

# 6. Run evaluations
python run_experiments.py

# 7. Check results
ls -la runs/exp_001*/results/
cat runs/exp_001*/results/summary.json | python -m json.tool
```

## Next Steps

1. **Submit training:** `bash submit_cluster_training.sh`
2. **Monitor:** `squeue -u $USER`
3. **Wait:** Training completes in ~12 hours (parallel)
4. **Evaluate:** `python run_experiments.py`
5. **Analyze:** Check results in `runs/` directory

## Questions?

For issues with:
- **SSH/cluster connection:** See `scripts/cluster/ssh_bootstrap.sh`
- **SLURM jobs:** Check `projects/algebra-ebm/slurm/logs/`
- **Model training:** Check SLURM error logs
- **Evaluation:** Check `run_experiments.py` output

All models and results are stored exclusively on the cluster at:
```
/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/
```

