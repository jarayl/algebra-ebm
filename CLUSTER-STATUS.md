# Cluster Diagnostic Experiment - Live Status

**Last Updated**: 2026-02-17 05:30 UTC
**Git Commit**: 6753b08

---

## 🚀 Job Submitted: Diagnostic Training

### Training Job Details
- **Job ID**: `60698745`
- **Script**: `slurm/train_diagnostic_no_norm.sbatch`
- **Partition**: gpu
- **Resources**: 1 GPU, 4 CPUs, 32GB RAM
- **Time Limit**: 6 hours
- **Expected Runtime**: ~3 hours

### Configuration
- **Rule**: distribute
- **Train Steps**: 10,000
- **Batch Size**: 2,048
- **Timesteps**: 10
- **Output**: `/n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm`

### What It's Testing
**Hypothesis**: Encoder normalization (`||embedding|| = 1.0`) breaks energy learning

**Fix Applied**: Disabled normalization in `src/algebra/algebra_encoder.py` line 135

**Expected Outcome**:
- Energy scale parameters learn meaningful values (not stuck at 1.0)
- Energy landscapes improve from 54% correct to >80%
- Single-rule accuracy improves from 6% to >30%

---

## 📊 Monitoring Commands

### Check Job Status
```bash
# Quick status
bash scripts/cluster/status.sh 60698745

# Detailed status with sacct
bash scripts/cluster/remote_cmd.sh "sacct -j 60698745 --format=JobID,JobName,State,Elapsed,TimeLimit,NodeList"
```

### View Live Logs
```bash
# Training output
bash scripts/cluster/remote_cmd.sh "tail -f /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.out"

# Training errors (if any)
bash scripts/cluster/remote_cmd.sh "tail -f /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.err"
```

### Check Progress
```bash
# See last 20 lines of output
bash scripts/cluster/remote_cmd.sh "tail -20 /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.out"

# Look for energy scale values
bash scripts/cluster/remote_cmd.sh "grep -i 'energy_scale' /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm/training.log 2>/dev/null | tail -5"
```

---

## ⏭️ Next Steps

### When Training Completes (~3 hours)

1. **Check Training Success**
```bash
# Verify model was saved
bash scripts/cluster/remote_cmd.sh "ls -lh /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm/distribute/model.pt"

# Check final training logs
bash scripts/cluster/remote_cmd.sh "tail -30 /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.out"
```

2. **Submit Evaluation Job**
```bash
bash scripts/cluster/submit.sh projects/algebra-ebm/slurm/eval_diagnostic_no_norm.sbatch algebra-ebm
```

3. **Monitor Evaluation** (~10 minutes)
```bash
# Check status (replace JOB_ID with actual ID from submit)
bash scripts/cluster/status.sh <JOB_ID>

# View logs
bash scripts/cluster/remote_cmd.sh "tail -f /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_eval_<JOB_ID>.out"
```

4. **Fetch Results and Analyze**
```bash
# Fetch evaluation results to local machine
bash scripts/cluster/remote_fetch.sh algebra-ebm

# Run analysis
python projects/algebra-ebm/scripts/analyze_energy_landscapes.py \
    projects/algebra-ebm/results/diagnostic_no_norm/evaluation
```

---

## 🎯 Success Criteria

### IF Energy Landscapes >80% Correct ✓
**Conclusion**: ROOT CAUSE CONFIRMED

**Next**: Full Retraining (T0b)
- Retrain all 5 models without normalization
- Expected: Single-rule 50-85%, Multi-rule 10-30%

### IF Energy Landscapes 60-80% Correct ⚠️
**Conclusion**: PARTIAL FIX

**Next**: Investigate Issue #2 (energy scale parameter learning)

### IF Energy Landscapes <60% Correct ✗
**Conclusion**: HYPOTHESIS REJECTED

**Next**: Investigate alternative root causes

---

## 📁 Files & Documentation

**SLURM Scripts**:
- `slurm/train_diagnostic_no_norm.sbatch` - Training script
- `slurm/eval_diagnostic_no_norm.sbatch` - Evaluation script

**Analysis**:
- `scripts/analyze_energy_landscapes.py` - Energy landscape analyzer

**Documentation**:
- `documentation/deep-dive-analysis.md` - Full technical analysis (11 pages)
- `documentation/CRITICAL-FINDINGS.md` - Executive summary (4 pages)
- `RUN-DIAGNOSTIC.md` - Complete how-to guide
- `DIAGNOSTIC-READY.md` - Preparation summary

**State**:
- `.state/diagnostic-experiment.json` - Experiment tracking

---

## 🔔 Email Notifications

Job completion emails will be sent to: `mkrasnow@fas.harvard.edu`

Both training and evaluation jobs have `--mail-type=END,FAIL` configured.

---

## ⏱️ Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Training | ~3 hours | RUNNING (job 60698745) |
| Evaluation | ~10 min | PENDING |
| Analysis | <1 min | PENDING |
| **Total** | **~3.5 hours** | |

**Expected Completion**: 2026-02-17 08:30 UTC (±30 minutes)

---

## 🆘 Troubleshooting

### If Job Fails Immediately
```bash
# Check error logs
bash scripts/cluster/remote_cmd.sh "cat /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.err"

# Common issues:
# - Module loading failed → Check CUDA/Python versions
# - GPU unavailable → Job will retry or use different node
# - Git clone failed → Check network/permissions
```

### If Job Times Out
```bash
# Check how far it got
bash scripts/cluster/remote_cmd.sh "tail -100 /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.out"

# If close to completion, increase time limit and resubmit
```

### If Model Not Saved
```bash
# Check if training actually completed
bash scripts/cluster/remote_cmd.sh "grep -i 'Training completed' /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_diagnostic_train_60698745.out"

# Check directory permissions
bash scripts/cluster/remote_cmd.sh "ls -ld /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/diagnostic_no_norm"
```

---

**Job submitted at**: 2026-02-17 05:30 UTC
**Current status**: Check with `bash scripts/cluster/status.sh 60698745`
