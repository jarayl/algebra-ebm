#!/bin/bash
# Submit all 5 algebra EBM training jobs to the cluster
# Models will be trained in parallel on GPU nodes
# Results will be stored on cluster at /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_SLUG="algebra-ebm"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Submitting Algebra EBM Training Jobs to Cluster          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Repository: $REPO_ROOT"
echo "Project: $PROJECT_SLUG"
echo "Timestamp: $(date)"
echo ""

# Get current git SHA
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
echo "Current git SHA: $GIT_SHA"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Submit each training job
echo "Submitting training jobs..."
echo "────────────────────────────────────────────────────────────"

rules=("distribute" "combine" "isolate" "divide")
for rule in "${rules[@]}"; do
    sbatch_file="$REPO_ROOT/projects/$PROJECT_SLUG/slurm/train_${rule}.sbatch"

    if [ ! -f "$sbatch_file" ]; then
        echo "❌ Error: SBATCH file not found: $sbatch_file"
        exit 1
    fi

    echo "Submitting: train_${rule}.sbatch"

    # Submit via cluster submit script (follows CLAUDE.md pattern)
    job_id="$("$REPO_ROOT/scripts/cluster/submit.sh" "$sbatch_file" "$PROJECT_SLUG")"

    if [ -z "$job_id" ]; then
        echo "❌ Failed to submit train_${rule}.sbatch"
        exit 1
    fi

    JOB_IDS+=("$job_id")
    echo "  ✓ Submitted with Job ID: $job_id"
done

# Submit monolithic job (longer partition since more data)
echo "Submitting: train_monolithic.sbatch (gpu partition)"
sbatch_file="$REPO_ROOT/projects/$PROJECT_SLUG/slurm/train_monolithic.sbatch"

if [ ! -f "$sbatch_file" ]; then
    echo "❌ Error: SBATCH file not found: $sbatch_file"
    exit 1
fi

job_id="$("$REPO_ROOT/scripts/cluster/submit.sh" "$sbatch_file" "$PROJECT_SLUG")"

if [ -z "$job_id" ]; then
    echo "❌ Failed to submit train_monolithic.sbatch"
    exit 1
fi

JOB_IDS+=("$job_id")
echo "  ✓ Submitted with Job ID: $job_id"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All training jobs submitted successfully!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Job Summary:"
echo "  Distribute:    Job ID ${JOB_IDS[0]}"
echo "  Combine:       Job ID ${JOB_IDS[1]}"
echo "  Isolate:       Job ID ${JOB_IDS[2]}"
echo "  Divide:        Job ID ${JOB_IDS[3]}"
echo "  Monolithic:    Job ID ${JOB_IDS[4]}"
echo ""

# Create a tracking file
TRACKING_FILE="$REPO_ROOT/projects/$PROJECT_SLUG/.state/cluster_training.json"
mkdir -p "$(dirname "$TRACKING_FILE")"

cat > "$TRACKING_FILE" << EOF
{
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_sha": "$GIT_SHA",
  "job_ids": {
    "distribute": "${JOB_IDS[0]}",
    "combine": "${JOB_IDS[1]}",
    "isolate": "${JOB_IDS[2]}",
    "divide": "${JOB_IDS[3]}",
    "monolithic": "${JOB_IDS[4]}"
  },
  "status": "submitted",
  "notes": "Models will be trained on cluster and stored at /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/"
}
EOF

echo "Tracking file saved: $TRACKING_FILE"
echo ""

# Show how to monitor
echo "To monitor job status:"
echo "  squeue -u \$USER"
echo ""
echo "To check job details:"
echo "  squeue -j <job_id>"
echo ""
echo "To view logs after completion:"
echo "  ls -la projects/algebra-ebm/slurm/logs/"
echo ""
echo "Models will be automatically copied to cluster storage:"
echo "  /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/"
echo ""
echo "After training completes, run evaluations with:"
echo "  cd projects/algebra-ebm"
echo "  python run_experiments.py"
echo ""
