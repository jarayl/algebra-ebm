#!/bin/bash
set -euo pipefail

USER="mkrasnow"
HOST="login.rc.fas.harvard.edu"
REMOTE="${USER}@${HOST}"

# Control socket path for connection sharing
CTRL_PATH="$HOME/.ssh/ctl_%h_%p_%r"

echo "Opening master SSH connection to ${REMOTE}..."
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath="${CTRL_PATH}" \
  -o ControlPersist=600 \
  "${REMOTE}"

echo "Running SCP transfers using shared connection..."

# Training scripts
scp -o ControlPath="${CTRL_PATH}" train_algebra.py                     "${REMOTE}:~/train_algebra.py"
scp -o ControlPath="${CTRL_PATH}" train_algebra_monolithic.py          "${REMOTE}:~/train_algebra_monolithic.py"
scp -o ControlPath="${CTRL_PATH}" run_train_algebra.sh                 "${REMOTE}:~/run_train_algebra.sh"
scp -o ControlPath="${CTRL_PATH}" run_train_algebra_quick.sh            "${REMOTE}:~/run_train_algebra_quick.sh"
scp -o ControlPath="${CTRL_PATH}" run_train_monolithic.sh              "${REMOTE}:~/run_train_monolithic.sh"

# Evaluation scripts  
scp -o ControlPath="${CTRL_PATH}" eval_algebra.py                      "${REMOTE}:~/eval_algebra.py"
scp -o ControlPath="${CTRL_PATH}" run_eval_algebra.sh                  "${REMOTE}:~/run_eval_algebra.sh"
scp -o ControlPath="${CTRL_PATH}" run_comparison_eval.sh               "${REMOTE}:~/run_comparison_eval.sh"

# Source code and scripts directory
scp -r -o ControlPath="${CTRL_PATH}" src                               "${REMOTE}:~/src"
scp -r -o ControlPath="${CTRL_PATH}" scripts                           "${REMOTE}:~/scripts"

echo "Closing master SSH connection..."
ssh -O exit -o ControlPath="${CTRL_PATH}" "${REMOTE}"

echo "All files copied."
