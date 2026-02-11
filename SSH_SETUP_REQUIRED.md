# SSH Setup Required for Cluster Training

**Status:** ⚠️ NO ACTIVE SSH SESSION DETECTED

Before submitting training jobs to the cluster, you need to establish an SSH connection.

## Quick Status Check

```bash
scripts/cluster/ensure_session.sh
```

**Current Result:**
```
No active SSH control session for mkrasnow@login.rc.fas.harvard.edu.
Run: scripts/cluster/ssh_bootstrap.sh
```

This means the SSH session is **not currently active** and must be established.

## SSH Setup Flow

### 1. Prerequisites ✓

The system has already verified:
- ✓ SSH keys exist in `~/.ssh/`
- ✓ Cluster configuration in `.claude/cluster.local.json`:
  - Host: `login.rc.fas.harvard.edu`
  - User: `mkrasnow`
  - Remote repo: `/n/home03/mkrasnow/research-repo`
  - SSH control path: `~/.ssh/cc-research-repo-%r@%h:%p`

### 2. Establish SSH Connection

**Run this command to establish SSH:**

```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
```

**What this does:**
1. Creates an SSH ControlMaster session
2. Keeps connection alive for 30 minutes (configurable)
3. Enables multiplexing for subsequent commands
4. **Requires interactive login + 2FA**

**Expected output:**
```
Bootstrapping SSH ControlMaster session for mkrasnow@login.rc.fas.harvard.edu ...
Complete login + 2FA interactively when prompted.
SSH session ready.
```

### 3. Verify Connection

After running `ssh_bootstrap.sh`, verify the session is active:

```bash
scripts/cluster/ensure_session.sh
echo $?  # Should return 0 if successful
```

## How It Works

### SSH ControlMaster Configuration

The codebase uses SSH ControlMaster to:
1. **Reduce authentication overhead** - One login, many commands
2. **Keep connection alive** - 30 minute persistence
3. **Enable parallel job submission** - Multiple commands simultaneously
4. **Automatic cleanup** - ControlPersist timeout

### Architecture

```
Local Development Machine
    ↓
    SSH ControlMaster (persistent socket)
    ↓
login.rc.fas.harvard.edu (Harvard cluster login node)
    ↓
SLURM job submission & cluster access
```

### Files Involved

**Configuration:**
```
.claude/cluster.local.json
├── ssh:
│   ├── user: mkrasnow
│   ├── host: login.rc.fas.harvard.edu
│   ├── port: 22
│   ├── control_path: ~/.ssh/cc-research-repo-%r@%h:%p
│   └── control_persist: 30m (keeps connection alive)
└── remote:
    └── repo_root: /n/home03/mkrasnow/research-repo
```

**Scripts:**
```
scripts/cluster/
├── ssh_bootstrap.sh       ← Establish initial connection (interactive 2FA)
├── ensure_session.sh      ← Check if connection exists
├── ssh.sh                 ← Execute commands over SSH
├── submit.sh              ← Submit SLURM jobs
└── remote_submit.sh       ← Remote job submission
```

## Step-by-Step: Complete Training Setup

### Step 1: Establish SSH Connection (One-time, Interactive)

```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
```

You will be prompted for:
- Harvard login password
- 2FA code (phone/email)

After successful login, SSH session remains active for 30 minutes.

### Step 2: Verify Connection

```bash
scripts/cluster/ensure_session.sh
echo $?  # Should output: 0
```

If you get `exit code 21`, the connection dropped and you need to re-run `ssh_bootstrap.sh`.

### Step 3: Submit Training Jobs

Once SSH is active, submit training:

```bash
cd projects/algebra-ebm
bash submit_cluster_training.sh
```

This will:
- Use the active SSH session (no additional 2FA needed)
- Submit 5 jobs to cluster
- Display job IDs
- Create tracking file with `.state/cluster_training.json`

### Step 4: Monitor Training

The SSH session remains active, so you can:

```bash
squeue -u $USER
```

Or login manually:
```bash
ssh cluster.local  # Uses existing control socket
```

## Troubleshooting SSH

### Problem: "Permission denied (publickey,password,keyboard-interactive)"

**Cause:** SSH key or credentials issue

**Solution:**
1. Verify SSH keys exist: `ls -la ~/.ssh/`
2. Check key permissions: `chmod 600 ~/.ssh/id_ed25519`
3. Run bootstrap again: `scripts/cluster/ssh_bootstrap.sh`

### Problem: "No active SSH control session" (Error 21)

**Cause:** Session timed out (default 30 minutes) or connection was closed

**Solution:**
```bash
scripts/cluster/ssh_bootstrap.sh  # Re-establish connection
```

### Problem: "2FA prompt not appearing"

**Cause:** Terminal not in interactive mode

**Solution:**
```bash
# Make sure you're in an interactive terminal
scripts/cluster/ssh_bootstrap.sh
# Don't run in background: background_script.sh &
```

### Problem: "Connection timed out"

**Cause:** Network issue or cluster not accessible

**Solution:**
1. Check internet connection: `ping 8.8.8.8`
2. Try manual SSH: `ssh mkrasnow@login.rc.fas.harvard.edu`
3. Check cluster status website (Harvard RC)

## SSH Session Lifecycle

### Creation
```bash
scripts/cluster/ssh_bootstrap.sh
# Creates: ~/.ssh/cc-research-repo-mkrasnow@login.rc.fas.harvard.edu:22
```

### Usage
All subsequent commands use the control socket:
```bash
scripts/cluster/submit.sh <sbatch_file> <project_slug>
scripts/cluster/ssh.sh "<remote_command>"
```

### Expiration
- **Default:** 30 minutes of inactivity
- **Automatic:** Connection closed, next command will fail with error 21
- **Manual:** Kill manually with `ssh -O exit ...`

### Renewal
Just run bootstrap again:
```bash
scripts/cluster/ssh_bootstrap.sh
```

## Complete Training Workflow with SSH

```bash
# 1. Navigate to repo
cd /Users/mkrasnow/Desktop/research-repo

# 2. FIRST TIME: Establish SSH (interactive, requires 2FA)
scripts/cluster/ssh_bootstrap.sh
# ↓ Enter password
# ↓ Enter 2FA code
# ✓ SSH session ready

# 3. Verify connection
scripts/cluster/ensure_session.sh
# ✓ Should return 0

# 4. Submit training (no additional login needed)
cd projects/algebra-ebm
bash submit_cluster_training.sh
# ✓ 5 jobs submitted with IDs displayed

# 5. Monitor training (uses existing SSH session)
squeue -u $USER
# ✓ Jobs running on GPU nodes

# 6. After ~12 hours, run evaluation
python run_experiments.py
```

## Important Notes

1. **Interactive Login Required**
   - Initial SSH connection requires 2FA
   - Subsequent commands use the established session
   - No additional 2FA prompts needed for 30 minutes

2. **Session Persistence**
   - SSH ControlMaster keeps connection alive
   - Default timeout: 30 minutes
   - If session expires, just run `ssh_bootstrap.sh` again

3. **Network Issues**
   - SSH session tied to specific network connection
   - If you switch networks (WiFi → Ethernet), restart bootstrap
   - SSH session won't work if network changes

4. **Multiple Terminals**
   - All terminals on same machine share the SSH session
   - No need to bootstrap separately in each terminal

## Quick Reference

| Command | Purpose | Requires SSH |
|---------|---------|--------------|
| `scripts/cluster/ssh_bootstrap.sh` | Establish SSH session | ✗ (First time) |
| `scripts/cluster/ensure_session.sh` | Check if connected | ✓ |
| `bash submit_cluster_training.sh` | Submit jobs | ✓ |
| `squeue -u $USER` | Check job status | ✓ |
| `ssh cluster.local` | Manual SSH login | ✓ |

## Next Steps

1. **Now:** `scripts/cluster/ssh_bootstrap.sh` (enter password + 2FA)
2. **Verify:** `scripts/cluster/ensure_session.sh`
3. **Submit:** `cd projects/algebra-ebm && bash submit_cluster_training.sh`
4. **Monitor:** `squeue -u $USER`
5. **Evaluate:** `python run_experiments.py` (after training)

---

**Ready?** Just run:

```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
```

The system will guide you through login + 2FA. Then you can submit training jobs!

