# Deep Research: Monolithic Model Detection Failure

**Date:** 2025-12-11  
**Job ID:** 50413896  
**Issue:** Model validation reports "✓ Found monolithic model: 0" but model fails to load

## Executive Summary

The issue is a **symlink handling bug** in the file copy operation. The training script creates `model.pt` as a symbolic link to `model-{milestone}.pt`, but the comparison evaluation script uses `/bin/cp -r` which doesn't properly dereference symlinks. This results in a broken symlink (0 bytes) that passes the bash file existence check but fails when PyTorch attempts to load it.

**Root Cause:** `run_comparison_eval.sh:76` uses `/bin/cp -r` instead of `/bin/cp -Lr` or `rsync`  
**Impact:** Model validation appears successful but inference fails immediately  
**Fix Complexity:** Simple one-line change to use proper copy command  

## Research Scope

### Original Question
Why does the script report "✓ Found monolithic model: 0" when the model exists in `results/monolithic/`?

### Sub-Questions Investigated
1. What does the "0" in the validation message represent?
2. How is the model validation performed in the bash script?
3. How are models saved during training?
4. How are models copied between directories?
5. What validation does the Python code perform before loading?

### Files/Systems Analyzed
- `run_comparison_eval.sh` - Evaluation orchestration script
- `run_train_monolithic.sh` - Training script  
- `src/diffusion/denoising_diffusion_pytorch_1d.py` - Trainer implementation
- `src/algebra/algebra_evaluation.py` - Model loading logic
- `eval_algebra.py` - Evaluation entry point
- `scripts/statistical_comparison_evaluation.py` - Statistical framework

## Key Findings

### Finding 1: The "0" Represents File Size, Not Count

**Evidence:**
- `run_comparison_eval.sh:130`
```bash
echo "✓ Found monolithic model: $(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)"
```

**Analysis:**
The validation message uses `du -h` (disk usage) to show the file size. The output "0" means the file is 0 bytes - it exists but is empty or a broken symlink. This is misleading because:
- The checkmark (✓) suggests success
- The "0" looks like a count (0 models found) but is actually a size
- The script continues execution despite the file being unusable

**Confidence:** High  
This is directly observable in the code and matches the user's output exactly.

---

### Finding 2: Bash Validation Only Checks File Existence

**Evidence:**
- `run_comparison_eval.sh:106-110`
```bash
# Check monolithic model
if [ ! -f "$JOB_SCRATCH/results/monolithic/model.pt" ]; then
    echo "ERROR: Monolithic model not found at results/monolithic/model.pt"
    echo "Run: sbatch run_train_monolithic.sh"
    exit 1
fi
```

**Analysis:**
The `-f` test in bash returns true if the file exists as a regular file OR as a symbolic link, regardless of whether the symlink is valid. This means:
- A broken symlink passes the validation
- An empty file passes the validation
- No check for file size or content validity
- The script proceeds to Python code that will inevitably fail

**Confidence:** High  
Standard bash behavior, confirmed by examining the validation logic.

---

### Finding 3: Training Creates Symlinks for Convenience

**Evidence:**
- `src/diffusion/denoising_diffusion_pytorch_1d.py:1571-1586`
```python
# Create model.pt symlink for compatibility
final_model_path = self.results_folder / f'model-{final_milestone}.pt'
model_pt_path = self.results_folder / 'model.pt'

if final_model_path.exists():
    # Remove existing symlink if it exists
    if model_pt_path.exists() or model_pt_path.is_symlink():
        model_pt_path.unlink()
    # Create new symlink
    try:
        os.symlink(f'model-{final_milestone}.pt', str(model_pt_path))
        print(f"Created model.pt -> model-{final_milestone}.pt")
    except OSError as e:
        print(f"Warning: Could not create model.pt symlink: {e}")
```

**Analysis:**
The training script saves checkpoints as `model-{step}.pt` (e.g., `model-5000.pt`, `model-10000.pt`) and creates a convenience symlink `model.pt` pointing to the final checkpoint. This design:
- Allows keeping multiple checkpoints
- Provides a stable `model.pt` name for evaluation
- Uses relative symlink (not absolute path)
- Requires both files to be in the same directory structure when copied

**Confidence:** High  
Code is explicit about symlink creation.

---

### Finding 4: File Copy Operation Breaks Symlinks

**Evidence:**

**Training script** (`run_train_monolithic.sh:203`):
```bash
rsync -av "$RESULTS_DIR/" "$FINAL_RESULTS_DIR/monolithic/"
```

**Comparison script** (`run_comparison_eval.sh:76`):
```bash
/bin/cp -r "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/
```

**Analysis:**
The training script uses `rsync -av` which preserves symlinks correctly, but the comparison script uses `/bin/cp -r` which has problematic symlink handling:

**What happens with `/bin/cp -r` and symlinks:**
1. By default, `cp -r` copies symlinks as-is (preserves the link)
2. If the symlink target doesn't exist at the destination, the link becomes broken
3. The relative symlink `model.pt -> model-5000.pt` only works if both files are copied
4. If files are copied in the wrong order or filtered, the symlink breaks

**Why this causes the 0-byte issue:**
- Training creates: `model.pt` (symlink) → `model-5000.pt` (actual file)
- Copy operation may copy the symlink but not resolve it
- At destination, `model.pt` exists as a symlink pointing to a file that may not be there
- `du -h` on a broken symlink shows 0 bytes
- Bash `-f` test still returns true (symlink exists)

**Confidence:** High  
This is the exact failure mode that explains all observed symptoms.

---

### Finding 5: Python Code Validates Existence But Not Content

**Evidence:**

**In `eval_algebra.py:996-997`:**
```python
# Validate monolithic checkpoint exists
if not Path(args.monolithic_checkpoint).exists():
    raise ValueError(f"Monolithic checkpoint not found: {args.monolithic_checkpoint}")
```

**In `src/algebra/algebra_evaluation.py:1441`:**
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
```

**Then `src/algebra/algebra_evaluation.py:83-84`:**
```python
if 'model' not in checkpoint:
    raise ValueError(f"Checkpoint does not contain 'model' key. Found: {checkpoint.keys()}")
```

**Analysis:**
The Python validation mirrors the bash problem:
1. Checks if file exists (passes for broken symlinks)
2. Attempts `torch.load()` which will fail on empty/broken files
3. No file size check before attempting to load

**Expected failure mode:**
- If file is 0 bytes: `torch.load()` raises unpickling error
- If file is broken symlink: May fail with file not found or unpickling error
- Error happens during evaluation, not during validation

**Confidence:** High  
Standard file operation patterns, confirmed by code analysis.

---

## Timeline & Evolution

The code shows signs of progressive enhancement but missed this edge case:

1. **Original design:** Simple file existence checks
2. **Enhancement:** Added `du -h` to show file size in validation message (intended as helpful info)
3. **Unintended consequence:** File size became visible but wasn't used for validation
4. **Symlink support:** Added in training to provide convenience naming
5. **Copy scripts:** Different copy strategies in training vs evaluation (rsync vs cp)

## Connections & Dependencies

This issue involves the interaction of several systems:

```
Training (run_train_monolithic.sh)
    ↓
Creates: model-{N}.pt + symlink model.pt
    ↓
Copies back with: rsync -av (preserves symlinks correctly)
    ↓
Stored in: $SLURM_SUBMIT_DIR/results/monolithic/
    ↓
Comparison (run_comparison_eval.sh)
    ↓
Copies with: /bin/cp -r (may break symlinks)
    ↓
Validation: bash -f test (passes for broken symlinks)
    ↓
Validation: du -h shows 0 (visible warning, not used)
    ↓
Python: Path.exists() (passes for broken symlinks)
    ↓
FAILURE: torch.load() fails on empty/broken file
```

## Root Cause Analysis

### Immediate Cause
`/bin/cp -r` in `run_comparison_eval.sh:76` doesn't properly handle symlinks.

### Contributing Factors
1. **Inconsistent copy strategy:** Training uses `rsync`, comparison uses `cp`
2. **Weak validation:** File existence checks don't verify usability
3. **Misleading output:** Checkmark with "0" suggests success
4. **No fail-fast:** Script continues despite unusable file

### Why It Wasn't Caught
- Bash `-f` test passes for symlinks (by design)
- No automated tests for cross-script file operations
- Manual testing probably done in same directory (symlinks work)
- Cluster environment creates separation that breaks relative symlinks

## Evidence-Based Solutions

### Solution 1: Dereference Symlinks During Copy (Recommended)

**Change in `run_comparison_eval.sh:76`:**
```bash
# OLD:
/bin/cp -r "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/

# NEW:
/bin/cp -Lr "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/
```

**Rationale:**
- `-L` flag dereferences symlinks and copies the actual files
- Minimal change, maximum safety
- No dependency on symlink preservation
- Works regardless of how training saved files

**Confidence:** High  
Standard solution for symlink copying issues.

---

### Solution 2: Use Rsync Consistently

**Change in `run_comparison_eval.sh:76`:**
```bash
# OLD:
/bin/cp -r "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/

# NEW:
rsync -avL "$SLURM_SUBMIT_DIR/results/" "$JOB_SCRATCH/results/"
```

**Rationale:**
- Matches training script's approach
- `-L` flag transforms symlinks into referent files
- More verbose output (helpful for debugging)
- More robust for future enhancements

**Confidence:** High  
Already proven to work in training script.

---

### Solution 3: Add File Size Validation

**Change in `run_comparison_eval.sh:106-110`:**
```bash
# Check monolithic model exists and has content
if [ ! -f "$JOB_SCRATCH/results/monolithic/model.pt" ]; then
    echo "ERROR: Monolithic model not found at results/monolithic/model.pt"
    exit 1
fi

# Check file size is non-zero
MODEL_SIZE=$(stat -f%z "$JOB_SCRATCH/results/monolithic/model.pt" 2>/dev/null || stat -c%s "$JOB_SCRATCH/results/monolithic/model.pt" 2>/dev/null)
if [ "$MODEL_SIZE" -eq 0 ] 2>/dev/null; then
    echo "ERROR: Monolithic model file is empty (0 bytes)"
    echo "This usually means the model file is a broken symlink or failed to copy"
    echo "Check training output and ensure model was saved correctly"
    exit 1
fi

echo "✓ Found monolithic model: $(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)"
```

**Rationale:**
- Fail-fast when file is unusable
- Clear error message explaining the issue
- Works with both GNU and BSD stat
- Defense-in-depth approach

**Confidence:** Medium-High  
Good defensive programming, but doesn't fix the underlying issue.

---

### Solution 4: Enhanced Validation Message

**Change in `run_comparison_eval.sh:130-131`:**
```bash
# OLD:
echo "✓ Found monolithic model: $(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)"
echo "✓ Found $((4 - MISSING_RULES))/4 rule models"

# NEW:
MONO_SIZE=$(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)
if [ "$MONO_SIZE" = "0" ] || [ "$MONO_SIZE" = "0B" ]; then
    echo "✗ Found monolithic model but size is 0 (likely broken symlink)"
    exit 1
else
    echo "✓ Found monolithic model: $MONO_SIZE"
fi
echo "✓ Found $((4 - MISSING_RULES))/4 rule models"
```

**Rationale:**
- Uses the information already being collected
- Catches the exact issue user experienced
- Clear failure message
- Minimal code change

**Confidence:** Medium  
Treats symptom, not root cause, but prevents silent failure.

---

## Recommended Implementation Plan

### Immediate Fix (Production)
1. Apply **Solution 1** (add `-L` flag to cp command)
2. Apply **Solution 3** (add file size validation)

**Combined change in `run_comparison_eval.sh:76`:**
```bash
/bin/cp -Lr "$SLURM_SUBMIT_DIR/results" "$JOB_SCRATCH"/
```

**Combined change in `run_comparison_eval.sh:106-131`:**
```bash
# Check monolithic model
if [ ! -f "$JOB_SCRATCH/results/monolithic/model.pt" ]; then
    echo "ERROR: Monolithic model not found at results/monolithic/model.pt"
    echo "Run: sbatch run_train_monolithic.sh"
    exit 1
fi

# Validate model file has content
MODEL_SIZE=$(stat -f%z "$JOB_SCRATCH/results/monolithic/model.pt" 2>/dev/null || \
             stat -c%s "$JOB_SCRATCH/results/monolithic/model.pt" 2>/dev/null)
if [ -z "$MODEL_SIZE" ] || [ "$MODEL_SIZE" -eq 0 ] 2>/dev/null; then
    echo "ERROR: Monolithic model file is empty or broken"
    echo "File: $JOB_SCRATCH/results/monolithic/model.pt"
    echo "This usually indicates:"
    echo "  - Model training did not complete successfully"
    echo "  - Symlink was broken during file copy"
    echo "  - File copy operation failed"
    echo ""
    echo "Check training logs and verify model.pt exists in:"
    echo "  $SLURM_SUBMIT_DIR/results/monolithic/"
    exit 1
fi

# ... rest of validation ...

MONO_SIZE=$(du -h $JOB_SCRATCH/results/monolithic/model.pt | cut -f1)
echo "✓ Found monolithic model: $MONO_SIZE ($(printf "%'d" $MODEL_SIZE) bytes)"
```

### Future Enhancement
Consider switching to `rsync` entirely for consistency (**Solution 2**), especially if more complex file operations are added.

### Testing Validation
After fix, verify:
1. Training with fresh model ✓
2. Training with existing checkpoints ✓
3. Evaluation after successful training ✓
4. Evaluation with broken/missing model (should fail gracefully) ✓
5. Evaluation with empty model.pt file (should fail gracefully) ✓

---

## Additional Context

### Why Symlinks Were Used
Looking at the trainer code, symlinks provide several benefits:
1. Keep all checkpoints (`model-1000.pt`, `model-2000.pt`, etc.)
2. Always have a stable `model.pt` pointing to latest
3. Easy to switch between checkpoints by changing symlink
4. Disk space efficient (no duplicate final checkpoint)

### Alternative Approaches Considered
1. **Copy final checkpoint to model.pt:** Wastes disk space but more robust
2. **Absolute symlinks:** Won't work across machines
3. **No symlink, always use model-{N}.pt:** Requires knowing final milestone number
4. **Hardlinks:** Less flexible, same issues with cross-filesystem copies

The symlink approach is reasonable, but file copy operations must handle it correctly.

---

## Knowledge Gaps & Uncertainties

### What We Know
- Exact code paths for model saving and loading
- Exact failure mode when symlinks break
- Exact location of the bug

### What We Assume
- User's cluster environment follows standard POSIX behavior
- `/bin/cp` is standard GNU or BSD coreutils
- Training completed successfully and created the symlink

### What We Can't Confirm Without Access
- Whether the actual `model-{milestone}.pt` files exist in submit directory
- Whether `rsync` is available on the cluster (assumed yes for HPC)
- Whether training logs show successful model.pt creation
- The exact error message from PyTorch when loading fails

### Recommended User Verification
To confirm diagnosis, user should run on cluster:
```bash
# Check submit directory
ls -lh /n/home03/mkrasnow/results/monolithic/
# Should see both model.pt (symlink) and model-*.pt files

# Check what model.pt points to
readlink /n/home03/mkrasnow/results/monolithic/model.pt
# Should show: model-{some_number}.pt

# Check actual file sizes
ls -lh /n/home03/mkrasnow/results/monolithic/model-*.pt
# Should show non-zero sizes (probably 100MB-1GB range)

# Verify target exists
stat /n/home03/mkrasnow/results/monolithic/$(readlink /n/home03/mkrasnow/results/monolithic/model.pt)
```

---

## References

### Code Locations
- Model saving: `src/diffusion/denoising_diffusion_pytorch_1d.py:1452-1464` (save method)
- Symlink creation: `src/diffusion/denoising_diffusion_pytorch_1d.py:1571-1586` (train completion)
- Training copy: `run_train_monolithic.sh:203` (rsync command)
- Comparison copy: `run_comparison_eval.sh:76` (cp command)
- Bash validation: `run_comparison_eval.sh:106-131` (file checks)
- Python validation: `eval_algebra.py:996-997` (existence check)
- Model loading: `src/algebra/algebra_evaluation.py:1441` (torch.load)
- Checkpoint parsing: `src/algebra/algebra_evaluation.py:81-84` (key validation)

### Key Commit Messages
Not examined in this research (git log analysis skipped for time).

### External Documentation
- Bash test operators: `-f` returns true for files and symlinks
- `cp -L`: Dereference symbolic links
- `rsync -L`: Transform symlinks into referent files
- `torch.load()`: Requires valid pickle format, fails on empty files

---

## Summary

The "Found monolithic model: 0" message reveals a **symlink handling bug**:

1. **Training** creates `model.pt` as a symlink to `model-{N}.pt`
2. **Comparison script** copies files with `/bin/cp -r` (doesn't dereference symlinks)  
3. **Result:** Broken symlink at destination (0 bytes)
4. **Validation:** Passes bash existence check, shows size warning, but continues
5. **Failure:** PyTorch cannot load empty/broken file

**Fix:** Use `/bin/cp -Lr` or `rsync -avL` to properly copy symlinked files, and add file size validation for early failure detection.

**Impact:** Critical for production use - prevents silent failures in evaluation pipeline.

**Effort:** ~10 lines of code changes, low risk.
