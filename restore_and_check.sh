#!/bin/bash
# =============================================================================
# TEKO Restore & Sanity Check Script
# =============================================================================
# This script:
# 1. Backs up your current changes
# 2. Restores the stable S25 version
# 3. Runs the sanity check
#
# Usage:
#   chmod +x restore_and_check.sh
#   ./restore_and_check.sh
# =============================================================================

set -e  # Exit on error

TEKO_DIR="${TEKO_DIR:-/home/schux00/teko}"
BACKUP_DIR="${TEKO_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
STABLE_COMMIT="1bc67de0ce54a3dea9d3d36d4f40b0141c6df930"

echo "=============================================="
echo "  TEKO Restore & Sanity Check"
echo "=============================================="
echo ""
echo "TEKO directory: $TEKO_DIR"
echo "Stable commit:  $STABLE_COMMIT"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check if TEKO directory exists
# -----------------------------------------------------------------------------
if [ ! -d "$TEKO_DIR" ]; then
    echo "ERROR: TEKO directory not found at $TEKO_DIR"
    echo "Set TEKO_DIR environment variable to your teko path"
    exit 1
fi

cd "$TEKO_DIR"

# -----------------------------------------------------------------------------
# Step 2: Check for uncommitted changes
# -----------------------------------------------------------------------------
echo "[1/5] Checking Git status..."

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo ""
    echo "⚠️  You have uncommitted changes!"
    echo ""
    read -p "Do you want to stash them? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git stash push -m "Auto-stash before restore $(date)"
        echo "✅ Changes stashed"
    else
        echo "Aborting. Please commit or stash your changes first."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 3: Create backup of current state
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Creating backup at $BACKUP_DIR..."

mkdir -p "$BACKUP_DIR"
cp -r source/teko/teko/tasks/direct/teko "$BACKUP_DIR/"
cp -r scripts "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ Backup created"

# -----------------------------------------------------------------------------
# Step 4: Restore stable version
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Restoring stable version ($STABLE_COMMIT)..."

# Restore key files
FILES_TO_RESTORE=(
    "source/teko/teko/tasks/direct/teko/teko_env.py"
    "source/teko/teko/tasks/direct/teko/teko_env_cfg.py"
    "source/teko/teko/tasks/direct/teko/robots/teko.py"
    "source/teko/teko/tasks/direct/teko/robots/teko_static.py"
    "source/teko/teko/tasks/direct/teko/rewards/reward_functions.py"
    "source/teko/teko/tasks/direct/teko/curriculum/curriculum_manager.py"
)

for file in "${FILES_TO_RESTORE[@]}"; do
    if git show "$STABLE_COMMIT:$file" > /dev/null 2>&1; then
        echo "  Restoring $file..."
        git checkout "$STABLE_COMMIT" -- "$file"
    else
        echo "  ⚠️  File not found in stable commit: $file"
    fi
done

echo "✅ Stable files restored"

# -----------------------------------------------------------------------------
# Step 5: Verify no merge conflicts
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Checking for merge conflicts..."

CONFLICT_COUNT=$(grep -r "<<<<<<< " source/teko/teko/tasks/direct/teko/ 2>/dev/null | wc -l || echo "0")

if [ "$CONFLICT_COUNT" -gt 0 ]; then
    echo "❌ Found $CONFLICT_COUNT merge conflict markers!"
    grep -r "<<<<<<< " source/teko/teko/tasks/direct/teko/ || true
    exit 1
else
    echo "✅ No merge conflicts found"
fi

# -----------------------------------------------------------------------------
# Step 6: Print summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  RESTORE COMPLETE"
echo "=============================================="
echo ""
echo "✅ Stable version restored"
echo "✅ Backup saved to: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Copy teko_sanity_check.py to your scripts directory"
echo "  2. Run: python scripts/teko_sanity_check.py --test all"
echo "  3. Or run interactively in Isaac Sim"
echo ""
echo "To undo and restore your changes:"
echo "  git stash pop  # If you stashed"
echo "  # Or copy from backup: $BACKUP_DIR"
echo ""