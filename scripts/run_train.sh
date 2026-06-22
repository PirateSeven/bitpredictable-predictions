#!/usr/bin/env bash
# Weekly LSTM retrain runner — called by cron (Sunday 02:00)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/cron-train-$(date +%Y%m%d).log"
PYTHON="$HOME/venv-bitpredictable/bin/python3"

mkdir -p "$LOG_DIR"
cd "$REPO"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) train start ==="

  "$PYTHON" -m pipeline.train
  TRAIN_EXIT=$?
  echo "train.py exited with status ${TRAIN_EXIT}"

  # Fallback git push — runs after the Python process has fully exited (RAM
  # freed), since train.py's own in-process push can fail to fork under the
  # Jetson's tight memory (observed 2026-06-23: training succeeds, push fails
  # with "Cannot allocate memory"). If train.py's CV MAE gate blocked the new
  # model, model.pt is unchanged and this diff check naturally no-ops.
  if ! git diff --quiet HEAD -- model.pt logs/cv_mae_history.jsonl 2>/dev/null; then
    git add model.pt logs/cv_mae_history.jsonl 2>/dev/null || true
    git commit -m "Update model v2 $(date -u +%Y%m%dT%H%M)"
    if git pull --rebase --autostash origin main; then
      git push origin main && echo "Fallback git push succeeded."
    else
      git rebase --abort
      echo "ERROR: rebase failed — aborted. Will retry next cron run."
    fi
  else
    echo "No model.pt/cv_mae_history changes to push (already pushed by train.py, or blocked by CV MAE gate)."
  fi

  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) train done (exit=${TRAIN_EXIT}) ==="
} 2>&1 | tee -a "$LOG_FILE"
