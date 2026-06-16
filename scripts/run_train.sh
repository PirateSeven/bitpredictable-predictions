#!/usr/bin/env bash
# Weekly model retraining — called by cron
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/cron-train-$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) train start ==="

  if [ -f "$VENV/bin/python" ]; then
    PYTHON="$VENV/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi

  echo "Python: $PYTHON"

  cd "$REPO"
  "$PYTHON" -m pipeline.train

  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) train done ==="
} >> "$LOG_FILE" 2>&1
