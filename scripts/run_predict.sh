#!/usr/bin/env bash
# Prediction runner — called by cron every 3 hours
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/cron-predict-$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) predict start ==="

  # Resolve Python: Mac (.venv) → Jetson in-repo venv → Jetson home venv → system python3
  if [ -f "$REPO/.venv/bin/python" ]; then
    PYTHON="$REPO/.venv/bin/python"
  elif [ -f "$REPO/venv-bitpredictable/bin/python3" ]; then
    PYTHON="$REPO/venv-bitpredictable/bin/python3"
  elif [ -f "$HOME/venv-bitpredictable/bin/python" ]; then
    PYTHON="$HOME/venv-bitpredictable/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi

  echo "Python: $PYTHON"
  cd "$REPO"

  # Run inference (writes predictions/ and attempts git push internally)
  "$PYTHON" -m pipeline.predict || true

  # Fallback git push: runs after Python exits (RAM freed), covers Jetson OOM case
  if ! git diff --quiet HEAD -- predictions/ 2>/dev/null; then
    git add predictions/
    git commit -m "Update predictions $(date -u +%Y%m%dT%H%M)"
    git push origin main
    echo "Fallback git push succeeded."
  fi

  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) predict done ==="
} 2>&1 | tee -a "$LOG_FILE"
