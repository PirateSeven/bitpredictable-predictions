#!/usr/bin/env bash
# Prediction runner — called by cron every hour
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/cron-predict-$(date +%Y%m%d).log"
ACE_ENV="$HOME/crypto-ace/.env"
MIN_FREE_MB=400

mkdir -p "$LOG_DIR"

# Load Telegram credentials from crypto-ace .env
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
if [ -f "$ACE_ENV" ]; then
  TELEGRAM_BOT_TOKEN="$(grep '^TELEGRAM_BOT_TOKEN=' "$ACE_ENV" 2>/dev/null | cut -d= -f2- || true)"
  TELEGRAM_CHAT_ID="$(grep '^TELEGRAM_CHAT_ID=' "$ACE_ENV" 2>/dev/null | cut -d= -f2- || true)"
fi

send_telegram() {
  [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ] && return 0
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" > /dev/null || true
}

PREDICT_OK=true

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) predict start ==="

  # ── OOM guard ────────────────────────────────────────────────────────────
  FREE_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  echo "Free memory: ${FREE_MB}MB (min: ${MIN_FREE_MB}MB)"
  if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
    echo "SKIP: insufficient memory"
    PREDICT_OK=false
  else

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
    if ! "$PYTHON" -m pipeline.predict; then
      echo "ERROR: pipeline.predict exited with non-zero status"
      PREDICT_OK=false
    fi

  fi

  # Weekly blog post — runs on Monday (weekday 1) at 09:00 UTC cron
  if [ "$(date -u +%u)" = "1" ] && [ "$(date -u +%H)" = "09" ]; then
    echo "--- weekly blog generation ---"
    if ! "$PYTHON" scripts/generate_blog.py; then
      echo "WARNING: blog generation failed (non-fatal)"
    fi
  fi

  # Fallback git push: runs after Python exits (RAM freed), covers Jetson OOM case
  CHANGED_DIRS=""
  if ! git diff --quiet HEAD -- predictions/ 2>/dev/null; then
    git add predictions/
    CHANGED_DIRS="predictions"
  fi
  if ! git diff --quiet HEAD -- blog/ 2>/dev/null; then
    git add blog/
    CHANGED_DIRS="${CHANGED_DIRS:+$CHANGED_DIRS }blog"
  fi

  if [ "$PREDICT_OK" = true ] && [ -n "$CHANGED_DIRS" ]; then
    git commit -m "Update ${CHANGED_DIRS} $(date -u +%Y%m%dT%H%M)"
    git pull --rebase origin main || { git rebase --abort; echo "ERROR: rebase failed — aborted. Will retry next cron run."; PREDICT_OK=false; }
    if [ "$PREDICT_OK" = true ]; then
      git push origin main
      echo "Fallback git push succeeded (${CHANGED_DIRS})."
    fi
  fi

  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) predict done (ok=${PREDICT_OK}) ==="
} 2>&1 | tee -a "$LOG_FILE"

# Telegram notifications (outside tee block so credentials aren't logged)
if [ "$PREDICT_OK" = false ]; then
  FREE_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  send_telegram "❌ crypto-ace predict FAILED at $(date -u +%Y-%m-%dT%H:%M:%SZ)
Free memory: ${FREE_MB}MB
Check: tail -50 ${LOG_FILE}"
fi
