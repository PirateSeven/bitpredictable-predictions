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
STATE_FILE="$REPO/data/missing_required_coins.txt"
mkdir -p "$REPO/data"

# `{ ... } | tee` would run the block in a subshell (pipes always fork the
# left side), silently discarding PREDICT_OK/CHANGED_DIRS updates made
# inside — the FAILED alert below never fired as a result. Redirecting to a
# temp file instead keeps the block in the current shell so those variables
# actually propagate.
RUN_LOG_TMP="$(mktemp)"
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

    # trading/log.json に v2 フィールドを追加
    if [ "$PREDICT_OK" = true ] && [ -f "$REPO/trading/log.json" ]; then
      "$PYTHON" "$REPO/trading/patch_log_v2.py" \
        --log "$REPO/trading/log.json" \
        --predictions "$REPO/predictions/" \
        || echo "WARNING: patch_log_v2 failed (non-fatal)"
    fi

  fi

  # Blog post — full weekly recap on Monday (weekday 1), shorter midweek
  # update on Thursday (weekday 4), both at 09:00 UTC. generate_blog.py
  # detects which one to write (see is_midweek_run()) and picks the slug
  # and prompt framing accordingly.
  if { [ "$(date -u +%u)" = "1" ] || [ "$(date -u +%u)" = "4" ]; } && [ "$(date -u +%H)" = "09" ]; then
    echo "--- blog generation ---"
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
  if ! git diff --quiet HEAD -- trading/ 2>/dev/null; then
    git add trading/log.json trading/patch_log_v2.py 2>/dev/null || true
    CHANGED_DIRS="${CHANGED_DIRS:+$CHANGED_DIRS }trading"
  fi
  if ! git diff --quiet HEAD -- blog/ 2>/dev/null; then
    git add blog/
    CHANGED_DIRS="${CHANGED_DIRS:+$CHANGED_DIRS }blog"
  fi

  if [ "$PREDICT_OK" = true ] && [ -n "$CHANGED_DIRS" ]; then
    git commit -m "Update ${CHANGED_DIRS} $(date -u +%Y%m%dT%H%M)"
    git pull --rebase --autostash origin main || { git rebase --abort; echo "ERROR: rebase failed — aborted. Will retry next cron run."; PREDICT_OK=false; }
    if [ "$PREDICT_OK" = true ]; then
      git push origin main
      echo "Fallback git push succeeded (${CHANGED_DIRS})."
    fi
  fi

  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) predict done (ok=${PREDICT_OK}) ==="
} > "$RUN_LOG_TMP" 2>&1
cat "$RUN_LOG_TMP" >> "$LOG_FILE"
RUN_OUTPUT="$(cat "$RUN_LOG_TMP")"
rm -f "$RUN_LOG_TMP"
echo "$RUN_OUTPUT"

# Telegram notifications (outside the redirected block so credentials aren't logged)
if [ "$PREDICT_OK" = false ]; then
  FREE_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  send_telegram "❌ crypto-ace predict FAILED at $(date -u +%Y-%m-%dT%H:%M:%SZ)
Free memory: ${FREE_MB}MB
Check: tail -50 ${LOG_FILE}"
fi

# Alert only on change (not every run) — a coin can stay outside the top-N
# for days (see crypto-ace.md 4.8), and re-sending the same alert hourly
# would just get muted/ignored.
CURRENT_MISSING="$(echo "$RUN_OUTPUT" | grep -o "Adding required coins outside top-[0-9]*: \[.*\]" | tail -1 || true)"
PREV_MISSING=""
[ -f "$STATE_FILE" ] && PREV_MISSING="$(cat "$STATE_FILE")"
if [ "$CURRENT_MISSING" != "$PREV_MISSING" ]; then
  if [ -n "$CURRENT_MISSING" ]; then
    send_telegram "⚠️ crypto-ace predict: coin(s) fell outside CoinGecko's top-N market cap and had to be force-included (REQUIRED_COINS).
${CURRENT_MISSING}
Predictions still work, but worth checking whether these are still valid trading candidates."
  elif [ -n "$PREV_MISSING" ]; then
    send_telegram "✅ crypto-ace predict: previously-missing required coin(s) are back in the top-N market cap ranking.
${PREV_MISSING}"
  fi
fi
echo "$CURRENT_MISSING" > "$STATE_FILE"
