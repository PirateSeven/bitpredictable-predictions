#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
if [ -x "$HOME/venv-bitpredictable/bin/python3" ]; then
  PYTHON="$HOME/venv-bitpredictable/bin/python3"
fi

"$PYTHON" scripts/update_market_data.py
SCRIPT_EXIT=$?

git add market-data/
if ! git diff --cached --quiet; then
  git commit -m "data: update market data $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git pull --rebase --autostash origin main
  git push origin main
fi

exit "$SCRIPT_EXIT"
