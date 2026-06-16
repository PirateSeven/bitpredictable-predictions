"""
Build predictions JSON and push to GitHub.
Called by predict.py after inference is done.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("predictions")
LATEST_PATH = OUTPUT_DIR / "latest.json"
COINS_PATH  = OUTPUT_DIR / "coins.json"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    kw.setdefault("check", True)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    result = subprocess.run(cmd, **kw)
    return result


def write_predictions(
    coin_results: list[dict[str, Any]],
    generated_at: datetime | None = None,
    cv_mae: float | None = None,
) -> None:
    """
    Write predictions JSON files and push them to GitHub.

    Each coin_result dict must have:
      {
        "coin_id": str,
        "symbol": str,
        "name": str,
        "current_price": float,
        "predicted_returns": {    # median % returns per future hour
            "q10": [float, ...],  # length=HORIZON
            "med": [float, ...],
            "q90": [float, ...],
        }
      }
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    # ── Compute summary stats per coin ────────────────────────────────────────
    coins_summary = []
    for r in coin_results:
        med = r["predicted_returns"]["med"]
        cumulative_return = _cumulative(med)
        direction = "up" if cumulative_return > 0 else "down"
        coins_summary.append({
            "coin_id":          r["coin_id"],
            "symbol":           r["symbol"],
            "name":             r["name"],
            "current_price":    r["current_price"],
            "direction":        direction,
            "cumulative_ret_pct": round(cumulative_return, 4),
            "pred_q10_pct":     round(_cumulative(r["predicted_returns"]["q10"]), 4),
            "pred_q90_pct":     round(_cumulative(r["predicted_returns"]["q90"]), 4),
        })

    # Sort by absolute magnitude of predicted move (most decisive first)
    coins_summary.sort(key=lambda x: abs(x["cumulative_ret_pct"]), reverse=True)

    # ── latest.json — compact summary (loaded by every page) ─────────────────
    latest = {
        "generated_at": generated_at.isoformat(),
        "cv_mae":        round(cv_mae, 4) if cv_mae is not None else None,
        "coins":         coins_summary,
    }
    LATEST_PATH.write_text(json.dumps(latest, ensure_ascii=False, indent=2))
    logger.info(f"Wrote {LATEST_PATH} ({len(coins_summary)} coins)")

    # ── coins.json — per-coin detail including quantile series ────────────────
    coin_detail = []
    for r in coin_results:
        coin_detail.append({
            "coin_id":       r["coin_id"],
            "symbol":        r["symbol"],
            "name":          r["name"],
            "current_price": r["current_price"],
            "predicted_returns": r["predicted_returns"],  # {q10, med, q90} arrays
        })

    COINS_PATH.write_text(json.dumps(coin_detail, ensure_ascii=False, indent=2))
    logger.info(f"Wrote {COINS_PATH}")

    # ── Git push ──────────────────────────────────────────────────────────────
    _git_push(generated_at)


def _cumulative(hourly_rets: list[float]) -> float:
    """Compound hourly % returns into a single 24h cumulative %."""
    factor = 1.0
    for r in hourly_rets:
        factor *= 1 + r / 100
    return (factor - 1) * 100


def _git_push(generated_at: datetime) -> None:
    tag = generated_at.strftime("%Y%m%d-%H%M")
    try:
        _run(["git", "add", str(LATEST_PATH), str(COINS_PATH)])
        result = _run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
        )
        if result.returncode == 0:
            logger.info("No changes to commit — predictions unchanged.")
            return
        _run(["git", "commit", "-m", f"Update predictions {tag}"])
        _run(["git", "push", "origin", "main"])
        logger.info(f"Pushed predictions to GitHub [{tag}]")
    except subprocess.CalledProcessError as e:
        logger.error(f"git push failed: {e.stderr}")
        raise
