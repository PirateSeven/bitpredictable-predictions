"""
Write per-coin prediction JSON files and push to GitHub.

Output structure (mirrors the Next.js CoinPrediction TypeScript type):
  predictions/{coinId}.json  — one file per coin
  predictions/index.json     — list of available coin IDs

Series format:
  past 7 days  — actualIndex (normalised price) + predictedIndex (backtest)
  next 24 h    — actualIndex=null + predictedIndex (forecast)
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("predictions")
MODEL_VERSION = "lstm-2.0.0"


# ── Main entry point ───────────────────────────────────────────────────────────

def write_predictions(
    coin_results: list[dict[str, Any]],
    generated_at_iso: str,
    model_version: str = MODEL_VERSION,
) -> None:
    """
    coin_results: list of dicts with keys:
      coin_id, symbol, name, series, signal, commentary
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    coin_ids = []

    for r in coin_results:
        cid = r["coin_id"]
        payload = {
            "coinId":       cid,
            "generatedAt":  generated_at_iso,
            "modelVersion": model_version,
            "series":       r["series"],     # [{time, actualIndex, predictedIndex}]
            "signal":       r["signal"],     # {direction, changePercent24h, confidence}
            "commentary":   r.get("commentary"),  # {en, ja, sources} | null
        }
        path = OUTPUT_DIR / f"{cid}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        coin_ids.append(cid)

    index_path = OUTPUT_DIR / "index.json"
    index_path.write_text(json.dumps(coin_ids, ensure_ascii=False, indent=2))

    logger.info(f"Wrote {len(coin_ids)} prediction files to {OUTPUT_DIR}/")
    _git_push(generated_at_iso)


# ── Series helpers ─────────────────────────────────────────────────────────────

def build_series(
    actual_prices: list[float],       # 168 hourly prices (7 days display window)
    actual_times: list[str],          # ISO timestamps for those 168 hours
    backtest_preds: list[list[float]], # 7 × 24 predicted hourly % returns (from 7 backtest windows)
    future_preds: list[float],        # 24 predicted hourly % returns (future window)
    future_times: list[str],          # ISO timestamps for the next 24 hours
) -> list[dict]:
    """
    Build a 192-point series (168 actual + 24 future).
    All indexes are normalised so that actual_prices[0] = 100.
    """
    base_price = actual_prices[0] if actual_prices[0] != 0 else 1.0

    series = []

    # ── Historical 168 hours ──────────────────────────────────────────────────
    for day in range(7):
        # Ground the predicted block to the actual index at the start of this 24h chunk
        anchor_actual = (actual_prices[day * 24] / base_price) * 100
        preds_24h = backtest_preds[day]  # 24 hourly % returns

        # compound starts at anchor; apply each return AFTER recording the current hour
        # so predictedIndex[h] aligns with actualIndex[h] at the same timestamp
        compound = anchor_actual
        for h in range(24):
            idx_actual = (actual_prices[day * 24 + h] / base_price) * 100
            series.append({
                "time":           actual_times[day * 24 + h],
                "actualIndex":    round(idx_actual, 4),
                "predictedIndex": round(compound, 4),
            })
            compound *= 1 + preds_24h[h] / 100

    # ── Future 24 hours ────────────────────────────────────────────────────────
    last_actual_idx = (actual_prices[-1] / base_price) * 100
    compound = last_actual_idx
    for h in range(24):
        compound *= 1 + future_preds[h] / 100
        series.append({
            "time":           future_times[h],
            "actualIndex":    None,
            "predictedIndex": round(compound, 4),
        })

    return series


def compute_signal(
    series: list[dict],
    q10_preds: list[float],
    q90_preds: list[float],
) -> dict:
    """
    Derive direction, changePercent24h, and confidence from the series and quantiles.
    """
    # Forecast is the last 24 points where actualIndex is None
    forecast = [p for p in series if p["actualIndex"] is None]
    if not forecast:
        return {"direction": "flat", "changePercent24h": 0.0, "confidence": 0.5}

    last_actual = next(
        (p["actualIndex"] for p in reversed(series) if p["actualIndex"] is not None),
        100.0,
    )
    final_pred = forecast[-1]["predictedIndex"]
    change_pct = (final_pred - last_actual) / last_actual * 100

    direction: str
    if change_pct > 0.5:
        direction = "up"
    elif change_pct < -0.5:
        direction = "down"
    else:
        direction = "flat"

    # Confidence: inverse of quantile spread relative to predicted magnitude
    if q10_preds and q90_preds:
        spread = sum(abs(hi - lo) for hi, lo in zip(q90_preds, q10_preds)) / len(q10_preds)
        magnitude = max(abs(change_pct), 0.1)
        # Tighter spread relative to magnitude = higher confidence
        confidence = max(0.1, min(0.95, 1.0 - spread / (magnitude * 4 + 1)))
    else:
        confidence = 0.5

    return {
        "direction":        direction,
        "changePercent24h": round(change_pct, 4),
        "confidence":       round(confidence, 4),
    }


# ── Git push ───────────────────────────────────────────────────────────────────

def _git_push(generated_at_iso: str) -> None:
    tag = generated_at_iso[:16].replace(":", "").replace("T", "-")
    try:
        subprocess.run(["git", "add", str(OUTPUT_DIR)], check=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], capture_output=True
        )
        if diff.returncode == 0:
            logger.info("No prediction changes to commit.")
            return
        subprocess.run(["git", "commit", "-m", f"Update predictions {tag}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logger.info(f"Pushed predictions to GitHub [{tag}]")
    except subprocess.CalledProcessError as e:
        logger.error(f"git push failed: {e}")
        raise
