"""
Write per-coin prediction JSON files and push to GitHub.

Output structure (mirrors the Next.js CoinPrediction TypeScript type):
  predictions/{coinId}.json  — one file per coin
  predictions/index.json     — list of available coin IDs

Series format:
  past 7 days  — actualIndex (normalised price) + predictedIndex (backtest)
  next 24 h    — actualIndex=null + predictedIndex (forecast)
"""


import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("predictions")
MODEL_VERSION = "lstm-2.0.0"


# ── Main entry point ───────────────────────────────────────────────────────────

def write_predictions(
    coin_results: List[Dict[str, Any]],
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
    actual_prices: List[float],          # 168 hourly prices (7 days display window)
    actual_times: List[str],             # ISO timestamps for those 168 hours
    backtest_returns_1h: List[float],    # 168 rolling 1h-ahead % return predictions
    future_preds: List[float],           # 24 predicted hourly % returns (future window)
    future_times: List[str],             # ISO timestamps for the next 24 hours
) -> List[Dict]:
    """
    Build a 192-point series (168 actual + 24 future).
    All indexes are normalised so that actual_prices[0] = 100.

    Backtest method: rolling 1h-ahead — each predictedIndex[h] is anchored
    to the previous actual price, avoiding sawtooth resets and error accumulation.
    """
    base_price = actual_prices[0] if actual_prices[0] != 0 else 1.0
    series = []

    # ── Historical 168 hours (rolling 1h-ahead) ───────────────────────────────
    for h in range(len(actual_prices)):
        idx_actual = (actual_prices[h] / base_price) * 100
        if h == 0:
            pred_idx = idx_actual  # anchor start to actual
        else:
            prev_actual = (actual_prices[h - 1] / base_price) * 100
            pred_idx = prev_actual * (1 + backtest_returns_1h[h] / 100)
        series.append({
            "time":           actual_times[h],
            "actualIndex":    round(idx_actual, 4),
            "predictedIndex": round(pred_idx, 4),
        })

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
    series: List[Dict],
    q10_preds: List[float],
    q90_preds: List[float],
) -> Dict:
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
        subprocess.run(["git", "commit", "-m", "Update predictions {}".format(tag)], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logger.info("Pushed predictions to GitHub [%s]", tag)
    except OSError as e:
        logger.warning("git push skipped (insufficient memory to fork): %s", e)
        logger.info("predictions/ written locally — run 'git add predictions/ && git push' manually.")
    except subprocess.CalledProcessError as e:
        logger.error("git push failed: %s", e)
        raise
