"""Per-coin prediction bias correction via EWMA of historical prediction errors.

Flow:
  Run N:   infer → log predictions + prices to cache/pred_log.jsonl
  Run N+1: reconcile (compare N's predictions to current prices) → update bias
           → apply updated bias to N+1's predictions → log N+1
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

ALPHA = 0.3          # EWMA weight for newest error; higher = adapts faster
_MAX_LOG_DAYS = 7    # discard log entries older than this

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_BIAS_PATH  = _CACHE_DIR / "coin_bias.pkl"
_PRED_LOG   = _CACHE_DIR / "pred_log.jsonl"


def load_bias() -> Dict[str, float]:
    """Return {coin_id: bias_%} from disk, or empty dict on first run."""
    if _BIAS_PATH.exists():
        with open(str(_BIAS_PATH), "rb") as fh:
            return pickle.load(fh)
    return {}


def _save_bias(bias: Dict[str, float]) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(str(_BIAS_PATH), "wb") as fh:
        pickle.dump(bias, fh)


def _parse_ts(ts_str: str) -> datetime:
    """Python 3.6-compatible ISO-8601 parse (no dateutil needed)."""
    s = ts_str.replace("Z", "").replace("+00:00", "")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)


def reconcile_and_update_bias(current_prices: Dict[str, float]) -> Dict[str, float]:
    """
    Compare predictions logged ~24h ago against current prices, update
    per-coin EWMA bias, persist to disk, and prune old log entries.
    Returns the updated bias dict (to be applied immediately to this run).
    """
    bias = load_bias()

    if not _PRED_LOG.exists():
        return bias

    now = datetime.now(timezone.utc)
    entries = []
    with open(str(_PRED_LOG), "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass

    reconciled = set()
    kept = []  # entries within _MAX_LOG_DAYS (written back to prune the file)

    for entry in entries:
        try:
            ts = _parse_ts(entry["ts"])
        except (KeyError, ValueError):
            continue

        age_h = (now - ts).total_seconds() / 3600.0

        if age_h > _MAX_LOG_DAYS * 24:
            continue  # prune: do not add to `kept`
        kept.append(entry)

        # Reconcile entries from the ≈24h window (12–36h tolerance)
        if not (12.0 <= age_h <= 36.0):
            continue

        coin       = entry.get("coin")
        price_then = entry.get("price_at_pred")
        pred_24h   = entry.get("pred_24h")
        price_now  = current_prices.get(coin)

        if None in (coin, price_then, pred_24h, price_now) or price_then == 0:
            continue

        actual_24h = (price_now - price_then) / price_then * 100.0
        error      = actual_24h - pred_24h   # positive = we under-predicted

        old_bias       = bias.get(coin, 0.0)
        new_bias       = ALPHA * error + (1.0 - ALPHA) * old_bias
        bias[coin]     = new_bias
        reconciled.add(coin)

        logger.info(
            "[%s] bias: pred=%+.2f%% actual=%+.2f%% err=%+.2f%%  "
            "bias %+.3f%% -> %+.3f%%",
            coin, pred_24h, actual_24h, error, old_bias, new_bias,
        )

    if reconciled:
        _save_bias(bias)
        logger.info("Bias updated for %d coins.", len(reconciled))

    # Rewrite log without expired entries
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(str(_PRED_LOG), "w") as fh:
        for entry in kept:
            fh.write(json.dumps(entry) + "\n")

    return bias


def apply_bias(results: List[Dict], bias: Dict[str, float]) -> None:
    """Correct signal.changePercent24h in-place using per-coin bias.
    Also recalculates confidence from the stored _spread so that bias
    corrections to change_pct propagate into the confidence score."""
    for r in results:
        b = bias.get(r["coin_id"], 0.0)
        if abs(b) < 0.01:
            continue
        old = r["signal"]["changePercent24h"]
        new_pct = old + b
        r["signal"]["changePercent24h"] = new_pct

        # Recalculate confidence using the pre-stored quantile spread
        spread = r["signal"].get("_spread")
        if spread is not None and abs(b) >= 0.01:
            magnitude = max(abs(new_pct), 0.1)
            r["signal"]["confidence"] = round(max(0.1, min(0.95, 1.0 - spread / (magnitude * 4 + 1))), 4)

        logger.info("[%s] bias correction %+.3f%%: %+.2f%% -> %+.2f%%  conf=%.3f",
                    r["coin_id"], b, old, new_pct, r["signal"]["confidence"])


def log_predictions(results: List[Dict], current_prices: Dict[str, float]) -> None:
    """Append today's predictions to pred_log.jsonl for tomorrow's reconciliation."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(str(_PRED_LOG), "a") as fh:
        for r in results:
            coin  = r["coin_id"]
            price = current_prices.get(coin)
            if price is None:
                continue
            fh.write(json.dumps({
                "ts":            ts,
                "coin":          coin,
                "price_at_pred": float(price),
                "pred_24h":      r["signal"]["changePercent24h"],
            }) + "\n")
    logger.info("Logged %d predictions for tomorrow's reconciliation.", len(results))
