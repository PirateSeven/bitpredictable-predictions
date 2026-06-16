"""
Daily prediction entry point.
Run via cron: python pipeline/predict.py

Features:
  - Lock file prevents duplicate execution
  - Cold start: trains model if model.pt is missing
  - Sanity checks on output before pushing
  - Full logging to logs/predict-YYYYMMDD.log
"""

import fcntl
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pipeline.fetch import fetch_coin_list, fetch_hourly
from pipeline.features import N_FEATURES, SEQ_LEN, HORIZON, build_sequences
from pipeline.output import write_predictions

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "model.pt"
LOCK_PATH  = REPO_ROOT / "predict.lock"
LOG_DIR    = REPO_ROOT / "logs"

TOP_N_COINS = 50
FETCH_DAYS  = 7    # only need recent data for inference (SEQ_LEN=48h)

# Sanity thresholds
MAX_VALID_RETURN_PCT   = 50.0   # flag if any coin's 24h median pred > 50%
MIN_COINS_REQUIRED     = 10     # fail if fewer than this many succeed


# ── Logging setup ──────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_file = LOG_DIR / f"predict-{date_str}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.info(f"Log file: {log_file}")


logger = logging.getLogger(__name__)


# ── Lock file ──────────────────────────────────────────────────────────────────
class SingleInstance:
    """Unix advisory lock: raises RuntimeError if another process is running."""

    def __init__(self, lock_path: Path):
        self._path = lock_path
        self._fh = None

    def __enter__(self):
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            raise RuntimeError(
                f"Another predict.py is already running. "
                f"Remove {self._path} manually if it's stale."
            )
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *_):
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
        self._path.unlink(missing_ok=True)


# ── Model loading ──────────────────────────────────────────────────────────────
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(device: torch.device):
    from pipeline.train import QuantileLSTM

    if not MODEL_PATH.exists():
        logger.warning("model.pt not found — running cold start training...")
        _cold_start()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    cfg = checkpoint["model_config"]
    model = QuantileLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,        # disable dropout at inference
        horizon=cfg["horizon"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.n_features_in_ = N_FEATURES
    scaler.feature_names_in_ = None

    cv_mae = checkpoint.get("cv_mae")
    logger.info(
        f"Loaded model (trained_at={checkpoint.get('trained_at')}, "
        f"cv_mae={cv_mae}, version={checkpoint.get('version')})"
    )
    return model, scaler, cv_mae


def _cold_start() -> None:
    """Train model from scratch when model.pt is missing."""
    logger.info("=== Cold start: training LSTM ===")
    from pipeline.train import main as train_main
    train_main()


# ── Sanity checks ──────────────────────────────────────────────────────────────
def _sanity_check(results: list[dict]) -> None:
    issues = []
    for r in results:
        med = r["predicted_returns"]["med"]
        cumret = sum(med)
        if abs(cumret) > MAX_VALID_RETURN_PCT:
            issues.append(
                f"[{r['coin_id']}] implausibly large prediction: {cumret:.1f}%"
            )
        q10 = r["predicted_returns"]["q10"]
        q90 = r["predicted_returns"]["q90"]
        violations = sum(1 for a, b in zip(q10, q90) if a > b)
        if violations > 0:
            issues.append(
                f"[{r['coin_id']}] {violations} q10>q90 monotonicity violations"
            )

    if issues:
        for issue in issues:
            logger.warning(f"Sanity: {issue}")
    else:
        logger.info("Sanity checks passed.")


# ── Main inference ─────────────────────────────────────────────────────────────
def run_inference() -> None:
    device = _get_device()
    logger.info(f"Device: {device}")

    model, scaler, cv_mae = _load_model(device)

    logger.info("Fetching coin list...")
    coin_ids = fetch_coin_list(TOP_N_COINS)

    logger.info("Fetching BTC/ETH market signals...")
    btc_df = fetch_hourly("bitcoin",  FETCH_DAYS)
    eth_df = fetch_hourly("ethereum", FETCH_DAYS)

    results = []
    for coin_id in coin_ids:
        try:
            df = fetch_hourly(coin_id, FETCH_DAYS)

            X = build_sequences(df, btc_df=btc_df, eth_df=eth_df, for_training=False)
            if len(X) == 0:
                logger.warning(f"[{coin_id}] no sequences — skipping")
                continue

            # Use only the most recent sequence (last row = latest state)
            X_last = X[[-1]]
            X_scaled = scaler.transform(X_last.reshape(1, -1)).reshape(1, SEQ_LEN, N_FEATURES)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                out = model(X_tensor).cpu().numpy()   # (1, 3, HORIZON)

            q10 = out[0, 0, :].tolist()
            med = out[0, 1, :].tolist()
            q90 = out[0, 2, :].tolist()

            # Enforce monotonicity at inference time
            q10 = [min(a, b) for a, b in zip(q10, med)]
            q90 = [max(a, b) for a, b in zip(med, q90)]

            current_price = float(df["price"].iloc[-1])

            results.append({
                "coin_id":       coin_id,
                "symbol":        coin_id,    # fetch.py doesn't return symbol; use id as fallback
                "name":          coin_id,
                "current_price": current_price,
                "predicted_returns": {
                    "q10": [round(v, 6) for v in q10],
                    "med": [round(v, 6) for v in med],
                    "q90": [round(v, 6) for v in q90],
                },
            })
            logger.info(f"[{coin_id}] OK — median 24h: {sum(med):.2f}%")

        except Exception as e:
            logger.error(f"[{coin_id}] failed: {e}", exc_info=True)

    if len(results) < MIN_COINS_REQUIRED:
        raise RuntimeError(
            f"Only {len(results)} coins succeeded (minimum {MIN_COINS_REQUIRED}). "
            "Aborting push."
        )

    _sanity_check(results)

    # Enrich symbol/name from coin list metadata if available
    _enrich_metadata(results, coin_ids)

    write_predictions(results, cv_mae=cv_mae)
    logger.info(f"Done. {len(results)} predictions pushed.")


def _enrich_metadata(results: list[dict], coin_ids: list[str]) -> None:
    """CoinGecko /coins/markets includes symbol and name — re-fetch for display."""
    import requests
    from pipeline.fetch import _fetch_with_retry, COINGECKO_API_BASE
    import time

    try:
        data = _fetch_with_retry(
            f"{COINGECKO_API_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": ",".join(coin_ids),
                "per_page": len(coin_ids),
                "page": 1,
            },
        )
        time.sleep(2.1)
        meta = {c["id"]: c for c in data}
        for r in results:
            m = meta.get(r["coin_id"])
            if m:
                r["symbol"] = m.get("symbol", r["coin_id"]).upper()
                r["name"]   = m.get("name",   r["coin_id"])
    except Exception as e:
        logger.warning(f"Metadata enrichment failed (non-fatal): {e}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_logging()
    logger.info("=== predict.py start ===")
    try:
        with SingleInstance(LOCK_PATH):
            run_inference()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    logger.info("=== predict.py done ===")


if __name__ == "__main__":
    main()
