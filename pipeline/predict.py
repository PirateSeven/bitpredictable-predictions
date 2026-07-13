"""
Daily inference entry point.
Run: python pipeline/predict.py

Per-coin steps:
  1. Fetch 14 days of hourly OHLCV (SEQ_LEN=96h lookback + 7-day backtest + buffer)
  2. Build 8 inference windows: 7 backtest (24h each) + 1 future (24h)
  3. Batch forward pass → q10/med/q90 per window
  4. Construct CoinPrediction series + signal
  5. Generate bilingual commentary (Gemini → rule-based fallback)
  6. Write predictions/{coinId}.json + index.json + git push

Safety:
  - Lock file prevents concurrent execution
  - Cold start: trains model if model.pt is missing
  - Sanity checks before pushing
"""

from typing import Dict, List, Optional
import fcntl
from tqdm import tqdm
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pipeline.fetch import fetch_coin_list, fetch_hourly, QuotaExhaustedError
from pipeline.features import N_FEATURES, SEQ_LEN, HORIZON, build_sequences, build_feature_df
from pipeline.news import fetch_fear_greed, fetch_global_market, fetch_coin_sentiment, fetch_market_headlines
from pipeline.commentary import generate_commentary
from pipeline.output import write_predictions, build_series, compute_signal
from pipeline.bias import reconcile_and_update_bias, apply_bias, log_predictions

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "model.pt"
LOCK_PATH  = REPO_ROOT / "predict.lock"
LOG_DIR    = REPO_ROOT / "logs"

TOP_N_COINS      = 50
FETCH_DAYS       = 14   # SEQ_LEN=96h lookback + 7-day display + buffer → need ~264h = 11 days
BACKTEST_WINDOWS = 7    # one 24h window per display day
MIN_RAW_ROWS     = SEQ_LEN + BACKTEST_WINDOWS * HORIZON  # 96 + 168 = 264
MIN_COINS        = 10
MAX_IMPLAUSIBLE  = 50.0  # % — flag if any 24h prediction exceeds this

# crypto-ace が取引対象にしているコイン。CoinGeckoの時価総額トップNランキングから
# 外れても予測を切らさないよう、top-N取得後に強制的に含める（2026-07: polkadotが
# 54位に後退し、trading/config.pyのDOT/BNBだけ9日以上予測が凍結した事故の再発防止）。
# crypto-ace/config.py の `symbols` を変更したらここも合わせて更新すること
REQUIRED_COINS = [
    "binancecoin", "solana", "cardano", "polkadot", "ripple",
    "litecoin", "tron", "hedera-hashgraph", "sui",
]


# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in [
        logging.FileHandler(LOG_DIR / f"predict-{date_str}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        root.addHandler(h)


logger = logging.getLogger(__name__)


# ── Lock file ──────────────────────────────────────────────────────────────────

class SingleInstance:
    def __init__(self, path: Path):
        self._path = path
        self._fh   = None

    def __enter__(self):
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            raise RuntimeError(
                f"Another predict.py is already running. Remove {self._path} if stale."
            )
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *_):
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
        if self._path.exists():
            self._path.unlink()


# ── Device + model loading ─────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def _load_model(device: torch.device):
    from pipeline.train import QuantileLSTM

    if not MODEL_PATH.exists():
        logger.warning("model.pt not found — cold start training...")
        from pipeline.train import main as train_main
        train_main()

    ckpt = torch.load(MODEL_PATH, map_location=device)
    cfg  = ckpt["model_config"]

    if cfg["input_size"] != N_FEATURES:
        raise RuntimeError(
            f"Model input_size={cfg['input_size']} but N_FEATURES={N_FEATURES}. "
            f"Feature set changed — retrain: python pipeline/train.py"
        )

    model = QuantileLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
        horizon=cfg["horizon"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_           = np.array(ckpt["scaler_mean"])
    scaler.scale_          = np.array(ckpt["scaler_scale"])
    scaler.n_features_in_  = N_FEATURES
    scaler.feature_names_in_ = None

    cv_mae = ckpt.get("cv_mae_24h") or ckpt.get("cv_mae")
    logger.info(
        f"Loaded model — trained_at={ckpt.get('trained_at')}, "
        f"cv_24h_mae={cv_mae}, version={ckpt.get('version')}"
    )
    return model, scaler, cv_mae


# ── Inference for one coin ─────────────────────────────────────────────────────

def _infer_coin(
    coin_id: str,
    df,           # full raw DataFrame (≥ MIN_RAW_ROWS rows)
    btc_df,
    eth_df,
    model,
    scaler,
    device: torch.device,
    generated_at: datetime,
) -> Optional[Dict]:
    """
    Returns a dict ready for write_predictions, or None on failure.
    """
    prices     = df["price"].values
    timestamps = df["timestamp"].tolist()

    n = len(prices)
    if n < MIN_RAW_ROWS:
        logger.warning(f"[{coin_id}] only {n} rows, need {MIN_RAW_ROWS} — skipping")
        return None

    # Use the last MIN_RAW_ROWS rows
    prices     = prices[-MIN_RAW_ROWS:]
    timestamps = timestamps[-MIN_RAW_ROWS:]

    # Feature matrix for the entire window (global_market removed from model features)
    sub_df      = df.iloc[-MIN_RAW_ROWS:].copy()
    feat_matrix = build_feature_df(sub_df, btc_df, eth_df).values.astype("float32")
    feat_matrix = np.where(np.isfinite(feat_matrix), feat_matrix, 0.0)

    # Build 169 inference windows:
    #   168 rolling 1h windows for backtest (feat_matrix[h : h+SEQ_LEN], h=0..167)
    #   1 future window (feat_matrix[-SEQ_LEN:])
    # Using 1h-ahead predictions avoids sawtooth resets and error accumulation.
    N_BACKTEST = BACKTEST_WINDOWS * HORIZON  # 168
    windows = []
    for h in range(N_BACKTEST):
        windows.append(feat_matrix[h : h + SEQ_LEN])
    windows.append(feat_matrix[-SEQ_LEN:])  # future

    X_batch  = np.stack(windows)  # (169, SEQ_LEN, N_FEATURES)
    X_scaled = scaler.transform(X_batch.reshape(-1, N_FEATURES)).reshape(
        len(windows), SEQ_LEN, N_FEATURES
    ).astype("float32")

    with torch.no_grad():
        out = model(torch.tensor(X_scaled).to(device)).cpu().numpy()  # (169, 3, HORIZON)

    # Take only the 1h-ahead prediction (index 0) from each rolling backtest window
    backtest_returns_1h = [float(out[h, 1, 0]) for h in range(N_BACKTEST)]
    future_med = out[N_BACKTEST, 1, :].tolist()
    future_q10 = out[N_BACKTEST, 0, :].tolist()
    future_q90 = out[N_BACKTEST, 2, :].tolist()

    # Enforce monotonicity at inference time
    future_q10 = [min(a, b) for a, b in zip(future_q10, future_med)]
    future_q90 = [max(a, b) for a, b in zip(future_med, future_q90)]

    # Display window: last 168 actual prices (= SEQ_LEN + 7*24 - SEQ_LEN ... wait:
    # prices[-MIN_RAW_ROWS:] = prices[0:216], the DISPLAY part is prices[SEQ_LEN:]
    display_prices = list(prices[SEQ_LEN:])           # 168 prices
    display_times  = [
        _iso(ts) for ts in timestamps[SEQ_LEN:]       # 168 ISO timestamps
    ]

    # Future timestamps
    last_ts = timestamps[-1]
    future_times = [_iso_offset(last_ts, h + 1) for h in range(HORIZON)]

    series = build_series(
        actual_prices=display_prices,
        actual_times=display_times,
        backtest_returns_1h=backtest_returns_1h,
        future_preds=future_med,
        future_times=future_times,
    )
    signal = compute_signal(series, future_q10, future_q90)

    # Last feature vector (for commentary)
    last_feat_row = feat_matrix[-1]
    from pipeline.features import FEATURE_NAMES
    last_features = dict(zip(FEATURE_NAMES, last_feat_row.tolist()))

    return {
        "prices":       display_prices,
        "series":       series,
        "signal":       signal,
        "last_features": last_features,
        "future_q10":   future_q10,
        "future_q90":   future_q90,
    }


def _iso(ts) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat().replace("+00:00", "Z")
    return str(ts)


def _iso_offset(ts, hours: int) -> str:
    from pandas import Timestamp
    base = Timestamp(ts) if not hasattr(ts, "hour") else ts
    return (base + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


# ── Sanity check ───────────────────────────────────────────────────────────────

def _sanity_check(results: List[Dict]) -> None:
    for r in results:
        chg = r["signal"]["changePercent24h"]
        if abs(chg) > MAX_IMPLAUSIBLE:
            logger.warning(
                f"[{r['coin_id']}] implausibly large prediction: {chg:+.1f}% — clamping"
            )
            r["signal"]["changePercent24h"] = max(-MAX_IMPLAUSIBLE, min(MAX_IMPLAUSIBLE, chg))
            r["signal"]["confidence"] = min(r["signal"]["confidence"], 0.4)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_inference() -> None:
    generated_at = datetime.now(timezone.utc)
    device = _get_device()
    logger.info(f"Device: {device}")

    model, scaler, cv_mae = _load_model(device)

    # Global signals (fetched once)
    logger.info("Fetching global market signals...")
    fear_greed    = fetch_fear_greed()
    global_market = fetch_global_market()
    market_news   = fetch_market_headlines(limit=5)

    logger.info(
        f"Fear & Greed: {fear_greed['value']} ({fear_greed['classification']}), "
        f"Market cap 24h: {global_market['total_mcap_ret_24h']:+.2f}%"
    )

    # Coin list + market signals
    logger.info("Fetching coin list...")
    coin_ids = fetch_coin_list(TOP_N_COINS)
    missing_required = [c for c in REQUIRED_COINS if c not in coin_ids]
    if missing_required:
        logger.info(f"Adding required coins outside top-{TOP_N_COINS}: {missing_required}")
        coin_ids = coin_ids + missing_required
    btc_df = fetch_hourly("bitcoin",  FETCH_DAYS)
    eth_df = fetch_hourly("ethereum", FETCH_DAYS)

    results = []
    current_prices = {}  # type: Dict[str, float]
    coin_bar = tqdm(coin_ids, desc="Inferring coins", unit="coin", dynamic_ncols=True)
    for coin_id in coin_bar:
        coin_bar.set_postfix(coin=coin_id, ok=len(results))
        try:
            df = fetch_hourly(coin_id, FETCH_DAYS)
            current_prices[coin_id] = float(df["price"].values[-1])

            inferred = _infer_coin(
                coin_id, df, btc_df, eth_df,
                model, scaler, device, generated_at,
            )
            if inferred is None:
                continue

            # Coin-specific sentiment (CoinGecko community votes)
            coin_sentiment = fetch_coin_sentiment(coin_id)

            # Commentary — use a human-readable name until _enrich_metadata runs
            symbol = coin_id.upper()
            name = coin_id.replace("-", " ").title()
            commentary = generate_commentary(
                coin_id=coin_id,
                symbol=symbol,
                name=name,
                direction=inferred["signal"]["direction"],
                change_pct_24h=inferred["signal"]["changePercent24h"],
                confidence=inferred["signal"]["confidence"],
                last_features=inferred["last_features"],
                fear_greed=fear_greed,
                global_market=global_market,
                coin_sentiment=coin_sentiment,
                news_headlines=market_news,
            )

            results.append({
                "coin_id":    coin_id,
                "series":     inferred["series"],
                "signal":     inferred["signal"],
                "commentary": commentary,
            })
            tqdm.write(
                f"[{coin_id}] {inferred['signal']['direction']} "
                f"{inferred['signal']['changePercent24h']:+.2f}%  "
                f"conf={inferred['signal']['confidence']:.2f}"
            )

        except QuotaExhaustedError as e:
            tqdm.write(f"[{coin_id}] {e}")
            logger.error(f"CoinGecko quota exhausted at [{coin_id}], aborting remaining coins: {e}")
            break
        except Exception as e:
            tqdm.write(f"[{coin_id}] failed: {e}")
            logger.error(f"[{coin_id}] failed: {e}", exc_info=True)

    if len(results) < MIN_COINS:
        raise RuntimeError(
            f"Only {len(results)} coins succeeded (minimum {MIN_COINS}). Aborting push."
        )

    # Reconcile yesterday's predictions against today's prices → update bias
    # Then apply the updated bias to today's results before writing
    bias = reconcile_and_update_bias(current_prices)
    apply_bias(results, bias)

    _sanity_check(results)

    # Enrich coin_id → name / symbol from CoinGecko markets
    _enrich_metadata(results, coin_ids)

    write_predictions(results, generated_at_iso=generated_at.isoformat().replace("+00:00", "Z"))

    # Log today's (bias-corrected) predictions for tomorrow's reconciliation
    log_predictions(results, current_prices)
    logger.info(f"Done — {len(results)} predictions pushed.")


def _enrich_metadata(results: List[Dict], coin_ids: List[str]) -> None:
    import time
    from pipeline.fetch import _fetch_with_retry, COINGECKO_API_BASE

    try:
        data = _fetch_with_retry(
            f"{COINGECKO_API_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "ids":       ",".join(coin_ids[:50]),
                "per_page":  len(coin_ids),
                "page":      1,
            },
        )
        time.sleep(2.1)
        meta = {c["id"]: c for c in data}
        for r in results:
            m = meta.get(r["coin_id"])
            if m:
                r["name"]   = m.get("name",   r["coin_id"])
                r["symbol"] = m.get("symbol", r["coin_id"]).upper()
                # Patch commentary to replace the .title() placeholder with the real display name
                if r.get("commentary"):
                    placeholder = r["coin_id"].replace("-", " ").title()
                    real_name = m["name"]
                    if placeholder != real_name:
                        for lang in ("en", "ja"):
                            text = r["commentary"].get(lang, "")
                            if placeholder in text:
                                r["commentary"][lang] = text.replace(placeholder, real_name)
    except Exception as e:
        logger.warning(f"Metadata enrichment failed (non-fatal): {e}")


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
