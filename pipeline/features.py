"""
Feature engineering: hourly price/volume DataFrame → LSTM input sequences.
Output shape: X (N, SEQ_LEN, N_FEATURES), y (N, HORIZON)
"""

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import ta as _ta

SEQ_LEN = 48    # hours of lookback (2 days); reduced from 96 for Jetson Nano Maxwell GPU
HORIZON = 24    # hours to predict ahead

FEATURE_NAMES = [
    # price returns (multi-timeframe momentum)
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h", "ret_48h",
    # trend
    "sma_ratio",   # SMA7 / SMA24
    "ema_ratio",   # EMA12 / EMA26
    # momentum
    "rsi_14",
    # volatility
    "bb_pct_b", "bb_width", "rolling_std_24h",
    # volume
    "volume_ratio_24h", "volume_trend_24h",
    # time encoding
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    # market signals (BTC / ETH cross-asset correlation)
    "btc_ret_1h", "btc_ret_6h", "btc_ret_24h",
    "eth_ret_1h", "eth_ret_24h",
]
N_FEATURES = len(FEATURE_NAMES)  # 23


def _price_features(prices: pd.Series) -> pd.DataFrame:
    p = prices

    ret = lambda n: p.pct_change(n) * 100  # noqa: E731

    sma7  = p.rolling(7).mean()
    sma24 = p.rolling(24).mean()
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()

    rsi = _ta.momentum.RSIIndicator(close=p, window=14).rsi()

    _bb = _ta.volatility.BollingerBands(close=p, window=20, window_dev=2)
    bb_upper = _bb.bollinger_hband()
    bb_lower = _bb.bollinger_lband()
    bb_mid   = _bb.bollinger_mavg()
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_pct_b = (p - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    rolling_std = p.pct_change().rolling(24).std() * 100

    return pd.DataFrame({
        "ret_1h":          ret(1),
        "ret_3h":          ret(3),
        "ret_6h":          ret(6),
        "ret_12h":         ret(12),
        "ret_24h":         ret(24),
        "ret_48h":         ret(48),
        "sma_ratio":       sma7 / sma24.replace(0, np.nan),
        "ema_ratio":       ema12 / ema26.replace(0, np.nan),
        "rsi_14":          rsi / 100,   # normalise to [0, 1]
        "bb_pct_b":        bb_pct_b,
        "bb_width":        bb_width,
        "rolling_std_24h": rolling_std,
    }, index=p.index)


def _volume_features(volumes: pd.Series) -> pd.DataFrame:
    v = volumes.replace(0, np.nan)
    avg_24h  = v.rolling(24).mean()
    prev_24h = v.shift(24).rolling(24).mean()

    return pd.DataFrame({
        "volume_ratio_24h": v / avg_24h.replace(0, np.nan),
        "volume_trend_24h": avg_24h / prev_24h.replace(0, np.nan),
    }, index=v.index)


def _time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour    = index.hour
    weekday = index.weekday
    return pd.DataFrame({
        "hour_sin":    np.sin(2 * np.pi * hour / 24),
        "hour_cos":    np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    }, index=index)


def _market_signal_features(
    index,          # pd.DatetimeIndex
    btc_df,         # Optional[pd.DataFrame]
    eth_df,         # Optional[pd.DataFrame]
) -> pd.DataFrame:
    def _ret(df, n):
        if df is None:
            return pd.Series(0.0, index=index)
        s = df.set_index("timestamp")["price"].reindex(index)
        return s.pct_change(n) * 100

    return pd.DataFrame({
        "btc_ret_1h":  _ret(btc_df, 1),
        "btc_ret_6h":  _ret(btc_df, 6),
        "btc_ret_24h": _ret(btc_df, 24),
        "eth_ret_1h":  _ret(eth_df, 1),
        "eth_ret_24h": _ret(eth_df, 24),
    }, index=index)


def build_feature_df(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    eth_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all 23 features for every row of df.
    Returns a DataFrame aligned to df's timestamp index.
    """
    idx     = pd.DatetimeIndex(df["timestamp"])
    prices  = pd.Series(df["price"].values,  index=idx)
    volumes = pd.Series(df["volume"].values, index=idx)

    feat = pd.concat([
        _price_features(prices),
        _volume_features(volumes),
        _time_features(idx),
        _market_signal_features(idx, btc_df, eth_df),
    ], axis=1)[FEATURE_NAMES]

    return feat


def build_sequences(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    eth_df: Optional[pd.DataFrame] = None,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    for_training: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Build LSTM input sequences.

    Training mode: returns (X, y)
      X: (N, seq_len, N_FEATURES)
      y: (N, horizon) — hourly % returns for next `horizon` hours

    Inference mode: returns X only (N, seq_len, N_FEATURES)
    """
    feat   = build_feature_df(df, btc_df, eth_df).values.astype(np.float32)
    prices = df["price"].values

    feat = np.where(np.isfinite(feat), feat, 0.0)

    if for_training:
        max_start = len(feat) - seq_len - horizon
        if max_start <= 0:
            return np.empty((0, seq_len, N_FEATURES), dtype=np.float32), \
                   np.empty((0, horizon), dtype=np.float32)

        X, y = [], []
        for i in range(max_start):
            X.append(feat[i : i + seq_len])
            future_prices  = prices[i + seq_len : i + seq_len + horizon + 1]
            hourly_returns = np.diff(future_prices) / future_prices[:-1] * 100
            y.append(hourly_returns)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    else:
        n = len(feat) - seq_len + 1
        if n <= 0:
            return np.empty((0, seq_len, N_FEATURES), dtype=np.float32)
        return np.array([feat[i : i + seq_len] for i in range(n)], dtype=np.float32)
