"""CoinGecko hourly price data fetching with retry and rate limiting."""

import logging
import os
import random
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
SLEEP_BETWEEN_CALLS = 2.1  # Demo key: 30 req/min
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


def _headers() -> dict:
    h = {"accept": "application/json"}
    if COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h


def _fetch_with_retry(url: str, params: dict) -> dict:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=10)

            if resp.ok:
                return resp.json()

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60 * (2 ** attempt)))
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait + random.uniform(0, 2))
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = 5 * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Server error {resp.status_code}. Waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()

        except requests.Timeout:
            if attempt < MAX_RETRIES:
                wait = 5 * (2 ** attempt)
                logger.warning(f"Timeout. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"CoinGecko request failed after {MAX_RETRIES} retries: {url}")


def fetch_coin_list(n: int = 50) -> list[str]:
    """Return top-N coin IDs ordered by market cap."""
    data = _fetch_with_retry(
        f"{COINGECKO_API_BASE}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": n,
            "page": 1,
        },
    )
    ids = [coin["id"] for coin in data]
    logger.info(f"Fetched coin list: {len(ids)} coins")
    return ids


def fetch_hourly(coin_id: str, days: int) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data for a coin.
    Returns DataFrame with columns: [timestamp (UTC), price, volume].
    Gaps are forward/backward filled so the index is always hourly.
    """
    data = _fetch_with_retry(
        f"{COINGECKO_API_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "hourly"},
    )
    time.sleep(SLEEP_BETWEEN_CALLS)

    prices = pd.DataFrame(data["prices"], columns=["ts_ms", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["ts_ms", "volume"])

    df = prices.merge(volumes, on="ts_ms")
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])

    df = _fill_gaps(df)
    logger.debug(f"[{coin_id}] fetched {len(df)} hourly rows")
    return df


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure strict hourly frequency; fill price gaps and zero-fill volume gaps."""
    df = df.set_index("timestamp")
    df = df.resample("1h").first()
    df["price"] = df["price"].ffill().bfill()
    df["volume"] = df["volume"].fillna(0)
    return df.reset_index()
