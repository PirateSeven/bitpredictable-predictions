"""
Free macro and sentiment signals.

Sources (all free, no additional API key needed except CoinGecko key you already have):
  - alternative.me Fear & Greed Index
  - CoinGecko /global (BTC dominance, total market cap change)
  - CoinGecko /coins/{id} (community sentiment votes)
  - CoinDesk / CoinTelegraph RSS feeds (no key needed)
"""


import logging
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_RSS_FEEDS = [
    ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph",  "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine","https://bitcoinmagazine.com/feed"),
]
_RSS_TIMEOUT = 6  # seconds


# ── Fear & Greed ───────────────────────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """
    Returns:
      {"value": int, "classification": str, "timestamp": str}
    Falls back to neutral (50) on any error.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1},
            timeout=8,
            headers={"User-Agent": "bitpredictable/1.0"},
        )
        data = resp.json()
        entry = data["data"][0]
        return {
            "value":          int(entry["value"]),
            "classification": entry["value_classification"],
            "timestamp":      entry.get("timestamp", ""),
        }
    except Exception as e:
        logger.warning(f"[news] Fear & Greed fetch failed: {e}")
        return {"value": 50, "classification": "Neutral", "timestamp": ""}


# ── CoinGecko global market ────────────────────────────────────────────────────

def fetch_global_market() -> dict:
    """
    Returns:
      {
        "btc_dominance":       float,   # current BTC dominance %
        "btc_dominance_delta": float,   # change in dominance (approx, not from API directly)
        "total_mcap_ret_24h":  float,   # total crypto market cap % change 24h
      }
    """
    from pipeline.fetch import _headers, COINGECKO_API_BASE

    _ZERO = {"btc_dominance": 0.0, "btc_dominance_delta": 0.0, "total_mcap_ret_24h": 0.0}
    try:
        resp = requests.get(
            f"{COINGECKO_API_BASE}/global",
            headers=_headers(),
            timeout=10,
        )
        time.sleep(2.1)
        if not resp.ok:
            return _ZERO
        g = resp.json().get("data", {})
        btc_dom = g.get("market_cap_percentage", {}).get("btc", 0.0)
        mcap_chg = g.get("market_cap_change_percentage_24h_usd", 0.0)
        return {
            "btc_dominance":       btc_dom,
            "btc_dominance_delta": 0.0,   # /global doesn't give previous dominance
            "total_mcap_ret_24h":  mcap_chg,
        }
    except Exception as e:
        logger.warning(f"[news] global market fetch failed: {e}")
        return _ZERO


# ── CoinGecko community sentiment ─────────────────────────────────────────────

def fetch_coin_sentiment(coin_id: str) -> Optional[Dict]:
    """
    Returns CoinGecko community data including sentiment_votes_up/down_percentage.
    Returns None on failure (non-fatal).
    """
    from pipeline.fetch import _headers, COINGECKO_API_BASE

    try:
        resp = requests.get(
            f"{COINGECKO_API_BASE}/coins/{coin_id}",
            headers=_headers(),
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "true",
                "developer_data": "false",
            },
            timeout=10,
        )
        time.sleep(2.1)
        if not resp.ok:
            return None
        data = resp.json()
        return {
            "sentiment_votes_up_percentage":   data.get("sentiment_votes_up_percentage"),
            "sentiment_votes_down_percentage": data.get("sentiment_votes_down_percentage"),
            "coingecko_score":                 data.get("coingecko_score"),
        }
    except Exception as e:
        logger.debug(f"[news] sentiment fetch failed for {coin_id}: {e}")
        return None


# ── RSS news headlines ─────────────────────────────────────────────────────────

def fetch_market_headlines(limit: int = 5) -> List[Dict]:
    """
    Fetch recent crypto market headlines from free RSS feeds.
    Returns list of {"title": str, "url": str | None, "source": str}.
    """
    headlines = []
    for source_name, rss_url in _RSS_FEEDS:
        try:
            req = Request(rss_url, headers={"User-Agent": "bitpredictable/1.0"})
            with urlopen(req, timeout=_RSS_TIMEOUT) as f:
                tree = ET.parse(f)
            root = tree.getroot()
            # Handle both RSS 2.0 and Atom
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items[:limit]:
                title = (
                    item.findtext("title")
                    or item.findtext("{http://www.w3.org/2005/Atom}title")
                    or ""
                ).strip()
                link = (
                    item.findtext("link")
                    or item.findtext("{http://www.w3.org/2005/Atom}link")
                    or ""
                ).strip()
                if title:
                    headlines.append({"title": title, "url": link or None, "source": source_name})
        except Exception as e:
            logger.debug(f"[news] RSS {source_name} failed: {e}")

    return headlines[:limit]
