#!/usr/bin/env python3
"""Fetch CoinGecko top-100 markets (USD & JPY) and write market-data/*.json.

Runs on a fixed schedule (Jetson cron), independent of site traffic, so the
CoinGecko quota is consumed at a predictable rate regardless of how many
people visit bitpredictable.com.
"""
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "market-data"
BASE_URL = "https://api.coingecko.com/api/v3/coins/markets"
PER_PAGE = 100
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")  # optional, same demo key as the web app


def fetch_markets(vs_currency: str) -> list:
    params = (
        f"?vs_currency={vs_currency}&order=market_cap_desc"
        f"&per_page={PER_PAGE}&page=1&price_change_percentage=1h,24h,7d"
    )
    req = urllib.request.Request(BASE_URL + params)
    req.add_header("accept", "application/json")
    req.add_header("user-agent", "bitpredictable.com/1.0 (+https://bitpredictable.com)")
    if COINGECKO_API_KEY:
        req.add_header("x-cg-demo-api-key", COINGECKO_API_KEY)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def map_coin(raw: dict) -> dict:
    pct24h = raw.get("price_change_percentage_24h_in_currency")
    if pct24h is None:
        pct24h = raw.get("price_change_percentage_24h")
    return {
        "id": raw["id"],
        "symbol": raw["symbol"],
        "name": raw["name"],
        "image": raw["image"],
        "currentPrice": raw["current_price"],
        "marketCap": raw["market_cap"],
        "marketCapRank": raw.get("market_cap_rank"),
        "totalVolume": raw["total_volume"],
        "priceChangePercentage1h": raw.get("price_change_percentage_1h_in_currency"),
        "priceChangePercentage24h": pct24h,
        "priceChangePercentage7d": raw.get("price_change_percentage_7d_in_currency"),
        "lastUpdated": raw["last_updated"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for currency in ("usd", "jpy"):
        try:
            raw_coins = fetch_markets(currency)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"[update_market_data] {currency} fetch failed: {e}")
            ok = False
            continue

        result = {
            "coins": [map_coin(c) for c in raw_coins],
            "stale": False,
            "fetchedAt": datetime.now(timezone.utc).isoformat(),
        }
        out_path = OUT_DIR / f"markets-{currency}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[update_market_data] wrote {out_path} ({len(result['coins'])} coins)")
        time.sleep(2)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
