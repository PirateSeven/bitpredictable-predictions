"""
Bilingual (EN/JA) prediction commentary with cited sources.

Primary:  Gemini API (free tier — set GEMINI_API_KEY in .env)
Fallback: Rule-based templates (no API key needed)

Each result includes a `sources` list for frontend citations.
"""


import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as _requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Free tier: 15 req/min → minimum 4.5s between calls
_GEMINI_MIN_INTERVAL = 4.5
_gemini_last_call = 0.0


def _gemini_rate_wait() -> None:
    global _gemini_last_call
    elapsed = time.time() - _gemini_last_call
    if elapsed < _GEMINI_MIN_INTERVAL:
        time.sleep(_GEMINI_MIN_INTERVAL - elapsed)
    _gemini_last_call = time.time()

logger = logging.getLogger(__name__)

CommentaryResult = Dict[str, Any]
# {
#   "en": str,
#   "ja": str,
#   "sources": [{"label": str, "value": str, "url": str | None}]
# }

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.0-flash:generateContent"
)


def generate_commentary(
    coin_id: str,
    symbol: str,
    name: str,
    direction: str,
    change_pct_24h: float,
    confidence: float,
    last_features: Dict[str, float],
    fear_greed: Dict[str, Any],
    global_market: Dict[str, float],
    coin_sentiment: Optional[Dict[str, Any]] = None,
    news_headlines: Optional[List[dict]] = None,
) -> CommentaryResult:
    ctx = _build_context(
        coin_id, symbol, name, direction, change_pct_24h, confidence,
        last_features, fear_greed, global_market, coin_sentiment,
    )
    sources = _build_sources(ctx, coin_id, fear_greed, global_market, coin_sentiment, news_headlines)

    if GEMINI_API_KEY:
        try:
            en, ja = _gemini_commentary(ctx, news_headlines or [])
            return {"en": en, "ja": ja, "sources": sources}
        except Exception as e:
            logger.warning(f"[commentary] Gemini API failed for {coin_id}: {e}")

    return {
        "en":      _compose_en(ctx),
        "ja":      _compose_ja(ctx),
        "sources": sources,
    }


# ── Gemini ─────────────────────────────────────────────────────────────────────

def _gemini_commentary(ctx: dict, headlines: list) -> Tuple[str, str]:
    headlines_text = (
        "\n".join("  - {} ({})".format(h["title"], h.get("source", "")) for h in headlines[:5])
        if headlines else "  (no recent headlines)"
    )

    prompt = (
        "You are a concise crypto market analyst. Generate prediction commentary"
        " for {name} ({symbol}).\n\n"
        "Technical signals:\n"
        "- Predicted 24h direction: {direction}, {change_pct:+.1f}% (confidence: {conf}%)\n"
        "- RSI-14: {rsi:.0f} — {rsi_signal}\n"
        "- Bollinger Band %B: {bb:.2f} — {bb_signal} band\n"
        "- 24h actual return: {ret_24h:+.2f}%\n"
        "- Volume ratio: {vol_r:.1f}x average ({vol_signal})\n"
        "- Short-term trend (SMA7/SMA24): {trend_signal}\n\n"
        "Market context:\n"
        "- Fear & Greed Index: {fg_val} — {fg_class}\n"
        "- Total crypto market cap 24h change: {mcap_ret:+.2f}%\n"
        "- BTC 24h return: {btc_ret:+.2f}%\n"
        "{sent_line}\n"
        "Recent news:\n{headlines_text}\n\n"
        "Write EXACTLY this format (no extra text, no markdown):\n"
        "[EN]\n(2-3 sentences in English. Factual, not promotional.)\n"
        "[JA]\n(Same content in natural Japanese.)"
    ).format(
        name=ctx["name"],
        symbol=ctx["symbol"],
        direction=ctx["direction"].upper(),
        change_pct=ctx["change_pct"],
        conf=int(ctx["confidence"] * 100),
        rsi=ctx["rsi"],
        rsi_signal=ctx["rsi_signal"],
        bb=ctx["bb"],
        bb_signal=ctx["bb_signal"],
        ret_24h=ctx["ret_24h"],
        vol_r=ctx["vol_r"],
        vol_signal=ctx["vol_signal"],
        trend_signal=ctx["trend_signal"],
        fg_val=ctx["fg_val"],
        fg_class=ctx["fg_class"],
        mcap_ret=ctx["mcap_ret"],
        btc_ret=ctx["btc_ret"],
        sent_line=(
            "- CoinGecko community: {:.0f}% bullish".format(ctx["sent_up"])
            if ctx["sent_up"] is not None else ""
        ),
        headlines_text=headlines_text,
    )

    for attempt in range(3):
        _gemini_rate_wait()
        resp = _requests.post(
            _GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning("[commentary] Gemini rate limited — waiting %ds (attempt %d/3)", retry_after, attempt + 1)
            time.sleep(retry_after)
            continue
        if not resp.ok:
            raise _requests.HTTPError(
                "{} {} Error: {}".format(resp.status_code, "Client" if resp.status_code < 500 else "Server", resp.reason),
                response=resp,
            )
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_bilingual(text)
    raise _requests.HTTPError("429 Too Many Requests after 3 attempts")


def _parse_bilingual(raw: str) -> Tuple[str, str]:
    en, ja = "", ""
    if "[EN]" in raw and "[JA]" in raw:
        try:
            en = raw.split("[EN]")[1].split("[JA]")[0].strip()
            ja = raw.split("[JA]")[1].strip()
        except IndexError:
            pass
    if not en:
        parts = raw.strip().split("\n\n", 1)
        en = parts[0].strip()
        ja = parts[1].strip() if len(parts) > 1 else en
    return en, ja


# ── Context ────────────────────────────────────────────────────────────────────

def _build_context(
    coin_id: str, symbol: str, name: str, direction: str, change_pct: float, confidence: float,
    feat: Dict[str, float], fg: dict, gm: dict, cs: Optional[dict],
) -> dict:
    rsi    = feat.get("rsi_14", 0.5) * 100
    bb     = feat.get("bb_pct_b", 0.5)
    vol_r  = feat.get("volume_ratio_24h", 1.0)
    btc_rt = feat.get("btc_ret_24h", 0.0)
    sma_r  = feat.get("sma_ratio", 1.0)

    sent_up = cs.get("sentiment_votes_up_percentage") if cs else None

    return dict(
        coin_id=coin_id, symbol=symbol, name=name,
        direction=direction, change_pct=change_pct, confidence=confidence,
        rsi=rsi,
        rsi_signal="overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
        bb=bb,
        bb_signal="upper" if bb > 0.8 else "lower" if bb < 0.2 else "mid",
        ret_24h=feat.get("ret_24h", 0.0),
        vol_r=vol_r,
        vol_signal="high" if vol_r > 1.5 else "low" if vol_r < 0.6 else "normal",
        btc_ret=btc_rt,
        btc_aligned=(btc_rt > 0 and direction == "up") or (btc_rt < 0 and direction == "down"),
        trend_signal="uptrend" if sma_r > 1.01 else "downtrend" if sma_r < 0.99 else "sideways",
        fg_val=fg.get("value", 50),
        fg_class=fg.get("classification", "Neutral"),
        mcap_ret=gm.get("total_mcap_ret_24h", 0.0),
        sent_up=sent_up,
    )


# ── Sources ────────────────────────────────────────────────────────────────────

def _build_sources(
    ctx: dict, coin_id: str, fg: dict, gm: dict,
    cs: Optional[dict], headlines: Optional[list],
) -> List[dict]:
    src = [
        {
            "label": "Fear & Greed Index",
            "value": f"{ctx['fg_val']} — {ctx['fg_class']}",
            "url":   "https://alternative.me/crypto/fear-and-greed-index/",
        },
        {
            "label": "RSI-14",
            "value": f"{ctx['rsi']:.1f} ({ctx['rsi_signal']})",
            "url":   f"https://www.coingecko.com/en/coins/{coin_id}",
        },
        {
            "label": "Bollinger Band %B",
            "value": f"{ctx['bb']:.2f} ({'near upper' if ctx['bb_signal']=='upper' else 'near lower' if ctx['bb_signal']=='lower' else 'mid range'})",
            "url":   None,
        },
        {
            "label": "Volume (24 h ratio)",
            "value": f"{ctx['vol_r']:.1f}× average",
            "url":   None,
        },
        {
            "label": "BTC 24 h return",
            "value": f"{ctx['btc_ret']:+.2f}%",
            "url":   "https://www.coingecko.com/en/coins/bitcoin",
        },
        {
            "label": "Crypto market cap (24 h)",
            "value": f"{gm.get('total_mcap_ret_24h', 0):+.2f}%",
            "url":   "https://www.coingecko.com/en/global-charts",
        },
    ]

    if cs and ctx["sent_up"] is not None:
        src.append({
            "label": "CoinGecko Community Sentiment",
            "value": f"{ctx['sent_up']:.0f}% bullish",
            "url":   f"https://www.coingecko.com/en/coins/{coin_id}",
        })

    for h in (headlines or [])[:3]:
        if h.get("title"):
            src.append({
                "label": h.get("source", "News"),
                "value": h["title"],
                "url":   h.get("url"),
            })

    return src


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _compose_en(c: dict) -> str:
    p = []
    dir_phrase = {"up": f"rise ~{abs(c['change_pct']):.1f}%", "down": f"decline ~{abs(c['change_pct']):.1f}%", "flat": "trade sideways"}[c["direction"]]
    conf = "high" if c["confidence"] > 0.7 else "moderate" if c["confidence"] > 0.5 else "low"
    p.append(f"The model predicts {c['name']} will {dir_phrase} over the next 24 hours with {conf} confidence ({int(c['confidence']*100)}%).")

    if c["rsi_signal"] == "overbought":
        p.append(f"RSI has reached overbought territory ({c['rsi']:.0f}), suggesting elevated selling pressure.")
    elif c["rsi_signal"] == "oversold":
        p.append(f"RSI is oversold ({c['rsi']:.0f}), indicating potential for a mean-reversion bounce.")
    elif c["bb_signal"] == "upper":
        p.append("Price is testing the upper Bollinger Band, a resistance level that often precedes short-term consolidation.")
    elif c["bb_signal"] == "lower":
        p.append("Price is near lower Bollinger Band support, where rebounds are historically more frequent.")
    elif c["vol_signal"] == "high":
        p.append(f"Volume is elevated at {c['vol_r']:.1f}× the daily average, amplifying the directional signal.")
    elif c["trend_signal"] == "uptrend":
        p.append("The short-term SMA has crossed above the medium-term SMA, signalling positive momentum.")
    elif c["trend_signal"] == "downtrend":
        p.append("The short-term SMA is below the medium-term SMA, indicating weakening momentum.")
    else:
        p.append(f"Price moved {c['ret_24h']:+.2f}% in the past 24 hours with volume at {c['vol_r']:.1f}× average.")

    if c["sent_up"] is not None:
        agree = (c["sent_up"] >= 50) == (c["direction"] == "up")
        p.append(
            f"Community sentiment on CoinGecko is {c['sent_up']:.0f}% bullish and "
            f"the Fear & Greed Index reads \"{c['fg_class']}\" ({c['fg_val']}), "
            + ("aligning with the model's forecast." if agree else "diverging from the model's forecast, adding uncertainty.")
        )
    else:
        p.append(
            f"Market sentiment stands at \"{c['fg_class']}\" (Fear & Greed: {c['fg_val']}); "
            f"the broader crypto market moved {c['mcap_ret']:+.1f}% over the past 24 hours."
        )
    return " ".join(p)


def _compose_ja(c: dict) -> str:
    p = []
    dir_ja = {"up": f"約{abs(c['change_pct']):.1f}%上昇", "down": f"約{abs(c['change_pct']):.1f}%下落", "flat": "横ばいで推移"}[c["direction"]]
    conf_ja = "高い" if c["confidence"] > 0.7 else "中程度の" if c["confidence"] > 0.5 else "低い"
    p.append(f"モデルは{c['name']}が今後24時間で{dir_ja}すると予測しています（信頼度: {conf_ja} {int(c['confidence']*100)}%）。")

    if c["rsi_signal"] == "overbought":
        p.append(f"RSIが{c['rsi']:.0f}と買われすぎ水準に達しており、短期的な売り圧力が高まりやすい局面です。")
    elif c["rsi_signal"] == "oversold":
        p.append(f"RSIが{c['rsi']:.0f}と売られすぎ水準で、平均回帰による反発が期待される局面です。")
    elif c["bb_signal"] == "upper":
        p.append("価格がボリンジャーバンド上限に到達しており、短期的な調整が起こりやすい状態です。")
    elif c["bb_signal"] == "lower":
        p.append("価格がボリンジャーバンド下限付近で、サポートからの反発が過去パターンより多く観察されます。")
    elif c["vol_signal"] == "high":
        p.append(f"出来高が平均の{c['vol_r']:.1f}倍と高水準で、方向性シグナルを強化しています。")
    elif c["trend_signal"] == "uptrend":
        p.append("短期移動平均が中期移動平均を上回り、ポジティブなモメンタムが示されています。")
    elif c["trend_signal"] == "downtrend":
        p.append("短期移動平均が中期移動平均を下回り、モメンタムの低下が示されています。")
    else:
        p.append(f"過去24時間のリターンは{c['ret_24h']:+.2f}%、出来高は平均の{c['vol_r']:.1f}倍です。")

    if c["sent_up"] is not None:
        agree = (c["sent_up"] >= 50) == (c["direction"] == "up")
        p.append(
            f"CoinGeckoコミュニティでは{c['sent_up']:.0f}%が強気を示し、"
            f"Fear & Greed指数は「{c['fg_class']}」（{c['fg_val']}）です。"
            + ("予測と方向性が一致しています。" if agree else "予測方向と乖離があり、不確実性が高まっています。")
        )
    else:
        p.append(
            f"Fear & Greed指数は「{c['fg_class']}」（{c['fg_val']}）で、"
            f"暗号資産市場全体の24時間変動率は{c['mcap_ret']:+.1f}%です。"
        )
    return "".join(p)
