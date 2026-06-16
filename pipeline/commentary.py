"""
Bilingual (EN/JA) prediction commentary with cited sources.

Primary:  Gemini API (free tier — set GEMINI_API_KEY in .env)
Fallback: Rule-based templates (no API key needed)

Each result includes a `sources` list for frontend citations.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)

CommentaryResult = dict[str, Any]
# {
#   "en": str,
#   "ja": str,
#   "sources": [{"label": str, "value": str, "url": str | None}]
# }


def generate_commentary(
    coin_id: str,
    symbol: str,
    name: str,
    direction: str,
    change_pct_24h: float,
    confidence: float,
    last_features: dict[str, float],
    fear_greed: dict[str, Any],
    global_market: dict[str, float],
    coin_sentiment: dict[str, Any] | None = None,
    news_headlines: list[dict] | None = None,
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

def _gemini_commentary(ctx: dict, headlines: list[dict]) -> tuple[str, str]:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    headlines_text = (
        "\n".join(f"  - {h['title']} ({h.get('source', '')})" for h in headlines[:5])
        if headlines else "  (no recent headlines)"
    )

    prompt = f"""You are a concise crypto market analyst. Generate prediction commentary for {ctx['name']} ({ctx['symbol']}).

Technical signals:
- Predicted 24h direction: {ctx['direction'].upper()}, {ctx['change_pct']:+.1f}% (confidence: {int(ctx['confidence']*100)}%)
- RSI-14: {ctx['rsi']:.0f} — {ctx['rsi_signal']}
- Bollinger Band %B: {ctx['bb']:.2f} — {ctx['bb_signal']} band
- 24h actual return: {ctx['ret_24h']:+.2f}%
- Volume ratio: {ctx['vol_r']:.1f}× average ({ctx['vol_signal']})
- Short-term trend (SMA7/SMA24): {ctx['trend_signal']}

Market context:
- Fear & Greed Index: {ctx['fg_val']} — {ctx['fg_class']}
- Total crypto market cap 24h change: {ctx['mcap_ret']:+.2f}%
- BTC 24h return: {ctx['btc_ret']:+.2f}% ({'aligned with forecast' if ctx['btc_aligned'] else 'diverging from forecast'})
{f"- CoinGecko community: {ctx['sent_up']:.0f}% bullish" if ctx['sent_up'] is not None else ""}

Recent news:
{headlines_text}

Write EXACTLY this format (no extra text, no markdown):
[EN]
(2-3 sentences in English explaining the prediction basis, citing specific signals. Factual, not promotional.)
[JA]
(Same content in natural Japanese.)"""

    response = model.generate_content(prompt)
    return _parse_bilingual(response.text)


def _parse_bilingual(raw: str) -> tuple[str, str]:
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
    feat: dict[str, float], fg: dict, gm: dict, cs: dict | None,
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
    cs: dict | None, headlines: list[dict] | None,
) -> list[dict]:
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
