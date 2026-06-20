#!/usr/bin/env python3
"""
Weekly blog post generator for bitpredictable-predictions.

Reads current predictions and trading log, generates a bilingual market
analysis post via Groq (free tier), and writes to blog/posts/ + blog/index.json.

Usage:
  python3 scripts/generate_blog.py
  python3 scripts/generate_blog.py --force   # overwrite existing post for today's week
"""

import argparse
import json
import os
import sys
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOG_DIR = REPO / "blog"
POSTS_DIR = BLOG_DIR / "posts"
INDEX_FILE = BLOG_DIR / "index.json"
PREDICTIONS_DIR = REPO / "predictions"
TRADING_LOG = REPO / "trading" / "log.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # better for blog writing; falls back to 8b if quota hit

# Top coins to highlight
TOP_COINS = ["bitcoin", "ethereum", "binancecoin", "solana", "ripple", "dogecoin"]

TAGS = ["weekly", "market-analysis", "crypto", "AI-forecast", "bitcoin"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def week_slug() -> str:
    """Monday of the current week as YYYY-MM-DD."""
    today = datetime.now(timezone.utc).date()
    monday = today - timedelta(days=today.weekday())
    return str(monday)


def gather_context() -> dict:
    """Collect prediction and trading data for the prompt."""
    context = {"predictions": {}, "trading": None, "week": week_slug()}

    for coin_id in TOP_COINS:
        pred = load_json(PREDICTIONS_DIR / f"{coin_id}.json")
        if pred:
            signal = pred.get("signal", {})
            context["predictions"][coin_id] = {
                "direction": signal.get("direction", "flat"),
                "changePercent24h": signal.get("changePercent24h", 0.0),
                "confidence": signal.get("confidence", 0.0),
            }

    log = load_json(TRADING_LOG)
    if log:
        perf = log.get("performance", {})
        context["trading"] = {
            "totalReturnPct": perf.get("totalReturnPct", 0.0),
            "winRate": perf.get("winRate", 0.0),
            "totalTrades": perf.get("totalTrades", 0),
            "sharpeRatio": perf.get("sharpeRatio", 0.0),
            "currentPosition": log.get("currentPosition"),
            "weeklyStats": log.get("weeklyStats"),
        }

    return context


def format_context_text(ctx: dict) -> str:
    lines = [f"Week of: {ctx['week']}", ""]
    lines.append("AI Forecast Summary:")
    for coin_id, pred in ctx["predictions"].items():
        arrow = "↑" if pred["direction"] == "up" else ("↓" if pred["direction"] == "down" else "→")
        lines.append(
            f"  {coin_id}: {arrow} {pred['direction']} "
            f"({pred['changePercent24h']:+.2f}% predicted, confidence {pred['confidence']:.0%})"
        )
    if ctx["trading"]:
        t = ctx["trading"]
        lines.append("")
        lines.append("Trading Agent (crypto-ace) Stats:")
        lines.append(f"  Cumulative return: {t['totalReturnPct']:+.2f}%")
        lines.append(f"  Win rate: {t['winRate']:.0%}")
        lines.append(f"  Total trades: {t['totalTrades']}")
        lines.append(f"  Sharpe ratio: {t['sharpeRatio']:.2f}")
        if t.get("weeklyStats"):
            ws = t["weeklyStats"]
            lines.append(f"  This week: {ws.get('wins', 0)}W / {ws.get('losses', 0)}L")
            comment = ws.get("comment", {})
            if comment.get("en"):
                lines.append(f"  Agent note: {comment['en']}")
    return "\n".join(lines)


def groq_generate(prompt: str, system: str) -> str:
    if not GROQ_API_KEY:
        return ""
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        # Rate limit — fallback to 8b
        if e.code == 429 and GROQ_MODEL != "llama-3.1-8b-instant":
            return groq_generate_model(prompt, system, "llama-3.1-8b-instant")
        print(f"Groq HTTP {e.code}: {body}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Groq error: {e}", file=sys.stderr)
        return ""


def groq_generate_model(prompt: str, system: str, model: str) -> str:
    if not GROQ_API_KEY:
        return ""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq fallback error: {e}", file=sys.stderr)
        return ""


def generate_post_content(ctx: dict) -> dict:
    ctx_text = format_context_text(ctx)
    week = ctx["week"]

    # Determine dominant market direction
    directions = [p["direction"] for p in ctx["predictions"].values()]
    up_count = directions.count("up")
    down_count = directions.count("down")
    if up_count > down_count:
        mood = "bullish"
        mood_ja = "強気"
    elif down_count > up_count:
        mood = "bearish"
        mood_ja = "弱気"
    else:
        mood = "mixed"
        mood_ja = "まちまち"

    # English post
    sys_en = (
        "You are a concise, data-driven crypto market analyst for BitPredictable, "
        "a site that publishes open-book AI trading results. Write clearly and factually. "
        "No hype, no price predictions as investment advice. Always note this is for informational purposes only. "
        "Use markdown headings (## for sections, ### for sub-sections). Write 3-4 sections, ~400 words total."
    )
    prompt_en = (
        f"Write a weekly crypto market analysis blog post for the week of {week}.\n\n"
        f"Market data from the BitPredictable LSTM AI model:\n{ctx_text}\n\n"
        "Structure:\n"
        "## Weekly Overview\n"
        "## Key Signals This Week\n"
        "## Trading Agent Update\n"
        "## What to Watch\n\n"
        "End with a one-line disclaimer that this is not investment advice."
    )

    # Japanese post
    sys_ja = (
        "あなたはBitPredictableのデータ駆動型の暗号資産マーケットアナリストです。"
        "サイトはAIトレードの結果を完全公開しています。明確・簡潔・事実に基づいて書いてください。"
        "誇大表現は禁止。投資助言ではないことを必ず明記。"
        "マークダウン見出し（##）を使い、3〜4セクション、合計350〜450字程度。"
    )
    prompt_ja = (
        f"{week}の週次暗号資産マーケット分析ブログ記事を書いてください。\n\n"
        f"BitPredictable LSTMモデルからの市場データ:\n{ctx_text}\n\n"
        "構成:\n"
        "## 今週の概況\n"
        "## 注目シグナル\n"
        "## トレードエージェントの状況\n"
        "## 来週のチェックポイント\n\n"
        "最後に「投資助言ではありません」と一行追加。"
    )

    body_en = groq_generate(prompt_en, sys_en)
    body_ja = groq_generate(prompt_ja, sys_ja)

    # Fallback if Groq unavailable
    if not body_en:
        body_en = (
            f"## Weekly Overview\n\n"
            f"The week of {week} shows a {mood} market based on BitPredictable's LSTM AI model. "
            f"{up_count} of {len(directions)} tracked coins are forecast to trend upward, "
            f"while {down_count} are forecast to decline.\n\n"
            f"## Key Signals This Week\n\n"
            + "\n".join(
                f"{coin_id}: {'↑' if p['direction']=='up' else ('↓' if p['direction']=='down' else '→')} "
                f"{p['direction']} ({p['changePercent24h']:+.2f}% predicted)"
                for coin_id, p in ctx["predictions"].items()
            )
            + "\n\n## Trading Agent Update\n\n"
            + (
                f"The crypto-ace agent has a cumulative return of {ctx['trading']['totalReturnPct']:+.2f}% "
                f"with a {ctx['trading']['winRate']:.0%} win rate across {ctx['trading']['totalTrades']} trades."
                if ctx["trading"] else "Trading data unavailable this week."
            )
            + "\n\n*This post is for informational purposes only. Not investment advice.*"
        )
    if not body_ja:
        body_ja = (
            f"## 今週の概況\n\n"
            f"{week}の週はBitPredictable LSTMモデルによると{mood_ja}相場です。"
            f"追跡中の{len(directions)}銘柄のうち{up_count}銘柄が上昇、{down_count}銘柄が下落予測です。\n\n"
            f"## 注目シグナル\n\n"
            + "\n".join(
                f"{coin_id}: {'↑' if p['direction']=='up' else ('↓' if p['direction']=='down' else '→')} "
                f"{p['direction']} ({p['changePercent24h']:+.2f}% 予測)"
                for coin_id, p in ctx["predictions"].items()
            )
            + "\n\n## トレードエージェントの状況\n\n"
            + (
                f"crypto-aceエージェントの累積収益率は{ctx['trading']['totalReturnPct']:+.2f}%、"
                f"勝率{ctx['trading']['winRate']:.0%}（{ctx['trading']['totalTrades']}取引）。"
                if ctx["trading"] else "今週の取引データは利用できません。"
            )
            + "\n\n*本記事は情報提供のみを目的としています。投資助言ではありません。*"
        )

    # Extract title from generated content
    title_match_en = re.match(r"^#{1,2}\s+(.+)", body_en)
    title_match_ja = re.match(r"^#{1,2}\s+(.+)", body_ja)
    title_en = (
        title_match_en.group(1) if title_match_en
        else f"Weekly Crypto Market Analysis — {week}"
    )
    title_ja = (
        title_match_ja.group(1) if title_match_ja
        else f"週次暗号資産マーケット分析 — {week}"
    )
    # Remove leading ## title from body if present (it becomes the page heading)
    if title_match_en:
        body_en = body_en[len(title_match_en.group(0)):].strip()
    if title_match_ja:
        body_ja = body_ja[len(title_match_ja.group(0)):].strip()

    # Summary: first non-empty paragraph
    def first_para(text: str) -> str:
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]
        return text[:200]

    return {
        "slug": f"weekly-{week}",
        "title": {"en": title_en, "ja": title_ja},
        "date": week,
        "summary": {
            "en": first_para(body_en),
            "ja": first_para(body_ja),
        },
        "body": {"en": body_en, "ja": body_ja},
        "tags": TAGS,
        "coins": [c for c in TOP_COINS if c in ctx["predictions"]],
    }


def update_index(post_meta: dict):
    index = load_json(INDEX_FILE) or {"posts": []}
    posts = index.get("posts", [])
    # Replace existing entry with same slug
    posts = [p for p in posts if p["slug"] != post_meta["slug"]]
    meta = {
        "slug": post_meta["slug"],
        "title": post_meta["title"],
        "date": post_meta["date"],
        "summary": post_meta["summary"],
        "tags": post_meta["tags"],
        "coins": post_meta["coins"],
    }
    posts.insert(0, meta)
    # Keep last 52 posts (1 year)
    posts = posts[:52]
    index["posts"] = posts
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing post for this week")
    args = parser.parse_args()

    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    slug = f"weekly-{week_slug()}"
    post_file = POSTS_DIR / f"{slug}.json"

    if post_file.exists() and not args.force:
        print(f"Post already exists: {post_file} (use --force to regenerate)")
        sys.exit(0)

    print(f"Gathering market context...")
    ctx = gather_context()
    print(f"Predictions loaded: {len(ctx['predictions'])} coins")
    print(f"Trading data: {'yes' if ctx['trading'] else 'no'}")

    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set — using template fallback (no AI generation)")

    print(f"Generating blog post '{slug}'...")
    post = generate_post_content(ctx)

    post_file.write_text(json.dumps(post, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote: {post_file}")

    update_index(post)
    print(f"Updated: {INDEX_FILE}")
    print(f"Done. Title (en): {post['title']['en']}")


if __name__ == "__main__":
    main()
