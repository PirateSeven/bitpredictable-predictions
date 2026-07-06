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
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAS_REQUESTS = False

REPO = Path(__file__).resolve().parent.parent

# Auto-load .env from repo root if python-dotenv is available, else parse manually
_env_file = REPO / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
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


NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]


def fetch_news(limit_per_feed=5, total_limit=6):
    """CoinDesk / Cointelegraph の公開RSSから直近の見出しを取得する。
    失敗しても記事生成自体は止めない（空リストを返すだけ）。"""
    items = []
    for source, url in NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (BitPredictableBot)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for it in root.findall("./channel/item")[:limit_per_feed]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if title and link:
                    items.append({"title": title, "link": link, "source": source})
        except Exception as e:
            print(f"News fetch error ({source}): {e}", file=sys.stderr)
    return items[:total_limit]


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


def is_midweek_run() -> bool:
    """True on any day the run_predict.sh cron trigger fires that isn't
    Monday — currently just Thursday. A midweek run publishes a shorter
    supplementary update instead of a second "weekly" post for the same
    week (which would either collide with Monday's slug or misrepresent
    itself as a second full weekly recap)."""
    return datetime.now(timezone.utc).isoweekday() != 1


def gather_context() -> dict:
    """Collect prediction and trading data for the prompt."""
    context = {
        "predictions": {},
        "trading": None,
        "week": week_slug(),
        "midweek": is_midweek_run(),
    }

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
            "currentPositions": log.get("currentPositions", []),
            "weeklyStats": log.get("weeklyStats"),
        }

    context["news"] = fetch_news()
    context["previous_opening"] = get_previous_post_opening()
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
    lines.append("")
    lines.append("Valid coin ids for linking (use exactly as shown): " + ", ".join(ctx["predictions"].keys()))
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
    if ctx.get("news"):
        lines.append("")
        lines.append("Recent crypto news headlines (use exact titles/urls, do not alter):")
        for n in ctx["news"]:
            lines.append(f"  - [{n['source']}] {n['title']} — {n['link']}")
    return "\n".join(lines)


def _groq_post(model: str, system: str, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    if _HAS_REQUESTS:
        resp = _requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code == 429 and model != "llama-3.1-8b-instant":
            return _groq_post("llama-3.1-8b-instant", system, prompt, max_tokens)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    else:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def groq_generate(prompt: str, system: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        return _groq_post(GROQ_MODEL, system, prompt, 1500)
    except Exception as e:
        print(f"Groq error: {e}", file=sys.stderr)
        return ""


def groq_generate_model(prompt: str, system: str, model: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        return _groq_post(model, system, prompt, 1200)
    except Exception as e:
        print(f"Groq fallback error: {e}", file=sys.stderr)
        return ""


COIN_DISPLAY = {
    "bitcoin":     ("Bitcoin", "ビットコイン"),
    "ethereum":    ("Ethereum", "イーサリアム"),
    "binancecoin": ("BNB", "BNB"),
    "solana":      ("Solana", "Solana"),
    "ripple":      ("Ripple", "リップル"),
    "dogecoin":    ("Dogecoin", "ドージコイン"),
}


def auto_link_coins(text: str, coin_ids, lang: str) -> str:
    """Groqがリンク指示に従わなかった場合の保険。本文中の最初のコイン名出現を /coins/{id} にリンクする。"""
    for coin_id in coin_ids:
        names = COIN_DISPLAY.get(coin_id)
        if not names:
            continue
        if "/coins/" + coin_id + ")" in text:
            continue
        name = names[0] if lang == "en" else names[1]
        flags = re.IGNORECASE if lang == "en" else 0
        pattern = re.compile((r"\b" + re.escape(name) + r"\b") if lang == "en" else re.escape(name), flags)
        m = pattern.search(text)
        if m:
            matched = text[m.start():m.end()]
            text = text[:m.start()] + "[" + matched + "](/coins/" + coin_id + ")" + text[m.end():]
    return text


def append_news_section(body: str, news: list, lang: str) -> str:
    """SEO目的の外部リンクを確実に入れるための保険。Groqが本文中で触れなかった
    見出しだけを末尾に追加する — 既に本文中でリンク済みのものを重ねて出すと
    生成記事特有の「機械的な繰り返し」感が出るため、リンク済みは除外する。"""
    if not news:
        return body
    remaining = [n for n in news if n["link"] not in body]
    if not remaining:
        return body
    header = "## In the News" if lang == "en" else "## 関連ニュース"
    parts = [header]
    for n in remaining[:3]:
        parts.append(f"{n['title']} ([{n['source']}]({n['link']}))")
    return body.rstrip() + "\n\n" + "\n\n".join(parts)


def get_previous_post_opening() -> dict:
    """直近の記事の書き出し文（英日）を返す。同じ導入パターンの繰り返しを
    避けるため、プロンプトに「これとは違う書き出しにしろ」という否定例として渡す。"""
    index = load_json(INDEX_FILE) or {"posts": []}
    posts = index.get("posts", [])
    if not posts:
        return {"en": None, "ja": None}
    prev_slug = posts[0]["slug"]
    prev_file = POSTS_DIR / f"{prev_slug}.json"
    prev = load_json(prev_file)
    if not prev:
        return {"en": None, "ja": None}

    def first_sentence(text: str) -> str:
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Cut at the first sentence boundary so the negative example
                # stays short — we only need the opening move, not the whole para.
                m = re.search(r"[.!?。]", line)
                return line[: m.end()] if m else line[:160]
        return ""

    return {
        "en": first_sentence(prev.get("body", {}).get("en", "")),
        "ja": first_sentence(prev.get("body", {}).get("ja", "")),
    }


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

    prev_en = ctx.get("previous_opening", {}).get("en")
    prev_ja = ctx.get("previous_opening", {}).get("ja")
    midweek = ctx.get("midweek", False)
    today_str = str(datetime.now(timezone.utc).date())

    # English post
    sys_en = (
        "You are a concise, data-driven crypto market analyst for BitPredictable, "
        "a site that publishes open-book AI trading results. Write clearly and factually. "
        "No hype, no price predictions as investment advice. Always note this is for informational purposes only. "
        "Use markdown headings (## for sections, ### for sub-sections). Write 3-4 sections, ~550-650 words total — "
        "use the extra length for texture and specifics, not padding. The first time you mention a tracked coin by "
        "name, link it using markdown: [Bitcoin](/coins/bitcoin) — use only the exact coin ids listed in the data, "
        "never invent a slug. Link each coin at most once per post. Write like a sharp, specific human analyst, not "
        "generic AI summary text. Never use cliches like \"in today's fast-paced market\", \"it's important to "
        "note\", \"navigate the landscape\", \"in the world of crypto\", \"looking ahead\", \"in summary\", "
        "\"overall\", or \"as always\". Avoid formulaic transitions (moreover, furthermore, in conclusion). Do not "
        "start more than one sentence in the whole post with the same word (especially \"The\" or \"This\"). Lean "
        "on the specific numbers given rather than vague language, and take a clear point of view grounded in the "
        "data instead of neutrally listing both sides. Vary sentence length — mix short, blunt sentences with "
        "longer ones. If a relevant recent news headline is provided in the data, you may reference it naturally "
        "with a markdown link, e.g. [headline text](url) — only if genuinely relevant, do not force it, and never "
        "just restate a headline verbatim without saying why it matters."
    )
    opening_constraint_en = (
        f"\n\nDo not open with a sentence shaped like last week's: \"{prev_en}\" — start this post differently. "
        "Lead with a specific number, a specific coin's move, a contrarian read, or a direct claim — not a general "
        "restatement that \"the market is expected to [mood]\"."
        if prev_en else ""
    )
    if midweek:
        prompt_en = (
            f"Write a short midweek crypto market update for {today_str}, a supplementary post published between "
            f"the regular Monday weekly posts. This is NOT a full weekly recap — assume the reader already saw "
            f"Monday's post and only wants what's new or has shifted since then. Keep it to 2-3 sections, "
            f"~300-350 words total.\n\n"
            f"Market data from the BitPredictable LSTM AI model:\n{ctx_text}\n\n"
            "Structure:\n"
            "## Midweek Update\n"
            "## What Changed\n"
            "## Trading Agent Update\n\n"
            "End with a one-line disclaimer that this is not investment advice."
            + opening_constraint_en
        )
    else:
        prompt_en = (
            f"Write a weekly crypto market analysis blog post for the week of {week}.\n\n"
            f"Market data from the BitPredictable LSTM AI model:\n{ctx_text}\n\n"
            "Structure:\n"
            "## Weekly Overview\n"
            "## Key Signals This Week\n"
            "## Trading Agent Update\n"
            "## What to Watch\n\n"
            "End with a one-line disclaimer that this is not investment advice."
            + opening_constraint_en
        )

    # Japanese post
    sys_ja = (
        "あなたはBitPredictableのデータ駆動型の暗号資産マーケットアナリストです。"
        "サイトはAIトレードの結果を完全公開しています。明確・簡潔・事実に基づいて書いてください。"
        "誇大表現は禁止。投資助言ではないことを必ず明記。"
        "マークダウン見出し（##）を使い、3〜4セクション、合計500〜650字程度 — 増やした分は具体性・深みに使い、水増しはしない。"
        "追跡中のコイン名を初めて言及する際は、[Bitcoin](/coins/bitcoin) のようにmarkdownリンクにしてください。"
        "データに記載された正確なcoin idのみ使用し、推測で作らないこと。1つのコインにつき記事内で1回までリンク。"
        "「急速に変化する市場」「〜することが重要です」「暗号資産の世界では」「今後の展望として」「総じて」"
        "のような決まり文句や、AIの要約っぽい無難な言い回しは禁止。"
        "記事全体を通して、同じ書き出しの単語（「〜は」「今週は」等）を2文以上続けて使わないこと。"
        "データの具体的な数字を使い、両論併記で終わらせず、データに基づいた明確な見立てを書くこと。"
        "文の長さにも変化をつけること — 短く言い切る文と、長く説明する文を混ぜる。"
        "提供されたニュース見出しの中に関連性の高いものがあれば、[見出し](url)のようにmarkdownリンクで自然に触れてよい。"
        "無理にこじつけないこと。見出しをただ言い換えるだけでなく、なぜそれが重要かを書くこと。"
    )
    opening_constraint_ja = (
        f"\n\n先週の書き出し「{prev_ja}」と同じパターンの書き出しにしないこと。"
        "具体的な数字、特定のコインの値動き、逆張りの視点、明確な主張のいずれかから始めること。"
        "「今週の市場は〜見込みです」のような一般的な言い換えで始めないこと。"
        if prev_ja else ""
    )
    if midweek:
        prompt_ja = (
            f"{today_str}向けの短い「週半ばアップデート」記事を書いてください。これは月曜の週次記事と週次記事の間に"
            f"公開する補足記事です。週次の総括ではなく、読者は既に月曜の記事を読んでいる前提で、"
            f"月曜以降に変化した点・新しく出てきた点だけを書いてください。2〜3セクション、合計300〜350字程度。\n\n"
            f"BitPredictable LSTMモデルからの市場データ:\n{ctx_text}\n\n"
            "構成:\n"
            "## 週半ばアップデート\n"
            "## 変化のポイント\n"
            "## トレードエージェントの状況\n\n"
            "最後に「投資助言ではありません」と一行追加。"
            + opening_constraint_ja
        )
    else:
        prompt_ja = (
            f"{week}の週次暗号資産マーケット分析ブログ記事を書いてください。\n\n"
            f"BitPredictable LSTMモデルからの市場データ:\n{ctx_text}\n\n"
            "構成:\n"
            "## 今週の概況\n"
            "## 注目シグナル\n"
            "## トレードエージェントの状況\n"
            "## 来週のチェックポイント\n\n"
            "最後に「投資助言ではありません」と一行追加。"
            + opening_constraint_ja
        )

    body_en = groq_generate(prompt_en, sys_en)
    body_ja = groq_generate(prompt_ja, sys_ja)

    # Fallback if Groq unavailable
    fallback_heading_en = "## Midweek Update" if midweek else "## Weekly Overview"
    fallback_heading_ja = "## 週半ばアップデート" if midweek else "## 今週の概況"
    fallback_intro_en = (
        f"Since Monday, BitPredictable's LSTM AI model shows a {mood} tilt: "
        if midweek
        else f"The week of {week} shows a {mood} market based on BitPredictable's LSTM AI model. "
    )
    fallback_intro_ja = (
        f"月曜以降、BitPredictableのLSTMモデルは{mood_ja}寄りの動きを示しています。"
        if midweek
        else f"{week}の週はBitPredictable LSTMモデルによると{mood_ja}相場です。"
    )
    if not body_en:
        body_en = (
            f"{fallback_heading_en}\n\n"
            f"{fallback_intro_en}"
            f"{up_count} of {len(directions)} tracked coins are forecast to trend upward, "
            f"while {down_count} are forecast to decline.\n\n"
            f"## Key Signals\n\n"
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
            f"{fallback_heading_ja}\n\n"
            f"{fallback_intro_ja}"
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

    body_en = auto_link_coins(body_en, ctx["predictions"].keys(), "en")
    body_ja = auto_link_coins(body_ja, ctx["predictions"].keys(), "ja")
    body_en = append_news_section(body_en, ctx.get("news", []), "en")
    body_ja = append_news_section(body_ja, ctx.get("news", []), "ja")

    # Extract title from generated content
    title_match_en = re.match(r"^#{1,2}\s+(.+)", body_en)
    title_match_ja = re.match(r"^#{1,2}\s+(.+)", body_ja)
    title_en = (
        title_match_en.group(1) if title_match_en
        else (f"Midweek Crypto Update — {today_str}" if midweek else f"Weekly Crypto Market Analysis — {week}")
    )
    title_ja = (
        title_match_ja.group(1) if title_match_ja
        else (f"週半ばアップデート — {today_str}" if midweek else f"週次暗号資産マーケット分析 — {week}")
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

    tags = [("midweek" if t == "weekly" else t) for t in TAGS] if midweek else TAGS

    return {
        "slug": f"midweek-{today_str}" if midweek else f"weekly-{week}",
        "title": {"en": title_en, "ja": title_ja},
        "date": today_str if midweek else week,
        "summary": {
            "en": first_para(body_en),
            "ja": first_para(body_ja),
        },
        "body": {"en": body_en, "ja": body_ja},
        "tags": tags,
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

    slug = (
        f"midweek-{datetime.now(timezone.utc).date()}"
        if is_midweek_run()
        else f"weekly-{week_slug()}"
    )
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
