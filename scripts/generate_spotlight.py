#!/usr/bin/env python3
"""
Single-coin "spotlight" blog post generator for bitpredictable-predictions.

Unlike the recurring weekly/midweek posts (generate_blog.py), a spotlight
post is a one-off deep dive on a single coin picked because today's signal
is unusually strong or noteworthy. Reuses the same Groq call, coin-linking,
and news-dedup helpers as generate_blog.py so both post types share the
same voice and don't duplicate logic.

Usage:
  python3 scripts/generate_spotlight.py <coin_id> [<coin_id> ...]
  python3 scripts/generate_spotlight.py --force <coin_id>   # overwrite today's post for this coin
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_blog import (  # noqa: E402
    REPO, POSTS_DIR, INDEX_FILE, PREDICTIONS_DIR,
    load_json, fetch_news, groq_generate, auto_link_coins,
    append_news_section, update_index, get_previous_post_opening,
    strip_orphan_source_labels,
)

MARKET_DATA_DIR = REPO / "market-data"

COIN_DISPLAY_FALLBACK = lambda coin_id: coin_id.replace("-", " ").title()  # noqa: E731


def load_market(coin_id: str) -> dict | None:
    snapshot = load_json(MARKET_DATA_DIR / "markets-usd.json")
    if not snapshot:
        return None
    for c in snapshot.get("coins", []):
        if c["id"] == coin_id:
            return c
    return None


def build_context(coin_id: str) -> dict | None:
    pred = load_json(PREDICTIONS_DIR / f"{coin_id}.json")
    if not pred or not pred.get("commentary", {}).get("en"):
        print(f"SKIP {coin_id}: no prediction or commentary data", file=sys.stderr)
        return None
    market = load_market(coin_id)
    return {
        "coin_id": coin_id,
        "name": market["name"] if market else COIN_DISPLAY_FALLBACK(coin_id),
        "signal": pred.get("signal", {}),
        "commentary": pred.get("commentary", {}),
        "market": market,
        "news": fetch_news(),
        "previous_opening": get_previous_post_opening(),
    }


def format_spotlight_context(ctx: dict) -> str:
    s = ctx["signal"]
    lines = [
        f"Coin: {ctx['name']} ({ctx['coin_id']})",
        f"AI forecast: {s.get('direction')} {s.get('changePercent24h', 0):+.2f}% over 24h, "
        f"confidence {s.get('confidence', 0):.0%}",
    ]
    if ctx["market"]:
        m = ctx["market"]
        lines.append(
            f"Market: price ${m['currentPrice']}, rank #{m['marketCapRank']}, "
            f"market cap ${m['marketCap']:,.0f}, 24h volume ${m['totalVolume']:,.0f}, "
            f"24h change {m.get('priceChangePercentage24h', 0):+.2f}%"
        )
    en_commentary = ctx["commentary"].get("en", "")
    if en_commentary:
        lines.append(f"\nModel's own reasoning for this signal: {en_commentary}")
    sources = ctx["commentary"].get("sources", [])
    if sources:
        lines.append("\nSupporting data points:")
        for src in sources:
            lines.append(f"  - {src['label']}: {src['value']}")
    if ctx.get("news"):
        lines.append("\nRecent crypto news headlines (use exact titles/urls, do not alter):")
        for n in ctx["news"]:
            lines.append(f"  - [{n['source']}] {n['title']} — {n['link']}")
    return "\n".join(lines)


def generate_spotlight(coin_id: str) -> dict | None:
    ctx = build_context(coin_id)
    if not ctx:
        return None

    ctx_text = format_spotlight_context(ctx)
    prev_en = ctx.get("previous_opening", {}).get("en")
    prev_ja = ctx.get("previous_opening", {}).get("ja")
    today = str(datetime.now(timezone.utc).date())

    sys_en = (
        "You are a data-driven crypto analyst for BitPredictable, a site that publishes open-book AI "
        "trading results. Write a one-off spotlight article on a single coin, explaining IN YOUR OWN WORDS "
        "why today's AI signal for this coin is worth a closer look — do not just restate the model's own "
        "reasoning sentence-for-sentence, add context (what the numbers imply, how this compares to a "
        "'normal' signal, what could invalidate the forecast). 3 sections, ~400-500 words. Choose your own "
        "section headings that fit this specific coin's situation — do not default to generic headings like "
        "'Overview' or 'Conclusion'. Link the coin once on first mention: [Name](/coins/id) using the exact "
        "id given. No hype, no investment advice, note this is informational only. Never use cliches like "
        "\"in today's fast-paced market\", \"it's important to note\", \"navigate the landscape\", \"in the "
        "world of crypto\", \"looking ahead\", \"in summary\", \"overall\", or \"as always\". Do not start more "
        "than one sentence in the whole post with the same word. Vary sentence length. If a news headline is "
        "genuinely relevant, link it naturally and say why it matters rather than just restating it."
    )
    opening_note_en = (
        f"\n\nDo not open with a sentence shaped like this recent post's opening: \"{prev_en}\" — use a "
        "different entry point."
        if prev_en else ""
    )
    prompt_en = (
        f"Write a spotlight post for {today} on {ctx['name']} ({coin_id}).\n\n"
        f"Data:\n{ctx_text}\n\n"
        "End with a one-line disclaimer that this is not investment advice."
        + opening_note_en
    )

    sys_ja = (
        "あなたはBitPredictableのデータ駆動型暗号資産アナリストです。サイトはAIトレード結果を完全公開しています。"
        "今回は単一銘柄に絞ったスポットライト記事を書いてください。モデル自身のコメンタリーをそのまま言い換えるのではなく、"
        "その数字が何を意味するか、通常のシグナルと比べてどう異質か、何が起きればこの予測が外れるか、といった文脈を加えてください。"
        "3セクション構成、合計400〜500字程度。見出しは「概況」のような汎用的なものではなく、この銘柄固有の状況に合わせて自分で考えること。"
        "銘柄名は初出時に一度だけ[名前](/coins/id)の形でリンクすること（idは与えられた正確な値を使う）。"
        "誇大表現・投資助言は禁止、情報提供目的である旨を明記。"
        "「急速に変化する市場」「〜することが重要です」「暗号資産の世界では」「今後の展望として」「総じて」のような決まり文句は禁止。"
        "記事全体を通して同じ単語で2文以上続けて書き出さないこと。文の長さに変化をつけること。"
        "関連性の高いニュースがあれば自然にリンクし、なぜ重要かを書くこと。"
    )
    opening_note_ja = (
        f"\n\n直近の記事の書き出し「{prev_ja}」と同じパターンにしないこと。別の切り口から始めること。"
        if prev_ja else ""
    )
    prompt_ja = (
        f"{today}向けに{ctx['name']}（{coin_id}）のスポットライト記事を書いてください。\n\n"
        f"データ:\n{ctx_text}\n\n"
        "最後に「投資助言ではありません」と一行追加。"
        + opening_note_ja
    )

    body_en = groq_generate(prompt_en, sys_en)
    body_ja = groq_generate(prompt_ja, sys_ja)
    if not body_en or not body_ja:
        print(f"SKIP {coin_id}: Groq generation failed (no template fallback for spotlight posts)", file=sys.stderr)
        return None

    body_en = strip_orphan_source_labels(body_en)
    body_ja = strip_orphan_source_labels(body_ja)
    body_en = auto_link_coins(body_en, [coin_id], "en")
    body_ja = auto_link_coins(body_ja, [coin_id], "ja")
    body_en = append_news_section(body_en, ctx.get("news", []), "en")
    body_ja = append_news_section(body_ja, ctx.get("news", []), "ja")

    title_match_en = re.match(r"^#{1,2}\s+(.+)", body_en)
    title_match_ja = re.match(r"^#{1,2}\s+(.+)", body_ja)
    title_en = title_match_en.group(1) if title_match_en else f"Spotlight: {ctx['name']}"
    title_ja = title_match_ja.group(1) if title_match_ja else f"スポットライト: {ctx['name']}"
    if title_match_en:
        body_en = body_en[len(title_match_en.group(0)):].strip()
    if title_match_ja:
        body_ja = body_ja[len(title_match_ja.group(0)):].strip()

    def first_para(text: str) -> str:
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]
        return text[:200]

    return {
        "slug": f"spotlight-{coin_id}-{today}",
        "title": {"en": title_en, "ja": title_ja},
        "date": today,
        "summary": {"en": first_para(body_en), "ja": first_para(body_ja)},
        "body": {"en": body_en, "ja": body_ja},
        "tags": ["spotlight", "coin-analysis", "crypto", "AI-forecast", ctx["coin_id"]],
        "coins": [coin_id],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coin_ids", nargs="+", help="Coin ids to generate spotlight posts for")
    parser.add_argument("--force", action="store_true", help="Overwrite today's post for this coin")
    args = parser.parse_args()

    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    for coin_id in args.coin_ids:
        today = str(datetime.now(timezone.utc).date())
        post_file = POSTS_DIR / f"spotlight-{coin_id}-{today}.json"
        if post_file.exists() and not args.force:
            print(f"Post already exists: {post_file} (use --force to regenerate)")
            continue

        print(f"Generating spotlight for {coin_id}...")
        post = generate_spotlight(coin_id)
        if not post:
            continue

        post_file.write_text(json.dumps(post, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote: {post_file}")
        update_index(post)
        print(f"Done: {post['title']['en']}")


if __name__ == "__main__":
    main()
