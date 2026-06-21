#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, os, sys
from datetime import datetime, timezone

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')

SYMBOL_MAP = {
    'bitcoin':'BTC','ethereum':'ETH','tether':'USDT','binancecoin':'BNB',
    'ripple':'XRP','solana':'SOL','dogecoin':'DOGE','cardano':'ADA',
    'avalanche-2':'AVAX','chainlink':'LINK','uniswap':'UNI','polkadot':'DOT',
    'litecoin':'LTC','stellar':'XLM','monero':'XMR','shiba-inu':'SHIB',
    'near':'NEAR','sui':'SUI','tron':'TRX','hedera-hashgraph':'HBAR',
}

def symbol_for(coin_id):
    return SYMBOL_MAP.get(coin_id, coin_id.split('-')[0].upper()[:6])

def compute_regime(pdir):
    try:
        coin_ids = load_json(os.path.join(pdir, 'index.json'))
        changes = []
        for cid in coin_ids[:15]:
            p = os.path.join(pdir, '{}.json'.format(cid))
            if not os.path.exists(p): continue
            sig = load_json(p).get('signal', {})
            chg = sig.get('changePercent24h', 0.0) or 0.0
            changes.append(chg)
        if not changes: return 'range'
        avg = sum(changes) / len(changes)
        std = math.sqrt(sum((c - avg)**2 for c in changes) / len(changes))
        if std > 3.0: return 'high_vol'
        if avg > 1.0: return 'bull'
        if avg < -1.0: return 'bear'
        return 'range'
    except Exception as e:
        print('[v2] regime error: {}'.format(e)); return 'range'

def compute_risk_metrics(log):
    perf = log.get('performance', {})
    pos = log.get('currentPosition')
    initial = float(log.get('initialCapital', 1.0))
    max_dd = perf.get('maxDrawdownPct', 0.0) or 0.0
    exposure_pct, open_pos = 0.0, 0
    if pos and pos.get('direction', 'flat') != 'flat':
        size = pos.get('sizeBtc', 0.0) or 0.0
        exposure_pct = round(size / initial * 100, 2) if initial > 0 else 0.0
        open_pos = 1
    win_rate = perf.get('winRate', 0.0) or 0.0
    return {
        'varPct': round(max_dd * 0.5, 4),
        'exposurePct': exposure_pct,
        'cbStatus': 'active',
        'kellyF': round(max(0.0, win_rate - (1.0 - win_rate)) * 0.25, 4),
        'openPositions': open_pos,
        'maxPositions': 3,
    }

def compute_signal_scores(pdir):
    try:
        coin_ids = load_json(os.path.join(pdir, 'index.json'))
        scores = []
        for cid in coin_ids:
            p = os.path.join(pdir, '{}.json'.format(cid))
            if not os.path.exists(p): continue
            pred = load_json(p)
            sig = pred.get('signal', {})
            chg = sig.get('changePercent24h', 0.0) or 0.0
            conf = sig.get('confidence', 0.5) or 0.5
            direction = sig.get('direction', 'flat')
            score = round(max(0.0, min(100.0, (chg + 10.0) / 20.0 * 100.0)), 2)
            if direction == 'up' and score >= 65: status = 'entering'
            elif score >= 55: status = 'watching'
            elif score >= 45: status = 'holding'
            elif score >= 30: status = 'weak'
            else: status = 'blocked'
            scores.append({'symbol': symbol_for(cid), 'coin': cid,
                           'score': score, 'confidence': round(conf, 4), 'status': status})
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
    except Exception as e:
        print('[v2] signals error: {}'.format(e)); return []

def compute_prediction_accuracy(pdir):
    try:
        coin_ids = load_json(os.path.join(pdir, 'index.json'))
        points = []
        for cid in coin_ids[:20]:
            p = os.path.join(pdir, '{}.json'.format(cid))
            if not os.path.exists(p): continue
            pred = load_json(p)
            series = pred.get('series', [])
            sig = pred.get('signal', {})
            actuals = [s for s in series if s.get('actualIndex') is not None]
            if len(actuals) < 25: continue
            pivot_val = actuals[-25].get('actualIndex') or 0.0
            cur_val = actuals[-1].get('actualIndex') or 0.0
            if pivot_val == 0: continue
            actual_chg = (cur_val - pivot_val) / pivot_val * 100.0
            pred_chg = sig.get('changePercent24h', 0.0) or 0.0
            hit = (pred_chg > 0 and actual_chg > 0) or (pred_chg < 0 and actual_chg < 0)
            points.append({'symbol': symbol_for(cid), 'date': actuals[-1]['time'][:10],
                           'predictedPct': round(pred_chg, 4), 'actualPct': round(actual_chg, 4), 'hit': hit})
        return points
    except Exception as e:
        print('[v2] accuracy error: {}'.format(e)); return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    if not os.path.exists(args.log):
        print('[v2] ERROR: {} not found'.format(args.log)); sys.exit(1)
    log = load_json(args.log)
    log['regime'] = compute_regime(args.predictions)
    log['riskMetrics'] = compute_risk_metrics(log)
    log['signalScores'] = compute_signal_scores(args.predictions)
    log['predictionAccuracy'] = compute_prediction_accuracy(args.predictions)
    log['updatedAt'] = datetime.now(timezone.utc).isoformat()
    save_json(args.log, log)
    print('[v2] done  regime={}  signals={}  accuracy={}'.format(
        log['regime'], len(log['signalScores']), len(log['predictionAccuracy'])))

if __name__ == '__main__':
    main()
