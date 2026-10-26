#!/usr/bin/env python3
"""
PRISM v34 — Walk-forward
=========================
Valide la stratégie hors échantillon : pour chaque fenêtre, on choisit les
paramètres Pattern D sur N mois (train) puis on les applique tels quels au
mois suivant (test), jamais vu pendant le choix. On compare avec les
paramètres v33 figés et avec Pattern C seul.

Réutilise le moteur et le cache de backtest_v33.py (données 1H OKX).

Usage:
  python3 walkforward_v34.py                    # 12 mois, train 4 / test 1
  python3 walkforward_v34.py --train 3 --test 1
  python3 walkforward_v34.py --lev-scale 0.5    # leviers ÷2
  python3 walkforward_v34.py --no-fetch         # données cachées uniquement
"""

import argparse
import sys

import numpy as np
import pandas as pd

import backtest_v33 as bt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# score_min_d = 101 → Pattern D désactivé (score plafonné à 100)
D_OFF = 101
GRID_D = [(sc, adx) for sc in (72, 76, 80) for adx in (22, 25, 28)] + [(D_OFF, bt.ADX_MIN_D)]

def scale_leverage(k: float):
    for name in ("BASE_LEVERAGE", "HIGH_LEVERAGE", "BASE_LEVERAGE_D", "HIGH_LEVERAGE_D"):
        setattr(bt, name, max(1, round(getattr(bt, name) * k)))

def stats(trades: list, curve: list, final_eq: float) -> dict:
    eq = np.array([c["equity"] for c in curve] + [final_eq])
    dd = float((1 - eq / np.maximum.accumulate(eq)).max()) if len(eq) else 0.0
    ret = final_eq / bt.INITIAL_CAPITAL - 1
    wins = sum(t["pnl"] > 0 for t in trades)
    return {"ret": ret, "dd": dd, "n": len(trades),
            "wr": wins / len(trades) if trades else 0.0,
            # Score de sélection : rendement pénalisé par le drawdown
            "score": ret - 1.5 * dd}

def run(sym_data, a, b, sc, adx) -> dict:
    return stats(*bt.run_backtest(sym_data, a, b, score_min_d=sc, adx_min_d=adx))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",    type=int,   default=12)
    parser.add_argument("--train",     type=int,   default=4)
    parser.add_argument("--test",      type=int,   default=1)
    parser.add_argument("--lev-scale", type=float, default=1.0)
    parser.add_argument("--no-fetch",  action="store_true")
    args = parser.parse_args()

    if args.lev_scale != 1.0:
        scale_leverage(args.lev_scale)

    print(f"\n  PRISM v34 — Walk-forward  ({args.months} mois, "
          f"train {args.train} / test {args.test}, levier ×{args.lev_scale})")
    print(f"  {'─'*60}")

    sym_data = {}
    for sym in bt.SYMBOLS:
        df = bt.load_or_fetch(sym, args.months, args.no_fetch)
        if df is not None and len(df) > 300:
            sym_data[sym] = bt.prepare(sym, df)
    if "BTC-USDT" not in sym_data:
        print("  ERREUR : BTC-USDT manquant — abandon.\n")
        return
    print(f"  {len(sym_data)}/{len(bt.SYMBOLS)} symboles chargés\n")

    btc_ts = sym_data["BTC-USDT"]["ts_index"]
    rich   = [sd for sd in sym_data.values() if len(sd["ts_index"]) >= 1000]
    first  = max(sd["ts_index"][300] for sd in rich)
    end    = pd.Timestamp(btc_ts[-2])
    start  = max(first, end - pd.DateOffset(months=args.months))

    folds = []
    t0 = start
    while True:
        t1 = t0 + pd.DateOffset(months=args.train)
        t2 = t1 + pd.DateOffset(months=args.test)
        if t2 > end:
            break
        folds.append((t0, t1, t2))
        t0 = t0 + pd.DateOffset(months=args.test)
    if not folds:
        print("  Pas assez de données pour une seule fenêtre.\n")
        return

    eq = {"wf": 1.0, "v33": 1.0, "c_only": 1.0}
    rows = []
    for t0, t1, t2 in folds:
        best = max(GRID_D, key=lambda p: run(sym_data, t0, t1, *p)["score"])
        wf   = run(sym_data, t1, t2, *best)
        v33  = run(sym_data, t1, t2, bt.SCORE_MIN_D, bt.ADX_MIN_D)
        c    = run(sym_data, t1, t2, D_OFF, bt.ADX_MIN_D)
        eq["wf"]     *= 1 + wf["ret"]
        eq["v33"]    *= 1 + v33["ret"]
        eq["c_only"] *= 1 + c["ret"]
        label = "D off" if best[0] == D_OFF else f"D {best[0]}/{best[1]}"
        rows.append((t1.date(), t2.date(), label, wf, v33, c))

    print(f"  {'Test':<23} {'Choix train':<12} {'Walk-fwd':>16} {'v33 figé':>16} {'C seul':>16}")
    for a, b, label, wf, v33, c in rows:
        fmt = lambda s: f"{s['ret']:+7.1%} dd{s['dd']:4.0%}"
        print(f"  {str(a)}→{str(b)}  {label:<12} {fmt(wf):>16} {fmt(v33):>16} {fmt(c):>16}")

    print(f"\n  Cumul hors échantillon ({len(folds)} mois de test, chaînés) :")
    print(f"    Walk-forward : {eq['wf'] - 1:+.1%}")
    print(f"    v33 figé     : {eq['v33'] - 1:+.1%}")
    print(f"    Pattern C    : {eq['c_only'] - 1:+.1%}")
    print("\n  NB : la whitelist Pattern D a été choisie sur l'historique ;")
    print("  le résultat « v33 figé » reste donc optimiste.\n")

if __name__ == "__main__":
    main()
