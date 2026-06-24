#!/usr/bin/env python3
"""
PRISM v33 — Backtester
=======================
Télécharge 6 mois de données 1H OKX (avec cache local),
rejoue le moteur exact barre par barre sur les 24 cryptos,
et génère un rapport complet de performance.

Usage:
  python3 backtest_v33.py                # 6 mois (défaut)
  python3 backtest_v33.py --months 12    # 12 mois
  python3 backtest_v33.py --no-fetch     # données cachées uniquement
  python3 backtest_v33.py --fetch-only   # téléchargement seul
"""

import argparse
import math
import os
import sys
import time

# Force UTF-8 sur Windows (console CP1252 par défaut)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "backtest_data"
RESULT_DIR  = BASE_DIR / "backtest_results"
DATA_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT",
    "AVAX-USDT","ADA-USDT","LINK-USDT","XRP-USDT","DOT-USDT","ATOM-USDT",
    "LTC-USDT","DOGE-USDT","NEAR-USDT","TRX-USDT","INJ-USDT","OP-USDT",
    "ARB-USDT","SUI-USDT","UNI-USDT","AAVE-USDT","TIA-USDT",
    "SEI-USDT","HBAR-USDT","ICP-USDT","JUP-USDT",
]

INITIAL_CAPITAL  = 2500.0
COMMISSION       = 0.001
SLIPPAGE         = 0.0005
EXIT_SLIPPAGE    = 0.0003
ATR_SL_MULT      = 1.5
RR_RATIO         = 4.0
ATR_SL_MIN_C     = 0.006
ATR_SL_MAX_C     = 0.025
SQUEEZE_BARS_C   = 4
VOL_RATIO_C      = 1.5
ADX_MIN_C        = 20
TIME_STOP_H      = 72
COOLDOWN_BARS    = 5
BASE_LEVERAGE    = 10
HIGH_LEVERAGE    = 15
DAILY_LOSS_CAP   = 0.12
MAX_MARGIN_RATIO = 0.60
RISK_PCT         = 0.10
MAX_POS          = 4
SCORE_MIN        = 65

SCORE_MIN_D     = 76   # relevé 68→76 (grid search : +123% vs -24% à 68)
ADX_MIN_D       = 25   # relevé 22→25 (filtre tendances faibles)
RISK_PCT_D      = 0.06
BASE_LEVERAGE_D = 8
HIGH_LEVERAGE_D = 10

# Univers autorisé Pattern D — actifs "trending" (WR≥27% ET PnL>0 sur 4 mois)
# Les actifs mean-reverting (ATOM, ARB, LTC, SEI…) génèrent des faux signaux.
PATTERN_D_WHITELIST = {
    "AVAX-USDT", "TIA-USDT", "OP-USDT", "JUP-USDT",
    "ICP-USDT",  "SUI-USDT", "DOT-USDT", "TRX-USDT",
}

# ── Téléchargement données ───────────────────────────────────────────────────
def fetch_symbol(sym: str, months: int = 6) -> pd.DataFrame | None:
    bars_needed = months * 30 * 24 + 400   # warmup indicateurs
    all_rows, after = [], None
    page_retries = 0
    while len(all_rows) < bars_needed:
        params = {"instId": sym, "bar": "1H", "limit": 100}
        if after:
            params["after"] = after
        try:
            r    = requests.get(OKX_URL, params=params, timeout=20)
            data = r.json()
        except Exception as e:
            page_retries += 1
            if page_retries > 8:
                break   # retourne ce qu'on a déjà
            time.sleep(3 * page_retries)
            continue
        code = data.get("code", "")
        if code == "50011":   # rate limit OKX → attente puis retry
            page_retries += 1
            time.sleep(5 * page_retries)
            continue
        if code != "0" or not data.get("data"):
            break
        batch = data["data"]
        all_rows.extend(batch)
        page_retries = 0
        if len(batch) < 100:
            break
        after = batch[-1][0]
        time.sleep(0.25)   # 4 req/s max — OKX limite à 10/s sur history-candles

    if len(all_rows) < 200:
        return None
    df = pd.DataFrame(all_rows,
                      columns=["timestamp","open","high","low","close","volume","a","b","c"])
    df = df[["timestamp","open","high","low","close","volume"]]
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(np.int64), unit="ms", utc=True
    ).dt.tz_convert(None)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["close"])
    return df

def load_or_fetch(sym: str, months: int, no_fetch: bool) -> pd.DataFrame | None:
    path = DATA_DIR / f"{sym.replace('-','_')}_{months}m.csv"
    if path.exists() and no_fetch:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < 6:
            return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_symbol(sym, months)
    if df is not None:
        df.to_csv(path)
    return df

# ── Indicateurs (identiques live_monitor_v33) ────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ml    = ema12 - ema26
    ms    = ml.ewm(span=9, adjust=False).mean()
    df["macd_hist"]  = ml - ms
    df["macd_slope"] = (ml - ms).diff()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    bbw            = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10)
    df["bbw"]      = bbw
    df["bbw_q15"]  = bbw.rolling(40).quantile(0.15)
    df["vol_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    tp_val = (h + l + c) / 3
    df["vwap"] = (tp_val * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    sk = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_k"] = sk
    df["stoch_d"] = sk.rolling(3).mean()
    tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    dm_p = (h - h.shift()).clip(lower=0)
    dm_m = (l.shift() - l).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0)
    dm_m = dm_m.where(dm_m > dm_p, 0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    dip   = 100 * dm_p.ewm(com=13, adjust=False).mean() / (atr14 + 1e-10)
    dim   = 100 * dm_m.ewm(com=13, adjust=False).mean() / (atr14 + 1e-10)
    dx    = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    df["atr14"]   = atr14
    df["adx"]     = dx.ewm(com=13, adjust=False).mean()
    df_4h    = df[["close"]].resample("4h").last().dropna()
    ema20_4h = df_4h["close"].ewm(span=20, adjust=False).mean()
    ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
    df["ema20_4h"] = ema20_4h.reindex(df.index, method="ffill")
    df["ema50_4h"] = ema50_4h.reindex(df.index, method="ffill")
    return df

def _compute_scores(sd: dict, n: int):
    buy_sc  = np.zeros(n, dtype=np.int32)
    sell_sc = np.zeros(n, dtype=np.int32)
    for i in range(n):
        bs = ss = 0
        e9, e21, e50 = sd["ema9"][i], sd["ema21"][i], sd["ema50"][i]
        if not any(math.isnan(v) for v in [e9, e21, e50]):
            if e9  > e21: bs += 12
            elif e9  < e21: ss += 12
            if e21 > e50: bs += 13
            elif e21 < e50: ss += 13
        r = sd["rsi14"][i]
        if not math.isnan(r):
            if 40 <= r <= 65:  bs += 15
            elif 35 <= r < 40: bs += 8
            elif 65 < r <= 70: bs += 5
            if 35 <= r <= 60:  ss += 15
            elif 60 < r <= 65: ss += 8
            elif 30 <= r < 35: ss += 5
        mh, mhs = sd["macd_hist"][i], sd["macd_slope"][i]
        if not any(math.isnan(v) for v in [mh, mhs]):
            if mh  > 0: bs += 12
            elif mh  < 0: ss += 12
            if mhs > 0: bs += 8
            elif mhs < 0: ss += 8
        vr = sd["vol_ratio"][i]
        if not math.isnan(vr):
            pts = 10 if vr >= 1.5 else 6 if vr >= 1.0 else 3 if vr >= 0.7 else 0
            bs += pts; ss += pts
        av = sd["adx"][i]
        if not math.isnan(av):
            pts = 10 if av >= 25 else 6 if av >= 18 else 0
            bs += pts; ss += pts
        cl, vw = sd["close"][i], sd["vwap"][i]
        if not any(math.isnan(v) for v in [cl, vw]):
            if cl > vw:  bs += 10
            elif cl < vw: ss += 10
        sk, sd_ = sd["stoch_k"][i], sd["stoch_d"][i]
        if not any(math.isnan(v) for v in [sk, sd_]):
            if sk > sd_ and sk < 75: bs += 10
            if sk < sd_ and sk > 25: ss += 10
        buy_sc[i]  = min(bs, 100)
        sell_sc[i] = min(ss, 100)
    return buy_sc, sell_sc

def prepare(sym: str, df: pd.DataFrame) -> dict:
    df = compute_indicators(df)
    ts_idx    = df.index.tolist()
    ts_to_pos = {ts: i for i, ts in enumerate(ts_idx)}
    cols = ["close","high","low","open","atr14","adx","bbw","bbw_q15",
            "bb_upper","bb_lower","vol_ratio","ema9","ema21","ema50",
            "macd_hist","macd_slope","rsi14","stoch_k","stoch_d","vwap",
            "ema20_4h","ema50_4h"]
    sd = {"name": sym, "ts_index": ts_idx, "ts_to_pos": ts_to_pos}
    for col in cols:
        sd[col] = df[col].values
    n = len(ts_idx)
    sd["buy_sc"], sd["sell_sc"] = _compute_scores(sd, n)
    return sd

def check_pattern_c(sd: dict, bar: int, adx_val: float):
    if bar < SQUEEZE_BARS_C + 3:
        return None
    try:
        bbw_arr, bbwq_arr = sd["bbw"], sd["bbw_q15"]
        bbw_cur, bbwq_cur = bbw_arr[bar], bbwq_arr[bar]
        if math.isnan(bbw_cur) or math.isnan(bbwq_cur):
            return None
        for i in range(1, SQUEEZE_BARS_C + 1):
            bw, bq = bbw_arr[bar - i], bbwq_arr[bar - i]
            if math.isnan(bw) or math.isnan(bq) or bw >= bq:
                return None
        if bbw_cur <= bbwq_cur:
            return None
        adx_prev = sd["adx"][bar - 2]
        if math.isnan(adx_val) or math.isnan(adx_prev):
            return None
        if adx_val <= adx_prev + 1.5 or adx_val < ADX_MIN_C:
            return None
        close    = sd["close"][bar]
        bb_upper = sd["bb_upper"][bar]
        bb_lower = sd["bb_lower"][bar]
        vol_r    = sd["vol_ratio"][bar]
        ema20_4h = sd["ema20_4h"][bar]
        ema50_4h = sd["ema50_4h"][bar]
        if any(math.isnan(v) for v in [close, bb_upper, bb_lower, vol_r, ema20_4h, ema50_4h]):
            return None
        if vol_r < VOL_RATIO_C:
            return None
        asset_4h_bull = ema20_4h > ema50_4h
        if close > bb_upper and asset_4h_bull:
            return "BUY"
        if close < bb_lower and not asset_4h_bull:
            return "SELL"
    except Exception:
        pass
    return None

def check_pattern_d(sd: dict, bar: int, adx_val: float):
    if bar < 152:
        return None
    try:
        e9, e21, e50 = sd["ema9"][bar], sd["ema21"][bar], sd["ema50"][bar]
        if any(math.isnan(v) for v in [e9, e21, e50]):
            return None
        rsi = sd["rsi14"][bar]
        if math.isnan(rsi):
            return None
        cl, vw = sd["close"][bar], sd["vwap"][bar]
        if any(math.isnan(v) for v in [cl, vw]):
            return None
        ema20_4h = sd["ema20_4h"][bar]
        ema50_4h = sd["ema50_4h"][bar]
        if any(math.isnan(v) for v in [ema20_4h, ema50_4h]):
            return None
        # Fix 2 : ADX doit monter depuis 2 barres (tendance s'accélère)
        adx_1 = sd["adx"][bar - 1]
        adx_2 = sd["adx"][bar - 2]
        if math.isnan(adx_1) or math.isnan(adx_2):
            return None
        if not (adx_val > adx_1 > adx_2):
            return None
        asset_4h_bull = ema20_4h > ema50_4h
        if e9 > e21 > e50 and 42 <= rsi <= 72 and cl > vw and asset_4h_bull:
            return "BUY"
        if e9 < e21 < e50 and 28 <= rsi <= 58 and cl < vw and not asset_4h_bull:
            return "SELL"
    except Exception:
        pass
    return None

# ── Moteur backtest ──────────────────────────────────────────────────────────
def run_backtest(sym_data: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                 activation_atr: float = 0.0, trail_distance_atr: float = 0.0,
                 time_stop_c: int = 0,
                 score_min_d: int = SCORE_MIN_D, adx_min_d: float = ADX_MIN_D):
    """
    activation_atr      : ATR multiplier — active le trailing quand profit ≥ n×ATR
                          0 = désactivé (comportement classique TP fixe)
    trail_distance_atr  : ATR multiplier — distance du SL suiveur derrière le max/min
    time_stop_c         : si > 0, ferme Pattern C après N barres sans profit significatif
    """
    btc_sd    = sym_data.get("BTC-USDT")
    all_ts    = sorted(btc_sd["ts_index"])
    test_ts   = [ts for ts in all_ts if start_ts <= ts <= end_ts]

    equity          = INITIAL_CAPITAL
    peak_equity     = INITIAL_CAPITAL
    day_start_eq    = INITIAL_CAPITAL
    current_day     = ""
    open_positions  = {}
    cooldown_tracker = {}
    trades          = []
    equity_curve    = [{"ts": str(start_ts), "equity": equity}]
    use_trail       = activation_atr > 0 and trail_distance_atr > 0

    for bar_ts in test_ts:
        today = str(bar_ts)[:10]
        if today != current_day:
            current_day   = today
            day_start_eq  = equity

        # ── Exits ────────────────────────────────────────────────────────────
        to_remove = []
        for pk, pos in list(open_positions.items()):
            sd = sym_data.get(pos["sym"])
            if sd is None:
                continue
            bar = sd["ts_to_pos"].get(bar_ts)
            if bar is None:
                continue
            hi, lo, cl = float(sd["high"][bar]), float(sd["low"][bar]), float(sd["close"][bar])
            entry = pos["entry_price"]
            sl    = pos["sl"]
            tp    = pos["tp"]
            side  = pos["side"]

            # ── Trailing ATR : mise à jour du SL suiveur ─────────────────────
            if use_trail:
                atr = float(sd["atr14"][bar])
                if not math.isnan(atr) and atr > 0:
                    if side == "long":
                        peak = max(pos.get("peak_price", entry), hi)
                        pos["peak_price"] = peak
                        # Activer trailing si profit ≥ activation_atr × ATR
                        if peak >= entry + activation_atr * atr:
                            trail_sl = peak - trail_distance_atr * atr
                            if trail_sl > sl:
                                pos["sl"] = sl = trail_sl
                                pos["trailing"] = True
                    else:  # short
                        trough = min(pos.get("peak_price", entry), lo)
                        pos["peak_price"] = trough
                        if trough <= entry - activation_atr * atr:
                            trail_sl = trough + trail_distance_atr * atr
                            if trail_sl < sl:
                                pos["sl"] = sl = trail_sl
                                pos["trailing"] = True

            # ── Vérification des exits ────────────────────────────────────────
            exit_price = exit_reason = None
            if side == "long":
                if hi >= tp and not pos.get("trailing"):
                    exit_price, exit_reason = tp, "take_profit"
                elif lo <= sl:
                    exit_reason = "trail_stop" if pos.get("trailing") else "stop_loss"
                    exit_price  = sl
            else:
                if lo <= tp and not pos.get("trailing"):
                    exit_price, exit_reason = tp, "take_profit"
                elif hi >= sl:
                    exit_reason = "trail_stop" if pos.get("trailing") else "stop_loss"
                    exit_price  = sl

            # ── Time Stop Pattern C ───────────────────────────────────────────
            entry_ts  = pos["entry_ts"]
            elapsed_h = (bar_ts - entry_ts).total_seconds() / 3600
            if exit_price is None:
                if time_stop_c > 0 and pos.get("pattern") == "C" and elapsed_h >= time_stop_c:
                    exit_price, exit_reason = cl, "time_stop_c"
                elif elapsed_h >= TIME_STOP_H:
                    exit_price, exit_reason = cl, "time_stop"

            if exit_price is not None:
                exit_price = (exit_price * (1 - EXIT_SLIPPAGE) if side == "long"
                              else exit_price * (1 + EXIT_SLIPPAGE))
                side_mult = 1 if side == "long" else -1
                notional  = pos["margin"] * pos["leverage"]
                raw_pnl   = (exit_price - entry) / (entry + 1e-10) * side_mult * notional
                fees      = notional * COMMISSION * 2
                net_pnl   = raw_pnl - fees
                equity   += net_pnl
                peak_equity = max(peak_equity, equity)
                trade = {
                    **pos,
                    "exit_ts":    bar_ts,
                    "exit_price": exit_price,
                    "reason":     exit_reason,
                    "trailing":   pos.get("trailing", False),
                    "pnl":        round(net_pnl, 2),
                    "equity_after": round(equity, 2),
                    "mae":        pos.get("mae", 0.0),
                    "mfe":        pos.get("mfe", 0.0),
                }
                trades.append(trade)
                to_remove.append(pk)
                equity_curve.append({"ts": str(bar_ts), "equity": round(equity, 2)})

        for k in to_remove:
            del open_positions[k]

        # Mise à jour MAE/MFE pour les positions ouvertes
        for pk, pos in open_positions.items():
            sd = sym_data.get(pos["sym"])
            if sd is None:
                continue
            bar = sd["ts_to_pos"].get(bar_ts)
            if bar is None:
                continue
            cl    = float(sd["close"][bar])
            entry = pos["entry_price"]
            side  = pos["side"]
            move  = (cl - entry) / entry if side == "long" else (entry - cl) / entry
            pos["mfe"] = max(pos.get("mfe", 0.0), move)
            pos["mae"] = min(pos.get("mae", 0.0), move)

        # ── Gardes entrée ─────────────────────────────────────────────────────
        # Fix 1 : pas d'entrée en session asiatique 00h-07h UTC (WR 19%)
        if bar_ts.hour < 7:
            continue
        day_pnl_pct = (equity - day_start_eq) / (day_start_eq + 1e-10)
        if day_pnl_pct <= -DAILY_LOSS_CAP:
            continue
        if len(open_positions) >= MAX_POS:
            continue
        drawdown = (peak_equity - equity) / (peak_equity + 1e-10)
        if drawdown > 0.40:
            continue

        # Filtre BTC macro
        btc_bar = btc_sd["ts_to_pos"].get(bar_ts)
        btc_4h_bull = None
        if btc_bar is not None:
            b20 = float(btc_sd["ema20_4h"][btc_bar])
            b50 = float(btc_sd["ema50_4h"][btc_bar])
            if not (math.isnan(b20) or math.isnan(b50)):
                btc_4h_bull = b20 > b50

        dd_scale         = max(0.5, 1.0 - drawdown * 2.5)
        margin_c         = equity * RISK_PCT   * dd_scale
        margin_d         = equity * RISK_PCT_D * dd_scale
        total_margin_used = sum(p["margin"] for p in open_positions.values())
        max_margin       = equity * MAX_MARGIN_RATIO
        syms_in_pos      = {p["sym"] for p in open_positions.values()}

        for sym, sd in sym_data.items():
            if sym == "BTC-USDT":
                continue
            if sym in syms_in_pos:
                continue
            if len(open_positions) >= MAX_POS:
                break

            bar = sd["ts_to_pos"].get(bar_ts)
            if bar is None or bar < 250:
                continue
            adx_val = float(sd["adx"][bar])
            if math.isnan(adx_val):
                continue

            # ── Pattern C ──────────────────────────────────────────────────
            ck = sym + "C"
            if ck not in [p.get("pattern_key","") for p in open_positions.values()]:
                bar_cd = cooldown_tracker.get(ck, -9999)
                if bar - bar_cd >= COOLDOWN_BARS:
                    action = check_pattern_c(sd, bar, adx_val)
                    if action is not None:
                        if btc_4h_bull is not None:
                            if action == "BUY"  and not btc_4h_bull: action = None
                            if action == "SELL" and btc_4h_bull:     action = None
                    if action is not None:
                        score = int(sd["buy_sc"][bar]) if action == "BUY" else int(sd["sell_sc"][bar])
                        if score >= SCORE_MIN and total_margin_used + margin_c <= max_margin:
                            lev = HIGH_LEVERAGE if adx_val > 28 and score >= 72 else BASE_LEVERAGE
                            side = "long" if action == "BUY" else "short"
                            open_px = float(sd["open"][bar])
                            ep  = open_px * (1 + SLIPPAGE) if side == "long" else open_px * (1 - SLIPPAGE)
                            atr = float(sd["atr14"][bar])
                            if math.isnan(atr) or atr <= 0:
                                atr = ep * 0.015
                            sl_pct = max(ATR_SL_MIN_C, min(ATR_SL_MAX_C, ATR_SL_MULT * atr / (ep + 1e-10)))
                            tp_pct = sl_pct * RR_RATIO
                            sl = ep * (1 - sl_pct) if side == "long" else ep * (1 + sl_pct)
                            tp = ep * (1 + tp_pct) if side == "long" else ep * (1 - tp_pct)
                            pk = ck + str(bar_ts)
                            open_positions[pk] = {
                                "sym": sym, "side": side, "pattern": "C",
                                "pattern_key": ck,
                                "entry_ts": bar_ts, "entry_price": ep,
                                "sl": sl, "tp": tp,
                                "margin": margin_c, "leverage": lev,
                                "score": score, "adx": adx_val,
                                "hour": bar_ts.hour,
                            }
                            cooldown_tracker[ck] = bar
                            total_margin_used += margin_c
                            syms_in_pos.add(sym)

            # ── Pattern D ──────────────────────────────────────────────────
            if sym in syms_in_pos:
                continue
            if sym not in PATTERN_D_WHITELIST:
                continue
            dk = sym + "D"
            bar_cd = cooldown_tracker.get(dk, -9999)
            if bar - bar_cd >= COOLDOWN_BARS and adx_val >= adx_min_d:
                action = check_pattern_d(sd, bar, adx_val)
                if action is not None:
                    if btc_4h_bull is not None:
                        if action == "BUY"  and not btc_4h_bull: action = None
                        if action == "SELL" and btc_4h_bull:     action = None
                if action is not None:
                    score = int(sd["buy_sc"][bar]) if action == "BUY" else int(sd["sell_sc"][bar])
                    if score >= score_min_d and total_margin_used + margin_d <= max_margin:
                        lev = HIGH_LEVERAGE_D if adx_val > 28 and score >= 68 else BASE_LEVERAGE_D
                        side = "long" if action == "BUY" else "short"
                        open_px = float(sd["open"][bar])
                        ep  = open_px * (1 + SLIPPAGE) if side == "long" else open_px * (1 - SLIPPAGE)
                        atr = float(sd["atr14"][bar])
                        if math.isnan(atr) or atr <= 0:
                            atr = ep * 0.015
                        sl_pct = max(ATR_SL_MIN_C, min(ATR_SL_MAX_C, ATR_SL_MULT * atr / (ep + 1e-10)))
                        tp_pct = sl_pct * RR_RATIO
                        sl = ep * (1 - sl_pct) if side == "long" else ep * (1 + sl_pct)
                        tp = ep * (1 + tp_pct) if side == "long" else ep * (1 - tp_pct)
                        pk = dk + str(bar_ts)
                        open_positions[pk] = {
                            "sym": sym, "side": side, "pattern": "D",
                            "pattern_key": dk,
                            "entry_ts": bar_ts, "entry_price": ep,
                            "sl": sl, "tp": tp,
                            "margin": margin_d, "leverage": lev,
                            "score": score, "adx": adx_val,
                            "hour": bar_ts.hour,
                        }
                        cooldown_tracker[dk] = bar
                        total_margin_used += margin_d
                        syms_in_pos.add(sym)

    # Fermer les positions restantes au dernier prix
    for pk, pos in open_positions.items():
        sd  = sym_data.get(pos["sym"])
        bar = sd["ts_to_pos"].get(test_ts[-1]) if sd else None
        if bar is not None:
            cl  = float(sd["close"][bar])
            ep  = pos["entry_price"]
            sm  = 1 if pos["side"] == "long" else -1
            not_ = pos["margin"] * pos["leverage"]
            raw  = (cl - ep) / (ep + 1e-10) * sm * not_
            net  = raw - not_ * COMMISSION * 2
            equity += net
            trades.append({**pos, "exit_ts": test_ts[-1], "exit_price": cl,
                            "reason": "end_of_backtest",
                            "pnl": round(net, 2),
                            "equity_after": round(equity, 2)})

    return trades, equity_curve, equity

# ── Rapport ──────────────────────────────────────────────────────────────────
def print_report(trades: list, equity_curve: list, final_equity: float,
                 months: int, start_ts, end_ts):
    n  = len(trades)
    if n == 0:
        print("\n  Aucun trade sur la période.")
        return

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tp_hits = [t for t in trades if t.get("reason") == "take_profit"]
    sl_hits = [t for t in trades if t.get("reason") == "stop_loss"]
    ts_hits = [t for t in trades if t.get("reason") == "time_stop"]

    total_pnl  = sum(t["pnl"] for t in trades)
    ret_pct    = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate   = len(wins) / n * 100
    avg_win    = sum(t["pnl"] for t in wins)  / max(len(wins), 1)
    avg_loss   = sum(t["pnl"] for t in losses) / max(len(losses), 1)
    profit_f   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Drawdown
    eq_vals  = [e["equity"] for e in equity_curve]
    peak_eq  = INITIAL_CAPITAL
    max_dd   = 0.0
    for eq in eq_vals:
        peak_eq = max(peak_eq, eq)
        dd = (peak_eq - eq) / peak_eq * 100
        max_dd = max(max_dd, dd)

    # Durée moyenne position
    durations = []
    for t in trades:
        try:
            ets = t["entry_ts"] if isinstance(t["entry_ts"], pd.Timestamp) else pd.Timestamp(t["entry_ts"])
            xts = t["exit_ts"]  if isinstance(t["exit_ts"],  pd.Timestamp) else pd.Timestamp(t["exit_ts"])
            durations.append((xts - ets).total_seconds() / 3600)
        except Exception:
            pass
    avg_dur = sum(durations) / len(durations) if durations else 0

    sep = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  PRISM v33 — RAPPORT BACKTEST  {months} mois")
    print(f"  {start_ts.date()} → {end_ts.date()}")
    print(f"{'═'*62}")
    print(f"\n  Capital initial  : €{INITIAL_CAPITAL:,.2f}")
    print(f"  Capital final    : €{final_equity:,.2f}")
    print(f"  Rendement total  : {ret_pct:+.1f}%")
    print(f"  Drawdown max     : {max_dd:.1f}%")
    print(f"\n{sep}")
    print(f"  Trades total     : {n}")
    print(f"  Taux de réussite : {win_rate:.1f}%  ({len(wins)} gagnants / {len(losses)} perdants)")
    print(f"  Gain moyen       : +€{avg_win:.2f}")
    print(f"  Perte moyenne    : €{avg_loss:.2f}")
    print(f"  Profit factor    : {profit_f:.2f}")
    print(f"  Durée moy. trade : {avg_dur:.1f}h")
    print(f"\n  TP atteints      : {len(tp_hits)} ({len(tp_hits)/n*100:.0f}%)")
    print(f"  SL touchés       : {len(sl_hits)} ({len(sl_hits)/n*100:.0f}%)")
    print(f"  Time stops       : {len(ts_hits)} ({len(ts_hits)/n*100:.0f}%)")

    # Par pattern
    for pat in ["C", "D"]:
        pt = [t for t in trades if t.get("pattern") == pat]
        if not pt:
            continue
        pw = [t for t in pt if t["pnl"] > 0]
        print(f"\n{sep}")
        print(f"  Pattern {pat} — {len(pt)} trades")
        print(f"    Wins     : {len(pw)}/{len(pt)}  ({len(pw)/len(pt)*100:.0f}%)")
        print(f"    PnL net  : {sum(t['pnl'] for t in pt):+.2f} €")
        print(f"    Gain moy : +€{sum(t['pnl'] for t in pw)/max(len(pw),1):.2f}")
        print(f"    Score moy: {sum(t['score'] for t in pt)/len(pt):.0f}")
        print(f"    ADX moy  : {sum(t['adx'] for t in pt)/len(pt):.1f}")

    # Par crypto (top 10)
    print(f"\n{sep}")
    print("  Performance par crypto (top 10 PnL)")
    sym_stats = {}
    for t in trades:
        s = t["sym"].replace("-USDT","")
        if s not in sym_stats:
            sym_stats[s] = {"pnl": 0, "n": 0, "wins": 0}
        sym_stats[s]["pnl"]  += t["pnl"]
        sym_stats[s]["n"]    += 1
        sym_stats[s]["wins"] += 1 if t["pnl"] > 0 else 0
    for s, st in sorted(sym_stats.items(), key=lambda x: -x[1]["pnl"])[:10]:
        wr = st["wins"] / st["n"] * 100
        bar_len = int(st["pnl"] / 5) if st["pnl"] > 0 else 0
        bar_str = "█" * min(bar_len, 20)
        print(f"  {s:<8} {st['pnl']:>+8.2f} €  {wr:>4.0f}% WR  {st['n']:>3} trades  {bar_str}")

    # Par heure (session)
    print(f"\n{sep}")
    print("  Performance par heure d'entrée (UTC)")
    hour_stats = {}
    for t in trades:
        h = t.get("hour", t["entry_ts"].hour if hasattr(t["entry_ts"], "hour") else 0)
        if h not in hour_stats:
            hour_stats[h] = {"pnl": 0, "n": 0, "wins": 0}
        hour_stats[h]["pnl"]  += t["pnl"]
        hour_stats[h]["n"]    += 1
        hour_stats[h]["wins"] += 1 if t["pnl"] > 0 else 0

    sessions = {"Londres 07-16h": range(7,17), "New York 13-22h": range(13,23), "Asie 00-09h": range(0,9)}
    for sess_name, sess_range in sessions.items():
        st = [t for t in trades if t.get("hour", 0) in sess_range]
        if not st:
            continue
        sw = [t for t in st if t["pnl"] > 0]
        print(f"  {sess_name:<20}  {len(st):>3} trades  {len(sw)/len(st)*100:>4.0f}% WR  {sum(t['pnl'] for t in st):>+8.2f} €")

    # Top 5 meilleurs et pires trades
    sorted_t = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    print(f"\n{sep}")
    print("  5 meilleurs trades")
    for t in sorted_t[:5]:
        sym = t["sym"].replace("-USDT","")
        print(f"  {sym:<8} {t['side']:<5} Pat.{t.get('pattern','?')}  {t['pnl']:>+8.2f} €  score={t['score']}  {str(t['entry_ts'])[:16]}")
    print(f"\n  5 pires trades")
    for t in sorted_t[-5:]:
        sym = t["sym"].replace("-USDT","")
        print(f"  {sym:<8} {t['side']:<5} Pat.{t.get('pattern','?')}  {t['pnl']:>+8.2f} €  score={t['score']}  {str(t['entry_ts'])[:16]}")

    print(f"\n{'═'*62}\n")

    # Sauvegarder les trades en CSV
    ts_str  = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = RESULT_DIR / f"backtest_{months}m_{ts_str}.csv"
    import csv
    keys = ["sym","side","pattern","score","adx","leverage","margin",
            "entry_ts","entry_price","exit_ts","exit_price","reason","pnl","equity_after"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)
    print(f"  Trades exportés → {out_csv}\n")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",     type=int,  default=6)
    parser.add_argument("--no-fetch",   action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    print(f"\n  PRISM v33 — Backtester  ({args.months} mois)")
    print(f"  {'─'*40}")

    # Téléchargement
    sym_data = {}
    print(f"  Téléchargement données ({len(SYMBOLS)} symboles)...")
    def load_sym(sym):
        df = load_or_fetch(sym, args.months, args.no_fetch)
        if df is not None:
            print(f"    {sym:<14} {len(df):>5} barres")
            return sym, prepare(sym, df)
        else:
            print(f"    {sym:<14} ERREUR")
            return sym, None

    # Séquentiel pour éviter rate-limit OKX (history-candles = 10 req/s shared)
    for sym in SYMBOLS:
        s, sd = load_sym(sym)
        if sd is not None:
            sym_data[s] = sd

    print(f"\n  {len(sym_data)}/{len(SYMBOLS)} symboles chargés")

    if args.fetch_only:
        print("  --fetch-only : téléchargement terminé.\n")
        return

    if "BTC-USDT" not in sym_data:
        print("  ERREUR : BTC-USDT manquant — abandon.\n")
        return

    # Plage de dates
    btc_ts = sym_data["BTC-USDT"]["ts_index"]
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=args.months)
    # Utiliser uniquement les symboles ayant au moins 1000 barres pour le start_ts
    rich_sds = [sd for sd in sym_data.values() if len(sd["ts_index"]) >= 1000]
    warmup_limit = max(sd["ts_index"][300] for sd in rich_sds) if rich_sds else cutoff
    start_ts = max(cutoff, warmup_limit)
    end_ts = pd.Timestamp(btc_ts[-2])  # -2 pour éviter barre en cours

    # Avertir sur les symboles avec peu de données
    short_syms = [sym for sym, sd in sym_data.items() if len(sd["ts_index"]) < 1000]
    if short_syms:
        print(f"  [INFO] Données courtes (<1000 barres): {', '.join(short_syms)}")

    n_bars = len([ts for ts in btc_ts if start_ts <= ts <= end_ts])
    print(f"  Période test : {start_ts.date()} → {end_ts.date()}  ({n_bars} barres)")

    # ── Baseline (sans trailing) ──────────────────────────────────────────────
    print(f"\n  [1/10] Baseline (TP fixe 4×ATR, pas de trailing)...")
    t0 = time.time()
    trades0, curve0, eq0 = run_backtest(sym_data, start_ts, end_ts)
    print(f"  Terminé en {time.time()-t0:.1f}s — {len(trades0)} trades")
    print_report(trades0, curve0, eq0, args.months, start_ts, end_ts)

    # ── Grid Search Trailing ATR ──────────────────────────────────────────────
    # activation_atr × trail_distance_atr : 9 combinaisons
    grid = [
        (0.8, 1.0), (0.8, 1.5), (0.8, 2.0),
        (1.2, 1.0), (1.2, 1.5), (1.2, 2.0),
        (1.8, 1.0), (1.8, 1.5), (1.8, 2.0),
    ]

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH — TRAILING ATR  ({len(grid)} combinaisons)")
    print(f"{'='*70}")

    def _stats(trades, eq_curve, final_eq):
        n     = len(trades)
        wins  = [t for t in trades if t["pnl"] > 0]
        losses= [t for t in trades if t["pnl"] <= 0]
        wr    = len(wins) / n * 100 if n else 0
        avg_w = sum(t["pnl"] for t in wins)  / max(len(wins),  1)
        avg_l = sum(t["pnl"] for t in losses) / max(len(losses), 1)
        pf    = abs(avg_w / avg_l) if avg_l != 0 else 999
        pnl   = sum(t["pnl"] for t in trades)
        trail_exits = [t for t in trades if t.get("reason") == "trail_stop"]
        tp_exits    = [t for t in trades if t.get("reason") == "take_profit"]
        peak = INITIAL_CAPITAL
        maxdd = 0.0
        for e in eq_curve:
            peak  = max(peak, e["equity"])
            maxdd = max(maxdd, (peak - e["equity"]) / peak * 100)
        return dict(n=n, wr=wr, pf=pf, pnl=pnl, final=final_eq,
                    maxdd=maxdd, n_trail=len(trail_exits), n_tp=len(tp_exits),
                    avg_w=avg_w, avg_l=avg_l)

    base = _stats(trades0, curve0, eq0)

    # En-tête du tableau
    hdr = f"  {'Act×ATR':>7} {'Trl×ATR':>7} {'Trades':>6} {'WR%':>5} {'PF':>5} {'PnL €':>8} {'Final €':>8} {'MaxDD%':>7} {'TrailExit':>9} {'TP':>5}"
    print(hdr)
    print(f"  {'-'*80}")

    def _row(label, st):
        return (f"  {label:<15} {st['n']:>6} {st['wr']:>5.1f} {st['pf']:>5.2f} "
                f"{st['pnl']:>+8.0f} {st['final']:>8.0f} {st['maxdd']:>7.1f}% "
                f"{st['n_trail']:>9} {st['n_tp']:>5}")

    print(_row("Baseline      ", base))
    print(f"  {'-'*80}")

    best_pf, best_cfg = base["pf"], "baseline"
    results = []

    for i, (act, trl) in enumerate(grid, 2):
        t_, c_, e_ = run_backtest(sym_data, start_ts, end_ts,
                                  activation_atr=act, trail_distance_atr=trl)
        st = _stats(t_, c_, e_)
        results.append((act, trl, st))
        label = f"act={act:.1f} trl={trl:.1f}"
        marker = " <-- BEST PF" if st["pf"] > best_pf else ""
        print(_row(label, st) + marker)
        if st["pf"] > best_pf:
            best_pf, best_cfg = st["pf"], label

    print(f"\n  Meilleure config trailing : {best_cfg}  (PF={best_pf:.2f})")

    # ── Grid Search Qualité Entrée (Score × ADX) ─────────────────────────────
    entry_grid = [
        (68, 22), (68, 25), (68, 28),
        (70, 22), (70, 25), (70, 28),
        (72, 22), (72, 25), (72, 28),
        (74, 22), (74, 25), (74, 28),
        (76, 22), (76, 25), (76, 28),
    ]

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH — QUALITE ENTREE PATTERN D  (score x ADX)")
    print(f"  Objectif : WR Pattern D > 35%  |  Baseline D : 24% WR, +26 EUR")
    print(f"{'='*70}")

    def _stats_d(trades):
        td = [t for t in trades if t.get("pattern") == "D"]
        n  = len(td)
        if n == 0:
            return dict(n=0, wr=0, pf=0, pnl=0)
        wins  = [t for t in td if t["pnl"] > 0]
        losses= [t for t in td if t["pnl"] <= 0]
        wr    = len(wins) / n * 100
        avg_w = sum(t["pnl"] for t in wins)  / max(len(wins), 1)
        avg_l = sum(t["pnl"] for t in losses) / max(len(losses), 1)
        pf    = abs(avg_w / avg_l) if avg_l != 0 else 999
        return dict(n=n, wr=wr, pf=pf, pnl=sum(t["pnl"] for t in td))

    def _stats_total(trades, eq_curve, final_eq):
        n    = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        wr   = len(wins) / n * 100 if n else 0
        pnl  = sum(t["pnl"] for t in trades)
        peak = INITIAL_CAPITAL; maxdd = 0.0
        for e in eq_curve:
            peak  = max(peak, e["equity"])
            maxdd = max(maxdd, (peak - e["equity"]) / peak * 100)
        return dict(n=n, wr=wr, pnl=pnl, final=final_eq, maxdd=maxdd)

    hdr2 = (f"  {'Sc':>3} {'ADX':>4} {'D trades':>8} {'D WR%':>6} {'D PF':>5} "
            f"{'D PnL':>7} {'Tot trades':>10} {'Tot PnL':>8} {'Final':>8} {'MaxDD%':>7}")
    print(hdr2)
    print(f"  {'-'*78}")

    best_entry_pf, best_entry_cfg = 0.0, ""
    entry_results = []

    for sc, adx in entry_grid:
        t_, c_, e_ = run_backtest(sym_data, start_ts, end_ts, score_min_d=sc, adx_min_d=adx)
        sd_  = _stats_d(t_)
        st_  = _stats_total(t_, c_, e_)
        entry_results.append((sc, adx, sd_, st_))
        marker = " <-- BEST D PF" if sd_["pf"] > best_entry_pf and sd_["n"] >= 20 else ""
        print(f"  {sc:>3}  {adx:>4}  {sd_['n']:>8}  {sd_['wr']:>6.1f}  {sd_['pf']:>5.2f} "
              f"  {sd_['pnl']:>+7.0f}  {st_['n']:>10}  {st_['pnl']:>+8.0f}  "
              f"{st_['final']:>8.0f}  {st_['maxdd']:>7.1f}%{marker}")
        if sd_["pf"] > best_entry_pf and sd_["n"] >= 20:
            best_entry_pf  = sd_["pf"]
            best_entry_cfg = f"score={sc} adx={adx}"

    print(f"\n  Meilleure config entree : {best_entry_cfg}  (D PF={best_entry_pf:.2f})")

    # ── Rapport détaillé de la meilleure config entrée ────────────────────────
    best_sc, best_adx, _, _ = max(
        [r for r in entry_results if r[2]["n"] >= 20],
        key=lambda x: x[2]["pf"]
    )
    print(f"\n{'='*70}")
    print(f"  RAPPORT DÉTAILLÉ — score_min_d={best_sc}  adx_min_d={best_adx}")
    print(f"{'='*70}")
    t_best, c_best, e_best = run_backtest(sym_data, start_ts, end_ts,
                                          score_min_d=best_sc, adx_min_d=best_adx)
    print_report(t_best, c_best, e_best, args.months, start_ts, end_ts)

if __name__ == "__main__":
    main()
