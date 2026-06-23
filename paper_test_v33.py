#!/usr/bin/env python3
"""
PRISM v33 — Exclusion STX-USDT | Purification univers de trading
=================================================================
Base : v32 (23 symboles, two-pass, margin cap 60%, score_min 70/65/60)
Delta v33 vs v32 :
  - Suppression STX-USDT → 22 symboles
  - STX : 2 trades, 0% WR sur Optimal et Agressif
    Optimal : -54.73€ | Agressif : -87.03€
  - Même critère d'exclusion qu'APT, WLD, FLOKI (2T / 0% WR)
  - STX = Bitcoin L2, comportement corrélé BTC mais IVOL élevée
    sur rebonds — les BB Squeeze se terminent en faux breakouts

Historique des exclusions :
  v27 : FIL, ETH, ALGO  →  WR 35% → 43%
  v28 : SOL
  v30 : APT, WLD        →  +97% → +122%
  v32 : FLOKI           →  +106% → +123% / Sharpe 2.81 → 3.26
  v33 : STX             →  ?
"""

import math, os, pickle, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = [
    # v33 : 22 symboles (v32 moins STX-USDT 0%WR/2T, -54€ Optimal)
    "BTC-USDT", "AVAX-USDT",
    "ADA-USDT", "LINK-USDT", "XRP-USDT", "DOT-USDT", "ATOM-USDT",
    "LTC-USDT", "DOGE-USDT", "NEAR-USDT", "TRX-USDT",
    "INJ-USDT", "OP-USDT",
    "ARB-USDT", "SUI-USDT", "UNI-USDT", "AAVE-USDT",
    "TIA-USDT", "SEI-USDT", "HBAR-USDT", "ICP-USDT",
    "JUP-USDT",
]

INITIAL_CAPITAL  = 1000.0
TIMEFRAME        = "1H"
MONTHS           = 6
WARMUP           = 250
COMMISSION       = 0.001
SLIPPAGE         = 0.0005
EXIT_SLIPPAGE    = 0.0003

# Fenêtre de backtest fixe — indépendante de la date de construction du cache
BACKTEST_START   = pd.Timestamp("2026-01-01")

# ATR Dynamic Stop-Loss (identique v22)
ATR_SL_MULT      = 1.5
RR_RATIO         = 4.0
ATR_SL_MIN_C     = 0.006
ATR_SL_MAX_C     = 0.025

# Pattern C — BB Squeeze 1H (identique v22 sauf COOLDOWN)
SQUEEZE_BARS_C   = 4
VOL_RATIO_C      = 1.5
ADX_MIN_C        = 20
TIME_STOP_H      = 72
COOLDOWN_BARS    = 5       # v22: 8 → v30: 5 (plus de signaux)

# Levier (identique v22 — prouvé supérieur)
BASE_LEVERAGE    = 10
HIGH_LEVERAGE    = 15      # ADX > 28 ET score >= 72

DAILY_LOSS_CAP   = 0.12
MAX_MARGIN_RATIO = 0.60  # cap total margin engagée (open + pending) à 60% equity

CONFIGS = [
    {"name": "Prudent",   "risk_pct": 0.06, "max_pos": 2, "score_min": 70},
    {"name": "Optimal",   "risk_pct": 0.10, "max_pos": 4, "score_min": 65},
    {"name": "Agressif",  "risk_pct": 0.14, "max_pos": 5, "score_min": 60},
]

CACHE_DIR       = os.path.join(os.path.dirname(__file__), ".cache_v20")
CACHE_MAX_AGE_H = 23
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"


# ---------------------------------------------------------------------------
# Download (réutilise cache v20/v22)
# ---------------------------------------------------------------------------

def _cache_path(sym: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = sym.replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{TIMEFRAME}_{MONTHS}m.pkl")


def _cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 3600 < CACHE_MAX_AGE_H


def _fetch_okx_raw(inst_id: str) -> list:
    until_ms = int(time.time() * 1000)
    since_ms  = until_ms - MONTHS * 30 * 24 * 60 * 60 * 1000
    all_rows, after = [], None
    for _ in range(200):
        params = {"instId": inst_id, "bar": TIMEFRAME, "limit": 100}
        if after:
            params["after"] = after
        try:
            r    = requests.get(OKX_CANDLES_URL, params=params, timeout=15)
            data = r.json()
        except Exception:
            break
        if data.get("code") != "0" or not data.get("data"):
            break
        batch = data["data"]
        all_rows.extend(batch)
        if int(batch[-1][0]) <= since_ms:
            break
        after = batch[-1][0]
        time.sleep(0.08)
    all_rows = [r for r in all_rows if int(r[0]) >= since_ms]
    all_rows.sort(key=lambda x: int(x[0]))
    return all_rows


def download_ohlcv(sym: str, force: bool = False) -> Optional[pd.DataFrame]:
    path = _cache_path(sym)
    if not force and _cache_valid(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    raw = _fetch_okx_raw(sym)
    if len(raw) < WARMUP + 50:
        return None
    df = pd.DataFrame(raw, columns=[
        "timestamp","open","high","low","close","volume","vol_ccy","vol_quote","confirm"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="ms", utc=True
    ).dt.tz_localize(None)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["close","open","high","low","volume"])
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


# ---------------------------------------------------------------------------
# Indicators (identiques v22)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ml    = ema12 - ema26
    ms    = ml.ewm(span=9, adjust=False).mean()
    df["macd_hist"]  = ml - ms
    df["macd_slope"] = (ml - ms).diff()

    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    bb_mid        = close.rolling(20).mean()
    bb_std        = close.rolling(20).std()
    df["bb_upper"]= bb_mid + 2 * bb_std
    df["bb_lower"]= bb_mid - 2 * bb_std
    bbw           = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10)
    df["bbw"]     = bbw
    df["bbw_q15"] = bbw.rolling(40).quantile(0.15)

    df["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)

    tp_val = (high + low + close) / 3
    df["vwap"] = (tp_val * volume).rolling(24).sum() / (volume.rolling(24).sum() + 1e-10)

    low14   = low.rolling(14).min()
    high14  = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_p = (high - high.shift()).clip(lower=0)
    dm_m = (low.shift() - low).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0)
    dm_m = dm_m.where(dm_m > dm_p, 0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    dip   = 100 * dm_p.ewm(com=13, adjust=False).mean() / (atr14 + 1e-10)
    dim   = 100 * dm_m.ewm(com=13, adjust=False).mean() / (atr14 + 1e-10)
    dx    = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    df["adx"]      = dx.ewm(com=13, adjust=False).mean()
    df["di_plus"]  = dip
    df["di_minus"] = dim
    df["atr14"]    = atr14

    df_4h    = df[["close"]].resample("4h").last().dropna()
    ema20_4h = df_4h["close"].ewm(span=20, adjust=False).mean()
    ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
    df["ema20_4h"] = ema20_4h.reindex(df.index, method="ffill")
    df["ema50_4h"] = ema50_4h.reindex(df.index, method="ffill")

    df_1d   = df[["close"]].resample("1D").last().dropna()
    ema50d  = df_1d["close"].ewm(span=50,  adjust=False).mean()
    ema200d = df_1d["close"].ewm(span=200, adjust=False).mean()
    df["ema50d"]  = ema50d.reindex(df.index,  method="ffill")
    df["ema200d"] = ema200d.reindex(df.index, method="ffill")

    return df


# ---------------------------------------------------------------------------
# Score (identique v22)
# ---------------------------------------------------------------------------

def _compute_scores(sd: dict, n: int):
    buy_sc  = np.zeros(n, dtype=np.int32)
    sell_sc = np.zeros(n, dtype=np.int32)
    close     = sd["close"]
    ema9      = sd["ema9"]
    ema21     = sd["ema21"]
    ema50     = sd["ema50"]
    rsi       = sd["rsi14"]
    mh        = sd["macd_hist"]
    mh_slope  = sd["macd_slope"]
    vol_ratio = sd["vol_ratio"]
    adx_arr   = sd["adx"]
    vwap      = sd["vwap"]
    stoch_k   = sd["stoch_k"]
    stoch_d   = sd["stoch_d"]

    for i in range(n):
        bs = ss = 0
        if not any(math.isnan(v) for v in [ema9[i], ema21[i], ema50[i]]):
            if ema9[i] > ema21[i]: bs += 12
            elif ema9[i] < ema21[i]: ss += 12
            if ema21[i] > ema50[i]: bs += 13
            elif ema21[i] < ema50[i]: ss += 13
        if not math.isnan(rsi[i]):
            r = rsi[i]
            if 40 <= r <= 65:   bs += 15
            elif 35 <= r < 40:  bs += 8
            elif 65 < r <= 70:  bs += 5
            if 35 <= r <= 60:   ss += 15
            elif 60 < r <= 65:  ss += 8
            elif 30 <= r < 35:  ss += 5
        if not any(math.isnan(v) for v in [mh[i], mh_slope[i]]):
            if mh[i] > 0: bs += 12
            elif mh[i] < 0: ss += 12
            if mh_slope[i] > 0: bs += 8
            elif mh_slope[i] < 0: ss += 8
        if not math.isnan(vol_ratio[i]):
            vr = vol_ratio[i]
            pts = 10 if vr >= 1.5 else 6 if vr >= 1.0 else 3 if vr >= 0.7 else 0
            bs += pts; ss += pts
        if not math.isnan(adx_arr[i]):
            av = adx_arr[i]
            pts = 10 if av >= 25 else 6 if av >= 18 else 0
            bs += pts; ss += pts
        if not any(math.isnan(v) for v in [close[i], vwap[i]]):
            if close[i] > vwap[i]:   bs += 10
            elif close[i] < vwap[i]: ss += 10
        if not any(math.isnan(v) for v in [stoch_k[i], stoch_d[i]]):
            sk, sd_ = stoch_k[i], stoch_d[i]
            if sk > sd_ and sk < 75: bs += 10
            if sk < sd_ and sk > 25: ss += 10
        buy_sc[i]  = min(bs, 100)
        sell_sc[i] = min(ss, 100)

    return buy_sc, sell_sc


def precompute(sym: str) -> Optional[dict]:
    df = download_ohlcv(sym)
    if df is None:
        return None
    df = compute_indicators(df)
    n  = len(df)
    sd = {
        "name":      sym,
        "ts_index":  df.index,
        "ts_to_pos": {ts: i for i, ts in enumerate(df.index)},
        "open":      df["open"].values.astype(float),
        "close":     df["close"].values.astype(float),
        "high":      df["high"].values.astype(float),
        "low":       df["low"].values.astype(float),
        "ema9":      df["ema9"].values.astype(float),
        "ema21":     df["ema21"].values.astype(float),
        "ema50":     df["ema50"].values.astype(float),
        "rsi14":     df["rsi14"].values.astype(float),
        "macd_hist": df["macd_hist"].values.astype(float),
        "macd_slope":df["macd_slope"].values.astype(float),
        "vol_ratio": df["vol_ratio"].values.astype(float),
        "vwap":      df["vwap"].values.astype(float),
        "stoch_k":   df["stoch_k"].values.astype(float),
        "stoch_d":   df["stoch_d"].values.astype(float),
        "adx":       df["adx"].values.astype(float),
        "atr14":     df["atr14"].values.astype(float),
        "bb_upper":  df["bb_upper"].values.astype(float),
        "bb_lower":  df["bb_lower"].values.astype(float),
        "bbw":       df["bbw"].values.astype(float),
        "bbw_q15":   df["bbw_q15"].values.astype(float),
        "ema20_4h":  df["ema20_4h"].values.astype(float),
        "ema50_4h":  df["ema50_4h"].values.astype(float),
        "ema50d":    df["ema50d"].values.astype(float),
        "ema200d":   df["ema200d"].values.astype(float),
    }
    buy_sc, sell_sc = _compute_scores(sd, n)
    sd["buy_sc"]  = buy_sc
    sd["sell_sc"] = sell_sc
    return sd


# ---------------------------------------------------------------------------
# Pattern C — BB Squeeze Breakout 1H (identique v22)
# ---------------------------------------------------------------------------

def _check_pattern_c(sd: dict, bar: int, adx_val: float) -> Optional[str]:
    if bar < SQUEEZE_BARS_C + 3:
        return None
    try:
        bbw_arr  = sd["bbw"]
        bbwq_arr = sd["bbw_q15"]
        bbw_cur  = bbw_arr[bar]
        bbwq_cur = bbwq_arr[bar]

        if math.isnan(bbw_cur) or math.isnan(bbwq_cur):
            return None

        for i in range(1, SQUEEZE_BARS_C + 1):
            bw = bbw_arr[bar - i]
            bq = bbwq_arr[bar - i]
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

        if any(math.isnan(v) for v in [close, bb_upper, bb_lower, vol_r,
                                        ema20_4h, ema50_4h]):
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


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol:      str
    side:        str
    pattern:     str
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    entry_price: float
    exit_price:  float
    margin_eur:  float
    pnl_eur:     float
    exit_reason: str
    leverage:    int
    score:       int


# ---------------------------------------------------------------------------
# Engine (identique v22 — levier ADX/score inchangé)
# ---------------------------------------------------------------------------

def run_engine(sym_data_list, timestamps, risk_pct, max_pos, score_min):
    equity           = INITIAL_CAPITAL
    open_positions   = {}
    pending_entries  = {}
    trades           = []
    eq_curve         = []
    day_wins         = {}
    current_day      = ""
    day_start_equity = INITIAL_CAPITAL
    equity_peak      = INITIAL_CAPITAL
    cooldown_tracker = {}
    sym_lookup = {sd["name"]: sd for sd in sym_data_list}
    btc_sd = sym_lookup.get("BTC-USDT")

    for ts in timestamps:
        eq_curve.append(equity)
        ts_day = str(ts)[:10]

        if ts_day != current_day:
            current_day      = ts_day
            day_start_equity = equity
            day_wins.setdefault(ts_day, {"wins": 0, "losses": 0})

        # ---- Execute pending entries ----
        for pk in list(pending_entries.keys()):
            if len(open_positions) >= max_pos:
                break
            p   = pending_entries.pop(pk)
            sd  = sym_lookup.get(p["sym"])
            if sd is None:
                continue
            bar = sd["ts_to_pos"].get(ts)
            if bar is None:
                continue
            open_px     = float(sd["open"][bar])
            side        = p["side"]
            entry_price = (open_px * (1 + SLIPPAGE) if side == "long"
                           else open_px * (1 - SLIPPAGE))
            atr_now = sd["atr14"][bar]
            if math.isnan(atr_now) or atr_now <= 0:
                atr_now = entry_price * 0.015
            atr_pct = atr_now / (entry_price + 1e-10)
            sl_pct  = max(ATR_SL_MIN_C, min(ATR_SL_MAX_C, ATR_SL_MULT * atr_pct))
            tp_pct  = sl_pct * RR_RATIO
            sl = (entry_price * (1 - sl_pct) if side == "long"
                  else entry_price * (1 + sl_pct))
            tp = (entry_price * (1 + tp_pct) if side == "long"
                  else entry_price * (1 - tp_pct))
            open_positions[p["sym"] + p["pattern"] + str(ts)] = {
                "sym":         p["sym"],
                "side":        side,
                "pattern":     p["pattern"],
                "entry_ts":    ts,
                "entry_price": entry_price,
                "sl":          sl,
                "tp":          tp,
                "margin":      p["margin"],
                "leverage":    p["leverage"],
                "score":       p["score"],
                "sl_pct":      sl_pct,
                "tp_pct":      tp_pct,
            }

        day_pnl_pct  = (equity - day_start_equity) / (day_start_equity + 1e-10)
        skip_entries = day_pnl_pct <= -DAILY_LOSS_CAP

        if equity > equity_peak:
            equity_peak = equity
        drawdown_from_peak = (equity_peak - equity) / (equity_peak + 1e-10)

        # ---- Process exits ----
        to_remove = []
        for pos_key, pos in list(open_positions.items()):
            sym = pos["sym"]
            sd  = sym_lookup.get(sym)
            if sd is None:
                continue
            bar = sd["ts_to_pos"].get(ts)
            if bar is None:
                continue

            h  = sd["high"][bar]
            lo = sd["low"][bar]
            cl = sd["close"][bar]
            entry  = pos["entry_price"]
            sl     = pos["sl"]
            tp     = pos["tp"]
            side   = pos["side"]

            exit_price  = None
            exit_reason = None

            if side == "long":
                if h >= tp:    exit_price, exit_reason = tp, "take_profit"
                elif lo <= sl: exit_price, exit_reason = sl, "stop_loss"
            else:
                if lo <= tp:   exit_price, exit_reason = tp, "take_profit"
                elif h >= sl:  exit_price, exit_reason = sl, "stop_loss"

            elapsed_h = (ts - pos["entry_ts"]).total_seconds() / 3600
            if exit_price is None and elapsed_h >= TIME_STOP_H:
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
                day_wins[ts_day]["wins" if net_pnl > 0 else "losses"] += 1

                trades.append(Trade(
                    symbol      = sym,
                    side        = side,
                    pattern     = pos["pattern"],
                    entry_ts    = pos["entry_ts"],
                    exit_ts     = ts,
                    entry_price = entry,
                    exit_price  = exit_price,
                    margin_eur  = pos["margin"],
                    pnl_eur     = net_pnl,
                    exit_reason = exit_reason,
                    leverage    = pos["leverage"],
                    score       = pos["score"],
                ))
                to_remove.append(pos_key)

        for k in to_remove:
            del open_positions[k]

        # ---- New entries — Two-Pass Priority Allocation ----
        if skip_entries or len(open_positions) + len(pending_entries) >= max_pos:
            continue
        if drawdown_from_peak > 0.40:
            continue

        # BTC 4H macro filter
        btc_4h_bull = None
        if btc_sd is not None:
            btc_bar = btc_sd["ts_to_pos"].get(ts)
            if btc_bar is not None:
                b20 = btc_sd["ema20_4h"][btc_bar]
                b50 = btc_sd["ema50_4h"][btc_bar]
                if not (math.isnan(b20) or math.isnan(b50)):
                    btc_4h_bull = bool(b20 > b50)

        # PASS 1 : scanner TOUS les symboles, collecter les candidats valides
        # Le cooldown n'est PAS consommé ici — seulement lors de l'allocation
        candidates = []
        for sd in sym_data_list:
            sym = sd["name"]
            if sym == "BTC-USDT":
                continue
            if sym + "C" in pending_entries:
                continue

            bar = sd["ts_to_pos"].get(ts)
            if bar is None or bar < WARMUP:
                continue

            adx_val = float(sd["adx"][bar])
            if math.isnan(adx_val):
                continue

            ck = sym + "C"
            if bar - cooldown_tracker.get(ck, -9999) < COOLDOWN_BARS:
                continue

            action = _check_pattern_c(sd, bar, adx_val)
            if action is None:
                continue

            if btc_4h_bull is not None:
                if action == "BUY"  and not btc_4h_bull:
                    continue
                if action == "SELL" and btc_4h_bull:
                    continue

            score = int(sd["buy_sc"][bar]) if action == "BUY" else int(sd["sell_sc"][bar])
            if score < score_min:
                continue

            leverage = HIGH_LEVERAGE if (adx_val > 28 and score >= 72) else BASE_LEVERAGE
            candidates.append({
                "sym":     sym,
                "ck":      ck,
                "bar":     bar,
                "action":  action,
                "score":   score,
                "adx":     adx_val,
                "leverage":leverage,
            })

        if not candidates:
            continue

        # PASS 2 : trier par score décroissant → meilleurs signaux en priorité
        # En cas d'égalité de score, ADX décroissant comme critère secondaire
        candidates.sort(key=lambda c: (c["score"], c["adx"]), reverse=True)

        # Margin cap : total engagé (open + pending) ≤ MAX_MARGIN_RATIO × equity
        dd_scale = max(0.5, 1.0 - drawdown_from_peak * 2.5)
        margin_per_trade = equity * risk_pct * dd_scale

        total_margin_engaged = (
            sum(p["margin"] for p in pending_entries.values()) +
            sum(pos["margin"] for pos in open_positions.values())
        )
        max_margin_allowed = equity * MAX_MARGIN_RATIO

        for c in candidates:
            if len(open_positions) + len(pending_entries) >= max_pos:
                break
            # Approach B : respecter le cap de margin totale
            if total_margin_engaged + margin_per_trade > max_margin_allowed:
                break

            pending_entries[c["sym"] + "C"] = {
                "sym":     c["sym"],
                "side":    "long" if c["action"] == "BUY" else "short",
                "pattern": "C",
                "margin":  margin_per_trade,
                "leverage":c["leverage"],
                "score":   c["score"],
            }
            # Cooldown consommé seulement maintenant — signal réellement alloué
            cooldown_tracker[c["ck"]] = c["bar"]
            total_margin_engaged += margin_per_trade

    return trades, eq_curve, day_wins


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(trades, eq_curve) -> dict:
    if not eq_curve:
        return {"n_trades": 0, "ret": 0.0, "max_dd": 0.0,
                "win_rate": 0.0, "net_pnl": 0.0, "final_equity": INITIAL_CAPITAL,
                "profit_factor": 0.0, "sharpe": 0.0}

    final_eq = eq_curve[-1]
    ret      = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
    peak     = INITIAL_CAPITAL
    max_dd   = 0.0
    for e in eq_curve:
        if e > peak: peak = e
        dd = (peak - e) / (peak + 1e-10)
        if dd > max_dd: max_dd = dd

    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "ret": ret, "max_dd": max_dd,
                "win_rate": 0.0, "net_pnl": 0.0, "final_equity": final_eq,
                "profit_factor": 0.0, "sharpe": 0.0}

    wins       = [t for t in trades if t.pnl_eur > 0]
    win_rate   = len(wins) / n
    net_pnl    = sum(t.pnl_eur for t in trades)
    gross_win  = sum(t.pnl_eur for t in wins)
    gross_loss = abs(sum(t.pnl_eur for t in trades if t.pnl_eur <= 0))
    pf         = gross_win / (gross_loss + 1e-10)

    monthly: dict = {}
    for t in trades:
        mk = t.exit_ts.strftime("%Y-%m")
        monthly[mk] = monthly.get(mk, 0) + t.pnl_eur
    mrs = list(monthly.values())
    if len(mrs) > 1:
        avg    = sum(mrs) / len(mrs)
        std    = (sum((r - avg) ** 2 for r in mrs) / len(mrs)) ** 0.5
        sharpe = (avg / (std + 1e-10)) * (12 ** 0.5)
    else:
        sharpe = 0.0

    return {
        "n_trades":      n,
        "ret":           ret,
        "max_dd":        max_dd,
        "win_rate":      win_rate,
        "net_pnl":       net_pnl,
        "profit_factor": pf,
        "final_equity":  final_eq,
        "sharpe":        sharpe,
        "monthly":       monthly,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(trades, eq_curve, cfg_name: str, t0: float):
    m  = _metrics(trades, eq_curve)
    rc = "green" if m["ret"] >= 0 else "red"

    console.print(f"\n[bold cyan]━━━ PRISM v33 | 22 Symboles | Config: {cfg_name} ━━━[/bold cyan]")
    console.print(
        f"  Capital: €{INITIAL_CAPITAL:.0f} → [bold]€{m['final_equity']:.2f}[/bold]  "
        f"Return: [{rc}]{m['ret']:+.1%}[/]  "
        f"MaxDD: [red]{m['max_dd']:.1%}[/]  "
        f"WinRate: {m['win_rate']:.0%}  "
        f"PF: {m.get('profit_factor',0):.2f}  "
        f"Sharpe: {m.get('sharpe',0):.2f}  "
        f"Trades: {m['n_trades']}"
    )

    monthly = m.get("monthly", {})
    if monthly:
        total_trades_per_month = {}
        for t in trades:
            mk = t.exit_ts.strftime("%Y-%m")
            total_trades_per_month[mk] = total_trades_per_month.get(mk, 0) + 1

        mtbl = Table(title=f"Mensuel — {MONTHS} mois | {len(SYMBOLS)} symboles", box=box.SIMPLE_HEAD)
        mtbl.add_column("Mois",   style="cyan")
        mtbl.add_column("Trades", justify="right")
        mtbl.add_column("Win%",   justify="right")
        mtbl.add_column("PnL €",  justify="right")
        mtbl.add_column("Return", justify="right")
        eq_running = INITIAL_CAPITAL
        for mk in sorted(monthly):
            tl  = [t for t in trades if t.exit_ts.strftime("%Y-%m") == mk]
            ws  = sum(1 for t in tl if t.pnl_eur > 0)
            wr  = ws / len(tl) if tl else 0
            pnl = monthly[mk]
            ret = pnl / (eq_running + 1e-10)
            eq_running += pnl
            col = "green" if pnl >= 0 else "red"
            mtbl.add_row(mk, str(len(tl)), f"{wr:.0%}",
                         f"[{col}]{pnl:+.2f}[/]", f"[{col}]{ret:+.1%}[/]")
        n = len(trades)
        aw = sum(1 for t in trades if t.pnl_eur > 0)
        mtbl.add_row("[bold]TOTAL[/bold]", f"[bold]{n}[/bold]",
                     f"[bold]{aw/n:.0%}[/bold]" if n else "—",
                     f"[bold][{rc}]{m['net_pnl']:+.2f}[/][/bold]",
                     f"[bold][{rc}]{m['ret']:+.1%}[/][/bold]")
        console.print(mtbl)

    # Exit breakdown
    if trades:
        etbl = Table(title="Sorties", box=box.SIMPLE_HEAD)
        etbl.add_column("Raison",  style="cyan")
        etbl.add_column("Count",   justify="right")
        etbl.add_column("Win%",    justify="right")
        etbl.add_column("Avg PnL", justify="right")
        for reason in ["take_profit", "stop_loss", "time_stop"]:
            tl = [t for t in trades if t.exit_reason == reason]
            if not tl: continue
            ws  = sum(1 for t in tl if t.pnl_eur > 0)
            avg = sum(t.pnl_eur for t in tl) / len(tl)
            col = "green" if avg >= 0 else "red"
            etbl.add_row(reason, str(len(tl)), f"{ws/len(tl):.0%}",
                         f"[{col}]{avg:+.2f}[/]")
        console.print(etbl)

    # Rapport par symbole — pour identifier les symboles à retirer
    if trades:
        sym_pnl: dict = {}
        sym_cnt: dict = {}
        sym_wins: dict = {}
        for t in trades:
            sym_pnl[t.symbol]  = sym_pnl.get(t.symbol, 0) + t.pnl_eur
            sym_cnt[t.symbol]  = sym_cnt.get(t.symbol, 0) + 1
            sym_wins[t.symbol] = sym_wins.get(t.symbol, 0) + (1 if t.pnl_eur > 0 else 0)
        ranked = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)

        stbl = Table(title="PnL par Symbole (v33 — 22 symboles)", box=box.SIMPLE_HEAD)
        stbl.add_column("Symbol",  style="cyan")
        stbl.add_column("Trades",  justify="right")
        stbl.add_column("Win%",    justify="right")
        stbl.add_column("PnL €",   justify="right")
        stbl.add_column("Avg €",   justify="right")
        stbl.add_column("Tag",     justify="left")
        for sym, pnl in ranked:
            cnt = sym_cnt[sym]
            wr  = sym_wins[sym] / cnt if cnt else 0
            avg = pnl / cnt if cnt else 0
            col = "green" if pnl >= 0 else "red"
            is_new = "NEW" if sym in [
                "ARB-USDT","APT-USDT","SUI-USDT","UNI-USDT","AAVE-USDT",
                "WLD-USDT","TIA-USDT","SEI-USDT","HBAR-USDT","ICP-USDT",
                "STX-USDT","FLOKI-USDT","JUP-USDT"
            ] else ""
            tag_col = "yellow" if is_new else "dim"
            stbl.add_row(sym, str(cnt), f"{wr:.0%}",
                         f"[{col}]{pnl:+.2f}[/]",
                         f"[{col}]{avg:+.2f}[/]",
                         f"[{tag_col}]{is_new}[/]")
        console.print(stbl)

        # Identifier les symboles dragging (< -€5 sur la période)
        losing_syms = [(s, p) for s, p in ranked if p < -5.0]
        if losing_syms:
            console.print("\n[yellow]  Symboles sous-performants (PnL < -€5):[/yellow]")
            for s, p in losing_syms:
                console.print(f"    [red]{s}[/] : {p:+.2f}€ ({sym_cnt[s]} trades)")
        else:
            console.print("\n[green]  Tous les symboles actifs sont profitables.[/green]")

    if monthly:
        pm = sum(1 for v in monthly.values() if v > 0)
        tm = len(monthly)
        avg_trades_pm = len(trades) / max(tm, 1)
        console.print(
            f"\n  Mois profitables: {pm}/{tm} ({pm/tm:.0%})  |  "
            f"Avg trades/mois: {avg_trades_pm:.1f}  |  "
            f"Elapsed: {time.time()-t0:.1f}s"
        )

    if len(eq_curve) > 10:
        console.print("\n[bold]Equity Curve[/bold]")
        sample = eq_curve[::max(1, len(eq_curve) // 70)]
        mn, mx = min(sample), max(sample)
        rng    = mx - mn + 1e-10
        rows   = 8
        grid   = [[" " for _ in range(len(sample))] for _ in range(rows)]
        for c, val in enumerate(sample):
            r = rows - 1 - int((val - mn) / rng * (rows - 1))
            grid[max(0, min(rows-1, r))][c] = "█"
        for row in grid:
            console.print("".join(row))
        console.print(f"  €{mn:.0f}{'':>66}€{mx:.0f}")


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

def run_backtest(cfg_name: str = "Agressif") -> dict:
    t0  = time.time()
    cfg = next((c for c in CONFIGS if c["name"] == cfg_name), CONFIGS[1])

    console.print(f"\n[bold]PRISM v33 — 22 Symboles | {cfg_name}[/bold]")
    console.print(
        f"  OKX REST | {MONTHS} mois | {TIMEFRAME} | Capital €{INITIAL_CAPITAL:.0f}\n"
        f"  Pattern C (BB Squeeze) | SL = ATR14 × {ATR_SL_MULT} | TP = SL × {RR_RATIO:.0f}\n"
        f"  SL range [{ATR_SL_MIN_C:.1%}-{ATR_SL_MAX_C:.1%}] | Vol ≥ {VOL_RATIO_C} | Squeeze ≥ {SQUEEZE_BARS_C} barres\n"
        f"  Cooldown: {COOLDOWN_BARS}h | Risk/trade: {cfg['risk_pct']:.0%} | "
        f"Levier: {BASE_LEVERAGE}-{HIGH_LEVERAGE}× | Score min: {cfg['score_min']}\n"
        f"  Symboles: {len(SYMBOLS)} | Margin cap: {MAX_MARGIN_RATIO:.0%} | Priority: score DESC"
    )
    console.print(f"\n  Téléchargement {len(SYMBOLS)} symboles...")

    results_map: dict = {}

    def _fetch(sym):
        cached = _cache_valid(_cache_path(sym))
        sd = precompute(sym)
        return sym, sd, cached

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, sym): sym for sym in SYMBOLS}
        for future in as_completed(futures):
            sym, sd, was_cached = future.result()
            status = "ok" if sd else "FAIL"
            tag    = " [dim](cache)[/dim]" if was_cached else ""
            idx    = SYMBOLS.index(sym) + 1
            console.print(f"    [{idx:2d}/{len(SYMBOLS)}] {sym:<14} {status}{tag}")
            if sd:
                results_map[sym] = sd

    sym_data_list = [results_map[s] for s in SYMBOLS if s in results_map]
    if not sym_data_list:
        console.print("[red]No data.[/red]")
        return {}

    # Timeline basée sur BTC (historique le plus long)
    # Chaque symbole participe quand il a des données (bar is None → ignoré dans engine)
    # Évite le piège intersection : nouveaux symboles avec historique court ne bloquent plus tout
    btc_sd_main = results_map.get("BTC-USDT")
    if btc_sd_main is not None:
        all_ts = [ts for ts in sorted(btc_sd_main["ts_index"]) if ts >= BACKTEST_START]
    else:
        ts_union: set = set()
        for sd in sym_data_list:
            ts_union.update(sd["ts_index"])
        all_ts = [ts for ts in sorted(ts_union) if ts >= BACKTEST_START]
    timestamps = all_ts[WARMUP:]

    first_ts = timestamps[0].strftime("%Y-%m-%d") if timestamps else "?"
    last_ts  = timestamps[-1].strftime("%Y-%m-%d") if timestamps else "?"
    console.print(
        f"\n  Moteur: {len(timestamps):,} barres | "
        f"{len(sym_data_list)} symboles | {first_ts} → {last_ts}"
    )

    trades, eq_curve, day_wins = run_engine(
        sym_data_list = sym_data_list,
        timestamps    = timestamps,
        risk_pct      = cfg["risk_pct"],
        max_pos       = cfg["max_pos"],
        score_min     = cfg["score_min"],
    )

    metrics = _metrics(trades, eq_curve)
    print_report(trades, eq_curve, cfg_name, t0)
    return {"config": cfg_name, "metrics": metrics, "trades": trades}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    console.print(
        "[bold yellow]PRISM v33 — Exclusion STX | 22 Symboles | Cooldown 5h[/bold yellow]"
    )
    console.print(
        "[bold cyan]  Base v22 (ADX/score leverage) + 13 nouveaux symboles[/bold cyan]\n"
    )

    results = []
    for cfg in CONFIGS:
        r = run_backtest(cfg_name=cfg["name"])
        if r and r.get("metrics"):
            results.append((cfg["name"], r["metrics"]))

    if results:
        console.print("\n")
        tbl = Table(
            title="Comparaison configs — 6 mois | 22 Symboles | Cooldown 5h",
            box=box.ROUNDED
        )
        tbl.add_column("Config",       style="cyan")
        tbl.add_column("Trades",       justify="right")
        tbl.add_column("Trades/mois",  justify="right")
        tbl.add_column("Return",       justify="right")
        tbl.add_column("MaxDD",        justify="right")
        tbl.add_column("WinRate",      justify="right")
        tbl.add_column("P.Factor",     justify="right")
        tbl.add_column("Sharpe",       justify="right")
        for name, m in results:
            rc = "green" if m["ret"] >= 0 else "red"
            dd_col = "red" if m["max_dd"] > 0.30 else "yellow" if m["max_dd"] > 0.15 else "green"
            n_months = len(m.get("monthly", {})) or 1
            tpm = m["n_trades"] / n_months
            tbl.add_row(
                name,
                str(m["n_trades"]),
                f"{tpm:.1f}",
                f"[{rc}]{m['ret']:+.1%}[/]",
                f"[{dd_col}]{m['max_dd']:.1%}[/]",
                f"{m['win_rate']:.0%}",
                f"{m.get('profit_factor',0):.2f}",
                f"{m.get('sharpe',0):.2f}",
            )
        console.print(tbl)

        best = max(results, key=lambda x: x[1]["ret"])
        bm   = best[1]
        n_months = len(bm.get("monthly", {})) or 1
        console.print(
            f"\n  Meilleure config: [bold]{best[0]}[/bold]  "
            f"→  €{INITIAL_CAPITAL:.0f} + €{bm['net_pnl']:+.0f} "
            f"= [bold green]€{bm['final_equity']:.0f}[/bold green]  "
            f"(MaxDD: [red]{bm['max_dd']:.1%}[/red], "
            f"{bm['n_trades']/n_months:.1f} trades/mois)"
        )

        if bm["ret"] >= 1.0:
            console.print("\n[bold green]  ✓ Objectif ×2 ATTEINT[/bold green]")
        elif bm["ret"] >= 0.5:
            console.print(f"\n[yellow]  ~ +{bm['ret']:.0%} — objectif proche mais pas ×2[/yellow]")
        elif bm["ret"] > 0:
            console.print(f"\n[yellow]  ~ Profitable (+{bm['ret']:.0%}) mais sous l'objectif[/yellow]")
        else:
            console.print(f"\n[red]  ✗ Perte sur cette période — marché défavorable[/red]")
