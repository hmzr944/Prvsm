#!/usr/bin/env python3
"""
PRISM v33 — Live Paper Trading Monitor
=======================================
Moteur identique v33. Mode paper : aucun ordre réel.
Logs persistants fichier + dashboard rich console.

Usage :
  python3 live_monitor_v33.py          # Boucle infinie (toutes les heures)
  python3 live_monitor_v33.py --once   # Une seule détection puis quitte
  python3 live_monitor_v33.py --status # Affiche l'état courant et quitte
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    import telegram_notif as _tg
except ImportError:
    _tg = None

console = Console()

# ── Répertoire de travail ────────────────────────────────────────────────────
BASE_DIR          = Path(os.environ.get("PRISM_DATA_DIR", str(Path(__file__).parent)))
LOG_DIR           = BASE_DIR / "live_logs"
STATE_FILE        = BASE_DIR / "live_state_v33.json"
SCAN_HISTORY_FILE = BASE_DIR / "scan_history_v33.json"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging fichier ──────────────────────────────────────────────────────────
def _setup_file_logger():
    today = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"live_session_{today}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log = logging.getLogger("prism_live")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(fh)
    return log

log = _setup_file_logger()

TRADE_LOG_PATH = LOG_DIR / f"live_trades_{datetime.now().strftime('%Y%m')}.csv"
_TRADE_HEADER  = ["exit_ts","sym","side","entry_ts","entry_px","exit_px",
                  "reason","margin","leverage","score","pnl","equity_after"]

def _write_trade(row: dict):
    exists = TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRADE_HEADER)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _TRADE_HEADER})

# ── Constantes (identiques v33) ──────────────────────────────────────────────
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT",
    "AVAX-USDT", "ADA-USDT", "LINK-USDT", "XRP-USDT", "DOT-USDT", "ATOM-USDT",
    "LTC-USDT", "DOGE-USDT", "NEAR-USDT", "TRX-USDT",
    "INJ-USDT", "OP-USDT",
    "ARB-USDT", "SUI-USDT", "UNI-USDT", "AAVE-USDT",
    "TIA-USDT", "SEI-USDT", "HBAR-USDT", "ICP-USDT",
    "JUP-USDT",
]

INITIAL_CAPITAL  = 2500.0
TIMEFRAME        = "1H"
BARS_NEEDED      = 350          # warmup suffisant pour tous les indicateurs
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

# ── Pattern D — EMA Trend Follow ─────────────────────────────────────────────
SCORE_MIN_D      = 76   # relevé 68→76 (grid search score×ADX : +123% vs -24%)
ADX_MIN_D        = 25   # relevé 22→25 (filtre tendances faibles)
VOL_RATIO_D      = 1.2
RISK_PCT_D       = 0.06   # 6% de l'equity (vs 10% Pattern C)
BASE_LEVERAGE_D  = 8
HIGH_LEVERAGE_D  = 10

# Univers autorisé Pattern D — actifs trending uniquement (WR>27%, PnL>0 sur 4 mois)
# Les actifs mean-reverting (ATOM, ARB, LTC, SEI, LINK…) génèrent des faux signaux.
PATTERN_D_WHITELIST = {
    "AVAX-USDT", "TIA-USDT", "OP-USDT", "JUP-USDT",
    "ICP-USDT",  "SUI-USDT", "DOT-USDT", "TRX-USDT",
}

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

# ── State (persistance JSON) ─────────────────────────────────────────────────
DEFAULT_STATE = {
    "equity":           INITIAL_CAPITAL,
    "peak_equity":      INITIAL_CAPITAL,
    "day_start_equity": INITIAL_CAPITAL,
    "current_day":      "",
    "open_positions":   {},
    "pending_entries":  {},
    "cooldown_tracker": {},
    "total_trades":     0,
    "total_wins":       0,
    "total_pnl":        0.0,
    "started_at":       datetime.now().isoformat(),
    "last_run_ts":      "",
}

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            for k, v in DEFAULT_STATE.items():
                s.setdefault(k, v)
            return s
        except Exception as e:
            log.warning(f"State corrompu ({e}) — reset")
    return dict(DEFAULT_STATE)

def save_state(s: dict):
    s["last_run_ts"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)

def save_scan_record(bar_ts, n_loaded: int, trades_closed: list,
                     signals_new: list, state: dict):
    record = {
        "ts":             datetime.now().isoformat(),
        "bar_ts":         str(bar_ts)[:16],
        "symbols_loaded": n_loaded,
        "signals": [
            {
                "sym":      s["sym"],
                "side":     "long" if s["action"] == "BUY" else "short",
                "score":    s["score"],
                "adx":      round(s["adx"], 1),
                "leverage": s["leverage"],
                "pattern":  s.get("pattern", "C"),
            }
            for s in signals_new
        ],
        "trades_closed": [
            {
                "sym":    t["sym"],
                "side":   t["side"],
                "reason": t["reason"],
                "pnl":    t["pnl"],
            }
            for t in trades_closed
        ],
        "equity":    round(state["equity"], 2),
        "n_open":    len(state["open_positions"]),
        "n_pending": len(state["pending_entries"]),
    }
    history = []
    if SCAN_HISTORY_FILE.exists():
        try:
            with open(SCAN_HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, record)
    history = history[:200]
    with open(SCAN_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)

# ── Download (live, pas de cache long) ──────────────────────────────────────
def fetch_live(inst_id: str, limit: int = 350) -> pd.DataFrame | None:
    """Récupère les `limit` dernières bougies 1H complètes."""
    all_rows, after = [], None
    needed = limit
    for _ in range(20):
        params = {"instId": inst_id, "bar": TIMEFRAME, "limit": min(100, needed)}
        if after:
            params["after"] = after
        try:
            r    = requests.get(OKX_URL, params=params, timeout=15)
            data = r.json()
        except Exception as e:
            log.error(f"fetch_live {inst_id}: {e}")
            return None
        if data.get("code") != "0" or not data.get("data"):
            break
        batch = data["data"]
        all_rows.extend(batch)
        needed -= len(batch)
        if needed <= 0:
            break
        after = batch[-1][0]
        time.sleep(0.12)

    if len(all_rows) < 50:
        return None

    df = pd.DataFrame(all_rows,
                      columns=["timestamp","open","high","low","close","volume","a","b","c"])
    df = df[["timestamp","open","high","low","close","volume"]]
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="ms", utc=True
    ).dt.tz_convert(None)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["close"])
    return df

# ── Indicateurs (identiques v33) ────────────────────────────────────────────
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
    df["atr14"]    = atr14
    df["adx"]      = dx.ewm(com=13, adjust=False).mean()
    df["di_plus"]  = dip
    df["di_minus"] = dim
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
            "di_plus","di_minus","ema20_4h","ema50_4h","ema50d","ema200d"]
    sd = {"name": sym, "ts_index": ts_idx, "ts_to_pos": ts_to_pos}
    for col in cols:
        sd[col] = df[col].values
    n = len(ts_idx)
    sd["buy_sc"], sd["sell_sc"] = _compute_scores(sd, n)
    return sd

# ── Pattern C (identique v33) ────────────────────────────────────────────────
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

# ── Pattern D — EMA Trend Follow ────────────────────────────────────────────
def check_pattern_d(sd: dict, bar: int, adx_val: float):
    """
    Pattern D : EMA9 > EMA21 > EMA50 (haussier) ou inverse (baissier)
                + RSI zone tendance + prix vs VWAP + alignement 4H.
    Pas de filtre vol_ratio : trend follow ne nécessite pas de pic de volume.
    RSI zones élargies pour capter les tendances matures sans sur-filtrer.
    Fix backtest : ADX doit monter depuis 2 barres (tendance s'accélère).
    """
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
        # ADX doit monter depuis 2 barres consécutives
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

# ── Étape moteur (une barre) ─────────────────────────────────────────────────
def engine_step(state: dict, sym_data: dict, current_bar_ts: pd.Timestamp):
    """
    Traite la barre `current_bar_ts` :
      1. Exécute les pending_entries (ouverture)
      2. Vérifie les exits (SL / TP / time_stop)
      3. Calcule skip_entries + drawdown
      4. Détection signaux BB Squeeze
    Modifie `state` en place. Retourne (trades_closed, signals_new).
    """
    equity           = state["equity"]
    peak_equity      = state["peak_equity"]
    open_positions   = state["open_positions"]
    pending_entries  = state["pending_entries"]
    cooldown_tracker = state["cooldown_tracker"]
    ts_str           = str(current_bar_ts)

    # Day tracking
    today = str(current_bar_ts)[:10]
    if today != state["current_day"]:
        if state["current_day"] and _tg:
            _tg.notify_daily_summary(
                equity,
                equity - state["day_start_equity"],
                state["total_trades"],
                state["total_wins"],
                len(open_positions),
            )
        state["current_day"]      = today
        state["day_start_equity"] = equity
    day_start_equity = state["day_start_equity"]

    trades_closed = []
    signals_new   = []

    # ── 1. Exécuter les pending_entries ──────────────────────────────────────
    for pk in list(pending_entries.keys()):
        if len(open_positions) >= MAX_POS:
            break
        p  = pending_entries.pop(pk)
        sd = sym_data.get(p["sym"])
        if sd is None:
            continue
        bar = sd["ts_to_pos"].get(current_bar_ts)
        if bar is None:
            # Remettre en attente pour la prochaine barre
            pending_entries[pk] = p
            continue

        # Revalidation : si les paramètres ont changé depuis la détection, annuler
        if p.get("pattern") == "D":
            if p["sym"] not in PATTERN_D_WHITELIST:
                log.warning(f"ANNULÉ {p['sym']} (hors whitelist Pattern D — paramètres mis à jour)")
                continue
            if p.get("score", 0) < SCORE_MIN_D:
                log.warning(f"ANNULÉ {p['sym']} (score={p.get('score')} < SCORE_MIN_D={SCORE_MIN_D})")
                continue

        open_px     = float(sd["open"][bar])
        side        = p["side"]
        entry_price = (open_px * (1 + SLIPPAGE) if side == "long"
                       else open_px * (1 - SLIPPAGE))
        atr_now = float(sd["atr14"][bar])
        if math.isnan(atr_now) or atr_now <= 0:
            atr_now = entry_price * 0.015
        sl_pct = max(ATR_SL_MIN_C, min(ATR_SL_MAX_C, ATR_SL_MULT * atr_now / (entry_price + 1e-10)))
        tp_pct = sl_pct * RR_RATIO
        sl = entry_price * (1 - sl_pct) if side == "long" else entry_price * (1 + sl_pct)
        tp = entry_price * (1 + tp_pct) if side == "long" else entry_price * (1 - tp_pct)

        pos_key = p["sym"] + p.get("pattern", "C") + ts_str
        open_positions[pos_key] = {
            "sym":         p["sym"],
            "side":        side,
            "pattern":     p.get("pattern", "C"),
            "entry_ts":    ts_str,
            "entry_price": entry_price,
            "sl":          sl,
            "tp":          tp,
            "margin":      p["margin"],
            "leverage":    p["leverage"],
            "score":       p["score"],
        }
        log.info(f"OPEN  {p['sym']:12s} {side:5s} | px={entry_price:.5g} "
                 f"sl={sl:.5g} tp={tp:.5g} | marge={p['margin']:.0f}€ ×{p['leverage']}")
        if _tg:
            _tg.notify_open(p["sym"], side, entry_price, sl, tp,
                            p["margin"], p["leverage"])

    # ── 2. Vérifier les exits ─────────────────────────────────────────────────
    to_remove = []
    for pos_key, pos in list(open_positions.items()):
        sd = sym_data.get(pos["sym"])
        if sd is None:
            continue
        bar = sd["ts_to_pos"].get(current_bar_ts)
        if bar is None:
            continue

        hi = float(sd["high"][bar])
        lo = float(sd["low"][bar])
        cl = float(sd["close"][bar])
        entry  = pos["entry_price"]
        sl, tp = pos["sl"], pos["tp"]
        side   = pos["side"]

        exit_price = exit_reason = None
        if side == "long":
            if hi >= tp:  exit_price, exit_reason = tp, "take_profit"
            elif lo <= sl: exit_price, exit_reason = sl, "stop_loss"
        else:
            if lo <= tp:  exit_price, exit_reason = tp, "take_profit"
            elif hi >= sl: exit_price, exit_reason = sl, "stop_loss"

        entry_ts = pd.Timestamp(pos["entry_ts"])
        elapsed_h = (current_bar_ts - entry_ts).total_seconds() / 3600
        if exit_price is None and elapsed_h >= TIME_STOP_H:
            exit_price, exit_reason = cl, "time_stop"

        # Clôture forcée si la position ne respecte plus les règles actuelles
        # (ex: paramètres changés depuis l'entrée — trade hors whitelist ou score périmé)
        if exit_price is None and pos.get("pattern") == "D":
            invalid = False
            if pos["sym"] not in PATTERN_D_WHITELIST:
                invalid = True
                log.warning(f"FORCE-CLOSE {pos['sym']} hors whitelist Pattern D (params mis à jour)")
            elif pos.get("score", 99) < SCORE_MIN_D:
                invalid = True
                log.warning(f"FORCE-CLOSE {pos['sym']} score={pos.get('score')} < {SCORE_MIN_D} (params mis à jour)")
            if invalid:
                exit_price, exit_reason = cl, "param_update"

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

            state["total_trades"] += 1
            state["total_pnl"]    += net_pnl
            if net_pnl > 0:
                state["total_wins"] += 1

            trade_row = {
                "exit_ts":    str(current_bar_ts),
                "sym":        pos["sym"],
                "side":       side,
                "entry_ts":   pos["entry_ts"],
                "entry_px":   entry,
                "exit_px":    exit_price,
                "reason":     exit_reason,
                "margin":     pos["margin"],
                "leverage":   pos["leverage"],
                "score":      pos.get("score", 0),
                "pnl":        round(net_pnl, 2),
                "equity_after": round(equity, 2),
            }
            trades_closed.append(trade_row)
            _write_trade(trade_row)
            to_remove.append(pos_key)
            emoji = "✓" if net_pnl > 0 else "✗"
            log.info(f"CLOSE {pos['sym']:12s} {side:5s} | {exit_reason:11s} | "
                     f"pnl={net_pnl:+.2f}€ {emoji} | equity={equity:.2f}€")
            if _tg:
                _tg.notify_close(pos["sym"], side, exit_reason, net_pnl, equity)

    for k in to_remove:
        del open_positions[k]

    # ── 3. Gardes d'entrée ────────────────────────────────────────────────────
    state["equity"]      = equity
    state["peak_equity"] = peak_equity

    day_pnl_pct  = (equity - day_start_equity) / (day_start_equity + 1e-10)
    skip_entries = day_pnl_pct <= -DAILY_LOSS_CAP
    drawdown     = (peak_equity - equity) / (peak_equity + 1e-10)

    # Pas d'entrée en session asiatique 00h-07h UTC (WR 19% sur backtest 4 mois)
    if current_bar_ts.hour < 7:
        return trades_closed, signals_new
    if skip_entries:
        log.info(f"  skip_entries=True (day_pnl={day_pnl_pct:.1%})")
        return trades_closed, signals_new
    if len(open_positions) + len(pending_entries) >= MAX_POS:
        return trades_closed, signals_new
    if drawdown > 0.40:
        log.info(f"  skip drawdown={drawdown:.1%}")
        return trades_closed, signals_new

    # ── 4. BTC 4H macro filter ────────────────────────────────────────────────
    btc_sd = sym_data.get("BTC-USDT")
    btc_4h_bull = None
    if btc_sd is not None:
        btc_bar = btc_sd["ts_to_pos"].get(current_bar_ts)
        if btc_bar is not None:
            b20 = float(btc_sd["ema20_4h"][btc_bar])
            b50 = float(btc_sd["ema50_4h"][btc_bar])
            if not (math.isnan(b20) or math.isnan(b50)):
                btc_4h_bull = bool(b20 > b50)

    # ── 5. Détection BB Squeeze — deux passes ─────────────────────────────────
    dd_scale         = max(0.5, 1.0 - drawdown * 2.5)
    margin_per_trade = equity * RISK_PCT * dd_scale
    total_margin     = (sum(p["margin"] for p in pending_entries.values()) +
                        sum(p["margin"] for p in open_positions.values()))
    max_margin_allowed = equity * MAX_MARGIN_RATIO

    candidates = []
    for sym, sd in sym_data.items():
        if sym == "BTC-USDT":
            continue
        if sym + "C" in pending_entries:
            continue
        bar = sd["ts_to_pos"].get(current_bar_ts)
        if bar is None or bar < 250:
            continue
        adx_val = float(sd["adx"][bar])
        if math.isnan(adx_val):
            continue
        ck = sym + "C"
        bar_cooldown = cooldown_tracker.get(ck, -9999)
        if bar - bar_cooldown < COOLDOWN_BARS:
            continue
        action = check_pattern_c(sd, bar, adx_val)
        if action is None:
            continue
        if btc_4h_bull is not None:
            if action == "BUY"  and not btc_4h_bull:
                continue
            if action == "SELL" and btc_4h_bull:
                continue
        score = int(sd["buy_sc"][bar]) if action == "BUY" else int(sd["sell_sc"][bar])
        if score < SCORE_MIN:
            continue
        lev = HIGH_LEVERAGE if (adx_val > 28 and score >= 72) else BASE_LEVERAGE
        candidates.append({"sym": sym, "ck": ck, "bar": bar,
                            "action": action, "score": score,
                            "adx": adx_val, "leverage": lev})

    candidates.sort(key=lambda c: (c["score"], c["adx"]), reverse=True)

    for c in candidates:
        if len(open_positions) + len(pending_entries) >= MAX_POS:
            break
        if total_margin + margin_per_trade > max_margin_allowed:
            break
        side = "long" if c["action"] == "BUY" else "short"
        pending_entries[c["sym"] + "C"] = {
            "sym":      c["sym"],
            "side":     side,
            "pattern":  "C",
            "margin":   margin_per_trade,
            "leverage": c["leverage"],
            "score":    c["score"],
        }
        cooldown_tracker[c["ck"]] = c["bar"]
        total_margin += margin_per_trade
        signals_new.append(c)
        log.info(f"SIGNAL-C {c['sym']:12s} {side:5s} | score={c['score']} "
                 f"adx={c['adx']:.1f} lev=×{c['leverage']} | "
                 f"entry prévu à la prochaine bougie")
        if _tg:
            _tg.notify_signal(c["sym"], side, c["score"], c["adx"],
                              c["leverage"], margin_per_trade)

    # ── 6. Pattern D — EMA Trend Follow ──────────────────────────────────────
    margin_d       = equity * RISK_PCT_D * dd_scale
    syms_with_pos  = {pos["sym"] for pos in open_positions.values()}

    candidates_d = []
    for sym, sd in sym_data.items():
        if sym == "BTC-USDT":
            continue
        if sym not in PATTERN_D_WHITELIST:
            continue
        if sym in syms_with_pos:
            continue
        if sym + "C" in pending_entries or sym + "D" in pending_entries:
            continue
        bar = sd["ts_to_pos"].get(current_bar_ts)
        if bar is None or bar < 150:
            continue
        adx_val = float(sd["adx"][bar])
        if math.isnan(adx_val) or adx_val < ADX_MIN_D:
            continue
        ck = sym + "D"
        bar_cooldown = cooldown_tracker.get(ck, -9999)
        if bar - bar_cooldown < COOLDOWN_BARS:
            continue
        action = check_pattern_d(sd, bar, adx_val)
        if action is None:
            continue
        if btc_4h_bull is not None:
            if action == "BUY"  and not btc_4h_bull:
                continue
            if action == "SELL" and btc_4h_bull:
                continue
        score = int(sd["buy_sc"][bar]) if action == "BUY" else int(sd["sell_sc"][bar])
        if score < SCORE_MIN_D:
            continue
        lev = HIGH_LEVERAGE_D if (adx_val > 28 and score >= SCORE_MIN_D) else BASE_LEVERAGE_D
        candidates_d.append({"sym": sym, "ck": ck, "bar": bar,
                              "action": action, "score": score,
                              "adx": adx_val, "leverage": lev, "pattern": "D"})

    candidates_d.sort(key=lambda c: (c["score"], c["adx"]), reverse=True)

    for c in candidates_d:
        if len(open_positions) + len(pending_entries) >= MAX_POS:
            break
        if total_margin + margin_d > max_margin_allowed:
            break
        side = "long" if c["action"] == "BUY" else "short"
        pending_entries[c["sym"] + "D"] = {
            "sym":      c["sym"],
            "side":     side,
            "pattern":  "D",
            "margin":   margin_d,
            "leverage": c["leverage"],
            "score":    c["score"],
        }
        cooldown_tracker[c["ck"]] = c["bar"]
        total_margin += margin_d
        signals_new.append(c)
        log.info(f"SIGNAL-D {c['sym']:12s} {side:5s} | score={c['score']} "
                 f"adx={c['adx']:.1f} lev=×{c['leverage']} marge={margin_d:.0f}€ | "
                 f"entry prévu à la prochaine bougie")
        if _tg:
            _tg.notify_signal(c["sym"], side, c["score"], c["adx"],
                              c["leverage"], margin_d)

    state["open_positions"]   = open_positions
    state["pending_entries"]  = pending_entries
    state["cooldown_tracker"] = cooldown_tracker
    return trades_closed, signals_new

# ── Dashboard ────────────────────────────────────────────────────────────────
def print_dashboard(state: dict, trades_closed: list, signals_new: list,
                    bar_ts: pd.Timestamp):
    equity   = state["equity"]
    peak     = state["peak_equity"]
    ret_pct  = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    dd_pct   = (peak - equity) / peak * 100 if peak > 0 else 0
    n_open   = len(state["open_positions"])
    n_pend   = len(state["pending_entries"])
    total_t  = state["total_trades"]
    wr_pct   = state["total_wins"] / total_t * 100 if total_t > 0 else 0

    ret_color = "green" if ret_pct >= 0 else "red"
    console.print(Panel(
        f"[bold]Barre traitée :[/bold] {bar_ts}\n"
        f"[bold]Équité        :[/bold] [bold {ret_color}]€{equity:,.2f}[/bold {ret_color}]  "
        f"[{ret_color}]{ret_pct:+.1f}%[/{ret_color}]  "
        f"MaxDD [yellow]{dd_pct:.1f}%[/yellow]\n"
        f"[bold]Positions     :[/bold] {n_open} ouvertes · {n_pend} en attente\n"
        f"[bold]Historique    :[/bold] {total_t} trades · WR {wr_pct:.0f}% · PnL total €{state['total_pnl']:+,.2f}",
        title="[bold cyan]PRISM v33 — Live Paper Monitor[/bold cyan]",
        border_style="cyan",
    ))

    if trades_closed:
        t = Table(box=box.SIMPLE_HEAD, title="Trades fermés cette barre")
        t.add_column("Symbole");  t.add_column("Side")
        t.add_column("Raison");   t.add_column("PnL", justify="right")
        t.add_column("Equity après", justify="right")
        for tr in trades_closed:
            col = "green" if tr["pnl"] > 0 else "red"
            t.add_row(tr["sym"], tr["side"], tr["reason"],
                      f"[{col}]€{tr['pnl']:+.2f}[/{col}]",
                      f"€{tr['equity_after']:,.2f}")
        console.print(t)

    if signals_new:
        s = Table(box=box.SIMPLE_HEAD, title="Nouveaux signaux (exécution prochaine barre)")
        s.add_column("Symbole"); s.add_column("Pat."); s.add_column("Direction")
        s.add_column("Score", justify="right"); s.add_column("ADX", justify="right")
        s.add_column("Levier", justify="right")
        for sig in signals_new:
            col = "green" if sig["action"] == "BUY" else "red"
            s.add_row(sig["sym"],
                      sig.get("pattern", "C"),
                      f"[{col}]{sig['action']}[/{col}]",
                      str(sig["score"]), f"{sig['adx']:.1f}", f"×{sig['leverage']}")
        console.print(s)

    if state["open_positions"]:
        p = Table(box=box.SIMPLE_HEAD, title="Positions ouvertes")
        p.add_column("Symbole"); p.add_column("Side"); p.add_column("Entrée")
        p.add_column("SL", justify="right"); p.add_column("TP", justify="right")
        p.add_column("Marge", justify="right"); p.add_column("Levier", justify="right")
        for pos in state["open_positions"].values():
            col = "green" if pos["side"] == "long" else "red"
            p.add_row(pos["sym"],
                      f"[{col}]{pos['side']}[/{col}]",
                      f"{pos['entry_price']:.5g}",
                      f"{pos['sl']:.5g}", f"{pos['tp']:.5g}",
                      f"€{pos['margin']:.0f}", f"×{pos['leverage']}")
        console.print(p)

# ── Boucle principale ────────────────────────────────────────────────────────
def run_once():
    """Télécharge les données, traite la dernière barre complète, sauvegarde."""
    console.print("[dim]Téléchargement données OKX...[/dim]", end="")
    sym_data = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_live, s): s for s in SYMBOLS}
        for fut in as_completed(futs):
            sym = futs[fut]
            df  = fut.result()
            if df is not None and len(df) >= 50:
                sym_data[sym] = prepare(sym, df)

    n_loaded = len(sym_data)
    console.print(f"\r[green]{n_loaded} symboles chargés[/green]       ")
    log.info(f"Données : {n_loaded}/{len(SYMBOLS)} symboles")

    if "BTC-USDT" not in sym_data:
        console.print("[red]BTC-USDT absent — abandon.[/red]")
        log.error("BTC-USDT absent")
        return

    # Dernière barre complète = avant-dernière entrée de BTC (la dernière est en cours)
    btc_ts = sym_data["BTC-USDT"]["ts_index"]
    # La dernière barre retournée par history-candles est la plus récente complète
    current_bar_ts = btc_ts[-1]

    state = load_state()
    trades_closed, signals_new = engine_step(state, sym_data, current_bar_ts)
    save_state(state)
    save_scan_record(current_bar_ts, n_loaded, trades_closed, signals_new, state)
    print_dashboard(state, trades_closed, signals_new, current_bar_ts)

    console.print(f"\n[dim]Logs : {LOG_DIR}[/dim]")
    console.print(f"[dim]State: {STATE_FILE}[/dim]")

def run_backfill():
    """Rejoue toutes les barres manquées depuis le dernier scan enregistré."""
    # Trouver la dernière barre traitée
    last_bar_ts = None
    if SCAN_HISTORY_FILE.exists():
        try:
            with open(SCAN_HISTORY_FILE) as f:
                history = json.load(f)
            if history:
                last_bar_ts = pd.Timestamp(history[0]["bar_ts"])
        except Exception:
            pass

    if last_bar_ts is None:
        console.print("[yellow]Aucun historique — exécution d'un scan normal.[/yellow]")
        run_once()
        return

    console.print(f"[cyan]Dernière barre traitée : {last_bar_ts}[/cyan]")
    console.print("[dim]Téléchargement données OKX...[/dim]", end="")

    sym_data = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_live, s): s for s in SYMBOLS}
        for fut in as_completed(futs):
            sym = futs[fut]
            df  = fut.result()
            if df is not None and len(df) >= 50:
                sym_data[sym] = prepare(sym, df)

    n_loaded = len(sym_data)
    console.print(f"\r[green]{n_loaded} symboles chargés[/green]       ")

    if "BTC-USDT" not in sym_data:
        console.print("[red]BTC-USDT absent — abandon.[/red]")
        return

    # Toutes les barres disponibles après la dernière traitée (sauf la dernière = en cours)
    btc_ts    = sym_data["BTC-USDT"]["ts_index"]
    missed    = [ts for ts in btc_ts if ts > last_bar_ts]
    if not missed:
        console.print("[yellow]Aucune barre manquée.[/yellow]")
        return
    # Exclure la dernière barre (potentiellement en cours de formation)
    missed = missed[:-1]
    if not missed:
        console.print("[yellow]Aucune barre complète manquée.[/yellow]")
        return

    console.print(f"[bold]{len(missed)} barres à rattraper :[/bold] "
                  f"{missed[0]} → {missed[-1]}")

    state = load_state()
    for bar_ts in missed:
        trades_closed, signals_new = engine_step(state, sym_data, bar_ts)
        save_state(state)
        save_scan_record(bar_ts, n_loaded, trades_closed, signals_new, state)

        tc_str  = f"  {len(trades_closed)} trade(s) fermé(s)" if trades_closed else ""
        sig_str = f"  {len(signals_new)} signal(s)" if signals_new else ""
        result  = (tc_str + sig_str).strip() or "aucun signal"
        console.print(f"  [dim]{bar_ts}[/dim]  →  {result}")

    console.print(f"\n[green]Backfill terminé — {len(missed)} barres traitées.[/green]")
    print_dashboard(state, [], [], missed[-1])


def _seconds_to_next_hour(offset_min: int = 3) -> float:
    """Secondes jusqu'à HH:(offset_min)."""
    now  = datetime.now()
    nxt  = now.replace(minute=offset_min, second=0, microsecond=0)
    if now.minute >= offset_min:
        nxt += timedelta(hours=1)
    return (nxt - now).total_seconds()

def run_loop():
    """Boucle infinie : exécute à HH:03 chaque heure."""
    console.print("[bold cyan]PRISM v33 Live Monitor démarré — Ctrl+C pour quitter[/bold cyan]")
    log.info("Live monitor démarré")
    _boot_state = load_state()
    save_state(_boot_state)          # marque last_run_ts dès le démarrage
    if _tg:
        _tg.notify_bot_start(_boot_state["equity"])
    while True:
        wait = _seconds_to_next_hour(offset_min=3)
        nxt  = datetime.now() + timedelta(seconds=wait)
        console.print(f"[dim]Prochaine exécution : {nxt.strftime('%H:%M:%S')} "
                      f"(dans {wait/60:.1f} min)[/dim]")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            console.print("\n[yellow]Arrêt demandé.[/yellow]")
            log.info("Live monitor arrêté manuellement")
            break
        try:
            run_once()
        except Exception as e:
            log.exception(f"Erreur run_once: {e}")
            console.print(f"[red]Erreur : {e}[/red]")
            if _tg:
                _tg.notify_error(str(e))

def show_status():
    state = load_state()
    equity  = state["equity"]
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_t = state["total_trades"]
    wr_pct  = state["total_wins"] / total_t * 100 if total_t > 0 else 0
    console.print(Panel(
        f"Equity      : €{equity:,.2f}  ({ret_pct:+.1f}%)\n"
        f"PnL total   : €{state['total_pnl']:+,.2f}\n"
        f"Trades      : {total_t}  |  WR {wr_pct:.0f}%\n"
        f"Positions   : {len(state['open_positions'])} open · {len(state['pending_entries'])} pending\n"
        f"Démarré le  : {state.get('started_at','?')}\n"
        f"Dernier run : {state.get('last_run_ts','jamais')}",
        title="[bold cyan]PRISM v33 — État courant[/bold cyan]",
    ))

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",     action="store_true", help="Exécute une seule fois")
    parser.add_argument("--status",   action="store_true", help="Affiche l'état et quitte")
    parser.add_argument("--reset",    action="store_true", help="Remet le state à zéro")
    parser.add_argument("--backfill", action="store_true", help="Rattrape les barres manquées")
    args = parser.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        console.print("[yellow]State réinitialisé.[/yellow]")
        sys.exit(0)

    if args.status:
        show_status()
        sys.exit(0)

    if args.backfill:
        run_backfill()
    elif args.once:
        run_once()
    else:
        run_loop()
