import sys, math
sys.path.insert(0, "/app")
import pandas as pd
import requests
from live_monitor_v33 import OKX_URL, prepare, check_pattern_c, check_pattern_d

def fetch_ohlcv(sym, timeframe="1H", n=350):
    rows, after = [], None
    for _ in range(4):
        params = {"instId": sym, "bar": timeframe, "limit": 100}
        if after:
            params["after"] = after
        r = requests.get(OKX_URL, params=params, timeout=15).json()
        if r.get("code") != "0" or not r.get("data"):
            break
        batch = r["data"]
        rows.extend(batch)
        if len(rows) >= n:
            break
        after = batch[-1][0]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts","o","h","l","c","vol","volccy","volccy2","confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df = df.set_index("ts").sort_index()
    for col in ["o","h","l","c","vol"]:
        df[col] = df[col].astype(float)
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","vol":"volume"}, inplace=True)
    return df

print("=== DIAGNOSTIC PRISM v33 (post-fix) ===")
btc_df = fetch_ohlcv("BTC-USDT")
btc_sd = prepare("BTC-USDT", btc_df) if btc_df is not None else None
btc_bull = None
if btc_sd:
    bar = len(btc_sd["ts_index"]) - 1
    b20 = float(btc_sd["ema20_4h"][bar])
    b50 = float(btc_sd["ema50_4h"][bar])
    btc_bull = b20 > b50
    print(f"BTC 4H EMA20={b20:.0f} EMA50={b50:.0f} => {'BULL' if btc_bull else 'BEAR'} => seuls {'BUY' if btc_bull else 'SELL'} autorise")

SYMS = [
    "ETH-USDT","SOL-USDT","LINK-USDT","AVAX-USDT","NEAR-USDT",
    "INJ-USDT","ARB-USDT","SUI-USDT","OP-USDT","ATOM-USDT",
    "ADA-USDT","XRP-USDT","DOT-USDT","AAVE-USDT","UNI-USDT",
    "LTC-USDT","DOGE-USDT","TRX-USDT","TIA-USDT","SEI-USDT",
    "HBAR-USDT","ICP-USDT","JUP-USDT",
]

hdr = f"{'Sym':<12} {'ADX':>5} {'RSI':>5} {'VR':>5} {'EMA':>9} {'4H':>5} {'ScB':>5} {'ScS':>5} {'PatC':>6} {'PatD':>6}  Blocage"
print(hdr)
print("-" * 98)

signals = []
for sym in SYMS:
    df = fetch_ohlcv(sym)
    if df is None:
        print(f"{sym:<12} NO DATA")
        continue
    sd = prepare(sym, df)
    bar = len(sd["ts_index"]) - 1
    adx = sd["adx"][bar]
    rsi = sd["rsi14"][bar]
    vr  = sd["vol_ratio"][bar]
    e9, e21, e50 = sd["ema9"][bar], sd["ema21"][bar], sd["ema50"][bar]
    ema_align = "9>21>50" if e9>e21>e50 else ("9<21<50" if e9<e21<e50 else "mixte")
    a4h = sd["ema20_4h"][bar] > sd["ema50_4h"][bar]
    cl  = sd["close"][bar]
    vw  = sd["vwap"][bar]
    sc_b = int(sd["buy_sc"][bar])
    sc_s = int(sd["sell_sc"][bar])
    pc  = check_pattern_c(sd, bar, adx)
    pd_ = check_pattern_d(sd, bar, adx)

    blocages = []
    # BTC macro
    if btc_bull is not None:
        if btc_bull and ema_align == "9<21<50":
            blocages.append("BTC_bull+EMA_bear")
        elif not btc_bull and ema_align == "9>21>50":
            blocages.append("BTC_bear+EMA_bull")
    # RSI
    if not (42 <= rsi <= 72) and not (28 <= rsi <= 58):
        blocages.append(f"RSI={rsi:.0f}")
    # VWAP
    if cl < vw and ema_align == "9>21>50":
        blocages.append("cl<VWAP(bull)")
    if cl > vw and ema_align == "9<21<50":
        blocages.append("cl>VWAP(bear)")
    # ADX
    if adx < 22:
        blocages.append(f"ADX={adx:.1f}<22")
    # 4H vs EMA
    if ema_align == "9<21<50" and a4h:
        blocages.append("4H_bull+EMA_bear")
    if ema_align == "9>21>50" and not a4h:
        blocages.append("4H_bear+EMA_bull")
    # Score
    if pd_ and sc_s < 60 and sc_b < 60:
        blocages.append(f"score_trop_bas(B={sc_b},S={sc_s})")

    tag = ", ".join(blocages) if blocages else ("SIGNAL!" if pd_ else "aucun")
    if pd_:
        signals.append(f"  => {sym} {pd_} score={'B'+str(sc_b) if pd_=='BUY' else 'S'+str(sc_s)} ADX={adx:.1f}")
    print(f"{sym:<12} {adx:>5.1f} {rsi:>5.1f} {vr:>5.2f} {ema_align:>9} {'bull' if a4h else 'bear':>5} {sc_b:>5} {sc_s:>5} {str(pc):>6} {str(pd_):>6}  {tag}")

print()
if signals:
    print(f"SIGNAUX DETECTES ({len(signals)}):")
    for s in signals:
        print(s)
else:
    print("Aucun signal detecte avec les nouveaux seuils.")
