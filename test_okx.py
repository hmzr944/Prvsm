import requests, json
symbols = ["BTC-USDT","AVAX-USDT","ADA-USDT","LINK-USDT","XRP-USDT","DOT-USDT","ATOM-USDT","LTC-USDT","DOGE-USDT","NEAR-USDT","TRX-USDT","INJ-USDT","OP-USDT","ARB-USDT","SUI-USDT","UNI-USDT","AAVE-USDT","TIA-USDT","SEI-USDT","HBAR-USDT","ICP-USDT","JUP-USDT"]
ok, fail = [], []
for s in symbols:
    try:
        r = requests.get("https://www.okx.com/api/v5/market/history-candles", params={"instId": s, "bar": "1H", "limit": 5}, timeout=10)
        d = r.json()
        if d.get("code") == "0" and d.get("data"):
            ok.append(s)
        else:
            fail.append(s + " → " + d.get("msg","?"))
    except Exception as e:
        fail.append(s + " → " + str(e)[:60])
print("OK (" + str(len(ok)) + "):", ok)
print("FAIL (" + str(len(fail)) + "):")
for f in fail: print("  ", f)
