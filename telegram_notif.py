#!/usr/bin/env python3
"""
PRISM v33 — Alertes Telegram
=============================
Setup (une seule fois) :
  1. Parle à @BotFather sur Telegram → /newbot → copie le token
  2. Envoie un message à ton bot
  3. Ouvre https://api.telegram.org/bot<TOKEN>/getUpdates → copie chat.id
  4. Crée telegram_config.json dans le même dossier :
     { "token": "123456:ABC...", "chat_id": "987654321" }

Test : python3 telegram_notif.py
"""

import json
import logging
import os
import requests
from pathlib import Path

CONFIG_FILE = Path(os.environ.get("PRISM_DATA_DIR", str(Path(__file__).parent))) / "telegram_config.json"
INITIAL_CAPITAL = 2500.0
log = logging.getLogger("prism_live")


def _cfg():
    if not CONFIG_FILE.exists():
        return None
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _send(text: str) -> bool:
    cfg = _cfg()
    if not cfg:
        return False
    token   = cfg.get("token", "").strip()
    chat_id = cfg.get("chat_id", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "link_preview_options": {"is_disabled": True}},
            timeout=10,
        )
        ok = r.json().get("ok", False)
        if not ok:
            log.warning(f"Telegram send failed: {r.text[:200]}")
        return ok
    except Exception as e:
        log.warning(f"Telegram error: {e}")
        return False


# ── Messages ─────────────────────────────────────────────────────────────────

def notify_signal(sym: str, side: str, score: int, adx: float,
                  leverage: int, margin: float):
    """Signal détecté — entrée en attente pour la prochaine bougie."""
    direction = "hausse 📈" if side == "long" else "baisse 📉"
    pair = sym.replace("-USDT", "")
    text = (
        f"🎯 <b>Nouveau signal — PRISM v33</b>\n"
        f"<b>{pair}</b> — Pari à la {direction}\n"
        f"Score : {score}/100 | ADX : {adx:.1f} | Levier ×{leverage}\n"
        f"Mise prévue : €{margin:.0f}\n"
        f"<i>Ouverture à la prochaine bougie 1H</i>"
    )
    _send(text)


def notify_open(sym: str, side: str, entry_price: float,
                sl: float, tp: float, margin: float, leverage: int):
    """Position ouverte — entrée exécutée."""
    direction = "hausse 📈" if side == "long" else "baisse 📉"
    pair   = sym.replace("-USDT", "")
    sl_pct = abs(sl - entry_price) / entry_price * 100
    tp_pct = abs(tp - entry_price) / entry_price * 100
    text = (
        f"✅ <b>Position ouverte</b>\n"
        f"<b>{pair}</b> — Pari à la {direction}\n"
        f"Entrée : <code>{entry_price:.5g}</code>\n"
        f"Coupe-circuit : <code>{sl:.5g}</code> (-{sl_pct:.1f}%)\n"
        f"Objectif : <code>{tp:.5g}</code> (+{tp_pct:.1f}%)\n"
        f"Mise : €{margin:.0f} × {leverage}"
    )
    _send(text)


def notify_close(sym: str, side: str, reason: str, pnl: float, equity: float):
    """Position fermée — résultat."""
    labels = {
        "take_profit": ("Objectif atteint", "✅"),
        "stop_loss":   ("Coupe-circuit",    "❌"),
        "time_stop":   ("Délai expiré",     "⏱"),
    }
    label, icon = labels.get(reason, (reason, "📊"))
    pair    = sym.replace("-USDT", "")
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    pnl_str = f"+€{pnl:.2f}" if pnl >= 0 else f"-€{abs(pnl):.2f}"
    text = (
        f"📊 <b>Position fermée — {label} {icon}</b>\n"
        f"<b>{pair}</b> — {'Pari hausse' if side == 'long' else 'Pari baisse'}\n"
        f"Résultat : <b>{pnl_str}</b>\n"
        f"Capital total : €{equity:,.2f} "
        f"(<b>{ret_pct:+.1f}%</b> depuis le départ)"
    )
    _send(text)


def notify_daily_summary(equity: float, day_pnl: float,
                         total_trades: int, total_wins: int, n_open: int):
    """Résumé quotidien — envoyé au début de chaque nouvelle journée."""
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr      = total_wins / total_trades * 100 if total_trades > 0 else 0
    sign    = "+" if day_pnl >= 0 else ""
    emoji   = "📈" if day_pnl >= 0 else "📉"
    text = (
        f"{emoji} <b>Résumé de la journée — PRISM v33</b>\n"
        f"Capital : €{equity:,.2f} ({ret_pct:+.1f}% depuis le départ)\n"
        f"Hier : <b>{sign}€{day_pnl:.2f}</b>\n"
        f"Historique : {total_wins} gagnants / {total_trades} trades "
        f"({wr:.0f}% réussite)\n"
        f"En cours : {n_open} position{'s' if n_open != 1 else ''}"
    )
    _send(text)


def notify_bot_start(equity: float):
    """Bot démarré."""
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    text = (
        f"🟢 <b>PRISM v33 démarré</b>\n"
        f"Capital : €{equity:,.2f} ({ret_pct:+.1f}%)\n"
        f"<i>Surveillance toutes les heures</i>"
    )
    _send(text)


def notify_error(message: str):
    """Erreur critique."""
    text = f"🔴 <b>PRISM v33 — Erreur</b>\n<code>{message[:300]}</code>"
    _send(text)


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CONFIG_FILE.exists():
        print(f"Crée d'abord : {CONFIG_FILE}")
        print('  { "token": "TON_TOKEN", "chat_id": "TON_CHAT_ID" }')
    else:
        print("Envoi message test...")
        ok = _send("✅ <b>PRISM v33 — Test OK</b>\nLes alertes Telegram fonctionnent.")
        print("Envoyé !" if ok else "Échec — vérifie token et chat_id")
