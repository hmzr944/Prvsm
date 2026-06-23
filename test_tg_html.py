import json, requests, sys

cfg = json.load(open('/data/telegram_config.json'))
token   = cfg["token"]
chat_id = cfg["chat_id"]

url = "https://api.telegram.org/bot" + token + "/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": "🟢 <b>PRISM v33 démarré</b>\nCapital : €2500,00 (+0.0%)\n<i>Surveillance toutes les heures</i>",
    "parse_mode": "HTML",
    "link_preview_options": {"is_disabled": True}
}
r = requests.post(url, json=payload, timeout=10)
sys.stdout.write("HTTP " + str(r.status_code) + "\n")
sys.stdout.write(r.text[:300] + "\n")
sys.stdout.flush()
