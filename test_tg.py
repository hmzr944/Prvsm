import json, requests, sys

cfg = json.load(open('/data/telegram_config.json'))
token   = cfg["token"]
chat_id = cfg["chat_id"]

sys.stdout.write("Token : " + token[:20] + "...\n")
sys.stdout.write("Chat ID : " + str(chat_id) + "\n")
sys.stdout.flush()

url = "https://api.telegram.org/bot" + token + "/sendMessage"
try:
    r = requests.post(url, json={"chat_id": chat_id, "text": "Test PRISM v33 OK"}, timeout=10)
    sys.stdout.write("HTTP " + str(r.status_code) + "\n")
    sys.stdout.write(r.text[:300] + "\n")
except Exception as e:
    sys.stdout.write("ERREUR : " + str(e) + "\n")
sys.stdout.flush()
