import sys, os
sys.path.insert(0, '/data')
import telegram_notif

# Show which file is loaded and config path
print("Module file:", telegram_notif.__file__)
print("Config path:", telegram_notif.CONFIG_FILE)
print("Config exists:", telegram_notif.CONFIG_FILE.exists())
print("PRISM_DATA_DIR:", os.environ.get("PRISM_DATA_DIR", "NOT SET"))

# Test _send directly
r = telegram_notif._send("Test HTML <b>OK</b>")
print("_send result:", r)
