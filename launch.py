import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = int(os.getenv("APP_PORT", "8767"))
URL = f"http://{HOST}:{PORT}"


def is_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/openapi.json", timeout=1) as response:
            paths = json.load(response)["paths"]
            return "/api/knowledge/ask" in paths
    except (OSError, ValueError, KeyError):
        return False


def open_when_ready():
    for _ in range(600):
        if is_ready():
            webbrowser.open(URL)
            return
        time.sleep(0.5)


if __name__ == "__main__":
    open_browser = "--no-browser" not in sys.argv
    if is_ready():
        if open_browser:
            webbrowser.open(URL)
        print(f"Already running: {URL}")
    else:
        print(f"Starting project at {URL}")
        if open_browser:
            threading.Thread(target=open_when_ready, daemon=True).start()
        uvicorn.run("app.main:app", host=HOST, port=PORT, access_log=False)
