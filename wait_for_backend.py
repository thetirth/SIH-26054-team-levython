"""wait_for_backend.py -- poll the FastAPI /health endpoint until ready.

Used by start.bat. Exits 0 on success, 1 on timeout (2 min).
"""
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/health"
TIMEOUT_S = 120

t0 = time.time()
while time.time() - t0 < TIMEOUT_S:
    try:
        urllib.request.urlopen(URL, timeout=5).read()
        print("[OK] Backend is up at http://127.0.0.1:8000")
        sys.exit(0)
    except Exception:
        time.sleep(3)

print("[ERROR] Backend did not answer /health within 120 s")
sys.exit(1)
