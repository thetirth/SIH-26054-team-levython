"""Targeted re-check: RTL latch + destruct + gps (fake clock)."""
import time as _t
import api

t0 = _t.monotonic()
_calls = [0]


def fake_mono():
    _calls[0] += 1
    return t0 + _calls[0] * 0.5


api.time.monotonic = fake_mono
T = api.TWIN


def run(n):
    p = None
    for _ in range(n):
        p = T.step()
    return p


print("== 4b. RTL completes and LATCHES at home ==", flush=True)
api.TWIN.reset("endurance")
T.fuel_l = 3.0
p = run(600)
modes = set()
T2fuel = T.fuel_l
print(f"end: mode {p['telemetry']['flight_mode']} dist {p['telemetry']['dist_home_km']} fuel {T2fuel:.1f}L", flush=True)
ev = [e["msg"] for e in p["events"]]
n_rtl = sum("RTL engaged" in m for m in ev)
n_rec = sum("RTL complete" in m for m in ev)
print(f"RTL engaged x{n_rtl}, recovered x{n_rec} -> {'PASS' if p['telemetry']['flight_mode'] == 'RECOVERED' and n_rtl == 1 else 'FLAKY/FAIL'}", flush=True)

print("== 5b. destruct still works ==", flush=True)
api.TWIN.reset("endurance")
T.lat, T.lon = T.lat0, T.lon0 + 0.05
T.fuel_l = 0.05
p = run(3)
print(f"mode {p['telemetry']['flight_mode']} -> {'PASS' if p['telemetry']['flight_mode'] == 'DESTROYED' else 'FAIL'}", flush=True)
print("DONE", flush=True)
