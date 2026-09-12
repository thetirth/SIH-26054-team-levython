"""Debug position integration + destruct math."""
import time as _t
import api

t0 = _t.monotonic()
_calls = [0]


def fake_mono():
    _calls[0] += 1
    return t0 + _calls[0] * 0.5


api.time.monotonic = fake_mono
T = api.TWIN

api.TWIN.reset("endurance")
T.fence_radius_km = 1.0
T.heading_deg = 90.0
T.target_lat = None
for i in range(6):
    p = T.step()
    t = p["telemetry"]
    print(f"step{i}: lat {t['lat']:.6f} lon {t['lon']:.6f} hdg {t['heading_deg']} gs {t['ground_speed_kt']} ias? dist {t['dist_home_km']} thr {t['throttle_pct']}", flush=True)

print("---- destruct math ----", flush=True)
api.TWIN.reset("endurance")
T.lat, T.lon = T.lat0, T.lon0 + 0.05
T.fuel_l = 0.2
p = T.step()
t = p["telemetry"]
print("fuel", t["fuel_l"], "flow?", T.flow_lph, "gs", t["ground_speed_kt"], "dist", t["dist_home_km"], "mode", t["flight_mode"], flush=True)
print("fuel_pct:", 100.0 * T.fuel_l / T.fuel_cap_l, flush=True)
