"""Ops verification: autothrottle, guidance, geofence, failsafes, fixes. Fake clock."""
import time as _t
import api

t0 = _t.monotonic()
_calls = [0]


def fake_mono():
    _calls[0] += 1
    return t0 + _calls[0] * 0.5  # dt clamps at 0.5 -> 0.5 sim-sec per step


api.time.monotonic = fake_mono
T = api.TWIN


def run(n):
    p = None
    for _ in range(n):
        p = T.step()
    return p


print("== 1. autothrottle vs wind ==", flush=True)
api.TWIN.reset("endurance")
T.wind_speed_kt, T.wind_from_deg = 40.0, T.heading_deg  # hard headwind
p = run(40)
th_head = p["telemetry"]["throttle_pct"]
ias_head = p["telemetry"]["airspeed_kt"]
api.TWIN.reset("endurance")
T.wind_speed_kt, T.wind_from_deg = 40.0, (T.heading_deg + 180.0) % 360.0  # tailwind
p = run(40)
th_tail = p["telemetry"]["throttle_pct"]
ias_tail = p["telemetry"]["airspeed_kt"]
print(f"headwind thr {th_head} vs tailwind thr {th_tail} (ias {ias_head}/{ias_tail} vs target 90) -> {'PASS' if th_head > th_tail + 2 and abs(ias_head-90) < 8 and abs(ias_tail-90) < 8 else 'FAIL'}", flush=True)

print("== 2. guidance to target ==", flush=True)
api.TWIN.reset("endurance")
T.target_lat, T.target_lon = T.lat0, T.lon0 + 0.03  # ~3 km east
T.target_reached = False
p = run(160)
d = T._dist_km(T.lat, T.lon, T.target_lat, T.target_lon)
ev = [e["msg"] for e in p["events"]]
print(f"heading {p['telemetry']['heading_deg']} reached={T.target_reached} arrival-event={any('Target reached' in m for m in ev)} -> {'PASS' if T.target_reached else 'FAIL'}", flush=True)

print("== 3. geofence hold then attack cross ==", flush=True)
api.TWIN.reset("endurance")
T.fence_radius_km = 1.0
T.heading_deg = 90.0
T.target_lat = None
p = run(120)
d = p["telemetry"]["dist_home_km"]
ev = [e["msg"] for e in p["events"]]
print(f"surveillance dist {d} (fence 1.0) hold={'PASS' if d <= 1.05 else 'FAIL'} geofence-event={any('Geofence' in m for m in ev)}", flush=True)
api.TWIN.reset("endurance")
TWIN_mode = api.set_mission_mode("ATTACK")
T.fence_radius_km = 1.0
T.heading_deg = 90.0
p = run(120)
d = p["telemetry"]["dist_home_km"]
ev = [e["msg"] for e in p["events"]]
print(f"attack dist {d} crossed={'PASS' if d > 1.0 else 'FAIL'} cross-event={any('Fence crossed' in m for m in ev)}", flush=True)

print("== 4. fuel-critical RTL (possible) ==", flush=True)
api.TWIN.reset("endurance")
T.fuel_l = 3.0
p = run(10)
print(f"mode {p['telemetry']['flight_mode']} (starts at home -> instant RECOVERED) -> {'PASS' if p['telemetry']['flight_mode'] in ('RTL', 'RECOVERED') else 'FAIL'}", flush=True)
p = run(400)
print(f"after long RTL: mode {p['telemetry']['flight_mode']} dist {p['telemetry']['dist_home_km']}", flush=True)

print("== 5. impossible RTL -> destruct ==", flush=True)
api.TWIN.reset("endurance")
T.lat, T.lon = T.lat0, T.lon0 + 0.05  # ~5 km out
T.fuel_l = 0.05  # ~1.7 km range at 5 L/h burn: truly stranded
p = run(3)
print(f"mode {p['telemetry']['flight_mode']} (expect DESTROYED) -> {'PASS' if p['telemetry']['flight_mode'] == 'DESTROYED' else 'FAIL'}", flush=True)

print("== 6. gps jam -> RTL ==", flush=True)
api.TWIN.reset("endurance")
api.inject_event("gps_jam")
p = run(5)
print(f"mode {p['telemetry']['flight_mode']} gps {p['telemetry']['gps_status']} (jam at home -> RTL then RECOVERED) -> {'PASS' if p['telemetry']['gps_status'] == 'jammed' and p['telemetry']['flight_mode'] in ('RTL', 'RECOVERED') else 'FAIL'}", flush=True)
api.inject_event("gps_clear")

print("== 7. non-critical fix proposal + accept ==", flush=True)
api.TWIN.reset("rapid_throttle")
got = None
for _ in range(400):
    p = T.step()
    if p["health"]["fault_mode"] == "injector_abnormality":
        got = p
        break
print("injector fault seen:", got is not None, flush=True)
if got:
    adv = api._advisory_for(got)
    print("advisory fix block:", adv[0].get("fix"), flush=True)
    fid = adv[0]["fix"]["fix_id"]
    print("accept:", api.accept_fix(fid), flush=True)
    print("repairs:", T.repairs, flush=True)

print("== 8. critical auto-fix ==", flush=True)
api.TWIN.reset("hot_weather")
T.cht = 152.0
T.oil_temp = 130.0
p = run(2)
print("fault:", p["health"]["fault_mode"], "| repairs:", T.repairs, "| applied:", T.applied_fixes, flush=True)
ev = [e["msg"] for e in p["events"]]
print("auto-fix event:", any("AUTO-FIX" in m for m in ev), "-> PASS" if T.repairs.get("cooling", 0) >= 0.9 else "-> FAIL", flush=True)
print("ALL DONE", flush=True)
