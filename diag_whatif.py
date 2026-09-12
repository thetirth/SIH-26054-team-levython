"""What-if: how far does MD fall if live signals used dataset formulas?"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from features import engineer_features, FEATURES_AD
from api import TWIN, MODELS

b = MODELS.bundle
sc, m = b["scaler"], b["anomaly_model"]
mean, Ci = np.asarray(m["mean"]), np.asarray(m["cov_inv"])

TWIN.reset("hot_weather")
recs = []
for i in range(120):
    p = TWIN.step(); t = p["telemetry"]
    recs.append({"altitude_m": t["altitude_m"], "airspeed_kt": t["airspeed_kt"],
           "ambient_temp": t["ambient_temp_c"], "throttle_pct": t["throttle_pct"] / 100.0,
           "engine_age_hours": 550.0, "rpm": t["rpm"], "engine_load": t["throttle_pct"] / 100.0,
           "fuel_flow": t["fuel_flow_lph"], "egt": t["egt_c"], "cht": t["cht_c"],
           "oil_temp": t["oil_temp_c"], "oil_pressure": t["oil_pressure_psi"], "vibration": t["vibration_g"],
           "vib_1x": t["vib_1x_g"], "vib_2x": t["vib_2x_g"], "vib_05x": t["vib_05x_g"],
           "battery_voltage": t["battery_v"], "alternator_voltage": t["alternator_v"],
           "battery_soc": t["battery_soc_pct"], "injection_timing": t["injection_timing_deg"],
           "cycle": 200, "scenario": "hot_weather"})
base = pd.DataFrame(recs)

def md(df):
    Z = sc.transform(engineer_features(df)[FEATURES_AD])
    d = Z - mean
    return float(np.median(np.einsum("ij,jk,ik->i", d, Ci, d)))

print("live as-is MD:", round(md(base), 1))
v2 = base.copy(); v2["battery_voltage"] = 12.6 + 0.0001 * v2["rpm"]
print("+dataset battery MD:", round(md(v2), 1))
v3 = v2.copy(); v3["battery_soc"] = 93.5
print("+dataset soc MD:", round(md(v3), 1))
v4 = v3.copy(); v4["oil_temp"] = 55.0 + (v4["oil_temp"] - v4["oil_temp"].mean())
print("+dataset-mean oil MD:", round(md(v4), 1))
v5 = v4.copy(); v5["oil_pressure"] = 45.0 + (v5["oil_pressure"] - v5["oil_pressure"].mean())
print("+dataset-mean oilP MD:", round(md(v5), 1))
print("(cap 535.8, healthy train 99.9th 91.5)", flush=True)
