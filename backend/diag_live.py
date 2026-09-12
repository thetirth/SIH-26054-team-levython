"""Diagnose live-sim gap, batched (fast)."""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from features import engineer_features, FEATURES_AD, FEATURES_FAULT
from api import TWIN, MODELS

b = MODELS.bundle
sc = b["scaler"]
m = b["anomaly_model"]
mean, Ci = np.asarray(m["mean"]), np.asarray(m["cov_inv"])
sup = b.get("supervisor")
clf, le = b["classifier"], b["labels"]
Dinv = np.sqrt(np.diag(np.linalg.pinv(Ci)) + 1e-9)

for scn in ["hot_weather", "endurance", "high_altitude", "rapid_throttle"]:
    TWIN.reset(scn)
    recs, faults = [], []
    for i in range(150):
        p = TWIN.step()
        t = p["telemetry"]
        recs.append({"altitude_m": t["altitude_m"], "airspeed_kt": t["airspeed_kt"],
               "ambient_temp": t["ambient_temp_c"], "throttle_pct": t["throttle_pct"] / 100.0,
               "engine_age_hours": 480.0 + TWIN.wear * 1800.0, "rpm": t["rpm"],
               "engine_load": t["throttle_pct"] / 100.0, "fuel_flow": t["fuel_flow_lph"],
               "egt": t["egt_c"], "cht": t["cht_c"], "oil_temp": t["oil_temp_c"],
               "oil_pressure": t["oil_pressure_psi"], "vibration": t["vibration_g"],
               "vib_1x": t["vib_1x_g"], "vib_2x": t["vib_2x_g"], "vib_05x": t["vib_05x_g"],
               "battery_voltage": t["battery_v"], "alternator_voltage": t["alternator_v"],
               "battery_soc": t["battery_soc_pct"], "injection_timing": t["injection_timing_deg"],
               "cycle": p["sequence"], "scenario": scn})
        faults.append(p["health"]["fault_mode"])
    eng = engineer_features(pd.DataFrame(recs))
    Z = sc.transform(eng[FEATURES_AD])
    d = Z - mean
    M = np.einsum("ij,jk,ik->i", d, Ci, d)
    z0 = (d[0] / Dinv) ** 2
    topz = sorted(zip(FEATURES_AD, z0), key=lambda kv: -kv[1])[:6]
    P = sup.predict_proba(eng[FEATURES_FAULT])[:, 1]
    F = le.inverse_transform(clf.predict(eng[FEATURES_FAULT]))
    print(f"{scn:15s} MD med {np.median(M):7.1f} p10 {np.percentile(M,10):7.1f} | supAnom {(P>0.5).mean():.2f} | phys {pd.Series(faults).unique()[:3]} | clfTop {pd.Series(F).value_counts().head(2).to_dict()}", flush=True)
    print("   top MD contributors:", [(k, round(float(v),1)) for k, v in topz], flush=True)
print("DONE", flush=True)
