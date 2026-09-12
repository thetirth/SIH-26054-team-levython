"""Baseline evaluation of shipped models (group-by-mission, leakage-free)."""
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

FEATURES = ["altitude_m","airspeed_kt","ambient_temp","throttle_pct","engine_age_hours","rpm","engine_load","fuel_flow","egt","cht","oil_temp","oil_pressure","vibration","vib_1x","vib_2x","vib_05x","battery_voltage","alternator_voltage","battery_soc","injection_timing"]

df = pd.read_csv("engine_data.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr, te = next(gss.split(df, groups=df["mission_id"]))
train, test = df.iloc[tr], df.iloc[te]
print(f"TRAIN {len(train)} TEST {len(test)}")

scaler = joblib.load("models/scaler_ad.joblib")
ad_type = joblib.load("models/ad_model_type.joblib")
print("AD type:", ad_type)
Xt = scaler.transform(test[FEATURES]); yt = test["is_anomaly"].values
if ad_type == "autoencoder":
    ae = joblib.load("models/model_ad_autoencoder.joblib")
    thr = joblib.load("models/ad_threshold.joblib")
    mse = np.mean(np.square(Xt - ae.predict(Xt)), axis=1)
    pred = (mse > thr).astype(int)
    try: print("AD ROC-AUC:", round(roc_auc_score(yt, mse), 4))
    except Exception as e: print("AUC n/a", e)
else:
    iso = joblib.load("models/model_ad_iforest.joblib")
    pred = np.where(iso.predict(Xt) == -1, 1, 0)
print("AD F1:", round(f1_score(yt, pred), 4))
print(classification_report(yt, pred, target_names=["Normal", "Anomaly"], digits=3))

clf = joblib.load("models/model_fc_xgboost.joblib")
le = joblib.load("models/label_encoder_fault.joblib")
anom = test[test["is_anomaly"] == 1]
yp = clf.predict(anom[FEATURES]); yt2 = le.transform(anom["fault_type"])
print("FAULT acc:", round(accuracy_score(yt2, yp), 4), "macro-F1:", round(f1_score(yt2, yp, average="macro"), 4))
print(classification_report(yt2, yp, target_names=list(le.classes_), digits=3))

reg = joblib.load("models/model_rul_xgboost.joblib")
rul = test[test["fault_type"] != "none"].dropna(subset=["RUL"])
pr = reg.predict(rul[FEATURES])
print(f"RUL MAE {mean_absolute_error(rul['RUL'], pr):.2f} RMSE {np.sqrt(mean_squared_error(rul['RUL'], pr)):.2f} R2 {r2_score(rul['RUL'], pr):.4f}")
for f in sorted(rul["fault_type"].unique()):
    m = rul[rul["fault_type"] == f]
    print(f"  {f:25s} n={len(m):4d} MAE {mean_absolute_error(m['RUL'], reg.predict(m[FEATURES])):6.2f}")
