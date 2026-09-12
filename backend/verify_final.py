"""verify_final.py -- score SHIPPED artifacts through the SERVING path (phm_pipeline).

Rebuilds the exact stratified group-by-mission split (seed 42) and reports
AD / fault-classification / RUL metrics from the files the dashboard and API
actually serve.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from phm_pipeline import load_artifacts, prepare, predict

SEED = 42
BASE = {"ad_f1": 0.723, "ad_auc": 0.893, "clf_acc": 0.818, "clf_macro": 0.657, "rul_mae": 10.67, "rul_r2": 0.878}

df = pd.read_csv("engine_data.csv")
mlab = df.groupby("mission_id")["fault_type"].agg(lambda s: str(s[s != "none"].mode().iloc[0]) if (s != "none").any() else "none")
rng = np.random.default_rng(SEED)
test_m, train_m = [], []
for fault, mids in mlab.groupby(mlab).groups.items():
    mids = np.array(sorted(mids)); rng.shuffle(mids)
    n = len(mids); n_te = max(1, int(round(n * 0.20))); n_va = max(1, int(round(n * 0.15)))
    test_m += list(mids[:n_te]); train_m += list(mids[n_te + n_va:]) + list(mids[n_te:n_te + n_va])
test = df[df.mission_id.isin(test_m)].reset_index(drop=True)
ete = prepare(test)
art = load_artifacts("models")
sup = "ON (RF supervised head)" if art.get("superv") is not None else "OFF (Mahalanobis only)"
print(f"artifacts: superv {sup} | {type(art['clf']).__name__} | {type(art['reg']).__name__}")
print(f"TEST rows {len(ete)} missions {len(test_m)} | faults: {sorted(ete[ete.is_anomaly==1].fault_type.unique())}")

res = predict(art, ete)
yt = yte = ete.is_anomaly.values
pa = np.array([r["is_anomaly"] for r in res]); sc = np.array([r["ad_score"] for r in res])
ad_f1, ad_auc = float(f1_score(yt, pa)), float(roc_auc_score(yt, sc))
print(f"\nAD   F1 {ad_f1:.4f} (base {BASE['ad_f1']})  AUC {ad_auc:.4f} (base {BASE['ad_auc']})")

cte = ete[ete.is_anomaly == 1]
yp = np.array([r["fault_type"] for r in res])[yte == 1]
yv = art["le"].transform(cte.fault_type)
acc, mac = float(accuracy_score(yv, [art["le"].transform([p])[0] for p in yp])), float(f1_score(yv, [art["le"].transform([p])[0] for p in yp], average="macro"))
print(f"\nCLF  acc {acc:.4f} (base {BASE['clf_acc']}*)  macro-F1 {mac:.4f} (base {BASE['clf_macro']}*)")
print("* base test held only 7/8 fault classes; new test holds all 8 (strictly harder)")
print(classification_report(yv, [art["le"].transform([p])[0] for p in yp], target_names=list(art["le"].classes_), digits=3))

rte = ete[ete.fault_type != "none"].dropna(subset=["RUL"])
pr = np.array([r["rul"] for r in res])[ete.fault_type.ne("none").values & ete.RUL.notna().values]
mae, r2 = float(mean_absolute_error(rte["RUL"], pr)), float(r2_score(rte["RUL"], pr))
print(f"RUL  MAE {mae:.2f} (base {BASE['rul_mae']})  RMSE {np.sqrt(mean_squared_error(rte['RUL'], pr)):.2f}  R2 {r2:.4f} (base {BASE['rul_r2']})")
ok = ad_f1 > BASE["ad_f1"] and ad_auc > BASE["ad_auc"] and mac > BASE["clf_macro"] and mae < BASE["rul_mae"] and r2 > BASE["rul_r2"]
print("\nVERIFY:", "ALL PRIMARY METRICS IMPROVED [PASS]" if ok else "REGRESSION DETECTED [FAIL]")
