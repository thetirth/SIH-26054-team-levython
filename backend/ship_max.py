"""ship_max.py -- final RF500 check on VAL, refit winners on train+val, score TEST once, ship v6."""
import json, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from features import FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED, MODELS = 42, "models"
BASELINE = {"ad_f1": 0.723, "ad_auc": 0.893, "clf_acc": 0.818, "clf_macro_f1": 0.657,
            "rul_mae": 10.67, "rul_r2": 0.878, "note": "previous shipped pipeline; test had 7/8 classes"}
V5 = {"ad_f1": 0.7899, "ad_auc": 0.9277, "clf_acc": 0.8074, "clf_macro": 0.8053, "rul_mae": 6.57, "rul_r2": 0.9404}

df = pd.read_csv("engine_data.csv")
mlab = df.groupby("mission_id")["fault_type"].agg(lambda s: str(s[s != "none"].mode().iloc[0]) if (s != "none").any() else "none")
rng = np.random.default_rng(SEED)
test_m, val_m, train_m = [], [], []
for fault, mids in mlab.groupby(mlab).groups.items():
    mids = np.array(sorted(mids)); rng.shuffle(mids)
    n = len(mids); n_te = max(1, int(round(n * 0.20))); n_va = max(1, int(round(n * 0.15)))
    test_m += list(mids[:n_te]); val_m += list(mids[n_te:n_te + n_va]); train_m += list(mids[n_te + n_va:])
train = df[df.mission_id.isin(train_m)].reset_index(drop=True)
val = df[df.mission_id.isin(val_m)].reset_index(drop=True)
test = df[df.mission_id.isin(test_m)].reset_index(drop=True)
etr, evl, ete = engineer_features(train), engineer_features(val), engineer_features(test)
le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])

# ---- extra VAL check: RF500 worth it? ----
print("VAL check: RF500 vs winners", flush=True)
ctr, cvl = etr[etr.is_anomaly == 1], evl[evl.is_anomaly == 1]
ytr, yvl = le.transform(ctr.fault_type), le.transform(cvl.fault_type)
rf500 = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
rf500.fit(ctr[FEATURES_FAULT], ytr)
f500 = float(f1_score(yvl, rf500.predict(cvl[FEATURES_FAULT]), average="macro"))
print(f"  CLF rf500 val macro-F1 {f500:.4f} (rf300 0.8582)", flush=True)
CLF_N = 500 if f500 > 0.8582 else 300

btr = etr.copy(); btr["y"] = (etr.is_anomaly == 1).astype(int)
bvl = evl.copy(); bvl["y"] = (evl.is_anomaly == 1).astype(int)
ad500 = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
ad500.fit(btr[FEATURES_FAULT], btr["y"])
a500 = float(f1_score(bvl["y"], ad500.predict(bvl[FEATURES_FAULT])))
print(f"  AD rf500 val F1 {a500:.4f} (rf200 0.9384)", flush=True)
AD_N = 500 if a500 > 0.9384 else 200

# ---- REFIT on train+val ----
print("Refitting on train+val...", flush=True)
full = pd.concat([etr, evl]).reset_index(drop=True)
ctr_f, cvl_t = ete[ete.is_anomaly == 1], None
yte_c = le.transform(ete[ete.is_anomaly == 1].fault_type)

superv = RandomForestClassifier(n_estimators=AD_N, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
superv.fit(full[FEATURES_FAULT], (full.is_anomaly == 1).astype(int))
ste = superv.predict_proba(ete[FEATURES_FAULT])[:, 1]
yte_ad = ete.is_anomaly.values
ad_f1 = float(f1_score(yte_ad, (ste > 0.5).astype(int))); ad_auc = float(roc_auc_score(yte_ad, ste))
print(f"TEST SUP-AD F1 {ad_f1:.4f} AUC {ad_auc:.4f}", flush=True)

clf = RandomForestClassifier(n_estimators=CLF_N, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
anf = full[full.is_anomaly == 1]
clf.fit(anf[FEATURES_FAULT], le.transform(anf.fault_type))
yp = clf.predict(ctr_f[FEATURES_FAULT])
acc = float(accuracy_score(yte_c, yp)); mac = float(f1_score(yte_c, yp, average="macro"))
print(f"TEST CLF acc {acc:.4f} macro-F1 {mac:.4f}", flush=True)
print(classification_report(yte_c, yp, target_names=list(le.classes_), digits=3), flush=True)

reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED)
rfull = full[full.fault_type != "none"].dropna(subset=["RUL"])
reg.fit(rfull[FEATURES_RUL], rfull["RUL"])
rte = ete[ete.fault_type != "none"].dropna(subset=["RUL"])
pr = reg.predict(rte[FEATURES_RUL])
mae = float(mean_absolute_error(rte["RUL"], pr)); r2 = float(r2_score(rte["RUL"], pr))
print(f"TEST RUL MAE {mae:.2f} RMSE {np.sqrt(mean_squared_error(rte['RUL'], pr)):.2f} R2 {r2:.4f}", flush=True)
for f in le.classes_:
    m = rte[rte.fault_type == f]
    if len(m): print(f"  {f:25s} n={len(m):4d} MAE {mean_absolute_error(m['RUL'], reg.predict(m[FEATURES_RUL])):6.2f}", flush=True)

# ---- Mahalanobis watchdog (refit, same protocol) ----
scaler = StandardScaler().fit(full[full.is_anomaly == 0][FEATURES_AD])
Zhtr = scaler.transform(full[full.is_anomaly == 0][FEATURES_AD])
mu, cov_inv = Zhtr.mean(0), np.linalg.pinv(np.cov(Zhtr.T) + 1e-6 * np.eye(Zhtr.shape[1]))

# ---- SAVE ----
joblib.dump(scaler, f"{MODELS}/scaler_ad.joblib"); joblib.dump("raw", f"{MODELS}/ad_features.joblib")
joblib.dump("mahalanobis", f"{MODELS}/ad_model_type.joblib")
joblib.dump({"mean": mu, "cov_inv": cov_inv}, f"{MODELS}/model_ad.joblib")
joblib.dump(superv, f"{MODELS}/model_superv_ad.joblib"); joblib.dump(0.5, f"{MODELS}/superv_threshold.joblib")
# watchdog operating threshold recomputed on train+val pool
Zall = scaler.transform(full[FEATURES_AD])
d = Zall - mu
sall = np.einsum("ij,jk,ik->i", d, cov_inv, d)
yall = (full.is_anomaly == 1).values
qs = np.percentile(sall, np.arange(70, 99.9, 0.25)); fs = [f1_score(yall, (sall > t).astype(int)) for t in qs]
t_final = float(qs[int(np.argmax(fs))])
joblib.dump(t_final, f"{MODELS}/ad_threshold.joblib")
joblib.dump({"winner": "mahalanobis", "threshold": t_final}, f"{MODELS}/ad_thresholds_both.joblib")
joblib.dump(clf, f"{MODELS}/model_fc_xgboost.joblib"); joblib.dump(le, f"{MODELS}/label_encoder_fault.joblib")
joblib.dump(reg, f"{MODELS}/model_rul_xgboost.joblib")

fulld = engineer_features(df)
sc2 = StandardScaler().fit(fulld[fulld.is_anomaly == 0][FEATURES_AD])
Z2 = sc2.transform(fulld[fulld.is_anomaly == 0][FEATURES_AD])
mu2, ci2 = Z2.mean(0), np.linalg.pinv(np.cov(Z2.T) + 1e-6 * np.eye(Z2.shape[1]))
Za = sc2.transform(fulld[FEATURES_AD]); da = Za - mu2
sa = np.einsum("ij,jk,ik->i", da, ci2, da)
ood = float(np.percentile(sa[fulld.is_anomaly == 1], 99.9))
le2 = LabelEncoder().fit(fulld[fulld.fault_type != "none"]["fault_type"])
an2 = fulld[fulld.fault_type != "none"].dropna(subset=["RUL"])
sup2 = RandomForestClassifier(n_estimators=AD_N, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
sup2.fit(fulld[FEATURES_FAULT], (fulld.is_anomaly == 1).astype(int))
clf2 = RandomForestClassifier(n_estimators=CLF_N, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
clf2.fit(an2[FEATURES_FAULT], le2.transform(an2["fault_type"]))
reg2 = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED)
reg2.fit(an2[FEATURES_RUL], an2["RUL"])
joblib.dump({"version": "deployment-v6-max", "features": FEATURES_FAULT, "features_rul": FEATURES_RUL,
             "features_ad": FEATURES_AD, "scaler": sc2, "anomaly_kind": "mahalanobis",
             "anomaly_model": {"mean": mu2, "cov_inv": ci2}, "anomaly_threshold": t_final, "ood_cap": ood,
             "supervisor": sup2, "supervisor_features": FEATURES_FAULT, "supervisor_threshold": 0.5,
             "classifier": clf2, "labels": le2, "rul": reg2}, f"{MODELS}/deployment_bundle.joblib")
print("saved deployment_bundle.joblib (deployment-v6-max), ood_cap", round(ood, 1), flush=True)

metrics = {"baseline": BASELINE, "v5": V5,
    "v6": {"sup_ad_test_f1": round(ad_f1, 4), "sup_ad_test_auc": round(ad_auc, 4),
           "clf_test_acc": round(acc, 4), "clf_test_macro_f1": round(mac, 4),
           "rul_test_mae": round(mae, 2), "rul_test_r2": round(r2, 4),
           "clf_trees": CLF_N, "sup_trees": AD_N}}
with open("metrics_v6.json", "w") as f: json.dump(metrics, f, indent=2)
with open("metrics_final.json", "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps(metrics["v6"], indent=2), flush=True)
print("SHIP MAX COMPLETE", flush=True)
