"""train_final.py -- final accuracy pass (v3 feature set, no scenario one-hots).

Candidates: AD {IF, AE, PCA, Mahalanobis}@raw | CLF {ET300, ET500, HGB1000}@44 feats
| RUL {HGB500, HGB1000}@44+cycle. Val selection -> refit train+val -> one test report.
Saves compat filenames + generic deployment bundle v5 + metrics_final.json.
"""
import json, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from features import FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED, MODELS = 42, "models"
BASELINE = {"ad_f1": 0.723, "ad_auc": 0.893, "clf_acc": 0.818, "clf_macro_f1": 0.657,
            "rul_mae": 10.67, "rul_r2": 0.878, "note": "leakage-free random mission split; test had 7/8 fault classes"}

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
print(f"TRAIN {len(train)} VAL {len(val)} TEST {len(test)} | feats CLF={len(FEATURES_FAULT)} RUL={len(FEATURES_RUL)}", flush=True)
etr, evl, ete = engineer_features(train), engineer_features(val), engineer_features(test)

# ================= AD =================
print("\nAD...", flush=True)
scaler = StandardScaler().fit(etr[etr.is_anomaly == 0][FEATURES_AD])
Zhtr = scaler.transform(etr[etr.is_anomaly == 0][FEATURES_AD])
Ztr = scaler.transform(etr[FEATURES_AD]); Zvl = scaler.transform(evl[FEATURES_AD]); Zte = scaler.transform(ete[FEATURES_AD])
yvl_ad, yte_ad = evl.is_anomaly.values, ete.is_anomaly.values
iso = IsolationForest(n_estimators=300, contamination=0.30, random_state=SEED, n_jobs=-1).fit(Zhtr)
ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="relu", solver="adam", max_iter=300, random_state=SEED).fit(Zhtr, Zhtr)
pca = PCA(n_components=8, random_state=SEED).fit(Zhtr)
mu, cov_inv = Zhtr.mean(0), np.linalg.pinv(np.cov(Zhtr.T) + 1e-6 * np.eye(Zhtr.shape[1]))

def ad_scores(kind, Z):
    if kind == "iforest": return -iso.decision_function(Z)
    if kind == "autoencoder": return np.mean(np.square(Z - ae.predict(Z)), axis=1)
    if kind == "pca": return np.mean(np.square(Z - pca.inverse_transform(pca.transform(Z))), axis=1)
    d = Z - mu; return np.einsum("ij,jk,ik->i", d, cov_inv, d)

valf1, fam = {}, {}
for k in ["iforest", "autoencoder", "pca", "mahalanobis"]:
    str_, svl = ad_scores(k, Zhtr), ad_scores(k, Zvl)
    valf1[k] = float(f1_score(yvl_ad, (svl > np.percentile(str_, 95)).astype(int)))
    fam[k] = (str_, svl); print(f"  {k:13s} VAL F1 {valf1[k]:.3f}", flush=True)
ad_win = max(valf1, key=valf1.get)
str_, svl = fam[ad_win]; ste = ad_scores(ad_win, Zte)
s_pool = np.concatenate([str_, svl]); y_pool = np.concatenate([np.zeros(len(str_)), yvl_ad])
qs = np.percentile(s_pool, np.arange(70, 99.9, 0.25)); fs = [f1_score(y_pool, (s_pool > t).astype(int)) for t in qs]
t_final = float(qs[int(np.argmax(fs))])
ad_test_f1 = float(f1_score(yte_ad, (ste > t_final).astype(int))); ad_test_auc = float(roc_auc_score(yte_ad, ste))
print(f"=> AD {ad_win} TEST F1 {ad_test_f1:.3f} AUC {ad_test_auc:.3f}", flush=True)

# ================= CLF =================
print("\nCLF...", flush=True)
le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])
ctr, cvl, cte = etr[etr.is_anomaly == 1], evl[evl.is_anomaly == 1], ete[ete.is_anomaly == 1]
ytr, yvl_c, yte_c = le.transform(ctr.fault_type), le.transform(cvl.fault_type), le.transform(cte.fault_type)
cc = {
    "et300": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "et500": ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "hgb1000": HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, early_stopping=False, class_weight="balanced", random_state=SEED),
}
vf = {}
for k, m in cc.items():
    m.fit(ctr[FEATURES_FAULT], ytr); vf[k] = float(f1_score(yvl_c, m.predict(cvl[FEATURES_FAULT]), average="macro"))
    print(f"  {k:8s} VAL macro-F1 {vf[k]:.3f}", flush=True)
cn = max(vf, key=vf.get); print("=> CLF", cn, flush=True)
full_c = pd.concat([ctr, cvl]).reset_index(drop=True)
clf = cc[cn].__class__(**cc[cn].get_params()); clf.fit(full_c[FEATURES_FAULT], le.transform(full_c.fault_type))
yp = clf.predict(cte[FEATURES_FAULT])
acc, mf1 = float(accuracy_score(yte_c, yp)), float(f1_score(yte_c, yp, average="macro"))
print(f"TEST acc {acc:.3f} macro-F1 {mf1:.3f}\n{classification_report(yte_c, yp, target_names=list(le.classes_), digits=3)}", flush=True)

# ================= RUL =================
print("RUL...", flush=True)
rtr = etr[etr.fault_type != "none"].dropna(subset=["RUL"]); rvl = evl[evl.fault_type != "none"].dropna(subset=["RUL"]); rte = ete[ete.fault_type != "none"].dropna(subset=["RUL"])
rc = {"hgb500": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
      "hgb1000": HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED)}
vm = {}
for k, m in rc.items():
    m.fit(rtr[FEATURES_RUL], rtr["RUL"]); vm[k] = float(mean_absolute_error(rvl["RUL"], m.predict(rvl[FEATURES_RUL])))
    print(f"  {k:8s} VAL MAE {vm[k]:.2f}", flush=True)
rn = min(vm, key=vm.get); print("=> RUL", rn, flush=True)
full_r = pd.concat([rtr, rvl]).reset_index(drop=True)
reg = rc[rn].__class__(**rc[rn].get_params()); reg.fit(full_r[FEATURES_RUL], full_r["RUL"])
pr = reg.predict(rte[FEATURES_RUL])
mae, r2 = float(mean_absolute_error(rte["RUL"], pr)), float(r2_score(rte["RUL"], pr))
print(f"TEST MAE {mae:.2f} RMSE {np.sqrt(mean_squared_error(rte['RUL'], pr)):.2f} R2 {r2:.4f}", flush=True)
for f in le.classes_:
    m = rte[rte.fault_type == f]
    if len(m): print(f"  {f:25s} n={len(m):4d} MAE {mean_absolute_error(m['RUL'], reg.predict(m[FEATURES_RUL])):6.2f}", flush=True)

# ================= SAVE =================
joblib.dump(scaler, f"{MODELS}/scaler_ad.joblib"); joblib.dump("raw", f"{MODELS}/ad_features.joblib")
joblib.dump(ad_win, f"{MODELS}/ad_model_type.joblib"); joblib.dump(float(t_final), f"{MODELS}/ad_threshold.joblib")
ad_model = {"mahalanobis": {"mean": mu, "cov_inv": cov_inv}, "pca": pca, "iforest": iso, "autoencoder": ae}[ad_win]
joblib.dump(ad_model, f"{MODELS}/model_ad.joblib")
joblib.dump(iso, f"{MODELS}/model_ad_iforest.joblib")
try: joblib.dump(ae, f"{MODELS}/model_ad_autoencoder.joblib")
except Exception: pass
joblib.dump({"winner": ad_win, "threshold": float(t_final)}, f"{MODELS}/ad_thresholds_both.joblib")
joblib.dump(clf, f"{MODELS}/model_fc_xgboost.joblib"); joblib.dump(le, f"{MODELS}/label_encoder_fault.joblib")
joblib.dump(reg, f"{MODELS}/model_rul_xgboost.joblib")

full = engineer_features(df)
sc2 = StandardScaler().fit(full[full.is_anomaly == 0][FEATURES_AD])
Z2 = sc2.transform(full[full.is_anomaly == 0][FEATURES_AD])
mu2, ci2 = Z2.mean(0), np.linalg.pinv(np.cov(Z2.T) + 1e-6 * np.eye(Z2.shape[1]))
d2 = Z2 - mu2; s2 = np.einsum("ij,jk,ik->i", d2, ci2, d2)
y2 = np.zeros(len(Z2))
le2 = LabelEncoder().fit(full[full.fault_type != "none"]["fault_type"])
an2 = full[full.fault_type != "none"].dropna(subset=["RUL"])
clf2 = clf.__class__(**clf.get_params()); clf2.fit(an2[FEATURES_FAULT], le2.transform(an2["fault_type"]))
reg2 = reg.__class__(**reg.get_params()); reg2.fit(an2[FEATURES_RUL], an2["RUL"])
joblib.dump({"version": "deployment-v5-physics", "features": FEATURES_FAULT, "features_rul": FEATURES_RUL,
             "features_ad": FEATURES_AD, "scaler": sc2, "anomaly_kind": "mahalanobis",
             "anomaly_model": {"mean": mu2, "cov_inv": ci2}, "anomaly_threshold": float(t_final),
             "classifier": clf2, "labels": le2, "rul": reg2}, f"{MODELS}/deployment_bundle.joblib")
print("saved deployment_bundle.joblib (deployment-v5-physics)", flush=True)
metrics = {"baseline_honest": BASELINE,
    "ad": {"winner": ad_win, "val_f1": {k: round(v, 4) for k, v in valf1.items()}, "test_f1": round(ad_test_f1, 4), "test_auc": round(ad_test_auc, 4)},
    "clf": {"winner": cn, "val_macro_f1": {k: round(v, 4) for k, v in vf.items()}, "test_acc": round(acc, 4), "test_macro_f1": round(mf1, 4)},
    "rul": {"winner": rn, "val_mae": {k: round(v, 2) for k, v in vm.items()}, "test_mae": round(mae, 2), "test_r2": round(r2, 4)}}
with open("metrics_final.json", "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("TRAIN FINAL COMPLETE", flush=True)
