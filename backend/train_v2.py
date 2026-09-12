"""train_v2.py -- second accuracy pass.

Fixes from v1 diagnosis:
  * AD back on RAW signals (engineered correlations collapsed AE/IF margins);
    4 candidates: IsolationForest / Autoencoder / PCA-reconstruction /
    Mahalanobis. Family picked on VAL at a fixed 95th-pct operating point,
    final threshold tuned once on pooled train+val (1 scalar), tested once.
  * CLF: decoupling residuals (cht_oil_ratio, egt_load_resid,
    oil_p_temp_comp, vib_excess) + scenario calibration one-hots attack the
    cooling/overheating/sensor_drift confusion; depth-capped ET candidate
    for edge-size; stronger HGB candidate.
  * RUL: deeper HGB candidate.
Honest baseline (leakage-free random mission split, evaluate_baseline.py):
  AD F1 0.723 / AUC 0.893; CLF acc 0.818 / macro-F1 0.657 (only 7 fault
  classes present); RUL MAE 10.67 / R2 0.878.
"""
import json, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from features import FEATURES_RAW, FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED, MODELS = 42, "models"
BASELINE = {"ad_f1": 0.723, "ad_auc": 0.893, "clf_acc": 0.818, "clf_macro_f1": 0.657,
            "rul_mae": 10.67, "rul_r2": 0.878, "note": "random mission split, 7/8 fault classes in test"}

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
print(f"TRAIN {len(train)} VAL {len(val)} TEST {len(test)} | TEST faults: {test[test.is_anomaly==1].groupby('fault_type').size().to_dict()}")
etr, evl, ete = engineer_features(train), engineer_features(val), engineer_features(test)

def pooled_threshold(scores_tr, y_tr, scores_pool_extra=None, y_extra=None):
    s = np.concatenate([scores_tr] + ([scores_pool_extra] if scores_pool_extra is not None else []))
    y = np.concatenate([y_tr] + ([y_extra] if y_extra is not None else []))
    qs = np.percentile(s, np.arange(70, 99.9, 0.25))
    fs = [f1_score(y, (s > t).astype(int)) for t in qs]
    return float(qs[int(np.argmax(fs))]), float(np.max(fs))

# ============================================================ AD (raw signals)
print("\n" + "=" * 70 + "\nAD -- raw signals, family on VAL@95th-pct, threshold on train+val\n" + "=" * 70)
scaler = StandardScaler().fit(etr[etr.is_anomaly == 0][FEATURES_AD])
H = lambda d: scaler.transform(d[d.is_anomaly == 0][FEATURES_AD])
A = lambda d: scaler.transform(d[FEATURES_AD])
Zhtr, Zhvl = H(etr), H(evl)
Ztr, Zvl, Zte = A(etr), A(evl), A(ete)
ytr_ad, yvl_ad, yte_ad = etr.is_anomaly.values, evl.is_anomaly.values, ete.is_anomaly.values

iso = IsolationForest(n_estimators=300, contamination=0.30, random_state=SEED, n_jobs=-1).fit(Zhtr)
ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="relu", solver="adam", max_iter=300, random_state=SEED).fit(Zhtr, Zhtr)
pca = PCA(n_components=8, random_state=SEED).fit(Zhtr)
mu, cov_inv = Zhtr.mean(0), np.linalg.pinv(np.cov(Zhtr.T) + 1e-6 * np.eye(Zhtr.shape[1]))

def scores(kind, Z):
    if kind == "iforest": return -iso.decision_function(Z)
    if kind == "autoencoder": return np.mean(np.square(Z - ae.predict(Z)), axis=1)
    if kind == "pca": return np.mean(np.square(Z - pca.inverse_transform(pca.transform(Z))), axis=1)
    d = Z - mu; return np.einsum("ij,jk,ik->i", d, cov_inv, d)

fams, valf1 = {}, {}
for k in ["iforest", "autoencoder", "pca", "mahalanobis"]:
    str_, svl = scores(k, Zhtr), scores(k, Zvl)
    t95 = float(np.percentile(str_, 95))
    valf1[k] = float(f1_score(yvl_ad, (svl > t95).astype(int)))
    fams[k] = (str_, svl)
    print(f"  {k:13s} VAL F1@95th-pct {valf1[k]:.3f}")
ad_win = max(valf1, key=valf1.get)
str_, svl = fams[ad_win]
ste = scores(ad_win, Zte)
# NOTE: str_ holds healthy-only scores (label 0); pool with val scores+labels:
s_pool = np.concatenate([str_, svl]); y_pool = np.concatenate([np.zeros(len(str_)), yvl_ad])
qs = np.percentile(s_pool, np.arange(70, 99.9, 0.25)); fs = [f1_score(y_pool, (s_pool > t).astype(int)) for t in qs]
t_final, f_pool = float(qs[int(np.argmax(fs))]), float(np.max(fs))
p_ad = (ste > t_final).astype(int)
ad_test_f1 = float(f1_score(yte_ad, p_ad)); ad_test_auc = float(roc_auc_score(yte_ad, ste))
print(f"=> AD winner: {ad_win} | pool-F1 {f_pool:.3f} @thr {t_final:.4f} | TEST F1 {ad_test_f1:.3f} AUC {ad_test_auc:.3f} (baseline F1 0.723/AUC 0.893)")

# ============================================================ CLF
print("\n" + "=" * 70 + "\nCLF -- select on VAL macro-F1\n" + "=" * 70)
le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])
ctr, cvl, cte = etr[etr.is_anomaly == 1], evl[evl.is_anomaly == 1], ete[ete.is_anomaly == 1]
ytr, yvl_c, yte_c = le.transform(ctr.fault_type), le.transform(cvl.fault_type), le.transform(cte.fault_type)
cands = {
    "et_eng": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "et_eng_shallow": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, max_depth=30, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "hgb_eng": HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, early_stopping=False, class_weight="balanced", random_state=SEED),
}
val_f1 = {}
for k, m in cands.items():
    m.fit(ctr[FEATURES_FAULT], ytr); val_f1[k] = float(f1_score(yvl_c, m.predict(cvl[FEATURES_FAULT]), average="macro"))
    print(f"  {k:14s} VAL macro-F1 {val_f1[k]:.3f}")
clf_name = max(val_f1, key=val_f1.get)
print("=> CLF winner:", clf_name)
full_c = pd.concat([ctr, cvl]).reset_index(drop=True)
clf = cands[clf_name].__class__(**cands[clf_name].get_params()); clf.fit(full_c[FEATURES_FAULT], le.transform(full_c.fault_type))
yp = clf.predict(cte[FEATURES_FAULT])
print(f"TEST acc {accuracy_score(yte_c, yp):.3f} macro-F1 {f1_score(yte_c, yp, average='macro'):.3f} (baseline acc 0.818/macro 0.657*)")
print(classification_report(yte_c, yp, target_names=list(le.classes_), digits=3))

# ============================================================ RUL
print("\n" + "=" * 70 + "\nRUL -- select on VAL MAE\n" + "=" * 70)
rtr = etr[etr.fault_type != "none"].dropna(subset=["RUL"]); rvl = evl[evl.fault_type != "none"].dropna(subset=["RUL"]); rte = ete[ete.fault_type != "none"].dropna(subset=["RUL"])
rcands = {
    "hgb_500": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
    "hgb_1000": HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
}
val_mae = {}
for k, m in rcands.items():
    m.fit(rtr[FEATURES_RUL], rtr["RUL"]); val_mae[k] = float(mean_absolute_error(rvl["RUL"], m.predict(rvl[FEATURES_RUL])))
    print(f"  {k:8s} VAL MAE {val_mae[k]:.2f}")
rul_name = min(val_mae, key=val_mae.get)
print("=> RUL winner:", rul_name)
full_r = pd.concat([rtr, rvl]).reset_index(drop=True)
reg = rcands[rul_name].__class__(**rcands[rul_name].get_params()); reg.fit(full_r[FEATURES_RUL], full_r["RUL"])
pr = reg.predict(rte[FEATURES_RUL])
print(f"TEST MAE {mean_absolute_error(rte['RUL'], pr):.2f} RMSE {np.sqrt(mean_squared_error(rte['RUL'], pr)):.2f} R2 {r2_score(rte['RUL'], pr):.4f} (baseline MAE 10.67/R2 0.878)")
for f in le.classes_:
    m = rte[rte.fault_type == f]
    if len(m): print(f"  {f:25s} n={len(m):4d} MAE {mean_absolute_error(m['RUL'], reg.predict(m[FEATURES_RUL])):6.2f}")

# ============================================================ SAVE
print("\n" + "=" * 70 + "\nSAVING\n" + "=" * 70)
joblib.dump(scaler, f"{MODELS}/scaler_ad.joblib")
joblib.dump("raw", f"{MODELS}/ad_features.joblib")
if ad_win == "autoencoder":
    joblib.dump(ae, f"{MODELS}/model_ad_autoencoder.joblib"); joblib.dump("autoencoder", f"{MODELS}/ad_model_type.joblib")
else:
    joblib.dump("iforest" if ad_win == "iforest" else ad_win, f"{MODELS}/ad_model_type.joblib")
    try: joblib.dump(ae, f"{MODELS}/model_ad_autoencoder.joblib")
    except Exception: pass
    joblib.dump(iso, f"{MODELS}/model_ad_iforest.joblib")
joblib.dump(float(t_final), f"{MODELS}/ad_threshold.joblib")
joblib.dump({"winner": ad_win, "threshold": float(t_final)}, f"{MODELS}/ad_thresholds_both.joblib")
if ad_win == "pca": joblib.dump(pca, f"{MODELS}/model_ad_pca.joblib")
if ad_win == "mahalanobis": joblib.dump({"mean": mu, "cov_inv": cov_inv}, f"{MODELS}/model_ad_mahal.joblib")
joblib.dump(clf, f"{MODELS}/model_fc_xgboost.joblib"); joblib.dump(le, f"{MODELS}/label_encoder_fault.joblib")
joblib.dump(reg, f"{MODELS}/model_rul_xgboost.joblib")

full = engineer_features(df)
sc2 = StandardScaler().fit(full[full.is_anomaly == 0][FEATURES_AD])
iso2 = IsolationForest(n_estimators=300, contamination=0.30, random_state=SEED, n_jobs=-1).fit(sc2.transform(full[full.is_anomaly == 0][FEATURES_AD]))
le2 = LabelEncoder().fit(full[full.fault_type != "none"]["fault_type"])
an2 = full[full.fault_type != "none"].dropna(subset=["RUL"])
clf2 = clf.__class__(**clf.get_params()); clf2.fit(an2[FEATURES_FAULT], le2.transform(an2["fault_type"]))
reg2 = reg.__class__(**reg.get_params()); reg2.fit(an2[FEATURES_RUL], an2["RUL"])
joblib.dump({"version": "deployment-v4-physics", "features": FEATURES_FAULT, "features_rul": FEATURES_RUL,
             "features_ad": FEATURES_AD, "scaler": sc2, "anomaly": iso2, "anomaly_threshold": float(t_final),
             "anomaly_winner": ad_win, "classifier": clf2, "labels": le2, "rul": reg2}, f"{MODELS}/deployment_bundle.joblib")
print("saved deployment_bundle.joblib (deployment-v4-physics)")

metrics = {"baseline_honest": BASELINE,
    "ad": {"winner": ad_win, "val_f1": {k: round(v, 4) for k, v in valf1.items()}, "test_f1": round(ad_test_f1, 4), "test_auc": round(ad_test_auc, 4)},
    "clf": {"winner": clf_name, "val_macro_f1": {k: round(v, 4) for k, v in val_f1.items()},
            "test_acc": round(float(accuracy_score(yte_c, yp)), 4), "test_macro_f1": round(float(f1_score(yte_c, yp, average="macro")), 4)},
    "rul": {"winner": rul_name, "val_mae": {k: round(v, 2) for k, v in val_mae.items()},
            "test_mae": round(float(mean_absolute_error(rte["RUL"], pr)), 2), "test_r2": round(float(r2_score(rte["RUL"], pr)), 4)}}
with open("metrics_improved.json", "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("TRAIN V2 COMPLETE")
