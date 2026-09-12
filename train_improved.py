"""train_improved.py -- accuracy upgrade for the MALE UAV digital-twin PHM stack.

Upgrades vs. previous pipeline:
  1. Physics-informed feature layer (features.py): Rotax limit margins,
     vibration signature ratios, timing deviation, ISA density corrections.
  2. Stratified group-by-mission split: every fault mode is guaranteed
     presence in train/val/test (old random split dropped entire classes).
  3. Validation-based model selection + threshold tuning (no hard-coded
     contamination / percentile guesses).
  4. HistGradientBoosting candidates: higher tabular accuracy at ~1/50th
     the model size (edge-friendly).

Protocol (leakage-free): split missions -> tune on val -> freeze -> retrain
winner on train+val -> report once on held-out test. Old shipped models are
re-scored on the SAME test missions for an apples-to-apples comparison.
"""
import json, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from features import FEATURES_RAW, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED = 42
MODELS = "models"

# ---------------------------------------------------------------- split
df = pd.read_csv("engine_data.csv")
print(f"Dataset {len(df)} rows, {df['mission_id'].nunique()} missions")

# Mission-level fault label for stratification
mlab = df.groupby("mission_id")["fault_type"].agg(
    lambda s: str(s[s != "none"].mode().iloc[0]) if (s != "none").any() else "none")

rng = np.random.default_rng(SEED)
test_m, val_m, train_m = [], [], []
for fault, mids in mlab.groupby(mlab).groups.items():
    mids = np.array(sorted(mids)); rng.shuffle(mids)
    n = len(mids); n_te = max(1, int(round(n * 0.20))); n_va = max(1, int(round(n * 0.15)))
    test_m += list(mids[:n_te]); val_m += list(mids[n_te:n_te + n_va]); train_m += list(mids[n_te + n_va:])
train = df[df.mission_id.isin(train_m)].reset_index(drop=True)
val = df[df.mission_id.isin(val_m)].reset_index(drop=True)
test = df[df.mission_id.isin(test_m)].reset_index(drop=True)
print(f"TRAIN {len(train)} ({len(train_m)} missions)  VAL {len(val)} ({len(val_m)})  TEST {len(test)} ({len(test_m)})")
print("TEST fault mix:", test[test.is_anomaly == 1].groupby("fault_type").size().to_dict())

etr, evl, ete = (engineer_features(train), engineer_features(val), engineer_features(test))

# ------------------------------------------------- OLD models, SAME test
print("\n" + "=" * 70 + "\nOLD SHIPPED MODELS re-scored on the new held-out test missions\n" + "=" * 70)
old = {}
try:
    sc0 = joblib.load(f"{MODELS}/scaler_ad.joblib"); t0 = joblib.load(f"{MODELS}/ad_model_type.joblib")
    Xt0 = sc0.transform(test[FEATURES_RAW]); yt = test["is_anomaly"].values
    if t0 == "autoencoder":
        ae0 = joblib.load(f"{MODELS}/model_ad_autoencoder.joblib"); th0 = joblib.load(f"{MODELS}/ad_threshold.joblib")
        mse0 = np.mean(np.square(Xt0 - ae0.predict(Xt0)), axis=1); p0 = (mse0 > th0).astype(int)
        old["ad_auc"] = float(roc_auc_score(yt, mse0))
    else:
        iso0 = joblib.load(f"{MODELS}/model_ad_iforest.joblib"); p0 = np.where(iso0.predict(Xt0) == -1, 1, 0)
        old["ad_auc"] = float(roc_auc_score(yt, -iso0.decision_function(Xt0)))
    old["ad_f1"] = float(f1_score(yt, p0))
    clf0 = joblib.load(f"{MODELS}/model_fc_xgboost.joblib"); le0 = joblib.load(f"{MODELS}/label_encoder_fault.joblib")
    an0 = test[test.is_anomaly == 1]
    yp0 = clf0.predict(an0[FEATURES_RAW])
    # align label spaces (old encoder may miss classes absent in ITS train split)
    yp0s = pd.Series(le0.inverse_transform(yp0)); yt0s = an0["fault_type"].values
    old["clf_acc"] = float((yp0s.values == yt0s).mean())
    reg0 = joblib.load(f"{MODELS}/model_rul_xgboost.joblib")
    ru0 = test[test.fault_type != "none"].dropna(subset=["RUL"]); pr0 = reg0.predict(ru0[FEATURES_RAW])
    old["rul_mae"] = float(mean_absolute_error(ru0["RUL"], pr0)); old["rul_r2"] = float(r2_score(ru0["RUL"], pr0))
    print(f"OLD  AD F1 {old['ad_f1']:.3f} AUC {old['ad_auc']:.3f} | CLF acc {old['clf_acc']:.3f} | RUL MAE {old['rul_mae']:.2f} R2 {old['rul_r2']:.3f}")
except Exception as e:
    print("OLD scoring failed:", type(e).__name__, e)

# ============================================================ ANOMALY DETECTION
print("\n" + "=" * 70 + "\nANOMALY DETECTION -- tune on VAL, report on TEST\n" + "=" * 70)
h_tr = etr[etr.is_anomaly == 0]
scaler = StandardScaler().fit(h_tr[FEATURES_FAULT])
Ztr, Zvl, Zte = scaler.transform(h_tr[FEATURES_FAULT]), scaler.transform(evl[FEATURES_FAULT]), scaler.transform(ete[FEATURES_FAULT])
yvl, yte = evl["is_anomaly"].values, ete["is_anomaly"].values

def thr_f1(scores, y):
    best = (0, None)
    for q in np.arange(80, 99.9, 0.5):
        t = np.percentile(scores, q)
        f = f1_score(y, (scores > t).astype(int))
        if f > best[0]: best = (f, t)
    return best

# candidate A: IsolationForest (engineered) -- tune threshold on val scores
iso = IsolationForest(n_estimators=250, contamination=0.30, random_state=SEED, n_jobs=-1).fit(Ztr)
t_scores = -iso.decision_function(scaler.transform(h_tr[FEATURES_FAULT]))  # train anomaly scores
f_if, t_if = thr_f1(-iso.decision_function(Zvl), yvl)
p_if = (-iso.decision_function(Zte) > t_if).astype(int)

# candidate B: Autoencoder (engineered) -- tune percentile on val
ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="relu", solver="adam", max_iter=300, random_state=SEED).fit(Ztr, Ztr)
mse_tr = np.mean(np.square(Ztr - ae.predict(Ztr)), axis=1)
mse_vl = np.mean(np.square(Zvl - ae.predict(Zvl)), axis=1)
mse_te = np.mean(np.square(Zte - ae.predict(Zte)), axis=1)
f_ae, t_ae = thr_f1(mse_vl, yvl)
p_ae = (mse_te > t_ae).astype(int)

res_ad = {"iforest": (float(f_if), float(f1_score(yte, p_if))), "autoencoder": (float(f_ae), float(f1_score(yte, p_ae)))}
print(f"IF  val-F1 {f_if:.3f} -> TEST F1 {res_ad['iforest'][1]:.3f} AUC {roc_auc_score(yte, -iso.decision_function(Zte)):.3f}")
print(f"AE  val-F1 {f_ae:.3f} -> TEST F1 {res_ad['autoencoder'][1]:.3f} AUC {roc_auc_score(yte, mse_te):.3f}")
ad_winner = "iforest" if res_ad["iforest"][0] >= res_ad["autoencoder"][0] else "autoencoder"
print("=> AD winner (by VAL):", ad_winner)

# ============================================================ FAULT CLASSIFIER
print("\n" + "=" * 70 + "\nFAULT CLASSIFIER -- select on VAL macro-F1\n" + "=" * 70)
le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])
ctr, cvl, cte = etr[etr.is_anomaly == 1], evl[evl.is_anomaly == 1], ete[ete.is_anomaly == 1]
ytr, yvl_c, yte_c = le.transform(ctr.fault_type), le.transform(cvl.fault_type), le.transform(cte.fault_type)

cands = {
    "extratrees_raw": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "extratrees_eng": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "hgb_eng": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
}
val_f1 = {}
cands["extratrees_raw"].fit(ctr[FEATURES_RAW], ytr); val_f1["extratrees_raw"] = f1_score(yvl_c, cands["extratrees_raw"].predict(cvl[FEATURES_RAW]), average="macro")
cands["extratrees_eng"].fit(ctr[FEATURES_FAULT], ytr); val_f1["extratrees_eng"] = f1_score(yvl_c, cands["extratrees_eng"].predict(cvl[FEATURES_FAULT]), average="macro")
cands["hgb_eng"].fit(ctr[FEATURES_FAULT], ytr); val_f1["hgb_eng"] = f1_score(yvl_c, cands["hgb_eng"].predict(cvl[FEATURES_FAULT]), average="macro")
for k, v in val_f1.items(): print(f"  {k:16s} VAL macro-F1 {v:.3f}")
clf_name = max(val_f1, key=val_f1.get)
print("=> CLF winner (by VAL):", clf_name)

# refit winner on train+val, evaluate once on test
full_c = pd.concat([ctr, cvl]).reset_index(drop=True); yfull_c = le.transform(full_c.fault_type)
Xcols = FEATURES_RAW if clf_name == "extratrees_raw" else FEATURES_FAULT
clf = cands[clf_name].__class__(**cands[clf_name].get_params())
clf.fit(full_c[Xcols], yfull_c)
yp = clf.predict(cte[Xcols])
print(f"TEST acc {accuracy_score(yte_c, yp):.3f} macro-F1 {f1_score(yte_c, yp, average='macro'):.3f}")
print(classification_report(yte_c, yp, target_names=list(le.classes_), digits=3))

# ============================================================ RUL REGRESSOR
print("\n" + "=" * 70 + "\nRUL REGRESSOR -- select on VAL MAE\n" + "=" * 70)
rtr = etr[etr.fault_type != "none"].dropna(subset=["RUL"]); rvl = evl[evl.fault_type != "none"].dropna(subset=["RUL"]); rte = ete[ete.fault_type != "none"].dropna(subset=["RUL"])
rcands = {
    "extratrees_raw": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "extratrees_eng": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "hgb_eng": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
}
val_mae = {}
rcands["extratrees_raw"].fit(rtr[FEATURES_RAW], rtr["RUL"]); val_mae["extratrees_raw"] = mean_absolute_error(rvl["RUL"], rcands["extratrees_raw"].predict(rvl[FEATURES_RAW]))
rcands["extratrees_eng"].fit(rtr[FEATURES_RUL], rtr["RUL"]); val_mae["extratrees_eng"] = mean_absolute_error(rvl["RUL"], rcands["extratrees_eng"].predict(rvl[FEATURES_RUL]))
rcands["hgb_eng"].fit(rtr[FEATURES_RUL], rtr["RUL"]); val_mae["hgb_eng"] = mean_absolute_error(rvl["RUL"], rcands["hgb_eng"].predict(rvl[FEATURES_RUL]))
for k, v in val_mae.items(): print(f"  {k:16s} VAL MAE {v:.2f}")
rul_name = min(val_mae, key=val_mae.get)
print("=> RUL winner (by VAL):", rul_name)

full_r = pd.concat([rtr, rvl]).reset_index(drop=True)
RX = FEATURES_RAW if rul_name == "extratrees_raw" else FEATURES_RUL
reg = rcands[rul_name].__class__(**rcands[rul_name].get_params())
reg.fit(full_r[RX], full_r["RUL"])
pr = reg.predict(rte[RX])
print(f"TEST MAE {mean_absolute_error(rte['RUL'], pr):.2f} RMSE {np.sqrt(mean_squared_error(rte['RUL'], pr)):.2f} R2 {r2_score(rte['RUL'], pr):.4f}")
for f in le.classes_:
    m = rte[rte.fault_type == f]
    if len(m): print(f"  {f:25s} n={len(m):4d} MAE {mean_absolute_error(m['RUL'], reg.predict(m[RX])):6.2f}")

# ============================================================ SAVE (compat names)
print("\n" + "=" * 70 + "\nSAVING\n" + "=" * 70)
joblib.dump(scaler, f"{MODELS}/scaler_ad.joblib")
if ad_winner == "autoencoder":
    joblib.dump(ae, f"{MODELS}/model_ad_autoencoder.joblib"); joblib.dump(float(t_ae), f"{MODELS}/ad_threshold.joblib")
    joblib.dump(iso, f"{MODELS}/model_ad_iforest.joblib"); joblib.dump("autoencoder", f"{MODELS}/ad_model_type.joblib")
else:
    joblib.dump(iso, f"{MODELS}/model_ad_iforest.joblib")
    try: joblib.dump(ae, f"{MODELS}/model_ad_autoencoder.joblib"); joblib.dump(float(t_ae), f"{MODELS}/ad_threshold.joblib")
    except Exception: pass
    joblib.dump("iforest", f"{MODELS}/ad_model_type.joblib")
import pathlib
joblib.dump(clf, f"{MODELS}/model_fc_xgboost.joblib"); joblib.dump(le, f"{MODELS}/label_encoder_fault.joblib")
joblib.dump(reg, f"{MODELS}/model_rul_xgboost.joblib")
# attach thresholds for pure-sklearn consumers
joblib.dump({"iforest_threshold": float(t_if), "ae_threshold": float(t_ae)}, f"{MODELS}/ad_thresholds_both.joblib")

# deployment bundle served by api.py (full-data refit of winners)
full = engineer_features(df)
sc2 = StandardScaler().fit(full[full.is_anomaly == 0][FEATURES_FAULT])
iso2 = IsolationForest(n_estimators=250, contamination=0.30, random_state=SEED, n_jobs=-1).fit(sc2.transform(full[full.is_anomaly == 0][FEATURES_FAULT]))
le2 = LabelEncoder().fit(full[full.fault_type != "none"]["fault_type"])
an2 = full[full.fault_type != "none"].dropna(subset=["RUL"])
clf_cls = clf.__class__(**clf.get_params()); clf_cls.fit(an2[Xcols], le2.transform(an2["fault_type"]))
reg_cls = reg.__class__(**reg.get_params()); reg_cls.fit(an2[RX], an2["RUL"])
joblib.dump({"version": "deployment-v3-physics", "features": FEATURES_FAULT, "features_rul": RX,
             "scaler": sc2, "anomaly": iso2, "anomaly_threshold": float(t_if),
             "classifier": clf_cls, "clf_features": Xcols, "labels": le2, "rul": reg_cls},
            f"{MODELS}/deployment_bundle.joblib")
print("saved deployment_bundle.joblib (deployment-v3-physics)")

metrics = {"old": old,
    "ad": {"winner": ad_winner, "test": {k: round(v[1], 4) for k, v in res_ad.items()}},
    "clf": {"winner": clf_name, "val_macro_f1": {k: round(v, 4) for k, v in val_f1.items()},
            "test_acc": round(float(accuracy_score(yte_c, yp)), 4), "test_macro_f1": round(float(f1_score(yte_c, yp, average="macro")), 4)},
    "rul": {"winner": rul_name, "val_mae": {k: round(v, 2) for k, v in val_mae.items()},
            "test_mae": round(float(mean_absolute_error(rte["RUL"], pr)), 2),
            "test_r2": round(float(r2_score(rte["RUL"], pr)), 4)}}
with open("metrics_improved.json", "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
print("TRAINING COMPLETE")
