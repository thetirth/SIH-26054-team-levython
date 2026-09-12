"""Diagnose new models: confusion matrix + AD threshold headroom analysis."""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from features import FEATURES_RAW, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED = 42
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

clf = joblib.load("models/model_fc_xgboost.joblib"); le = joblib.load("models/label_encoder_fault.joblib")
Xc = FEATURES_FAULT
cte = ete[ete.is_anomaly == 1]; yte = le.transform(cte.fault_type); yp = clf.predict(cte[Xc])
print("CLASSES:", list(le.classes_))
print(classification_report(yte, yp, target_names=list(le.classes_), digits=3))
print("CONFUSION (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(yte, yp), index=le.classes_, columns=le.classes_).to_string())
# top confusions
cm = confusion_matrix(yte, yp)
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] >= 20:
            print(f"  true={le.classes_[i]:22s} pred={le.classes_[j]:22s} n={cm[i,j]}")

# worst-mission analysis: which test missions are hardest?
cte2 = cte.copy(); cte2["pred"] = le.inverse_transform(yp); cte2["ok"] = (cte2["fault_type"] == cte2["pred"]).astype(int)
print("\nPer-mission CLF accuracy (worst 8):")
ma = cte2.groupby("mission_id")["ok"].mean()
print(ma.sort_values().head(8).to_string())
print("\nScenario of worst missions:")
worst = ma.sort_values().head(8).index.get_level_values("mission_id")
print(df[df.mission_id.isin(worst)].groupby(["mission_id", "scenario", "fault_type"]).size().to_string())

# ---- AD headroom: best achievable TEST F1 per score (analysis only) ----
scaler = joblib.load("models/scaler_ad.joblib")
Zte = scaler.transform(ete[FEATURES_FAULT]); yte_ad = ete["is_anomaly"].values
Zvl = scaler.transform(evl[FEATURES_FAULT]); yvl_ad = evl["is_anomaly"].values
Ztr_h = scaler.transform(etr[etr.is_anomaly == 0][FEATURES_FAULT])
ae = joblib.load("models/model_ad_autoencoder.joblib")
iso = joblib.load("models/model_ad_iforest.joblib")
mse_te = np.mean(np.square(Zte - ae.predict(Zte)), axis=1)
mse_vl = np.mean(np.square(Zvl - ae.predict(Zvl)), axis=1)
mse_tr = np.mean(np.square(Ztr_h - ae.predict(Ztr_h)), axis=1)
s_te = -iso.decision_function(Zte); s_vl = -iso.decision_function(Zvl)
print(f"\nAE MSE: train95={np.percentile(mse_tr,95):.4f} train99={np.percentile(mse_tr,99):.4f} | val-best?', test-headroom:")
for name, sv, st in [("AE", mse_te, mse_vl), ("IF", s_te, s_vl)]:
    # val-tuned (as done in training)
    best = max(((f1_score(yvl_ad, (st > t).astype(int)), t) for t in np.percentile(st, np.arange(80, 99.9, 0.5))), key=lambda x: x[0])
    f_te_valthr = f1_score(yte_ad, (sv > best[1]).astype(int))
    # test-oracle headroom
    oracle = max(f1_score(yte_ad, (sv > t).astype(int)) for t in np.percentile(sv, np.arange(50, 99.9, 0.5)))
    print(f"  {name}: val-thr test-F1={f_te_valthr:.3f} | oracle test-F1={oracle:.3f} AUC={roc_auc_score(yte_ad, sv):.3f}")
# raw-feature AE headroom (old approach, tuned threshold)
from sklearn.preprocessing import StandardScaler
sc0 = StandardScaler().fit(etr[etr.is_anomaly == 0][FEATURES_RAW])
from sklearn.neural_network import MLPRegressor
print("(raw-feature AE would need retraining -- skipped; see train script)")
