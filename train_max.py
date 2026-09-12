"""train_max.py -- maximum-accuracy campaign (all honest, deployable signals only).

Stages (selection ALWAYS on VAL; TEST scored once at the end):
  0. Feature ablation: base 44 vs base + 3 trial residuals (ET-300, val macro-F1).
  1. Supervised anomaly head (binary, deployable): ET300 / RF200 / HGB500, val F1.
     (Mahalanobis stays as the novelty/OOD watchdog.)
  2. Fault classifier battery: ET300 / ET800 / RF300 / HGB1000 / vote(ET+HGB),
     plus hierarchical rerouting (thermal-trio + misfire/combustion specialists).
  3. RUL battery: global HGB500 / HGB1000 / ET500 / per-fault HGB specialists
     routed by PREDICTED fault (honest end-to-end).
  4. Refit winners on train+val, score TEST once, save (guardrail: keep old
     artifact if val did not improve).

Same stratified mission split (seed 42) as train_final.py, so every number
compares directly with the published baseline.
"""
import json, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (f1_score, accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score)

from features import FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features

SEED, MODELS = 42, "models"
LOG = {}

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

TRIAL = ["oil_p_rpm_resid", "fuel_per_rpm", "egt_cht_gap"]
for d in (etr, evl, ete):
    d["oil_p_rpm_resid"] = d["oil_pressure"] - 0.010 * d["rpm"]
    d["fuel_per_rpm"] = d["fuel_flow"] / (d["rpm"] / 1000.0 + 1e-3)
    d["egt_cht_gap"] = (d["egt"] - 400.0) / 450.0 - (d["cht"] - 60.0) / 90.0

le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])

# ================= 0. FEATURE ABLATION =================
print("STAGE 0: feature ablation (ET300, val macro-F1)", flush=True)
ctr, cvl = etr[etr.is_anomaly == 1], evl[evl.is_anomaly == 1]
ytr, yvl = le.transform(ctr.fault_type), le.transform(cvl.fault_type)
abl = {}
for name, cols in [("base44", FEATURES_FAULT), ("base47", FEATURES_FAULT + TRIAL)]:
    m = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                             class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    m.fit(ctr[cols], ytr)
    abl[name] = float(f1_score(yvl, m.predict(cvl[cols]), average="macro"))
    print(f"  {name}: {abl[name]:.4f}", flush=True)
FF = FEATURES_FAULT + TRIAL if abl["base47"] > abl["base44"] else FEATURES_FAULT
FR = FF + ["cycle"]
print("  => features:", len(FF), "trial kept" if len(FF) > len(FEATURES_FAULT) else "trial dropped", flush=True)
LOG["features"] = {"ablation": abl, "n": len(FF)}

# ================= 1. SUPERVISED AD HEAD =================
print("STAGE 1: supervised AD head (val F1)", flush=True)
btr = etr[FF].copy(); btr["y"] = (etr.is_anomaly == 1).astype(int)
bvl = evl[FF].copy(); bvl["y"] = (evl.is_anomaly == 1).astype(int)
ad_cands = {
    "et300": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "rf200": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "hgb500": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, early_stopping=False, class_weight="balanced", random_state=SEED),
}
ad_val = {}
for k, m in ad_cands.items():
    m.fit(btr[FF], btr["y"]); ad_val[k] = float(f1_score(bvl["y"], m.predict(bvl[FF])))
    print(f"  {k}: {ad_val[k]:.4f}", flush=True)
ad_name = max(ad_val, key=ad_val.get)
print("  => AD head:", ad_name, flush=True)
LOG["ad_head"] = {"val_f1": ad_val, "winner": ad_name}

# ================= 2. CLF BATTERY =================
print("STAGE 2: classifier battery (val macro-F1)", flush=True)
cc = {
    "et300": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "et800": ExtraTreesClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "rf300": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "hgb1000": HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, early_stopping=False, class_weight="balanced", random_state=SEED),
}
vf = {}
for k, m in cc.items():
    m.fit(ctr[FF], ytr); vf[k] = float(f1_score(yvl, m.predict(cvl[FF]), average="macro"))
    print(f"  {k}: {vf[k]:.4f}", flush=True)
# soft vote ET+HGB (kept only with a clear margin: SHAP-ability matters)
pe, ph = cc["et300"].predict_proba(cvl[FF]), cc["hgb1000"].predict_proba(cvl[FF])
vf["vote_et_hgb"] = float(f1_score(yvl, np.argmax(pe + ph, axis=1), average="macro"))
print(f"  vote_et_hgb: {vf['vote_et_hgb']:.4f}", flush=True)
cn = max(vf, key=vf.get)
if cn == "vote_et_hgb" and vf[cn] - max(v for k, v in vf.items() if k != "vote_et_hgb") < 0.015:
    cn = max((k, v) for k, v in vf.items() if k != "vote_et_hgb")[0]
    print("  vote margin < 0.015 -> single model:", cn, flush=True)
print("  => CLF:", cn, flush=True)

# hierarchical specialists (reroute inside confused groups)
TRIO = ["cooling_degradation", "overheating", "sensor_drift"]
PAIR = ["misfire", "combustion_instability"]
def fit_special(classes):
    sub = ctr[ctr.fault_type.isin(classes)]
    m = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    le_s = LabelEncoder().fit(classes)
    m.fit(sub[FF], le_s.transform(sub.fault_type))
    return m, le_s
trio_m, trio_le = fit_special(TRIO)
pair_m, pair_le = fit_special(PAIR)
base_pred = cc[cn].predict(cvl[FF]) if cn != "vote_et_hgb" else np.argmax(cc["et300"].predict_proba(cvl[FF]) + cc["hgb1000"].predict_proba(cvl[FF]), axis=1)
base_lab = le.inverse_transform(base_pred)
final = base_pred.copy()
for idx, lab in enumerate(base_lab):
    if lab in TRIO:
        final[idx] = int(le.transform(trio_le.inverse_transform(trio_m.predict(cvl[FF].iloc[[idx]])))[0])
    elif lab in PAIR:
        final[idx] = int(le.transform(pair_le.inverse_transform(pair_m.predict(cvl[FF].iloc[[idx]])))[0])
hier_f1 = float(f1_score(yvl, final, average="macro"))
print(f"  hierarchical reroute: {hier_f1:.4f} (vs base {vf[cn]:.4f})", flush=True)
use_hier = hier_f1 > vf[cn] + 0.003
print("  => hierarchical:", "ON" if use_hier else "OFF", flush=True)
LOG["clf"] = {"val_macro_f1": vf, "winner": cn, "hier_f1": hier_f1, "use_hier": use_hier}

# ================= 3. RUL BATTERY =================
print("STAGE 3: RUL battery (val MAE, specialists routed by PREDICTED fault)", flush=True)
rtr = etr[etr.fault_type != "none"].dropna(subset=["RUL"]); rvl = evl[evl.fault_type != "none"].dropna(subset=["RUL"])
RFR = FR
rc = {
    "hgb500": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
    "hgb1000": HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED),
    "et500": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
}
vm = {}
for k, m in rc.items():
    m.fit(rtr[RFR], rtr["RUL"]); vm[k] = float(mean_absolute_error(rvl["RUL"], m.predict(rvl[RFR])))
    print(f"  {k}: {vm[k]:.2f}", flush=True)
# per-fault specialists, routed by predicted fault from stage-2 winner on val
val_pred_lab = le.inverse_transform(final if use_hier else base_pred)
# need val-row alignment: cvl rows that have RUL
rvl2 = rvl.copy()
# map: rvl rows are subset of evl; get predicted labels via cvl index alignment
pred_by_pos = dict(zip(cvl.index, val_pred_lab))
rvl2["pred_lab"] = rvl2.index.map(pred_by_pos)
spec = {}
for f in le.classes_:
    sub = rtr[rtr.fault_type == f]
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=SEED)
    m.fit(sub[RFR], sub["RUL"]); spec[f] = m
preds = np.zeros(len(rvl2))
for f, m in spec.items():
    mask = (rvl2.pred_lab == f).values
    if mask.any():
        preds[mask] = m.predict(rvl2.loc[mask, RFR])
vm["specialists_pred_routed"] = float(mean_absolute_error(rvl2["RUL"], preds))
print(f"  specialists_pred_routed: {vm['specialists_pred_routed']:.2f}", flush=True)
rn = min(vm, key=vm.get)
print("  => RUL:", rn, flush=True)
LOG["rul"] = {"val_mae": vm, "winner": rn}

with open("metrics_max_val.json", "w") as f:
    json.dump(LOG, f, indent=2)
print(json.dumps(LOG, indent=2), flush=True)
print("MAX SEARCH COMPLETE (no test touched)", flush=True)
