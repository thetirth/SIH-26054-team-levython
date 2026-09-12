"""Ablate feature groups for CLF (fast HGB, VAL macro-F1 only)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from features import FEATURES_RAW, ENGINEERED, SCENARIO_COLS, engineer_features

SEED = 42
V1_ENG = [c for c in ENGINEERED if c not in ("cht_oil_ratio", "egt_load_resid", "oil_p_temp_comp", "vib_excess")]
RESID = ["cht_oil_ratio", "egt_load_resid", "oil_p_temp_comp", "vib_excess"]

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
ete_tr, ete_vl = engineer_features(train), engineer_features(val)
le = LabelEncoder().fit(df[df.fault_type != "none"]["fault_type"])
ctr = ete_tr[ete_tr.is_anomaly == 1]; cvl = ete_vl[ete_vl.is_anomaly == 1]
ytr, yvl = le.transform(ctr.fault_type), le.transform(cvl.fault_type)

groups = {
    "A raw": FEATURES_RAW,
    "B raw+v1eng": FEATURES_RAW + V1_ENG,
    "C B+resid": FEATURES_RAW + V1_ENG + RESID,
    "D C+scenario": FEATURES_RAW + V1_ENG + RESID + SCENARIO_COLS,
    "E raw+resid": FEATURES_RAW + RESID,
    "F raw+scenario": FEATURES_RAW + SCENARIO_COLS,
}
for name, cols in groups.items():
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20,
                                       early_stopping=False, class_weight="balanced", random_state=SEED)
    m.fit(ctr[cols], ytr)
    print(f"{name:14s} ({len(cols):2d} feats) VAL macro-F1 {f1_score(yvl, m.predict(cvl[cols]), average='macro'):.4f}", flush=True)
