"""
generate_eval_charts.py
=======================
Final-model evaluation charts for DRDO SIH-26054 (shipped v6 artifacts).

Evaluates the ACTUAL serving models in models/ on honest held-out TEST
missions (stratified Group-by-Mission split, seed 42 — same protocol as
train_final.py / ship_max.py), and saves publication charts + metrics.

Outputs -> eval_charts/ :
  01_ad_roc_comparison.png        ROC curves + AUC (Supervised RF200, Mahalanobis, IF, AE, PCA)
  02_ad_pr_comparison.png         Precision-Recall curves + AP
  03_ad_score_distributions.png   Score histograms (healthy vs anomaly)
  04_outlier_detector_benchmark.png  Bar chart F1 + AUC across all outlier detectors
  05_confusion_matrix_counts.png   Fault classifier 8x8 counts
  06_confusion_matrix_normalized.png  row-normalized recall matrix
  07_perclass_prf1.png             Per-fault precision/recall/F1 bars
  08_rul_predicted_vs_actual.png  RUL scatter + MAE/RMSE/R2 box
  09_rul_residuals_hist.png        RUL error histogram + per-fault MAE bars
  10_feature_importance_clf.png    Top-20 RF-300 importances
  11_feature_importance_rul.png    Top-20 HGB RUL permutation importances
  + eval_metrics.json             All numbers (AUC, F1, CM, report, RUL)

Run: python generate_eval_charts.py
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, classification_report,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, roc_curve,
)
from sklearn.inspection import permutation_importance

from features import FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features
from phm_pipeline import load_artifacts

SEED = 42
OUT = "eval_charts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.size": 11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

COLORS = {
    "superv": "#58a6ff", "mahalanobis": "#f97316", "iforest": "#2ecc71",
    "autoencoder": "#e74c3c", "pca": "#9b59b6",
}
NAMES = {
    "superv": "Supervised RF-200 (primary)",
    "mahalanobis": "Mahalanobis (watchdog)",
    "iforest": "IsolationForest-300",
    "autoencoder": "Autoencoder MLP(12,6,12)",
    "pca": "PCA-8 recon.",
}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved {p}")


print("=" * 64)
print("FINAL-MODEL EVAL CHARTS  (shipped v6 artifacts, held-out TEST)")
print("=" * 64)

# ---------- 1. Data + honest stratified mission split (seed 42) ----------
df = pd.read_csv("engine_data.csv")
print(f"rows={len(df):,} missions={df['mission_id'].nunique()}")

mlab = df.groupby("mission_id")["fault_type"].agg(
    lambda s: str(s[s != "none"].mode().iloc[0]) if (s != "none").any() else "none")
rng = np.random.default_rng(SEED)
test_m, val_m, train_m = [], [], []
for fault, mids in mlab.groupby(mlab).groups.items():
    mids = np.array(sorted(mids))
    rng.shuffle(mids)
    n = len(mids)
    n_te = max(1, int(round(n * 0.20)))
    n_va = max(1, int(round(n * 0.15)))
    test_m += list(mids[:n_te])
    val_m += list(mids[n_te:n_te + n_va])
    train_m += list(mids[n_te + n_va:])
trainval_m = train_m + val_m
print(f"missions train={len(train_m)} val={len(val_m)} test={len(test_m)}")

ete = engineer_features(df[df.mission_id.isin(test_m)].reset_index(drop=True))
etv = engineer_features(df[df.mission_id.isin(trainval_m)].reset_index(drop=True))

# ---------- 2. Load shipped artifacts ----------
art = load_artifacts("models")
print(f"AD kind={art['ad_kind']} clf={type(art['clf']).__name__} "
      f"reg={type(art['reg']).__name__} superv={'yes' if art.get('superv') is not None else 'no'}")

y_ad = ete["is_anomaly"].values.astype(int)

# supervised scores (primary detector)
s_sup = np.asarray(art["superv"].predict_proba(ete[FEATURES_FAULT])[:, 1])
# mahalanobis scores (watchdog, same serving path)
Zt = art["scaler"].transform(ete[FEATURES_AD])
m = art["ad_model"]
d = Zt - np.asarray(m["mean"])
s_mah = np.einsum("ij,jk,ik->i", d, np.asarray(m["cov_inv"]), d)
# legacy isolation forest (same scaler for fair comparison)
iso = joblib.load("models/model_ad_iforest.joblib")
s_if = -iso.decision_function(Zt)
# legacy autoencoder MLP
try:
    ae = joblib.load("models/model_ad_autoencoder.joblib")
    s_ae = np.mean(np.square(Zt - ae.predict(Zt)), axis=1)
    has_ae = True
except Exception as e:
    print(f"  [warn] autoencoder unavailable: {e}")
    s_ae, has_ae = None, False
# PCA-8 fitted on trainval healthy (same recipe as train_final.py)
Zhv = art["scaler"].transform(etv[etv.is_anomaly == 0][FEATURES_AD])
pca = PCA(n_components=8, random_state=SEED).fit(Zhv)
s_pca = np.mean(np.square(Zt - pca.inverse_transform(pca.transform(Zt))), axis=1)

scores = {"superv": s_sup, "mahalanobis": s_mah, "iforest": s_if, "pca": s_pca}
if has_ae:
    scores["autoencoder"] = s_ae

# thresholds WITHOUT test leakage: superv fixed 0.5; others = 95th pct of trainval-healthy
Zhv_all = Zhv
thr = {"superv": float(art.get("superv_threshold", 0.5))}
d_hv = Zhv_all - np.asarray(m["mean"])
thr["mahalanobis"] = float(np.percentile(
    np.einsum("ij,jk,ik->i", d_hv, np.asarray(m["cov_inv"]), d_hv), 95))
thr["iforest"] = float(np.percentile(-iso.decision_function(Zhv_all), 95))
thr["pca"] = float(np.percentile(
    np.mean(np.square(Zhv_all - pca.inverse_transform(pca.transform(Zhv_all))), axis=1), 95))
if has_ae:
    thr["autoencoder"] = float(np.percentile(
        np.mean(np.square(Zhv_all - ae.predict(Zhv_all)), axis=1), 95))
# shipped operating point for reference
thr["mahalanobis_shipped"] = float(art["ad_threshold"])

metrics = {"split": {"train": len(train_m), "val": len(val_m), "test": len(test_m),
                     "test_rows": int(len(ete))},
           "ad": {}, "thresholds_trainval95": {k: round(float(v), 5) for k, v in thr.items() if not k.endswith("shipped")},
           "ad_shipped_mahalanobis_threshold": round(float(art["ad_threshold"]), 5)}

for k, s in scores.items():
    fpr, tpr, _ = roc_curve(y_ad, s)
    prec, rec, _ = precision_recall_curve(y_ad, s)
    pred = (s > thr[k]).astype(int)
    metrics["ad"][k] = {
        "auc": round(float(auc(fpr, tpr)), 4),
        "ap": round(float(average_precision_score(y_ad, s)), 4),
        "f1_at_trainval95_thr": round(float(f1_score(y_ad, pred)), 4),
    }
# shipped-threshold operating point for primary + watchdog
metrics["ad"]["superv"]["f1_at_0.5"] = metrics["ad"]["superv"].pop("f1_at_trainval95_thr")
pred_ship = (s_mah > art["ad_threshold"]).astype(int)
metrics["ad"]["mahalanobis"]["f1_at_shipped_thr"] = round(float(f1_score(y_ad, pred_ship)), 4)
print(json.dumps(metrics["ad"], indent=2))

# ---------- 3. 01 ROC comparison ----------
fig, ax = plt.subplots(figsize=(9, 8))
for k, s in scores.items():
    fpr, tpr, _ = roc_curve(y_ad, s)
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[k], linewidth=2.4, label=f"{NAMES[k]} (AUC={a:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#8b949e", lw=1, label="Chance")
ax.fill_between(*roc_curve(y_ad, s_sup)[:2], alpha=0.06, color=COLORS["superv"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Anomaly Detection — ROC Curves (held-out TEST missions)\n"
             "Shipped v6: Supervised RF-200 primary + Mahalanobis watchdog", pad=12)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
save(fig, "01_ad_roc_comparison.png")

# ---------- 4. 02 PR comparison ----------
fig, ax = plt.subplots(figsize=(9, 8))
for k, s in scores.items():
    prec, rec, _ = precision_recall_curve(y_ad, s)
    ap = average_precision_score(y_ad, s)
    ax.plot(rec, prec, color=COLORS[k], linewidth=2.4, label=f"{NAMES[k]} (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Anomaly Detection — Precision-Recall (held-out TEST)", pad=12)
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
save(fig, "02_ad_pr_comparison.png")

# ---------- 5. 03 score distributions ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
h, f_ = y_ad == 0, y_ad == 1
axes[0].hist(s_sup[h], bins=60, color="#2ecc71", alpha=0.65, label=f"Healthy (n={h.sum():,})")
axes[0].hist(s_sup[f_], bins=60, color="#e74c3c", alpha=0.65, label=f"Anomaly (n={f_.sum():,})")
axes[0].axvline(0.5, color="#f39c12", ls="--", lw=1.5, label="Operating pt 0.5")
axes[0].set_xlabel("Supervised RF-200 P(anomaly)")
axes[0].set_title("Primary detector scores")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
lm = np.log10(np.maximum(s_mah, 1e-9))
axes[1].hist(lm[h], bins=60, color="#2ecc71", alpha=0.65, label="Healthy")
axes[1].hist(lm[f_], bins=60, color="#e74c3c", alpha=0.65, label="Anomaly")
axes[1].axvline(np.log10(art["ad_threshold"]), color="#f39c12", ls="--", lw=1.5,
                label=f"Shipped thr {art['ad_threshold']:.2f}")
axes[1].set_xlabel("Mahalanobis distance (log10)")
axes[1].set_title("Watchdog scores")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
fig.suptitle("AD Score Distributions — Healthy vs Anomaly (TEST)", fontsize=13, y=1.03)
fig.tight_layout()
save(fig, "03_ad_score_distributions.png")

# ---------- 6. 04 outlier-detector benchmark bars ----------
keys = [k for k in ["superv", "mahalanobis", "iforest", "pca", "autoencoder"] if k in scores]
f1s, aucs = [], []
for k in keys:
    aucs.append(metrics["ad"][k]["auc"])
    d = metrics["ad"][k]
    f1s.append(d.get("f1_at_0.5", d.get("f1_at_shipped_thr", d.get("f1_at_trainval95_thr"))))
x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(11, 6.5))
w = 0.36
b1 = ax.bar(x - w / 2, f1s, w, color="#f97316", edgecolor="#30363d", label="F1 (operating point)")
b2 = ax.bar(x + w / 2, aucs, w, color="#58a6ff", edgecolor="#30363d", label="ROC-AUC (threshold-free)")
ax.set_xticks(x)
ax.set_xticklabels([NAMES[k] for k in keys], rotation=14, ha="right", fontsize=9)
ax.set_ylim(0, 1.02)
ax.set_ylabel("Score")
ax.set_title("Outlier / Anomaly Detector Benchmark (held-out TEST)", pad=12)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, color="#e6edf3")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "04_outlier_detector_benchmark.png")

# ---------- 7. Fault classifier: confusion matrices ----------
anom_te = ete[ete.is_anomaly == 1].reset_index(drop=True)
le = art["le"]
y_c = le.transform(anom_te["fault_type"])
yp_c = art["clf"].predict(anom_te[FEATURES_FAULT])
cm = confusion_matrix(y_c, yp_c)
rep = classification_report(y_c, yp_c, target_names=list(le.classes_), output_dict=True, digits=4)
metrics["clf"] = {"accuracy": round(float((y_c == yp_c).mean()), 4),
                  "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
                  "per_class": {c: {"precision": round(float(rep[c]["precision"]), 4),
                                    "recall": round(float(rep[c]["recall"]), 4),
                                    "f1": round(float(rep[c]["f1-score"]), 4),
                                    "support": int(rep[c]["support"])} for c in le.classes_}}
print("CLF acc", metrics["clf"]["accuracy"], "macro-F1", metrics["clf"]["macro_f1"])

labels = [c.replace("_", "\n") for c in le.classes_]
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = cm[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if v > cm.max() * 0.5 else "#0d1117")
plt.colorbar(im, ax=ax, label="Count", shrink=0.82)
ax.set_xlabel("Predicted fault")
ax.set_ylabel("Actual fault")
ax.set_title(f"Fault Classification Confusion Matrix (TEST, RF-300)\n"
             f"Acc {metrics['clf']['accuracy']:.3f} | macro-F1 {metrics['clf']['macro_f1']:.3f}", pad=12)
fig.tight_layout()
save(fig, "05_confusion_matrix_counts.png")

cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cmn, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if cmn[i, j] > 0.5 else "#0d1117")
plt.colorbar(im, ax=ax, label="Recall fraction", shrink=0.82)
ax.set_xlabel("Predicted fault")
ax.set_ylabel("Actual fault")
ax.set_title("Confusion Matrix — Row-Normalized Recall (TEST)", pad=12)
fig.tight_layout()
save(fig, "06_confusion_matrix_normalized.png")

# ---------- 8. 07 per-class P/R/F1 ----------
cls = list(le.classes_)
P = [metrics["clf"]["per_class"][c]["precision"] for c in cls]
R = [metrics["clf"]["per_class"][c]["recall"] for c in cls]
F = [metrics["clf"]["per_class"][c]["f1"] for c in cls]
x = np.arange(len(cls))
fig, ax = plt.subplots(figsize=(13, 6.5))
w = 0.26
ax.bar(x - w, P, w, color="#58a6ff", label="Precision")
ax.bar(x, R, w, color="#2ecc71", label="Recall")
ax.bar(x + w, F, w, color="#f97316", label="F1")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ") for c in cls], rotation=22, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Per-Fault Precision / Recall / F1 (TEST, RF-300)", pad=12)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "07_perclass_prf1.png")

# ---------- 9. RUL scatter + residuals ----------
rul_te = ete[ete.fault_type != "none"].dropna(subset=["RUL"]).reset_index(drop=True)
yt = rul_te["RUL"].values
ypr = art["reg"].predict(rul_te[FEATURES_RUL])
mae = float(mean_absolute_error(yt, ypr))
rmse = float(np.sqrt(mean_squared_error(yt, ypr)))
r2 = float(r2_score(yt, ypr))
metrics["rul"] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4),
                  "n": int(len(yt)),
                  "per_fault_mae": {f: round(float(mean_absolute_error(
                      rul_te[rul_te.fault_type == f]["RUL"],
                      art["reg"].predict(rul_te[rul_te.fault_type == f][FEATURES_RUL]))), 2)
                      for f in sorted(rul_te.fault_type.unique())}}
print(f"RUL MAE {mae:.2f} RMSE {rmse:.2f} R2 {r2:.4f}")

fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(yt, ypr, c="#58a6ff", alpha=0.22, s=10, edgecolors="none")
lim = [0, max(yt.max(), ypr.max()) * 1.05]
ax.plot(lim, lim, "--", color="#e74c3c", lw=2, label="Perfect")
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title("RUL — Predicted vs Actual (TEST, HGB-500)", pad=12)
ax.text(0.05, 0.95, f"MAE = {mae:.2f} cycles\nRMSE = {rmse:.2f}\nR² = {r2:.4f}\nn = {len(yt):,}",
        transform=ax.transAxes, fontsize=12, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#58a6ff", alpha=0.9))
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(lim)
ax.set_ylim(lim)
fig.tight_layout()
save(fig, "08_rul_predicted_vs_actual.png")

resid = ypr - yt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
a1.hist(resid, bins=60, color="#58a6ff", edgecolor="#30363d", alpha=0.85)
a1.axvline(0, color="#e74c3c", ls="--", lw=1.5)
a1.axvline(np.median(resid), color="#f39c12", ls=":", lw=1.5, label=f"median {np.median(resid):+.2f}")
a1.set_xlabel("Residual (predicted − actual, cycles)")
a1.set_ylabel("Count")
a1.set_title("RUL residuals")
a1.legend(fontsize=9)
a1.grid(True, alpha=0.3)
fe = sorted(metrics["rul"]["per_fault_mae"])
a2.barh([f.replace("_", " ") for f in fe],
        [metrics["rul"]["per_fault_mae"][f] for f in fe], color="#f97316", edgecolor="#30363d")
a2.set_xlabel("MAE (cycles)")
a2.set_title("RUL MAE per fault")
a2.grid(True, axis="x", alpha=0.3)
fig.suptitle("RUL Error Analysis (TEST)", fontsize=13, y=1.03)
fig.tight_layout()
save(fig, "09_rul_residuals_hist.png")

# ---------- 10. Feature importances ----------
imp = np.asarray(getattr(art["clf"], "feature_importances_"))
idx = np.argsort(imp)[-20:]
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(20), imp[idx], color="#58a6ff", edgecolor="#30363d")
for b in bars[-5:]:
    b.set_color("#f97316")
ax.set_yticks(range(20))
ax.set_yticklabels([FEATURES_FAULT[i].replace("_", " ") for i in idx], fontsize=9)
ax.set_xlabel("Gini importance (RF-300)")
ax.set_title("Top-20 Features — Fault Classifier (shipped RF-300)", pad=12)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "10_feature_importance_clf.png")
metrics["top_features_clf"] = {FEATURES_FAULT[i]: round(float(imp[i]), 5)
                               for i in idx[::-1][:10]}

# permutation importance for HGB RUL (sampled for speed)
samp = rul_te.sample(min(2000, len(rul_te)), random_state=SEED)
pi = permutation_importance(art["reg"], samp[FEATURES_RUL], samp["RUL"],
                            n_repeats=5, random_state=SEED, n_jobs=-1)
order = np.argsort(pi.importances_mean)[-20:]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(20), pi.importances_mean[order], xerr=pi.importances_std[order],
        color="#2ecc71", ecolor="#8b949e", edgecolor="#30363d")
ax.set_yticks(range(20))
ax.set_yticklabels([FEATURES_RUL[i].replace("_", " ") for i in order], fontsize=9)
ax.set_xlabel("Permutation importance (ΔMAE, cycles)")
ax.set_title("Top-20 Features — RUL Regressor (HGB-500, TEST sample n=2000)", pad=12)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "11_feature_importance_rul.png")
metrics["top_features_rul"] = {FEATURES_RUL[i]: round(float(pi.importances_mean[i]), 4)
                               for i in order[::-1][:10]}

with open(os.path.join(OUT, "eval_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nAll metrics -> {OUT}/eval_metrics.json")
print(json.dumps({k: v for k, v in metrics.items() if k in ('ad', 'clf', 'rul')}, indent=2))
print("EVAL CHARTS COMPLETE")
