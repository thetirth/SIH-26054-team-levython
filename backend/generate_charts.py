"""
generate_charts.py
==================
DRDO SIH-26054 | MALE UAV Engine Digital Twin
Comprehensive EDA & Model Performance Charts

Generates 12 publication-quality charts into charts/ folder:
  1.  Correlation heatmap (20 sensor features)
  2.  Sensor distribution boxplots (healthy vs faulty)
  3.  Fault separation via PCA (2D projection)
  4.  RUL prediction scatter (predicted vs actual)
  5.  Confusion matrix (9-class fault classification)
  6.  Anomaly detection ROC curves
  7.  Feature importance bar chart (top-20)
  8.  Mission profile sensor traces (4 scenarios)
  9.  Degradation severity progression (8 fault types)
  10. Engine age vs health index (5 engines)
  11. Flight phase distribution
  12. Scenario vs fault heatmap

Run: python generate_charts.py
Output: charts/*.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import joblib

# Attempt sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import GroupShuffleSplit

# ====================================================================
# CONFIG
# ====================================================================
DATA_FILE = "engine_data.csv"
CHARTS_DIR = "charts"
MODEL_DIR = "models"

FEATURES = [
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct", "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt", "cht", "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc", "injection_timing"
]

SENSOR_FEATURES = [
    "rpm", "cht", "egt", "oil_pressure", "oil_temp",
    "fuel_flow", "vibration", "battery_voltage", "alternator_voltage",
    "injection_timing"
]

# Plot style — dark theme matching dashboard
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
    "font.family": "sans-serif",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

DRDO_COLORS = ["#58a6ff", "#f97316", "#2ecc71", "#e74c3c", "#9b59b6",
               "#f39c12", "#1abc9c", "#e91e63", "#3498db"]
FAULT_COLORS = {
    "none": "#2ecc71", "misfire": "#e74c3c", "injector_abnormality": "#f97316",
    "cooling_degradation": "#3498db", "lubrication_issue": "#f39c12",
    "sensor_drift": "#9b59b6", "combustion_instability": "#e91e63",
    "overheating": "#ff6b6b", "abnormal_vibration": "#1abc9c"
}

os.makedirs(CHARTS_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ====================================================================
# LOAD DATA
# ====================================================================
print("=" * 60)
print("DRDO SIH-26054 — EDA Chart Generator")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df):,} rows, {df['mission_id'].nunique()} missions, "
      f"{df['engine_id'].nunique()} engines\n")

# ====================================================================
# 1. CORRELATION HEATMAP
# ====================================================================
print("[1/12] Correlation heatmap...")

fig, ax = plt.subplots(figsize=(14, 11))
corr = df[FEATURES].corr()
cmap = LinearSegmentedColormap.from_list("drdo", ["#3498db", "#0d1117", "#e74c3c"])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
short_names = [f.replace("_", "\n") for f in FEATURES]
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_names, fontsize=8)
# Annotate cells
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        val = corr.values[i, j]
        if abs(val) > 0.3:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(val) > 0.5 else "#8b949e")
plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.82)
ax.set_title("Sensor Feature Correlation Matrix\nDRDO SIH-26054 | Rotax 912 ULS", fontsize=14, pad=16)
fig.tight_layout()
save(fig, "01_correlation_heatmap.png")

# ====================================================================
# 2. SENSOR DISTRIBUTION BOXPLOTS (Healthy vs Faulty)
# ====================================================================
print("[2/12] Sensor distribution boxplots...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for idx, feat in enumerate(SENSOR_FEATURES):
    ax = axes[idx]
    healthy = df[df["is_anomaly"] == 0][feat].values
    faulty = df[df["is_anomaly"] == 1][feat].values
    bp = ax.boxplot(
        [healthy, faulty], tick_labels=["Healthy", "Faulty"],
        patch_artist=True, widths=0.6, showfliers=False,
        medianprops=dict(color="#f39c12", linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e")
    )
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_title(feat.replace("_", " ").title(), fontsize=10, color="#c9d1d9")
    ax.grid(True, alpha=0.3)
fig.suptitle("Sensor Distributions: Healthy vs Faulty States\nDRDO SIH-26054", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "02_sensor_distributions_boxplot.png")

# ====================================================================
# 3. FAULT SEPARATION — PCA 2D Projection
# ====================================================================
print("[3/12] PCA fault separation...")

# Sample for performance
sample_size = min(8000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)
X_sample = df_sample[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 9))
fault_types = df_sample["fault_type"].unique()
for ft in sorted(fault_types):
    mask = df_sample["fault_type"].values == ft
    color = FAULT_COLORS.get(ft, "#8b949e")
    alpha = 0.3 if ft == "none" else 0.7
    size = 8 if ft == "none" else 18
    label = ft.replace("_", " ").title()
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=size,
               alpha=alpha, label=label, edgecolors="none")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Fault Type Separation — PCA 2D Projection\nDRDO SIH-26054 | 20-Feature Sensor Space", fontsize=14, pad=12)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=1.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "03_pca_fault_separation.png")

# ====================================================================
# 4-7: MODEL PERFORMANCE CHARTS
# ====================================================================
# Train/test split (group by mission to avoid leakage)
print("[4/12] Training models for performance charts...")

anomalous = df[df["fault_type"] != "none"].dropna(subset=["RUL"]).copy()
healthy = df[df["is_anomaly"] == 0].copy()

# Group split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(anomalous, groups=anomalous["mission_id"]))
train_data = anomalous.iloc[train_idx]
test_data = anomalous.iloc[test_idx]

X_train = train_data[FEATURES]
X_test = test_data[FEATURES]

# Fault classifier
le = LabelEncoder().fit(anomalous["fault_type"])
y_train_fc = le.transform(train_data["fault_type"])
y_test_fc = le.transform(test_data["fault_type"])

clf = ExtraTreesClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
clf.fit(X_train, y_train_fc)
y_pred_fc = clf.predict(X_test)

# RUL regressor
y_train_rul = train_data["RUL"].values
y_test_rul = test_data["RUL"].values

rul_model = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
rul_model.fit(X_train, y_train_rul)
y_pred_rul = rul_model.predict(X_test)

# Anomaly detection
scaler_ad = StandardScaler().fit(healthy[FEATURES])
X_healthy_scaled = scaler_ad.transform(healthy[FEATURES])

iforest = IsolationForest(n_estimators=180, contamination=0.05, random_state=42, n_jobs=-1)
iforest.fit(X_healthy_scaled)

# Get anomaly scores on combined set for ROC
all_data_for_roc = pd.concat([
    healthy.sample(min(5000, len(healthy)), random_state=42),
    anomalous.sample(min(5000, len(anomalous)), random_state=42)
], ignore_index=True)
X_roc_scaled = scaler_ad.transform(all_data_for_roc[FEATURES])
y_roc_true = all_data_for_roc["is_anomaly"].values
scores_if = -iforest.score_samples(X_roc_scaled)  # negate so higher = more anomalous

print("  Models trained. Generating charts...")

# ====================================================================
# 4. RUL PREDICTION SCATTER
# ====================================================================
print("[4/12] RUL prediction scatter...")

mae = mean_absolute_error(y_test_rul, y_pred_rul)
rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul))
r2 = r2_score(y_test_rul, y_pred_rul)

fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(y_test_rul, y_pred_rul, c="#58a6ff", alpha=0.25, s=10, edgecolors="none")
lims = [0, max(y_test_rul.max(), y_pred_rul.max()) * 1.05]
ax.plot(lims, lims, "--", color="#e74c3c", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
ax.set_title("Remaining Useful Life — Predicted vs Actual\nDRDO SIH-26054 | ExtraTrees Regressor", fontsize=14, pad=12)
textbox = f"MAE = {mae:.1f} cycles\nRMSE = {rmse:.1f} cycles\nR² = {r2:.3f}"
ax.text(0.05, 0.95, textbox, transform=ax.transAxes, fontsize=12,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
        edgecolor="#58a6ff", alpha=0.9), color="#e6edf3")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(lims)
ax.set_ylim(lims)
fig.tight_layout()
save(fig, "04_rul_prediction_scatter.png")

# ====================================================================
# 5. CONFUSION MATRIX
# ====================================================================
print("[5/12] Confusion matrix...")

cm = confusion_matrix(y_test_fc, y_pred_fc)
class_names = le.classes_

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
short_labels = [c.replace("_", "\n") for c in class_names]
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = cm[i, j]
        ax.text(j, i, str(val), ha="center", va="center", fontsize=10,
                color="white" if val > cm.max() * 0.5 else "#0d1117",
                fontweight="bold")
plt.colorbar(im, ax=ax, label="Count", shrink=0.82)
acc = np.trace(cm) / cm.sum() * 100
ax.set_xlabel("Predicted Fault", fontsize=12)
ax.set_ylabel("Actual Fault", fontsize=12)
ax.set_title(f"Fault Classification Confusion Matrix\nAccuracy: {acc:.1f}% | ExtraTrees 220 estimators", fontsize=14, pad=12)
fig.tight_layout()
save(fig, "05_confusion_matrix.png")

# ====================================================================
# 6. ANOMALY DETECTION ROC CURVE
# ====================================================================
print("[6/12] Anomaly detection ROC...")

fpr, tpr, _ = roc_curve(y_roc_true, scores_if)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(fpr, tpr, color="#58a6ff", linewidth=2.5,
        label=f"IsolationForest (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#8b949e", linewidth=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.12, color="#58a6ff")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Anomaly Detection ROC Curve\nDRDO SIH-26054 | IsolationForest", fontsize=14, pad=12)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
fig.tight_layout()
save(fig, "06_anomaly_roc_curve.png")

# ====================================================================
# 7. FEATURE IMPORTANCE BAR CHART
# ====================================================================
print("[7/12] Feature importance bar chart...")

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)
top_n = min(20, len(sorted_idx))
top_idx = sorted_idx[-top_n:]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(top_n), importances[top_idx], color="#58a6ff", edgecolor="#30363d")
# Highlight top 5
for i, bar in enumerate(bars):
    if i >= top_n - 5:
        bar.set_color("#f97316")
ax.set_yticks(range(top_n))
ax.set_yticklabels([FEATURES[i].replace("_", " ").title() for i in top_idx], fontsize=10)
ax.set_xlabel("Gini Importance", fontsize=12)
ax.set_title("Feature Importance — Fault Classification\nDRDO SIH-26054 | ExtraTrees 220 estimators | Top-20 Features",
             fontsize=14, pad=12)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "07_feature_importance.png")

# ====================================================================
# 8. MISSION PROFILE SENSOR TRACES (4 scenarios)
# ====================================================================
print("[8/12] Mission profile sensor traces...")

trace_sensors = ["rpm", "cht", "egt", "oil_pressure", "vibration", "fuel_flow"]
scenario_list = ["normal", "hot_weather", "high_altitude", "rapid_throttle"]
scenario_colors = {"normal": "#2ecc71", "hot_weather": "#e74c3c",
                   "high_altitude": "#3498db", "rapid_throttle": "#f97316"}

fig, axes = plt.subplots(len(trace_sensors), 1, figsize=(16, 18), sharex=True)

# Pick one healthy mission per scenario
for scenario in scenario_list:
    scenario_df = df[(df["scenario"] == scenario) & (df["fault_type"] == "none")]
    if scenario_df.empty:
        scenario_df = df[df["scenario"] == scenario]
    mission = scenario_df["mission_id"].iloc[0]
    m_data = df[df["mission_id"] == mission].sort_values("cycle")
    for idx, sensor in enumerate(trace_sensors):
        axes[idx].plot(m_data["cycle"], m_data[sensor],
                       color=scenario_colors[scenario], alpha=0.85, linewidth=1.2,
                       label=scenario.replace("_", " ").title() if idx == 0 else "")
        if idx == 0:
            axes[idx].legend(loc="upper right", fontsize=9)

for idx, sensor in enumerate(trace_sensors):
    axes[idx].set_ylabel(sensor.replace("_", " ").title(), fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    # Add limit lines for key sensors
    if sensor == "cht":
        axes[idx].axhline(y=135, color="#f39c12", linestyle="--", alpha=0.6, linewidth=1)
        axes[idx].axhline(y=150, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1)
    elif sensor == "egt":
        axes[idx].axhline(y=800, color="#f39c12", linestyle="--", alpha=0.6, linewidth=1)
        axes[idx].axhline(y=850, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1)
    elif sensor == "vibration":
        axes[idx].axhline(y=0.30, color="#f39c12", linestyle="--", alpha=0.6, linewidth=1)

axes[-1].set_xlabel("Mission Cycle", fontsize=12)
fig.suptitle("Mission Profile Sensor Traces — 4 Operational Scenarios\nDRDO SIH-26054 | Rotax 912 ULS | Healthy Missions",
             fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "08_mission_profile_traces.png")

# ====================================================================
# 9. DEGRADATION SEVERITY PROGRESSION
# ====================================================================
print("[9/12] Degradation severity progression...")

fault_types_list = [ft for ft in df["fault_type"].unique() if ft != "none"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for idx, ft in enumerate(sorted(fault_types_list)):
    ax = axes[idx]
    ft_data = df[df["fault_type"] == ft]
    # Plot each mission's degradation curve
    for mission_id in ft_data["mission_id"].unique()[:3]:  # max 3 per fault
        m_data = ft_data[ft_data["mission_id"] == mission_id].sort_values("cycle")
        color = FAULT_COLORS.get(ft, "#8b949e")
        ax.plot(m_data["cycle"], m_data["degradation_severity"],
                color=color, alpha=0.7, linewidth=1.5)
    ax.set_title(ft.replace("_", " ").title(), fontsize=10, color=FAULT_COLORS.get(ft, "#c9d1d9"))
    ax.set_xlabel("Cycle", fontsize=8)
    ax.set_ylabel("Severity", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.15, color="#f39c12", linestyle=":", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)

fig.suptitle("Degradation Severity Progression — Paris-Law Power Curve (γ=1.5)\nDRDO SIH-26054 | 8 FMEA Fault Modes",
             fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "09_degradation_severity.png")

# ====================================================================
# 10. ENGINE AGE vs HEALTH INDEX
# ====================================================================
print("[10/12] Engine age vs health index...")

fig, ax = plt.subplots(figsize=(11, 7))
engine_colors = {"E001": "#58a6ff", "E002": "#f97316", "E003": "#2ecc71",
                 "E004": "#e74c3c", "E005": "#9b59b6"}

for eng_id in sorted(df["engine_id"].unique()):
    eng_data = df[df["engine_id"] == eng_id]
    ax.scatter(eng_data["engine_age_hours"], eng_data["health_index"],
               c=engine_colors.get(eng_id, "#8b949e"), s=5, alpha=0.15,
               label=eng_id, edgecolors="none")

ax.set_xlabel("Engine Age (hours)", fontsize=12)
ax.set_ylabel("Health Index", fontsize=12)
ax.set_title("Engine Age vs Health Index — Fleet Aging Analysis\nDRDO SIH-26054 | 5 Engine Lifecycles | TBO ≈ 1500-2000h",
             fontsize=14, pad=12)
ax.legend(loc="lower left", fontsize=10, markerscale=4)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
fig.tight_layout()
save(fig, "10_engine_age_vs_health.png")

# ====================================================================
# 11. FLIGHT PHASE DISTRIBUTION
# ====================================================================
print("[11/12] Flight phase distribution...")

phase_order = ["idle", "takeoff", "climb", "cruise", "descent", "landing"]
phase_counts = df["flight_phase"].value_counts().reindex(phase_order)
phase_colors = ["#8b949e", "#e74c3c", "#f97316", "#58a6ff", "#3498db", "#2ecc71"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
bars = ax1.bar(range(len(phase_order)), phase_counts.values,
               color=phase_colors, edgecolor="#30363d", width=0.7)
ax1.set_xticks(range(len(phase_order)))
ax1.set_xticklabels([p.title() for p in phase_order], fontsize=10)
ax1.set_ylabel("Number of Samples", fontsize=11)
ax1.set_title("Data Distribution by Flight Phase", fontsize=12)
for bar, val in zip(bars, phase_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{val:,}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")
ax1.grid(True, axis="y", alpha=0.3)

# Pie chart
wedges, texts, autotexts = ax2.pie(
    phase_counts.values, labels=[p.title() for p in phase_order],
    colors=phase_colors, autopct="%1.1f%%", startangle=90,
    textprops={"color": "#e6edf3", "fontsize": 9},
    wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
)
for at in autotexts:
    at.set_color("#0d1117")
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax2.set_title("Phase Proportion", fontsize=12)

fig.suptitle("Flight Phase Distribution\nDRDO SIH-26054 | 6-Phase Mission Envelope",
             fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "11_flight_phase_distribution.png")

# ====================================================================
# 12. SCENARIO vs FAULT HEATMAP
# ====================================================================
print("[12/12] Scenario vs fault heatmap...")

# Cross-tab: for each mission, get its dominant fault and scenario
mission_info = df.groupby("mission_id").agg({
    "scenario": "first",
    "fault_type": lambda x: x[x != "none"].mode().iloc[0] if (x != "none").any() else "none"
}).reset_index()

ct = pd.crosstab(mission_info["scenario"], mission_info["fault_type"])
# Reorder
scenario_order = ["normal", "hot_weather", "high_altitude", "rapid_throttle"]
fault_order = ["none", "misfire", "injector_abnormality", "cooling_degradation",
               "lubrication_issue", "sensor_drift", "combustion_instability",
               "overheating", "abnormal_vibration"]
ct = ct.reindex(index=scenario_order, columns=[c for c in fault_order if c in ct.columns], fill_value=0)

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(ct.columns)))
ax.set_yticks(range(len(ct.index)))
ax.set_xticklabels([c.replace("_", "\n") for c in ct.columns], rotation=45, ha="right", fontsize=9)
ax.set_yticklabels([s.replace("_", " ").title() for s in ct.index], fontsize=10)
for i in range(len(ct.index)):
    for j in range(len(ct.columns)):
        val = ct.values[i, j]
        ax.text(j, i, str(val), ha="center", va="center", fontsize=12,
                color="white" if val > ct.values.max() * 0.4 else "#0d1117",
                fontweight="bold")
plt.colorbar(im, ax=ax, label="Mission Count", shrink=0.8)
ax.set_xlabel("Fault Type", fontsize=12)
ax.set_ylabel("Scenario", fontsize=12)
ax.set_title("Scenario vs Fault Type — Mission Cross-Tabulation\nDRDO SIH-26054 | 100 Missions × 8 Fault Modes",
             fontsize=14, pad=12)
fig.tight_layout()
save(fig, "12_scenario_fault_heatmap.png")

# ====================================================================
# SUMMARY
# ====================================================================
charts = sorted(os.listdir(CHARTS_DIR))
print(f"\n{'=' * 60}")
print(f"COMPLETE — Generated {len(charts)} charts in {CHARTS_DIR}/")
print(f"{'=' * 60}")
for c in charts:
    size_kb = os.path.getsize(os.path.join(CHARTS_DIR, c)) / 1024
    print(f"  {c} ({size_kb:.0f} KB)")

# ====================================================================
# UPDATE eda_report.json
# ====================================================================
import json

eda_report = {
    "dataset": {
        "file": DATA_FILE,
        "rows": len(df),
        "columns": len(df.columns),
        "missions": int(df["mission_id"].nunique()),
        "engines": int(df["engine_id"].nunique()),
        "healthy_missions": 20,
        "faulty_missions": 80,
    },
    "anomaly_stats": {
        "anomalous_rows": int(df["is_anomaly"].sum()),
        "nominal_rows": int((df["is_anomaly"] == 0).sum()),
        "anomaly_pct": round(df["is_anomaly"].mean() * 100, 2),
    },
    "scenario_distribution": df["scenario"].value_counts().to_dict(),
    "fault_distribution": {
        "none": 20, "misfire": 10, "injector_abnormality": 10,
        "cooling_degradation": 10, "lubrication_issue": 10,
        "sensor_drift": 10, "combustion_instability": 10,
        "overheating": 10, "abnormal_vibration": 10
    },
    "sensor_ranges": {
        feat: {"min": round(float(df[feat].min()), 3),
               "max": round(float(df[feat].max()), 3),
               "mean": round(float(df[feat].mean()), 3),
               "std": round(float(df[feat].std()), 3)}
        for feat in FEATURES
    },
    "model_performance": {
        "fault_classifier": {
            "model": "ExtraTreesClassifier (n=220)",
            "accuracy_pct": round(float(acc), 2),
            "split": "GroupShuffleSplit by mission_id (80/20)",
        },
        "rul_regressor": {
            "model": "ExtraTreesRegressor (n=220)",
            "mae_cycles": round(float(mae), 2),
            "rmse_cycles": round(float(rmse), 2),
            "r2": round(float(r2), 4),
        },
        "anomaly_detector": {
            "model": "IsolationForest (n=180, contam=0.05)",
            "roc_auc": round(float(roc_auc), 4),
        },
    },
    "top_features": {FEATURES[i]: round(float(importances[i]), 4)
                     for i in sorted_idx[-10:]},
    "charts_generated": charts,
}

with open("eda_report.json", "w") as f:
    json.dump(eda_report, f, indent=2, default=str)
print(f"\n✓ Updated eda_report.json ({os.path.getsize('eda_report.json')} bytes)")
print("\nDone.")
