"""Versioned, deployment-safe ML bundle for the MALE engine twin.

The original research artifacts remain in ``models/``. This module creates a
reproducible bundle using the exact sklearn runtime that serves the API, so the
demo cannot silently rely on incompatible serialized dependencies.

v6 recipe (matches ship_max.py winners):
  * Mahalanobis novelty watchdog on RAW signals + OOD abstention cap
  * RandomForest-200 supervised known-fault head (primary detector)
  * RandomForest-300 (balanced) fault classifier on 44 physics-informed features
  * HistGradientBoosting-500 RUL regressor on 44 features + cycle
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features

FEATURES = FEATURES_FAULT  # serving-time classifier inputs (back-compat alias)


def train_bundle(data_path: Path, bundle_path: Path) -> dict:
    data = engineer_features(pd.read_csv(data_path))
    healthy = data[data["is_anomaly"] == 0]
    anomalous = data[data["fault_type"] != "none"].dropna(subset=["RUL"])

    scaler = StandardScaler().fit(healthy[FEATURES_AD])
    Z = scaler.transform(healthy[FEATURES_AD])
    mean, cov_inv = Z.mean(0), np.linalg.pinv(np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1]))
    d = Z - mean
    scores = np.einsum("ij,jk,ik->i", d, cov_inv, d)
    threshold = float(np.percentile(scores, 95))
    # OOD abstention cap: worst fault extremes seen in training. Live-loop
    # points beyond this are outside the dataset envelope -- serving abstains.
    Za = scaler.transform(anomalous[FEATURES_AD])
    da = Za - mean
    ood_cap = float(np.percentile(np.einsum("ij,jk,ik->i", da, cov_inv, da), 99.9))

    labels = LabelEncoder().fit(anomalous["fault_type"])
    classifier = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    classifier.fit(anomalous[FEATURES_FAULT], labels.transform(anomalous["fault_type"]))

    supervisor = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                        class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    supervisor.fit(data[FEATURES_FAULT], (data["is_anomaly"] == 1).astype(int))

    rul = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                                        min_samples_leaf=20, random_state=42)
    rul.fit(anomalous[FEATURES_RUL], anomalous["RUL"])

    bundle = {"version": "deployment-v6-max", "features": FEATURES_FAULT, "features_rul": FEATURES_RUL,
              "features_ad": FEATURES_AD, "scaler": scaler, "anomaly_kind": "mahalanobis",
              "anomaly_model": {"mean": mean, "cov_inv": cov_inv}, "anomaly_threshold": threshold,
              "ood_cap": ood_cap,
              "supervisor": supervisor, "supervisor_features": FEATURES_FAULT, "supervisor_threshold": 0.5,
              "classifier": classifier, "labels": labels, "rul": rul}
    joblib.dump(bundle, bundle_path)
    return bundle


def load_or_train(root: Path) -> dict:
    bundle_path = root / "models" / "deployment_bundle.joblib"
    bundle = joblib.load(bundle_path) if bundle_path.exists() else train_bundle(root / "engine_data.csv", bundle_path)
    # Back-compat: v1 bundles used IsolationForest under "anomaly".
    if "anomaly_kind" not in bundle:
        bundle["anomaly_kind"] = "iforest"
        bundle["features_ad"] = bundle.get("features_ad", FEATURES_AD)
        bundle["features_rul"] = bundle.get("features_rul", bundle.get("features", FEATURES_FAULT))
    return bundle
