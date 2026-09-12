"""
inference.py
============
Demonstrates real-time inference on a simulated ECU/CAN payload.

Steps:
  1. Grabs an anomalous telemetry row.
  2. Simulates CAN encoding -> transmission -> decoding.
  3. Physics feature engineering (features.py) -- same layer as training.
  4. Anomaly detection (Mahalanobis / IsolationForest / Autoencoder).
  5. Fault classification (ExtraTrees) -> fault type + confidence.
  6. RUL prediction (HistGradientBoosting) -> remaining cycles.
  7. Explainable AI: SHAP when installed, else model importances.
"""

import pandas as pd

from can_simulator import row_to_frames, frames_to_dict
from features import FEATURES_RAW
from phm_pipeline import load_artifacts, prepare, predict, explain

import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 70)
    print("PHASE 2-4: REAL-TIME INFERENCE & EXPLAINABLE AI")
    print("=" * 70)

    # 1. Load data & pick a faulty sample
    df = pd.read_csv("engine_data.csv")
    faulty_rows = df[df["degradation_severity"] > 0.8]
    if len(faulty_rows) == 0:
        print("No heavily degraded rows found in dataset.")
        return
    raw_row = df.iloc[faulty_rows.index[0]]
    print(f"\n[INFO] Simulating ECU CAN feed for Mission {raw_row['mission_id']}, Cycle {raw_row['cycle']}")
    print(f"       Ground Truth: Fault = {raw_row['fault_type']}, RUL = {raw_row['RUL']} cycles")

    # 2. CAN framing (ECU -> GCS)
    frames = row_to_frames(raw_row)
    decoded_dict = frames_to_dict(frames)

    # 3. Reconstruct feature vector. GCS-known fields (engine age from Hobbs
    #    meter, elapsed cycle from mission clock, scenario from mission config)
    #    fall back to the dataset row when absent from the CAN payload.
    feature_dict = {}
    for f in FEATURES_RAW + ["cycle", "scenario"]:
        if f in decoded_dict:
            feature_dict[f] = decoded_dict[f]
        elif f in raw_row:
            feature_dict[f] = raw_row[f]
        else:
            feature_dict[f] = 0.0

    art = load_artifacts("models")
    if art.get("superv") is not None:
        ad_label = f"RF-supervised head (thr={art.get('superv_threshold', 0.5):.2f}) + Mahalanobis watchdog"
    else:
        ad_label = f"{art['ad_kind']} (thr={art['ad_threshold']:.4f})"
    print(f"\n[Models] AD={ad_label} | "
          f"CLF={type(art['clf']).__name__} | RUL={type(art['reg']).__name__}")

    df_eng = prepare(feature_dict)
    res = predict(art, df_eng)[0]

    print("\n--- HEALTH MONITORING SYSTEM ---")
    if art.get("superv") is not None:
        print(f"Anomaly probability = {res['ad_score']:.4f} (operating point = {art.get('superv_threshold', 0.5):.2f})")
    else:
        print(f"Anomaly score = {res['ad_score']:.4f} (threshold = {art['ad_threshold']:.4f})")
    print(">> ANOMALY DETECTED <<" if res["is_anomaly"] else ">> SYSTEM NORMAL <<\n   (Forcing diagnostic pipeline for demonstration...)")

    print("\n--- FAULT DIAGNOSIS ---")
    print(f"Diagnosis : {res['fault_type'].upper()}")
    print(f"Confidence: {res['confidence']:.1f}%")

    print("\n--- PROGNOSTICS ---")
    print(f"Est. Remaining Useful Life (RUL): {res['rul']:.0f} cycles")

    print("\n--- EXPLAINABLE AI (XAI) ---")
    print(f"Why did the AI diagnose '{res['fault_type']}'?")
    exp = explain(art, df_eng)
    print(f"(method: {exp['method']})")
    print("\nTop Contributing Features:")
    for feat, impact in exp["top"][:5]:
        print(f"  * {feat:<18}: {impact:>8.3f}")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
