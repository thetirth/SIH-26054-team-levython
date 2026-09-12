"""
edge_benchmark.py
=================
PHASE 7 -- Edge AI & Deployment Benchmark

Benchmarks inference latency and memory footprint of the trained models
to demonstrate suitability for deployment on embedded edge devices
onboard the UAV (e.g., Jetson Nano, Raspberry Pi 4, STM32H7).

Uses the shared serving path (phm_pipeline) so benchmarked latency is
exactly what the dashboard and API serve.

Ref: DRDO SIH-26054 Innovation Areas:
  - Edge AI for UAV applications
  - Lightweight onboard analytics
  - Secure telemetry architecture
"""

import os
import time
import numpy as np
import pandas as pd

from phm_pipeline import load_artifacts, prepare, predict


def get_file_size_kb(path):
    """Get file size in KB."""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024.0
    return 0.0


def bench(fn, n=1000):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 99))


def benchmark():
    print("=" * 70)
    print("PHASE 7: EDGE AI BENCHMARK & DEPLOYMENT ANALYSIS")
    print("DRDO SIH-26054 | MALE UAV Digital Twin")
    print("=" * 70)

    art = load_artifacts("models")
    print(f"\n[Models] AD={art['ad_kind']} | CLF={type(art['clf']).__name__} | "
          f"RUL={type(art['reg']).__name__}")

    df = pd.read_csv("engine_data.csv")
    df_eng = prepare(df.iloc[[100]].to_dict(orient="records")[0])

    N_RUNS = 1000

    print(f"\n--- Feature Engineering (44 physics features) ---")
    raw = df.iloc[100].to_dict()
    avg_fe, p99_fe = bench(lambda: prepare(raw), N_RUNS)
    print(f"  Avg latency  : {avg_fe:.3f} ms")

    print(f"\n--- Anomaly Detection ({art['ad_kind']}) ---")
    from phm_pipeline import ad_scores
    avg_ad, p99_ad = bench(lambda: ad_scores(art, df_eng), N_RUNS)
    print(f"  Avg latency  : {avg_ad:.3f} ms")
    print(f"  P99 latency  : {p99_ad:.3f} ms")

    print(f"\n--- Fault Classification ({type(art['clf']).__name__}) ---")
    from features import FEATURES_FAULT
    Xc = df_eng[FEATURES_FAULT]
    avg_fc, p99_fc = bench(lambda: art["clf"].predict(Xc), N_RUNS)
    print(f"  Avg latency  : {avg_fc:.3f} ms")
    print(f"  P99 latency  : {p99_fc:.3f} ms")

    print(f"\n--- RUL Prediction ({type(art['reg']).__name__}) ---")
    from features import FEATURES_RUL
    Xr = df_eng[FEATURES_RUL]
    avg_rul, p99_rul = bench(lambda: art["reg"].predict(Xr), N_RUNS)
    print(f"  Avg latency  : {avg_rul:.3f} ms")
    print(f"  P99 latency  : {p99_rul:.3f} ms")

    total_avg = avg_fe + avg_ad + avg_fc + avg_rul
    edge_avg = avg_fe + avg_ad
    print(f"\n--- Full PHM Pipeline (features + AD + FC + RUL) ---")
    print(f"  Avg latency  : {total_avg:.3f} ms")
    print(f"  Throughput    : {1000.0/total_avg:.0f} inferences/sec")

    telemetry_rate_hz = 10  # 10 Hz ECU telemetry
    budget_ms = 1000.0 / telemetry_rate_hz  # 100 ms budget
    print(f"\n  ECU Rate      : {telemetry_rate_hz} Hz")
    print(f"  Time Budget   : {budget_ms:.0f} ms per frame (EDGE stage)")
    print(f"\n--- Deployment verdict ---")
    print(f"  EDGE stage (features + Mahalanobis AD): {edge_avg:.1f} ms "
          f"-> {'[PASS]' if edge_avg < budget_ms else '[WARN]'} fits the 10 Hz onboard budget")
    print(f"  GCS stage (+ ET-300 classifier + HGB RUL): {total_avg:.1f} ms, "
          f"runs async at the ground station (advisory cadence, non-blocking)")

    print("\n--- Model File Sizes ---")
    model_files = [
        ("scaler_ad.joblib", "StandardScaler"),
        ("model_ad.joblib", f"AD ({art['ad_kind']})"),
        ("model_fc_xgboost.joblib", "Fault Classifier (compat name)"),
        ("model_rul_xgboost.joblib", "RUL Regressor (compat name)"),
        ("label_encoder_fault.joblib", "Label Encoder"),
    ]
    total_kb = 0.0
    for fname, desc in model_files:
        size_kb = get_file_size_kb(os.path.join("models", fname))
        total_kb += size_kb
        print(f"  {desc:<30}: {size_kb:>10.1f} KB")
    print(f"  {'TOTAL':<30}: {total_kb:>10.1f} KB ({total_kb/1024:.2f} MB)")
    print("\n  Note: the Mahalanobis AD head (scaler + mean/cov, a few KB, "
          "<0.1 ms) is the edge-suitable onboard stage; the tree ensembles "
          "serve at the GCS/cloud layer.")

    print(f"\n{'='*70}")
    print("EDGE vs GROUND CONTROL STATION (GCS) PROCESSING SPLIT")
    print(f"{'='*70}")
    print("""
    +------------------------------------------------------------+
    |                    UAV ONBOARD (EDGE)                      |
    |  - CAN bus data ingestion (can_simulator.py)               |
    |  - Physics features + Mahalanobis anomaly score (KBs)      |
    |  - Lightweight anomaly flag: NORMAL / ANOMALY              |
    |  - OOD abstention: unknown regimes stay with physics twin  |
    |  - Latency budget: < 100 ms per frame                      |
    |  - Hardware: ARM Cortex-M7 / Jetson Nano                   |
    +------------------------------------------------------------+
                           |
                      Telemetry Link
                      (AES-256 encrypted)
                           |
    +------------------------------------------------------------+
    |              GROUND CONTROL STATION (GCS)                  |
    |  - Full fault classification (ExtraTrees-300)              |
    |  - RUL prediction (HistGradientBoosting)                   |
    |  - SHAP / importance explainability analysis               |
    |  - Trend analysis & degradation monitoring                 |
    |  - Dashboard visualization (Streamlit / Next.js GCS)       |
    |  - Maintenance advisory generation                         |
    |  - Mission replay & historical analysis                    |
    |  - Hardware: x86 workstation / cloud VM                    |
    +------------------------------------------------------------+
    """)

    print(f"{'='*70}")
    print("SECURE TELEMETRY ARCHITECTURE (Design)")
    print(f"{'='*70}")
    print("""
    Data-in-Transit:
      - AES-256-GCM encryption for all telemetry payloads
      - Mutual TLS (mTLS) authentication between UAV and GCS
      - Per-session ephemeral keys via ECDH key exchange

    Data-at-Rest:
      - Encrypted local storage on UAV (AES-256-XTS)
      - Tamper-evident logging with HMAC-SHA256 checksums

    Access Control:
      - Role-based access (Pilot, Maintenance, Engineer, Admin)
      - Certificate-based device authentication (X.509)

    Federated Learning (Future Work):
      - Fleet-wide model improvement without centralizing raw data
      - Each UAV trains a local model update; only gradients are shared
      - Privacy-preserving aggregation at GCS
      - Ref: McMahan et al., "Communication-Efficient Learning" (2017)
    """)

    print(f"\n{'='*70}")
    print("EDGE AI BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    benchmark()
