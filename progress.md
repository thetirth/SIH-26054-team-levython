# DRDO SIH-26054: Project Progress Tracker

This document tracks the ongoing development of the AI-Enabled Real-Time Digital Twin System for Aero Piston Engines.

---

## ✅ Phase 1: Digital Twin Core & Physics Engine
**Status: COMPLETED**
- Modeled **BRP-Rotax 912 ULS** aero engine thermodynamics using the ICAO Standard Atmosphere (ISA).
- Implemented core engine physics for dynamic mapping of power, torque, cooling effectiveness, and fuel flow against RPM, throttle, and altitude.
- Developed multi-component frequency vibration modeling (1× firing, 2× harmonic, 0.5× sub-harmonic) for FFT-style diagnostics.
- Engineered realistic FADEC **CAN 2.0A Bus communications** (19 telemetry signals mapped into 32-bit automotive payloads).
- Injected 8 specific FMEA fault types (e.g., misfire, cooling degradation, abnormal vibration) showing progressive degradation over engine life.

## ✅ Phase 2: Health Monitoring System
**Status: COMPLETED**
- Designed the continuous state estimator (`DigitalTwinCore`).
- Derived four primary health indices from raw data: **Thermal Load (%)**, **Electrical Health (%)**, **Combustion Quality (%)**, and **Lubrication Health (%)**.
- Trained a two-layer detection stack: a **RandomForest-200 supervised head** (primary detector, deployable signals only) over a **Mahalanobis novelty watchdog** (catches never-seen faults + OOD gate). Held-out test: **F1 0.889, ROC-AUC 0.982** (baseline Isolation Forest/Autoencoder: F1 0.723, AUC 0.893).

## ✅ Phase 3: Fault Diagnosis & Predictive Analytics
**Status: COMPLETED**
- Built a **RandomForest-300 Classifier** (balanced) on **44 physics-informed features** (`features.py`: Rotax limit margins, vibration signature ratios, timing deviation, ISA density corrections, fault-decoupling residuals such as `cht_oil_ratio`) to pinpoint exactly which of the 8 FMEA fault modes an engine is experiencing.
- Utilized a rigorous **stratified Group-by-Mission** train/val/test split to completely eliminate data leakage and guarantee real-world generalization (all 8 fault modes present in every split).
- Held-out test: **accuracy 0.834, macro-F1 0.833** across all 8 faults (baseline: acc 0.818 / macro-F1 0.657 on an easier 7-class test). Beaten candidates (ExtraTrees, HGB, voting, per-group specialists) all lost on validation — full battery in `train_max.py`.
- Incorporated **SHAP (SHapley Additive exPlanations)** with a model-importance fallback to provide Explainable AI (XAI). The system explicitly details *why* a fault was predicted by breaking down sensor impact (e.g., CHT/oil decoupling drove the cooling alert).

## ✅ Phase 4: Remaining Useful Life (RUL) Prediction
**Status: COMPLETED**
- Developed a **HistGradientBoosting Regressor** (44 physics features + elapsed cycle) to predict the exact number of flight cycles remaining before total engine failure.
- Validated the model on fully unseen flight missions: **MAE 6.6 cycles, R² 0.940** (baseline: MAE 10.7, R² 0.878).

## ✅ Phase 5: Simulation & Replay Capability
**Status: COMPLETED**
- Stored completed runs as CSV dataset for comprehensive playback.
- Simulated dynamic mission profiles (Endurance, High Altitude, Hot Weather, Rapid Throttle) directly influencing physics metrics.
- Built interactive historical replay views into the dashboard.

## ✅ Phase 6: Visualization Dashboard (GCS UI)
**Status: COMPLETED**
- Developed a complete **Streamlit Ground Control Station Dashboard** (`dashboard.py`).
- Integrated dynamic Plotly gauges for real-time monitoring of critical sensors.
- Implemented **Maintenance Advisory** panels to translate ML predictions into actionable tasks based on RUL urgency.
- Added live, interactive SHAP value bar charts to visualize model explainability dynamically.
- Added a polished **Next.js Mission Control** (`gcs-app`) with a live Three.js scene: textured **MQ-9 Reaper airframe** (`public/mq9`, spinning pusher prop, fault glow on real part groups, nav strobes), real Rotax 912 engine shell view, scenario controls, health and fault state, CAN framing, XAI drivers, subsystem health, mission replay, and maintenance advisory for the jury-facing demonstration.

## ✅ Phase 7: Edge AI & Deployment Considerations
**Status: COMPLETED**
- Built an edge deployment benchmarking script (`edge_benchmark.py`, runs on the exact serving path via `phm_pipeline.py`).
- Verified the **edge stage (physics features + Mahalanobis AD, a few KB) runs in ~14 ms**, well within the 100 ms budget for a 10 Hz ECU; the full pipeline (RF-300 classifier + HGB RUL) serves in ~66 ms asynchronously at the GCS.
- Serving uses **OOD abstention**: inputs beyond the worst training-fault extremes are declined (physical twin stays authoritative) instead of emitting an untrustworthy diagnosis.

## ✅ Phase 8: Deliverables Package
**Status: COMPLETED**
- Generated a formal **Failure Mode and Effects Analysis (FMEA)** table (`fmea_table.py`) with computed RPN scores.
- Documented the exact **Edge vs. Cloud architecture split** and secure mTLS/AES-256 telemetry links in `architecture_design.md`.
- All code is modular, fully validated, and ready for hackathon demonstration.
