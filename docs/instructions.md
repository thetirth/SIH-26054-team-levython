# DRDO SIH-26054: AI-Enabled Real-Time Digital Twin System for Aero Piston Engines

This repository contains the digital twin and prognostics & health management (PHM) system for MALE UAV aero piston engines (specifically modeled on the BRP-Rotax 912 ULS).

## Quick Start (Windows)

Double-click **`start.bat`** — it launches the FastAPI backend (`http://127.0.0.1:8000`) and the Next.js GCS frontend (`http://localhost:3000`), skipping anything already running. Run **`stop.bat`** to shut the stack down. (Streamlit dashboard separately: `streamlit run dashboard.py` → `http://localhost:8501`.)

## Setup & Dependencies

1. **Python version**: Ensure you have Python 3.9 or newer installed.
2. **Install dependencies**: 
   Run the following command to install the required packages:
   ```bash
   pip install numpy pandas scikit-learn joblib python-can fastapi uvicorn streamlit plotly
   ```
   *(Note: `python-can` is used for simulating the actual CAN bus interface the FADEC would use. `shap`/`xgboost` are optional -- inference falls back gracefully without them.)*

---

## Phase 1: Physics Simulation & Digital Twin Core

Phase 1 focuses on building a physically accurate synthetic dataset and the core digital twin state manager.

### Step 1: Generate the Engine Dataset
Run the physics-informed data generator to create the engine telemetry dataset (`engine_data.csv`).
```bash
python generate_engine_data.py
```
**What it does:** 
Generates 100 missions across 5 engine lifecycles. It uses real thermodynamics (Otto cycle efficiency, ISA atmospheric density ratio) and multi-component vibration frequencies (1×, 2×, 0.5×) to simulate healthy flights and 8 specific FMEA-aligned fault modes (e.g., misfire, cooling degradation, abnormal vibration).

### Step 2: Validate the Digital Twin
Run the validation suite to ensure the data is physically sound and the systems are fully operational.
```bash
python phase1_validation.py
```
**What it does:**
Checks that all generated data falls within strict Rotax 912 ULS operational limits. It also verifies that the CAN bus simulation can encode and decode all 19 telemetry signals without data loss, and successfully updates the `DigitalTwinCore` state machine.

### Step 3: Run the End-to-End Demo
Run the demo to see how a single telemetry row flows through the entire system.
```bash
python phase1_demo.py
```
**What it does:**
Extracts a healthy cruise row from the dataset, encodes it into 32-bit CAN payloads, decodes it back, and feeds it into the `DigitalTwinCore`. It prints the real-time physical state (Power, Torque, BSFC) and derived health indices (Thermal Load, Combustion Quality, Electrical Health).

---

## Phase 2, 3, 4: AI/ML Health Monitoring & RUL Prediction

This phase implements the Machine Learning layer to detect anomalies, classify faults, and predict Remaining Useful Life (RUL) using explainable AI.

### Step 4: Train the ML Models
Run the training script to build the AI pipeline.
```bash
python train_final.py
```
**What it does:**
1. Loads the physics dataset and applies a **stratified group-by-mission** split (train/val/test) to prevent data leakage and guarantee every fault mode appears in every split.
2. Engineers **44 physics-informed features** (`features.py`: Rotax limit margins, vibration signature ratios, timing deviation, ISA density corrections, fault-decoupling residuals).
3. Trains a **Mahalanobis anomaly detector** on healthy data (beats Isolation Forest / Autoencoder on validation F1).
4. Trains an **ExtraTrees-300 Classifier** (balanced) to categorize faults into one of the 8 FMEA modes.
5. Trains a **HistGradientBoosting Regressor** to predict the Remaining Useful Life (RUL) in cycles.
6. Saves all models into the `models/` directory for real-time inference, plus `metrics_final.json` with held-out test scores.
*(Legacy `train_models.py` / `retrain_goated.py` need `xgboost`; the current pipeline is pure `scikit-learn`.)*

### Step 5: Run Real-Time Inference & XAI
Simulate a real-time ECU telemetry stream and run the AI models on a live data packet.
```bash
python inference.py
```
**What it does:**
Pulls a highly degraded engine state, simulates CAN transmission, and runs it through the ML pipeline. It outputs the anomaly status, the fault diagnosis (with confidence %), the predicted RUL, and explains *why* the AI made that diagnosis (SHAP values when the `shap` package is installed, otherwise model importances) — e.g., showing that CHT/oil decoupling drove the "cooling degradation" prediction.

---

## Future Phases (Project Roadmap)

*As we build out the system, new instructions will be appended here.*

- [x] **Phase 1: Digital Twin Core** - Physics simulation & CAN data generation.
- [x] **Phase 2: Health Monitoring System** - Anomaly detection model (Isolation Forest).
- [x] **Phase 3: Fault Detection** - XGBoost classifier + SHAP explainability.
- [x] **Phase 4: RUL Prediction** - XGBoost regressor for time-to-failure.
- [x] **Phase 5 & 6: Real-Time Dashboard** - Visualizing the digital twin in a web-based Ground Control Station (GCS) UI.
- [x] **Phase 7: Edge-Cloud Architecture** - Benchmark script for onboard edge AI simulation.
- [x] **Phase 8: Deliverables** - FMEA Table and Architecture Design document.

---

## 4. Launching the Ground Control Station Dashboard (Phase 5 & 6)

The Streamlit dashboard serves as the Ground Control Station (GCS) interface. It visualizes the live CAN data stream, runs inference, displays the SHAP explainability, and plots engine trends.

```bash
streamlit run dashboard.py
```
**What it does:**
Opens an interactive web application where you can:
- **Monitor Real-Time Data:** View live Plotly gauges for RPM, CHT, EGT, and Vibration.
- **Mission Replay:** Replay past missions and view sensor time series.
- **Fault Alerts & XAI:** See exact model predictions, confidence levels, and the SHAP feature impact chart.
- **Maintenance Advisory:** View rule-based actionable insights generated from the ML predictions.

---

## 5. Running the Edge AI Benchmark & FMEA (Phase 7 & 8)

**FMEA Table Generation:**
```bash
python fmea_table.py
```
*Outputs a standalone `fmea_table.csv` and prints the Risk Priority Number (RPN) matrix.*

**Edge AI Benchmarking:**
```bash
python edge_benchmark.py
```
*Measures per-stage inference latency on the exact serving path (`phm_pipeline.py`): edge stage (features + Mahalanobis AD, ~13 ms, within the 100 ms / 10 Hz budget) and GCS stage (classifier + RUL, async).*
