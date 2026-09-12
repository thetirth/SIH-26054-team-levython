# ML Model Requirements — UAV Engine Digital Twin

## 1. Objective

Develop ML models for the AI-enabled Digital Twin of a MALE UAV piston engine.

The ML system must:
1. Detect abnormal engine behavior.
2. Identify/classify the likely engine fault.
3. Predict Remaining Useful Life (RUL).
4. Support engine health monitoring and mission reliability analysis.
5. Provide outputs that can later be integrated into the Digital Twin dashboard.

The current dataset is synthetic and physics-informed. It is a hackathon prototype dataset, not certified real-world aircraft-engine data.

## 2. Current Dataset

Primary file:

`engine_data.csv`

Dataset:
- 45,169 rows
- 100 missions
- 5 simulated engines
- 20 healthy missions
- 80 faulty missions
- 10 missions for each of 8 fault types
- 31 columns

Fault types:
- `misfire`
- `injector_abnormality`
- `lubrication_issue`
- `cooling_degradation`
- `sensor_drift`
- `combustion_instability`
- `overheating`
- `alternator_failure`

Scenarios:
- `normal`
- `hot_weather`
- `high_altitude`
- `rapid_throttle`

Flight phases:
- `idle`
- `takeoff`
- `climb`
- `cruise`
- `descent`
- `landing`

The dataset has already been validated for mission distribution, failure labels, RUL behavior, fault onset, sensor drift behavior, alternator behavior, duplicates, numeric sanity, and sensor ranges.

## 3. Available Input Features

### Engine/Sensor Features
- `rpm`
- `engine_load`
- `fuel_flow`
- `egt`
- `cht`
- `cht_true`
- `oil_temp`
- `oil_pressure`
- `vibration`
- `battery_voltage`
- `alternator_voltage`
- `battery_soc`
- `injection_timing`

### Operating/Environmental Features
- `flight_phase`
- `scenario`
- `altitude_m`
- `airspeed`
- `ambient_temp`
- `throttle_pct`

### Engine Condition Features
- `engine_age_hours`
- `health_index`
- `degradation_severity`

### Temporal Features
- `cycle`
- `elapsed_time_sec`

Do NOT use these as predictive input features:
- `fault_type`
- `is_anomaly`
- `failure`
- `RUL`
- `fault_onset_cycle`

They are labels/target information or directly reveal future fault information.

## 4. Required ML Components

### A. Anomaly Detection

**Goal:** Detect whether the engine is behaving abnormally.

Output should include:
- `anomaly_status`: Normal / Anomaly
- `anomaly_probability`: 0–1, if supported

Possible approaches:
- Isolation Forest
- One-Class SVM
- Autoencoder
- Statistical baseline
- Supervised classifier if a labeled approach is selected

For a hackathon, a simple baseline plus one stronger model is preferable to many models without clear evaluation.

### B. Fault Classification

**Goal:** Identify the likely fault when abnormal behavior is detected.

Target:
`fault_type`

Classes:
- `misfire`
- `injector_abnormality`
- `lubrication_issue`
- `cooling_degradation`
- `sensor_drift`
- `combustion_instability`
- `overheating`
- `alternator_failure`

Healthy data can be represented separately as `none`.

Output:
- predicted fault
- confidence/probability
- preferably top 2–3 predictions

Suitable starting models:
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Logistic Regression baseline

### C. RUL Prediction

**Goal:** Predict how many cycles remain before simulated failure.

Target:
`RUL`

For faulty missions, RUL decreases to 0. Healthy missions have no assigned failure point and therefore contain missing RUL.

Output:
- predicted RUL in cycles
- preferably confidence/uncertainty

Suitable starting models:
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor

LSTM/GRU should only be considered after a simpler baseline works.

## 5. Time-Series Requirements

The data is sequential. Rows within a mission are ordered by:
- `mission_id`
- `cycle`

Where appropriate, use temporal windows such as:
- last 10 cycles
- last 20 cycles
- last 30 cycles

Useful derived features:
- rolling mean
- rolling standard deviation
- rate of change
- moving minimum/maximum
- sensor trend
- vibration variability
- temperature trend
- oil-pressure trend

Examples:
- CHT trend over the last 20 cycles
- oil-pressure change over the last 10 cycles
- RPM variability over the last 20 cycles

## 6. Train/Validation/Test Split

**Do not randomly split individual rows.**

Rows from the same mission are highly related. Random row-level splitting can cause data leakage and unrealistically high performance.

Preferred:

```text
Training Missions
       ↓
Validation Missions
       ↓
Test Missions
```

Keep complete missions together.

For a stronger generalization test, consider an engine-level split where test engines are not present in training.

## 7. Data Leakage Prevention

Never use future information.

Do not use:
- `fault_type`
- `is_anomaly`
- `failure`
- `RUL`
- `fault_onset_cycle`

Do not calculate features from future sensor values.

For current-cycle RUL prediction, only information available up to the current cycle may be used.

## 8. Preprocessing

The ML pipeline should:
1. Load `engine_data.csv`.
2. Inspect missing values.
3. Encode categorical variables.
4. Scale features when required.
5. Split data by mission.
6. Train models.
7. Evaluate on unseen missions.
8. Save trained models.

Categorical features:
- `flight_phase`
- `scenario`

Tree-based models generally need less scaling. SVM, neural networks, and linear models generally benefit from scaling.

## 9. Evaluation Metrics

### Anomaly Detection
Use:
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where applicable

Recall for abnormal conditions is particularly important.

### Fault Classification
Use:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

Also identify commonly confused fault classes.

### RUL Regression
Use:
- MAE
- RMSE
- R²

MAE should be reported in cycles because it is easy to interpret.

## 10. Explainability

The system should provide some explanation for predictions.

Recommended:
- Feature importance
- SHAP values, if practical
- Top contributing sensor readings

Example:

```text
Predicted Fault: Lubrication Issue

Main indicators:
- Oil pressure decreased
- Oil temperature increased
- Vibration increased
```

This will help the Digital Twin/dashboard explain predictions rather than only displaying a fault name.

## 11. Recommended ML Pipeline

```text
engine_data.csv
       |
       v
Data Preprocessing
       |
       v
Mission-Level Train/Test Split
       |
       +----------------------+
       |                      |
       v                      v
Anomaly Detection      Feature Engineering
       |                      |
       v                      v
Normal / Anomaly       Fault Classification
                              |
                              v
                     Fault + Confidence

       Feature / Time-Series Data
                |
                v
           RUL Prediction
                |
                v
         Remaining Cycles
```

Final outputs for the Digital Twin:

```text
Engine State
     |
     +--> Health Score
     +--> Anomaly Status
     +--> Fault Type
     +--> Fault Confidence
     +--> Predicted RUL
     +--> Important Sensor Indicators
```

## 12. Model Saving

Recommended structure:

```text
models/
├── anomaly_model.pkl
├── fault_classifier.pkl
├── rul_model.pkl
└── scaler.pkl
```

Also save:
- feature list
- preprocessing configuration
- label encoder, if used
- model version

Use the appropriate native format when required by the selected ML framework.

## 13. Required Deliverables From the ML Team

### 1. Training script

```text
train_models.py
```

### 2. Inference script

```text
predict_engine.py
```

### 3. Trained models

```text
models/
```

### 4. Evaluation results

Include:
- anomaly detection metrics
- fault classification metrics
- RUL metrics
- confusion matrix
- important features

### 5. Prediction interface

Example:

```json
{
  "health_index": 72.5,
  "anomaly_status": "Anomaly",
  "anomaly_probability": 0.94,
  "predicted_fault": "cooling_degradation",
  "fault_confidence": 0.89,
  "predicted_rul": 142
}
```

The exact implementation may differ, but the output should be easy for the Digital Twin/dashboard to consume.

## 14. Minimum Acceptable Prototype

### Required
- One anomaly detection/classification model
- One fault classification model
- One RUL regression model
- Mission-level train/test split
- Proper evaluation metrics
- Saved trained models
- Simple inference script

### Preferred
- Time-series/rolling features
- Feature importance
- Confidence scores
- Prediction visualization
- RUL trend graph

### Optional
- LSTM/GRU
- Autoencoder
- SHAP
- Ensemble models
- Uncertainty estimation

Do not sacrifice correct data splitting and validation just to use a more complex model.

## 15. Important Dataset Note

The current dataset is synthetic.

Therefore:
- Accuracy on this dataset does not prove real-world aircraft-engine performance.
- Results should be presented as simulation/prototype results.
- Real sensor data should be able to replace the synthetic CSV later.
- Feature names and the prediction interface should remain stable where possible.

## 16. Final ML Goal

The ML system should answer:

### Is the engine healthy?
`Normal / Anomaly`

### If abnormal, what is wrong?
`Fault Type + Confidence`

### How much useful life remains?
`Predicted RUL in cycles`

### Why did the model make this prediction?
`Important sensor indicators / feature importance`

These outputs will later be integrated into the Digital Twin and dashboard.
