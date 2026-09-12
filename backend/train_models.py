"""
train_models.py
===============
PHASE 2, 3, 4 -- MALE UAV Engine Digital Twin AI/ML Pipeline

Upgraded to "Top Notch" Hackathon Quality:
  1. Anomaly Detection: Compares Isolation Forest vs. Autoencoder (Neural Network).
     Automatically selects the model with the best F1-score for deployment.
  2. Fault Diagnosis: XGBoost Classifier with tuned hyperparameters.
  3. RUL Prediction: XGBoost Regressor for Remaining Useful Life.

Strict group-by-mission splitting is enforced to prevent data leakage.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# ====================================================================
# CONFIGURATION
# ====================================================================

DATA_PATH = "engine_data.csv"
MODELS_DIR = "models"

# Features available from CAN bus / ECU
FEATURES = [
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct", "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt", "cht", "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc", "injection_timing"
]

RANDOM_SEED = 42

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ====================================================================
# 1. DATA LOADING AND LEAKAGE-FREE SPLITTING
# ====================================================================

def load_and_split_data():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # We must split by mission_id to prevent data leakage.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df['mission_id']))
    
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()
    
    print(f"Dataset split by mission:")
    print(f"  Train: {len(train_df)} rows ({train_df['mission_id'].nunique()} missions)")
    print(f"  Test : {len(test_df)} rows ({test_df['mission_id'].nunique()} missions)")
    
    return train_df, test_df

# ====================================================================
# 2. ANOMALY DETECTION (Isolation Forest vs Autoencoder)
# ====================================================================

def train_anomaly_detector(train_df, test_df):
    print("\n" + "="*70)
    print("PHASE 2: ANOMALY DETECTION (Model Comparison)")
    print("Requirement: Compare at least two approaches (Isolation Forest vs Autoencoder)")
    print("="*70)
    
    # Train ONLY on healthy data (is_anomaly == 0) to learn "normal"
    healthy_train = train_df[train_df["is_anomaly"] == 0]
    
    scaler = StandardScaler()
    X_train_healthy = scaler.fit_transform(healthy_train[FEATURES])
    
    # Test set contains both healthy and anomalous data
    X_test = scaler.transform(test_df[FEATURES])
    y_test = test_df["is_anomaly"].values

    # --- Approach A: Isolation Forest ---
    print(f"\n[Training Approach A: Isolation Forest]")
    iso_forest = IsolationForest(
        n_estimators=150, 
        max_samples='auto', 
        contamination=0.05,  # Tune for higher recall
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    iso_forest.fit(X_train_healthy)
    
    preds_if = iso_forest.predict(X_test)
    y_pred_if = np.where(preds_if == -1, 1, 0)
    f1_if = f1_score(y_test, y_pred_if, pos_label=1)
    print("Isolation Forest Results:")
    print(classification_report(y_test, y_pred_if, target_names=["Normal (0)", "Anomaly (1)"]))

    # --- Approach B: Deep Autoencoder (MLPRegressor) ---
    print(f"\n[Training Approach B: Autoencoder]")
    # Bottleneck architecture: 20 -> 12 -> 6 -> 12 -> 20
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(12, 6, 12),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=RANDOM_SEED
    )
    # Target is the input itself (reconstruction)
    autoencoder.fit(X_train_healthy, X_train_healthy)
    
    # Predict reconstruction on test set
    reconstructions_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.square(X_test - reconstructions_test), axis=1)
    
    # Calculate threshold based on 95th percentile of healthy training errors
    reconstructions_train = autoencoder.predict(X_train_healthy)
    mse_train = np.mean(np.square(X_train_healthy - reconstructions_train), axis=1)
    ae_threshold = np.percentile(mse_train, 95)
    
    y_pred_ae = (mse_test > ae_threshold).astype(int)
    f1_ae = f1_score(y_test, y_pred_ae, pos_label=1)
    print("Autoencoder Results:")
    print(classification_report(y_test, y_pred_ae, target_names=["Normal (0)", "Anomaly (1)"]))

    # --- Model Selection Rationale ---
    print("\n[Rationale for Deployment]")
    print(f"  Isolation Forest F1-Score (Anomaly) : {f1_if:.3f}")
    print(f"  Autoencoder F1-Score (Anomaly)      : {f1_ae:.3f}")
    
    if f1_ae > f1_if:
        print("  -> Autoencoder outperforms Isolation Forest. Deploying Autoencoder.")
        best_model_name = "model_ad_autoencoder.joblib"
        best_model = autoencoder
        # We need to save the threshold as well for inference
        joblib.dump(ae_threshold, os.path.join(MODELS_DIR, "ad_threshold.joblib"))
        joblib.dump("autoencoder", os.path.join(MODELS_DIR, "ad_model_type.joblib"))
    else:
        print("  -> Isolation Forest outperforms Autoencoder. Deploying Isolation Forest.")
        best_model_name = "model_ad_iforest.joblib"
        best_model = iso_forest
        joblib.dump("iforest", os.path.join(MODELS_DIR, "ad_model_type.joblib"))

    # Save scaler and best model
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_ad.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, best_model_name))
    print(f"Saved {best_model_name} and Scaler.")

# ====================================================================
# 3. FAULT CLASSIFICATION (XGBoost)
# ====================================================================

def train_fault_classifier(train_df, test_df):
    print("\n" + "="*70)
    print("PHASE 3: FAULT DIAGNOSIS (XGBoost Classifier)")
    print("="*70)
    
    train_anom = train_df[train_df["is_anomaly"] == 1]
    test_anom  = test_df[test_df["is_anomaly"] == 1]
    
    X_train = train_anom[FEATURES]
    y_train = train_anom["fault_type"]
    X_test = test_anom[FEATURES]
    y_test = test_anom["fault_type"]
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    
    print(f"Training XGBoost Classifier on {len(X_train)} anomalous samples...")
    
    # Tuned hyperparameters for top-tier accuracy
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train_enc)
    
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    print("\nFault Classification Results (Test Set - Anomalous Rows Only):")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_fault.joblib"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "model_fc_xgboost.joblib"))
    print("Saved XGBoost Classifier and Label Encoder.")

# ====================================================================
# 4. REMAINING USEFUL LIFE (RUL) PREDICTION
# ====================================================================

def train_rul_regressor(train_df, test_df):
    print("\n" + "="*70)
    print("PHASE 4: RUL PREDICTION (XGBoost Regressor)")
    print("="*70)
    
    train_rul = train_df[train_df["fault_type"] != "none"].dropna(subset=["RUL"])
    test_rul  = test_df[test_df["fault_type"] != "none"].dropna(subset=["RUL"])
    
    X_train = train_rul[FEATURES]
    y_train = train_rul["RUL"]
    X_test = test_rul[FEATURES]
    y_test = test_rul["RUL"]
    
    print(f"Training XGBoost Regressor on {len(X_train)} samples for RUL prediction...")
    
    # Tuned hyperparameters
    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nRUL Prediction Results (Test Set):")
    print(f"  Mean Absolute Error (MAE) : {mae:.2f} cycles")
    print(f"  Root Mean Sq Error (RMSE) : {rmse:.2f} cycles")
    print(f"  R-squared (R2)            : {r2:.3f}")
    
    joblib.dump(reg, os.path.join(MODELS_DIR, "model_rul_xgboost.joblib"))
    print("Saved XGBoost Regressor.")

# ====================================================================
# MAIN
# ====================================================================

def main():
    train_df, test_df = load_and_split_data()
    train_anomaly_detector(train_df, test_df)
    train_fault_classifier(train_df, test_df)
    train_rul_regressor(train_df, test_df)
    
    print("\n" + "="*70)
    print("PHM AI/ML PIPELINE TRAINING COMPLETE")
    print("======================================================================")

if __name__ == "__main__":
    main()
