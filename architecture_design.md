# Digital Twin Architecture Design (DRDO SIH-26054)

This document outlines the end-to-end architecture of the AI-Enabled Real-Time Digital Twin System for Aero Piston Engines (Rotax 912 ULS), split between the UAV Onboard Edge and the Ground Control Station (GCS).

## High-Level Architecture Diagram

```mermaid
flowchart TD
    %% Define Styles
    classDef edge fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#e5e7eb
    classDef gcs fill:#111827,stroke:#10b981,stroke-width:2px,color:#e5e7eb
    classDef link fill:none,stroke:#f59e0b,stroke-width:2px,stroke-dasharray: 5 5
    
    subgraph Edge["UAV Onboard (Edge Processing)"]
        A[Virtual Engine Physics Model<br/>ISA, Thermodynamics, Vibration] -->|Raw Sensor Data| B[FADEC / ECU Emulator<br/>can_simulator.py]
        B -->|CAN 2.0A Frames<br/>19 Signals, 32-bit payloads| C[Feature Pipeline<br/>State Manager]
        C -->|Normalized Features| D[Edge AI: Anomaly Detection<br/>Deep Autoencoder]
        D -->|Normal / Anomaly Flag| E[Telemetry Transmitter<br/>AES-256-GCM]
    end
    
    E -.->|Encrypted RF Data Link<br/>10 Hz| F
    
    subgraph GCS["Ground Control Station (Cloud/Server)"]
        F[Telemetry Receiver<br/>mTLS Auth] --> G[DigitalTwinCore<br/>Health Indices]
        G --> H{Is Anomaly?}
        H -- Yes --> I[Fault Classifier<br/>XGBoost Multi-class]
        H -- No --> J[System Normal Log]
        
        I --> K[Prognostics<br/>XGBoost RUL Regressor]
        I --> L[Explainable AI<br/>SHAP Explainer]
        
        K --> M[GCS Dashboard UI<br/>Next.js Mission Control + Streamlit]
        L --> M
        G --> M
        
        M --> N[Maintenance Advisory<br/>Actionable Rules]
        M --> O[Mission Data Storage<br/>Historical Replay]
    end

    %% Apply Styles
    class A,B,C,D,E edge
    class F,G,H,I,J,K,L,M,N,O gcs
```

## Component Breakdown

### 1. UAV Onboard (Edge)
- **Virtual Engine (Phase 1):** Simulates the Rotax 912 ULS with first-principles physics (Newton's cooling law, Otto Cycle efficiency, ISA atmosphere). It injects physically realistic degradation curves for 8 FMEA fault modes.
- **FADEC/ECU (Phase 1):** Encodes 19 continuous telemetry signals into standard CAN 2.0A bus frames using 11-bit arbitration IDs.
- **Edge AI (Phase 7):** A highly quantized Deep Autoencoder model evaluates the data locally in under `0.1 ms`. It emits a binary "Anomaly / Normal" flag to save telemetry bandwidth.

### 2. Telemetry Link
- **Security:** AES-256-GCM encryption with Mutual TLS (mTLS).
- **Rate:** 10 Hz telemetry frames.

### 3. Ground Control Station (GCS)
- **Digital Twin Core (Phase 2):** Continuously derives 4 key health indices (Thermal Load, Electrical Health, Combustion Quality, Lubrication Health) using strict rule-based logic anchored to the Rotax manual.
- **Fault Classifier (Phase 3):** An XGBoost classifier identifies exactly which of the 8 FMEA faults is occurring.
- **Prognostics (Phase 4):** An XGBoost regressor estimates the Remaining Useful Life (RUL) in cycles.
- **Explainable AI (XAI):** SHAP values are extracted to provide human-readable causality for the ML predictions.
- **Dashboard (Phase 6):** A Next.js Mission Control (`gcs-app`) presents a Three.js MALE UAV/engine simulation, live engine status, FMEA diagnosis, SHAP drivers, RUL, CAN frames, mission scenarios, replay, and maintenance advisory. The Streamlit dashboard remains available for analytical exploration and model validation.

## Edge vs. Cloud Rationale
By placing the **Anomaly Detection** on the edge, the UAV only needs to transmit full high-frequency diagnostic packets when an anomaly is detected, drastically reducing bandwidth requirements. The heavier, resource-intensive **XGBoost classifiers, RUL regressors, and SHAP explainers** reside on the GCS where computing power is abundant.
