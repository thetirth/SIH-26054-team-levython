"""
dashboard.py
============
PHASE 5 & 6 -- Ground Control Station (GCS) Dashboard
DRDO SIH-26054 | MALE UAV Engine Digital Twin

Real-time interactive dashboard providing:
  - Live engine health monitoring with Plotly gauges
  - Fault alerts with severity and SHAP-based XAI explanations
  - Engine efficiency trends over mission timeline
  - Maintenance advisory panel with predictive recommendations
  - Mission-wise health reports
  - Simulation replay capability for 4 operational scenarios

Run with: streamlit run dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time

from features import FEATURES_RAW
from phm_pipeline import load_artifacts, prepare, predict, explain

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="DRDO UAV Engine Digital Twin",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# CONSTANTS
# ====================================================================
FEATURES = [
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct", "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt", "cht", "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc", "injection_timing"
]

# Rotax 912 ULS Limits (Ref: Operator's Manual Section 2)
LIMITS = {
    "cht": {"caution": 135, "limit": 150, "unit": "degC"},
    "egt": {"caution": 800, "limit": 850, "unit": "degC"},
    "oil_pressure": {"min": 22, "max": 72, "unit": "PSI"},
    "oil_temp": {"caution": 130, "limit": 140, "unit": "degC"},
    "vibration": {"caution": 0.30, "limit": 0.50, "unit": "g RMS"},
    "rpm": {"idle": 1400, "cruise": 4800, "max_cont": 5500, "max_to": 5800, "unit": "RPM"},
}

# ====================================================================
# DATA & MODEL LOADING (cached)
# ====================================================================

@st.cache_data
def load_data():
    return pd.read_csv("engine_data.csv")

@st.cache_resource
def load_models():
    return load_artifacts("models")

@st.cache_data
def load_fmea():
    if os.path.exists("fmea_table.csv"):
        return pd.read_csv("fmea_table.csv")
    return None

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def create_gauge(value, title, min_val, max_val, caution=None, limit=None, unit=""):
    """Create a Plotly gauge indicator."""
    steps = []
    bar_color = "#2ecc71"  # green
    
    if caution and limit:
        steps = [
            {"range": [min_val, caution], "color": "#1a1a2e"},
            {"range": [caution, limit], "color": "#f39c12"},
            {"range": [limit, max_val], "color": "#e74c3c"},
        ]
        if value > limit:
            bar_color = "#e74c3c"
        elif value > caution:
            bar_color = "#f39c12"
    else:
        steps = [{"range": [min_val, max_val], "color": "#1a1a2e"}]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"<b>{title}</b><br><span style='font-size:12px'>{unit}</span>",
               "font": {"size": 14, "color": "#e0e0e0"}},
        number={"font": {"size": 28, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#666",
                     "tickfont": {"color": "#999", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "#0d1117",
            "bordercolor": "#333",
            "steps": steps,
            "threshold": {
                "line": {"color": "#e74c3c", "width": 2},
                "thickness": 0.8,
                "value": limit if limit else max_val
            }
        }
    ))
    fig.update_layout(
        height=200, margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e0e0e0"}
    )
    return fig

def run_inference(row_data, art):
    """Run the full PHM inference pipeline on a single data row.

    row_data: mapping with raw CAN/GCS signals (+ cycle/scenario when known).
    art: artifact dict from load_artifacts().
    Returns dict with is_anomaly, ad_score, fault_type, confidence, rul, shap.
    `shap` maps feature -> signed impact (SHAP values when shap is installed,
    otherwise gain-importance magnitudes for the predicted class).
    """
    raw = {f: row_data[f] for f in FEATURES_RAW}
    for extra in ("cycle", "scenario"):
        if extra in row_data:
            raw[extra] = row_data[extra]
    df_eng = prepare(raw)
    res = predict(art, df_eng)[0]
    exp = explain(art, df_eng)
    if exp["method"] == "shap":
        pred_idx = exp["class"]
        shap_dict = {f: 0.0 for f in FEATURES_RAW}
        for f, v in exp["top"]:
            shap_dict[f] = float(v)
        # fill remaining with 0 already; keep full-length mapping below
        full = dict(shap_dict)
    else:
        full = {f: 0.0 for f in FEATURES_RAW}
        for f, v in exp["top"]:
            if f in full:
                full[f] = float(v)
    res["shap"] = full
    res["_xai_method"] = exp["method"]
    return res

def get_maintenance_advisory(result, row):
    """Generate human-readable maintenance advisory from model output."""
    advisories = []
    fault = result["fault_type"]
    rul = result["rul"]
    
    urgency = "ROUTINE"
    if rul < 20:
        urgency = "CRITICAL"
    elif rul < 50:
        urgency = "URGENT"
    elif rul < 100:
        urgency = "SCHEDULED"
    
    fault_actions = {
        "misfire": "Inspect spark plugs and ignition modules. Check fuel filter for contamination.",
        "injector_abnormality": "Clean or replace fuel injectors. Verify ECU fuel mapping tables.",
        "cooling_degradation": "Check coolant level and radiator blockage. Verify thermostat operation.",
        "lubrication_issue": "Replace oil and filter immediately. Inspect oil pump and check for leaks.",
        "sensor_drift": "Recalibrate affected sensors. Check wiring harness for corrosion or EMI.",
        "combustion_instability": "Verify fuel octane rating. Retard ignition timing. Decarbonize combustion chamber.",
        "overheating": "Reduce power setting. Increase airspeed for cooling. Inspect cooling system.",
        "abnormal_vibration": "Inspect propeller balance and engine mounts. Check bearing condition.",
        "none": "No corrective action required. Continue monitoring.",
    }
    
    advisories.append({
        "urgency": urgency,
        "fault": fault.replace("_", " ").title(),
        "rul_cycles": rul,
        "action": fault_actions.get(fault, "Perform general inspection."),
        "confidence": result["confidence"],
    })
    
    # Check parameter limits
    if row.get("cht", 0) > 135:
        advisories.append({"urgency": "WARNING", "fault": "CHT Elevated",
                          "action": f"CHT at {row['cht']:.1f} degC exceeds caution limit (135 degC).",
                          "rul_cycles": None, "confidence": None})
    if row.get("oil_pressure", 100) < 28:
        advisories.append({"urgency": "WARNING", "fault": "Low Oil Pressure",
                          "action": f"Oil pressure at {row['oil_pressure']:.1f} PSI approaching minimum (22 PSI).",
                          "rul_cycles": None, "confidence": None})
    if row.get("vibration", 0) > 0.30:
        advisories.append({"urgency": "WARNING", "fault": "High Vibration",
                          "action": f"Vibration at {row['vibration']:.3f}g exceeds caution threshold (0.30g).",
                          "rul_cycles": None, "confidence": None})
    
    return advisories

# ====================================================================
# CUSTOM CSS
# ====================================================================
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp { background-color: #0d1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #8b949e; font-size: 13px; margin: 0; font-weight: 400; }
    .metric-card h1 { color: #e6edf3; font-size: 32px; margin: 4px 0 0 0; }
    
    .status-normal { color: #2ecc71; font-weight: bold; }
    .status-caution { color: #f39c12; font-weight: bold; }
    .status-critical { color: #e74c3c; font-weight: bold; animation: pulse 1s infinite; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .alert-box {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .alert-critical { background: #2d1b1b; border-color: #e74c3c; color: #f8d7da; }
    .alert-urgent { background: #2d2a1b; border-color: #f39c12; color: #fff3cd; }
    .alert-routine { background: #1b2d1e; border-color: #2ecc71; color: #d4edda; }
    
    .shap-bar-positive { background: #e74c3c; height: 12px; border-radius: 3px; display: inline-block; }
    .shap-bar-negative { background: #3498db; height: 12px; border-radius: 3px; display: inline-block; }
    
    div[data-testid="stSidebar"] { background: #161b22; }
    
    .big-title {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
        text-align: center;
        padding: 8px 0;
        border-bottom: 2px solid #30363d;
        margin-bottom: 16px;
    }
    .sub-title {
        font-size: 14px;
        color: #8b949e;
        text-align: center;
        margin-top: -12px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# SIDEBAR
# ====================================================================
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Select View", [
        "Real-Time Monitoring",
        "Mission Replay",
        "Fault Alerts & XAI",
        "Efficiency Trends",
        "Maintenance Advisory",
        "FMEA Reference",
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### Mission Configuration")
    
    df = load_data()
    engines = sorted(df["engine_id"].unique())
    selected_engine = st.selectbox("Engine", engines)
    
    engine_df = df[df["engine_id"] == selected_engine]
    missions = sorted(engine_df["mission_id"].unique())
    selected_mission = st.selectbox("Mission", missions)
    
    mission_df = engine_df[engine_df["mission_id"] == selected_mission].copy()
    mission_df = mission_df.sort_values("cycle").reset_index(drop=True)
    
    scenario = mission_df.iloc[0]["scenario"]
    _faults = mission_df[mission_df["fault_type"] != "none"]["fault_type"]
    fault = str(_faults.mode().iloc[0]) if len(_faults) else "none"
    st.markdown(f"**Scenario:** `{scenario}`")
    st.markdown(f"**Fault Type:** `{fault}`")
    st.markdown(f"**Cycles:** {len(mission_df)}")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:11px;'>"
        "DRDO SIH-26054<br>MALE UAV Digital Twin<br>"
        "Rotax 912 ULS<br>v2.0</div>",
        unsafe_allow_html=True
    )

# Load models
art = load_models()

# ====================================================================
# HEADER
# ====================================================================
st.markdown(
    "<div class='big-title'>AI-Enabled Digital Twin | MALE UAV Aero Engine</div>"
    "<div class='sub-title'>DRDO SIH-26054 | Rotax 912 ULS | Ground Control Station</div>",
    unsafe_allow_html=True
)

# ====================================================================
# PAGE: REAL-TIME MONITORING
# ====================================================================
if page == "Real-Time Monitoring":
    
    # Cycle slider for the selected mission
    cycle_idx = st.slider("Mission Timeline (Cycle)", 0, len(mission_df)-1,
                          len(mission_df)//2, key="rt_slider")
    row = mission_df.iloc[cycle_idx]
    
    # Run inference
    result = run_inference(row, art)
    
    # Status bar
    if result["is_anomaly"]:
        status_class = "status-critical" if result["rul"] < 30 else "status-caution"
        status_text = "ANOMALY DETECTED" if result["rul"] < 30 else "CAUTION"
    else:
        status_class = "status-normal"
        status_text = "NORMAL"
    
    # Top KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='metric-card'><h3>Status</h3>"
                    f"<h1 class='{status_class}'>{status_text}</h1></div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3>Flight Phase</h3>"
                    f"<h1>{row['flight_phase'].upper()}</h1></div>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3>Altitude</h3>"
                    f"<h1>{row['altitude_m']:.0f} m</h1></div>",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h3>Engine Age</h3>"
                    f"<h1>{row['engine_age_hours']:.0f} h</h1></div>",
                    unsafe_allow_html=True)
    with c5:
        rul_color = "#e74c3c" if result["rul"] < 30 else ("#f39c12" if result["rul"] < 80 else "#2ecc71")
        st.markdown(f"<div class='metric-card'><h3>RUL Prediction</h3>"
                    f"<h1 style='color:{rul_color}'>{result['rul']:.0f} cycles</h1></div>",
                    unsafe_allow_html=True)
    
    st.markdown("")
    
    # Gauges row
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1:
        st.plotly_chart(create_gauge(row["rpm"], "RPM", 0, 6000,
                                     5500, 5800, "RPM"), use_container_width=True)
    with g2:
        st.plotly_chart(create_gauge(row["cht"], "CHT", 0, 180,
                                     135, 150, "degC"), use_container_width=True)
    with g3:
        st.plotly_chart(create_gauge(row["egt"], "EGT", 400, 900,
                                     800, 850, "degC"), use_container_width=True)
    with g4:
        st.plotly_chart(create_gauge(row["oil_pressure"], "Oil Press", 0, 80,
                                     None, None, "PSI"), use_container_width=True)
    with g5:
        st.plotly_chart(create_gauge(row["oil_temp"], "Oil Temp", 0, 160,
                                     130, 140, "degC"), use_container_width=True)
    with g6:
        st.plotly_chart(create_gauge(row["vibration"], "Vibration", 0, 0.6,
                                     0.30, 0.50, "g RMS"), use_container_width=True)
    
    # Detailed sensor readout
    st.markdown("#### Sensor Readout")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Fuel Flow", f"{row['fuel_flow']:.1f} L/h")
    sc1.metric("Throttle", f"{row['throttle_pct']*100:.0f}%")
    sc2.metric("Battery V", f"{row['battery_voltage']:.1f} V")
    sc2.metric("Alt Voltage", f"{row['alternator_voltage']:.1f} V")
    sc3.metric("Battery SOC", f"{row['battery_soc']:.0f}%")
    sc3.metric("Inj. Timing", f"{row['injection_timing']:.1f} deg BTDC")
    sc4.metric("Vib 1x", f"{row['vib_1x']:.4f} g")
    sc4.metric("Vib 2x / 0.5x", f"{row['vib_2x']:.4f} / {row['vib_05x']:.4f} g")

# ====================================================================
# PAGE: MISSION REPLAY (Phase 5)
# ====================================================================
elif page == "Mission Replay":
    
    st.markdown("#### Mission Replay -- Sensor Time Series")
    st.caption(f"Engine {selected_engine} | Mission {selected_mission} | Scenario: {scenario}")
    
    # Multi-select for which sensors to plot
    sensor_options = ["rpm", "cht", "egt", "oil_pressure", "oil_temp",
                      "fuel_flow", "vibration", "battery_voltage", "alternator_voltage"]
    selected_sensors = st.multiselect("Select sensors to plot", sensor_options,
                                       default=["rpm", "cht", "egt", "vibration"])
    
    if selected_sensors:
        for sensor in selected_sensors:
            fig = px.line(mission_df, x="cycle", y=sensor,
                          color_discrete_sequence=["#58a6ff"],
                          title=f"{sensor.upper()} over Mission Timeline")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                height=300,
                margin=dict(t=40, b=30, l=50, r=20),
                xaxis_title="Cycle",
                yaxis_title=sensor,
            )
            
            # Add limit lines
            if sensor in LIMITS:
                lim = LIMITS[sensor]
                if "caution" in lim:
                    fig.add_hline(y=lim["caution"], line_dash="dash",
                                  line_color="#f39c12", annotation_text="Caution")
                if "limit" in lim:
                    fig.add_hline(y=lim["limit"], line_dash="dash",
                                  line_color="#e74c3c", annotation_text="Limit")
                if "min" in lim:
                    fig.add_hline(y=lim["min"], line_dash="dash",
                                  line_color="#e74c3c", annotation_text="Min")
            
            # Highlight fault onset
            fault_rows = mission_df[mission_df["is_anomaly"] == 1]
            if len(fault_rows) > 0:
                fault_start = fault_rows.iloc[0]["cycle"]
                fig.add_vrect(x0=fault_start, x1=mission_df["cycle"].max(),
                              fillcolor="#e74c3c", opacity=0.08,
                              annotation_text="Fault Region")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Degradation severity over time
    if "degradation_severity" in mission_df.columns:
        fig_deg = px.line(mission_df, x="cycle", y="degradation_severity",
                          color_discrete_sequence=["#f97316"],
                          title="Degradation Severity Progression")
        fig_deg.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="#0d1117", height=250)
        st.plotly_chart(fig_deg, use_container_width=True)

# ====================================================================
# PAGE: FAULT ALERTS & XAI
# ====================================================================
elif page == "Fault Alerts & XAI":
    
    st.markdown("#### Fault Diagnosis & Explainable AI (SHAP)")
    
    cycle_idx = st.slider("Select Cycle", 0, len(mission_df)-1,
                          len(mission_df)-10, key="xai_slider")
    row = mission_df.iloc[cycle_idx]
    result = run_inference(row, art)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Diagnosis")
        if result["is_anomaly"]:
            st.error(f"ANOMALY DETECTED: **{result['fault_type'].replace('_',' ').upper()}**")
        else:
            st.success("SYSTEM NORMAL")
        
        st.metric("Fault Type", result["fault_type"].replace("_", " ").title())
        st.metric("Confidence", f"{result['confidence']:.1f}%")
        st.metric("Remaining Useful Life", f"{result['rul']:.0f} cycles")
        st.metric("Ground Truth", row["fault_type"])
    
    with c2:
        st.markdown("##### Why did the AI make this prediction? (SHAP)")
        st.caption(f"Explanation method: `{result.get('_xai_method', 'shap')}`"
                   " (SHAP values when the `shap` package is installed, else model importances).")

        shap_items = sorted(result["shap"].items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        
        shap_df = pd.DataFrame(shap_items, columns=["Feature", "SHAP Value"])
        shap_df["Abs"] = shap_df["SHAP Value"].abs()
        shap_df["Direction"] = shap_df["SHAP Value"].apply(
            lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
        shap_df = shap_df.sort_values("Abs", ascending=True)
        
        fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                          color="Direction",
                          color_discrete_map={"Increases Risk": "#e74c3c",
                                              "Decreases Risk": "#3498db"},
                          title="Feature Impact on Prediction")
        fig_shap.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1117",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ====================================================================
# PAGE: EFFICIENCY TRENDS
# ====================================================================
elif page == "Efficiency Trends":
    
    st.markdown("#### Engine Efficiency & Performance Trends")
    
    # Power and Torque over mission
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(
        x=mission_df["cycle"], y=mission_df["power_kw"],
        name="Power (kW)", line=dict(color="#58a6ff", width=2)))
    fig_power.add_trace(go.Scatter(
        x=mission_df["cycle"], y=mission_df["torque_nm"],
        name="Torque (N.m)", line=dict(color="#f97316", width=2), yaxis="y2"))
    fig_power.update_layout(
        title="Power & Torque over Mission",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117", height=350,
        yaxis=dict(title="Power (kW)"),
        yaxis2=dict(title="Torque (N.m)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_power, use_container_width=True)
    
    # Fuel flow vs engine load
    fig_fuel = px.scatter(mission_df, x="engine_load", y="fuel_flow",
                          color="flight_phase",
                          title="Fuel Flow vs Engine Load (Willans Line)",
                          color_discrete_sequence=px.colors.qualitative.Set2)
    fig_fuel.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="#0d1117", height=350)
    st.plotly_chart(fig_fuel, use_container_width=True)
    
    # Health indices over mission
    health_cols = []
    for col_name in ["thermal_load_pct", "electrical_health_pct", "combustion_quality_pct", "lubrication_health_pct"]:
        if col_name in mission_df.columns:
            health_cols.append(col_name)
    
    if health_cols:
        fig_health = go.Figure()
        colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
        for i, col in enumerate(health_cols):
            fig_health.add_trace(go.Scatter(
                x=mission_df["cycle"], y=mission_df[col],
                name=col.replace("_pct", "").replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2)))
        fig_health.update_layout(
            title="Health Indices Over Mission",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1117", height=350,
            yaxis=dict(title="Health Index (%)", range=[0, 105]),
        )
        st.plotly_chart(fig_health, use_container_width=True)

# ====================================================================
# PAGE: MAINTENANCE ADVISORY
# ====================================================================
elif page == "Maintenance Advisory":
    
    st.markdown("#### Predictive Maintenance Advisory Panel")
    
    # Run inference on the last row of the mission
    last_row = mission_df.iloc[-1]
    result = run_inference(last_row, art)
    advisories = get_maintenance_advisory(result, last_row)
    
    for adv in advisories:
        urgency = adv["urgency"]
        css_class = {
            "CRITICAL": "alert-critical",
            "URGENT": "alert-urgent",
            "WARNING": "alert-urgent",
            "SCHEDULED": "alert-routine",
            "ROUTINE": "alert-routine",
        }.get(urgency, "alert-routine")
        
        rul_text = f" | RUL: {adv['rul_cycles']:.0f} cycles" if adv.get("rul_cycles") else ""
        conf_text = f" | Confidence: {adv['confidence']:.1f}%" if adv.get("confidence") else ""
        
        st.markdown(
            f"<div class='alert-box {css_class}'>"
            f"<strong>[{urgency}]</strong> {adv['fault']}{rul_text}{conf_text}<br>"
            f"<em>Action:</em> {adv['action']}</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("#### Mission Health Summary")
    
    # Summary statistics for the mission
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Max CHT", f"{mission_df['cht'].max():.1f} degC")
    mc2.metric("Max EGT", f"{mission_df['egt'].max():.1f} degC")
    mc3.metric("Min Oil Pressure", f"{mission_df['oil_pressure'].min():.1f} PSI")
    mc4.metric("Max Vibration", f"{mission_df['vibration'].max():.3f} g")
    
    # Anomaly distribution over mission
    anomaly_count = mission_df["is_anomaly"].sum()
    total_count = len(mission_df)
    st.progress(anomaly_count / max(total_count, 1))
    st.caption(f"Anomalous cycles: {anomaly_count} / {total_count} ({100*anomaly_count/max(total_count,1):.1f}%)")

# ====================================================================
# PAGE: FMEA REFERENCE
# ====================================================================
elif page == "FMEA Reference":
    
    st.markdown("#### FMEA Table -- Rotax 912 ULS Aero Piston Engine")
    st.caption("Ref: SAE J1739, MIL-STD-1629A")
    
    fmea_df = load_fmea()
    if fmea_df is not None:
        # Color RPN values
        st.dataframe(fmea_df, use_container_width=True, height=500)
        
        # RPN bar chart
        fig_rpn = px.bar(fmea_df.sort_values("RPN", ascending=True),
                         x="RPN", y="Failure Mode", orientation="h",
                         color="RPN",
                         color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                         title="Risk Priority Number (RPN) by Failure Mode")
        fig_rpn.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="#0d1117", height=400)
        st.plotly_chart(fig_rpn, use_container_width=True)
    else:
        st.warning("FMEA table not found. Run `python fmea_table.py` first.")
