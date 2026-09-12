"""
phase1_demo.py
==============
End-to-end Phase 1 demonstration:
  CSV -> CAN framing -> CAN decoding -> Digital Twin state update

Shows a real engine data row going through the full Phase 1 pipeline.
"""

import json
import math
import pandas as pd
from digital_twin_core import DigitalTwinCore
from can_simulator import row_to_frames, frames_to_dict, print_frame_table, SIGNALS

REQUIRED = [
    "engine_id", "mission_id", "cycle", "flight_phase", "scenario",
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct",
    "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt", "cht",
    "oil_temp", "oil_pressure", "vibration",
    "battery_voltage", "alternator_voltage", "battery_soc",
    "injection_timing",
]

print("=" * 70)
print("PHASE 1 -- DIGITAL TWIN CORE END-TO-END DEMO")
print("Rotax 912 ULS | MALE UAV | DRDO SIH-26054")
print("=" * 70)

# 1. Load CSV
df = pd.read_csv("engine_data.csv")
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")
print(f"\n[PASS] CSV loaded: {len(df):,} rows, {len(df.columns)} columns")

# Pick a healthy cruise row for the demo
cruise_rows = df[(df["flight_phase"] == "cruise") & (df["fault_type"] == "none")]
if len(cruise_rows) == 0:
    row = df.iloc[0]
else:
    row = cruise_rows.iloc[len(cruise_rows) // 2]

print(f"[INFO] Demo row: Engine={row['engine_id']}  Mission={row['mission_id']}"
      f"  Cycle={row['cycle']}  Phase={row['flight_phase']}")

# 2. CAN framing
frames = row_to_frames(row)
print(f"\n[PASS] CAN encoding: {len(frames)} frames generated ({len(SIGNALS)} signals defined)")

# 3. CAN decoding
decoded = frames_to_dict(frames)
print(f"[PASS] CAN decoding: {len(decoded)} signals recovered")

# 4. Build full record (CAN-decoded signals + non-CAN context fields)
record = {
    "engine_id":      row["engine_id"],
    "mission_id":     int(row["mission_id"]),
    "cycle":          int(row["cycle"]),
    "flight_phase":   row["flight_phase"],
    "scenario":       row["scenario"],
    "engine_age_hours": float(row["engine_age_hours"]),
    **decoded,   # all CAN-decoded sensor + environmental values
}

# 5. Digital Twin update
twin  = DigitalTwinCore()
state = twin.update(record)
print(f"[PASS] Digital Twin state updated")

# 6. Display
print(f"\n{'='*70}")
print("ENGINE STATE SUMMARY")
print(f"{'='*70}")
print(f"\n  Status          : {state.operating_status}")
print(f"  Warnings        : '{state.active_warnings}'")
print(f"\n  --- Raw Sensors ---")
print(f"  RPM             : {state.rpm:.0f} RPM")
print(f"  CHT             : {state.cht:.1f} degC       [limit: 150]")
print(f"  EGT             : {state.egt:.1f} degC       [limit: 850]")
print(f"  Oil Pressure    : {state.oil_pressure:.1f} PSI        [min: 22]")
print(f"  Oil Temperature : {state.oil_temp:.1f} degC       [max: 140]")
print(f"  Fuel Flow       : {state.fuel_flow:.2f} L/h")
print(f"  Vibration (RMS) : {state.vibration:.4f} g         [caution: 0.30]")
print(f"  Vib 1x / 2x / 0.5x: {state.vib_1x:.4f} / {state.vib_2x:.4f} / {state.vib_05x:.4f} g")
print(f"  Alt Voltage     : {state.alternator_voltage:.2f} V")
print(f"  Bat SOC         : {state.battery_soc:.1f} %")
print(f"  Inj. Timing     : {state.injection_timing:.1f} deg BTDC")
print(f"\n  --- ISA & Performance ---")
print(f"  Altitude        : {state.altitude_m:.0f} m")
print(f"  Air Density (sigma) : {state.sigma:.4f}   [sea-level = 1.0]")
print(f"  Power Est.      : {state.power_kw_est:.1f} kW")
print(f"  Torque Est.     : {state.torque_nm_est:.1f} N.m")
bsfc_str = f"{state.bsfc_est:.0f} g/kW.h" if (state.bsfc_est and math.isfinite(state.bsfc_est)) else "N/A"
print(f"  BSFC Est.       : {bsfc_str}         [ref: 285 g/kW.h]")
print(f"  Otto Efficiency : {state.otto_efficiency_pct:.1f}%        [ideal]")
print(f"\n  --- Health Indices [0-100] ---")
print(f"  Thermal Load    : {state.thermal_load_pct:.1f}%")
print(f"  Electrical Hlth : {state.electrical_health_pct:.1f}%")
print(f"  Combustion Qual.: {state.combustion_quality:.1f}%")
print(f"  Lubrication Hlth: {state.lubrication_health:.1f}%")

print(f"\n{'='*70}")
print("JSON FOR ML / DASHBOARD PIPELINE:")
print(f"{'='*70}")
print(twin.to_json())
