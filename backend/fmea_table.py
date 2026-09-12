"""
fmea_table.py
=============
PHASE 3 DELIVERABLE -- FMEA Table for DRDO SIH-26054

Failure Mode and Effects Analysis (FMEA) for Rotax 912 ULS Aero Piston Engine.
Generates a standalone FMEA reference document as CSV and prints a formatted table.

Ref: SAE J1739 (FMEA Standard)
     MIL-STD-1629A (Military Standard for FMEA)
     BRP-Rotax 912 ULS Operator's Manual, Section 2
"""

import pandas as pd

FMEA_DATA = [
    {
        "Fault ID": "F-001",
        "Failure Mode": "Misfire Conditions",
        "Potential Cause": "Fouled spark plug, ignition module degradation, fuel quality",
        "Affected Subsystem": "Ignition / Combustion",
        "Effect on Engine": "Reduced power output, uneven running, increased EGT scatter",
        "Detection Method": "EGT deviation between cylinders, RPM fluctuation, vib_0.5x spike",
        "Severity (1-10)": 7,
        "Occurrence (1-10)": 5,
        "Detection (1-10)": 3,
        "RPN": 105,
        "Recommended Action": "Inspect spark plugs, verify ignition timing, check fuel filter",
        "Digital Twin Indicator": "combustion_quality < 50%, vib_05x elevated"
    },
    {
        "Fault ID": "F-002",
        "Failure Mode": "Injector Abnormality",
        "Potential Cause": "Clogged injector nozzle, fuel contamination, electrical fault",
        "Affected Subsystem": "Fuel Injection",
        "Effect on Engine": "Lean/rich mixture, power loss, elevated EGT on affected cylinder",
        "Detection Method": "Fuel flow deviation from expected curve, injection timing drift",
        "Severity (1-10)": 6,
        "Occurrence (1-10)": 4,
        "Detection (1-10)": 4,
        "RPN": 96,
        "Recommended Action": "Clean/replace injectors, inspect fuel lines, verify ECU mapping",
        "Digital Twin Indicator": "fuel_flow anomaly, injection_timing deviation > 5 deg"
    },
    {
        "Fault ID": "F-003",
        "Failure Mode": "Cooling Degradation",
        "Potential Cause": "Coolant leak, thermostat failure, blocked radiator fins, airflow restriction",
        "Affected Subsystem": "Cooling System",
        "Effect on Engine": "Progressive CHT rise, potential thermal seizure if unchecked",
        "Detection Method": "CHT trending above normal band, CHT rate-of-change positive at steady-state",
        "Severity (1-10)": 8,
        "Occurrence (1-10)": 4,
        "Detection (1-10)": 2,
        "RPN": 64,
        "Recommended Action": "Check coolant level, inspect radiator, verify thermostat operation",
        "Digital Twin Indicator": "thermal_load_pct > 30%, CHT > 135 degC at cruise"
    },
    {
        "Fault ID": "F-004",
        "Failure Mode": "Lubrication Issue",
        "Potential Cause": "Oil pump wear, oil filter blockage, oil degradation, external leak",
        "Affected Subsystem": "Lubrication System",
        "Effect on Engine": "Accelerated bearing wear, potential seizure, elevated oil temperature",
        "Detection Method": "Oil pressure below minimum, oil temperature above normal, vibration increase",
        "Severity (1-10)": 9,
        "Occurrence (1-10)": 3,
        "Detection (1-10)": 2,
        "RPN": 54,
        "Recommended Action": "Replace oil and filter, inspect pump, check for external leaks",
        "Digital Twin Indicator": "lubrication_health < 60%, oil_pressure < 28 PSI"
    },
    {
        "Fault ID": "F-005",
        "Failure Mode": "Sensor Drift / Failure",
        "Potential Cause": "Thermocouple aging, wiring corrosion, ADC degradation, EMI",
        "Affected Subsystem": "Instrumentation",
        "Effect on Engine": "Incorrect readings leading to false alarms or missed faults",
        "Detection Method": "Cross-correlation check between redundant sensors, rate-of-change limits",
        "Severity (1-10)": 5,
        "Occurrence (1-10)": 6,
        "Detection (1-10)": 5,
        "RPN": 150,
        "Recommended Action": "Calibrate sensors, replace degraded thermocouples, check wiring harness",
        "Digital Twin Indicator": "Sensor residual > 3-sigma from physics model prediction"
    },
    {
        "Fault ID": "F-006",
        "Failure Mode": "Combustion Instability (Knock)",
        "Potential Cause": "Low octane fuel, excessive carbon deposits, incorrect ignition advance",
        "Affected Subsystem": "Combustion Chamber",
        "Effect on Engine": "Detonation/pre-ignition, piston damage, power loss",
        "Detection Method": "EGT spike pattern, vibration 2x harmonic increase, combustion quality drop",
        "Severity (1-10)": 9,
        "Occurrence (1-10)": 3,
        "Detection (1-10)": 3,
        "RPN": 81,
        "Recommended Action": "Verify fuel quality, retard ignition timing, decarbonize combustion chamber",
        "Digital Twin Indicator": "combustion_quality < 40%, vib_2x elevated, EGT spike"
    },
    {
        "Fault ID": "F-007",
        "Failure Mode": "Overheating Trend",
        "Potential Cause": "Sustained high power operation, cooling system partial failure, hot ambient",
        "Affected Subsystem": "Thermal Management",
        "Effect on Engine": "Material fatigue, gasket failure, lubricant breakdown",
        "Detection Method": "CHT and oil temp both trending upward, thermal load index rising",
        "Severity (1-10)": 8,
        "Occurrence (1-10)": 5,
        "Detection (1-10)": 2,
        "RPN": 80,
        "Recommended Action": "Reduce power setting, increase airspeed for cooling, abort mission if critical",
        "Digital Twin Indicator": "thermal_load_pct > 50%, CHT > 140 degC, oil_temp > 120 degC"
    },
    {
        "Fault ID": "F-008",
        "Failure Mode": "Abnormal Vibration Pattern",
        "Potential Cause": "Propeller imbalance, bearing wear, crankshaft fatigue, mount loosening",
        "Affected Subsystem": "Mechanical / Structural",
        "Effect on Engine": "Accelerated fatigue, mount failure, secondary damage to accessories",
        "Detection Method": "Vibration RMS increase, vib_1x or vib_2x spectral shift, broadband energy rise",
        "Severity (1-10)": 8,
        "Occurrence (1-10)": 3,
        "Detection (1-10)": 2,
        "RPN": 48,
        "Recommended Action": "Inspect propeller balance, check engine mounts, bearing condition inspection",
        "Digital Twin Indicator": "vibration > 0.3g RMS, vib_1x or vib_2x > baseline + 2-sigma"
    },
]

def generate_fmea():
    df = pd.DataFrame(FMEA_DATA)
    # Sort by RPN descending (highest risk first)
    df = df.sort_values("RPN", ascending=False).reset_index(drop=True)
    
    print("=" * 80)
    print("FMEA TABLE -- Rotax 912 ULS Aero Piston Engine")
    print("DRDO SIH-26054 | MALE UAV Digital Twin")
    print("Ref: SAE J1739, MIL-STD-1629A")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n[{row['Fault ID']}] {row['Failure Mode']}  (RPN={row['RPN']})")
        print(f"  Cause     : {row['Potential Cause']}")
        print(f"  Effect    : {row['Effect on Engine']}")
        print(f"  Detection : {row['Detection Method']}")
        print(f"  Severity={row['Severity (1-10)']}  Occurrence={row['Occurrence (1-10)']}  Detection={row['Detection (1-10)']}")
        print(f"  Action    : {row['Recommended Action']}")
        print(f"  DT Signal : {row['Digital Twin Indicator']}")
    
    # Save to CSV
    csv_path = "fmea_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFMEA table saved to {csv_path}")
    return df

if __name__ == "__main__":
    generate_fmea()
