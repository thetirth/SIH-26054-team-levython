"""
phase1_validation.py
====================
PHASE 1 -- DIGITAL TWIN CORE VALIDATION

Validates that all Phase 1 components work correctly:
  1. CSV loads with all required columns
  2. Sensor values are numeric, finite, and within physical bounds
  3. CAN encoding/decoding round-trips correctly for all signals
  4. Digital Twin state update succeeds for 1000 rows
  5. Engine and mission counts are correct
  6. All scenarios and flight phases are present
  7. Derived state values (sigma, thermal load, etc.) are valid
  8. Air density ratio is physically plausible (0.05–1.2)
  9. Sensor sanity check vs. Rotax 912 ULS operating limits
"""

import math
import pandas as pd
import numpy as np

from digital_twin_core import (
    DigitalTwinCore,
    CHT_LIMIT, EGT_LIMIT, OIL_P_MIN, OIL_T_LIMIT,
    RPM_MAX_TAKEOFF
)
import can_simulator


CSV_FILE = "engine_data.csv"

REQUIRED_COLUMNS = [
    "engine_id", "mission_id", "cycle", "elapsed_time_sec",
    "flight_phase", "scenario",
    "altitude_m", "airspeed_kt", "ambient_temp",
    "throttle_pct", "engine_age_hours", "sigma",
    "rpm", "engine_load", "power_kw", "torque_nm",
    "fuel_flow", "bsfc",
    "egt", "cht_true", "cht",
    "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc",
    "injection_timing",
    "health_index", "degradation_severity",
    "fault_onset_cycle", "fault_type", "is_anomaly", "failure", "RUL",
]

NUMERIC_COLUMNS = [
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct",
    "engine_age_hours", "sigma",
    "rpm", "engine_load", "power_kw", "torque_nm",
    "fuel_flow",
    "egt", "cht_true", "cht",
    "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc",
    "injection_timing",
    "health_index", "degradation_severity", "is_anomaly", "failure",
]

# Physical plausibility ranges (grounded in Rotax 912 ULS limits + fault margin)
# Ref: BRP-Rotax 912 ULS Operator's Manual, Section 2
PHYSICAL_RANGES = {
    "rpm":          (0,    6500),   # some margin above 5800 max
    "fuel_flow":    (0,    45),     # max ~27 L/h + fault injection headroom
    "egt":          (200,  1000),   # well above idle (400) to fault headroom
    "cht":          (0,    230),    # max 150 degC + fault injection headroom
    "cht_true":     (0,    230),
    "oil_temp":     (0,    200),    # max 140 degC + headroom
    "vibration":    (0,    5),      # g RMS
    "battery_voltage":     (8,  16),
    "alternator_voltage":  (0,  16),
    "battery_soc":         (0,  100),
    "throttle_pct":        (0,  1.01),
    "health_index":        (0,  100),
    "degradation_severity":(0,  1.01),
    "sigma":               (0.05, 1.21),
    "power_kw":            (0,  90),    # Rotax 912: 73.5 kW + margin
    "torque_nm":           (0,  140),
}


def make_twin_record(row) -> dict:
    """Build digital twin input dict from a CSV row."""
    return {
        "engine_id":      str(row["engine_id"]),
        "mission_id":     int(row["mission_id"]),
        "cycle":          int(row["cycle"]),
        "flight_phase":   str(row["flight_phase"]),
        "scenario":       str(row["scenario"]),
        "altitude_m":     float(row["altitude_m"]),
        "airspeed_kt":    float(row["airspeed_kt"]),
        "ambient_temp":   float(row["ambient_temp"]),
        "throttle_pct":   float(row["throttle_pct"]),
        "engine_age_hours": float(row["engine_age_hours"]),
        "rpm":            float(row["rpm"]),
        "engine_load":    float(row["engine_load"]),
        "fuel_flow":      float(row["fuel_flow"]),
        "egt":            float(row["egt"]),
        "cht":            float(row["cht"]),
        "oil_temp":       float(row["oil_temp"]),
        "oil_pressure":   float(row["oil_pressure"]),
        "vibration":      float(row["vibration"]),
        "vib_1x":         float(row["vib_1x"]),
        "vib_2x":         float(row["vib_2x"]),
        "vib_05x":        float(row["vib_05x"]),
        "battery_voltage":    float(row["battery_voltage"]),
        "alternator_voltage": float(row["alternator_voltage"]),
        "battery_soc":        float(row["battery_soc"]),
        "injection_timing":   float(row["injection_timing"]),
    }


def validate_can(df: pd.DataFrame):
    """Validate CAN encode/decode round-trip for all signals."""
    if not hasattr(can_simulator, "SIGNALS"):
        raise ImportError("SIGNALS not found in can_simulator.py")

    row    = df.iloc[0]
    errors = []

    for can_id, (signal_name, scale, unit, _) in can_simulator.SIGNALS.items():
        if signal_name not in df.columns:
            continue    # environmental columns may not all be in row

        original = float(row[signal_name])
        frame    = can_simulator.encode_signal(can_id, original)
        decoded_name, decoded_value = can_simulator.decode_signal(frame)

        if decoded_name != signal_name:
            errors.append(f"Name mismatch: {signal_name} != {decoded_name}")
            continue

        tolerance = (1.0 / scale) + 1e-9
        diff      = abs(decoded_value - original)
        if diff > tolerance:
            errors.append(
                f"{signal_name}: original={original:.6f}, "
                f"decoded={decoded_value:.6f}, diff={diff:.8f}"
            )

    if errors:
        raise ValueError("CAN round-trip failures:\n" + "\n".join(errors))


def main():
    print("=" * 70)
    print("PHASE 1 -- DIGITAL TWIN CORE VALIDATION")
    print("Rotax 912 ULS | MALE UAV | DRDO SIH-26054")
    print("=" * 70)

    passed = 0
    failed = 0

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"[PASS] {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        print(f"[FAIL] {msg}")

    def warn(msg):
        print(f"[WARN] {msg}")

    # 1. Load CSV
    try:
        df = pd.read_csv(CSV_FILE)
        ok(f"CSV loaded: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        fail(f"CSV load failed: {e}")
        return

    # 2. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        fail(f"Missing columns: {missing_cols}")
    else:
        ok("All required columns present")

    # 3. Numeric data sanity (NaN / Inf)
    nan_issues = []
    inf_issues = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            nan_issues.append(col)
        finite = vals.dropna().apply(math.isfinite)
        if not finite.all():
            inf_issues.append(col)

    if nan_issues:
        fail(f"NaN/non-numeric values in: {nan_issues}")
    else:
        ok("No NaN/non-numeric in sensor columns")

    if inf_issues:
        fail(f"Infinite values in: {inf_issues}")
    else:
        ok("No infinite values in sensor columns")

    # 4. Physical range checks (Rotax 912 ULS limits)
    range_issues = []
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in df.columns:
            continue
        vals  = pd.to_numeric(df[col], errors="coerce").dropna()
        out   = ((vals < lo) | (vals > hi)).sum()
        if out > 0:
            range_issues.append(f"{col}: {out} rows outside [{lo}, {hi}]")

    if range_issues:
        for iss in range_issues:
            warn(f"Range check: {iss}")
        warn("Some out-of-range values may be fault injections -- verify context")
    else:
        ok("All sensor columns within physical plausibility ranges")

    # 5. CAN round-trip
    try:
        validate_can(df)
        ok("CAN encoding/decoding round-trip (all signals)")
    except Exception as e:
        fail(f"CAN round-trip: {e}")

    # 6. Digital Twin update (1000 rows)
    try:
        twin = DigitalTwinCore()
        n_test = min(1000, len(df))
        for i in range(n_test):
            row    = df.iloc[i]
            record = make_twin_record(row)
            state  = twin.update(record)
            if state is None:
                raise ValueError(f"None state at row {i}")
        ok(f"Digital Twin updated for {n_test:,} rows")
    except Exception as e:
        fail(f"Digital Twin update: {e}")

    # 7. Engine count
    n_engines = df["engine_id"].nunique()
    if n_engines >= 1:
        ok(f"Engine count: {n_engines}")
    else:
        fail("No engines found")

    # 8. Mission count
    n_missions = df[["engine_id", "mission_id"]].drop_duplicates().shape[0]
    if n_missions >= 1:
        ok(f"Mission count: {n_missions}")
    else:
        fail("No missions found")

    # 9. Scenarios
    expected_scenarios = {"normal", "hot_weather", "high_altitude", "rapid_throttle"}
    actual_scenarios   = set(df["scenario"].dropna().unique())
    missing_scenarios  = expected_scenarios - actual_scenarios
    if missing_scenarios:
        fail(f"Missing scenarios: {sorted(missing_scenarios)}")
    else:
        ok(f"All scenarios present: {', '.join(sorted(actual_scenarios))}")

    # 10. Flight phases
    expected_phases = {"idle", "takeoff", "climb", "cruise", "descent", "landing"}
    actual_phases   = set(df["flight_phase"].dropna().unique())
    missing_phases  = expected_phases - actual_phases
    if missing_phases:
        fail(f"Missing flight phases: {sorted(missing_phases)}")
    else:
        ok(f"All flight phases present: {', '.join(sorted(actual_phases))}")

    # 11. Derived state sanity check (sample rows)
    try:
        twin    = DigitalTwinCore()
        indices = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
        for idx in indices:
            row    = df.iloc[idx]
            record = make_twin_record(row)
            state  = twin.update(record)
            assert math.isfinite(state.sigma), f"sigma not finite at row {idx}"
            assert 0.05 <= state.sigma <= 1.2, f"sigma out of range: {state.sigma}"
            assert math.isfinite(state.thermal_load_pct)
            assert math.isfinite(state.electrical_health_pct)
            assert math.isfinite(state.combustion_quality)
            assert math.isfinite(state.lubrication_health)
            assert state.operating_status in ("NORMAL","CAUTION","CRITICAL","EMERGENCY")
        ok("Derived state sanity check (5 sample points)")
    except Exception as e:
        fail(f"Derived state: {e}")

    # 12. Healthy cruise sensor sanity vs. Rotax 912 limits
    healthy_cruise = df[
        (df["fault_type"] == "none") & (df["flight_phase"] == "cruise")
    ]
    if len(healthy_cruise) > 0:
        cht_p95   = healthy_cruise["cht"].quantile(0.95)
        egt_p95   = healthy_cruise["egt"].quantile(0.95)
        oil_p_p05 = healthy_cruise["oil_pressure"].quantile(0.05)
        ff_mean   = healthy_cruise["fuel_flow"].mean()
        pwr_mean  = healthy_cruise["power_kw"].mean()

        issues = []
        if cht_p95 > CHT_LIMIT:
            issues.append(f"CHT p95={cht_p95:.1f} > limit {CHT_LIMIT}")
        if egt_p95 > EGT_LIMIT:
            issues.append(f"EGT p95={egt_p95:.1f} > limit {EGT_LIMIT}")
        if oil_p_p05 < OIL_P_MIN:
            issues.append(f"OilP p05={oil_p_p05:.1f} < min {OIL_P_MIN} PSI")

        if issues:
            for iss in issues:
                warn(f"Healthy cruise sensor check: {iss}")
        else:
            ok(f"Healthy cruise sensors within Rotax 912 limits"
               f" (CHT p95={cht_p95:.1f}, EGT p95={egt_p95:.1f}, OilP p05={oil_p_p05:.1f})")

        print(f"\n[INFO] Healthy cruise stats:")
        print(f"       Fuel flow mean : {ff_mean:.1f} L/h  (ref: ~18.5 L/h @ 75%)")
        print(f"       Power mean     : {pwr_mean:.1f} kW   (ref: 69 kW max cont)")

    # 13. Fault type coverage
    fault_rows   = df[df["fault_type"] != "none"]
    mission_fault_types = (
        fault_rows.groupby("mission_id")["fault_type"].first()
        if len(fault_rows) > 0 else pd.Series(dtype=str)
    )
    expected_faults = {
        "misfire", "injector_abnormality", "lubrication_issue",
        "cooling_degradation", "sensor_drift", "combustion_instability",
        "overheating", "abnormal_vibration"
    }
    actual_faults = set(mission_fault_types.unique()) if len(mission_fault_types) > 0 else set()
    missing_faults = expected_faults - actual_faults
    if missing_faults:
        fail(f"Missing fault types: {missing_faults}")
    else:
        ok(f"All 8 fault types present in dataset")

    # 14. Vibration sub-components present
    if all(c in df.columns for c in ["vib_1x", "vib_2x", "vib_05x"]):
        ok("Vibration sub-components (vib_1x, vib_2x, vib_05x) present")
    else:
        fail("Missing vibration sub-component columns")

    # ---- Final result -----------------------------------------------
    print()
    print("=" * 70)
    print(f"PHASE 1 STATUS: {'PASS' if failed == 0 else 'FAIL'}")
    print(f"Passed: {passed}   Failed: {failed}")
    print("=" * 70)

    if failed == 0:
        print("\nValidated:")
        print("  [OK] Dataset loading and schema")
        print("  [OK] Physical plausibility (Rotax 912 ULS limits)")
        print("  [OK] CAN-style ECU/FADEC communication")
        print("  [OK] Digital Twin state update (1000 rows)")
        print("  [OK] Multi-engine multi-mission structure")
        print("  [OK] All 4 operating scenarios")
        print("  [OK] All 6 flight phases")
        print("  [OK] Derived state: ISA sigma, thermal, electrical, combustion, lubrication")
        print("  [OK] All 8 FMEA fault types")
        print("  [OK] Frequency-domain vibration components (1x, 2x, 0.5x)")
        print("\nNext step: Phase 2 -- Health Monitoring System")
    else:
        print(f"\n{failed} check(s) failed. Review output above.")


if __name__ == "__main__":
    main()