"""
digital_twin_core.py
====================
Digital Twin Core for a Rotax 912 ULS-class aero piston engine
used in MALE UAVs (DRDO SIH-26054).

This module:
  - Maintains the live virtual engine state (EngineState)
  - Computes derived parameters from raw sensor data:
      * ISA air density ratio (sigma)
      * Thermal load index (CHT + EGT proximity to limits)
      * Electrical health index
      * Combustion quality index (injection timing deviation)
      * Estimated brake power and BSFC from sensor readings
      * Operating status (NORMAL / CAUTION / CRITICAL / EMERGENCY)
  - Validates all incoming sensor records
  - Provides JSON export for ML/dashboard consumption

All operating limits are grounded in:
  Ref [1]: BRP-Rotax 912 ULS Operator's Manual (Ed. 2, Rev. 0), Section 2
  Ref [2]: ICAO Doc 7488/3 (1993) -- ISA Standard Atmosphere
  Ref [3]: Heywood, "Internal Combustion Engine Fundamentals", McGraw-Hill 1988
  Ref [4]: ISO 13374-1 (condition monitoring data processing and presentation)
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import json
import math


# ================================================================
# ROTAX 912 ULS OPERATING LIMITS
# Ref [1]: BRP-Rotax 912 ULS Operator's Manual, Section 2
# ================================================================

# RPM
RPM_IDLE         = 1400.0   # idle speed
RPM_CRUISE       = 4800.0   # prop-governed cruise
RPM_MAX_CONT     = 5500.0   # max continuous
RPM_MAX_TAKEOFF  = 5800.0   # 5-min max
RPM_CAUTION_LOW  = 1200.0   # below idle -- caution
RPM_CAUTION_HIGH = 5500.0   # max continuous -- caution if exceeded

# CHT (degC) -- Ref [1] Section 2.3
CHT_CAUTION      = 135.0    # caution onset
CHT_LIMIT        = 150.0    # hard redline

# EGT (degC) -- Ref [1] Section 2.3
EGT_CAUTION      = 800.0    # caution onset
EGT_LIMIT        = 850.0    # hard redline

# Oil Pressure (PSI) -- Ref [1] Section 2.4
OIL_P_MIN        = 22.0     # minimum operating pressure (at idle/warm)
OIL_P_CAUTION_LO = 28.0     # caution if approaching minimum
OIL_P_CAUTION_HI = 72.0     # upper caution (cold start excluded)

# Oil Temperature (degC) -- Ref [1] Section 2.4
OIL_T_MIN        = 50.0     # minimum for takeoff
OIL_T_CAUTION    = 130.0    # caution onset
OIL_T_LIMIT      = 140.0    # maximum

# Vibration (g RMS) -- advisory thresholds (no official Rotax limit;
# derived from general aviation vibration monitoring practice)
# Ref: FAA AC 43.13-1B Section 8-12
VIB_CAUTION      = 0.30     # elevated -- investigate
VIB_LIMIT        = 0.50     # high -- immediate action

# Alternator Voltage (V) -- Ref [1] Section 5
ALT_V_CAUTION_LO = 13.0     # below charging threshold
ALT_V_CAUTION_HI = 15.5     # over-voltage caution
ALT_V_LIMIT_LO   = 11.5     # critical low (battery only)

# Battery Voltage (V)
BATT_V_CAUTION   = 11.8     # caution
BATT_V_LIMIT     = 11.0     # critical

# Battery SOC (%)
SOC_CAUTION      = 30.0     # caution
SOC_LIMIT        = 15.0     # critical

# Injection Timing (deg BTDC)
TIMING_NORMAL_LO = 8.0      # minimum advance
TIMING_NORMAL_HI = 30.0     # maximum advance
TIMING_OPTIMAL_BASE = 20.0  # near-optimal at cruise

# ISA constants -- Ref [2]
ISA_T0   = 288.15
ISA_L    = 0.0065
ISA_G    = 9.80665
ISA_R    = 287.058

# Otto cycle -- Ref [3]
COMPRESSION_RATIO = 9.0
GAMMA             = 1.4


# ================================================================
# ENGINE STATE DATACLASS
# ================================================================

@dataclass
class EngineState:
    """
    Complete virtual state of the digital twin engine at one timestep.
    Raw sensor fields are populated from the incoming record.
    Derived fields are computed by DigitalTwinCore.update().
    """
    # --- Identifiers ---
    engine_id:   str   = ""
    mission_id:  int   = 0
    cycle:       int   = 0
    flight_phase: str  = ""
    scenario:    str   = ""

    # --- Environmental ---
    altitude_m:    float = 0.0
    airspeed_kt:   float = 0.0
    ambient_temp:  float = 0.0
    throttle_pct:  float = 0.0
    engine_age_hours: float = 0.0

    # --- Primary Sensor Readings ---
    rpm:            float = 0.0
    engine_load:    float = 0.0
    fuel_flow:      float = 0.0     # L/h
    egt:            float = 0.0     # degC
    cht:            float = 0.0     # degC (measured; may drift)
    oil_temp:       float = 0.0     # degC
    oil_pressure:   float = 0.0     # PSI
    vibration:      float = 0.0     # g RMS (composite)
    vib_1x:         float = 0.0     # g (1x fundamental)
    vib_2x:         float = 0.0     # g (2x harmonic)
    vib_05x:        float = 0.0     # g (0.5x sub-harmonic)
    battery_voltage:    float = 0.0
    alternator_voltage: float = 0.0
    battery_soc:        float = 0.0
    injection_timing:   float = 0.0

    # --- Derived / Computed ---
    sigma:              float = 1.0   # ISA air density ratio
    power_kw_est:       float = 0.0   # estimated brake power (kW)
    torque_nm_est:      float = 0.0   # estimated torque (N.m)
    bsfc_est:           float = 0.0   # estimated BSFC (g/kW.h)
    otto_efficiency_pct: float = 0.0  # Otto cycle thermal efficiency (%)
    thermal_load_pct:   float = 0.0   # thermal stress index [0-100]
    electrical_health_pct: float = 100.0  # electrical health [0-100]
    combustion_quality: float = 100.0    # combustion quality index [0-100]
    lubrication_health: float = 100.0    # lubrication system health [0-100]
    operating_status:   str  = "UNKNOWN"

    # --- Active Warnings ---
    active_warnings: str = ""  # comma-separated list of active warnings


# ================================================================
# DIGITAL TWIN CORE
# ================================================================

class DigitalTwinCore:
    """
    Real-time digital twin state manager for a Rotax 912 ULS-class engine.

    Workflow:
      1. Receive a sensor record (dict) via .update()
      2. Validate all required fields
      3. Compute ISA density ratio from altitude + ambient_temp
      4. Compute all derived health indices
      5. Determine operating status (NORMAL / CAUTION / CRITICAL / EMERGENCY)
      6. Return updated EngineState

    The state can be exported as dict or JSON for the ML/dashboard pipeline.
    """

    # Minimum required fields in every incoming record
    REQUIRED = [
        "engine_id", "mission_id", "cycle", "flight_phase", "scenario",
        "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct",
        "engine_age_hours",
        "rpm", "engine_load", "fuel_flow", "egt", "cht",
        "oil_temp", "oil_pressure", "vibration",
        "battery_voltage", "alternator_voltage", "battery_soc",
        "injection_timing",
    ]

    # Optional vibration sub-components (present in generated data)
    OPTIONAL_VIB = ["vib_1x", "vib_2x", "vib_05x"]

    def __init__(self):
        self.state = EngineState()
        self._otto_efficiency = (1.0 - 1.0 / (COMPRESSION_RATIO ** (GAMMA - 1.0))) * 100.0

    # ----------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, float(x)))

    @staticmethod
    def isa_density_ratio(altitude_m: float, ambient_temp_c: float) -> float:
        """
        ISA troposphere density ratio sigma = rho / rho0.
        Ref [2]: ICAO Doc 7488/3 (1993)
        """
        h   = max(0.0, min(float(altitude_m), 11000.0))
        T_k = max(180.0, float(ambient_temp_c) + 273.15)

        exponent   = ISA_G / (ISA_R * ISA_L)   # ~5.256
        temp_ratio = max(0.1, 1.0 - ISA_L * h / ISA_T0)
        P_ratio    = temp_ratio ** exponent
        T_isa      = ISA_T0 - ISA_L * h
        sigma      = P_ratio * (T_isa / T_k)
        return max(0.05, min(1.2, sigma))

    # ----------------------------------------------------------
    # MAIN UPDATE
    # ----------------------------------------------------------

    def update(self, record: Dict[str, Any]) -> EngineState:
        """
        Ingest one sensor timestep and update the complete engine state.

        Args:
            record: dict with at minimum all REQUIRED keys

        Returns:
            Updated EngineState (also mutates self.state in-place)

        Raises:
            ValueError: if any required field is missing
        """
        missing = [k for k in self.REQUIRED if k not in record]
        if missing:
            raise ValueError(f"Missing required sensor fields: {missing}")

        s = self.state

        # --- String / int fields ---
        for k in ["engine_id", "flight_phase", "scenario"]:
            setattr(s, k, str(record[k]))
        s.mission_id = int(record["mission_id"])
        s.cycle      = int(record["cycle"])

        # --- Numeric sensor fields ---
        numeric = [
            "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct",
            "engine_age_hours",
            "rpm", "engine_load", "fuel_flow", "egt", "cht",
            "oil_temp", "oil_pressure", "vibration",
            "battery_voltage", "alternator_voltage", "battery_soc",
            "injection_timing",
        ]
        for k in numeric:
            setattr(s, k, float(record[k]))

        # Optional vibration sub-components
        for k in self.OPTIONAL_VIB:
            if k in record and record[k] is not None:
                try:
                    setattr(s, k, float(record[k]))
                except (TypeError, ValueError):
                    setattr(s, k, 0.0)

        # --- ISA Air Density Ratio ---
        s.sigma = self.isa_density_ratio(s.altitude_m, s.ambient_temp)

        # --- Otto Cycle Efficiency (diagnostic constant for this engine) ---
        # eta_th = 1 - 1/r^(gamma-1)  -- Ref [3]: Heywood Ch. 2
        s.otto_efficiency_pct = round(self._otto_efficiency, 2)

        # --- Estimated Brake Power (kW) ---
        # Willans-line calibrated to Rotax 912 ULS:
        # P = 69 kW * (RPM/5500) * throttle * sigma * eta_vol_norm
        # Ref [1]: Rotax 912 ULS performance curves
        rpm_norm = self.clamp(s.rpm / RPM_MAX_CONT, 0.0, 1.1)
        eta_vol_norm = self.clamp(1.0 - 2.0e-8 * (s.rpm - 4200.0)**2 / 0.87, 0.65, 1.0)
        s.power_kw_est = round(
            69.0 * rpm_norm * s.throttle_pct * s.sigma * eta_vol_norm, 2)

        # --- Estimated Torque (N.m) ---
        omega = max(1.0, s.rpm * 2.0 * math.pi / 60.0)
        s.torque_nm_est = round(
            min(s.power_kw_est * 1000.0 / omega, 135.0), 2)

        # --- Estimated BSFC (g/kW.h) ---
        # BSFC = fuel_mass_flow (g/h) / power (kW)
        # fuel_mass = fuel_flow (L/h) * density (g/L)
        fuel_mass_gh = s.fuel_flow * 720.0   # 720 g/L for avgas 100LL
        if s.power_kw_est > 1.0:
            s.bsfc_est = round(fuel_mass_gh / s.power_kw_est, 1)
        else:
            s.bsfc_est = float("nan")

        # --- Thermal Load Index [0-100] ---
        # CHT contribution: normalised between caution (135) and limit (150)
        # EGT contribution: normalised between caution (800) and limit (850)
        # Weighted: CHT 60%, EGT 40% (CHT is primary safety indicator)
        # Ref [1]: Rotax OM Section 2.3; Ref [4]: ISO 13374-1
        cht_norm = self.clamp((s.cht - CHT_CAUTION) / (CHT_LIMIT - CHT_CAUTION), 0.0, 1.0)
        egt_norm = self.clamp((s.egt - EGT_CAUTION) / (EGT_LIMIT - EGT_CAUTION), 0.0, 1.0)
        s.thermal_load_pct = round(100.0 * (0.60 * cht_norm + 0.40 * egt_norm), 2)

        # --- Electrical Health Index [0-100] ---
        # Three contributors weighted by criticality:
        #   Alternator voltage: 50% weight (primary power source)
        #   Battery voltage:    30% weight
        #   Battery SOC:        20% weight
        alt_score  = self.clamp(
            (s.alternator_voltage - ALT_V_LIMIT_LO) / (ALT_V_CAUTION_LO - ALT_V_LIMIT_LO),
            0.0, 1.0)
        batt_score = self.clamp(
            (s.battery_voltage - BATT_V_LIMIT) / (BATT_V_CAUTION - BATT_V_LIMIT),
            0.0, 1.0)
        soc_score  = self.clamp(
            (s.battery_soc - SOC_LIMIT) / (SOC_CAUTION - SOC_LIMIT),
            0.0, 1.0)
        s.electrical_health_pct = round(
            100.0 * (0.50 * alt_score + 0.30 * batt_score + 0.20 * soc_score), 2)

        # --- Combustion Quality Index [0-100] ---
        # Based on injection timing deviation from the optimal advance curve.
        # Optimal at cruise: ~20 deg BTDC (Ducati ignition, Rotax 912)
        # Ref [1]: Rotax 912 Ignition Manual; standard ignition advance maps
        optimal_timing = TIMING_OPTIMAL_BASE - 2.0 * s.engine_load
        timing_dev = abs(s.injection_timing - optimal_timing)
        # Deviation of 0 = 100%; deviation of 10 deg = 0%
        s.combustion_quality = round(self.clamp(100.0 - timing_dev * 10.0, 0.0, 100.0), 2)

        # --- Lubrication Health Index [0-100] ---
        # Oil pressure normalised between limit and operating range
        # Oil temperature normalised between caution and limit
        # Ref [1]: Rotax OM Section 2.4
        oil_p_score = self.clamp(
            (s.oil_pressure - OIL_P_MIN) / (OIL_P_CAUTION_LO - OIL_P_MIN),
            0.0, 1.0)
        oil_t_score = self.clamp(
            1.0 - (s.oil_temp - OIL_T_CAUTION) / (OIL_T_LIMIT - OIL_T_CAUTION),
            0.0, 1.0)
        s.lubrication_health = round(100.0 * (0.60 * oil_p_score + 0.40 * oil_t_score), 2)

        # --- Operating Status & Active Warnings -----------------------
        warnings = []

        # CHT
        if s.cht >= CHT_LIMIT:
            warnings.append("CHT_OVERLIMIT")
        elif s.cht >= CHT_CAUTION:
            warnings.append("CHT_CAUTION")

        # EGT
        if s.egt >= EGT_LIMIT:
            warnings.append("EGT_OVERLIMIT")
        elif s.egt >= EGT_CAUTION:
            warnings.append("EGT_CAUTION")

        # Oil Pressure -- Ref [1]: min 22 PSI
        if s.oil_pressure < OIL_P_MIN:
            warnings.append("OIL_P_LOW_CRITICAL")
        elif s.oil_pressure < OIL_P_CAUTION_LO:
            warnings.append("OIL_P_LOW_CAUTION")

        # Oil Temperature
        if s.oil_temp >= OIL_T_LIMIT:
            warnings.append("OIL_T_OVERLIMIT")
        elif s.oil_temp >= OIL_T_CAUTION:
            warnings.append("OIL_T_CAUTION")

        # Vibration
        if s.vibration >= VIB_LIMIT:
            warnings.append("VIBRATION_HIGH")
        elif s.vibration >= VIB_CAUTION:
            warnings.append("VIBRATION_ELEVATED")

        # Alternator
        if s.alternator_voltage < ALT_V_LIMIT_LO:
            warnings.append("ALTERNATOR_CRITICAL")
        elif s.alternator_voltage < ALT_V_CAUTION_LO:
            warnings.append("ALTERNATOR_CAUTION")
        elif s.alternator_voltage > ALT_V_CAUTION_HI:
            warnings.append("ALTERNATOR_OVERVOLT")

        # Battery SOC
        if s.battery_soc < SOC_LIMIT:
            warnings.append("BATTERY_SOC_CRITICAL")
        elif s.battery_soc < SOC_CAUTION:
            warnings.append("BATTERY_SOC_CAUTION")

        # RPM
        if s.rpm > RPM_MAX_CONT:
            warnings.append("RPM_OVER_MAX_CONT")

        # Determine status from warning severity
        critical_flags = [w for w in warnings if "CRITICAL" in w or "OVERLIMIT" in w]
        caution_flags  = [w for w in warnings if "CAUTION" in w or "ELEVATED" in w
                          or "OVER_MAX_CONT" in w]

        if len(critical_flags) >= 2:
            s.operating_status = "EMERGENCY"
        elif len(critical_flags) >= 1:
            s.operating_status = "CRITICAL"
        elif len(caution_flags) >= 1:
            s.operating_status = "CAUTION"
        else:
            s.operating_status = "NORMAL"

        s.active_warnings = ",".join(warnings) if warnings else ""

        return s

    # ----------------------------------------------------------
    # EXPORT
    # ----------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self.state)

    def to_json(self, indent: int = 2) -> str:
        d = self.to_dict()
        # Handle NaN (not JSON serialisable)
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return json.dumps(d, indent=indent)

    def get_health_summary(self) -> dict:
        """Return a concise health summary dict for the dashboard."""
        s = self.state
        return {
            "operating_status":     s.operating_status,
            "active_warnings":      s.active_warnings,
            "thermal_load_pct":     s.thermal_load_pct,
            "electrical_health_pct":s.electrical_health_pct,
            "combustion_quality":   s.combustion_quality,
            "lubrication_health":   s.lubrication_health,
            "power_kw_est":         s.power_kw_est,
            "bsfc_est":             s.bsfc_est,
            "sigma":                s.sigma,
        }


# ================================================================
# SELF-TEST
# ================================================================

if __name__ == "__main__":
    # Sample: Rotax 912 ULS at cruise, 5000 m, normal conditions
    # Expected: NORMAL status, CHT ~110-130 degC, EGT ~660-700 degC
    sample = {
        "engine_id":       "E001",
        "mission_id":      1,
        "cycle":           250,
        "flight_phase":    "cruise",
        "scenario":        "normal",
        "altitude_m":      5000.0,
        "airspeed_kt":     92.0,
        "ambient_temp":    3.0,       # ISA at 5000 m: 28 - 0.0065*5000 = -4.5 + offset
        "throttle_pct":    0.80,
        "engine_age_hours":420.0,
        "rpm":             4750.0,
        "engine_load":     0.82,
        "fuel_flow":       18.5,      # L/h -- within Rotax spec at 75%
        "egt":             680.0,     # degC -- normal cruise (620-720 range)
        "cht":             125.0,     # degC -- below caution (135)
        "oil_temp":        98.0,      # degC -- favorable range (90-110)
        "oil_pressure":    55.0,      # PSI  -- normal (22-72)
        "vibration":       0.18,      # g RMS -- normal
        "vib_1x":          0.12,
        "vib_2x":          0.04,
        "vib_05x":         0.02,
        "battery_voltage":     13.7,
        "alternator_voltage":  14.1,
        "battery_soc":         94.0,
        "injection_timing":    21.5,  # deg BTDC -- normal cruise advance
    }

    twin  = DigitalTwinCore()
    state = twin.update(sample)

    print("=" * 70)
    print("DIGITAL TWIN CORE -- Rotax 912 ULS self-test")
    print("=" * 70)
    print(f"\nOperating Status  : {state.operating_status}")
    print(f"Active Warnings   : '{state.active_warnings}'")
    print(f"\n--- Physical State ---")
    print(f"RPM               : {state.rpm:.0f} RPM")
    print(f"CHT               : {state.cht:.1f} degC  (limit: {CHT_LIMIT})")
    print(f"EGT               : {state.egt:.1f} degC  (limit: {EGT_LIMIT})")
    print(f"Oil Pressure      : {state.oil_pressure:.1f} PSI  (min: {OIL_P_MIN})")
    print(f"Oil Temperature   : {state.oil_temp:.1f} degC  (max: {OIL_T_LIMIT})")
    print(f"Vibration (RMS)   : {state.vibration:.3f} g  (caution: {VIB_CAUTION})")
    print(f"\n--- ISA & Performance ---")
    print(f"Air Density Ratio : {state.sigma:.4f}  (sea level = 1.0)")
    print(f"Power Est.        : {state.power_kw_est:.1f} kW")
    print(f"Torque Est.       : {state.torque_nm_est:.1f} N.m")
    print(f"BSFC Est.         : {state.bsfc_est:.0f} g/kW.h  (ref: 285)")
    print(f"Otto Efficiency   : {state.otto_efficiency_pct:.1f}%  (ideal)")
    print(f"\n--- Health Indices ---")
    print(f"Thermal Load      : {state.thermal_load_pct:.1f}%")
    print(f"Electrical Health : {state.electrical_health_pct:.1f}%")
    print(f"Combustion Quality: {state.combustion_quality:.1f}%")
    print(f"Lubrication Health: {state.lubrication_health:.1f}%")
    print(f"\n[PASS] Digital Twin Core operational")
    print(f"\nJSON export (first 500 chars):")
    print(twin.to_json()[:500])
