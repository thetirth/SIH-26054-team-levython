"""
features.py
===========
Physics-informed feature engineering layer (shared by training + inference).

All derived features are pure functions of the 20 raw CAN/GCS signals --
no labels, no leakage. They encode Rotax 912 ULS operating limits,
ISA atmosphere physics, and FMEA fault signatures so tree models can
split on threshold-like quantities directly.

Raw inputs (unchanged, CAN/GCS-compatible):
    FEATURES_RAW -- the 20 signals used everywhere before.

Model inputs:
    FEATURES_FAULT -- raw + engineered (fault classifier + anomaly detector)
    FEATURES_RUL   -- FEATURES_FAULT + ['cycle'] (elapsed time is known to GCS;
                      RUL is inherently time-dependent: RUL = N - cycle)
"""

import numpy as np
import pandas as pd

FEATURES_RAW = [
    "altitude_m", "airspeed_kt", "ambient_temp", "throttle_pct", "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt", "cht", "oil_temp", "oil_pressure",
    "vibration", "vib_1x", "vib_2x", "vib_05x",
    "battery_voltage", "alternator_voltage", "battery_soc", "injection_timing",
]

# Rotax 912 ULS limits (Operator's Manual, Section 2)
CHT_LIMIT, CHT_CAUTION = 150.0, 135.0
EGT_LIMIT, EGT_CAUTION = 850.0, 800.0
OIL_P_MIN = 22.0
OIL_T_LIMIT = 140.0

ENGINEERED = [
    "cht_margin", "egt_margin", "oil_p_margin", "oil_t_margin",
    "thermal_load", "vib_1x_ratio", "vib_2x_ratio", "vib_05x_ratio",
    "timing_dev", "volt_diff", "soc_deficit",
    "fuel_per_load", "egt_per_load",
    "oil_temp_rise", "cht_rise", "rpm_per_throttle",
    "sigma_isa", "thin_air", "egt_alt_corr", "cht_cooling_gap",
    # v2: fault-decoupling residuals (PHM residual-diagnostics practice)
    "cht_oil_ratio", "egt_load_resid", "oil_p_temp_comp", "vib_excess",
]

# Scenario context (GCS mission config; fault-independent by design, used only
# for baseline calibration, e.g. hot-day CHT vs genuine cooling fault).
# Fixed schema so live API scenarios (endurance/custom) map safely.
SCENARIO_COLS = ["scn_normal", "scn_hot_weather", "scn_high_altitude",
                 "scn_rapid_throttle", "scn_endurance"]

FEATURES_AD = FEATURES_RAW                      # unsupervised AD stays on raw signals
# NOTE: scenario one-hots are engineered (above) but EXCLUDED from model
# inputs: ablation showed they lose 0.02-0.07 macro-F1 (65 train missions
# are too few -- trees memorize sparse scenario x fault coincidences).
FEATURES_FAULT = FEATURES_RAW + ENGINEERED
FEATURES_RUL = FEATURES_FAULT + ["cycle"]


def isa_sigma(altitude_m, ambient_c):
    """ISA troposphere density ratio (ICAO Doc 7488/3). Vectorised."""
    h = np.clip(np.asarray(altitude_m, dtype=float), 0.0, 11000.0)
    t_k = np.maximum(np.asarray(ambient_c, dtype=float) + 273.15, 180.0)
    tr = np.maximum(1.0 - 0.0065 * h / 288.15, 0.1)
    p_ratio = tr ** 5.2561
    t_isa = 288.15 - 0.0065 * h
    return np.clip(p_ratio * (t_isa / t_k), 0.05, 1.2)


def engineer_features(df):
    """Add physics-informed columns. Accepts DataFrame with FEATURES_RAW
    (+ optionally 'cycle'). Returns a new DataFrame with model features."""
    out = df.copy()
    g = lambda c: np.asarray(out[c], dtype=float)

    # --- Safety margins (distance to Rotax redlines; fault thresholds) ---
    out["cht_margin"] = CHT_LIMIT - g("cht")
    out["egt_margin"] = EGT_LIMIT - g("egt")
    out["oil_p_margin"] = g("oil_pressure") - OIL_P_MIN
    out["oil_t_margin"] = OIL_T_LIMIT - g("oil_temp")

    # --- Thermal load index (same weighting as DigitalTwinCore) ---
    cht_n = np.clip((g("cht") - CHT_CAUTION) / (CHT_LIMIT - CHT_CAUTION), 0, 1)
    egt_n = np.clip((g("egt") - EGT_CAUTION) / (EGT_LIMIT - EGT_CAUTION), 0, 1)
    out["thermal_load"] = 0.60 * cht_n + 0.40 * egt_n

    # --- Vibration signature ratios (misfire -> 1x spike; bearing -> 2x;
    #     prop imbalance -> 0.5x). Ratios are load-invariant fault cues. ---
    out["vib_1x_ratio"] = g("vib_1x") / (g("vibration") + 1e-6)
    out["vib_2x_ratio"] = g("vib_2x") / (g("vib_1x") + 1e-6)
    out["vib_05x_ratio"] = g("vib_05x") / (g("vib_1x") + 1e-6)

    # --- Combustion quality: deviation from optimal advance curve ---
    out["timing_dev"] = np.abs(g("injection_timing") - (20.0 - 2.0 * g("engine_load")))

    # --- Electrical: charging margin + stored-energy deficit ---
    out["volt_diff"] = g("alternator_voltage") - g("battery_voltage")
    out["soc_deficit"] = 100.0 - g("battery_soc")

    # --- Load-normalised consumption / temperature (isolates injector,
    #     cooling faults from throttle changes) ---
    out["fuel_per_load"] = g("fuel_flow") / (g("engine_load") + 1e-3)
    out["egt_per_load"] = g("egt") / (g("engine_load") + 1e-3)
    out["rpm_per_throttle"] = g("rpm") / (g("throttle_pct") + 1e-3)

    # --- Temperature rises above ambient (cooling-system effectiveness) ---
    out["oil_temp_rise"] = g("oil_temp") - g("ambient_temp")
    out["cht_rise"] = g("cht") - g("ambient_temp")

    # --- ISA density + altitude corrections (separates thin-air physics
    #     from genuine cooling/combustion degradation) ---
    sig = isa_sigma(g("altitude_m"), g("ambient_temp"))
    out["sigma_isa"] = sig
    out["thin_air"] = 1.0 - sig
    out["egt_alt_corr"] = g("egt") - g("altitude_m") / 1524.0 * 30.0
    out["cht_cooling_gap"] = out["cht_rise"] * sig

    # --- v2 fault-decoupling residuals ---
    # sensor_drift raises measured CHT with NO oil/EGT coupling, cooling
    # raises CHT WITH oil coupling, overheating couples CHT+oil+pressure.
    out["cht_oil_ratio"] = out["cht_rise"] / (out["oil_temp_rise"] + 5.0)
    # EGT residual after removing load + altitude physics (isolates
    # injector/misfire/combustion EGT signatures from throttle changes)
    out["egt_load_resid"] = out["egt_alt_corr"] - (400.0 + 270.0 * g("engine_load"))
    # Viscosity-temperature compensation: removes oil thinning with heat,
    # isolating level/wear faults (lubrication) from temperature effects
    out["oil_p_temp_comp"] = g("oil_pressure") + 0.15 * g("oil_temp")
    # RPM-normalised vibration excess (isolates fault vibration from the
    # normal rpm-dependent baseline ~1.05*(0.05+0.10*rpm/5500))
    out["vib_excess"] = g("vibration") - 1.05 * (0.05 + 0.10 * g("rpm") / 5500.0)

    # --- scenario one-hots (fixed schema; unknown -> all zeros) ---
    scn = out["scenario"].astype(str) if "scenario" in out.columns else ""
    mapping = {"normal": "scn_normal", "hot_weather": "scn_hot_weather",
               "high_altitude": "scn_high_altitude",
               "rapid_throttle": "scn_rapid_throttle", "endurance": "scn_endurance"}
    for col in SCENARIO_COLS:
        out[col] = 0.0
    if isinstance(scn, pd.Series):
        for raw, col in mapping.items():
            out.loc[scn == raw, col] = 1.0

    return out
