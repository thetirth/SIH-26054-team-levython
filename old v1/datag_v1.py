"""
Engine Data Generator — AI-Enabled Digital Twin System (MALE UAV Piston Engine)
--------------------------------------------------------------------------------
Generates a synthetic, physics-linked dataset of engine sensor readings.
No real engine needed — every sensor is derived from one shared "engine state"
(RPM/load/mission-phase), so all readings stay consistent with each other and
with the fault label on every row.

Output: engine_data.csv
Columns:
    mission_id        - which simulated flight this row belongs to
    cycle             - time step within that flight
    rpm, fuel_flow, egt, cht, oil_temp, oil_pressure,
    vibration, battery_voltage, injection_timing   - the 9 sensor readings
    fault_type        - none / misfire / injector_abnormality / lubrication_issue /
                        cooling_degradation / sensor_drift / combustion_instability /
                        overheating
    is_anomaly        - 0/1 flag version of fault_type
    RUL               - Remaining Useful Life (cycles until failure).
                        NaN for missions that stay healthy the whole flight.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# CONFIG — tweak these freely, nothing else needs to change
# ----------------------------------------------------------------
RANDOM_SEED = 42
N_HEALTHY_MISSIONS = 20          # flights that never fail
N_FAULT_MISSIONS_PER_TYPE = 5    # flights per fault type (7 fault types)
MIN_CYCLES = 300                 # shortest simulated flight (time steps)
MAX_CYCLES = 600                 # longest simulated flight
DEGRADATION_FRACTION = 0.25      # last 25% of a "fault" flight shows degradation

FAULT_TYPES = [
    "misfire",
    "injector_abnormality",
    "lubrication_issue",
    "cooling_degradation",
    "sensor_drift",
    "combustion_instability",
    "overheating",
]

rng = np.random.default_rng(RANDOM_SEED)


# ----------------------------------------------------------------
# 1) MISSION PROFILE — the one shared "state" every sensor is built from
# ----------------------------------------------------------------
def build_mission_profile(n_cycles):
    """
    Builds the flight-phase timeline for one mission:
    ground idle -> climb -> cruise -> descent -> landing idle
    Returns: target_rpm, throttle_pct, ambient_temp, airspeed (all length n_cycles)
    """
    idle_len = max(1, int(n_cycles * 0.05))
    climb_len = max(1, int(n_cycles * 0.15))
    descent_len = max(1, int(n_cycles * 0.15))
    cruise_len = n_cycles - (2 * idle_len + climb_len + descent_len)
    cruise_len = max(1, cruise_len)

    def ramp(start, end, length):
        return np.linspace(start, end, length)

    target_rpm = np.concatenate([
        ramp(1000, 1000, idle_len),     # ground idle
        ramp(1000, 4500, climb_len),    # climb power
        ramp(4500, 3800, cruise_len),   # cruise power
        ramp(3800, 1200, descent_len),  # throttle back for descent
        ramp(1200, 1000, idle_len),     # landing idle
    ])[:n_cycles]

    airspeed = np.concatenate([
        ramp(0, 0, idle_len),
        ramp(0, 120, climb_len),
        ramp(120, 140, cruise_len),
        ramp(140, 40, descent_len),
        ramp(40, 0, idle_len),
    ])[:n_cycles]

    ambient_temp = np.concatenate([
        ramp(30, 30, idle_len),
        ramp(30, 10, climb_len),        # cooler climbing to altitude
        ramp(10, 5, cruise_len),
        ramp(5, 20, descent_len),
        ramp(20, 30, idle_len),
    ])[:n_cycles]

    throttle_pct = target_rpm / target_rpm.max()

    return target_rpm, throttle_pct, ambient_temp, airspeed


# ----------------------------------------------------------------
# 2) SENSOR PHYSICS FORMULAS — same table we planned, turned into code
# ----------------------------------------------------------------
def compute_sensors(target_rpm, throttle_pct, ambient_temp, airspeed):
    n = len(target_rpm)

    # RPM: mission target + small sensor noise
    rpm = target_rpm + rng.normal(0, 15, n)

    # Fuel flow: scales with RPM and throttle position
    fuel_flow = 0.02 * rpm * throttle_pct + rng.normal(0, 0.5, n)

    # EGT: rises with RPM and fuel flow
    egt = 300 + 0.08 * rpm + 4 * fuel_flow + rng.normal(0, 5, n)

    # CHT: ambient temp + engine load - cooling from airspeed
    load = rpm / rpm.max()
    cht = ambient_temp + 140 * load - 0.3 * airspeed + rng.normal(0, 3, n)

    # Oil temp: rises SLOWLY toward a target (thermal lag, not instant)
    oil_temp = np.zeros(n)
    oil_temp[0] = ambient_temp[0] + 10
    for i in range(1, n):
        target_oil_temp = ambient_temp[i] + 0.02 * rpm[i]
        oil_temp[i] = oil_temp[i - 1] + 0.05 * (target_oil_temp - oil_temp[i - 1])
    oil_temp = oil_temp + rng.normal(0, 1, n)

    # Oil pressure: rises with RPM, falls as oil gets hotter/thinner
    oil_pressure = 20 + 0.01 * rpm - 0.15 * oil_temp + rng.normal(0, 1, n)

    # Vibration: baseline wave at (RPM-based) firing frequency + noise
    firing_freq = rpm / 60.0 * 2
    t = np.arange(n)
    vibration = 0.5 + 0.1 * np.sin(2 * np.pi * firing_freq * t / max(n, 1)) + rng.normal(0, 0.05, n)
    vibration = np.abs(vibration)

    # Battery / alternator voltage: alternator output scales with RPM
    battery_voltage = 12 + 0.0015 * rpm + rng.normal(0, 0.1, n)

    # Injection timing: target curve based on RPM + small noise
    injection_timing = 10 + 0.002 * rpm + rng.normal(0, 0.3, n)

    return {
        "rpm": rpm,
        "fuel_flow": fuel_flow,
        "egt": egt,
        "cht": cht,
        "oil_temp": oil_temp,
        "oil_pressure": oil_pressure,
        "vibration": vibration,
        "battery_voltage": battery_voltage,
        "injection_timing": injection_timing,
    }


# ----------------------------------------------------------------
# 3) FAULT INJECTION — only touches the sensors that SHOULD move
#    for that fault (this is what keeps labels honest)
# ----------------------------------------------------------------
def inject_fault(sensors, fault_type, degradation_start):
    n = len(sensors["rpm"])
    severity = np.zeros(n)
    if degradation_start < n:
        ramp_len = n - degradation_start
        # severity climbs from 0 to 1, accelerating near the end (realistic decay curve)
        severity[degradation_start:] = np.linspace(0, 1, ramp_len) ** 1.5

    s = {k: v.copy() for k, v in sensors.items()}

    if fault_type == "misfire":
        s["rpm"] -= severity * rng.normal(150, 40, n)
        s["vibration"] += severity * rng.uniform(0.3, 0.8, n)

    elif fault_type == "injector_abnormality":
        s["fuel_flow"] += severity * rng.normal(0, 3, n) * np.sign(rng.normal(0, 1, n))
        s["egt"] -= severity * 40

    elif fault_type == "lubrication_issue":
        s["oil_pressure"] -= severity * 15
        s["oil_temp"] += severity * 20

    elif fault_type == "cooling_degradation":
        s["cht"] += severity * 60

    elif fault_type == "sensor_drift":
        # only ONE sensor slowly reads wrong — no real underlying cause
        s["cht"] += severity * 25

    elif fault_type == "combustion_instability":
        s["rpm"] += severity * rng.normal(0, 100, n)
        s["egt"] += severity * rng.normal(0, 30, n)

    elif fault_type == "overheating":
        s["cht"] += severity * 55
        s["oil_temp"] += severity * 25

    return s, severity


# ----------------------------------------------------------------
# 4) BUILD ONE MISSION (healthy or degrading)
# ----------------------------------------------------------------
def generate_mission(mission_id, fault_type=None):
    n_cycles = int(rng.integers(MIN_CYCLES, MAX_CYCLES))
    target_rpm, throttle_pct, ambient_temp, airspeed = build_mission_profile(n_cycles)
    sensors = compute_sensors(target_rpm, throttle_pct, ambient_temp, airspeed)

    if fault_type is None:
        fault_type_col = np.array(["none"] * n_cycles)
        is_anomaly = np.zeros(n_cycles, dtype=int)
        rul = np.full(n_cycles, np.nan)  # healthy mission never fails -> RUL unknown
    else:
        degradation_start = int(n_cycles * (1 - DEGRADATION_FRACTION))
        sensors, severity = inject_fault(sensors, fault_type, degradation_start)
        # label a row with the fault ONLY once its matching sensors actually moved
        is_anomaly = (severity > 0.15).astype(int)
        fault_type_col = np.where(is_anomaly == 1, fault_type, "none")
        rul = (n_cycles - 1) - np.arange(n_cycles)  # counts down to 0 at the failure point

    df = pd.DataFrame(sensors)
    df.insert(0, "cycle", np.arange(1, n_cycles + 1))
    df.insert(0, "mission_id", mission_id)
    df["fault_type"] = fault_type_col
    df["is_anomaly"] = is_anomaly
    df["RUL"] = rul
    return df


# ----------------------------------------------------------------
# 5) BUILD THE FULL DATASET
# ----------------------------------------------------------------
def main():
    missions = []
    mission_id = 1

    for _ in range(N_HEALTHY_MISSIONS):
        missions.append(generate_mission(mission_id, fault_type=None))
        mission_id += 1

    for fault_type in FAULT_TYPES:
        for _ in range(N_FAULT_MISSIONS_PER_TYPE):
            missions.append(generate_mission(mission_id, fault_type=fault_type))
            mission_id += 1

    dataset = pd.concat(missions, ignore_index=True)
    dataset.to_csv("engine_data.csv", index=False)

    print(f"Generated {len(dataset)} rows across {mission_id - 1} missions.")
    print("\nRows per fault_type:")
    print(dataset["fault_type"].value_counts())


if __name__ == "__main__":
    main()