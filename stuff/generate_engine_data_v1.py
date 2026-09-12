import numpy as np
import pandas as pd

# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_SEED = 42

N_ENGINES = 5
MISSIONS_PER_ENGINE = 20

MIN_CYCLES = 300
MAX_CYCLES = 600

FAULT_PROBABILITY = 0.35

FAULT_TYPES = [
    "misfire",
    "injector_abnormality",
    "lubrication_issue",
    "cooling_degradation",
    "sensor_drift",
    "combustion_instability",
    "overheating",
    "alternator_failure"
]

SCENARIOS = [
    "normal",
    "hot_weather",
    "high_altitude",
    "rapid_throttle"
]

rng = np.random.default_rng(RANDOM_SEED)


# ================================================================
# 1. MISSION PROFILE
# ================================================================

def build_mission_profile(n_cycles, scenario):

    # ------------------------------------------------------------
    # Flight phase lengths
    # ------------------------------------------------------------

    idle_len = int(n_cycles * 0.05)
    takeoff_len = int(n_cycles * 0.05)
    climb_len = int(n_cycles * 0.15)
    descent_len = int(n_cycles * 0.15)

    cruise_len = (
        n_cycles
        - idle_len
        - takeoff_len
        - climb_len
        - descent_len
        - idle_len
    )

    def ramp(start, end, length):
        return np.linspace(start, end, length)

    # ------------------------------------------------------------
    # Flight phase
    # ------------------------------------------------------------

    flight_phase = np.concatenate([
        np.array(["idle"] * idle_len),

        np.array(["takeoff"] * takeoff_len),

        np.array(["climb"] * climb_len),

        np.array(["cruise"] * cruise_len),

        np.array(["descent"] * descent_len),

        np.array(["landing"] * idle_len)
    ])[:n_cycles]

    # ------------------------------------------------------------
    # RPM
    # ------------------------------------------------------------

    target_rpm = np.concatenate([

        ramp(1000, 1000, idle_len),

        ramp(1000, 4300, takeoff_len),

        ramp(4300, 4500, climb_len),

        ramp(4500, 3900, cruise_len),

        ramp(3900, 1500, descent_len),

        ramp(1500, 1000, idle_len)

    ])[:n_cycles]

    # ------------------------------------------------------------
    # Altitude
    # ------------------------------------------------------------

    altitude = np.concatenate([

        ramp(0, 0, idle_len),

        ramp(0, 500, takeoff_len),

        ramp(500, 5000, climb_len),

        ramp(5000, 5000, cruise_len),

        ramp(5000, 500, descent_len),

        ramp(500, 0, idle_len)

    ])[:n_cycles]

    # ------------------------------------------------------------
    # Airspeed
    # ------------------------------------------------------------

    airspeed = np.concatenate([

        ramp(0, 0, idle_len),

        ramp(0, 70, takeoff_len),

        ramp(70, 120, climb_len),

        ramp(120, 140, cruise_len),

        ramp(140, 60, descent_len),

        ramp(60, 0, idle_len)

    ])[:n_cycles]

    # ------------------------------------------------------------
    # Ambient temperature
    # ------------------------------------------------------------

    if scenario == "hot_weather":
        base_temp = 40

    elif scenario == "high_altitude":
        base_temp = 25

    else:
        base_temp = 30

    ambient_temp = np.concatenate([

        ramp(base_temp, base_temp, idle_len),

        ramp(base_temp, base_temp - 2, takeoff_len),

        ramp(base_temp - 2, base_temp - 15, climb_len),

        ramp(base_temp - 15, base_temp - 18, cruise_len),

        ramp(base_temp - 18, base_temp - 5, descent_len),

        ramp(base_temp - 5, base_temp, idle_len)

    ])[:n_cycles]

    # ------------------------------------------------------------
    # Throttle
    # ------------------------------------------------------------

    throttle_pct = target_rpm / 4500

    # Rapid throttle scenario
    if scenario == "rapid_throttle":

        throttle_variation = rng.normal(0, 0.08, n_cycles)

        throttle_pct = np.clip(
            throttle_pct + throttle_variation,
            0.20,
            1.0
        )

        target_rpm = 1000 + throttle_pct * 3500

    # ------------------------------------------------------------
    # High altitude scenario
    # ------------------------------------------------------------

    if scenario == "high_altitude":

        altitude = altitude * 1.25
        airspeed = airspeed * 0.95

    # ------------------------------------------------------------
    # Environmental noise
    # ------------------------------------------------------------

    ambient_temp += rng.normal(0, 1.0, n_cycles)
    airspeed += rng.normal(0, 2.0, n_cycles)
    altitude += rng.normal(0, 20, n_cycles)

    altitude = np.maximum(altitude, 0)
    airspeed = np.maximum(airspeed, 0)

    return (
        flight_phase,
        target_rpm,
        throttle_pct,
        altitude,
        airspeed,
        ambient_temp
    )


# ================================================================
# 2. ENGINE SENSOR PHYSICS
# ================================================================

def compute_sensors(
    target_rpm,
    throttle_pct,
    altitude,
    airspeed,
    ambient_temp,
    engine_age
):

    n = len(target_rpm)

    # ------------------------------------------------------------
    # RPM
    # ------------------------------------------------------------

    rpm = (
        target_rpm
        + rng.normal(0, 15, n)
    )

    rpm = np.maximum(rpm, 0)

    # ------------------------------------------------------------
    # Engine load
    # ------------------------------------------------------------

    engine_load = np.clip(
        throttle_pct * 0.85
        + (rpm / 5000) * 0.15,
        0,
        1
    )

    # ------------------------------------------------------------
    # Fuel flow
    # ------------------------------------------------------------

    fuel_flow = (
        0.018 * rpm
        * throttle_pct
        * (1 + 0.05 * engine_load)
        + rng.normal(0, 0.4, n)
    )

    fuel_flow = np.maximum(fuel_flow, 0)

    # ------------------------------------------------------------
    # EGT
    # ------------------------------------------------------------

    egt = (
        300
        + 0.075 * rpm
        + 3.5 * fuel_flow
        + 30 * engine_load
        + rng.normal(0, 5, n)
    )

    # ------------------------------------------------------------
    # CHT
    # ------------------------------------------------------------

    # Higher altitude → lower air density → slightly reduced cooling
    altitude_effect = altitude / 10000 * 10

    cht = (
        ambient_temp
        + 70 * engine_load
        + altitude_effect
        - 0.20 * airspeed
        + rng.normal(0, 2.5, n)
    )

    # ------------------------------------------------------------
    # Oil temperature with thermal lag
    # ------------------------------------------------------------

    oil_temp = np.zeros(n)

    oil_temp[0] = ambient_temp[0] + 10

    for i in range(1, n):

        target_oil_temp = (
            ambient_temp[i]
            + 0.018 * rpm[i]
            + 20 * engine_load[i]
        )

        oil_temp[i] = (
            oil_temp[i - 1]
            + 0.05
            * (target_oil_temp - oil_temp[i - 1])
        )

    oil_temp += rng.normal(0, 1, n)

    # ------------------------------------------------------------
    # Oil pressure
    # ------------------------------------------------------------

    oil_pressure = (
        25
        + 0.008 * rpm
        - 0.12 * oil_temp
        - 0.003 * engine_age
        + rng.normal(0, 1, n)
    )

    oil_pressure = np.maximum(oil_pressure, 0)

    # ------------------------------------------------------------
    # Vibration
    # ------------------------------------------------------------

    rpm_component = rpm / 4500 * 0.15

    harmonic_component = (
        0.05
        * np.sin(
            2 * np.pi
            * np.arange(n)
            / 20
        )
    )

    vibration = (
        0.30
        + rpm_component
        + harmonic_component
        + rng.normal(0, 0.04, n)
    )

    vibration = np.abs(vibration)

    # ------------------------------------------------------------
    # Battery voltage
    # ------------------------------------------------------------

    battery_voltage = (
        12.2
        + 0.00035 * rpm
        + rng.normal(0, 0.08, n)
    )

    # ------------------------------------------------------------
    # Alternator voltage
    # ------------------------------------------------------------

    alternator_voltage = (
        13.2
        + 0.00035 * rpm
        + rng.normal(0, 0.08, n)
    )

    # ------------------------------------------------------------
    # Battery state of charge
    # ------------------------------------------------------------

    battery_soc = np.full(n, 95.0)

    # Small natural discharge when RPM is low
    for i in range(1, n):

        if rpm[i] < 1500:

            battery_soc[i] = battery_soc[i - 1] - 0.01

        else:

            battery_soc[i] = min(
                100,
                battery_soc[i - 1] + 0.005
            )

    battery_soc += rng.normal(0, 0.1, n)

    battery_soc = np.clip(
        battery_soc,
        0,
        100
    )

    # ------------------------------------------------------------
    # Injection timing
    # ------------------------------------------------------------

    injection_timing = (
        10
        + 0.002 * rpm
        + rng.normal(0, 0.25, n)
    )

    return {
        "rpm": rpm,
        "engine_load": engine_load,
        "fuel_flow": fuel_flow,
        "egt": egt,
        "cht": cht,
        "oil_temp": oil_temp,
        "oil_pressure": oil_pressure,
        "vibration": vibration,
        "battery_voltage": battery_voltage,
        "alternator_voltage": alternator_voltage,
        "battery_soc": battery_soc,
        "injection_timing": injection_timing
    }


# ================================================================
# 3. FAULT INJECTION
# ================================================================

def inject_fault(
    sensors,
    fault_type,
    degradation_start
):

    n = len(sensors["rpm"])

    severity = np.zeros(n)

    if degradation_start < n:

        remaining = n - degradation_start

        severity[degradation_start:] = (
            np.linspace(0, 1, remaining) ** 1.5
        )

    s = {
        key: value.copy()
        for key, value in sensors.items()
    }

    # ------------------------------------------------------------
    # MISFIRE
    # ------------------------------------------------------------

    if fault_type == "misfire":

        s["rpm"] -= severity * 180

        s["vibration"] += severity * 0.5

        s["egt"] += (
            severity
            * rng.normal(0, 20, n)
        )

    # ------------------------------------------------------------
    # INJECTOR ABNORMALITY
    # ------------------------------------------------------------

    elif fault_type == "injector_abnormality":

        direction = rng.choice(
            [-1, 1]
        )

        s["fuel_flow"] += (
            severity
            * direction
            * 3
        )

        s["egt"] -= severity * 35

        s["injection_timing"] += (
            severity
            * 2
        )

    # ------------------------------------------------------------
    # LUBRICATION ISSUE
    # ------------------------------------------------------------

    elif fault_type == "lubrication_issue":

        s["oil_pressure"] -= severity * 15

        s["oil_temp"] += severity * 20

        s["vibration"] += severity * 0.15

    # ------------------------------------------------------------
    # COOLING DEGRADATION
    # ------------------------------------------------------------

    elif fault_type == "cooling_degradation":

        s["cht"] += severity * 55

        s["oil_temp"] += severity * 12

        s["egt"] += severity * 10

    # ------------------------------------------------------------
    # SENSOR DRIFT
    # ------------------------------------------------------------

    elif fault_type == "sensor_drift":

        s["cht"] += severity * 25

    # ------------------------------------------------------------
    # COMBUSTION INSTABILITY
    # ------------------------------------------------------------

    elif fault_type == "combustion_instability":

        s["rpm"] += (
            severity
            * rng.normal(0, 120, n)
        )

        s["egt"] += (
            severity
            * rng.normal(0, 35, n)
        )

        s["vibration"] += severity * 0.25

    # ------------------------------------------------------------
    # OVERHEATING
    # ------------------------------------------------------------

    elif fault_type == "overheating":

        s["cht"] += severity * 60

        s["oil_temp"] += severity * 25

        s["oil_pressure"] -= severity * 8

    # ------------------------------------------------------------
    # ALTERNATOR FAILURE
    # ------------------------------------------------------------

    elif fault_type == "alternator_failure":

        s["alternator_voltage"] -= severity * 3

        s["battery_voltage"] -= severity * 1.5

        s["battery_soc"] -= severity * 25

    return s, severity


# ================================================================
# 4. GENERATE ONE MISSION
# ================================================================

def generate_mission(
    engine_id,
    mission_id,
    engine_age
):

    n_cycles = int(
        rng.integers(
            MIN_CYCLES,
            MAX_CYCLES + 1
        )
    )

    # ------------------------------------------------------------
    # Choose scenario
    # ------------------------------------------------------------

    scenario = rng.choice(
        SCENARIOS
    )

    (
        flight_phase,
        target_rpm,
        throttle_pct,
        altitude,
        airspeed,
        ambient_temp
    ) = build_mission_profile(
        n_cycles,
        scenario
    )

    # ------------------------------------------------------------
    # Base sensors
    # ------------------------------------------------------------

    sensors = compute_sensors(
        target_rpm,
        throttle_pct,
        altitude,
        airspeed,
        ambient_temp,
        engine_age
    )

    # ------------------------------------------------------------
    # Decide whether this mission has a fault
    # ------------------------------------------------------------

    has_fault = (
        rng.random()
        < FAULT_PROBABILITY
    )

    if has_fault:

        fault_type = rng.choice(
            FAULT_TYPES
        )

        # Different missions fail at different points
        degradation_fraction = rng.uniform(
            0.15,
            0.35
        )

        degradation_start = int(
            n_cycles
            * (1 - degradation_fraction)
        )

        sensors, severity = inject_fault(
            sensors,
            fault_type,
            degradation_start
        )

    else:

        fault_type = "none"

        severity = np.zeros(
            n_cycles
        )

        degradation_start = n_cycles

    # ------------------------------------------------------------
    # Anomaly
    # ------------------------------------------------------------

    is_anomaly = (
        severity > 0.15
    ).astype(int)

    # ------------------------------------------------------------
    # Failure
    # ------------------------------------------------------------

    failure = np.zeros(
        n_cycles,
        dtype=int
    )

    if has_fault:

        failure[-1] = 1

    # ------------------------------------------------------------
    # RUL
    # ------------------------------------------------------------

    if has_fault:

        rul = (
            n_cycles - 1
            - np.arange(n_cycles)
        )

    else:

        # Healthy mission does not fail during this flight.
        rul = np.full(
            n_cycles,
            np.nan
        )

    # ------------------------------------------------------------
    # Health index
    # ------------------------------------------------------------

    health_index = (
        100
        - severity * 80
    )

    # Natural small aging effect
    health_index -= min(
        engine_age / 10000 * 10,
        10
    )

    health_index = np.clip(
        health_index,
        0,
        100
    )

    # ------------------------------------------------------------
    # Fault labels
    # ------------------------------------------------------------

    fault_labels = np.where(
        is_anomaly == 1,
        fault_type,
        "none"
    )

    # ------------------------------------------------------------
    # Create dataframe
    # ------------------------------------------------------------

    df = pd.DataFrame({
        "engine_id": engine_id,
        "mission_id": mission_id,
        "cycle": np.arange(1, n_cycles + 1),

        "flight_phase": flight_phase,
        "scenario": scenario,

        "altitude_m": altitude,
        "airspeed": airspeed,
        "ambient_temp": ambient_temp,

        "throttle_pct": throttle_pct,

        "engine_age_hours": engine_age,

        "rpm": sensors["rpm"],
        "engine_load": sensors["engine_load"],
        "fuel_flow": sensors["fuel_flow"],
        "egt": sensors["egt"],
        "cht": sensors["cht"],
        "oil_temp": sensors["oil_temp"],
        "oil_pressure": sensors["oil_pressure"],
        "vibration": sensors["vibration"],
        "battery_voltage": sensors["battery_voltage"],
        "alternator_voltage": sensors["alternator_voltage"],
        "battery_soc": sensors["battery_soc"],
        "injection_timing": sensors["injection_timing"],

        "health_index": health_index,
        "degradation_severity": severity,

        "fault_type": fault_labels,
        "is_anomaly": is_anomaly,
        "failure": failure,

        "RUL": rul
    })

    return df


# ================================================================
# 5. GENERATE COMPLETE DATASET
# ================================================================

def main():

    all_missions = []

    mission_id = 1

    for engine_number in range(
        1,
        N_ENGINES + 1
    ):

        engine_id = (
            f"E{engine_number:03d}"
        )

        # Starting age differs between engines
        engine_age = rng.uniform(
            100,
            1000
        )

        for _ in range(
            MISSIONS_PER_ENGINE
        ):

            df = generate_mission(
                engine_id,
                mission_id,
                engine_age
            )

            all_missions.append(
                df
            )

            # Increase engine age after every mission
            engine_age += (
                len(df) / 60
            )

            mission_id += 1

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    dataset = pd.concat(
        all_missions,
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    dataset.to_csv(
        "engine_data.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------

    print("=" * 60)

    print("ENGINE DATASET GENERATED")

    print("=" * 60)

    print(
        f"Total rows: {len(dataset):,}"
    )

    print(
        f"Total missions: {dataset['mission_id'].nunique()}"
    )

    print(
        f"Total engines: {dataset['engine_id'].nunique()}"
    )

    print("\nFault distribution:")

    print(
        dataset["fault_type"]
        .value_counts()
    )

    print("\nScenario distribution:")

    print(
        dataset["scenario"]
        .value_counts()
    )

    print("\nSaved as:")

    print("engine_data.csv")


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()