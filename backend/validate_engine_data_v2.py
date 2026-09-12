import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FILE = "engine_data.csv"

print("=" * 65)
print("ENGINE DATASET V2 VALIDATION")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(FILE):
    print("ERROR: engine_data.csv not found.")
    exit()

df = pd.read_csv(FILE)

print("\n[1] BASIC INFORMATION")
print("-" * 45)

print("Rows     :", len(df))
print("Columns  :", len(df.columns))
print("Missions :", df["mission_id"].nunique())
print("Engines  :", df["engine_id"].nunique())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

print("\n[2] REQUIRED COLUMNS")
print("-" * 45)

required = [
    "engine_id", "mission_id", "cycle", "elapsed_time_sec",
    "flight_phase", "scenario", "altitude_m", "airspeed_kt",
    "ambient_temp", "throttle_pct", "engine_age_hours",
    "rpm", "engine_load", "fuel_flow", "egt",
    "cht_true", "cht", "oil_temp", "oil_pressure",
    "vibration", "battery_voltage", "alternator_voltage",
    "battery_soc", "injection_timing", "health_index",
    "degradation_severity", "fault_onset_cycle",
    "fault_type", "is_anomaly", "failure", "RUL"
]

missing_columns = [x for x in required if x not in df.columns]

if len(missing_columns) == 0:
    print("PASS: All required columns exist.")
else:
    print("FAIL: Missing columns:", missing_columns)


# ============================================================
# DUPLICATES
# ============================================================

print("\n[3] DUPLICATES")
print("-" * 45)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:
    print("PASS")
else:
    print("FAIL")


# ============================================================
# MISSING VALUES
# ============================================================

print("\n[4] MISSING VALUES")
print("-" * 45)

missing = df.isnull().sum()

for col, count in missing.items():
    if count > 0:
        print(col, ":", count)

print("RUL missing values are expected for healthy missions.")


# ============================================================
# CORRECT MISSION-LEVEL FAULT DETECTION
# ============================================================

print("\n[5] MISSION-LEVEL FAULT DISTRIBUTION")
print("-" * 45)

# IMPORTANT:
# Do NOT use .first() on the entire mission because the
# first rows of faulty missions are "none" before fault onset.

fault_rows = df[df["fault_type"] != "none"]

mission_faults = (
    fault_rows.groupby("mission_id")["fault_type"]
    .first()
)

# Missions having no fault rows = healthy
all_missions = set(df["mission_id"].unique())
faulty_missions = set(mission_faults.index)
healthy_missions = all_missions - faulty_missions

print("Healthy missions:", len(healthy_missions))
print("Faulty missions :", len(faulty_missions))

print("\nFault counts:")

expected = {
    "cooling_degradation": 10,
    "misfire": 10,
    "injector_abnormality": 10,
    "lubrication_issue": 10,
    "abnormal_vibration": 10,
    "combustion_instability": 10,
    "overheating": 10,
    "sensor_drift": 10
}

mission_fault_count = mission_faults.value_counts()

mission_distribution_pass = True

if len(healthy_missions) == 20:
    print("PASS: none =", len(healthy_missions))
else:
    print("FAIL: Expected 20 healthy missions.")
    mission_distribution_pass = False

for fault, expected_count in expected.items():

    actual = mission_fault_count.get(fault, 0)

    print(
        fault,
        ":",
        actual,
        "/ expected",
        expected_count
    )

    if actual != expected_count:
        mission_distribution_pass = False

if mission_distribution_pass:
    print("\nPASS: Mission-level fault distribution is correct.")


# ============================================================
# FAILURE CHECK
# ============================================================

print("\n[6] FAILURE CHECK")
print("-" * 45)

failure_pass = True

total_failures = df["failure"].sum()

print("Total failure rows:", total_failures)

# Every faulty mission must have exactly one failure
for mission in faulty_missions:

    data = df[df["mission_id"] == mission]

    failures = data["failure"].sum()

    if failures != 1:
        print(
            "FAIL:",
            mission,
            "has",
            failures,
            "failure rows"
        )
        failure_pass = False

# Healthy missions must have zero failures
for mission in healthy_missions:

    data = df[df["mission_id"] == mission]

    failures = data["failure"].sum()

    if failures != 0:
        print(
            "FAIL:",
            mission,
            "is healthy but contains failure"
        )
        failure_pass = False

if failure_pass:
    print("PASS: Failure labels are correct.")


# ============================================================
# RUL CHECK
# ============================================================

print("\n[7] RUL CHECK")
print("-" * 45)

rul_pass = True

# Faulty missions
for mission in faulty_missions:

    data = (
        df[df["mission_id"] == mission]
        .sort_values("cycle")
    )

    rul = data["RUL"].values

    # No missing RUL
    if np.isnan(rul).any():
        print("FAIL:", mission, "contains missing RUL.")
        rul_pass = False
        continue

    # Final RUL must be zero
    if rul[-1] != 0:
        print(
            "FAIL:",
            mission,
            "final RUL is",
            rul[-1]
        )
        rul_pass = False

    # RUL should decrease by exactly 1
    if len(rul) > 1:

        differences = np.diff(rul)

        if not np.all(differences == -1):
            print(
                "FAIL:",
                mission,
                "RUL does not decrease by 1."
            )
            rul_pass = False


# Healthy missions should have NaN RUL
for mission in healthy_missions:

    data = df[df["mission_id"] == mission]

    if data["RUL"].notna().any():

        print(
            "FAIL:",
            mission,
            "healthy mission contains RUL."
        )

        rul_pass = False


if rul_pass:
    print("PASS: RUL is correct.")


# ============================================================
# FAULT ONSET CHECK
# ============================================================

print("\n[8] FAULT ONSET CHECK")
print("-" * 45)

onset_pass = True

for mission in faulty_missions:

    data = (
        df[df["mission_id"] == mission]
        .sort_values("cycle")
    )

    onset_values = data["fault_onset_cycle"].dropna().unique()

    if len(onset_values) != 1:

        print("FAIL:", mission, "invalid onset.")
        onset_pass = False
        continue

    onset = onset_values[0]

    actual_fault_start = data[
        data["fault_type"] != "none"
    ]["cycle"].min()

    if actual_fault_start != onset:

        print(
            "FAIL:",
            mission,
            "declared onset =",
            onset,
            "actual =",
            actual_fault_start
        )

        onset_pass = False

if onset_pass:
    print("PASS: Fault onset is correct.")


# ============================================================
# NUMERIC CHECK
# ============================================================

print("\n[9] NUMERIC CHECK")
print("-" * 45)

numeric = df.select_dtypes(include=np.number)

negative_values = []

for col in numeric.columns:

    if (numeric[col].dropna() < 0).any():
        negative_values.append(col)

if len(negative_values) == 0:
    print("PASS: No negative numeric values.")
else:
    print("WARNING: Negative values:", negative_values)


infinite_values = np.isinf(numeric).sum().sum()

print("Infinite values:", infinite_values)

if infinite_values == 0:
    print("PASS")


# ============================================================
# SENSOR RANGE CHECK
# ============================================================

print("\n[10] SENSOR RANGE CHECK")
print("-" * 45)

ranges = {
    "rpm": (0, 10000),
    "fuel_flow": (0, 100),
    "egt": (0, 1500),
    "cht": (0, 300),
    "cht_true": (0, 300),
    "oil_temp": (0, 200),
    "vibration": (0, 20),
    "battery_voltage": (0, 30),
    "alternator_voltage": (0, 30),
    "battery_soc": (0, 100),
    "throttle_pct": (0, 1),
    "health_index": (0, 100),
    "degradation_severity": (0, 1)
}

for col, (low, high) in ranges.items():

    invalid = (
        (df[col] < low) |
        (df[col] > high)
    ).sum()

    if invalid == 0:
        print("PASS:", col)
    else:
        print(
            "WARNING:",
            col,
            "has",
            invalid,
            "values outside range"
        )


# ============================================================
# ANOMALY CHECK
# ============================================================

print("\n[11] ANOMALY CHECK")
print("-" * 45)

anomalies = df["is_anomaly"].sum()

print("Anomaly rows:", anomalies)

if anomalies > 0:
    print("PASS")
else:
    print("FAIL")


# ============================================================
# SENSOR DRIFT CHECK
# ============================================================

print("\n[12] SENSOR DRIFT CHECK")
print("-" * 45)

drift_missions = mission_faults[
    mission_faults == "sensor_drift"
].index

drift_pass = True

for mission in drift_missions:

    data = (
        df[df["mission_id"] == mission]
        .sort_values("cycle")
    )

    onset = data["fault_onset_cycle"].iloc[0]

    before = data[data["cycle"] < onset]
    after = data[data["cycle"] >= onset]

    if len(before) == 0 or len(after) == 0:
        continue

    before_error = (
        before["cht"] - before["cht_true"]
    ).abs().mean()

    after_error = (
        after["cht"] - after["cht_true"]
    ).abs().mean()

    if after_error <= before_error:
        drift_pass = False

if drift_pass:
    print("PASS: Sensor drift behavior is correct.")
else:
    print("WARNING: Sensor drift needs review.")


# ============================================================
# ALTERNATOR CHECK
# ============================================================

print("\n[13] ALTERNATOR FAILURE CHECK")
print("-" * 45)

alt_missions = mission_faults[
    mission_faults == "abnormal_vibration"
].index

alt_pass = True

for mission in alt_missions:

    data = (
        df[df["mission_id"] == mission]
        .sort_values("cycle")
    )

    onset = data["fault_onset_cycle"].iloc[0]

    before = data[data["cycle"] < onset]
    after = data[data["cycle"] >= onset]

    if len(before) == 0 or len(after) == 0:
        continue

    before_voltage = before["alternator_voltage"].mean()
    after_voltage = after["alternator_voltage"].mean()

    if after_voltage >= before_voltage:
        alt_pass = False

if alt_pass:
    print("PASS: Alternator failure affects electrical system.")
else:
    print("WARNING: Alternator behavior needs review.")


# ============================================================
# ROW-LEVEL DISTRIBUTION
# ============================================================

print("\n[14] ROW-LEVEL FAULT DISTRIBUTION")
print("-" * 45)

print(df["fault_type"].value_counts())


# ============================================================
# SCENARIO DISTRIBUTION
# ============================================================

print("\n[15] SCENARIO DISTRIBUTION")
print("-" * 45)

print(df["scenario"].value_counts())


# ============================================================
# FLIGHT PHASE DISTRIBUTION
# ============================================================

print("\n[16] FLIGHT PHASE DISTRIBUTION")
print("-" * 45)

print(df["flight_phase"].value_counts())


# ============================================================
# HEALTHY MISSION GRAPH
# ============================================================

print("\n[17] GENERATING HEALTHY GRAPH")
print("-" * 45)

healthy_mission = sorted(healthy_missions)[0]

data = (
    df[df["mission_id"] == healthy_mission]
    .sort_values("cycle")
)

plt.figure(figsize=(12, 6))

plt.plot(
    data["cycle"],
    data["rpm"],
    label="RPM"
)

plt.xlabel("Cycle")
plt.ylabel("RPM")
plt.title("Healthy Mission - " + str(healthy_mission))

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "validation_healthy_rpm.png",
    dpi=150
)

plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)

print("\nDataset:")
print("Rows     :", len(df))
print("Missions :", df["mission_id"].nunique())
print("Engines  :", df["engine_id"].nunique())

print("\nExpected:")
print("20 healthy missions")
print("10 missions for each of 8 faults")
print("80 faulty missions")
print("80 failure rows")

print("\nCheck the PASS / FAIL / WARNING messages above.")
print("=" * 65)