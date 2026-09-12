import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATASET
# ============================================================

FILE_NAME = "engine_data.csv"

df = pd.read_csv(FILE_NAME)

print("=" * 70)
print("ENGINE DATASET VALIDATION")
print("=" * 70)


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n[1] BASIC INFORMATION")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missions:", df["mission_id"].nunique())
print("Engines:", df["engine_id"].nunique())

print("\nColumns:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n[2] MISSING VALUES")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("PASS: No missing values")
else:
    print("WARNING: Missing values found")
    print(missing[missing > 0])


# ============================================================
# 3. DUPLICATES
# ============================================================

print("\n[3] DUPLICATES")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:
    print("PASS: No duplicate rows")
else:
    print("WARNING: Duplicate rows found")


# ============================================================
# 4. NUMERIC DATA SANITY CHECK
# ============================================================

print("\n[4] NUMERIC DATA SANITY")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

invalid_found = False

for column in numeric_columns:

    negative_count = (df[column] < 0).sum()

    infinite_count = np.isinf(
        df[column].dropna()
    ).sum()

    if negative_count > 0:
        print(
            f"WARNING: {column} has "
            f"{negative_count} negative values"
        )
        invalid_found = True

    if infinite_count > 0:
        print(
            f"WARNING: {column} has "
            f"{infinite_count} infinite values"
        )
        invalid_found = True

if not invalid_found:
    print("PASS: No negative/infinite numeric values")


# ============================================================
# 5. SENSOR STATISTICS
# ============================================================

print("\n[5] SENSOR STATISTICS")

sensor_columns = [
    "rpm",
    "engine_load",
    "fuel_flow",
    "egt",
    "cht",
    "oil_temp",
    "oil_pressure",
    "vibration",
    "battery_voltage",
    "alternator_voltage",
    "battery_soc",
    "injection_timing"
]

print(
    df[sensor_columns]
    .describe()
    .round(2)
)


# ============================================================
# 6. FAULT DISTRIBUTION
# ============================================================

print("\n[6] FAULT DISTRIBUTION")

print(
    df["fault_type"]
    .value_counts()
)


# ============================================================
# 7. SCENARIO DISTRIBUTION
# ============================================================

print("\n[7] SCENARIO DISTRIBUTION")

print(
    df["scenario"]
    .value_counts()
)


# ============================================================
# 8. FLIGHT PHASE DISTRIBUTION
# ============================================================

print("\n[8] FLIGHT PHASE DISTRIBUTION")

print(
    df["flight_phase"]
    .value_counts()
)


# ============================================================
# 9. ANOMALY / FAILURE CHECK
# ============================================================

print("\n[9] ANOMALY / FAILURE")

print(
    "Normal rows:",
    (df["is_anomaly"] == 0).sum()
)

print(
    "Anomaly rows:",
    (df["is_anomaly"] == 1).sum()
)

print(
    "Failure rows:",
    (df["failure"] == 1).sum()
)


# ============================================================
# 10. RUL CHECK
# ============================================================

print("\n[10] RUL VALIDATION")

fault_data = df[
    df["fault_type"] != "none"
]

if len(fault_data) > 0:

    print(
        "Minimum RUL:",
        fault_data["RUL"].min()
    )

    print(
        "Maximum RUL:",
        fault_data["RUL"].max()
    )

    failure_rul = fault_data[
        fault_data["failure"] == 1
    ]["RUL"]

    print(
        "RUL at failure:",
        failure_rul.unique()
    )


# ============================================================
# 11. PHYSICS RELATIONSHIP CHECK
# ============================================================

print("\n[11] PHYSICS RELATIONSHIPS")

correlation_columns = [
    "rpm",
    "engine_load",
    "fuel_flow",
    "egt",
    "cht",
    "oil_temp",
    "oil_pressure",
    "vibration",
    "battery_voltage",
    "alternator_voltage"
]

correlation = df[
    correlation_columns
].corr()

print(
    correlation
    .round(2)
)


# ============================================================
# 12. FAULT EFFECT ANALYSIS
# ============================================================

print("\n[12] FAULT EFFECT ANALYSIS")

for fault in df["fault_type"].unique():

    if fault == "none":
        continue

    normal = df[
        df["fault_type"] == "none"
    ]

    faulty = df[
        df["fault_type"] == fault
    ]

    print("\n------------------------------------------")
    print("FAULT:", fault)
    print("------------------------------------------")

    important_columns = [
        "rpm",
        "fuel_flow",
        "egt",
        "cht",
        "oil_temp",
        "oil_pressure",
        "vibration",
        "battery_voltage",
        "alternator_voltage"
    ]

    comparison = pd.DataFrame({
        "normal_mean":
            normal[important_columns].mean(),

        "fault_mean":
            faulty[important_columns].mean()
    })

    comparison["change"] = (
        comparison["fault_mean"]
        - comparison["normal_mean"]
    )

    print(
        comparison
        .round(2)
    )


# ============================================================
# 13. PLOT HELPER
# ============================================================

def plot_mission(
    mission_id,
    columns,
    title
):

    mission = df[
        df["mission_id"] == mission_id
    ]

    plt.figure(figsize=(12, 5))

    for column in columns:

        plt.plot(
            mission["cycle"],
            mission[column],
            label=column
        )

    plt.xlabel("Cycle")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# 14. FIND EXAMPLE MISSIONS
# ============================================================

print("\n[13] EXAMPLE MISSIONS")

fault_missions = (
    df[
        df["fault_type"] != "none"
    ]["mission_id"]
    .unique()
)

healthy_missions = (
    df[
        df["fault_type"] == "none"
    ]["mission_id"]
    .unique()
)

if len(healthy_missions) > 0:

    healthy_id = healthy_missions[0]

    print(
        "Healthy mission:",
        healthy_id
    )

else:

    healthy_id = None


if len(fault_missions) > 0:

    faulty_id = fault_missions[0]

    print(
        "Faulty mission:",
        faulty_id
    )

else:

    faulty_id = None


# ============================================================
# 15. GENERATE GRAPHS
# ============================================================

if healthy_id is not None:

    plot_mission(
        healthy_id,
        [
            "rpm",
            "engine_load"
        ],
        "Healthy Mission - RPM and Engine Load"
    )


if faulty_id is not None:

    plot_mission(
        faulty_id,
        [
            "rpm",
            "cht",
            "oil_temp",
            "oil_pressure",
            "vibration"
        ],
        "Faulty Mission - Engine Parameters"
    )


# ============================================================
# 16. RUL GRAPH
# ============================================================

if faulty_id is not None:

    mission = df[
        df["mission_id"] == faulty_id
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        mission["cycle"],
        mission["RUL"]
    )

    plt.xlabel("Cycle")
    plt.ylabel("Remaining Useful Life")
    plt.title("RUL vs Cycle")

    plt.grid(True)

    plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)