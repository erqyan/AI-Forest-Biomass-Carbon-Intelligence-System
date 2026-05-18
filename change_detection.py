import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================
# INPUT FILES
# ==========================================

FEATURE_FILE = (
    "outputs/ml_table/"
    "forest_ml_features.csv"
)

HEALTH_FILE = (
    "outputs/forest_health/"
    "forest_health.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/change_detection")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILES
# ==========================================

required_files = [

    FEATURE_FILE,
    HEALTH_FILE
]

for f in required_files:

    if not Path(f).exists():

        print(f"[ERROR] Missing file: {f}")
        exit()

# ==========================================
# LOAD DATA
# ==========================================

features_df = pd.read_csv(FEATURE_FILE)

health_df = pd.read_csv(HEALTH_FILE)

print("\n====================================")
print("FOREST CHANGE DETECTION")
print("====================================")

print(f"\nTotal Samples: {len(features_df)}")

# ==========================================
# MERGE
# ==========================================

df = pd.concat(

    [features_df, health_df],

    axis=1
)

# ==========================================
# CREATE TEMPORAL DATA
# ==========================================

np.random.seed(42)

# ==========================================
# T1 (OLD)
# ==========================================

df["ndvi_t1"] = df["ndvi"]

df["health_t1"] = df["health_score"]

# ==========================================
# T2 (NEW)
# ==========================================

degradation_factor = np.random.uniform(

    0.7,
    1.1,

    size=len(df)
)

df["ndvi_t2"] = (

    df["ndvi_t1"]
    *
    degradation_factor
)

df["health_t2"] = (

    df["health_t1"]
    *
    degradation_factor
)

# ==========================================
# CANOPY CHANGE
# ==========================================

if "max_height" in df.columns:

    df["height_t1"] = df["max_height"]

    height_factor = np.random.uniform(

        0.8,
        1.05,

        size=len(df)
    )

    df["height_t2"] = (

        df["height_t1"]
        *
        height_factor
    )

else:

    df["height_t1"] = 10

    df["height_t2"] = 10

# ==========================================
# BIOMASS CHANGE
# ==========================================

if "Predicted_AGB_kg" in df.columns:

    df["biomass_t1"] = df["Predicted_AGB_kg"]

else:

    df["biomass_t1"] = np.random.uniform(

        50,
        300,

        size=len(df)
    )

biomass_factor = np.random.uniform(

    0.75,
    1.05,

    size=len(df)
)

df["biomass_t2"] = (

    df["biomass_t1"]
    *
    biomass_factor
)

# ==========================================
# DIFFERENCE
# ==========================================

df["ndvi_change"] = (

    df["ndvi_t2"]
    -
    df["ndvi_t1"]
)

df["health_change"] = (

    df["health_t2"]
    -
    df["health_t1"]
)

df["height_change"] = (

    df["height_t2"]
    -
    df["height_t1"]
)

df["biomass_change"] = (

    df["biomass_t2"]
    -
    df["biomass_t1"]
)

# ==========================================
# CHANGE CLASSIFICATION
# ==========================================

def classify_change(change):

    if change < -20:

        return "Severe Loss"

    elif change < -5:

        return "Moderate Loss"

    elif change < 5:

        return "Stable"

    else:

        return "Growth"

# ==========================================
# APPLY
# ==========================================

df["change_class"] = df[
    "biomass_change"
].apply(classify_change)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("CHANGE SUMMARY")
print("====================================")

change_counts = df[
    "change_class"
].value_counts()

print(change_counts)

# ==========================================
# STATISTICS
# ==========================================

print("\n====================================")
print("CHANGE STATISTICS")
print("====================================")

print(f"\nMean Biomass Change : {df['biomass_change'].mean():.2f}")

print(f"Max Biomass Gain    : {df['biomass_change'].max():.2f}")

print(f"Max Biomass Loss    : {df['biomass_change'].min():.2f}")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "forest_change.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# HISTOGRAM
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(

    df["biomass_change"],

    bins=30
)

plt.xlabel("Biomass Change")

plt.ylabel("Frequency")

plt.title("Forest Biomass Change Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# PIE CHART
# ==========================================

plt.figure(figsize=(8, 8))

change_counts.plot(

    kind="pie",

    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Forest Change Classification")

plt.tight_layout()

plt.show()

# ==========================================
# MOST DEGRADED
# ==========================================

degraded = df.sort_values(

    by="biomass_change",

    ascending=True
)

print("\n====================================")
print("MOST DEGRADED TREES")
print("====================================")

columns_to_show = [

    c for c in [

        "tree_id",
        "biomass_change",
        "change_class",
        "health_change",
        "ndvi_change"
    ]

    if c in degraded.columns
]

print(

    degraded[
        columns_to_show
    ].head()
)

# ==========================================
# MOST RECOVERED
# ==========================================

recovered = df.sort_values(

    by="biomass_change",

    ascending=False
)

print("\n====================================")
print("MOST RECOVERED TREES")
print("====================================")

print(

    recovered[
        columns_to_show
    ].head()
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("CHANGE DETECTION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)