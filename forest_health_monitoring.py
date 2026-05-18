import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/ml_table/"
    "forest_ml_features.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/forest_health")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] forest_ml_features.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("FOREST HEALTH MONITORING")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# REQUIRED FEATURES
# ==========================================

required_features = [

    "ndvi",
    "ndre"
]

for feature in required_features:

    if feature not in df.columns:

        print(f"[ERROR] Missing feature: {feature}")
        exit()

# ==========================================
# OPTIONAL FEATURES
# ==========================================

if "canopy_density" not in df.columns:

    df["canopy_density"] = 0.5

if "spectral_entropy" not in df.columns:

    df["spectral_entropy"] = 1.0

# ==========================================
# NORMALIZATION
# ==========================================

df["ndvi_norm"] = (

    df["ndvi"] + 1
) / 2

df["ndre_norm"] = (

    df["ndre"] + 1
) / 2

# ==========================================
# HEALTH SCORE
# ==========================================

df["health_score"] = (

    (
        df["ndvi_norm"]
        * 0.35
    )

    +

    (
        df["ndre_norm"]
        * 0.40
    )

    +

    (
        df["canopy_density"]
        * 0.15
    )

    +

    (
        (
            1 /
            (
                df["spectral_entropy"]
                + 1e-6
            )
        )
        * 0.10
    )
)

# ==========================================
# CLIP
# ==========================================

df["health_score"] = np.clip(

    df["health_score"],

    0,
    1
)

# ==========================================
# HEALTH CLASSIFICATION
# ==========================================

def classify_health(score):

    if score >= 0.75:

        return "Healthy"

    elif score >= 0.55:

        return "Moderate"

    elif score >= 0.35:

        return "Unhealthy"

    else:

        return "Critical"

# ==========================================
# APPLY CLASSIFICATION
# ==========================================

df["health_class"] = df[
    "health_score"
].apply(classify_health)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("HEALTH SUMMARY")
print("====================================")

health_counts = df[
    "health_class"
].value_counts()

print(health_counts)

# ==========================================
# HEALTH STATISTICS
# ==========================================

print("\n====================================")
print("HEALTH STATISTICS")
print("====================================")

print(f"\nMean Health Score : {df['health_score'].mean():.3f}")

print(f"Max Health Score  : {df['health_score'].max():.3f}")

print(f"Min Health Score  : {df['health_score'].min():.3f}")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "forest_health.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# PIE CHART
# ==========================================

plt.figure(figsize=(8, 8))

health_counts.plot(

    kind="pie",

    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Forest Health Distribution")

plt.tight_layout()

plt.show()

# ==========================================
# HEALTH SCORE HISTOGRAM
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(

    df["health_score"],

    bins=30
)

plt.xlabel("Health Score")

plt.ylabel("Frequency")

plt.title("Forest Health Score Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# TOP HEALTHY TREES
# ==========================================

top_healthy = df.sort_values(

    by="health_score",

    ascending=False
)

print("\n====================================")
print("TOP HEALTHY TREES")
print("====================================")

columns_to_show = [

    c for c in [

        "tree_id",
        "health_score",
        "health_class",
        "ndvi",
        "ndre"
    ]

    if c in top_healthy.columns
]

print(

    top_healthy[
        columns_to_show
    ].head()
)

# ==========================================
# MOST CRITICAL TREES
# ==========================================

critical_trees = df.sort_values(

    by="health_score",

    ascending=True
)

print("\n====================================")
print("MOST CRITICAL TREES")
print("====================================")

print(

    critical_trees[
        columns_to_show
    ].head()
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("FOREST HEALTH MONITORING FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)