import pandas as pd
import numpy as np

from pathlib import Path

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

OUTPUT_DIR = Path("outputs/biomass")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] Feature table not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("ALLOMETRIC BIOMASS ESTIMATION")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# CREATE REQUIRED FEATURES
# ==========================================

# ==========================================
# DBH APPROXIMATION
# ==========================================

# karena belum ada DBH asli,
# gunakan crown diameter proxy

if "crown_diameter" in df.columns:

    df["DBH"] = (
        df["crown_diameter"]
        * 0.5
    )

else:

    df["DBH"] = 20

# ==========================================
# HEIGHT
# ==========================================

if "max_height" in df.columns:

    df["Height"] = df["max_height"]

elif "mean_height" in df.columns:

    df["Height"] = df["mean_height"]

else:

    df["Height"] = 15

# ==========================================
# WOOD DENSITY
# ==========================================

# default tropical wood density

df["Wood_Density"] = 0.6

# ==========================================
# HANDLE INVALID
# ==========================================

df["DBH"] = df["DBH"].clip(lower=1)

df["Height"] = df["Height"].clip(lower=1)

# ==========================================
# ALLOMETRIC EQUATION
# ==========================================

agb = (
    0.0673
    *
    (
        df["Wood_Density"]
        *
        (df["DBH"] ** 2)
        *
        df["Height"]
    ) ** 0.976
)

# ==========================================
# SAVE BIOMASS
# ==========================================

df["AGB_kg"] = agb

# ==========================================
# CARBON
# ==========================================

df["Carbon_kg"] = (
    df["AGB_kg"]
    *
    0.47
)

# ==========================================
# CO2 EQUIVALENT
# ==========================================

df["CO2e_kg"] = (
    df["Carbon_kg"]
    *
    3.67
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("SUMMARY")
print("====================================")

print(f"\nMean Biomass : {df['AGB_kg'].mean():.2f} kg")

print(f"Max Biomass  : {df['AGB_kg'].max():.2f} kg")

print(f"Total Biomass: {df['AGB_kg'].sum():.2f} kg")

print(f"\nTotal Carbon : {df['Carbon_kg'].sum():.2f} kg")

print(f"Total CO2e   : {df['CO2e_kg'].sum():.2f} kg")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "biomass_allometric.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# TOP TREES
# ==========================================

top_trees = df.sort_values(

    by="AGB_kg",

    ascending=False
)

print("\n====================================")
print("TOP BIOMASS TREES")
print("====================================")

print(

    top_trees[
        [
            "tree_id",
            "AGB_kg",
            "Carbon_kg",
            "CO2e_kg"
        ]
    ].head()
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("BIOMASS ESTIMATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)