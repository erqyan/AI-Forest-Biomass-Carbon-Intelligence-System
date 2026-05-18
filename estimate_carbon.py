import pandas as pd
from pathlib import Path

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/biomass_prediction/"
    "predicted_biomass.csv"
)

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = Path("outputs/carbon")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] predicted_biomass.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("CARBON ESTIMATION")
print("====================================")

print(f"\nTotal Trees: {len(df)}")

# ==========================================
# CHECK BIOMASS COLUMN
# ==========================================

if "Predicted_AGB_kg" not in df.columns:

    print("[ERROR] Predicted_AGB_kg column missing!")
    exit()

# ==========================================
# CARBON ESTIMATION
# ==========================================

df["Carbon_kg"] = (

    df["Predicted_AGB_kg"]
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
# TON CONVERSION
# ==========================================

df["Carbon_ton"] = (

    df["Carbon_kg"]
    / 1000
)

df["CO2e_ton"] = (

    df["CO2e_kg"]
    / 1000
)

# ==========================================
# SUMMARY
# ==========================================

total_biomass = df["Predicted_AGB_kg"].sum()

total_carbon = df["Carbon_kg"].sum()

total_co2e = df["CO2e_kg"].sum()

mean_carbon = df["Carbon_kg"].mean()

max_carbon = df["Carbon_kg"].max()

print("\n====================================")
print("CARBON SUMMARY")
print("====================================")

print(f"\nTotal Biomass : {total_biomass:.2f} kg")

print(f"Total Carbon  : {total_carbon:.2f} kg")

print(f"Total CO2e    : {total_co2e:.2f} kg")

print(f"\nMean Carbon   : {mean_carbon:.2f} kg/tree")

print(f"Max Carbon    : {max_carbon:.2f} kg/tree")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "carbon_estimation.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# TOP CARBON TREES
# ==========================================

top_trees = df.sort_values(

    by="Carbon_kg",

    ascending=False
)

print("\n====================================")
print("TOP CARBON TREES")
print("====================================")

columns_to_show = [

    c for c in [

        "tree_id",
        "Predicted_AGB_kg",
        "Carbon_kg",
        "CO2e_kg"
    ]

    if c in top_trees.columns
]

print(

    top_trees[
        columns_to_show
    ].head()
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("CARBON ESTIMATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)