import pandas as pd
from pathlib import Path

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/carbon/"
    "carbon_estimation.csv"
)

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = Path("outputs/co2e")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] carbon_estimation.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("CO2 EQUIVALENT ESTIMATION")
print("====================================")

print(f"\nTotal Trees: {len(df)}")

# ==========================================
# CHECK CARBON COLUMN
# ==========================================

if "Carbon_kg" not in df.columns:

    print("[ERROR] Carbon_kg column missing!")
    exit()

# ==========================================
# CO2E ESTIMATION
# ==========================================

df["CO2e_kg"] = (

    df["Carbon_kg"]
    *
    3.67
)

# ==========================================
# TON CONVERSION
# ==========================================

df["CO2e_ton"] = (

    df["CO2e_kg"]
    / 1000
)

# ==========================================
# SUMMARY
# ==========================================

total_carbon = df["Carbon_kg"].sum()

total_co2e = df["CO2e_kg"].sum()

mean_co2e = df["CO2e_kg"].mean()

max_co2e = df["CO2e_kg"].max()

print("\n====================================")
print("CO2e SUMMARY")
print("====================================")

print(f"\nTotal Carbon : {total_carbon:.2f} kg")

print(f"Total CO2e   : {total_co2e:.2f} kg")

print(f"\nMean CO2e    : {mean_co2e:.2f} kg/tree")

print(f"Max CO2e     : {max_co2e:.2f} kg/tree")

print(f"\nTotal CO2e   : {total_co2e/1000:.2f} ton")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "co2e_estimation.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# TOP CO2 TREES
# ==========================================

top_trees = df.sort_values(

    by="CO2e_kg",

    ascending=False
)

print("\n====================================")
print("TOP CO2 TREES")
print("====================================")

columns_to_show = [

    c for c in [

        "tree_id",
        "Carbon_kg",
        "CO2e_kg",
        "CO2e_ton"
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
print("CO2e ESTIMATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)