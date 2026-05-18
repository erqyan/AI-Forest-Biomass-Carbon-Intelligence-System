import pandas as pd
import numpy as np
import joblib

from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================
# MODEL FILE
# ==========================================

MODEL_FILE = (
    "models/biomass/"
    "best_biomass_model.pkl"
)

# ==========================================
# INPUT FEATURES
# ==========================================

INPUT_CSV = (
    "outputs/ml_table/"
    "forest_ml_features.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/biomass_prediction")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILES
# ==========================================

if not Path(MODEL_FILE).exists():

    print("[ERROR] Biomass model not found!")
    exit()

if not Path(INPUT_CSV).exists():

    print("[ERROR] Feature table not found!")
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print("\n====================================")
print("LOADING MODEL")
print("====================================")

model = joblib.load(MODEL_FILE)

print("\nModel Loaded")

# ==========================================
# LOAD FEATURES
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("BIOMASS PREDICTION")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# FEATURE LIST
# ==========================================

candidate_features = [

    "crown_area",
    "crown_density",
    "mean_height",
    "max_height",
    "std_height",

    "canopy_density",

    "ndvi",
    "ndre",

    "spectral_entropy",
    "spectral_energy"
]

# ==========================================
# AVAILABLE FEATURES
# ==========================================

features = [

    f for f in candidate_features

    if f in df.columns
]

print("\nUsing Features:")

for f in features:

    print(f"✔ {f}")

# ==========================================
# FEATURE MATRIX
# ==========================================

X = df[features]

X = X.fillna(0)

# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X)

# ==========================================
# SAVE PREDICTION
# ==========================================

df["Predicted_AGB_kg"] = predictions

# ==========================================
# CARBON
# ==========================================

df["Predicted_Carbon_kg"] = (

    df["Predicted_AGB_kg"]
    *
    0.47
)

# ==========================================
# CO2 EQUIVALENT
# ==========================================

df["Predicted_CO2e_kg"] = (

    df["Predicted_Carbon_kg"]
    *
    3.67
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("PREDICTION SUMMARY")
print("====================================")

print(f"\nMean Biomass : {df['Predicted_AGB_kg'].mean():.2f} kg")

print(f"Max Biomass  : {df['Predicted_AGB_kg'].max():.2f} kg")

print(f"Min Biomass  : {df['Predicted_AGB_kg'].min():.2f} kg")

print(f"\nTotal Biomass:")
print(f"{df['Predicted_AGB_kg'].sum():.2f} kg")

print(f"\nTotal Carbon:")
print(f"{df['Predicted_Carbon_kg'].sum():.2f} kg")

print(f"\nTotal CO2e:")
print(f"{df['Predicted_CO2e_kg'].sum():.2f} kg")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "predicted_biomass.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 6))

plt.hist(

    df["Predicted_AGB_kg"],

    bins=30
)

plt.xlabel("Predicted Biomass (kg)")

plt.ylabel("Frequency")

plt.title("Biomass Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# TOP BIOMASS TREES
# ==========================================

top_trees = df.sort_values(

    by="Predicted_AGB_kg",

    ascending=False
)

print("\n====================================")
print("TOP BIOMASS TREES")
print("====================================")

columns_to_show = [

    c for c in [

        "tree_id",
        "Predicted_AGB_kg",
        "Predicted_Carbon_kg",
        "Predicted_CO2e_kg"
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
print("BIOMASS PREDICTION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)