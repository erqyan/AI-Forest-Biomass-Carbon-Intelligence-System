import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    r2_score
)

import matplotlib.pyplot as plt

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/biomass_prediction/"
    "predicted_biomass.csv"
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
print("BIOMASS VALIDATION")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_cols = [

    "AGB_kg",

    "Predicted_AGB_kg"
]

for col in required_cols:

    if col not in df.columns:

        print(f"[ERROR] Missing column: {col}")
        exit()

# ==========================================
# REMOVE INVALID
# ==========================================

df = df.dropna(

    subset=[
        "AGB_kg",
        "Predicted_AGB_kg"
    ]
)

print(f"\nValid Samples: {len(df)}")

# ==========================================
# TRUE / PRED
# ==========================================

y_true = df["AGB_kg"]

y_pred = df["Predicted_AGB_kg"]

# ==========================================
# METRICS
# ==========================================

rmse = np.sqrt(

    mean_squared_error(
        y_true,
        y_pred
    )
)

mae = mean_absolute_error(

    y_true,
    y_pred
)

r2 = r2_score(

    y_true,
    y_pred
)

bias = np.mean(

    y_pred - y_true
)

# ==========================================
# MAPE
# ==========================================

mape = np.mean(

    np.abs(
        (
            y_true - y_pred
        )
        /
        (
            y_true + 1e-6
        )
    )
) * 100

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("VALIDATION RESULT")
print("====================================")

print(f"\nRMSE : {rmse:.2f} kg")

print(f"MAE  : {mae:.2f} kg")

print(f"MAPE : {mape:.2f} %")

print(f"R²   : {r2:.4f}")

print(f"Bias : {bias:.2f} kg")

# ==========================================
# INTERPRETATION
# ==========================================

print("\n====================================")
print("MODEL QUALITY")
print("====================================")

if r2 > 0.9:

    quality = "Excellent"

elif r2 > 0.75:

    quality = "Very Good"

elif r2 > 0.6:

    quality = "Good"

elif r2 > 0.4:

    quality = "Moderate"

else:

    quality = "Poor"

print(f"\nBiomass Model Quality: {quality}")

# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8, 8))

plt.scatter(

    y_true,
    y_pred,

    alpha=0.7
)

# ==========================================
# 1:1 LINE
# ==========================================

min_val = min(
    y_true.min(),
    y_pred.min()
)

max_val = max(
    y_true.max(),
    y_pred.max()
)

plt.plot(

    [min_val, max_val],

    [min_val, max_val],

    linestyle="--"
)

# ==========================================
# LABELS
# ==========================================

plt.xlabel("Reference Biomass (kg)")

plt.ylabel("Predicted Biomass (kg)")

plt.title(

    f"Biomass Validation\nR² = {r2:.4f}"
)

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# ERROR DISTRIBUTION
# ==========================================

df["error"] = (

    df["Predicted_AGB_kg"]
    -
    df["AGB_kg"]
)

plt.figure(figsize=(10, 5))

plt.hist(

    df["error"],

    bins=30
)

plt.xlabel("Biomass Error (kg)")

plt.ylabel("Frequency")

plt.title("Biomass Error Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# SAVE RESULT
# ==========================================

OUTPUT_DIR = Path("outputs/validation")

OUTPUT_DIR.mkdir(

    parents=True,
    exist_ok=True
)

output_csv = (
    OUTPUT_DIR
    /
    "biomass_validation.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# SAVE METRICS
# ==========================================

metrics_df = pd.DataFrame({

    "metric": [

        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Bias"
    ],

    "value": [

        rmse,
        mae,
        mape,
        r2,
        bias
    ]
})

metrics_csv = (
    OUTPUT_DIR
    /
    "biomass_metrics.csv"
)

metrics_df.to_csv(

    metrics_csv,

    index=False
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("BIOMASS VALIDATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)

print(metrics_csv)