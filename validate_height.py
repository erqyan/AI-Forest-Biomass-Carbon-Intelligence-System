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
    "outputs/"
    "tree_height_comparison.csv"
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] tree_height_comparison.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("HEIGHT VALIDATION")
print("====================================")

print(f"\nTotal Records: {len(df)}")

# ==========================================
# REMOVE MISSING
# ==========================================

df = df.dropna(

    subset=[
        "field_height",
        "lidar_height"
    ]
)

print(f"\nValid Samples: {len(df)}")

# ==========================================
# CHECK DATA
# ==========================================

if len(df) == 0:

    print("[ERROR] No valid samples!")
    exit()

# ==========================================
# TRUE / PRED
# ==========================================

y_true = df["field_height"]

y_pred = df["lidar_height"]

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
# SUMMARY
# ==========================================

print("\n====================================")
print("VALIDATION RESULT")
print("====================================")

print(f"\nRMSE : {rmse:.2f} meter")

print(f"MAE  : {mae:.2f} meter")

print(f"R²   : {r2:.4f}")

print(f"Bias : {bias:.2f} meter")

# ==========================================
# INTERPRETATION
# ==========================================

print("\n====================================")
print("INTERPRETATION")
print("====================================")

if rmse < 2:

    quality = "Excellent"

elif rmse < 5:

    quality = "Good"

elif rmse < 10:

    quality = "Moderate"

else:

    quality = "Poor"

print(f"\nLiDAR Height Quality: {quality}")

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

plt.xlabel("Field Height (m)")

plt.ylabel("LiDAR Height (m)")

plt.title(

    f"Height Validation\nRMSE = {rmse:.2f} m"
)

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# ERROR ANALYSIS
# ==========================================

df["error"] = (

    df["lidar_height"]
    -
    df["field_height"]
)

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
    "height_validation.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# ERROR HISTOGRAM
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(

    df["error"],

    bins=30
)

plt.xlabel("Height Error (m)")

plt.ylabel("Frequency")

plt.title("LiDAR Height Error Distribution")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("HEIGHT VALIDATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)