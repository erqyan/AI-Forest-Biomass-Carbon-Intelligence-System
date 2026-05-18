import laspy
import numpy as np
import pandas as pd

from pathlib import Path

# ==========================================
# INPUT DIRECTORY
# ==========================================

TREE_DIR = Path("outputs/tree_points")

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = Path("outputs/lidar_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD TREE FILES
# ==========================================

tree_files = list(
    TREE_DIR.glob("*.las")
)

# ==========================================
# CHECK FILES
# ==========================================

if len(tree_files) == 0:

    print("[ERROR] No tree point clouds found!")
    exit()

# ==========================================
# RESULTS
# ==========================================

results = []

# ==========================================
# PROCESS EACH TREE
# ==========================================

for tree_file in tree_files:

    print("\n====================================")
    print(f"PROCESSING: {tree_file.name}")
    print("====================================")

    # ======================================
    # READ LAS
    # ======================================

    las = laspy.read(tree_file)

    x = las.x
    y = las.y
    z = las.z

    # ======================================
    # EMPTY CHECK
    # ======================================

    if len(z) == 0:

        print("[WARNING] Empty point cloud")
        continue

    # ======================================
    # BASIC STATISTICS
    # ======================================

    max_height = np.max(z)

    mean_height = np.mean(z)

    std_height = np.std(z)

    min_height = np.min(z)

    point_count = len(z)

    # ======================================
    # HEIGHT PERCENTILES
    # ======================================

    p95 = np.percentile(z, 95)

    p75 = np.percentile(z, 75)

    p50 = np.percentile(z, 50)

    # ======================================
    # CANOPY DENSITY
    # ======================================

    canopy_points = z > 2

    canopy_density = (
        np.sum(canopy_points)
        /
        len(z)
    )

    # ======================================
    # HEIGHT RANGE
    # ======================================

    height_range = (
        max_height
        -
        min_height
    )

    # ======================================
    # CROWN AREA APPROXIMATION
    # ======================================

    crown_width = x.max() - x.min()

    crown_length = y.max() - y.min()

    crown_area = (
        crown_width
        *
        crown_length
    )

    # ======================================
    # CROWN VOLUME PROXY
    # ======================================

    crown_volume_proxy = (
        crown_area
        *
        mean_height
    )

    # ======================================
    # SAVE RECORD
    # ======================================

    results.append({

        "tree_file": tree_file.stem,

        "point_count": point_count,

        "max_height": max_height,

        "mean_height": mean_height,

        "std_height": std_height,

        "height_range": height_range,

        "p95_height": p95,

        "p75_height": p75,

        "median_height": p50,

        "canopy_density": canopy_density,

        "crown_width": crown_width,

        "crown_length": crown_length,

        "crown_area": crown_area,

        "crown_volume_proxy": crown_volume_proxy
    })

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(results)

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "lidar_features.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("LIDAR FEATURE EXTRACTION FINISHED")
print("====================================")

print(f"\nTotal Trees:")
print(len(df))

print(f"\nSaved:")
print(output_csv)

print("\nPreview:")

print(df.head())