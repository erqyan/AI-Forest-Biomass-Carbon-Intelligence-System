import pandas as pd
from pathlib import Path

# ==========================================
# INPUT FILES
# ==========================================

CROWN_FILE = (
    "outputs/crown_features/"
    "crown_features.csv"
)

LIDAR_FILE = (
    "outputs/lidar_features/"
    "lidar_features.csv"
)

HSI_FILE = (
    "outputs/hyperspectral_features/"
    "hyperspectral_features.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/ml_table")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILES
# ==========================================

required_files = [

    CROWN_FILE,
    LIDAR_FILE,
    HSI_FILE
]

for f in required_files:

    if not Path(f).exists():

        print(f"[ERROR] Missing file: {f}")
        exit()

# ==========================================
# LOAD DATA
# ==========================================

print("\n====================================")
print("BUILDING FEATURE TABLE")
print("====================================")

crown_df = pd.read_csv(CROWN_FILE)

lidar_df = pd.read_csv(LIDAR_FILE)

hsi_df = pd.read_csv(HSI_FILE)

# ==========================================
# INFO
# ==========================================

print("\nLoaded Features:")

print(f"Crown Features : {len(crown_df)}")

print(f"LiDAR Features : {len(lidar_df)}")

print(f"HSI Features   : {len(hsi_df)}")

# ==========================================
# CREATE TREE ID
# ==========================================

# Crown Feature
crown_df["tree_id"] = (
    crown_df.index + 1
)

# LiDAR Feature
lidar_df["tree_id"] = (
    lidar_df.index + 1
)

# ==========================================
# MERGE CROWN + LIDAR
# ==========================================

merged_df = pd.merge(

    crown_df,

    lidar_df,

    on="tree_id",

    how="inner",

    suffixes=(
        "_crown",
        "_lidar"
    )
)

# ==========================================
# MERGE HSI
# ==========================================

if "tree_id" in hsi_df.columns:

    merged_df = pd.merge(

        merged_df,

        hsi_df,

        on="tree_id",

        how="inner"
    )

else:

    print("\n[WARNING] HSI tree_id missing")

# ==========================================
# REMOVE DUPLICATES
# ==========================================

merged_df = merged_df.loc[
    :,
    ~merged_df.columns.duplicated()
]

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

merged_df = merged_df.fillna(0)

# ==========================================
# FEATURE SUMMARY
# ==========================================

print("\n====================================")
print("FEATURE SUMMARY")
print("====================================")

print(f"\nTotal Samples:")
print(len(merged_df))

print(f"\nTotal Features:")
print(len(merged_df.columns))

# ==========================================
# FEATURE GROUPS
# ==========================================

crown_features = [

    c for c in merged_df.columns

    if "crown" in c
]

lidar_features = [

    c for c in merged_df.columns

    if (
        "height" in c
        or
        "density" in c
        or
        "point" in c
    )
]

spectral_features = [

    c for c in merged_df.columns

    if (
        "band_" in c
        or
        "ndvi" in c
        or
        "ndre" in c
        or
        "spectral" in c
    )
]

print(f"\nCrown Features : {len(crown_features)}")

print(f"LiDAR Features : {len(lidar_features)}")

print(f"Spectral Features : {len(spectral_features)}")

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "forest_ml_features.csv"
)

merged_df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# SAVE FEATURE LIST
# ==========================================

feature_list = pd.DataFrame({

    "feature_name": merged_df.columns
})

feature_list.to_csv(

    OUTPUT_DIR / "feature_columns.csv",

    index=False
)

# ==========================================
# PREVIEW
# ==========================================

print("\n====================================")
print("DATA PREVIEW")
print("====================================")

print(merged_df.head())

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("FEATURE TABLE FINISHED")
print("====================================")

print(f"\nSaved:")

print(output_csv)