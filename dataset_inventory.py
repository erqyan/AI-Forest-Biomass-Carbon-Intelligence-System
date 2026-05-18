import pandas as pd
from pathlib import Path

# ==========================================
# BASE DIRECTORY
# ==========================================

BASE_DIR = Path(".")

# ==========================================
# DATASET DIRECTORIES
# ==========================================

RGB_DIR = BASE_DIR / "evaluation" / "RGB"
LIDAR_DIR = BASE_DIR / "evaluation" / "LiDAR"
HSI_DIR = BASE_DIR / "evaluation" / "Hyperspectral"
CHM_DIR = BASE_DIR / "evaluation" / "CHM"
ANNOTATION_DIR = BASE_DIR / "annotations"

# ==========================================
# LOAD RGB FILES
# ==========================================

rgb_files = list(RGB_DIR.glob("*.tif"))

# ==========================================
# DATASET SUMMARY
# ==========================================

results = []

print("\n====================================")
print("SCANNING DATASET")
print("====================================")

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

    print(f"Checking: {plot_name}")

    # ======================================
    # RGB
    # ======================================

    rgb_exists = True

    # ======================================
    # LIDAR
    # ======================================

    lidar_las = LIDAR_DIR / f"{plot_name}.las"
    lidar_laz = LIDAR_DIR / f"{plot_name}.laz"

    lidar_exists = (
        lidar_las.exists()
        or
        lidar_laz.exists()
    )

    lidar_type = None

    if lidar_las.exists():

        lidar_type = ".las"

    elif lidar_laz.exists():

        lidar_type = ".laz"

    # ======================================
    # HYPERSPECTRAL
    # ======================================

    hsi_file = HSI_DIR / f"{plot_name}.tif"

    hsi_exists = hsi_file.exists()

    # ======================================
    # CHM
    # ======================================

    chm_file = CHM_DIR / f"{plot_name}_CHM.tif"

    chm_exists = chm_file.exists()

    # ======================================
    # ANNOTATION
    # ======================================

    xml_file = ANNOTATION_DIR / f"{plot_name}.xml"

    annotation_exists = xml_file.exists()

    # ======================================
    # FILE SIZE
    # ======================================

    rgb_size_mb = round(
        rgb_file.stat().st_size / (1024 * 1024),
        2
    )

    # ======================================
    # STORE RESULT
    # ======================================

    results.append({

        "plot_name": plot_name,

        "rgb_exists": rgb_exists,

        "lidar_exists": lidar_exists,

        "lidar_type": lidar_type,

        "hyperspectral_exists": hsi_exists,

        "chm_exists": chm_exists,

        "annotation_exists": annotation_exists,

        "rgb_size_mb": rgb_size_mb
    })

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(results)

# ==========================================
# SAVE CSV
# ==========================================

output_csv = "dataset_inventory.csv"

df.to_csv(output_csv, index=False)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("DATASET SUMMARY")
print("====================================")

print(f"Total RGB Files        : {len(df)}")
print(f"LiDAR Available        : {df['lidar_exists'].sum()}")
print(f"Hyperspectral Available: {df['hyperspectral_exists'].sum()}")
print(f"CHM Available          : {df['chm_exists'].sum()}")
print(f"Annotations Available  : {df['annotation_exists'].sum()}")

print("\n====================================")
print("CSV SAVED")
print("====================================")

print(f"\nSaved to: {output_csv}")

# ==========================================
# SHOW SAMPLE
# ==========================================

print("\n====================================")
print("SAMPLE DATA")
print("====================================")

print(df.head())