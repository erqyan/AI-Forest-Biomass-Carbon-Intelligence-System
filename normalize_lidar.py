import laspy
import rasterio
import numpy as np
from pathlib import Path

# ==========================================
# INPUT / OUTPUT
# ==========================================

LIDAR_DIR = Path("data_clean/LiDAR")
DTM_DIR = Path("outputs/dtm")

OUTPUT_DIR = Path("outputs/normalized_lidar")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD FILES
# ==========================================

las_files = list(LIDAR_DIR.glob("*.las"))
laz_files = list(LIDAR_DIR.glob("*.laz"))

lidar_files = las_files + laz_files

# ==========================================
# PROCESS EACH FILE
# ==========================================

for lidar_file in lidar_files:

    plot_name = lidar_file.stem

    print("\n====================================")
    print(f"NORMALIZING: {plot_name}")
    print("====================================")

    # ======================================
    # DTM FILE
    # ======================================

    dtm_file = DTM_DIR / f"{plot_name}_DTM.tif"

    if not dtm_file.exists():

        print("[WARNING] DTM not found")
        continue

    # ======================================
    # READ LIDAR
    # ======================================

    las = laspy.read(lidar_file)

    x = las.x
    y = las.y
    z = las.z

    # ======================================
    # READ DTM
    # ======================================

    with rasterio.open(dtm_file) as dtm_src:

        dtm = dtm_src.read(1)

        transform = dtm_src.transform

    # ======================================
    # SAMPLE GROUND HEIGHT
    # ======================================

    rows, cols = rasterio.transform.rowcol(
        transform,
        x,
        y
    )

    rows = np.clip(rows, 0, dtm.shape[0]-1)
    cols = np.clip(cols, 0, dtm.shape[1]-1)

    ground_height = dtm[rows, cols]

    # ======================================
    # NORMALIZE HEIGHT
    # ======================================

    normalized_z = z - ground_height

    normalized_z[normalized_z < 0] = 0

    # ======================================
    # UPDATE LAS
    # ======================================

    las.z = normalized_z

    # ======================================
    # SAVE
    # ======================================

    output_file = OUTPUT_DIR / f"{plot_name}_normalized.las"

    las.write(output_file)

    # ======================================
    # STATISTICS
    # ======================================

    print("\nNormalized Height Statistics")

    print(f"MIN : {normalized_z.min():.2f}")
    print(f"MAX : {normalized_z.max():.2f}")
    print(f"MEAN: {normalized_z.mean():.2f}")

    print(f"\nSaved:")
    print(output_file)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("LIDAR NORMALIZATION FINISHED")
print("====================================")