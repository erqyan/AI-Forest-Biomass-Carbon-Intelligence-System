import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

# ==========================================
# INPUT / OUTPUT DIRECTORY
# ==========================================

LIDAR_DIR = Path("data_clean/LiDAR")

OUTPUT_DIR = Path("outputs/dsm")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD LIDAR FILES
# ==========================================

las_files = list(LIDAR_DIR.glob("*.las"))
laz_files = list(LIDAR_DIR.glob("*.laz"))

lidar_files = las_files + laz_files

# ==========================================
# CHECK FILES
# ==========================================

if len(lidar_files) == 0:

    print("[ERROR] No LiDAR files found!")
    exit()

# ==========================================
# DSM PARAMETERS
# ==========================================

RESOLUTION = 1.0  # meter

# ==========================================
# PROCESS EACH FILE
# ==========================================

for lidar_file in lidar_files:

    print("\n====================================")
    print(f"PROCESSING: {lidar_file.name}")
    print("====================================")

    # ======================================
    # READ LIDAR
    # ======================================

    las = laspy.read(lidar_file)

    x = las.x
    y = las.y
    z = las.z

    print(f"Total Points: {len(x):,}")

    # ======================================
    # SPATIAL EXTENT
    # ======================================

    xmin = x.min()
    xmax = x.max()

    ymin = y.min()
    ymax = y.max()

    # ======================================
    # GRID SIZE
    # ======================================

    width = int((xmax - xmin) / RESOLUTION) + 1
    height = int((ymax - ymin) / RESOLUTION) + 1

    print(f"Raster Size: {width} x {height}")

    # ======================================
    # CREATE EMPTY DSM
    # ======================================

    dsm = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    # ======================================
    # POINT TO GRID
    # ======================================

    col = ((x - xmin) / RESOLUTION).astype(int)

    row = ((ymax - y) / RESOLUTION).astype(int)

    # ======================================
    # FILL DSM WITH MAX HEIGHT
    # ======================================

    for r, c, height_value in zip(row, col, z):

        if np.isnan(dsm[r, c]):

            dsm[r, c] = height_value

        else:

            dsm[r, c] = max(
                dsm[r, c],
                height_value
            )

    # ======================================
    # HANDLE EMPTY PIXELS
    # ======================================

    dsm = np.nan_to_num(
        dsm,
        nan=0
    )

    # ======================================
    # TRANSFORM
    # ======================================

    transform = from_origin(
        xmin,
        ymax,
        RESOLUTION,
        RESOLUTION
    )

    # ======================================
    # OUTPUT FILE
    # ======================================

    output_file = OUTPUT_DIR / f"{lidar_file.stem}_DSM.tif"

    # ======================================
    # SAVE DSM
    # ======================================

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=dsm.shape[0],
        width=dsm.shape[1],
        count=1,
        dtype=dsm.dtype,
        crs="EPSG:32611",
        transform=transform
    ) as dst:

        dst.write(dsm, 1)

    # ======================================
    # DSM STATISTICS
    # ======================================

    print("\nDSM Statistics")

    print(f"MIN : {dsm.min():.2f}")
    print(f"MAX : {dsm.max():.2f}")
    print(f"MEAN: {dsm.mean():.2f}")

    print(f"\nDSM Saved:")
    print(output_file)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("DSM GENERATION FINISHED")
print("====================================")