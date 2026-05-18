import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

# ==========================================
# INPUT / OUTPUT DIRECTORY
# ==========================================

LIDAR_DIR = Path("data_clean/LiDAR")

OUTPUT_DIR = Path("outputs/dtm")

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
# DTM PARAMETERS
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
    # CLASSIFICATION FILTER
    # ======================================

    # ASPRS:
    # 2 = ground

    if hasattr(las, "classification"):

        ground_mask = las.classification == 2

        if ground_mask.sum() > 0:

            print(f"Ground Points: {ground_mask.sum():,}")

            x = x[ground_mask]
            y = y[ground_mask]
            z = z[ground_mask]

        else:

            print("[WARNING] No classified ground points found")
            print("Using LOWEST point approach")

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
    # CREATE EMPTY DTM
    # ======================================

    dtm = np.full(
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
    # FILL DTM WITH MIN HEIGHT
    # ======================================

    for r, c, height_value in zip(row, col, z):

        if np.isnan(dtm[r, c]):

            dtm[r, c] = height_value

        else:

            dtm[r, c] = min(
                dtm[r, c],
                height_value
            )

    # ======================================
    # HANDLE EMPTY PIXELS
    # ======================================

    dtm = np.nan_to_num(
        dtm,
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

    output_file = OUTPUT_DIR / f"{lidar_file.stem}_DTM.tif"

    # ======================================
    # SAVE DTM
    # ======================================

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=dtm.shape[0],
        width=dtm.shape[1],
        count=1,
        dtype=dtm.dtype,
        crs="EPSG:32611",
        transform=transform
    ) as dst:

        dst.write(dtm, 1)

    # ======================================
    # DTM STATISTICS
    # ======================================

    print("\nDTM Statistics")

    print(f"MIN : {dtm.min():.2f}")
    print(f"MAX : {dtm.max():.2f}")
    print(f"MEAN: {dtm.mean():.2f}")

    print(f"\nDTM Saved:")
    print(output_file)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("DTM GENERATION FINISHED")
print("====================================")