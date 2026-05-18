import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path

# ==========================================
# INPUT / OUTPUT DIRECTORY
# ==========================================

DSM_DIR = Path("outputs/dsm")
DTM_DIR = Path("outputs/dtm")

OUTPUT_DIR = Path("outputs/chm")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD DSM FILES
# ==========================================

dsm_files = list(DSM_DIR.glob("*_DSM.tif"))

# ==========================================
# CHECK FILES
# ==========================================

if len(dsm_files) == 0:

    print("[ERROR] No DSM files found!")
    exit()

# ==========================================
# PROCESS EACH DSM
# ==========================================

for dsm_file in dsm_files:

    plot_name = dsm_file.stem.replace("_DSM", "")

    print("\n====================================")
    print(f"PROCESSING: {plot_name}")
    print("====================================")

    # ======================================
    # MATCH DTM FILE
    # ======================================

    dtm_file = DTM_DIR / f"{plot_name}_DTM.tif"

    if not dtm_file.exists():

        print("[WARNING] DTM not found")
        continue

    # ======================================
    # READ DSM
    # ======================================

    with rasterio.open(dsm_file) as dsm_src:

        dsm = dsm_src.read(1).astype(np.float32)

        dsm_profile = dsm_src.profile

        dsm_transform = dsm_src.transform

        dsm_crs = dsm_src.crs

        dsm_height = dsm_src.height
        dsm_width = dsm_src.width

    # ======================================
    # READ DTM
    # ======================================

    with rasterio.open(dtm_file) as dtm_src:

        dtm = dtm_src.read(1).astype(np.float32)

        dtm_transform = dtm_src.transform

        dtm_crs = dtm_src.crs

    # ======================================
    # CREATE REPROJECTED DTM
    # ======================================

    dtm_resampled = np.empty(
        (dsm_height, dsm_width),
        dtype=np.float32
    )

    # ======================================
    # HANDLE CRS MISSING
    # ======================================

    if dsm_crs is None:

        dsm_crs = "EPSG:32611"

    if dtm_crs is None:

        dtm_crs = "EPSG:32611"

    # ======================================
    # RESAMPLE DTM TO DSM
    # ======================================

    reproject(
        source=dtm,
        destination=dtm_resampled,

        src_transform=dtm_transform,
        src_crs=dtm_crs,

        dst_transform=dsm_transform,
        dst_crs=dsm_crs,

        resampling=Resampling.bilinear
    )

    # ======================================
    # GENERATE CHM
    # ======================================

    chm = dsm - dtm_resampled

    # ======================================
    # REMOVE NEGATIVE VALUE
    # ======================================

    chm[chm < 0] = 0

    # ======================================
    # HANDLE NAN
    # ======================================

    chm = np.nan_to_num(
        chm,
        nan=0
    )

    # ======================================
    # CHM STATISTICS
    # ======================================

    print("\nCHM Statistics")

    print(f"MIN : {chm.min():.2f}")
    print(f"MAX : {chm.max():.2f}")
    print(f"MEAN: {chm.mean():.2f}")

    # ======================================
    # OUTPUT FILE
    # ======================================

    output_file = OUTPUT_DIR / f"{plot_name}_CHM.tif"

    # ======================================
    # SAVE CHM
    # ======================================

    dsm_profile.update(
        dtype=rasterio.float32,
        count=1
    )

    with rasterio.open(
        output_file,
        "w",
        **dsm_profile
    ) as dst:

        dst.write(
            chm.astype(rasterio.float32),
            1
        )

    print(f"\nCHM Saved:")
    print(output_file)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("CHM GENERATION FINISHED")
print("====================================")