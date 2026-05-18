import shutil
from pathlib import Path
import random

# ==========================================
# SOURCE DIRECTORIES
# ==========================================

BASE_DIR = Path(".")

RGB_DIR = BASE_DIR / "evaluation" / "RGB"
LIDAR_DIR = BASE_DIR / "evaluation" / "LiDAR"
HSI_DIR = BASE_DIR / "evaluation" / "Hyperspectral"
ANNOTATION_DIR = BASE_DIR / "annotations"

# ==========================================
# OUTPUT CLEAN DATASET
# ==========================================

CLEAN_DIR = BASE_DIR / "data_clean"

CLEAN_RGB = CLEAN_DIR / "RGB"
CLEAN_LIDAR = CLEAN_DIR / "LiDAR"
CLEAN_HSI = CLEAN_DIR / "Hyperspectral"
CLEAN_ANN = CLEAN_DIR / "annotations"

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for folder in [
    CLEAN_RGB,
    CLEAN_LIDAR,
    CLEAN_HSI,
    CLEAN_ANN
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# FIND VALID PLOTS
# ==========================================

print("\n====================================")
print("SEARCHING VALID DATA")
print("====================================")

valid_plots = []

rgb_files = list(RGB_DIR.glob("*.tif"))

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

    # ======================================
    # LIDAR CHECK
    # ======================================

    lidar_las = LIDAR_DIR / f"{plot_name}.las"
    lidar_laz = LIDAR_DIR / f"{plot_name}.laz"

    lidar_exists = (
        lidar_las.exists()
        or
        lidar_laz.exists()
    )

    # ======================================
    # HYPERSPECTRAL CHECK
    # ======================================

    # contoh:
    # 2018_SJER_3_252000_4104000_image_628_hyperspectral.tif

    hsi_pattern = f"{plot_name}_hyperspectral.tif"

    hsi_file = HSI_DIR / hsi_pattern

    hsi_exists = hsi_file.exists()

    # ======================================
    # ANNOTATION CHECK
    # ======================================

    xml_file = ANNOTATION_DIR / f"{plot_name}.xml"

    annotation_exists = xml_file.exists()

    # ======================================
    # VALIDATION
    # ======================================

    if (
        lidar_exists
        and
        hsi_exists
        and
        annotation_exists
    ):

        valid_plots.append(plot_name)

# ==========================================
# SUMMARY
# ==========================================

print(f"\nValid plots found: {len(valid_plots)}")

if len(valid_plots) == 0:

    print("\n[ERROR] No valid plots found!")
    exit()

# ==========================================
# RANDOM SELECT
# ==========================================

TOTAL_SAMPLE = min(100, len(valid_plots))

selected_plots = random.sample(
    valid_plots,
    TOTAL_SAMPLE
)

print(f"\nSelected {TOTAL_SAMPLE} plots")

# ==========================================
# COPY FILES
# ==========================================

print("\n====================================")
print("COPYING FILES")
print("====================================")

for plot_name in selected_plots:

    print(f"\nCopying: {plot_name}")

    # ======================================
    # COPY RGB
    # ======================================

    shutil.copy2(
        RGB_DIR / f"{plot_name}.tif",
        CLEAN_RGB / f"{plot_name}.tif"
    )

    # ======================================
    # COPY LIDAR
    # ======================================

    lidar_las = LIDAR_DIR / f"{plot_name}.las"
    lidar_laz = LIDAR_DIR / f"{plot_name}.laz"

    if lidar_las.exists():

        shutil.copy2(
            lidar_las,
            CLEAN_LIDAR / f"{plot_name}.las"
        )

    elif lidar_laz.exists():

        shutil.copy2(
            lidar_laz,
            CLEAN_LIDAR / f"{plot_name}.laz"
        )

    # ======================================
    # COPY HYPERSPECTRAL
    # ======================================

    original_hsi = HSI_DIR / f"{plot_name}_hyperspectral.tif"

    # rename tanpa kata hyperspectral
    clean_hsi_name = f"{plot_name}.tif"

    shutil.copy2(
        original_hsi,
        CLEAN_HSI / clean_hsi_name
    )

    # ======================================
    # COPY ANNOTATION
    # ======================================

    shutil.copy2(
        ANNOTATION_DIR / f"{plot_name}.xml",
        CLEAN_ANN / f"{plot_name}.xml"
    )

# ==========================================
# SAVE PLOT LIST
# ==========================================

with open(
    CLEAN_DIR / "selected_plots.txt",
    "w"
) as f:

    for item in selected_plots:

        f.write(item + "\n")

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("DATA CLEAN FINISHED")
print("====================================")

print(f"\nSaved to: {CLEAN_DIR}")
print(f"Total copied: {TOTAL_SAMPLE}")