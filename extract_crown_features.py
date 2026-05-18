import cv2
import numpy as np
import pandas as pd
import rasterio

from pathlib import Path

# ==========================================
# INPUT DIRECTORIES
# ==========================================

MASK_DIR = Path("outputs/segmentation")

CHM_DIR = Path("outputs/chm")

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/crown_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD MASK FILES
# ==========================================

mask_files = list(
    MASK_DIR.glob("*_mask.png")
)

# ==========================================
# CHECK FILES
# ==========================================

if len(mask_files) == 0:

    print("[ERROR] No segmentation masks found!")
    exit()

# ==========================================
# RESULTS
# ==========================================

results = []

# ==========================================
# PROCESS EACH MASK
# ==========================================

for mask_file in mask_files:

    plot_name = mask_file.stem.replace("_mask", "")

    print("\n====================================")
    print(f"PROCESSING: {plot_name}")
    print("====================================")

    # ======================================
    # MATCH CHM FILE
    # ======================================

    chm_file = CHM_DIR / f"{plot_name}_CHM.tif"

    if not chm_file.exists():

        print("[WARNING] CHM missing")
        continue

    # ======================================
    # READ MASK
    # ======================================

    mask = cv2.imread(

        str(mask_file),

        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        print("[WARNING] Cannot read mask")
        continue

    # ======================================
    # BINARY MASK
    # ======================================

    binary = (mask > 0).astype(np.uint8)

    # ======================================
    # CONTOURS
    # ======================================

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        print("[WARNING] No contour found")
        continue

    # ======================================
    # LARGEST CONTOUR
    # ======================================

    contour = max(

        contours,

        key=cv2.contourArea
    )

    # ======================================
    # CROWN AREA
    # ======================================

    crown_area = cv2.contourArea(
        contour
    )

    # ======================================
    # PERIMETER
    # ======================================

    crown_perimeter = cv2.arcLength(

        contour,

        True
    )

    # ======================================
    # BOUNDING BOX
    # ======================================

    x, y, w, h = cv2.boundingRect(
        contour
    )

    crown_diameter = max(w, h)

    # ======================================
    # COMPACTNESS
    # ======================================

    if crown_perimeter > 0:

        compactness = (
            4
            *
            np.pi
            *
            crown_area
        ) / (crown_perimeter ** 2)

    else:

        compactness = 0

    # ======================================
    # READ CHM
    # ======================================

    with rasterio.open(chm_file) as src:

        chm = src.read(1)

    # ======================================
    # RESIZE CHM IF NEEDED
    # ======================================

    if chm.shape != binary.shape:

        chm = cv2.resize(

            chm,

            (
                binary.shape[1],
                binary.shape[0]
            ),

            interpolation=cv2.INTER_LINEAR
        )

    # ======================================
    # EXTRACT HEIGHT PIXELS
    # ======================================

    crown_heights = chm[binary == 1]

    # ======================================
    # HEIGHT FEATURES
    # ======================================

    if len(crown_heights) > 0:

        mean_height = np.mean(
            crown_heights
        )

        max_height = np.max(
            crown_heights
        )

        std_height = np.std(
            crown_heights
        )

        crown_density = np.sum(
            crown_heights > 2
        ) / len(crown_heights)

    else:

        mean_height = 0
        max_height = 0
        std_height = 0
        crown_density = 0

    # ======================================
    # SAVE RECORD
    # ======================================

    results.append({

        "plot": plot_name,

        "crown_area": crown_area,

        "crown_perimeter": crown_perimeter,

        "crown_diameter": crown_diameter,

        "compactness": compactness,

        "mean_height": mean_height,

        "max_height": max_height,

        "std_height": std_height,

        "crown_density": crown_density
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
    "crown_features.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("FEATURE EXTRACTION FINISHED")
print("====================================")

print(f"\nTotal Crowns:")
print(len(df))

print(f"\nSaved:")
print(output_csv)

print("\nPreview:")

print(df.head())