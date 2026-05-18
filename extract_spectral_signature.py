import rasterio
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# INPUT DIRECTORIES
# ==========================================

HSI_DIR = Path("data_clean/Hyperspectral")

ANN_DIR = Path("data_clean/annotations")

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/spectral_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD FILES
# ==========================================

hsi_files = list(HSI_DIR.glob("*.tif"))

# ==========================================
# CHECK FILES
# ==========================================

if len(hsi_files) == 0:

    print("[ERROR] No hyperspectral files found!")
    exit()

# ==========================================
# RESULTS
# ==========================================

results = []

# ==========================================
# PROCESS EACH FILE
# ==========================================

for hsi_file in hsi_files:

    plot_name = hsi_file.stem

    print("\n====================================")
    print(f"PROCESSING: {plot_name}")
    print("====================================")

    # ======================================
    # XML FILE
    # ======================================

    xml_file = ANN_DIR / f"{plot_name}.xml"

    if not xml_file.exists():

        print("[WARNING] Annotation missing")
        continue

    # ======================================
    # READ HYPERSPECTRAL
    # ======================================

    with rasterio.open(hsi_file) as src:

        total_bands = src.count

        width = src.width
        height = src.height

        print(f"Total Bands: {total_bands}")

        # ==================================
        # READ ALL BANDS
        # ==================================

        hsi_cube = src.read().astype(np.float32)

    # ======================================
    # READ XML
    # ======================================

    tree = ET.parse(xml_file)

    root = tree.getroot()

    objects = root.findall("object")

    print(f"Total Trees: {len(objects)}")

    # ======================================
    # PROCESS EACH TREE
    # ======================================

    for idx, obj in enumerate(objects):

        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # ==================================
        # CLIP RANGE
        # ==================================

        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = min(width - 1, xmax)
        ymax = min(height - 1, ymax)

        # ==================================
        # EXTRACT TREE CUBE
        # ==================================

        tree_cube = hsi_cube[
            :,
            ymin:ymax,
            xmin:xmax
        ]

        # ==================================
        # EMPTY CHECK
        # ==================================

        if tree_cube.size == 0:

            continue

        # ==================================
        # MEAN SPECTRAL SIGNATURE
        # ==================================

        spectral_signature = np.mean(

            tree_cube,

            axis=(1, 2)
        )

        # ==================================
        # CREATE RECORD
        # ==================================

        record = {

            "plot": plot_name,

            "tree_id": idx + 1
        }

        # ==================================
        # SAVE EACH BAND
        # ==================================

        for band_idx, value in enumerate(
            spectral_signature
        ):

            record[
                f"band_{band_idx+1}"
            ] = float(value)

        results.append(record)

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
    "spectral_signatures.csv"
)

df.to_csv(
    output_csv,
    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("SPECTRAL EXTRACTION FINISHED")
print("====================================")

print(f"\nTotal Trees:")
print(len(df))

print(f"\nTotal Features:")
print(len(df.columns)-2)

print(f"\nSaved:")
print(output_csv)

print("\nPreview:")

print(df.head())