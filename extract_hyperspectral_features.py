import rasterio
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path

from scipy.stats import entropy

# ==========================================
# INPUT DIRECTORIES
# ==========================================

HSI_DIR = Path("data_clean/Hyperspectral")

ANN_DIR = Path("data_clean/annotations")

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/hyperspectral_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD FILES
# ==========================================

hsi_files = list(
    HSI_DIR.glob("*.tif")
)

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

        cube = src.read().astype(np.float32)

        total_bands = src.count

        height = src.height
        width = src.width

    print(f"Total Bands: {total_bands}")

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

        tree_cube = cube[
            :,
            ymin:ymax,
            xmin:xmax
        ]

        if tree_cube.size == 0:

            continue

        # ==================================
        # FLATTEN
        # ==================================

        pixels = tree_cube.reshape(
            total_bands,
            -1
        )

        # ==================================
        # BASIC SPECTRAL FEATURES
        # ==================================

        mean_reflectance = np.mean(
            pixels
        )

        std_reflectance = np.std(
            pixels
        )

        min_reflectance = np.min(
            pixels
        )

        max_reflectance = np.max(
            pixels
        )

        # ==================================
        # BAND-WISE MEAN
        # ==================================

        spectral_signature = np.mean(

            pixels,

            axis=1
        )

        # ==================================
        # NDVI
        # ==================================

        RED_BAND = min(58, total_bands)

        NIR_BAND = min(90, total_bands)

        red = spectral_signature[
            RED_BAND - 1
        ]

        nir = spectral_signature[
            NIR_BAND - 1
        ]

        ndvi = (
            (nir - red)
            /
            (nir + red + 1e-6)
        )

        # ==================================
        # NDRE
        # ==================================

        RED_EDGE_BAND = min(75, total_bands)

        red_edge = spectral_signature[
            RED_EDGE_BAND - 1
        ]

        ndre = (
            (nir - red_edge)
            /
            (nir + red_edge + 1e-6)
        )

        # ==================================
        # SPECTRAL ENTROPY
        # ==================================

        normalized_signature = (
            spectral_signature
            /
            (
                spectral_signature.sum()
                + 1e-6
            )
        )

        spectral_entropy = entropy(
            normalized_signature
        )

        # ==================================
        # SPECTRAL ENERGY
        # ======================================

        spectral_energy = np.sum(

            spectral_signature ** 2
        )

        # ======================================
        # SAVE RECORD
        # ======================================

        record = {

            "plot": plot_name,

            "tree_id": idx + 1,

            "mean_reflectance": mean_reflectance,

            "std_reflectance": std_reflectance,

            "min_reflectance": min_reflectance,

            "max_reflectance": max_reflectance,

            "ndvi": ndvi,

            "ndre": ndre,

            "spectral_entropy": spectral_entropy,

            "spectral_energy": spectral_energy
        }

        # ======================================
        # SAVE IMPORTANT BANDS
        # ======================================

        important_bands = [

            10,
            30,
            50,
            70,
            90,
            110,
            130,
            150
        ]

        for band in important_bands:

            if band <= total_bands:

                record[
                    f"band_{band}"
                ] = spectral_signature[
                    band - 1
                ]

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
    "hyperspectral_features.csv"
)

df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("HYPERSPECTRAL FEATURE EXTRACTION")
print("====================================")

print(f"\nTotal Trees:")
print(len(df))

print(f"\nTotal Features:")
print(len(df.columns)-2)

print(f"\nSaved:")
print(output_csv)

print("\nPreview:")

print(df.head())