import rasterio
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================
# INPUT
# ==========================================

RGB_DIR = Path("data_clean/RGB")

ANN_DIR = Path("data_clean/annotations")

BIOMASS_FILE = (
    "outputs/biomass_prediction/"
    "predicted_biomass.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/biomass_map")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(BIOMASS_FILE).exists():

    print("[ERROR] predicted_biomass.csv not found!")
    exit()

# ==========================================
# LOAD BIOMASS
# ==========================================

biomass_df = pd.read_csv(BIOMASS_FILE)

print("\n====================================")
print("GENERATE BIOMASS MAP")
print("====================================")

print(f"\nTotal Records: {len(biomass_df)}")

# ==========================================
# RGB FILES
# ==========================================

rgb_files = list(
    RGB_DIR.glob("*.tif")
)

# ==========================================
# PROCESS EACH PLOT
# ==========================================

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

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
    # OPEN RGB
    # ======================================

    with rasterio.open(rgb_file) as src:

        profile = src.profile

        width = src.width

        height = src.height

    # ======================================
    # CREATE EMPTY MAP
    # ======================================

    biomass_map = np.zeros(

        (height, width),

        dtype=np.float32
    )

    # ======================================
    # READ XML
    # ======================================

    tree = ET.parse(xml_file)

    root = tree.getroot()

    objects = root.findall("object")

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
        # BIOMASS VALUE
        # ==================================

        if idx >= len(biomass_df):

            continue

        biomass_value = biomass_df.iloc[
            idx
        ]["Predicted_AGB_kg"]

        # ==================================
        # FILL MAP
        # ==================================

        biomass_map[
            ymin:ymax,
            xmin:xmax
        ] = biomass_value

    # ======================================
    # SAVE GEOTIFF
    # ======================================

    profile.update(

        dtype=rasterio.float32,

        count=1
    )

    tif_output = (
        OUTPUT_DIR
        /
        f"{plot_name}_biomass.tif"
    )

    with rasterio.open(

        tif_output,

        "w",

        **profile

    ) as dst:

        dst.write(
            biomass_map,
            1
        )

    # ======================================
    # VISUALIZATION
    # ======================================

    plt.figure(figsize=(10, 8))

    img = plt.imshow(

        biomass_map,

        cmap="turbo"
    )

    plt.colorbar(

        img,

        label="Biomass (kg)"
    )

    plt.title(
        f"Biomass Heatmap\n{plot_name}",
        fontsize=14
    )

    plt.axis("off")

    plt.tight_layout()

    # ======================================
    # SAVE PNG
    # ======================================

    png_output = (
        OUTPUT_DIR
        /
        f"{plot_name}_biomass_heatmap.png"
    )

    plt.savefig(

        png_output,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # SUMMARY
    # ======================================

    print(f"\nSaved GeoTIFF:")
    print(tif_output)

    print(f"\nSaved Heatmap:")
    print(png_output)

    print(f"\nStatistics:")

    print(f"MIN : {biomass_map.min():.2f}")

    print(f"MAX : {biomass_map.max():.2f}")

    print(f"MEAN: {biomass_map.mean():.2f}")

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("BIOMASS MAP FINISHED")
print("====================================")