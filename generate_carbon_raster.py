import rasterio
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path

# ==========================================
# INPUT
# ==========================================

RGB_DIR = Path("data_clean/RGB")

ANN_DIR = Path("data_clean/annotations")

CO2_FILE = (
    "outputs/co2e/"
    "co2e_estimation.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/carbon_raster")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(CO2_FILE).exists():

    print("[ERROR] co2e_estimation.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

carbon_df = pd.read_csv(CO2_FILE)

print("\n====================================")
print("GENERATE CARBON RASTER")
print("====================================")

print(f"\nTotal Records: {len(carbon_df)}")

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
    # OPEN RGB FOR GEOREFERENCE
    # ======================================

    with rasterio.open(rgb_file) as src:

        profile = src.profile

        width = src.width

        height = src.height

        transform = src.transform

        crs = src.crs

    # ======================================
    # CREATE EMPTY RASTER
    # ======================================

    carbon_raster = np.zeros(

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
    # PROCESS TREES
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
        # GET CARBON VALUE
        # ==================================

        if idx >= len(carbon_df):

            continue

        carbon_value = carbon_df.iloc[
            idx
        ]["Carbon_kg"]

        # ==================================
        # FILL RASTER
        # ==================================

        carbon_raster[
            ymin:ymax,
            xmin:xmax
        ] = carbon_value

    # ======================================
    # UPDATE PROFILE
    # ======================================

    profile.update(

        dtype=rasterio.float32,

        count=1
    )

    # ======================================
    # OUTPUT FILE
    # ======================================

    output_file = (
        OUTPUT_DIR
        /
        f"{plot_name}_carbon.tif"
    )

    # ======================================
    # SAVE GEOTIFF
    # ======================================

    with rasterio.open(

        output_file,

        "w",

        **profile

    ) as dst:

        dst.write(
            carbon_raster,
            1
        )

    # ======================================
    # SUMMARY
    # ======================================

    print(f"\nSaved:")
    print(output_file)

    print(f"\nCarbon Statistics:")

    print(f"MIN : {carbon_raster.min():.2f}")

    print(f"MAX : {carbon_raster.max():.2f}")

    print(f"MEAN: {carbon_raster.mean():.2f}")

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("CARBON RASTER FINISHED")
print("====================================")