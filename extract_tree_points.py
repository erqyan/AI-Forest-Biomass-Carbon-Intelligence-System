import laspy
import xml.etree.ElementTree as ET
import rasterio
from pathlib import Path
import numpy as np

# ==========================================
# INPUT
# ==========================================

RGB_DIR = Path("data_clean/RGB")

LIDAR_DIR = Path("outputs/normalized_lidar")

ANN_DIR = Path("data_clean/annotations")

OUTPUT_DIR = Path("outputs/tree_points")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# LOAD RGB FILES
# ==========================================

rgb_files = list(RGB_DIR.glob("*.tif"))

# ==========================================
# PROCESS EACH PLOT
# ==========================================

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

    print("\n====================================")
    print(f"PROCESSING: {plot_name}")
    print("====================================")

    # ======================================
    # FILE PATHS
    # ======================================

    lidar_file = LIDAR_DIR / f"{plot_name}_normalized.las"

    xml_file = ANN_DIR / f"{plot_name}.xml"

    if not lidar_file.exists():

        print("[WARNING] Normalized LiDAR missing")
        continue

    if not xml_file.exists():

        print("[WARNING] Annotation missing")
        continue

    # ======================================
    # READ RGB GEOREFERENCE
    # ======================================

    with rasterio.open(rgb_file) as rgb_src:

        transform = rgb_src.transform

    # ======================================
    # READ LIDAR
    # ======================================

    las = laspy.read(lidar_file)

    x = las.x
    y = las.y
    z = las.z

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

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # ==================================
        # PIXEL TO GEO
        # ==================================

        geo_xmin, geo_ymax = rasterio.transform.xy(
            transform,
            ymin,
            xmin
        )

        geo_xmax, geo_ymin = rasterio.transform.xy(
            transform,
            ymax,
            xmax
        )

        # ==================================
        # FILTER POINT CLOUD
        # ==================================

        mask = (
            (x >= geo_xmin)
            &
            (x <= geo_xmax)
            &
            (y >= geo_ymin)
            &
            (y <= geo_ymax)
        )

        if mask.sum() == 0:

            continue

        # ==================================
        # CREATE TREE LAS
        # ==================================

        tree_las = laspy.create(
            point_format=las.header.point_format,
            file_version=las.header.version
        )

        tree_las.points = las.points[mask]

        # ==================================
        # SAVE
        # ==================================

        output_file = OUTPUT_DIR / (
            f"{plot_name}_tree_{idx+1}.las"
        )

        tree_las.write(output_file)

    print("Tree extraction finished")

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("TREE POINT EXTRACTION FINISHED")
print("====================================")