import os
from pathlib import Path
import xml.etree.ElementTree as ET

# =====================================
# PROJECT ROOT
# =====================================

BASE_DIR = Path(".")

# =====================================
# DATASET DIRECTORIES
# =====================================

RGB_DIR = BASE_DIR / "evaluation" / "RGB"
LIDAR_DIR = BASE_DIR / "evaluation" / "LiDAR"
CHM_DIR = BASE_DIR / "evaluation" / "CHM"
HSI_DIR = BASE_DIR / "evaluation" / "Hyperspectral"
ANNOTATION_DIR = BASE_DIR / "annotations"

# =====================================
# CHECK DIRECTORY EXISTENCE
# =====================================

print("\n====================================")
print("CHECKING DATASET DIRECTORIES")
print("====================================")

dirs = {
    "RGB": RGB_DIR,
    "LiDAR": LIDAR_DIR,
    "CHM": CHM_DIR,
    "Hyperspectral": HSI_DIR,
    "Annotations": ANNOTATION_DIR
}

for name, path in dirs.items():

    if path.exists():

        print(f"[OK] {name} directory found")

    else:

        print(f"[ERROR] {name} directory missing")

# =====================================
# LOAD RGB FILES
# =====================================

rgb_files = list(RGB_DIR.glob("*.tif"))

# remove competition files if any
rgb_files = [
    f for f in rgb_files
    if "competition" not in f.name.lower()
]

print("\n====================================")
print("TOTAL RGB FILES")
print("====================================")
print(len(rgb_files))

# =====================================
# SENSOR VALIDATION
# =====================================

missing_chm = []
missing_hsi = []
missing_lidar = []

print("\n====================================")
print("CHECKING SENSOR SYNCHRONIZATION")
print("====================================")

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

    print(f"\nChecking Plot: {plot_name}")

    # =================================
    # CHM
    # =================================

    chm_file = CHM_DIR / f"{plot_name}_CHM.tif"

    if chm_file.exists():

        print("[OK] CHM")

    else:

        print("[MISSING] CHM")

        missing_chm.append(plot_name)

    # =================================
    # HYPERSPECTRAL
    # =================================

    hsi_file = HSI_DIR / f"{plot_name}.tif"

    if hsi_file.exists():

        print("[OK] Hyperspectral")

    else:

        print("[MISSING] Hyperspectral")

        missing_hsi.append(plot_name)

    # =================================
    # LiDAR
    # =================================

    lidar_las = LIDAR_DIR / f"{plot_name}.las"
    lidar_laz = LIDAR_DIR / f"{plot_name}.laz"

    if lidar_las.exists() or lidar_laz.exists():

        print("[OK] LiDAR")

    else:

        print("[MISSING] LiDAR")

        missing_lidar.append(plot_name)

# =====================================
# ANNOTATION VALIDATION
# =====================================

print("\n====================================")
print("CHECKING ANNOTATIONS")
print("====================================")

xml_files = list(ANNOTATION_DIR.glob("*.xml"))

annotation_errors = []

for xml_file in xml_files:

    annotation_name = xml_file.stem

    print(f"\nChecking Annotation: {annotation_name}")

    try:

        tree = ET.parse(xml_file)

        root = tree.getroot()

        filename_node = root.find("filename")

        if filename_node is None:

            print("[ERROR] filename tag missing")

            annotation_errors.append(annotation_name)

            continue

        rgb_filename = filename_node.text

        rgb_name = Path(rgb_filename).stem

        # =================================
        # CHECK RGB MATCH
        # =================================

        rgb_path = RGB_DIR / f"{rgb_name}.tif"

        if rgb_path.exists():

            print("[OK] RGB match")

        else:

            print("[ERROR] RGB file not found")

            annotation_errors.append(annotation_name)

    except Exception as e:

        print(f"[ERROR] XML Parsing Failed: {e}")

        annotation_errors.append(annotation_name)

# =====================================
# SUMMARY
# =====================================

print("\n====================================")
print("DATA VALIDATION SUMMARY")
print("====================================")

print(f"Total RGB Files          : {len(rgb_files)}")
print(f"Missing CHM              : {len(missing_chm)}")
print(f"Missing Hyperspectral    : {len(missing_hsi)}")
print(f"Missing LiDAR            : {len(missing_lidar)}")
print(f"Annotation Errors        : {len(annotation_errors)}")

# =====================================
# DETAIL REPORT
# =====================================

if len(missing_chm) > 0:

    print("\nMissing CHM:")
    for x in missing_chm:
        print("-", x)

if len(missing_hsi) > 0:

    print("\nMissing Hyperspectral:")
    for x in missing_hsi:
        print("-", x)

if len(missing_lidar) > 0:

    print("\nMissing LiDAR:")
    for x in missing_lidar:
        print("-", x)

if len(annotation_errors) > 0:

    print("\nAnnotation Errors:")
    for x in annotation_errors:
        print("-", x)

print("\n====================================")
print("DATA VALIDATION FINISHED")
print("====================================")