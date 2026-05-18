import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ==========================================
# INPUT DIRECTORIES
# ==========================================

RGB_DIR = Path("data_clean/RGB")
ANN_DIR = Path("data_clean/annotations")

# ==========================================
# OUTPUT YOLO DATASET
# ==========================================

YOLO_DIR = Path("yolo_dataset")

TRAIN_IMG_DIR = YOLO_DIR / "images" / "train"
VAL_IMG_DIR = YOLO_DIR / "images" / "val"

TRAIN_LABEL_DIR = YOLO_DIR / "labels" / "train"
VAL_LABEL_DIR = YOLO_DIR / "labels" / "val"

# ==========================================
# CREATE FOLDERS
# ==========================================

for folder in [
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

# ==========================================
# LOAD RGB FILES
# ==========================================

rgb_files = list(RGB_DIR.glob("*.tif"))

# ==========================================
# CHECK FILES
# ==========================================

if len(rgb_files) == 0:

    print("[ERROR] No RGB files found!")
    exit()

# ==========================================
# SHUFFLE DATA
# ==========================================

random.shuffle(rgb_files)

# ==========================================
# TRAIN / VAL SPLIT
# ==========================================

split_index = int(len(rgb_files) * 0.8)

train_files = rgb_files[:split_index]
val_files = rgb_files[split_index:]

print("\n====================================")
print("YOLO DATASET PREPARATION")
print("====================================")

print(f"\nTotal Files : {len(rgb_files)}")
print(f"Train Files : {len(train_files)}")
print(f"Val Files   : {len(val_files)}")

# ==========================================
# FUNCTION
# ==========================================

def convert_xml_to_yolo(
    xml_path,
    output_txt,
    img_width,
    img_height
):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    objects = root.findall("object")

    lines = []

    for obj in objects:

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # ==================================
        # YOLO FORMAT
        # ==================================

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height

        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # ==================================
        # CLASS ID
        # ==================================

        class_id = 0  # tree

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        lines.append(line)

    # ======================================
    # SAVE TXT
    # ======================================

    with open(output_txt, "w") as f:

        for line in lines:

            f.write(line + "\n")

# ==========================================
# PROCESS FUNCTION
# ==========================================

def process_dataset(
    file_list,
    image_output_dir,
    label_output_dir
):

    for rgb_file in file_list:

        plot_name = rgb_file.stem

        print(f"\nProcessing: {plot_name}")

        # ==================================
        # XML FILE
        # ==================================

        xml_file = ANN_DIR / f"{plot_name}.xml"

        if not xml_file.exists():

            print("[WARNING] XML missing")
            continue

        # ==================================
        # IMAGE SIZE
        # ==================================

        with Image.open(rgb_file) as img:

            width, height = img.size

        # ==================================
        # COPY IMAGE
        # ==================================

        output_image = image_output_dir / rgb_file.name

        shutil.copy2(
            rgb_file,
            output_image
        )

        # ==================================
        # OUTPUT LABEL
        # ==================================

        output_txt = (
            label_output_dir
            /
            f"{plot_name}.txt"
        )

        # ==================================
        # CONVERT XML → YOLO
        # ==================================

        convert_xml_to_yolo(
            xml_file,
            output_txt,
            width,
            height
        )

# ==========================================
# PROCESS TRAIN
# ==========================================

print("\n====================================")
print("PROCESSING TRAIN DATA")
print("====================================")

process_dataset(
    train_files,
    TRAIN_IMG_DIR,
    TRAIN_LABEL_DIR
)

# ==========================================
# PROCESS VAL
# ==========================================

print("\n====================================")
print("PROCESSING VALIDATION DATA")
print("====================================")

process_dataset(
    val_files,
    VAL_IMG_DIR,
    VAL_LABEL_DIR
)

# ==========================================
# CREATE DATA.YAML
# ==========================================

yaml_content = f"""
path: {YOLO_DIR}

train: images/train
val: images/val

names:
  0: tree
"""

yaml_file = YOLO_DIR / "data.yaml"

with open(yaml_file, "w") as f:

    f.write(yaml_content)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("YOLO DATASET FINISHED")
print("====================================")

print(f"\nSaved to:")
print(YOLO_DIR)