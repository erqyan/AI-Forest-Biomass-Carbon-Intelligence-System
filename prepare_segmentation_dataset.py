import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# ==========================================
# INPUT DIRECTORIES
# ==========================================

RGB_DIR = Path("data_clean/RGB")
ANN_DIR = Path("data_clean/annotations")

# ==========================================
# OUTPUT DATASET
# ==========================================

SEG_DIR = Path("segmentation_dataset")

TRAIN_IMG_DIR = SEG_DIR / "images/train"
VAL_IMG_DIR = SEG_DIR / "images/val"

TRAIN_MASK_DIR = SEG_DIR / "masks/train"
VAL_MASK_DIR = SEG_DIR / "masks/val"

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for folder in [

    TRAIN_IMG_DIR,
    VAL_IMG_DIR,

    TRAIN_MASK_DIR,
    VAL_MASK_DIR

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
# SHUFFLE
# ==========================================

random.shuffle(rgb_files)

# ==========================================
# SPLIT DATASET
# ==========================================

split_idx = int(len(rgb_files) * 0.8)

train_files = rgb_files[:split_idx]
val_files = rgb_files[split_idx:]

print("\n====================================")
print("SEGMENTATION DATASET PREPARATION")
print("====================================")

print(f"\nTotal Images : {len(rgb_files)}")
print(f"Train Images : {len(train_files)}")
print(f"Val Images   : {len(val_files)}")

# ==========================================
# PROCESS FUNCTION
# ==========================================

def process_dataset(
    file_list,
    image_output_dir,
    mask_output_dir
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
        # READ IMAGE
        # ==================================

        image = cv2.imread(str(rgb_file))

        if image is None:

            print("[WARNING] Cannot read image")
            continue

        height, width = image.shape[:2]

        # ==================================
        # CREATE EMPTY MASK
        # ==================================

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        # ==================================
        # READ XML
        # ==================================

        tree = ET.parse(xml_file)

        root = tree.getroot()

        objects = root.findall("object")

        # ==================================
        # DRAW MASK
        # ==================================

        for obj in objects:

            bbox = obj.find("bndbox")

            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            # ==============================
            # DRAW FILLED RECTANGLE
            # ==============================

            cv2.rectangle(
                mask,
                (xmin, ymin),
                (xmax, ymax),
                255,
                -1
            )

        # ==================================
        # SAVE IMAGE
        # ==================================

        output_image = (
            image_output_dir
            /
            f"{plot_name}.png"
        )

        cv2.imwrite(
            str(output_image),
            image
        )

        # ==================================
        # SAVE MASK
        # ==================================

        output_mask = (
            mask_output_dir
            /
            f"{plot_name}.png"
        )

        cv2.imwrite(
            str(output_mask),
            mask
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
    TRAIN_MASK_DIR
)

# ==========================================
# PROCESS VALIDATION
# ==========================================

print("\n====================================")
print("PROCESSING VALIDATION DATA")
print("====================================")

process_dataset(

    val_files,

    VAL_IMG_DIR,
    VAL_MASK_DIR
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("SEGMENTATION DATASET FINISHED")
print("====================================")

print(f"\nSaved to:")
print(SEG_DIR)