import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# ==========================================
# CLEAN DATASET DIRECTORIES
# ==========================================

RGB_DIR = Path("data_clean/RGB")
ANN_DIR = Path("data_clean/annotations")

# ==========================================
# LOAD RGB FILES
# ==========================================

rgb_files = list(RGB_DIR.glob("*.tif"))

if len(rgb_files) == 0:

    print("[ERROR] No RGB files found!")
    exit()

# ==========================================
# RANDOM RGB FILE
# ==========================================

rgb_file = random.choice(rgb_files)

plot_name = rgb_file.stem

print("\n====================================")
print("VISUALIZE ANNOTATIONS")
print("====================================")

print(f"\nSelected Plot:")
print(plot_name)

# ==========================================
# ANNOTATION FILE
# ==========================================

xml_file = ANN_DIR / f"{plot_name}.xml"

if not xml_file.exists():

    print("[ERROR] Annotation file not found!")
    exit()

# ==========================================
# READ RGB IMAGE
# ==========================================

with rasterio.open(rgb_file) as src:

    red = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(3).astype(np.float32)

# ==========================================
# STACK RGB
# ==========================================

rgb = np.dstack((red, green, blue))

# ==========================================
# PERCENTILE STRETCH
# ==========================================

p2 = np.percentile(rgb, 2)
p98 = np.percentile(rgb, 98)

rgb = np.clip(
    (rgb - p2) / (p98 - p2),
    0,
    1
)

# ==========================================
# PARSE XML
# ==========================================

tree = ET.parse(xml_file)

root = tree.getroot()

objects = root.findall("object")

print(f"\nTotal Annotations: {len(objects)}")

# ==========================================
# VISUALIZATION
# ==========================================

fig, ax = plt.subplots(figsize=(14, 14))

ax.imshow(rgb)

# ==========================================
# DRAW BOUNDING BOX
# ==========================================

for obj in objects:

    bbox = obj.find("bndbox")

    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    width = xmax - xmin
    height = ymax - ymin

    rect = patches.Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=1.5,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)

# ==========================================
# TITLE
# ==========================================

plt.title(
    f"RGB + Tree Annotations\n{plot_name}",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()

plt.show()

print("\nVisualization Finished")