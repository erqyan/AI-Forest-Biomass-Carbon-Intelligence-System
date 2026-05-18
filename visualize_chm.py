import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import random

# ==========================================
# CHM DIRECTORY
# ==========================================

CHM_DIR = Path("evaluation/CHM")

# ==========================================
# LOAD CHM FILES
# ==========================================

chm_files = list(CHM_DIR.glob("*.tif"))

# ==========================================
# CHECK FILES
# ==========================================

if len(chm_files) == 0:

    print("[ERROR] No CHM files found!")
    exit()

# ==========================================
# RANDOM FILE
# ==========================================

chm_file = random.choice(chm_files)

print("\n====================================")
print("CHM VISUALIZATION")
print("====================================")

print(f"\nSelected File:")
print(chm_file.name)

# ==========================================
# READ CHM
# ==========================================

with rasterio.open(chm_file) as src:

    chm = src.read(1).astype(np.float32)

# ==========================================
# HANDLE NODATA
# ==========================================

chm[chm < 0] = np.nan

# ==========================================
# STATISTICS
# ==========================================

print("\n====================================")
print("CHM STATISTICS")
print("====================================")

print(f"MIN Height : {np.nanmin(chm):.2f} m")
print(f"MAX Height : {np.nanmax(chm):.2f} m")
print(f"MEAN Height: {np.nanmean(chm):.2f} m")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 12))

img = plt.imshow(
    chm,
    cmap="viridis"
)

# ==========================================
# COLORBAR
# ==========================================

cbar = plt.colorbar(img)

cbar.set_label(
    "Canopy Height (m)"
)

# ==========================================
# TITLE
# ==========================================

plt.title(
    f"Canopy Height Model (CHM)\n{chm_file.name}",
    fontsize=14
)

plt.axis("off")

plt.tight_layout()

plt.show()

print("\nVisualization Finished")