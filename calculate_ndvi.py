import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ==========================================
# HYPERSPECTRAL DIRECTORY
# ==========================================

HSI_DIR = Path("data_clean/Hyperspectral")

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = Path("outputs/ndvi")

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
# RANDOM FILE
# ==========================================

hsi_file = random.choice(hsi_files)

print("\n====================================")
print("NDVI CALCULATION")
print("====================================")

print(f"\nSelected File:")
print(hsi_file.name)

# ==========================================
# READ HYPERSPECTRAL
# ==========================================

with rasterio.open(hsi_file) as src:

    total_bands = src.count

    print(f"\nTotal Bands: {total_bands}")

    # ======================================
    # APPROXIMATE BAND SELECTION
    # ======================================

    # RED ≈ 660nm
    # NIR ≈ 800nm

    # asumsi band hyperspectral NEON
    # sekitar:

    RED_BAND = min(58, total_bands)
    NIR_BAND = min(90, total_bands)

    print(f"RED Band : {RED_BAND}")
    print(f"NIR Band : {NIR_BAND}")

    red = src.read(RED_BAND).astype(np.float32)

    nir = src.read(NIR_BAND).astype(np.float32)

    profile = src.profile

# ==========================================
# HANDLE INVALID
# ==========================================

red[red <= 0] = np.nan
nir[nir <= 0] = np.nan

# ==========================================
# NDVI
# ==========================================

ndvi = (nir - red) / (nir + red)

# ==========================================
# HANDLE NAN
# ==========================================

ndvi = np.nan_to_num(
    ndvi,
    nan=0
)

# ==========================================
# CLIP RANGE
# ==========================================

ndvi = np.clip(
    ndvi,
    -1,
    1
)

# ==========================================
# SAVE NDVI
# ==========================================

output_file = (
    OUTPUT_DIR
    /
    f"{hsi_file.stem}_NDVI.tif"
)

profile.update(
    dtype=rasterio.float32,
    count=1
)

with rasterio.open(

    output_file,

    "w",

    **profile

) as dst:

    dst.write(
        ndvi.astype(np.float32),
        1
    )

# ==========================================
# NDVI STATISTICS
# ==========================================

print("\n====================================")
print("NDVI STATISTICS")
print("====================================")

print(f"MIN : {ndvi.min():.3f}")
print(f"MAX : {ndvi.max():.3f}")
print(f"MEAN: {ndvi.mean():.3f}")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 10))

img = plt.imshow(

    ndvi,

    cmap="RdYlGn",

    vmin=-1,
    vmax=1
)

plt.colorbar(
    img,
    label="NDVI"
)

plt.title(
    f"NDVI Map\n{hsi_file.name}",
    fontsize=14
)

plt.axis("off")

plt.tight_layout()

plt.show()

print("\n====================================")
print("NDVI FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_file)