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

OUTPUT_DIR = Path("outputs/ndre")

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
print("NDRE CALCULATION")
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

    # RED EDGE ≈ 720nm
    # NIR ≈ 790-820nm

    RED_EDGE_BAND = min(75, total_bands)

    NIR_BAND = min(90, total_bands)

    print(f"RED EDGE Band : {RED_EDGE_BAND}")
    print(f"NIR Band      : {NIR_BAND}")

    red_edge = src.read(
        RED_EDGE_BAND
    ).astype(np.float32)

    nir = src.read(
        NIR_BAND
    ).astype(np.float32)

    profile = src.profile

# ==========================================
# HANDLE INVALID
# ==========================================

red_edge[red_edge <= 0] = np.nan

nir[nir <= 0] = np.nan

# ==========================================
# NDRE
# ==========================================

ndre = (
    (nir - red_edge)
    /
    (nir + red_edge)
)

# ==========================================
# HANDLE NAN
# ==========================================

ndre = np.nan_to_num(

    ndre,

    nan=0
)

# ==========================================
# CLIP RANGE
# ==========================================

ndre = np.clip(

    ndre,

    -1,
    1
)

# ==========================================
# SAVE NDRE
# ==========================================

output_file = (
    OUTPUT_DIR
    /
    f"{hsi_file.stem}_NDRE.tif"
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
        ndre.astype(np.float32),
        1
    )

# ==========================================
# NDRE STATISTICS
# ==========================================

print("\n====================================")
print("NDRE STATISTICS")
print("====================================")

print(f"MIN : {ndre.min():.3f}")
print(f"MAX : {ndre.max():.3f}")
print(f"MEAN: {ndre.mean():.3f}")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 10))

img = plt.imshow(

    ndre,

    cmap="RdYlGn",

    vmin=-1,
    vmax=1
)

plt.colorbar(
    img,
    label="NDRE"
)

plt.title(
    f"NDRE Map\n{hsi_file.name}",
    fontsize=14
)

plt.axis("off")

plt.tight_layout()

plt.show()

print("\n====================================")
print("NDRE FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_file)