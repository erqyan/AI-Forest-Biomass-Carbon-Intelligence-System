import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import random

# ==========================================
# HYPERSPECTRAL DIRECTORY
# ==========================================

HSI_DIR = Path("data_clean/Hyperspectral")

# ==========================================
# LOAD FILES
# ==========================================

hsi_files = list(HSI_DIR.glob("*.tif"))

if len(hsi_files) == 0:

    print("[ERROR] No hyperspectral files found!")
    exit()

# ==========================================
# RANDOM FILE
# ==========================================

hsi_file = random.choice(hsi_files)

print("\nSelected:")
print(hsi_file.name)

# ==========================================
# READ HYPERSPECTRAL
# ==========================================

with rasterio.open(hsi_file) as src:

    total_bands = src.count

    print(f"\nTotal Bands: {total_bands}")

    # ======================================
    # FALSE COLOR BANDS
    # ======================================

    band_r = min(60, total_bands)
    band_g = min(30, total_bands)
    band_b = min(10, total_bands)

    print(f"R Band: {band_r}")
    print(f"G Band: {band_g}")
    print(f"B Band: {band_b}")

    red = src.read(band_r).astype(np.float32)
    green = src.read(band_g).astype(np.float32)
    blue = src.read(band_b).astype(np.float32)

# ==========================================
# NORMALIZATION FUNCTION
# ==========================================

def normalize_band(band):

    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)

    band = np.clip(
        (band - p2) / (p98 - p2),
        0,
        1
    )

    return band

# ==========================================
# NORMALIZE EACH BAND
# ==========================================

red = normalize_band(red)
green = normalize_band(green)
blue = normalize_band(blue)

# ==========================================
# STACK RGB
# ==========================================

rgb = np.dstack((red, green, blue))

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 12))

plt.imshow(rgb)

plt.title(
    f"Hyperspectral False Color\n{hsi_file.name}",
    fontsize=14
)

plt.axis("off")

plt.tight_layout()

plt.show()