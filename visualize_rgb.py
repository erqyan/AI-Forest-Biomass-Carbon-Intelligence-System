import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import random

# ==========================================
# RGB DIRECTORY
# ==========================================

RGB_DIR = Path("data_clean/RGB")

rgb_files = list(RGB_DIR.glob("*.tif"))

# ==========================================
# RANDOM IMAGE
# ==========================================

rgb_file = random.choice(rgb_files)

print("\nSelected:")
print(rgb_file.name)

# ==========================================
# READ IMAGE
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
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 12))

plt.imshow(rgb)

plt.title(
    f"RGB Orthomosaic\n{rgb_file.name}",
    fontsize=14
)

plt.axis("off")

plt.tight_layout()

plt.show()