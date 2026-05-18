import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ==========================================
# CLEAN LIDAR DIRECTORY
# ==========================================

LIDAR_DIR = Path("data_clean/LiDAR")

# ==========================================
# LOAD LIDAR FILES
# ==========================================

las_files = list(LIDAR_DIR.glob("*.las"))
laz_files = list(LIDAR_DIR.glob("*.laz"))

lidar_files = las_files + laz_files

# ==========================================
# CHECK FILES
# ==========================================

if len(lidar_files) == 0:

    print("[ERROR] No LiDAR files found!")
    exit()

# ==========================================
# RANDOM FILE
# ==========================================

lidar_file = random.choice(lidar_files)

print("\n====================================")
print("LIDAR VISUALIZATION")
print("====================================")

print(f"\nSelected File:")
print(lidar_file.name)

# ==========================================
# READ LIDAR
# ==========================================

las = laspy.read(lidar_file)

# ==========================================
# EXTRACT XYZ
# ==========================================

x = las.x
y = las.y
z = las.z

print("\n====================================")
print("POINT CLOUD INFO")
print("====================================")

print(f"Total Points : {len(x):,}")

print(f"X Range      : {x.min():.2f} - {x.max():.2f}")
print(f"Y Range      : {y.min():.2f} - {y.max():.2f}")
print(f"Z Range      : {z.min():.2f} - {z.max():.2f}")

# ==========================================
# RANDOM SAMPLING
# ==========================================

MAX_POINTS = 100000

if len(x) > MAX_POINTS:

    idx = np.random.choice(
        len(x),
        MAX_POINTS,
        replace=False
    )

    x = x[idx]
    y = y[idx]
    z = z[idx]

    print(f"\nPoint cloud sampled to {MAX_POINTS:,} points")

# ==========================================
# VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    x,
    y,
    z,
    c=z,
    s=0.5
)

# ==========================================
# LABELS
# ==========================================

ax.set_title(
    f"LiDAR Point Cloud\n{lidar_file.name}",
    fontsize=14
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Height (Z)")

# ==========================================
# COLORBAR
# ==========================================

cbar = plt.colorbar(
    scatter,
    ax=ax,
    shrink=0.6
)

cbar.set_label("Elevation")

# ==========================================
# SHOW
# ==========================================

plt.tight_layout()

plt.show()

print("\nVisualization Finished")