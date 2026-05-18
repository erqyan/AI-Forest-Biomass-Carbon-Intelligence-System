import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import random

# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = (
    "models/unet/"
    "unet_crown_segmentation.keras"
)

# ==========================================
# IMAGE DIRECTORY
# ==========================================

IMAGE_DIR = Path("data_clean/RGB")

# ==========================================
# PARAMETERS
# ==========================================

IMG_SIZE = 256

# ==========================================
# CHECK MODEL
# ==========================================

if not Path(MODEL_PATH).exists():

    print("[ERROR] U-Net model not found!")
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print("\n====================================")
print("LOADING U-NET MODEL")
print("====================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded")

# ==========================================
# LOAD RGB FILES
# ==========================================

image_files = list(IMAGE_DIR.glob("*.tif"))

# ==========================================
# CHECK FILES
# ==========================================

if len(image_files) == 0:

    print("[ERROR] No RGB images found!")
    exit()

# ==========================================
# RANDOM IMAGE
# ==========================================

image_path = random.choice(image_files)

print("\n====================================")
print("CANOPY SEGMENTATION")
print("====================================")

print(f"\nSelected Image:")
print(image_path.name)

# ==========================================
# READ ORIGINAL IMAGE
# ==========================================

original = cv2.imread(str(image_path))

original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

orig_h, orig_w = original.shape[:2]

# ==========================================
# PREPROCESS IMAGE
# ==========================================

image = cv2.resize(
    original,
    (IMG_SIZE, IMG_SIZE)
)

image_input = image.astype(np.float32) / 255.0

image_input = np.expand_dims(
    image_input,
    axis=0
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(image_input)

mask = prediction[0, :, :, 0]

# ==========================================
# THRESHOLD
# ==========================================

mask_binary = (mask > 0.5).astype(np.uint8)

# ==========================================
# RESIZE TO ORIGINAL SIZE
# ==========================================

mask_resized = cv2.resize(

    mask_binary,

    (orig_w, orig_h),

    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# CREATE OVERLAY
# ==========================================

overlay = original.copy()

overlay[mask_resized == 1] = [0, 255, 0]

# ==========================================
# BLEND
# ==========================================

blended = cv2.addWeighted(

    original,
    0.7,

    overlay,
    0.3,

    0
)

# ==========================================
# VISUALIZATION
# ==========================================

fig, axes = plt.subplots(

    1,
    3,

    figsize=(18, 6)
)

# ==========================================
# ORIGINAL
# ==========================================

axes[0].imshow(original)

axes[0].set_title("Original RGB")

axes[0].axis("off")

# ==========================================
# MASK
# ==========================================

axes[1].imshow(

    mask_resized,

    cmap="gray"
)

axes[1].set_title("Predicted Crown Mask")

axes[1].axis("off")

# ==========================================
# OVERLAY
# ==========================================

axes[2].imshow(blended)

axes[2].set_title("Canopy Segmentation Overlay")

axes[2].axis("off")

# ==========================================
# SHOW
# ==========================================

plt.tight_layout()

plt.show()

# ==========================================
# SAVE OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/segmentation")

OUTPUT_DIR.mkdir(

    parents=True,
    exist_ok=True
)

# ==========================================
# SAVE MASK
# ==========================================

mask_path = (
    OUTPUT_DIR
    /
    f"{image_path.stem}_mask.png"
)

cv2.imwrite(

    str(mask_path),

    mask_resized * 255
)

# ==========================================
# SAVE OVERLAY
# ==========================================

overlay_path = (
    OUTPUT_DIR
    /
    f"{image_path.stem}_overlay.png"
)

cv2.imwrite(

    str(overlay_path),

    cv2.cvtColor(
        blended,
        cv2.COLOR_RGB2BGR
    )
)

print("\n====================================")
print("SEGMENTATION FINISHED")
print("====================================")

print(f"\nMask Saved:")
print(mask_path)

print(f"\nOverlay Saved:")
print(overlay_path)