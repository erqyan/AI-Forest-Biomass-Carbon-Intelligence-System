import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# ==========================================
# DATASET DIRECTORY
# ==========================================

IMAGE_DIR = Path("segmentation_dataset/images/train")
MASK_DIR = Path("segmentation_dataset/masks/train")

# ==========================================
# PARAMETERS
# ==========================================

IMG_SIZE = 256

BATCH_SIZE = 4

EPOCHS = 50

# ==========================================
# LOAD FILES
# ==========================================

image_files = sorted(list(IMAGE_DIR.glob("*.png")))

mask_files = sorted(list(MASK_DIR.glob("*.png")))

# ==========================================
# CHECK DATA
# ==========================================

if len(image_files) == 0:

    print("[ERROR] No training images found!")
    exit()

print("\n====================================")
print("U-NET CROWN SEGMENTATION")
print("====================================")

print(f"\nTotal Images: {len(image_files)}")

# ==========================================
# LOAD DATA
# ==========================================

images = []
masks = []

# ==========================================
# READ DATA
# ==========================================

for img_path, mask_path in zip(image_files, mask_files):

    # ======================================
    # READ IMAGE
    # ======================================

    image = cv2.imread(str(img_path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32) / 255.0

    # ======================================
    # READ MASK
    # ======================================

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE)
    )

    mask = mask.astype(np.float32) / 255.0

    mask = np.expand_dims(mask, axis=-1)

    images.append(image)

    masks.append(mask)

# ==========================================
# TO NUMPY
# ==========================================

X = np.array(images)

Y = np.array(masks)

print(f"Image Shape : {X.shape}")
print(f"Mask Shape  : {Y.shape}")

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, Y_train, Y_val = train_test_split(

    X,
    Y,

    test_size=0.2,

    random_state=42
)

# ==========================================
# U-NET MODEL
# ==========================================

def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x

# ==========================================
# BUILD U-NET
# ==========================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

# ==========================================
# ENCODER
# ==========================================

c1 = conv_block(inputs, 64)

p1 = layers.MaxPooling2D()(c1)

c2 = conv_block(p1, 128)

p2 = layers.MaxPooling2D()(c2)

c3 = conv_block(p2, 256)

p3 = layers.MaxPooling2D()(c3)

# ==========================================
# BOTTLENECK
# ==========================================

b1 = conv_block(p3, 512)

# ==========================================
# DECODER
# ==========================================

u1 = layers.UpSampling2D()(b1)

u1 = layers.concatenate([u1, c3])

c4 = conv_block(u1, 256)

u2 = layers.UpSampling2D()(c4)

u2 = layers.concatenate([u2, c2])

c5 = conv_block(u2, 128)

u3 = layers.UpSampling2D()(c5)

u3 = layers.concatenate([u3, c1])

c6 = conv_block(u3, 64)

# ==========================================
# OUTPUT
# ==========================================

outputs = layers.Conv2D(

    1,

    1,

    activation="sigmoid"

)(c6)

# ==========================================
# MODEL
# ==========================================

model = Model(
    inputs,
    outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==========================================
# SUMMARY
# ==========================================

model.summary()

# ==========================================
# TRAINING
# ==========================================

history = model.fit(

    X_train,
    Y_train,

    validation_data=(X_val, Y_val),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS
)

# ==========================================
# CREATE OUTPUT DIR
# ==========================================

MODEL_DIR = Path("models/unet")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    MODEL_DIR / "unet_crown_segmentation.keras"
)

print("\n====================================")
print("MODEL SAVED")
print("====================================")

# ==========================================
# PLOT LOSS
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Val Loss"
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("\nTraining Finished")