from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = (
    "runs/detect/models/yolo26n_tree_detection/weights/best.pt"
)

# ==========================================
# INPUT RGB DIRECTORY
# ==========================================

RGB_DIR = Path("data_clean/RGB")

# ==========================================
# CHECK MODEL
# ==========================================

if not Path(MODEL_PATH).exists():

    print("[ERROR] YOLO model not found!")
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

print("\n====================================")
print("TREE DETECTION INFERENCE")
print("====================================")

# ==========================================
# LOAD RGB FILES
# ==========================================

rgb_files = list(RGB_DIR.glob("*.tif"))

if len(rgb_files) == 0:

    print("[ERROR] No RGB images found!")
    exit()

# ==========================================
# RANDOM IMAGE
# ==========================================

image_path = random.choice(rgb_files)

print(f"\nSelected Image:")
print(image_path.name)

# ==========================================
# YOLO PREDICTION
# ==========================================

results = model.predict(

    source=str(image_path),

    conf=0.25,

    imgsz=640,

    save=False
)

# ==========================================
# GET RESULT
# ==========================================

result = results[0]

# ==========================================
# READ IMAGE
# ==========================================

image = cv2.imread(str(image_path))

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# GET DETECTION
# ==========================================

boxes = result.boxes

print(f"\nTotal Trees Detected: {len(boxes)}")

# ==========================================
# DRAW BOXES
# ==========================================

for box in boxes:

    coords = box.xyxy[0].cpu().numpy()

    x1, y1, x2, y2 = coords.astype(int)

    confidence = float(box.conf[0])

    # ======================================
    # DRAW RECTANGLE
    # ======================================

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    # ======================================
    # LABEL
    # ======================================

    label = f"Tree {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 0, 0),

        2
    )

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(14, 14))

plt.imshow(image)

plt.title(
    f"YOLO Tree Detection\n{image_path.name}",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()

plt.show()

print("\nInference Finished")