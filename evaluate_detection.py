import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path

# ==========================================
# INPUT DIRECTORIES
# ==========================================

GT_DIR = Path("data_clean/annotations")

PRED_DIR = Path("runs/detect/predict/labels")

RGB_DIR = Path("data_clean/RGB")

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("outputs/evaluation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# IOU FUNCTION
# ==========================================

def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])

    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])

    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)

    inter_height = max(0, yB - yA)

    inter_area = inter_width * inter_height

    boxA_area = (
        (boxA[2] - boxA[0])
        *
        (boxA[3] - boxA[1])
    )

    boxB_area = (
        (boxB[2] - boxB[0])
        *
        (boxB[3] - boxB[1])
    )

    union_area = (
        boxA_area
        +
        boxB_area
        -
        inter_area
    )

    if union_area == 0:

        return 0

    return inter_area / union_area

# ==========================================
# LOAD XML GROUND TRUTH
# ==========================================

def load_ground_truth(xml_file):

    tree = ET.parse(xml_file)

    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):

        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        boxes.append([

            xmin,
            ymin,
            xmax,
            ymax
        ])

    return boxes

# ==========================================
# LOAD YOLO PREDICTIONS
# ==========================================

def load_predictions(
    txt_file,
    image_width,
    image_height
):

    boxes = []

    with open(txt_file, "r") as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:

            continue

        _, xc, yc, w, h = map(
            float,
            parts[:5]
        )

        # ==================================
        # YOLO TO PIXEL
        # ==================================

        xc *= image_width
        yc *= image_height

        w *= image_width
        h *= image_height

        xmin = int(xc - w / 2)
        ymin = int(yc - h / 2)

        xmax = int(xc + w / 2)
        ymax = int(yc + h / 2)

        boxes.append([

            xmin,
            ymin,
            xmax,
            ymax
        ])

    return boxes

# ==========================================
# RGB FILES
# ==========================================

rgb_files = list(
    RGB_DIR.glob("*.tif")
)

# ==========================================
# EVALUATION STORAGE
# ==========================================

all_results = []

# ==========================================
# IOU THRESHOLD
# ==========================================

IOU_THRESHOLD = 0.5

# ==========================================
# PROCESS EACH IMAGE
# ==========================================

for rgb_file in rgb_files:

    plot_name = rgb_file.stem

    print("\n====================================")
    print(f"EVALUATING: {plot_name}")
    print("====================================")

    xml_file = GT_DIR / f"{plot_name}.xml"

    pred_file = PRED_DIR / f"{plot_name}.txt"

    # ======================================
    # CHECK FILES
    # ======================================

    if not xml_file.exists():

        print("[WARNING] Missing XML")
        continue

    if not pred_file.exists():

        print("[WARNING] Missing Prediction")
        continue

    # ======================================
    # IMAGE SIZE
    # ======================================

    import rasterio

    with rasterio.open(rgb_file) as src:

        width = src.width
        height = src.height

    # ======================================
    # LOAD DATA
    # ======================================

    gt_boxes = load_ground_truth(
        xml_file
    )

    pred_boxes = load_predictions(

        pred_file,

        width,
        height
    )

    print(f"\nGround Truth : {len(gt_boxes)}")

    print(f"Predictions  : {len(pred_boxes)}")

    # ======================================
    # MATCHING
    # ======================================

    matched_gt = set()

    tp = 0
    fp = 0

    iou_scores = []

    # ======================================
    # CHECK PREDICTIONS
    # ======================================

    for pred_box in pred_boxes:

        best_iou = 0

        best_gt_idx = -1

        for idx, gt_box in enumerate(gt_boxes):

            iou = calculate_iou(

                pred_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou

                best_gt_idx = idx

        iou_scores.append(best_iou)

        # ==================================
        # TRUE POSITIVE
        # ==================================

        if (

            best_iou >= IOU_THRESHOLD

            and

            best_gt_idx not in matched_gt
        ):

            tp += 1

            matched_gt.add(best_gt_idx)

        else:

            fp += 1

    # ======================================
    # FALSE NEGATIVE
    # ======================================

    fn = len(gt_boxes) - len(matched_gt)

    # ======================================
    # METRICS
    # ======================================

    precision = (

        tp / (tp + fp + 1e-6)
    )

    recall = (

        tp / (tp + fn + 1e-6)
    )

    mean_iou = np.mean(iou_scores) \
        if len(iou_scores) > 0 else 0

    # ======================================
    # SUMMARY
    # ======================================

    print(f"\nTP : {tp}")

    print(f"FP : {fp}")

    print(f"FN : {fn}")

    print(f"\nPrecision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"Mean IoU  : {mean_iou:.4f}")

    # ======================================
    # SAVE RESULT
    # ======================================

    all_results.append({

        "plot": plot_name,

        "TP": tp,

        "FP": fp,

        "FN": fn,

        "precision": precision,

        "recall": recall,

        "mean_iou": mean_iou
    })

# ==========================================
# CREATE DATAFRAME
# ==========================================

results_df = pd.DataFrame(all_results)

# ==========================================
# SAVE CSV
# ==========================================

output_csv = (
    OUTPUT_DIR
    /
    "detection_evaluation.csv"
)

results_df.to_csv(

    output_csv,

    index=False
)

# ==========================================
# OVERALL METRICS
# ==========================================

overall_precision = results_df[
    "precision"
].mean()

overall_recall = results_df[
    "recall"
].mean()

overall_iou = results_df[
    "mean_iou"
].mean()

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n====================================")
print("OVERALL RESULT")
print("====================================")

print(f"\nMean Precision : {overall_precision:.4f}")

print(f"Mean Recall    : {overall_recall:.4f}")

print(f"Mean IoU       : {overall_iou:.4f}")

# ==========================================
# INTERPRETATION
# ==========================================

print("\n====================================")
print("MODEL QUALITY")
print("====================================")

if overall_iou > 0.75:

    quality = "Excellent"

elif overall_iou > 0.6:

    quality = "Good"

elif overall_iou > 0.4:

    quality = "Moderate"

else:

    quality = "Poor"

print(f"\nDetection Quality: {quality}")

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("DETECTION EVALUATION FINISHED")
print("====================================")

print(f"\nSaved:")
print(output_csv)