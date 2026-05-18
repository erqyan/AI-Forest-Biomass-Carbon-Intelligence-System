from ultralytics import YOLO
from pathlib import Path
from multiprocessing import freeze_support

# ==========================================
# MAIN
# ==========================================

def main():

    # ======================================
    # YOLO DATASET
    # ======================================

    DATA_YAML = Path("yolo_dataset/data.yaml")

    # ======================================
    # CHECK DATA
    # ======================================

    if not DATA_YAML.exists():

        print("[ERROR] data.yaml not found!")
        return

    # ======================================
    # LOAD MODEL
    # ======================================

    model = YOLO("yolo26n.pt")

    # ======================================
    # TRAIN
    # ======================================

    model.train(

        # ==================================
        # DATASET
        # ==================================

        data=str(DATA_YAML),

        # ==================================
        # TRAINING
        # ==================================

        epochs=100,

        imgsz=640,

        batch=8,

        device=0,

        workers=0,

        # ==================================
        # OPTIMIZER
        # ==================================

        optimizer="AdamW",

        lr0=0.001,

        # ==================================
        # AUGMENTATION
        # ==================================

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        translate=0.1,
        scale=0.5,

        fliplr=0.5,

        mosaic=1.0,

        # ==================================
        # PROJECT
        # ==================================

        project="models",

        name="yolo26n_tree_detection",

        # ==================================
        # VALIDATION
        # ==================================

        val=True,

        plots=True
    )

    print("\n====================================")
    print("YOLO TRAINING FINISHED")
    print("====================================")

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    freeze_support()

    main()