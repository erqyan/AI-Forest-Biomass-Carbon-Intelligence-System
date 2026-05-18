import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (

    mean_squared_error,

    r2_score
)

import joblib

import matplotlib.pyplot as plt

# ==========================================
# OPTIONAL MODELS
# ==========================================

try:

    from xgboost import XGBRegressor

    xgb_available = True

except:

    xgb_available = False

try:

    from lightgbm import LGBMRegressor

    lgbm_available = True

except:

    lgbm_available = False

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/biomass/"
    "biomass_allometric.csv"
)

# ==========================================
# OUTPUT
# ==========================================

OUTPUT_DIR = Path("models/biomass")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] biomass_allometric.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("BIOMASS MACHINE LEARNING")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# SELECT FEATURES
# ==========================================

candidate_features = [

    "crown_area",
    "crown_density",
    "mean_height",
    "max_height",
    "std_height",

    "canopy_density",

    "ndvi",
    "ndre",

    "spectral_entropy",
    "spectral_energy"
]

# ==========================================
# AVAILABLE FEATURES
# ==========================================

features = [

    f for f in candidate_features

    if f in df.columns
]

print("\nSelected Features:")

for f in features:

    print(f"✔ {f}")

# ==========================================
# FEATURES / TARGET
# ==========================================

X = df[features]

y = df["AGB_kg"]

# ==========================================
# HANDLE NAN
# ==========================================

X = X.fillna(0)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# ==========================================
# MODELS
# ==========================================

models = {}

# ==========================================
# RANDOM FOREST
# ==========================================

models["RandomForest"] = RandomForestRegressor(

    n_estimators=300,

    max_depth=20,

    random_state=42,

    n_jobs=-1
)

# ==========================================
# XGBOOST
# ==========================================

if xgb_available:

    models["XGBoost"] = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=10,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42
    )

# ==========================================
# LIGHTGBM
# ==========================================

if lgbm_available:

    models["LightGBM"] = LGBMRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=10,

        random_state=42
    )

# ==========================================
# RESULTS
# ==========================================

results = []

best_model = None

best_r2 = -999

best_name = None

# ==========================================
# TRAIN EACH MODEL
# ==========================================

for name, model in models.items():

    print("\n====================================")
    print(f"TRAINING: {name}")
    print("====================================")

    # ======================================
    # TRAIN
    # ======================================

    model.fit(

        X_train,
        y_train
    )

    # ======================================
    # PREDICT
    # ======================================

    y_pred = model.predict(X_test)

    # ======================================
    # METRICS
    # ======================================

    rmse = np.sqrt(

        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(

        y_test,
        y_pred
    )

    print(f"\nRMSE : {rmse:.2f}")

    print(f"R²   : {r2:.4f}")

    # ======================================
    # SAVE RESULT
    # ======================================

    results.append({

        "model": name,

        "rmse": rmse,

        "r2": r2
    })

    # ======================================
    # BEST MODEL
    # ======================================

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_name = name

# ==========================================
# SAVE BEST MODEL
# ==========================================

model_path = (
    OUTPUT_DIR
    /
    "best_biomass_model.pkl"
)

joblib.dump(

    best_model,

    model_path
)

# ==========================================
# RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(results)

# ==========================================
# SAVE RESULTS
# ==========================================

results_csv = (
    OUTPUT_DIR
    /
    "model_results.csv"
)

results_df.to_csv(

    results_csv,

    index=False
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": features,

        "importance": importance
    })

    importance_df = importance_df.sort_values(

        by="importance",

        ascending=False
    )

    importance_csv = (
        OUTPUT_DIR
        /
        "feature_importance.csv"
    )

    importance_df.to_csv(

        importance_csv,

        index=False
    )

    # ======================================
    # VISUALIZATION
    # ======================================

    plt.figure(figsize=(10, 6))

    plt.barh(

        importance_df["feature"],

        importance_df["importance"]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        f"Feature Importance ({best_name})"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()

# ==========================================
# SUMMARY
# ==========================================

print("\n====================================")
print("BEST MODEL")
print("====================================")

print(f"\nModel : {best_name}")

print(f"R²    : {best_r2:.4f}")

print(f"\nSaved:")

print(model_path)

print(results_csv)

print("\n====================================")
print("BIOMASS ML FINISHED")
print("====================================")