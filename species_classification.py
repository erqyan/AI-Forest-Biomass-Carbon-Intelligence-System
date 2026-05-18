import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt

# ==========================================
# INPUT FILE
# ==========================================

INPUT_CSV = (
    "outputs/spectral_features/"
    "spectral_signatures.csv"
)

# ==========================================
# CHECK FILE
# ==========================================

if not Path(INPUT_CSV).exists():

    print("[ERROR] spectral_signatures.csv not found!")
    exit()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("\n====================================")
print("TREE SPECIES CLASSIFICATION")
print("====================================")

print(f"\nTotal Samples: {len(df)}")

# ==========================================
# FEATURE COLUMNS
# ==========================================

feature_cols = [

    c for c in df.columns

    if c.startswith("band_")
]

# ==========================================
# CREATE DUMMY LABELS
# ==========================================

# simulasi species

species_classes = [

    "Oak",
    "Pine",
    "Fir",
    "Cedar",
    "Maple"
]

np.random.seed(42)

df["species"] = np.random.choice(

    species_classes,

    size=len(df)
)

print("\nSpecies Distribution:")

print(df["species"].value_counts())

# ==========================================
# FEATURES / LABELS
# ==========================================

X = df[feature_cols]

y = df["species"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=20,

    random_state=42,

    n_jobs=-1
)

# ==========================================
# TRAIN
# ==========================================

print("\n====================================")
print("TRAINING MODEL")
print("====================================")

model.fit(

    X_train,
    y_train
)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(

    y_test,
    y_pred
)

print("\n====================================")
print("RESULT")
print("====================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(

    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "band": feature_cols,

    "importance": importance
})

importance_df = importance_df.sort_values(

    by="importance",

    ascending=False
)

# ==========================================
# TOP 20 BANDS
# ==========================================

top_features = importance_df.head(20)

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 8))

plt.barh(

    top_features["band"],

    top_features["importance"]
)

plt.xlabel("Importance")

plt.ylabel("Spectral Band")

plt.title("Top Spectral Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# ==========================================
# SAVE MODEL
# ==========================================

OUTPUT_DIR = Path("models/species")

OUTPUT_DIR.mkdir(

    parents=True,
    exist_ok=True
)

# ==========================================
# SAVE FEATURE IMPORTANCE
# ==========================================

importance_df.to_csv(

    OUTPUT_DIR / "feature_importance.csv",

    index=False
)

print("\n====================================")
print("MODEL FINISHED")
print("====================================")

print("\nSaved:")

print(
    OUTPUT_DIR / "feature_importance.csv"
)