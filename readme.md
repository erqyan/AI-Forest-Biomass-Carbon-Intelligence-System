# 🌳 AI Forest Biomass & Carbon Intelligence System

An end-to-end Artificial Intelligence system for:

- 🌲 Individual Tree Detection
- 🌿 Crown Segmentation
- 📡 LiDAR Processing
- 🌈 Hyperspectral Analysis
- 📊 Biomass Estimation
- 🌍 Carbon Stock Mapping
- 🛰 Forest Monitoring
- 🤖 Deep Learning Forestry Intelligence

Built using:

- Python
- YOLO
- U-Net
- PointNet
- LiDAR
- Hyperspectral Imaging
- GIS / GeoTIFF
- Machine Learning
- Remote Sensing

---

# 📌 Overview

This project is a complete multi-sensor forestry AI pipeline designed to automate:

- Tree-level forest inventory
- Biomass estimation
- Carbon estimation
- Canopy analysis
- Forest health monitoring
- Multi-temporal forest change detection

The system integrates:

| Sensor | Function |
|---|---|
| RGB | Tree detection |
| LiDAR | 3D forest structure |
| CHM | Canopy height |
| Hyperspectral | Vegetation chemistry |
| Annotation | Ground truth |

---

# 🧠 Main Capabilities

## 🌲 Tree Detection

- YOLO-based individual tree detection
- Bounding box extraction
- Large-scale orthomosaic analysis

---

## 🌿 Crown Segmentation

- U-Net canopy segmentation
- Individual crown extraction
- Crown geometry analysis

---

## 📡 LiDAR Intelligence

- DSM generation
- DTM generation
- CHM generation
- Point cloud normalization
- Point cloud extraction

---

## 🌈 Hyperspectral Analysis

- NDVI calculation
- NDRE calculation
- Spectral signature extraction
- Species classification

---

## 🌍 Biomass & Carbon

- Allometric biomass estimation
- AI-based biomass prediction
- Carbon stock estimation
- CO₂ equivalent estimation
- Carbon raster generation

---

## 📊 Forest Intelligence

- Forest health monitoring
- Vegetation stress detection
- Degradation analysis
- Multi-temporal forest change detection

---

## 🤖 Advanced AI

- YOLO object detection
- U-Net segmentation
- PointNet point cloud deep learning
- Random Forest
- XGBoost
- LightGBM

---

# 🏗 System Architecture

```text
                ┌─────────────────┐
                │  RGB Imagery    │
                └────────┬────────┘
                         │
                         ▼
               ┌──────────────────┐
               │ YOLO Detection   │
               └────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Crown Segmentation│
              └────────┬──────────┘
                       │
                       ▼

┌────────────┐   ┌─────────────┐   ┌────────────────┐
│ LiDAR      │→ │ CHM / Height │→ │ Structure Feat │
└────────────┘   └─────────────┘   └────────────────┘

┌────────────────┐
│ Hyperspectral  │
└───────┬────────┘
        ▼
┌─────────────────┐
│ Spectral Feature│
└────────┬────────┘
         ▼

      ┌────────────────────┐
      │ Feature Fusion     │
      └────────┬───────────┘
               ▼

      ┌────────────────────┐
      │ Biomass AI Model   │
      └────────┬───────────┘
               ▼

      ┌────────────────────┐
      │ Carbon Estimation  │
      └────────┬───────────┘
               ▼

      ┌────────────────────┐
      │ Forest Intelligence│
      └────────────────────┘
```

---

# 📂 Project Structure

```text
AI Forest Biomass & Carbon Intelligence System/
│
├── data/
├── data_clean/
│
├── outputs/
│   ├── biomass/
│   ├── carbon/
│   ├── lidar_features/
│   ├── crown_features/
│   ├── hyperspectral_features/
│   ├── validation/
│   ├── biomass_map/
│   ├── carbon_raster/
│   ├── forest_health/
│   └── change_detection/
│
├── models/
│   ├── yolo/
│   ├── unet/
│   ├── biomass/
│   └── pointnet/
│
├── runs/
├── dashboard/
├── api/
└── scripts/
```

---

# 📦 Dataset

This project uses the:

# 🌲 NeonTreeEvaluation Dataset

GitHub:  
https://github.com/weecology/NeonTreeEvaluation

Zenodo:  
https://zenodo.org/record/5914554


The system integrates:

|    Sensor     |        Function         |
|---------------|-------------------------|
|      RGB      |   Orthomosaic imagery   |
|     LiDAR     |    3D forest structure  |
|      CHM      |       Canopy height     |
| Hyperspectral |     426-band imagery    |
|  Annotation   |   Tree bounding boxes   |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/erqyan/AI-Forest-Biomass-Carbon-Intelligence-System
cd AI-Forest-Biomass-Carbon-Intelligence-System
```

---

## Create Environment

```bash
conda create -n forest_ai python=3.10
conda activate forest_ai
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Complete Workflow

# TAHAP 1 — DATA VALIDATION

|      Program     |           Function            |
|------------------|-------------------------------|
| check_dataset.py | Validate dataset completeness |

---

# TAHAP 2 — VISUALIZATION

|          Program           |             Function            |
|----------------------------|---------------------------------|
|      visualize_rgb.py      |        RGB visualization        |
|  visualize_annotations.py  |   Bounding box visualization    |
|     visualize_lidar.py     | LiDAR point cloud visualization |
|      visualize_chm.py      |        CHM visualization        |
| visualize_hyperspectral.py |   Hyperspectral visualization   |

---

# TAHAP 3 — PREPROCESSING

|        Program         |          Function          |
|------------------------|----------------------------|
|     generate_dsm.py    |        Generate DSM        |
|     generate_dtm.py    |        Generate DTM        |
|     generate_chm.py    |        Generate CHM        |
|    normalize_lidar.py  |       Normalize LiDAR      |
| extract_tree_points.py |  Extract tree point cloud  |

---

# TAHAP 4 — RGB TREE DETECTION

|          Program          |          Function          |
|---------------------------|----------------------------|
|  prepare_yolo_dataset.py  | Convert XML to YOLO format |
|       train_yolo.py       |     Train YOLO detector    |
| predict_tree_detection.py |  Tree detection inference  |

---

# TAHAP 5 — CROWN SEGMENTATION

|             Program             |                Function               |
|---------------------------------|---------------------------------------|
| prepare_segmentation_dataset.py |      Generate segmentation masks      |
|          train_unet.py          |        Train crown segmentation       |
|     predict_segmentation.py     |     Automatic canopy segmentation     |

---

# TAHAP 6 — HYPERSPECTRAL ANALYSIS

|            Program            |            Function            |
|-------------------------------|--------------------------------|
|       calculate_ndvi.py       |        NDVI calculation        |
|       calculate_ndre.py       |        NDRE calculation        |
| extract_spectral_signature.py |  Spectral signature extraction |
|   species_classification.py   |   Tree species classification  |

---

# TAHAP 7 — FEATURE EXTRACTION

|              Program              |           Function           |
|-----------------------------------|------------------------------|
|     extract_crown_features.py     |    Crown geometry features   |
|     extract_lidar_features.py     |   LiDAR structural features  |
| extract_hyperspectral_features.py | Spectral vegetation features |
|       build_feature_table.py      |  Multi-sensor feature fusion |

---

# TAHAP 8 — BIOMASS MODEL

|             Program            |           Function            |
|--------------------------------|-------------------------------|
| estimate_biomass_allometric.py | Allometric biomass estimation |
|       train_biomass_ml.py      |      Biomass AI training      |
|       predict_biomass.py       |       Biomass prediction      |

---

# TAHAP 9 — CARBON SYSTEM

|          Program          |           Function          |
|---------------------------|-----------------------------|
|    estimate_carbon.py     |   Carbon stock estimation   |
|     estimate_co2e.py      |  CO₂ equivalent estimation  |
| generate_carbon_raster.py |  Carbon GeoTIFF generation  |
|  generate_biomass_map.py  |  Biomass heatmap generation |

---

# TAHAP 10 — VALIDATION

|        Program        |               Function              |
|-----------------------|-------------------------------------|
|  validate_height.py   |        Height RMSE validation       |
|  validate_biomass.py  |          Biomass validation         |
| evaluate_detection.py | IoU / Precision / Recall evaluation |

---

# TAHAP 11 — ADVANCED AI

|           Program           |             Function             |
|-----------------------------|----------------------------------|
|      train_pointnet.py      |     Point cloud segmentation     |
| forest_health_monitoring.py |         Forest health AI         |
|     change_detection.py     | Multi-temporal forest monitoring |

---

# TAHAP 12 — VISUALIZATION SYSTEM

|         Program        |        Function         |
|------------------------|-------------------------|
| dashboard_streamlit.py |  Interactive dashboard  |
|    webgis_forest.py    |    Interactive WebGIS   |
| visualize_3d_forest.py | 3D forest visualization |

---

# TAHAP 13 — DEPLOYMENT

|           Program            |        Function       |
|------------------------------|-----------------------|
|         api_server.py        |       REST API        |
| batch_processing_pipeline.py | Automated AI pipeline |

---

# 📊 Outputs

The system generates:

- 🌲 Tree detection
- 🌿 Crown segmentation
- 📏 Tree height
- 🌈 Spectral signature
- 📦 Biomass estimation
- 🌍 Carbon stock
- ☁ CO₂ equivalent
- 🗺 GeoTIFF raster
- 🔥 Biomass heatmap
- 🌿 Forest health index
- 📉 Forest degradation map

---

# 📈 Machine Learning Models

|          Task          |               Model                |
|------------------------|------------------------------------|
|        Detection       |                YOLO                |
|      Segmentation      |                U-Net               |
|   Biomass Regression   | Random Forest / XGBoost / LightGBM |
|     Point Cloud AI     |               PointNet             |
| Species Classification |             Random Forest          |

---

# 🌍 GIS Compatibility

Outputs are compatible with:

- QGIS
- ArcGIS
- Google Earth Engine
- GeoPandas
- Rasterio

---

# 🔬 Scientific Applications

- Forest Inventory
- Carbon Stock Estimation
- REDD+
- Forest Monitoring
- Ecosystem Intelligence
- Biodiversity Analysis
- Climate Change Monitoring
- Smart Forestry

---

# 📊 Core Equations

## NDVI

```text
NDVI = (NIR - Red) / (NIR + Red)
```

---

## NDRE

```text
NDRE = (NIR - RedEdge) / (NIR + RedEdge)
```

---

## Biomass Equation

```text
AGB = 0.0673 × (ρD²H)^0.976
```

Where:

| Variable |      Description     |
|----------|----------------------|
|    AGB   | Above Ground Biomass |
|     ρ    |      Wood Density    |
|     D    |        Diameter      |
|     H    |         Height       |

---

## Carbon Estimation

```text
Carbon = Biomass × 0.47
```

---

## CO₂ Equivalent

```text
CO₂e = Carbon × 3.67
```

---

# 📌 Future Improvements

- Multi-temporal satellite integration
- Real-time drone processing
- Transformer-based segmentation
- Self-supervised hyperspectral AI
- Large-scale cloud deployment
- Forest Digital Twin
- Real-time forest intelligence

---

# 👨‍💻 Author

Developed for:

AI-Powered Forestry Intelligence Research

---

# 📜 License

This project is intended for:

- Research
- Education
- Environmental monitoring
- Forestry intelligence
- Remote sensing research

---

# 🙏 Acknowledgement

Special thanks to:

- NEON
- Weecology
- NeonTreeEvaluation
- Open-source geospatial community
- Remote sensing research community
- Forestry AI research community