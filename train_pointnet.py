import os
import glob
import random
import numpy as np
import laspy

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONFIG
# ==========================================

POINT_DIR = Path("outputs/tree_points")

MODEL_DIR = Path("models/pointnet")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_POINTS = 1024

BATCH_SIZE = 8

EPOCHS = 20

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

# ==========================================
# DATASET
# ==========================================

class TreePointDataset(Dataset):

    def __init__(self, point_files):

        self.point_files = point_files

    def __len__(self):

        return len(self.point_files)

    def __getitem__(self, idx):

        file = self.point_files[idx]

        las = laspy.read(file)

        points = np.vstack([

            las.x,
            las.y,
            las.z

        ]).T

        # ==================================
        # RANDOM SAMPLING
        # ==================================

        if len(points) >= NUM_POINTS:

            indices = np.random.choice(

                len(points),

                NUM_POINTS,

                replace=False
            )

        else:

            indices = np.random.choice(

                len(points),

                NUM_POINTS,

                replace=True
            )

        points = points[indices]

        # ==================================
        # NORMALIZATION
        # ==================================

        points = points - np.mean(

            points,

            axis=0
        )

        scale = np.max(

            np.linalg.norm(
                points,
                axis=1
            )
        )

        points = points / scale

        # ==================================
        # DUMMY LABELS
        # ==================================

        labels = np.ones(NUM_POINTS)

        return (

            torch.tensor(
                points,
                dtype=torch.float32
            ),

            torch.tensor(
                labels,
                dtype=torch.long
            )
        )

# ==========================================
# SIMPLE POINTNET
# ==========================================

class PointNet(nn.Module):

    def __init__(self):

        super(PointNet, self).__init__()

        self.mlp1 = nn.Sequential(

            nn.Linear(3, 64),

            nn.ReLU(),

            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Linear(128, 256),

            nn.ReLU()
        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        # x = [B, N, 3]

        x = self.mlp1(x)

        x = self.classifier(x)

        return x

# ==========================================
# LOAD FILES
# ==========================================

point_files = list(
    POINT_DIR.glob("*.las")
)

# ==========================================
# CHECK FILES
# ==========================================

if len(point_files) == 0:

    print("[ERROR] No point cloud found!")
    exit()

print("\n====================================")
print("POINTNET TRAINING")
print("====================================")

print(f"\nTotal Point Clouds: {len(point_files)}")

# ==========================================
# DATASET
# ==========================================

dataset = TreePointDataset(
    point_files
)

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)

# ==========================================
# MODEL
# ==========================================

model = PointNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001
)

# ==========================================
# TRAINING
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for points, labels in loader:

        points = points.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(points)

        outputs = outputs.view(-1, 2)

        labels = labels.view(-1)

        loss = criterion(

            outputs,

            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(

        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

model_path = (
    MODEL_DIR
    /
    "pointnet_model.pth"
)

torch.save(

    model.state_dict(),

    model_path
)

# ==========================================
# FINISHED
# ==========================================

print("\n====================================")
print("POINTNET TRAINING FINISHED")
print("====================================")

print(f"\nSaved:")
print(model_path)