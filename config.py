"""
config.py
---------
Central configuration for the EuroSAT classification project.

All hyperparameters and paths live here so that nothing is hard-coded
across the other modules. If you need to change a batch size, learning
rate, or output directory, this is the only file you touch.
"""

import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data/eurosat"   # Root folder for the EuroSAT dataset.
                                # torchvision will create this on first download.
OUTPUT_DIR = "./outputs"        # All plots, checkpoints, and logs go here.

# ─── Dataset Split Ratios ─────────────────────────────────────────────────────
# 70 / 15 / 15 is a standard split for a dataset of this size.
# Stratification is applied so every class is proportionally represented
# in all three splits — important for reliable per-class F1 computation.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─── Training ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 10
BATCH_SIZE   = 32       # 32 is safer than 64 on a free Colab T4 GPU,
                        # especially for ViT at 224x224.
NUM_WORKERS  = 2        # Number of parallel data-loading workers.
                        # Set to 0 on Windows if you encounter errors.
NUM_EPOCHS   = 20       # Applies to all three models.
LEARNING_RATE = 1e-4   # Works well as a starting point for fine-tuning.
                        # Too high and pre-trained weights are destroyed;
                        # too low and training stalls.
WEIGHT_DECAY  = 1e-4   # L2 regularisation via AdamW. Helps prevent
                        # overfitting on the relatively small training set.
SEED = 42              # Fixed seed for reproducibility across runs.

# ─── Image Sizes ──────────────────────────────────────────────────────────────
# EuroSAT native resolution is 64x64.
# ViT-B/16 was pre-trained on 224x224 and requires that input size.
# Baseline CNN and ResNet50 operate on the native 64x64.
IMG_SIZE_CNN = 64
IMG_SIZE_VIT = 224

# ─── Two-Phase Fine-Tuning for ResNet50 ───────────────────────────────────────
# Phase 1: train only the new classification head with the backbone frozen.
# This prevents the random head weights from destroying the pre-trained
# backbone during the first few gradient updates.
# Phase 2: unfreeze everything and fine-tune end-to-end at a low LR.
RESNET_PHASE1_EPOCHS = 5
RESNET_PHASE2_EPOCHS = NUM_EPOCHS - RESNET_PHASE1_EPOCHS  # = 15

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Class Names (EuroSAT) ────────────────────────────────────────────────────
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
