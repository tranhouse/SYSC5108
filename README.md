# EuroSAT Land Use Classification
### SYSC 5108 — Neural Networks and Deep Learning

---

## Project Overview

This project trains and compares three deep learning models for land use and
land cover (LULC) classification on the EuroSAT satellite imagery dataset.
The methods are motivated by Fisheries and Oceans Canada (DFO) work on
ghost gear detection from satellite imagery, where operational data is
restricted. EuroSAT serves as the public proxy dataset.

**Models compared:**
| Model | Pre-trained | Architecture |
|---|---|---|
| Baseline CNN | No | Convolutional |
| ResNet50 | Yes (ImageNet) | Convolutional + Residual |
| ViT-B/16 | Yes (ImageNet-21k) | Transformer |

---

## File Structure

```
eurosat/
├── config.py       All hyperparameters and paths
├── data.py         Dataset download, splitting, normalisation, dataloaders
├── models.py       Model definitions (BaselineCNN, ResNet50, ViT-B/16)
├── train.py        Training loop, optimiser, scheduler, checkpointing
├── evaluate.py     Metrics, confusion matrices, all plots
├── main.py         Orchestrates the full pipeline end to end
└── README.md       This file
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm
```

- **torch / torchvision** — PyTorch and its dataset utilities
- **timm** — PyTorch Image Models library; provides ViT-B/16 with pre-trained weights
- **scikit-learn** — stratified splitting, F1, confusion matrix
- **matplotlib / seaborn** — all plots
- **tqdm** — progress bars during training

### 2. (Optional) Verify GPU availability

```python
import torch
print(torch.cuda.is_available())   # True if a GPU is detected
print(torch.cuda.get_device_name(0))
```

If no GPU is available, training will run on CPU. Expect roughly 5–10x
longer training times. Reduce `NUM_EPOCHS` to 10 if training on CPU.

---

## Step-by-Step Pipeline

### Step 1 — Configuration (`config.py`)

All hyperparameters live in `config.py`. Before running, review:

```python
BATCH_SIZE   = 32      # Reduce to 16 if you get CUDA out-of-memory errors
NUM_EPOCHS   = 20      # Reduce to 10 for a faster first run
LEARNING_RATE = 1e-4
DATA_DIR     = "./data/eurosat"
OUTPUT_DIR   = "./outputs"
```

**Why these values?**
- `BATCH_SIZE = 32`: Safe for a 16GB GPU even with ViT at 224x224. If you
  have a smaller GPU (e.g., Colab T4 with 15GB), start here and increase
  only if memory allows.
- `LEARNING_RATE = 1e-4`: A standard starting point for fine-tuning
  pre-trained models. High enough to learn, low enough not to destroy
  pre-trained features.
- `SEED = 42`: Fixes all random generators so results are reproducible
  across separate runs on the same machine.

---

### Step 2 — Data Pipeline (`data.py`)

Running `main.py` triggers `build_dataloaders()` which performs:

#### 2a. Download
```python
EuroSAT(root=DATA_DIR, download=True)
```
torchvision downloads the RGB EuroSAT zip (~90 MB) automatically on first
run. Subsequent runs detect the existing folder and skip the download.

#### 2b. Stratified 70/15/15 Split
```python
train_idx, temp_idx = train_test_split(indices, test_size=0.30,
                                        stratify=targets)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50,
                                        stratify=targets[temp_idx])
```
**Why stratify?** A purely random split could, by chance, give the test
set more samples from one class than another. Stratification guarantees
that every class appears in each split at its natural frequency, making
per-class metrics meaningful and fair to compare.

#### 2c. Compute Normalisation Statistics
```python
mean, std = compute_mean_std(raw_dataset, train_idx)
```
Per-channel mean and standard deviation are computed from the **training
split only**. If computed from the full dataset, information from the
validation and test sets would leak into preprocessing, biasing evaluation.

#### 2d. Augmentation (Training Split Only)
```python
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```
**Why these augmentations?**
- Satellite images have no canonical "up" direction, so flips and rotations
  are realistic and safe transformations.
- Augmentation is **not** applied to validation or test splits — those
  must represent real, unmodified data.

#### 2e. Image Sizing
- Baseline CNN and ResNet50: native **64×64**
- ViT-B/16: resized to **224×224** (required by the pre-trained model's
  patch embedding)

---

### Step 3 — Baseline CNN (`models.py`)

```python
cnn_model = BaselineCNN(num_classes=10)
```

Architecture: 4 convolutional blocks → AdaptiveAvgPool → MLP classifier

```
Conv2D(3→32)  → BN → ReLU → MaxPool   # 64x64 → 32x32
Conv2D(32→64) → BN → ReLU → MaxPool   # 32x32 → 16x16
Conv2D(64→128)→ BN → ReLU → MaxPool   # 16x16 → 8x8
Conv2D(128→256)→BN → ReLU → AvgPool   # 8x8   → 4x4
Flatten → Linear(4096→512) → ReLU → Dropout(0.4) → Linear(512→10)
```

**Why this serves as a baseline?**
It establishes the performance level achievable without any pre-training.
The gap between this and the pre-trained models directly quantifies the
value of transfer learning on satellite imagery.

Training:
```bash
# Called automatically from main.py:
run_training(cnn_model, train_64, val_64, model_name="BaselineCNN")
```

---

### Step 4 — ResNet50 with Transfer Learning (`models.py`, `train.py`)

#### Phase 1: Train head only (5 epochs)
```python
model = build_resnet50(freeze_backbone=True)
run_training(model, ..., num_epochs=5)
```
**Why freeze the backbone first?**
The randomly initialised head produces large gradient signals at the
start. If the backbone is unfrozen, those gradients propagate back and
corrupt the carefully pre-trained ImageNet weights before the head
has had any chance to stabilise.

#### Phase 2: Fine-tune everything (15 epochs, 10× lower LR)
```python
model = unfreeze_resnet(model)
run_training(model, ..., num_epochs=15, lr=1e-5)
```
**Why a lower learning rate for Phase 2?**
The backbone already has good features. We want to make small,
targeted adjustments to adapt to satellite imagery — not overwrite
everything learned from ImageNet.

---

### Step 5 — ViT-B/16 with Transfer Learning (`models.py`)

```python
vit_model = build_vit(num_classes=10)
run_training(vit_model, train_224, val_224, model_name="ViT-B16")
```

**Why 224×224?**
ViT-B/16 splits the image into 16×16 patches. At 224×224, this gives
N = (224/16)² = **196 patches**. The positional embeddings were trained
for exactly 196 positions. At 64×64 we would get only 16 patches, which
is a fundamentally different sequence length that the model was not
trained to handle.

**Why ImageNet-21k pre-training for ViT?**
Unlike CNNs, ViT has weaker inductive biases (no built-in assumption
about local structure). It requires more pre-training data to learn
useful features. ImageNet-21k (~14M images, 21K classes) provides a
much richer starting point than ImageNet-1k (~1.2M images, 1K classes).

---

### Step 6 — Training Engine (`train.py`)

All three models use the same training loop:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
```

**Why AdamW over Adam?**
In standard Adam, weight decay is coupled with the adaptive learning rate
in a way that reduces its regularising effect. AdamW decouples them,
giving true L2 regularisation and generally better generalisation.

**Why ReduceLROnPlateau?**
Different models converge at different speeds. A fixed schedule would
require separate tuning per model. ReduceLROnPlateau adapts automatically:
if validation loss hasn't improved for 3 epochs, the LR is halved.

**Best model checkpointing:**
The checkpoint with the highest validation accuracy is saved, not the
final epoch. Training loss always falls; validation loss may begin rising
(overfitting). Using the best checkpoint prevents evaluating an overfitted
model on the test set.

---

### Step 7 — Evaluation and Plots (`evaluate.py`)

After training, for each model:

1. **Collect predictions** over the test set
2. **Compute metrics**: overall accuracy, macro F1, per-class report
3. **Plot confusion matrix**: row-normalised (shows per-class recall)
4. **Plot training curves**: loss and accuracy vs. epoch

After all models are trained:

5. **Side-by-side confusion matrix comparison** — all three in one figure
6. **Model comparison bar chart** — accuracy and macro F1 side by side

All figures are saved to `./outputs/`.

---

## Running the Full Pipeline

```bash
cd eurosat/
python main.py
```

Expected output files:
```
outputs/
├── BaselineCNN_best.pth
├── BaselineCNN_curves.png
├── BaselineCNN_confusion_matrix.png
├── resnet50_final_best.pth
├── ResNet50_curves.png
├── ResNet50_confusion_matrix.png
├── ViT-B16_best.pth
├── ViT-B16_curves.png
├── ViT-B16_confusion_matrix.png
├── confusion_matrix_comparison.png
└── model_comparison.png
```
---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 16 in `config.py` |
| Slow training on CPU | Reduce `NUM_EPOCHS` to 10 |
| `timm` not found | `pip install timm` |
| Windows `num_workers` error | Set `NUM_WORKERS = 0` in `config.py` |
| Download fails | Download manually from https://zenodo.org/records/7711810 and extract to `./data/eurosat/` |

---

## References
References outlined in the report are downloaded in the reference folder for review
