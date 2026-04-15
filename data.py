"""
data.py
-------
Handles everything related to the EuroSAT dataset:
  - Downloading via torchvision
  - Computing normalisation statistics from the training split
  - Stratified train / val / test splitting
  - Augmentation and transform pipelines
  - DataLoader construction
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT
from sklearn.model_selection import train_test_split

import config


# ─── Step 1: Download the Dataset ─────────────────────────────────────────────

def get_raw_dataset(img_size: int) -> EuroSAT:
    """
    Downloads EuroSAT (RGB, 64x64) if not already present and returns the
    full dataset with only a ToTensor transform. This is used for computing
    normalisation statistics and for building split subsets.

    Parameters
    ----------
    img_size : int
        Target image size. If 64, no resize is applied (native resolution).
        If 224, images are resized to satisfy ViT's input requirement.
    """
    if img_size == config.IMG_SIZE_CNN:
        base_transform = transforms.ToTensor()
    else:
        base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    os.makedirs(config.DATA_DIR, exist_ok=True)

    dataset = EuroSAT(
        root=config.DATA_DIR,
        transform=base_transform,
        download=True,
    )
    return dataset


# ─── Step 2: Stratified Split ─────────────────────────────────────────────────

def make_splits(dataset: EuroSAT):
    """
    Stratified split
    A plain random split could, by chance, over-represent one class in the test set and under-represent it in training. Stratification guarantees
    that every class appears in each split at its natural frequency, making per-class metrics meaningful and comparable across models.
    Produces stratified train / val / test index arrays.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    # First cut: separate training from the rest.
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - config.TRAIN_RATIO),
        stratify=targets,
        random_state=config.SEED,
    )

    # Second cut: split the remainder equally into val and test.
    val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=targets[temp_idx],
        random_state=config.SEED,
    )

    print(f"[Data] Split sizes → Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# ─── Step 3: Compute Normalisation Statistics ──────────────────────────────────

def compute_mean_std(dataset: EuroSAT, train_idx: np.ndarray) -> tuple:
    """
    Compute normalisation from training data only if we computed mean/std across the whole dataset, information from the
    validation and test sets would leak into the training preprocessing.
    By computing from the training subset only we keep the evaluation sets truly held-out.
    
    Computes per-channel mean and standard deviation over the training subset.

    Uses a running accumulation rather than loading everything into memory,
    making it safe for large datasets.

    Parameters
    ----------
    dataset   : EuroSAT with ToTensor transform applied (values in [0, 1]).
    train_idx : Indices of the training split.

    Returns
    -------
    mean, std : torch.Tensor of shape (3,)
    """
    print("[Data] Computing normalisation statistics from training split ...")

    train_subset = Subset(dataset, train_idx)
    loader = DataLoader(
        train_subset,
        batch_size=256,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_samples = 0

    for images, _ in loader:
        b = images.size(0)
        # Flatten spatial dimensions: (B, C, H, W) → (B, C, H*W)
        images = images.view(b, 3, -1)
        mean += images.mean(dim=2).sum(dim=0)
        std  += images.std(dim=2).sum(dim=0)
        n_samples += b

    mean /= n_samples
    std  /= n_samples

    print(f"[Data] Mean: {mean.tolist()}")
    print(f"[Data] Std : {std.tolist()}")
    return mean, std


# ─── Step 4: Build Transform Pipelines ────────────────────────────────────────

def build_transforms(img_size: int, mean: torch.Tensor, std: torch.Tensor):
    """
    Returns (train_transform, eval_transform) using the computed statistics.

    Train transform includes augmentation.
    Flip and rotation augmentations
        Satellite images are taken from directly above, so there is no natural "up" direction. Horizontal flips, vertical flips, and arbitrary rotations
        are all realistic transformations for this domain.

    Eval transform is normalisation only (no augmentation).
    """
    mean_list = mean.tolist()
    std_list  = std.tolist()

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # ±15° rotation. Larger angles risk rotating road/river features
        # out of the patch, which would corrupt the label.
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])

    return train_transform, eval_transform


# ─── Step 5: Build DataLoaders ────────────────────────────────────────────────

def build_dataloaders(img_size: int):
    """
    Separate image sizes for CNN vs ViT
    ViT-B/16 was pre-trained on 224x224. Feeding it 64x64 images would change the number of patches (N = HW/P^2) relative to what the positional
    embeddings were trained on, degrading performance significantly. Baseline CNN and ResNet50 take the native 64x64.

    Full pipeline: download → split → compute stats → build loaders.

    Parameters
    ----------
    img_size : int
        64 for Baseline CNN and ResNet50; 224 for ViT-B/16.

    Returns
    -------
    train_loader, val_loader, test_loader, mean, std
    """
    # Download and get raw dataset (ToTensor only, no normalisation yet)
    raw_dataset = get_raw_dataset(img_size=config.IMG_SIZE_CNN)

    # Stratified split
    train_idx, val_idx, test_idx = make_splits(raw_dataset)

    # Compute normalisation statistics from training split only
    mean, std = compute_mean_std(raw_dataset, train_idx)

    # Build transform pipelines with the computed statistics
    train_tf, eval_tf = build_transforms(img_size, mean, std)

    # Build three separate dataset instances, each with the correct transform.
    # We cannot reuse one instance because transforms are set at construction.
    def make_split_dataset(transform):
        return EuroSAT(
            root=config.DATA_DIR,
            transform=transform,
            download=False,   # already downloaded above
        )

    train_ds = Subset(make_split_dataset(train_tf), train_idx)
    val_ds   = Subset(make_split_dataset(eval_tf),  val_idx)
    test_ds  = Subset(make_split_dataset(eval_tf),  test_idx)

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, mean, std
