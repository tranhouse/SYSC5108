"""
models.py
---------
Defines the three deep learning models used in this project:

  1. BaselineCNN  — trained from scratch, no pre-training.
  2. ResNet50     — ImageNet pre-trained, two-phase fine-tuning.
  3. ViT_B16      — ImageNet-21k pre-trained, fine-tuned on EuroSAT.

RATIONALE
----------
Having all three models in one file makes it easy to see the architectural
differences side by side and ensures the classification head design is
consistent across models.

Each builder function returns a model ready for training — the caller does
not need to know whether weights were loaded from a checkpoint or randomly
initialised.
"""

import torch.nn as nn
import torchvision.models as tv_models
import timm

import config


# ─── Model 1: Baseline CNN ────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    A compact 4-block convolutional network trained from scratch.

    PURPOSE
    -------
    Acts as the lower-bound reference. Any gap between this model and the
    pre-trained models quantifies the value of transfer learning.

    ARCHITECTURE
    ------------
    Each block: Conv2D → BatchNorm → ReLU → MaxPool
    Channel progression: 3 → 32 → 64 → 128 → 256

    BatchNorm is included in each block because it:
      - Stabilises training by normalising activations layer by layer
      - Acts as a mild regulariser
      - Allows higher learning rates

    AdaptiveAvgPool2d at the end collapses the spatial map to a fixed size
    regardless of the input resolution, keeping the classifier head simple.

    Dropout (p=0.4) before the final linear layer reduces overfitting on
    the training set.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8x8 → 4x4 (via adaptive pooling)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── Model 2: ResNet50 with Transfer Learning ─────────────────────────────────

def build_resnet50(num_classes: int = config.NUM_CLASSES,
                   freeze_backbone: bool = False) -> nn.Module:
    """
    ResNet50 with ImageNet pre-trained weights. The final fully connected
    layer is replaced with a new classification head for `num_classes` outputs.

    RATIONALE FOR ARCHITECTURE CHOICES
    ------------------------------------
    Pre-trained weights:
        ResNet50 pre-trained on ImageNet already knows how to detect edges,
        textures, shapes, and high-level visual patterns. Fine-tuning adapts
        these features to satellite imagery without learning everything from
        scratch, which requires far less data and training time.

    Replaced FC layer:
        The original ResNet50 FC layer has 1000 outputs (ImageNet classes).
        We replace it with a two-layer head: Linear → ReLU → Dropout → Linear.
        The intermediate layer gives the model some capacity to re-map the
        2048-dimensional backbone features to the new 10-class space.

    freeze_backbone=True (Phase 1):
        The randomly initialised head produces large gradient signals at the
        start of training. If the backbone is unfrozen at this point, those
        large gradients propagate back and corrupt the pre-trained weights
        before the head has had any chance to stabilise. Freezing the backbone
        for a few epochs lets the head converge first.

    Parameters
    ----------
    freeze_backbone : bool
        If True, all backbone parameters have requires_grad=False.
        Used for Phase 1 training. Call with False for Phase 2.
    """
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classification head (backbone output dim is 2048)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    # The new head always has requires_grad=True regardless of freeze_backbone
    return model


def unfreeze_resnet(model: nn.Module) -> nn.Module:
    """
    Unfreezes all backbone parameters for Phase 2 fine-tuning.
    Called after Phase 1 training is complete.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


# ─── Model 3: ViT-B/16 with Transfer Learning ─────────────────────────────────

def build_vit(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """
    Vision Transformer ViT-B/16 with ImageNet-21k pre-trained weights,
    loaded via the `timm` library.

    RATIONALE
    ----------
    timm (PyTorch Image Models) is the standard library for pre-trained
    vision models in research. It has ViT-B/16 with multiple checkpoint
    options. We use the ImageNet-21k pre-trained variant because:
      - ImageNet-21k has ~14M images across 21,000 classes
      - A richer pre-training distribution improves transfer to satellite
        imagery, which looks very different from standard ImageNet photos
      - ViT generally requires more pre-training data than CNNs to work well
        (its inductive biases are weaker), so the larger pre-training set helps

    timm's create_model() with num_classes= automatically replaces the head.

    Input resolution:
        ViT-B/16 divides the image into 16x16 patches. At 224x224, this gives
        N = (224/16)^2 = 196 patches. The positional embeddings were trained
        for exactly this number of patches. Feeding 64x64 images would give
        N = 16 patches, requiring interpolation of positional embeddings and
        degrading transfer quality. We resize all images to 224x224 for ViT.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


# ─── Parameter Count Utility ──────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    print(f"[Model] {name}")
    print(f"        Total parameters    : {total:,}")
    print(f"        Trainable parameters: {trainable:,}")
