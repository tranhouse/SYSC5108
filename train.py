"""
train.py
--------
Contains the training and validation loop used by all three models.

RATIONALE FOR TRAINING CHOICES
--------------------------------
Loss — Cross-Entropy:
    Standard choice for multi-class classification. Cross-entropy penalises
    confident wrong predictions heavily, which encourages the model to be
    well-calibrated.

Optimiser — AdamW:
    Adam with decoupled weight decay. AdamW is preferred over vanilla Adam
    for fine-tuning because weight decay in Adam was shown to be coupled with
    the adaptive learning rate in a way that reduces its regularising effect.
    AdamW fixes this, giving better generalisation especially with pre-trained
    models.

Scheduler — ReduceLROnPlateau:
    Halves the learning rate when validation loss has not improved for
    `patience` epochs. This is appropriate here because we do not know in
    advance how many epochs each model needs to converge. A fixed schedule
    would require separate tuning for each model.

Best-model checkpointing:
    We save the model weights corresponding to the best validation accuracy,
    not the final epoch. Training loss will always decrease monotonically
    (the model memorises the training data), but the model that generalises
    best to unseen data is the one with the lowest validation loss / highest
    validation accuracy. Using final-epoch weights risks using an overfitted
    model for test evaluation.
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one full pass over the training set.

    Returns
    -------
    avg_loss : float — mean cross-entropy loss over all samples
    accuracy : float — fraction of correctly classified samples
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping prevents occasional large gradient spikes from
        # destabilising training, particularly useful with Transformers.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on a validation or test loader without updating
    any weights.

    @torch.no_grad() disables gradient tracking for the entire function,
    which reduces memory usage and speeds up evaluation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def run_training(model, train_loader, val_loader, model_name: str,
                 num_epochs: int = None, lr: float = None):
    """
    Full training loop for one model.

    Parameters
    ----------
    model       : nn.Module — the model to train (already on the correct device)
    train_loader: DataLoader
    val_loader  : DataLoader
    model_name  : str — used for checkpoint filenames and print statements
    num_epochs  : int — overrides config.NUM_EPOCHS if provided
    lr          : float — overrides config.LEARNING_RATE if provided

    Returns
    -------
    model   : nn.Module with best validation weights restored
    history : dict with keys train_loss, val_loss, train_acc, val_acc
    """
    device     = config.DEVICE
    num_epochs = num_epochs or config.NUM_EPOCHS
    lr         = lr or config.LEARNING_RATE

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Only pass parameters that require gradients to the optimiser.
    # During Phase 1 of ResNet50 training the backbone is frozen, so
    # filter(requires_grad) ensures the optimiser only updates the head.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Patience of 3 means the LR is halved after 3 consecutive epochs
    # with no improvement in validation loss.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_acc  = 0.0
    best_weights  = None
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}   ({num_epochs} epochs)")
    print(f"  Device  : {device}")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% "
            f"| LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(
                config.OUTPUT_DIR, f"{model_name}_best.pth"
            )
            torch.save(best_weights, ckpt_path)
            print(f"    ✓ New best val accuracy: {best_val_acc*100:.2f}%"
                  f"  → saved to {ckpt_path}")

    # Restore best weights before returning
    model.load_state_dict(best_weights)
    print(f"\n  Best validation accuracy: {best_val_acc*100:.2f}%")
    return model, history


def train_resnet_two_phase(train_loader, val_loader):
    """
    Two-phase fine-tuning strategy for ResNet50.

    Phase 1 — Backbone frozen, head only (5 epochs):
        The randomly initialised head is trained until it produces
        reasonable outputs. This prevents large gradient signals from
        propagating into the carefully pre-trained backbone.

    Phase 2 — All layers unfrozen, full fine-tuning (15 epochs):
        Now that the head is stable, we unfreeze the backbone and allow
        the entire network to adapt to satellite imagery. A lower learning
        rate is used to make small, careful adjustments to the backbone
        features rather than overwriting them.

    Returns
    -------
    model   : ResNet50 with best weights from Phase 2
    history : merged history dict across both phases
    """
    from models import build_resnet50, unfreeze_resnet, print_model_summary

    print("\n[ResNet50] Phase 1 — training head only (backbone frozen)")
    model = build_resnet50(freeze_backbone=True)
    print_model_summary("ResNet50 (Phase 1, frozen backbone)", model)

    model, h1 = run_training(
        model, train_loader, val_loader,
        model_name="resnet50_phase1",
        num_epochs=config.RESNET_PHASE1_EPOCHS,
    )

    print("\n[ResNet50] Phase 2 — full fine-tuning (all layers unfrozen)")
    model = unfreeze_resnet(model)
    print_model_summary("ResNet50 (Phase 2, all unfrozen)", model)

    # Rebuild optimiser so it includes the now-unfrozen backbone parameters.
    # The old optimiser only tracked the head parameters.
    model, h2 = run_training(
        model, train_loader, val_loader,
        model_name="resnet50_final",
        num_epochs=config.RESNET_PHASE2_EPOCHS,
        lr=config.LEARNING_RATE * 0.1,   # 10x lower LR for fine-tuning
    )

    # Merge the two history dicts so we can plot one continuous curve
    combined_history = {k: h1[k] + h2[k] for k in h1}
    return model, combined_history
