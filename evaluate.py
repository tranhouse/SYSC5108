"""
evaluate.py — updated
----------------------
Produces all outputs referenced in the report:

  Figures
  -------
  {model_name}_curves.png            Training/validation loss and accuracy
                                     ResNet50 version includes phase boundary line
  {model_name}_confusion_matrix.png  Normalised confusion matrix per model
  confusion_matrix_comparison.png    All three matrices side by side (Fig 4.7)
  model_comparison.png               Accuracy and macro F1 bar chart (Fig 4.8)

  Data files
  ----------
  results_summary.json               Accuracy, F1, precision, recall for all models
                                     → use to fill Tables 4.1 and 4.2 in report
  {model_name}_per_class.csv         Per-class precision, recall, F1
  top_misclassifications.csv         Most confused class pairs across all models
                                     → use to fill misclassification table in report
"""

import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

import config


# ─── Collect Predictions ──────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader) -> tuple:
    device = config.DEVICE
    model.eval()
    model.to(device)
    all_true, all_pred = [], []
    for images, labels in tqdm(loader, desc="  Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
    return np.array(all_true), np.array(all_pred)


# ─── Compute Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  Test Results: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {macro_p:.4f}  (macro)")
    print(f"  Recall     : {macro_r:.4f}  (macro)")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"\n  Per-Class Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=config.CLASSES, digits=4))

    return {"accuracy": acc, "macro_f1": macro_f1,
            "macro_precision": macro_p, "macro_recall": macro_r}


# ─── Save Results Summary JSON ────────────────────────────────────────────────

def save_results_summary(all_metrics):
    """
    Saves a JSON with all model metrics.
    Use the values here to fill Tables 4.1 and 4.2 in the report.
    """
    path = os.path.join(config.OUTPUT_DIR, "results_summary.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  [Saved] results_summary.json  → {path}")
    print(f"\n  Table fill-in values:")
    print(f"  {'Model':<20} {'Accuracy':>10}  {'Precision':>10}"
          f"  {'Recall':>10}  {'F1':>10}")
    print(f"  {'-'*64}")
    for name, m in all_metrics.items():
        print(f"  {name:<20} "
              f"{m['accuracy']*100:>9.2f}%  "
              f"{m['macro_precision']*100:>9.2f}%  "
              f"{m['macro_recall']*100:>9.2f}%  "
              f"{m['macro_f1']*100:>9.2f}%")


# ─── Save Per-Class CSV ───────────────────────────────────────────────────────

def save_per_class_report(y_true, y_pred, model_name):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    path = os.path.join(config.OUTPUT_DIR, f"{model_name}_per_class.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
        for i, cls in enumerate(config.CLASSES):
            writer.writerow([cls, f"{precision[i]:.4f}",
                             f"{recall[i]:.4f}", f"{f1[i]:.4f}", int(support[i])])
    print(f"  [Saved] Per-class CSV         → {path}")


# ─── Top Misclassification Table ──────────────────────────────────────────────

def save_top_misclassifications(all_results, top_n=10):
    """
    Aggregates off-diagonal confusion counts across all models and saves the
    most common true→predicted confusion pairs to CSV.
    This produces the equivalent of Table 4-3 in the reference ASL report.
    """
    K = config.NUM_CLASSES
    agg_cm = np.zeros((K, K), dtype=int)
    for res in all_results.values():
        agg_cm += confusion_matrix(res["y_true"], res["y_pred"],
                                   labels=list(range(K)))

    pairs = []
    for ti in range(K):
        for pi in range(K):
            if ti != pi and agg_cm[ti, pi] > 0:
                pairs.append({
                    "true_class":      config.CLASSES[ti],
                    "predicted_class": config.CLASSES[pi],
                    "count":           int(agg_cm[ti, pi]),
                })
    pairs = sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_n]

    path = os.path.join(config.OUTPUT_DIR, "top_misclassifications.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["true_class", "predicted_class", "count"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"\n  [Saved] Top misclassifications → {path}")
    print(f"\n  {'True Class':<25} {'Predicted As':<25} {'Count':>6}")
    print(f"  {'-'*58}")
    for p in pairs:
        print(f"  {p['true_class']:<25} {p['predicted_class']:<25}"
              f" {p['count']:>6}")
    return pairs


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history, model_name, phase1_epochs=None):
    """
    Plots loss and accuracy curves.
    phase1_epochs : if set (ResNet50 only), draws a vertical dashed line
                    at the Phase 1 / Phase 2 boundary. The report caption
                    explicitly references this boundary at epoch 5.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=13)

    for ax, loss_key, acc_key, ylabel, title in [
        (axes[0], "train_loss", "val_loss",  "Cross-Entropy Loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",   "Accuracy (%)",        "Accuracy"),
    ]:
        train_vals = history[loss_key] if "loss" in loss_key else \
                     [v * 100 for v in history[acc_key]]
        val_vals   = history[loss_key.replace("train", "val")] \
                     if "loss" in loss_key else \
                     [v * 100 for v in history[acc_key.replace("train", "val")]]

        ax.plot(epochs, train_vals, marker="o", markersize=3, label="Train")
        ax.plot(epochs, val_vals,   marker="s", markersize=3,
                linestyle="--", label="Validation")

        if phase1_epochs is not None:
            ax.axvline(x=phase1_epochs + 0.5, color="gray",
                       linestyle=":", linewidth=1.5)
            y_pos = max(train_vals) * 0.92 if "loss" in loss_key \
                    else min(val_vals) + 1
            ax.text(phase1_epochs + 0.8, y_pos,
                    "← P1 | P2 →", fontsize=8, color="gray")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Training curves        → {path}")


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short   = ["AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
               "Pasture", "PermCrop", "Residential", "River", "SeaLake"]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=short, yticklabels=short,
                ax=ax, vmin=0, vmax=1)
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{model_name} — Normalised Confusion Matrix"
                 f"  (Acc: {acc*100:.2f}%)", fontsize=12)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR,
                        f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Confusion matrix       → {path}")


# ─── Side-by-Side Comparison ──────────────────────────────────────────────────

def plot_confusion_matrix_comparison(all_results):
    short = ["AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
             "Pasture", "PermCrop", "Residential", "River", "SeaLake"]
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 9))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        cm      = confusion_matrix(res["y_true"], res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=short, yticklabels=short,
                    ax=ax, vmin=0, vmax=1, cbar=False)
        acc = accuracy_score(res["y_true"], res["y_pred"])
        ax.set_title(f"{name}\nAcc: {acc*100:.2f}%", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.suptitle("Normalised Confusion Matrix Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "confusion_matrix_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Comparison matrix      → {path}")


# ─── Model Comparison Bar Chart ───────────────────────────────────────────────

def plot_model_comparison(results):
    names = list(results.keys())
    accs  = [results[m]["accuracy"]  * 100 for m in names]
    f1s   = [results[m]["macro_f1"] * 100 for m in names]
    x, w  = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, accs, w, label="Test Accuracy (%)")
    b2 = ax.bar(x + w/2, f1s,  w, label="Macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — EuroSAT Test Set")
    ax.set_ylim(0, 108)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Model comparison       → {path}")


# ─── Full Evaluation for One Model ────────────────────────────────────────────

def evaluate_model(model, test_loader, model_name):
    y_true, y_pred = get_predictions(model, test_loader)
    metrics = compute_metrics(y_true, y_pred, model_name)
    save_per_class_report(y_true, y_pred, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    return {**metrics, "y_true": y_true, "y_pred": y_pred}
