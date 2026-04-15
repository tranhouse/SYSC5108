"""
main.py — updated
------------------
Orchestrates the full pipeline and produces every output the report needs.

Outputs after a complete run
-----------------------------
outputs/
  BaselineCNN_best.pth
  BaselineCNN_curves.png
  BaselineCNN_confusion_matrix.png
  BaselineCNN_per_class.csv

  resnet50_final_best.pth
  ResNet50_curves.png                  ← includes Phase 1/2 boundary annotation
  ResNet50_confusion_matrix.png
  ResNet50_per_class.csv

  ViT-B16_best.pth
  ViT-B16_curves.png
  ViT-B16_confusion_matrix.png
  ViT-B16_per_class.csv

  confusion_matrix_comparison.png      ← Fig 4.7 in report
  model_comparison.png                 ← Fig 4.8 in report
  top_misclassifications.csv           ← misclassification table in report
  results_summary.json                 ← fill Tables 4.1 and 4.2 from here
"""

import random
import os
import numpy as np
import torch

import config
from data import build_dataloaders
from models import BaselineCNN, build_vit, print_model_summary
from train import run_training, train_resnet_two_phase
from evaluate import (
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix_comparison,
    plot_model_comparison,
    save_results_summary,
    save_top_misclassifications,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"[Config] Device    : {config.DEVICE}")
    print(f"[Config] Batch size: {config.BATCH_SIZE}")
    print(f"[Config] Epochs    : {config.NUM_EPOCHS}")

    all_results  = {}   # y_true, y_pred per model — for comparison plots
    metric_table = {}   # accuracy, F1, precision, recall — for JSON and tables

    # ── Step 2: Data at 64x64 (CNN + ResNet50) ────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 2: Dataloaders at 64×64 (CNN / ResNet50)")
    print("─"*55)
    train_64, val_64, test_64, _, _ = build_dataloaders(
        img_size=config.IMG_SIZE_CNN)

    # ── Step 3: Baseline CNN ──────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 3: Baseline CNN (trained from scratch)")
    print("─"*55)
    cnn_model = BaselineCNN(num_classes=config.NUM_CLASSES)
    print_model_summary("Baseline CNN", cnn_model)
    cnn_model, cnn_history = run_training(
        cnn_model, train_64, val_64, model_name="BaselineCNN")

    # No phase boundary for baseline CNN
    plot_training_curves(cnn_history, "BaselineCNN")
    cnn_res = evaluate_model(cnn_model, test_64, "BaselineCNN")
    all_results["Baseline CNN"]  = cnn_res
    metric_table["Baseline CNN"] = {k: v for k, v in cnn_res.items()
                                    if k not in ("y_true", "y_pred")}
    del cnn_model
    torch.cuda.empty_cache()

    # ── Step 4: ResNet50 two-phase ────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 4: ResNet50 with Transfer Learning (two-phase)")
    print("─"*55)
    resnet_model, resnet_history = train_resnet_two_phase(train_64, val_64)

    # Pass phase1_epochs so the curve plot shows the Phase 1/2 boundary line.
    # The report caption explicitly references this boundary at epoch 5.
    plot_training_curves(resnet_history, "ResNet50",
                         phase1_epochs=config.RESNET_PHASE1_EPOCHS)

    resnet_res = evaluate_model(resnet_model, test_64, "ResNet50")
    all_results["ResNet50"]  = resnet_res
    metric_table["ResNet50"] = {k: v for k, v in resnet_res.items()
                                if k not in ("y_true", "y_pred")}
    del resnet_model
    torch.cuda.empty_cache()

    # ── Step 5: Data at 224x224 (ViT) ────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 5: Dataloaders at 224×224 (ViT-B/16)")
    print("─"*55)
    train_224, val_224, test_224, _, _ = build_dataloaders(
        img_size=config.IMG_SIZE_VIT)

    # ── Step 6: ViT-B/16 ─────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 6: ViT-B/16 with Transfer Learning")
    print("─"*55)
    vit_model = build_vit(num_classes=config.NUM_CLASSES)
    print_model_summary("ViT-B/16", vit_model)
    vit_model, vit_history = run_training(
        vit_model, train_224, val_224, model_name="ViT-B16")

    plot_training_curves(vit_history, "ViT-B16")
    vit_res = evaluate_model(vit_model, test_224, "ViT-B16")
    all_results["ViT-B/16"]  = vit_res
    metric_table["ViT-B/16"] = {k: v for k, v in vit_res.items()
                                if k not in ("y_true", "y_pred")}
    del vit_model
    torch.cuda.empty_cache()

    # ── Step 7: Comparison outputs ────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Step 7: Generating comparison outputs")
    print("─"*55)

    # Fig 4.7 — side-by-side confusion matrices
    plot_confusion_matrix_comparison(all_results)

    # Fig 4.8 — accuracy and F1 bar chart
    plot_model_comparison(metric_table)

    # Misclassification table — equivalent of Table 4-3 in the ASL report
    save_top_misclassifications(all_results, top_n=10)

    # Results JSON — fill in Tables 4.1 and 4.2 from this file
    save_results_summary(metric_table)

    print(f"\n  All outputs saved to: {os.path.abspath(config.OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
