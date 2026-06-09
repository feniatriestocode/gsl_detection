from pathlib import Path

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
import typer

from gsl_detect.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, CSV, REPORTS_DIR
)
from gsl_detect.modeling.dataset import GSLDataset, collate_fn
from gsl_detect.modeling.model import GSLTransformer
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

app = typer.Typer()

def plot_test_summary(top1, top5, f1, save_path=None):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    })
    metrics = {"Top-1 Acc": top1, "Top-5 Acc": top5, "Weighted F1": f1}
    
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#5a9e6f", "#4a8abf", "#e0a045"], edgecolor="none")
    
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Performance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_inference(model, loader, device, num_classes):
    """Run model over the test loader, return all logits, preds and labels."""
    model.eval()

    top1_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    top5_metric = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    f1_metric   = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    cm_metric   = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for sequences, masks, labels in tqdm(loader, desc="Evaluating"):
            sequences = sequences.to(device)
            masks     = masks.to(device)
            labels    = labels.to(device)

            logits = model(sequences, masks)

            top1_metric.update(logits, labels)
            top5_metric.update(logits, labels)
            f1_metric.update(logits, labels)
            cm_metric.update(logits, labels)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "top1":   top1_metric.compute().item(),
        "top5":   top5_metric.compute().item(),
        "f1":     f1_metric.compute().item(),
        "cm":     cm_metric.compute().cpu().numpy(),
        "preds":  all_preds,
        "labels": all_labels,
    }


def plot_confusion_top_n(cm, label_names, n=20, save_path=None):
    """Plot the N most confused class pairs as a ranked horizontal bar chart."""
    rows, cols = np.where(cm > 0)
    confused_pairs = []
    for r, c in zip(rows, cols):
        if r != c:
            confused_pairs.append((cm[r, c], label_names[r], label_names[c]))

    confused_pairs.sort(reverse=True)
    top_pairs = confused_pairs[:n]

    labels_plot = [f"{true} → {pred}" for _, true, pred in top_pairs]
    counts      = [count for count, _, _ in top_pairs]

    fig, ax = plt.subplots(figsize=(8, n * 0.38 + 1.2))
    bars = ax.barh(labels_plot[::-1], counts[::-1], color="#e07b54", edgecolor="none")
    ax.set_xlabel("Misclassification count")
    ax.set_title(f"Top-{n} Confused Class Pairs", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion pairs plot → {save_path}")
    plt.close(fig)


def plot_per_class_accuracy(cm, label_names, save_path=None):
    """Bar chart of per-class accuracy, sorted ascending."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    sorted_idx = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[sorted_idx]
    sorted_names = [label_names[i] for i in sorted_idx]

    # Show all classes but only label worst/best 20
    n = len(sorted_names)
    tick_labels = [""] * n
    for i in range(min(20, n)):
        tick_labels[i] = sorted_names[i]           # 20 worst
    for i in range(max(0, n - 20), n):
        tick_labels[i] = sorted_names[i]           # 20 best

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#d94f3d" if a < 0.5 else "#5a9e6f" if a >= 0.8 else "#e0a045"
              for a in sorted_acc]
    ax.bar(range(n), sorted_acc, color=colors, edgecolor="none", width=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(sorted_acc.mean(), color="black", linewidth=1,
               linestyle="--", label=f"Mean: {sorted_acc.mean():.2%}")
    ax.set_title("Per-Class Accuracy (sorted ascending)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved per-class accuracy plot → {save_path}")
    plt.close(fig)


def plot_confusion_matrix_submatrix(cm, label_names, n_worst=30, save_path=None):
    """Heatmap of the N worst-performing classes (readable submatrix)."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    worst_idx = np.argsort(per_class_acc)[:n_worst]
    sub_cm    = cm[np.ix_(worst_idx, worst_idx)]
    sub_names = [label_names[i] for i in worst_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_cm, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(n_worst))
    ax.set_yticks(range(n_worst))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=7)
    ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {n_worst} Hardest Classes",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion submatrix → {save_path}")
    plt.close(fig)


@app.command()
def main(
    data_dir:   Path  = PROCESSED_DATA_DIR,
    csv_path:   Path  = CSV,
    model_path: Path  = MODELS_DIR / "best_model.pt",
    d_model:    int   = 256,
    nhead:      int   = 4,
    num_layers: int   = 3,
    batch_size: int   = 64,
    top_n_confused: int = 20,
    n_worst_classes: int = 30,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_dataset = GSLDataset(csv_path, data_dir / "train")
    test_dataset  = GSLDataset(
        csv_path, data_dir / "test",
        label_encoder=train_dataset.label_encoder
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    num_classes  = train_dataset.num_classes
    label_names  = list(train_dataset.label_to_gloss)   # index → gloss string
    logger.info(f"Classes: {num_classes} | Test samples: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GSLTransformer(
        input_dim=186,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    results = run_inference(model, test_loader, device, num_classes)

    logger.success(f"Top-1 Accuracy : {results['top1']:.4%}")
    logger.success(f"Top-5 Accuracy : {results['top5']:.4%}")
    logger.success(f"Weighted F1    : {results['f1']:.4f}")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    cm = results["cm"]
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    per_class_f1  = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        per_class_f1.append(float(2 * tp / denom) if denom > 0 else 0.0)

    per_class_df = pd.DataFrame({
        "gloss":    label_names,
        "accuracy": per_class_acc,
        "f1":       per_class_f1,
        "support":  cm.sum(axis=1).astype(int),
    }).sort_values("accuracy")

    # ── Save metrics summary ──────────────────────────────────────────────────
    summary = {
        "top1_accuracy": round(results["top1"], 6),
        "top5_accuracy": round(results["top5"], 6),
        "weighted_f1":   round(results["f1"],   6),
        "num_classes":   num_classes,
        "test_samples":  len(test_dataset),
    }
    summary_path = REPORTS_DIR / "test_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved metrics summary → {summary_path}")

    per_class_csv_path = REPORTS_DIR / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class metrics → {per_class_csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots...")

    plot_per_class_accuracy(
        cm, label_names,
        save_path=FIGURES_DIR / "per_class_accuracy.pdf"
    )
    plot_confusion_top_n(
        cm, label_names,
        n=top_n_confused,
        save_path=FIGURES_DIR / "top_confused_pairs.pdf"
    )
    plot_confusion_matrix_submatrix(
        cm, label_names,
        n_worst=n_worst_classes,
        save_path=FIGURES_DIR / "confusion_submatrix.pdf"
    )

    plot_test_summary(
    results["top1"], results["top5"], results["f1"],
    save_path=FIGURES_DIR / "test_summary.pdf"
    )

    logger.success("Evaluation complete.")
    logger.success(f"Figures saved to: {FIGURES_DIR}")
    logger.success(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    app()