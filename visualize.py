"""
============================================================
Financial News Sentiment & Risk Prediction System
Visualization Module
============================================================
Contains:
  - Training loss curves (total, sentiment, risk)
  - Accuracy curves
  - Confusion matrix heatmap
  - Risk score distribution
  - Sentiment distribution pie chart
  - Prediction summary dashboard
============================================================
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from config import SENTIMENT_LABELS, PLOTS_DIR


# ── Style Configuration ──
plt.style.use("dark_background")
COLORS = {
    "primary":   "#00D2FF",
    "secondary": "#FF6B6B",
    "accent":    "#7C4DFF",
    "positive":  "#4CAF50",
    "negative":  "#F44336",
    "neutral":   "#FFC107",
    "bg":        "#1a1a2e",
    "grid":      "#2a2a4a",
}


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss curves.
    
    Shows three subplots:
      1. Total Loss (train vs val)
      2. Sentiment Loss
      3. Risk Loss
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # ── Total Loss ──
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(epochs, history["train_loss"], "-o", color=COLORS["primary"],
            label="Train", linewidth=2, markersize=5)
    ax.plot(epochs, history["val_loss"], "-s", color=COLORS["secondary"],
            label="Validation", linewidth=2, markersize=5)
    ax.set_title("Total Loss", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    # ── Sentiment Loss ──
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(epochs, history["train_sent_loss"], "-o", color=COLORS["primary"],
            label="Train", linewidth=2, markersize=5)
    ax.plot(epochs, history["val_sent_loss"], "-s", color=COLORS["secondary"],
            label="Validation", linewidth=2, markersize=5)
    ax.set_title("Sentiment Loss (CrossEntropy)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    # ── Risk Loss ──
    ax = axes[2]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(epochs, history["train_risk_loss"], "-o", color=COLORS["primary"],
            label="Train", linewidth=2, markersize=5)
    ax.plot(epochs, history["val_risk_loss"], "-s", color=COLORS["secondary"],
            label="Validation", linewidth=2, markersize=5)
    ax.set_title("Risk Loss (MSE)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    plt.tight_layout()
    
    save_path = save_path or os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Visualize] Saved training curves → {save_path}")


def plot_accuracy_curves(history, save_path=None):
    """
    Plot training and validation accuracy curves.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    
    epochs = range(1, len(history["train_sent_acc"]) + 1)
    
    ax.plot(epochs, history["train_sent_acc"], "-o", color=COLORS["primary"],
            label="Train Accuracy", linewidth=2, markersize=6)
    ax.plot(epochs, history["val_sent_acc"], "-s", color=COLORS["secondary"],
            label="Val Accuracy", linewidth=2, markersize=6)
    
    ax.set_title("Sentiment Classification Accuracy", fontsize=14,
                 fontweight="bold", color="white")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Accuracy", color="white")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.3, fontsize=11)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    plt.tight_layout()
    
    save_path = save_path or os.path.join(PLOTS_DIR, "accuracy_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Visualize] Saved accuracy curves → {save_path}")


def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot a styled confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    
    labels = [SENTIMENT_LABELS[i] for i in range(len(SENTIMENT_LABELS))]
    
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=2, linecolor=COLORS["bg"],
        annot_kws={"size": 16, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    
    ax.set_title("Confusion Matrix — Sentiment Classification",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Predicted", fontsize=12, color="white")
    ax.set_ylabel("Actual", fontsize=12, color="white")
    ax.tick_params(colors="white")
    
    plt.tight_layout()
    
    save_path = save_path or os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Visualize] Saved confusion matrix → {save_path}")


def plot_risk_distribution(risk_preds, risk_labels, save_path=None):
    """
    Plot predicted vs. actual risk score distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    
    # ── Predicted Risk Distribution ──
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    ax.hist(risk_preds, bins=20, color=COLORS["primary"], alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.4, color=COLORS["positive"], linestyle="--",
               linewidth=2, label="Low/Med boundary")
    ax.axvline(x=0.7, color=COLORS["negative"], linestyle="--",
               linewidth=2, label="Med/High boundary")
    ax.set_title("Predicted Risk Score Distribution", fontsize=13,
                 fontweight="bold", color="white")
    ax.set_xlabel("Risk Score", color="white")
    ax.set_ylabel("Count", color="white")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    # ── Predicted vs Actual Scatter ──
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    ax.scatter(risk_labels, risk_preds, c=COLORS["accent"], alpha=0.7,
               s=60, edgecolors="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["secondary"], linewidth=2,
            label="Perfect prediction")
    ax.set_title("Predicted vs Actual Risk Scores", fontsize=13,
                 fontweight="bold", color="white")
    ax.set_xlabel("Actual Risk", color="white")
    ax.set_ylabel("Predicted Risk", color="white")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    
    plt.tight_layout()
    
    save_path = save_path or os.path.join(PLOTS_DIR, "risk_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Visualize] Saved risk distribution → {save_path}")


def plot_sentiment_distribution(sentiment_preds, save_path=None):
    """
    Plot sentiment prediction distribution as a pie chart.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    
    # Count each sentiment
    unique, counts = np.unique(sentiment_preds, return_counts=True)
    labels = [SENTIMENT_LABELS[u] for u in unique]
    colors = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]
    colors = colors[:len(unique)]
    
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=COLORS["bg"], linewidth=3),
        textprops=dict(color="white", fontsize=12),
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    
    ax.set_title("Sentiment Prediction Distribution",
                 fontsize=14, fontweight="bold", color="white", pad=20)
    
    plt.tight_layout()
    
    save_path = save_path or os.path.join(PLOTS_DIR, "sentiment_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Visualize] Saved sentiment distribution → {save_path}")


def generate_all_visualizations(history, metrics, all_preds):
    """
    Generate all visualization plots from training results.
    
    Args:
        history:   Training history dict
        metrics:   Evaluation metrics dict
        all_preds: Raw predictions dict
    """
    print("\n[Visualize] Generating all plots...")
    
    plot_training_curves(history)
    plot_accuracy_curves(history)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_risk_distribution(all_preds["risk_preds"], all_preds["risk_labels"])
    plot_sentiment_distribution(all_preds["sentiment_preds"])
    
    print(f"[Visualize] All plots saved to: {PLOTS_DIR}")


# ── Quick test ──
if __name__ == "__main__":
    # Test with dummy data
    dummy_history = {
        "train_loss": [1.5, 1.2, 0.9, 0.7, 0.5],
        "val_loss": [1.6, 1.3, 1.0, 0.8, 0.7],
        "train_sent_loss": [1.2, 1.0, 0.7, 0.5, 0.4],
        "train_risk_loss": [0.3, 0.2, 0.2, 0.2, 0.1],
        "val_sent_loss": [1.3, 1.1, 0.8, 0.6, 0.5],
        "val_risk_loss": [0.3, 0.2, 0.2, 0.2, 0.2],
        "train_sent_acc": [0.4, 0.5, 0.6, 0.7, 0.8],
        "val_sent_acc": [0.3, 0.4, 0.5, 0.6, 0.65],
    }
    
    plot_training_curves(dummy_history)
    plot_accuracy_curves(dummy_history)
    print("Dummy plots generated!")
