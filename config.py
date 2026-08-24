"""
============================================================
Financial News Sentiment & Risk Prediction System
Configuration Module
============================================================
Contains all hyperparameters, paths, constants, and device
configuration used across the entire system.
============================================================
"""

import torch
import os

# ──────────────────────────────────────────────
# Device Configuration (auto-detect GPU/MPS/CPU)
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[Config] Using device: {DEVICE}")

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"       # Pretrained BERT model
MAX_SEQ_LENGTH = 128                    # Maximum token sequence length
HIDDEN_SIZE = 768                       # BERT hidden dimension
DROPOUT_RATE = 0.3                      # Dropout probability

# ──────────────────────────────────────────────
# Label Mappings
# ──────────────────────────────────────────────
SENTIMENT_LABELS = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABEL_TO_ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
NUM_SENTIMENT_CLASSES = 3

# ──────────────────────────────────────────────
# Risk Thresholds for Decision Engine
# ──────────────────────────────────────────────
RISK_LEVELS = {
    "Low":    (0.0, 0.4),
    "Medium": (0.4, 0.7),
    "High":   (0.7, 1.0),
}

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
EPOCHS = 15                             # Number of training epochs
BATCH_SIZE = 8                          # Batch size
LEARNING_RATE_BERT = 2e-5               # Learning rate for BERT layers
LEARNING_RATE_HEAD = 5e-4               # Learning rate for custom heads
WARMUP_RATIO = 0.1                      # Warmup ratio for learning rate scheduler
WEIGHT_DECAY = 0.01                     # AdamW weight decay
SENTIMENT_LOSS_WEIGHT = 1.0             # Weight for sentiment loss
RISK_LOSS_WEIGHT = 0.5                  # Weight for risk loss
VALIDATION_SPLIT = 0.2                  # Fraction of data for validation

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_model.pth")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Random Seed for Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
