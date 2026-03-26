"""
============================================================
Financial News Sentiment & Risk Prediction System
Main Pipeline Orchestration
============================================================
This script runs the complete pipeline:
  1. Load & prepare data
  2. Initialize model
  3. Train the model
  4. Evaluate performance
  5. Generate visualizations
  6. Run predictions on sample headlines
  7. Save trained model
  8. (Optional) Show explainability content
============================================================
Usage:
  python main.py              — Run full pipeline
  python main.py --explain    — Show concept explanations
  python main.py --predict    — Interactive prediction mode
============================================================
"""

import sys
import os
import torch
import random
import numpy as np

# ── Local Modules ──
from config import (
    DEVICE, RANDOM_SEED, MODEL_SAVE_PATH, OUTPUT_DIR, EPOCHS
)
from dataset import get_data_loaders
from model import FinancialSentimentRiskModel
from train import train_model
from evaluate import evaluate_model
from predict import (
    predict_batch, display_batch_predictions, interactive_prediction
)
from visualize import generate_all_visualizations
from explainability import show_all_explanations


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, tokenizer, path=MODEL_SAVE_PATH):
    """Save the trained model state dict."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "FinancialSentimentRiskModel",
    }, path)
    print(f"\n[Main] Model saved → {path}")


def load_model(path=MODEL_SAVE_PATH):
    """Load a saved model."""
    model = FinancialSentimentRiskModel()
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"[Main] Model loaded from {path}")
    return model


# ══════════════════════════════════════════════
# Test Headlines for Demonstration
# ══════════════════════════════════════════════
DEMO_HEADLINES = [
    # Positive
    "Apple announces groundbreaking AI chip boosting stock price",
    "Amazon quarterly revenue exceeds all analyst expectations",
    
    # Negative
    "Major bank collapses amid liquidity crisis and fraud allegations",
    "Company shares fell sharply after massive revenue miss",
    "Global recession fears intensify as markets tumble",
    
    # Neutral
    "Federal Reserve keeps interest rates unchanged at meeting",
    "Company appoints new chief financial officer effective next month",
    
    # Ambiguous / Complex
    "Tech startup raises funding despite industry wide layoffs",
    "Oil prices fluctuate as OPEC discusses production adjustments",
    "Pharmaceutical company settles lawsuit for undisclosed amount",
]


def run_full_pipeline():
    """
    Execute the complete Financial Sentiment + Risk pipeline.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  🏦 FINANCIAL NEWS SENTIMENT & RISK PREDICTION SYSTEM".center(58) + "║")
    print("║" + "  ─────────────────────────────────────────────────".center(58) + "║")
    print("║" + "  PyTorch + HuggingFace BERT | Multi-Task Learning".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # ── Step 0: Set Seeds ──
    set_seed()
    
    # ── Step 1: Load Data ──
    print("\n" + "━" * 60)
    print("  📂 STEP 1: Loading & Preparing Data")
    print("━" * 60)
    train_loader, val_loader, tokenizer = get_data_loaders()
    
    # ── Step 2: Initialize Model ──
    print("\n" + "━" * 60)
    print("  🤖 STEP 2: Initializing Model")
    print("━" * 60)
    model = FinancialSentimentRiskModel()
    
    # Freeze early BERT layers to prevent overfitting on small dataset
    model.freeze_bert_layers(num_layers_to_freeze=10)
    
    # ── Step 3: Train ──
    print("\n" + "━" * 60)
    print("  🔧 STEP 3: Training Model")
    print("━" * 60)
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    
    # ── Step 4: Evaluate ──
    print("\n" + "━" * 60)
    print("  📊 STEP 4: Evaluating Model")
    print("━" * 60)
    metrics, all_preds = evaluate_model(model, val_loader)
    
    # ── Step 5: Visualizations ──
    print("\n" + "━" * 60)
    print("  📈 STEP 5: Generating Visualizations")
    print("━" * 60)
    generate_all_visualizations(history, metrics, all_preds)
    
    # ── Step 6: Demo Predictions ──
    print("\n" + "━" * 60)
    print("  🔮 STEP 6: Running Demo Predictions")
    print("━" * 60)
    results = predict_batch(model, tokenizer, DEMO_HEADLINES)
    display_batch_predictions(results)
    
    # ── Step 7: Save Model ──
    print("\n" + "━" * 60)
    print("  💾 STEP 7: Saving Trained Model")
    print("━" * 60)
    save_model(model, tokenizer)
    
    # ── Summary ──
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  ✅ PIPELINE COMPLETE!".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + f"  Accuracy:  {metrics['accuracy']:.4f}".ljust(58) + "║")
    print("║" + f"  F1 Score:  {metrics['f1_score']:.4f}".ljust(58) + "║")
    print("║" + f"  ROC-AUC:   {metrics['risk_auc']:.4f}".ljust(58) + "║")
    print("║" + f"  Risk MAE:  {metrics['risk_mae']:.4f}".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + f"  Model saved: {MODEL_SAVE_PATH}".ljust(58) + "║")
    print("║" + f"  Plots saved: {os.path.basename(OUTPUT_DIR)}/plots/".ljust(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    return model, tokenizer, history, metrics


def main():
    """Entry point with CLI argument support."""
    args = sys.argv[1:]
    
    if "--explain" in args:
        show_all_explanations()
        return
    
    if "--predict" in args:
        # Load saved model for interactive prediction
        if os.path.exists(MODEL_SAVE_PATH):
            from transformers import AutoTokenizer
            from config import MODEL_NAME
            model = load_model()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            interactive_prediction(model, tokenizer)
        else:
            print("No saved model found. Running full pipeline first...")
            model, tokenizer, _, _ = run_full_pipeline()
            interactive_prediction(model, tokenizer)
        return
    
    # Default: run full pipeline
    model, tokenizer, history, metrics = run_full_pipeline()
    
    # Ask if user wants interactive mode
    print("\n  Would you like to enter interactive prediction mode? (y/n)")
    try:
        if input("  > ").strip().lower() in ("y", "yes"):
            interactive_prediction(model, tokenizer)
    except (EOFError, KeyboardInterrupt):
        print("\n  Goodbye!")


if __name__ == "__main__":
    main()
