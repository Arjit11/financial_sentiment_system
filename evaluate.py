"""
============================================================
Financial News Sentiment & Risk Prediction System
Evaluation Module
============================================================
Contains:
  - Accuracy, F1 Score (macro), Classification Report
  - ROC-AUC for risk scores
  - Confusion matrix generation
  - Full evaluation pipeline
============================================================
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

from config import DEVICE, SENTIMENT_LABELS


def evaluate_model(model, data_loader):
    """
    Run full evaluation on a dataset and return all metrics.
    
    Computes:
      - Sentiment: Accuracy, F1 (macro), classification report, confusion matrix
      - Risk: Mean Absolute Error, ROC-AUC (binarized at 0.5)
    
    Args:
        model:       Trained FinancialSentimentRiskModel
        data_loader: DataLoader to evaluate on
    
    Returns:
        metrics: Dict containing all computed metrics
        all_preds: Dict with raw predictions for visualization
    """
    model.eval()
    model.to(DEVICE)
    
    all_sent_preds = []
    all_sent_labels = []
    all_risk_preds = []
    all_risk_labels = []
    all_sent_probs = []
    
    print("\n[Evaluation] Running evaluation...")
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sentiment_labels = batch["sentiment_label"].to(DEVICE)
            risk_labels = batch["risk_score"].to(DEVICE)
            
            # Forward pass
            sentiment_logits, risk_preds = model(input_ids, attention_mask)
            
            # Sentiment predictions
            sent_probs = torch.softmax(sentiment_logits, dim=1)
            sent_preds = torch.argmax(sent_probs, dim=1)
            
            all_sent_preds.extend(sent_preds.cpu().numpy())
            all_sent_labels.extend(sentiment_labels.cpu().numpy())
            all_sent_probs.extend(sent_probs.cpu().numpy())
            
            # Risk predictions
            all_risk_preds.extend(risk_preds.cpu().numpy())
            all_risk_labels.extend(risk_labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_sent_preds = np.array(all_sent_preds)
    all_sent_labels = np.array(all_sent_labels)
    all_sent_probs = np.array(all_sent_probs)
    all_risk_preds = np.array(all_risk_preds)
    all_risk_labels = np.array(all_risk_labels)
    
    # ── Sentiment Metrics ──
    accuracy = accuracy_score(all_sent_labels, all_sent_preds)
    f1 = f1_score(all_sent_labels, all_sent_preds, average="macro", zero_division=0)
    
    label_names = [SENTIMENT_LABELS[i] for i in range(len(SENTIMENT_LABELS))]
    cls_report = classification_report(
        all_sent_labels, all_sent_preds,
        target_names=label_names, zero_division=0
    )
    conf_matrix = confusion_matrix(all_sent_labels, all_sent_preds)
    
    # ── Risk Metrics ──
    risk_mae = np.mean(np.abs(all_risk_preds - all_risk_labels))
    
    # ROC-AUC (binarize: risk > 0.5 = high risk)
    try:
        risk_binary_labels = (all_risk_labels > 0.5).astype(int)
        risk_auc = roc_auc_score(risk_binary_labels, all_risk_preds)
    except ValueError:
        risk_auc = 0.0  # If only one class present
    
    # ── Compile Metrics ──
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": cls_report,
        "confusion_matrix": conf_matrix,
        "risk_mae": risk_mae,
        "risk_auc": risk_auc,
    }
    
    all_preds = {
        "sentiment_preds": all_sent_preds,
        "sentiment_labels": all_sent_labels,
        "sentiment_probs": all_sent_probs,
        "risk_preds": all_risk_preds,
        "risk_labels": all_risk_labels,
    }
    
    # ── Print Results ──
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  📊 Sentiment Classification:")
    print(f"     Accuracy:  {accuracy:.4f}")
    print(f"     F1 Score:  {f1:.4f}")
    print(f"\n  📈 Risk Score Prediction:")
    print(f"     MAE:       {risk_mae:.4f}")
    print(f"     ROC-AUC:   {risk_auc:.4f}")
    print(f"\n  📋 Classification Report:")
    print(cls_report)
    print(f"  🔢 Confusion Matrix:")
    print(f"     {conf_matrix}")
    print("=" * 60)
    
    return metrics, all_preds


# ── Quick test ──
if __name__ == "__main__":
    from model import FinancialSentimentRiskModel
    from dataset import get_data_loaders
    
    train_loader, val_loader, tokenizer = get_data_loaders()
    model = FinancialSentimentRiskModel()
    
    # Evaluate untrained model (baseline)
    metrics, preds = evaluate_model(model, val_loader)
