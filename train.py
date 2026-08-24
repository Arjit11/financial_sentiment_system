"""
============================================================
Financial News Sentiment & Risk Prediction System
Training Module
============================================================
Contains:
  - Multi-task training loop
  - Differential learning rates (BERT vs. custom heads)
  - Combined loss: CrossEntropyLoss + MSELoss
  - Epoch-level logging & history tracking
============================================================
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import copy

from config import (
    DEVICE, EPOCHS, LEARNING_RATE_BERT, LEARNING_RATE_HEAD,
    WEIGHT_DECAY, SENTIMENT_LOSS_WEIGHT, RISK_LOSS_WEIGHT,
    WARMUP_RATIO
)


def get_optimizer(model):
    """
    Create AdamW optimizer with differential learning rates.
    
    Strategy:
      - BERT layers:  Lower LR (2e-5) to preserve pretrained knowledge
      - Custom heads:  Higher LR (1e-3) for faster adaptation
    
    This is a best practice for fine-tuning transformers.
    """
    # Separate BERT and head parameters
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.sentiment_head.parameters()) +
        list(model.risk_head.parameters())
    )
    
    optimizer = AdamW([
        {"params": bert_params, "lr": LEARNING_RATE_BERT},
        {"params": head_params, "lr": LEARNING_RATE_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    
    return optimizer


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    """
    Train the multi-task model with combined sentiment + risk loss.
    
    Loss = (SENTIMENT_LOSS_WEIGHT × CrossEntropy) + (RISK_LOSS_WEIGHT × MSE)
    
    Args:
        model:         FinancialSentimentRiskModel instance
        train_loader:  Training DataLoader
        val_loader:    Validation DataLoader
        epochs:        Number of training epochs
    
    Returns:
        history: Dict with training metrics per epoch
                 Keys: train_loss, val_loss, train_sent_loss,
                       train_risk_loss, val_sent_loss, val_risk_loss
    """
    model.to(DEVICE)
    
    # ── Loss Functions ──
    sentiment_criterion = nn.CrossEntropyLoss()  # For classification
    risk_criterion = nn.MSELoss()                 # For regression
    
    # ── Optimizer & Scheduler ──
    optimizer = get_optimizer(model)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # ── Training History ──
    history = {
        "train_loss": [], "val_loss": [],
        "train_sent_loss": [], "train_risk_loss": [],
        "val_sent_loss": [], "val_risk_loss": [],
        "train_sent_acc": [], "val_sent_acc": [],
    }
    
    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)
    print(f"  Epochs:          {epochs}")
    print(f"  Device:          {DEVICE}")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    print(f"  Total steps:     {total_steps} (warmup: {warmup_steps})")
    print(f"  Sentiment weight: {SENTIMENT_LOSS_WEIGHT}")
    print(f"  Risk weight:      {RISK_LOSS_WEIGHT}")
    print("=" * 60)
    
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ────────────────────────────────────
        # Training Phase
        # ────────────────────────────────────
        model.train()
        total_train_loss = 0
        total_sent_loss = 0
        total_risk_loss = 0
        correct_preds = 0
        total_samples = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sentiment_labels = batch["sentiment_label"].to(DEVICE)
            risk_labels = batch["risk_score"].to(DEVICE)
            
            # Forward pass
            sentiment_logits, risk_preds = model(input_ids, attention_mask)
            
            # Compute losses
            sent_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            risk_loss = risk_criterion(risk_preds, risk_labels)
            
            # Combined multi-task loss
            total_loss = (SENTIMENT_LOSS_WEIGHT * sent_loss +
                         RISK_LOSS_WEIGHT * risk_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_train_loss += total_loss.item()
            total_sent_loss += sent_loss.item()
            total_risk_loss += risk_loss.item()
            
            preds = torch.argmax(sentiment_logits, dim=1)
            correct_preds += (preds == sentiment_labels).sum().item()
            total_samples += sentiment_labels.size(0)
        
        # Average training metrics
        n_batches = len(train_loader)
        avg_train_loss = total_train_loss / n_batches
        avg_sent_loss = total_sent_loss / n_batches
        avg_risk_loss = total_risk_loss / n_batches
        train_acc = correct_preds / total_samples
        
        # ────────────────────────────────────
        # Validation Phase
        # ────────────────────────────────────
        model.eval()
        total_val_loss = 0
        total_val_sent = 0
        total_val_risk = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                sentiment_labels = batch["sentiment_label"].to(DEVICE)
                risk_labels = batch["risk_score"].to(DEVICE)
                
                sentiment_logits, risk_preds = model(input_ids, attention_mask)
                
                sent_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
                risk_loss = risk_criterion(risk_preds, risk_labels)
                total_loss = (SENTIMENT_LOSS_WEIGHT * sent_loss +
                             RISK_LOSS_WEIGHT * risk_loss)
                
                total_val_loss += total_loss.item()
                total_val_sent += sent_loss.item()
                total_val_risk += risk_loss.item()
                
                preds = torch.argmax(sentiment_logits, dim=1)
                val_correct += (preds == sentiment_labels).sum().item()
                val_total += sentiment_labels.size(0)
        
        # Average validation metrics
        n_val = len(val_loader)
        avg_val_loss = total_val_loss / n_val if n_val > 0 else 0
        avg_val_sent = total_val_sent / n_val if n_val > 0 else 0
        avg_val_risk = total_val_risk / n_val if n_val > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_sent_loss"].append(avg_sent_loss)
        history["train_risk_loss"].append(avg_risk_loss)
        history["val_sent_loss"].append(avg_val_sent)
        history["val_risk_loss"].append(avg_val_risk)
        history["train_sent_acc"].append(train_acc)
        history["val_sent_acc"].append(val_acc)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_marker = " ★ Best"
        else:
            best_marker = ""
        
        elapsed = time.time() - epoch_start
        
        # ── Epoch Summary ──
        print(f"\n  Epoch {epoch + 1}/{epochs}  ({elapsed:.1f}s){best_marker}")
        print(f"  ├─ Train Loss: {avg_train_loss:.4f}  "
              f"(Sent: {avg_sent_loss:.4f}, Risk: {avg_risk_loss:.4f})  "
              f"Acc: {train_acc:.3f}")
        print(f"  └─ Val   Loss: {avg_val_loss:.4f}  "
              f"(Sent: {avg_val_sent:.4f}, Risk: {avg_val_risk:.4f})  "
              f"Acc: {val_acc:.3f}")
    
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n[Train] Restored best model weights from checkpoint.")
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    return history


# ── Quick test ──
if __name__ == "__main__":
    from model import FinancialSentimentRiskModel
    from dataset import get_data_loaders
    
    train_loader, val_loader, tokenizer = get_data_loaders()
    model = FinancialSentimentRiskModel()
    model.freeze_bert_layers(num_layers_to_freeze=10)
    
    history = train_model(model, train_loader, val_loader, epochs=3)
    print(f"\nTraining history keys: {history.keys()}")
