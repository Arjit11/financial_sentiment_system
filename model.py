"""
============================================================
Financial News Sentiment & Risk Prediction System
Model Module
============================================================
Contains:
  - FinancialSentimentRiskModel: BERT + custom dual-head
    architecture for simultaneous sentiment classification
    and risk score regression.

Architecture:
  BERT Encoder (pretrained)
       ↓
  [CLS] token embedding (768-dim)
       ↓
  ┌────────────────┬──────────────────────────────┐
  │ Sentiment Head │ Risk Scoring Head            │
  │ Dropout(0.3)   │ Dropout(0.3)                 │
  │ Linear(768→3)  │ Linear(768→128) → ReLU       │
  │                │ Linear(128→1) → Sigmoid       │
  └────────────────┴──────────────────────────────┘
============================================================
"""

import torch
import torch.nn as nn
from transformers import BertModel

from config import MODEL_NAME, HIDDEN_SIZE, DROPOUT_RATE, NUM_SENTIMENT_CLASSES


class SentimentHead(nn.Module):
    """
    Classification head for sentiment prediction.
    
    Takes the [CLS] token embedding from BERT and maps it
    to 3 sentiment classes: Positive, Negative, Neutral.
    """
    
    def __init__(self, hidden_size=HIDDEN_SIZE, num_classes=NUM_SENTIMENT_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super(SentimentHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, cls_embedding):
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)  # Raw logits (softmax applied in loss)
        return logits


class RiskScoringHead(nn.Module):
    """
    Regression head for risk score prediction.
    
    Takes the [CLS] token embedding and predicts a continuous
    risk score between 0 (low risk) and 1 (high risk).
    
    Uses a deeper architecture with ReLU activation for
    better non-linear risk modeling.
    """
    
    def __init__(self, hidden_size=HIDDEN_SIZE, intermediate_size=128,
                 dropout_rate=DROPOUT_RATE):
        super(RiskScoringHead, self).__init__()
        self.network = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Lighter dropout in middle
            nn.Linear(intermediate_size, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
    
    def forward(self, cls_embedding):
        risk_score = self.network(cls_embedding)
        return risk_score.squeeze(-1)  # Shape: (batch_size,)


class FinancialSentimentRiskModel(nn.Module):
    """
    ⭐ Main Model: BERT + Custom Dual-Head Architecture
    
    This model combines a pretrained BERT encoder with two
    custom neural network heads for multi-task learning:
    
    1. Sentiment Classification (3 classes)
    2. Risk Score Regression (continuous 0–1)
    
    The model leverages BERT's bidirectional contextual
    understanding of financial text, then applies specialized
    heads for each prediction task.
    
    Parameters:
        model_name (str): HuggingFace model identifier
        
    Outputs:
        sentiment_logits: Raw logits for 3 sentiment classes
        risk_score: Float value in [0, 1]
    """
    
    def __init__(self, model_name=MODEL_NAME):
        super(FinancialSentimentRiskModel, self).__init__()
        
        # ── Pretrained BERT Encoder ──
        # Loads bert-base-uncased with 12 transformer layers,
        # 768 hidden size, and 110M parameters
        self.bert = BertModel.from_pretrained(model_name)
        
        # ── Custom Task-Specific Heads ──
        self.sentiment_head = SentimentHead()
        self.risk_head = RiskScoringHead()
        
        print(f"[Model] Loaded {model_name}")
        print(f"[Model] BERT parameters: {sum(p.numel() for p in self.bert.parameters()):,}")
        print(f"[Model] Sentiment head parameters: {sum(p.numel() for p in self.sentiment_head.parameters()):,}")
        print(f"[Model] Risk head parameters: {sum(p.numel() for p in self.risk_head.parameters()):,}")
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Total parameters: {total:,}")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT → dual heads.
        
        Args:
            input_ids:      Token IDs from BERT tokenizer, shape (batch, seq_len)
            attention_mask:  Mask for padding tokens, shape (batch, seq_len)
        
        Returns:
            sentiment_logits: Shape (batch, 3) — raw class scores
            risk_score:       Shape (batch,) — values in [0, 1]
        """
        # ── BERT Encoding ──
        # Returns: last_hidden_state (batch, seq_len, 768), pooler_output (batch, 768)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Extract [CLS] token representation (first token)
        # This captures the overall semantic meaning of the headline
        cls_embedding = bert_output.pooler_output  # Shape: (batch, 768)
        
        # ── Dual-Head Predictions ──
        sentiment_logits = self.sentiment_head(cls_embedding)
        risk_score = self.risk_head(cls_embedding)
        
        return sentiment_logits, risk_score
    
    def freeze_bert_layers(self, num_layers_to_freeze=10):
        """
        Freeze early BERT layers to prevent overfitting on small datasets.
        Only fine-tunes the last few transformer layers + custom heads.
        
        Args:
            num_layers_to_freeze: Number of BERT encoder layers to freeze (out of 12)
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        for i in range(num_layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Frozen {num_layers_to_freeze}/12 BERT layers")
        print(f"[Model] Trainable: {trainable:,} / {total:,} parameters")


# ── Quick test ──
if __name__ == "__main__":
    from config import DEVICE
    
    model = FinancialSentimentRiskModel()
    model.to(DEVICE)
    model.freeze_bert_layers(num_layers_to_freeze=10)
    
    # Test with dummy input
    dummy_ids = torch.randint(0, 1000, (2, 128)).to(DEVICE)
    dummy_mask = torch.ones(2, 128, dtype=torch.long).to(DEVICE)
    
    sentiment_logits, risk_scores = model(dummy_ids, dummy_mask)
    print(f"\nSentiment logits shape: {sentiment_logits.shape}")  # (2, 3)
    print(f"Risk scores shape:      {risk_scores.shape}")          # (2,)
    print(f"Sentiment logits:       {sentiment_logits}")
    print(f"Risk scores:            {risk_scores}")
