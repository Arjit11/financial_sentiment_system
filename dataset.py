"""
============================================================
Financial News Sentiment & Risk Prediction System
Dataset Module
============================================================
Contains:
  - Curated financial news dataset (~60 labeled headlines)
  - FinancialNewsDataset class (PyTorch Dataset)
  - BERT tokenization integration
  - Risk score generation based on sentiment
  - Train/validation split utility
============================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import numpy as np
import random

from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, LABEL_TO_ID, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED
)


# ══════════════════════════════════════════════
# Built-in Financial News Dataset
# ══════════════════════════════════════════════
# Each entry: (headline, sentiment_label)
# This dataset simulates real financial news for training/demo.

RAW_FINANCIAL_DATA = [
    # ── Positive Headlines ──
    ("Tesla reports record quarterly profits exceeding expectations", "Positive"),
    ("Apple stock surges after strong iPhone sales report", "Positive"),
    ("Amazon announces massive expansion into healthcare sector", "Positive"),
    ("Google parent Alphabet posts record revenue growth", "Positive"),
    ("Microsoft cloud division shows exceptional growth this quarter", "Positive"),
    ("Netflix subscriber numbers beat analyst predictions significantly", "Positive"),
    ("NVIDIA stock hits all time high on AI chip demand", "Positive"),
    ("JPMorgan reports strongest earnings in banking history", "Positive"),
    ("Walmart raises full year guidance on strong consumer spending", "Positive"),
    ("Meta platforms sees advertising revenue boom in Q4", "Positive"),
    ("Company announces breakthrough partnership with major retailer", "Positive"),
    ("Startup secures billion dollar funding round from top investors", "Positive"),
    ("Renewable energy sector sees unprecedented investment growth", "Positive"),
    ("Pharmaceutical company gets FDA approval for new drug", "Positive"),
    ("Tech giant reports better than expected earnings for third quarter", "Positive"),
    ("Interest rates expected to drop boosting market optimism", "Positive"),
    ("Major merger creates largest company in the industry", "Positive"),
    ("Export numbers reach historic highs this fiscal year", "Positive"),
    ("Consumer confidence index rises to highest level in decade", "Positive"),
    ("Electric vehicle sales growth exceeds all market forecasts", "Positive"),

    # ── Negative Headlines ──
    ("Bank faces massive fraud investigation by federal authorities", "Negative"),
    ("Company revenue drops by twenty percent in latest quarter", "Negative"),
    ("Major retailer files for bankruptcy after years of declining sales", "Negative"),
    ("Stock market crashes amid global recession fears", "Negative"),
    ("Tech company announces layoffs of ten thousand employees", "Negative"),
    ("Oil prices plummet as demand outlook weakens significantly", "Negative"),
    ("Company shares fell sharply after revenue miss", "Negative"),
    ("Credit rating agency downgrades nation debt outlook to negative", "Negative"),
    ("Cryptocurrency exchange collapses losing billions in customer funds", "Negative"),
    ("Trade war escalation threatens global economic stability", "Negative"),
    ("Housing market shows signs of severe downturn ahead", "Negative"),
    ("Inflation surges to highest level in forty years", "Negative"),
    ("Company faces billion dollar lawsuit over data breach", "Negative"),
    ("Supply chain disruptions cause massive losses for manufacturers", "Negative"),
    ("Hedge fund reports worst quarterly performance in its history", "Negative"),
    ("Government shutdown threatens economic growth projections", "Negative"),
    ("Company recalls products after safety violations discovered", "Negative"),
    ("Pension fund reports significant losses in risky investments", "Negative"),
    ("Earnings miss sends stock price tumbling by fifteen percent", "Negative"),
    ("Rising interest rates put pressure on corporate borrowing costs", "Negative"),

    # ── Neutral Headlines ──
    ("Federal Reserve maintains current interest rate policy unchanged", "Neutral"),
    ("Company announces leadership transition with new CEO appointment", "Neutral"),
    ("Quarterly earnings meet analyst expectations with no surprises", "Neutral"),
    ("Market trading volume remains steady with mixed sector performance", "Neutral"),
    ("Government releases updated economic growth forecast for next year", "Neutral"),
    ("Company restructures operations in line with industry trends", "Neutral"),
    ("Regulatory body announces review of financial sector policies", "Neutral"),
    ("Central bank signals cautious approach to monetary policy changes", "Neutral"),
    ("Industry conference highlights emerging technology trends", "Neutral"),
    ("Analysts maintain hold rating on stock citing market uncertainty", "Neutral"),
    ("Company completes scheduled share buyback program as planned", "Neutral"),
    ("New trade agreement signed between major economies", "Neutral"),
    ("Commodity prices show mixed movement across different sectors", "Neutral"),
    ("Board of directors approves annual dividend unchanged from last year", "Neutral"),
    ("Company announces plans to enter new geographic market", "Neutral"),
    ("Financial regulators issue guidelines for digital currency trading", "Neutral"),
    ("Annual shareholder meeting concludes without major policy changes", "Neutral"),
    ("Market analysts predict sideways trading pattern for coming weeks", "Neutral"),
    ("Company reports revenue in line with previous guidance", "Neutral"),
    ("Economic indicators show mixed signals about growth trajectory", "Neutral"),
]


def generate_risk_scores(data):
    """
    Generate synthetic risk scores based on sentiment labels.
    
    Risk Score Logic:
      - Positive sentiment → Low risk (0.05 – 0.35)
      - Negative sentiment → High risk (0.65 – 0.95)
      - Neutral  sentiment → Medium risk (0.30 – 0.65)
    
    Small random noise is added for realism.
    """
    np.random.seed(RANDOM_SEED)
    risk_scores = []
    
    for _, sentiment in data:
        if sentiment == "Positive":
            risk = np.random.uniform(0.05, 0.35)
        elif sentiment == "Negative":
            risk = np.random.uniform(0.65, 0.95)
        else:  # Neutral
            risk = np.random.uniform(0.30, 0.65)
        risk_scores.append(round(risk, 4))
    
    return risk_scores


class FinancialNewsDataset(Dataset):
    """
    PyTorch Dataset for Financial News Sentiment & Risk Prediction.
    
    Each sample contains:
      - input_ids:      Tokenized headline (BERT format)
      - attention_mask:  Attention mask for padding
      - sentiment_label: Integer label (0=Positive, 1=Negative, 2=Neutral)
      - risk_score:      Float in [0, 1]
    """
    
    def __init__(self, headlines, sentiment_labels, risk_scores, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.headlines = headlines
        self.sentiment_labels = sentiment_labels
        self.risk_scores = risk_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        headline = self.headlines[idx]
        sentiment = self.sentiment_labels[idx]
        risk = self.risk_scores[idx]
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            headline,
            add_special_tokens=True,     # [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding="max_length",         # Pad to max_length
            truncation=True,              # Truncate if too long
            return_tensors="pt",          # Return PyTorch tensors
        )
        
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # Shape: (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),   # Shape: (max_length,)
            "sentiment_label": torch.tensor(sentiment, dtype=torch.long),
            "risk_score":      torch.tensor(risk, dtype=torch.float),
        }


def get_data_loaders():
    """
    Prepare and return train & validation DataLoaders.
    
    Steps:
      1. Load the built-in financial dataset
      2. Generate risk scores
      3. Initialize BERT tokenizer
      4. Create dataset & split into train/val
      5. Return DataLoaders
      
    Returns:
        train_loader, val_loader, tokenizer
    """
    print("\n[Dataset] Loading financial news data...")
    
    # Extract headlines and labels
    headlines = [item[0] for item in RAW_FINANCIAL_DATA]
    sentiment_strs = [item[1] for item in RAW_FINANCIAL_DATA]
    sentiment_ids = [LABEL_TO_ID[s] for s in sentiment_strs]
    
    # Generate risk scores
    risk_scores = generate_risk_scores(RAW_FINANCIAL_DATA)
    
    print(f"[Dataset] Total samples: {len(headlines)}")
    print(f"[Dataset] Sentiment distribution: "
          f"Positive={sentiment_strs.count('Positive')}, "
          f"Negative={sentiment_strs.count('Negative')}, "
          f"Neutral={sentiment_strs.count('Neutral')}")
    
    # Initialize BERT tokenizer
    print(f"[Dataset] Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create full dataset
    full_dataset = FinancialNewsDataset(
        headlines=headlines,
        sentiment_labels=sentiment_ids,
        risk_scores=risk_scores,
        tokenizer=tokenizer,
    )
    
    # Train / Validation split
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    print(f"[Dataset] Train samples: {train_size}, Validation samples: {val_size}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, tokenizer


# ── Quick test ──
if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_data_loaders()
    
    # Show a sample batch
    batch = next(iter(train_loader))
    print(f"\nSample batch keys: {batch.keys()}")
    print(f"Input IDs shape:   {batch['input_ids'].shape}")
    print(f"Attention mask:    {batch['attention_mask'].shape}")
    print(f"Sentiment labels:  {batch['sentiment_label']}")
    print(f"Risk scores:       {batch['risk_score']}")
