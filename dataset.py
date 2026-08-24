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
# Each entry: (headline, sentiment_label, calibrated_risk_score)

RAW_FINANCIAL_DATA = [
    # ── 1. Positive Headlines (Standard) ──
    ("Tesla reports record quarterly profits exceeding expectations", "Positive", 0.15),
    ("Apple stock surges after strong iPhone sales report", "Positive", 0.12),
    ("Amazon announces massive expansion into healthcare sector", "Positive", 0.22),
    ("Google parent Alphabet posts record revenue growth", "Positive", 0.16),
    ("Microsoft cloud division shows exceptional growth this quarter", "Positive", 0.14),
    ("Netflix subscriber numbers beat analyst predictions significantly", "Positive", 0.20),
    ("NVIDIA stock hits all time high on AI chip demand", "Positive", 0.15),
    ("JPMorgan reports strongest earnings in banking history", "Positive", 0.18),
    ("Walmart raises full year guidance on strong consumer spending", "Positive", 0.19),
    ("Meta platforms sees advertising revenue boom in Q4", "Positive", 0.21),
    ("Company announces breakthrough partnership with major retailer", "Positive", 0.25),
    ("Startup secures billion dollar funding round from top investors", "Positive", 0.28),
    ("Renewable energy sector sees unprecedented investment growth", "Positive", 0.24),
    ("Pharmaceutical company gets FDA approval for new drug", "Positive", 0.20),
    ("Tech giant reports better than expected earnings for third quarter", "Positive", 0.17),
    ("Interest rates expected to drop boosting market optimism", "Positive", 0.22),
    ("Major merger creates largest company in the industry", "Positive", 0.26),
    ("Export numbers reach historic highs this fiscal year", "Positive", 0.21),
    ("Consumer confidence index rises to highest level in decade", "Positive", 0.19),
    ("Electric vehicle sales growth exceeds all market forecasts", "Positive", 0.23),
    ("Semiconductor giant announces massive dividend increase and share buyback", "Positive", 0.14),
    ("Biotech firm shows ninety percent efficacy in phase three clinical trials", "Positive", 0.20),
    ("Aerospace leader signs multibillion dollar commercial jet supply contract", "Positive", 0.22),
    ("E-commerce sales surge thirty percent during annual holiday shopping festival", "Positive", 0.18),
    ("Cloud software provider surpasses one billion dollar annual recurring revenue", "Positive", 0.16),
    ("Central bank rate cuts ignite broad rally across equity markets", "Positive", 0.20),
    ("Clean energy startup receives lucrative government infrastructure grant", "Positive", 0.25),
    ("Medical device company receives international regulatory approval for flagship product", "Positive", 0.19),
    ("Telecom giant completes nationwide high speed network rollout ahead of schedule", "Positive", 0.21),
    ("Cybersecurity firm reports eighty percent growth in enterprise client bookings", "Positive", 0.17),

    # ── 2. Contrastive / Multi-Horizon Positive (Short-Term Dip/Headwind → Medium/Long-Term Rebound/Catalyst) ──
    ("The Stock Of Apple is expected to fall this quarter, but in the next quarter the price is expected to bounce back, because of the latest iphone", "Positive", 0.30),
    ("Apple stock is expected to fall this quarter, but next quarter the price will rebound because of strong iPhone demand", "Positive", 0.28),
    ("Company shares may drop in the short term, but long term growth is projected to accelerate due to new AI chips", "Positive", 0.27),
    ("Stock prices are expected to decline this quarter, but analysts forecast a strong rebound next year driven by cloud demand", "Positive", 0.30),
    ("Near-term earnings may face headwinds, but new product launches will drive robust revenue expansion next quarter", "Positive", 0.29),
    ("Revenue missed expectations this quarter, however full-year guidance was raised significantly on strong order backlog", "Positive", 0.32),
    ("Tesla margins contracted this quarter, but record vehicle deliveries and autonomous driving software boost long term outlook", "Positive", 0.31),
    ("Microsoft faces short-term gaming division slowdown, but cloud and enterprise AI adoption ensure massive future profit growth", "Positive", 0.22),
    ("Nvidia sees temporary supply chain constraints, but unprecedented AI accelerator demand guarantees exponential rebound", "Positive", 0.20),
    ("Retail sales dipped last month, but holiday season preorder volumes point to a record-breaking surge", "Positive", 0.28),
    ("Semiconductor firm warns of near-term inventory adjustment, but projects triple-digit sales growth next year", "Positive", 0.33),
    ("Stock tumbled four percent today on earnings whisper, but Wall Street unanimously reiterates Buy rating with forty percent upside", "Positive", 0.26),
    ("Temporary production delay resolved, company set to resume aggressive market expansion next quarter", "Positive", 0.27),
    ("Biotech firm reports clinical delay, but receives breakthrough FDA fast-track designation for lead drug", "Positive", 0.34),
    ("EV maker expects softer Q1 deliveries, but new low-cost model rollout will triple production capacity by Q3", "Positive", 0.29),
    ("Software maker experiences quarterly slowdown, but annual recurring revenue jumps forty percent", "Positive", 0.25),
    ("Bank provisions for credit losses rise, but net interest income reaches all-time record", "Positive", 0.32),
    ("Short-term supply disruptions affect shipments, but strong pricing power preserves double-digit profit margins", "Positive", 0.30),
    ("Initial quarter shows modest sales dip, yet subsequent quarters poised for explosive recovery on product cycle", "Positive", 0.28),
    ("Company posts slight quarterly loss due to R&D investment, but commercial rollout promises multi-billion dollar return", "Positive", 0.31),
    ("Earnings slipped marginally this quarter, but next quarter expected to surge to historic highs on strong consumer demand", "Positive", 0.26),
    ("Stock faces near-term pressure from currency fluctuations, but underlying international demand remains at record highs", "Positive", 0.29),
    ("Short term pullback presents attractive buying opportunity as fundamental growth trajectory remains intact", "Positive", 0.25),
    ("While factory upgrade paused manufacturing temporarily, new automated lines will double output next month", "Positive", 0.28),
    ("Operating expenses increased due to new store openings, which analysts predict will generate record holiday sales", "Positive", 0.30),
    ("Company warns of soft Q2 guidance, but predicts explosive second-half rebound driven by new product ecosystem", "Positive", 0.31),
    ("Gross margins slipped slightly, yet customer retention rates hit a five-year peak driving recurring revenue", "Positive", 0.27),
    ("Shares dipped on cautious executive comments, but institutional investors aggressively accumulate stock on dips", "Positive", 0.29),
    ("Quarterly revenue declined three percent, but upcoming next-gen platform launch anticipated to spark massive upgrade cycle", "Positive", 0.30),
    ("Temporary logistics bottleneck cleared, positioning firm for record export volumes in coming quarters", "Positive", 0.26),

    # ── 3. Negative Headlines (Standard) ──
    ("Bank faces massive fraud investigation by federal authorities", "Negative", 0.92),
    ("Company revenue drops by twenty percent in latest quarter", "Negative", 0.78),
    ("Major retailer files for bankruptcy after years of declining sales", "Negative", 0.95),
    ("Stock market crashes amid global recession fears", "Negative", 0.88),
    ("Tech company announces layoffs of ten thousand employees", "Negative", 0.74),
    ("Oil prices plummet as demand outlook weakens significantly", "Negative", 0.80),
    ("Company shares fell sharply after revenue miss", "Negative", 0.79),
    ("Credit rating agency downgrades nation debt outlook to negative", "Negative", 0.85),
    ("Cryptocurrency exchange collapses losing billions in customer funds", "Negative", 0.96),
    ("Trade war escalation threatens global economic stability", "Negative", 0.82),
    ("Housing market shows signs of severe downturn ahead", "Negative", 0.81),
    ("Inflation surges to highest level in forty years", "Negative", 0.83),
    ("Company faces billion dollar lawsuit over data breach", "Negative", 0.89),
    ("Supply chain disruptions cause massive losses for manufacturers", "Negative", 0.77),
    ("Hedge fund reports worst quarterly performance in its history", "Negative", 0.86),
    ("Government shutdown threatens economic growth projections", "Negative", 0.75),
    ("Company recalls products after safety violations discovered", "Negative", 0.84),
    ("Pension fund reports significant losses in risky investments", "Negative", 0.87),
    ("Earnings miss sends stock price tumbling by fifteen percent", "Negative", 0.83),
    ("Rising interest rates put pressure on corporate borrowing costs", "Negative", 0.73),
    ("CEO abruptly resigns amid internal probe into accounting irregularities", "Negative", 0.91),
    ("Automaker recalls two million vehicles over critical battery fire risk", "Negative", 0.88),
    ("Tech startup runs out of cash after funding deal falls through", "Negative", 0.94),
    ("Commercial real estate defaults hit highest level since financial crisis", "Negative", 0.89),
    ("Pharma firm scraps phase three trial after drug fails primary endpoint", "Negative", 0.86),
    ("National debt default looms as parliament fails to reach budget agreement", "Negative", 0.93),
    ("Airlines ground fleet due to critical mechanical failures and safety concerns", "Negative", 0.87),
    ("Manufacturing output contracts for sixth consecutive month amid slumping orders", "Negative", 0.80),
    ("Major utility company sued for catastrophic environmental disaster", "Negative", 0.90),
    ("Semiconductor company reports forty percent drop in forward chip orders", "Negative", 0.82),

    # ── 4. Contrastive / False-Hope Negative (Short-Term Bump → Severe Long-Term Collapse/Risk) ──
    ("Shares surged five percent today, but analysts warn of severe structural debt default in coming months", "Negative", 0.84),
    ("Company posted temporary revenue uptick, but ongoing federal fraud investigation threatens imminent bankruptcy", "Negative", 0.91),
    ("Short-term stock rally expected to fizzle out as core customer churn reaches critical levels", "Negative", 0.78),
    ("Stock rebounded slightly this week, but escalating patent litigation poses existential threat to core product", "Negative", 0.86),
    ("Quarterly revenue stabilized, but aggressive competitor price cuts destroy long-term margin viability", "Negative", 0.77),
    ("Company reported minor sales beat, but massive debt burden forces urgent emergency refinancing", "Negative", 0.82),
    ("Stock bounced on buyout rumors, but board confirms no credible acquisition offers exist", "Negative", 0.80),
    ("Short-term trading volume spiked, yet regulatory sanctions threaten to revoke primary operating license", "Negative", 0.89),
    ("Earnings appeared steady, but forensic audit reveals severe revenue overstatement and manipulation", "Negative", 0.93),
    ("Shares opened higher on dividend news, but dividend was financed through unsustainable high-yield debt", "Negative", 0.81),

    # ── 5. Neutral Headlines (Standard) ──
    ("Federal Reserve maintains current interest rate policy unchanged", "Neutral", 0.48),
    ("Company announces leadership transition with new CEO appointment", "Neutral", 0.45),
    ("Quarterly earnings meet analyst expectations with no surprises", "Neutral", 0.42),
    ("Market trading volume remains steady with mixed sector performance", "Neutral", 0.50),
    ("Government releases updated economic growth forecast for next year", "Neutral", 0.46),
    ("Company restructures operations in line with industry trends", "Neutral", 0.49),
    ("Regulatory body announces review of financial sector policies", "Neutral", 0.52),
    ("Central bank signals cautious approach to monetary policy changes", "Neutral", 0.47),
    ("Industry conference highlights emerging technology trends", "Neutral", 0.40),
    ("Analysts maintain hold rating on stock citing market uncertainty", "Neutral", 0.51),
    ("Company completes scheduled share buyback program as planned", "Neutral", 0.41),
    ("New trade agreement signed between major economies", "Neutral", 0.43),
    ("Commodity prices show mixed movement across different sectors", "Neutral", 0.49),
    ("Board of directors approves annual dividend unchanged from last year", "Neutral", 0.42),
    ("Company announces plans to enter new geographic market", "Neutral", 0.44),
    ("Financial regulators issue guidelines for digital currency trading", "Neutral", 0.50),
    ("Annual shareholder meeting concludes without major policy changes", "Neutral", 0.42),
    ("Market analysts predict sideways trading pattern for coming weeks", "Neutral", 0.52),
    ("Company reports revenue in line with previous guidance", "Neutral", 0.41),
    ("Economic indicators show mixed signals about growth trajectory", "Neutral", 0.53),
    ("Stock index rebalancing takes effect with minor constituent adjustments", "Neutral", 0.44),
    ("Treasury yields hold steady ahead of upcoming inflation report", "Neutral", 0.47),
    ("Company schedules date for annual investor day presentation", "Neutral", 0.40),
    ("Mining firm files routine environmental compliance documentation", "Neutral", 0.46),
    ("Tech firm files standard quarterly SEC disclosure report", "Neutral", 0.42),
    ("Retail banking division reports balanced deposit and withdrawal flows", "Neutral", 0.45),
    ("Logistics provider renews existing multi-year freight forwarding contract", "Neutral", 0.43),
    ("Auto manufacturer announces planned routine factory retooling for upcoming models", "Neutral", 0.44),
    ("Energy regulator opens public comment period on proposed grid updates", "Neutral", 0.48),
    ("Company participates in annual international consumer electronics exhibition", "Neutral", 0.39),
    ("Quarterly dividend declared payable on scheduled distribution date", "Neutral", 0.41),
    ("Bond market trades within narrow range following routine debt auction", "Neutral", 0.47),
    ("Corporate governance report confirms board compliance with standard criteria", "Neutral", 0.42),
    ("Regional banking index reflects stable deposit levels across member institutions", "Neutral", 0.45),
    ("Telecom company conducts scheduled network bandwidth expansion tests", "Neutral", 0.40),
]


def generate_risk_scores(data):
    """
    Extract or generate calibrated continuous risk scores.
    
    If risk score is provided in the tuple (headline, sentiment, risk), use it with
    minor realistic noise. Otherwise, fallback to sentiment range.
    """
    np.random.seed(RANDOM_SEED)
    risk_scores = []
    
    for item in data:
        if len(item) == 3:
            headline, sentiment, calibrated_risk = item
            # Add tiny jitter (+-0.015) for natural variation
            jitter = np.random.uniform(-0.015, 0.015)
            risk = np.clip(calibrated_risk + jitter, 0.02, 0.98)
        else:
            headline, sentiment = item[0], item[1]
            if sentiment == "Positive":
                risk = np.random.uniform(0.10, 0.35)
            elif sentiment == "Negative":
                risk = np.random.uniform(0.68, 0.94)
            else:  # Neutral
                risk = np.random.uniform(0.40, 0.58)
        risk_scores.append(round(float(risk), 4))
    
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
