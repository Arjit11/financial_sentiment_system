# 🏦 Financial News Sentiment & Risk Prediction System

An AI-powered system that analyzes financial news headlines, predicts sentiment, generates risk scores, and provides investment recommendations using **BERT** and **PyTorch**.

---

## 🏗 Architecture

```
User Input (Financial Headline)
     ↓
BERT Tokenizer (WordPiece)
     ↓
BERT Encoder (bert-base-uncased, 12 layers)
     ↓
[CLS] Token Embedding (768-dim)
     ↓
┌──────────────────┬──────────────────────┐
│  Sentiment Head  │   Risk Scoring Head  │
│  Dropout → FC    │   Dropout → FC → FC  │
│  → 3 classes     │   → Sigmoid (0–1)    │
└──────────────────┴──────────────────────┘
     ↓                    ↓
  Sentiment          Risk Score
  (Pos/Neg/Neu)      (0.0 – 1.0)
     ↓                    ↓
     └──── Decision Engine ────┘
                ↓
      Investment Suggestion
     (Invest / Hold / Avoid)
```

---

## 🧰 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| Transformer Model | HuggingFace `bert-base-uncased` |
| Metrics | scikit-learn |
| Visualization | matplotlib, seaborn |
| Environment | Kaggle / Local (GPU recommended) |

---

## 📁 Project Structure

```
financial_sentiment_system/
├── config.py          # Hyperparameters & device configuration
├── dataset.py         # Financial news dataset & tokenization
├── model.py           # BERT + dual-head model architecture
├── train.py           # Multi-task training loop
├── evaluate.py        # Metrics: Accuracy, F1, ROC-AUC
├── predict.py         # Prediction & investment decision engine
├── visualize.py       # Charts: loss curves, confusion matrix, etc.
├── explainability.py  # Educational concept explanations
├── main.py            # Full pipeline orchestration
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```
This will: load data → train → evaluate → visualize → predict on demo headlines → save model.

### Interactive Prediction Mode
```bash
python main.py --predict
```

### View Concept Explanations
```bash
python main.py --explain
```

---

## 📊 Sample Output

```
📰 Headline: Company shares fell after revenue miss
─────────────────────────────────────
🎭 Sentiment:    Negative
🔒 Confidence:   92.34%
⚠️  Risk Score:   0.8234
📈 Risk Level:   High
💼 Suggestion:   ❌ Avoid Investment
```

---

## 💡 Investment Decision Logic

| Sentiment | Risk Score | Suggestion |
|-----------|-----------|------------|
| Positive | < 0.4 | ✅ Invest |
| Neutral | < 0.6 | ⏸️ Hold |
| Negative | any | ❌ Avoid |
| any | > 0.7 | ❌ Avoid |
| otherwise | otherwise | ⏸️ Hold |

---

## 🏆 Real-World Use Cases

1. **Stock Market Trading Automation** — Real-time sentiment signals for algorithmic trading
2. **Loan Risk Assessment in Banks** — Evaluate news about borrowers/sectors
3. **Hedge Fund Sentiment Monitoring** — Portfolio-wide news sentiment tracking
4. **Portfolio Advisory Apps** — Automated investment suggestions for retail investors
5. **Fraud Detection Monitoring** — Flag suspicious news patterns about entities

---

## 📈 Visualizations Generated

The system generates the following plots in `outputs/plots/`:
- **Training loss curves** (total, sentiment, risk)
- **Accuracy curves** (train vs. validation)
- **Confusion matrix** (sentiment classification)
- **Risk score distribution** (predicted vs. actual)
- **Sentiment distribution** (pie chart)

---

## 📘 Key Concepts (Run `python main.py --explain`)

- What is BERT & why it's used for NLP
- Fine-tuning pretrained models
- Transformers vs. LSTMs
- BERT tokenization (WordPiece)
- Sentiment analysis in finance
- Risk modeling with neural networks

---

## 🔧 Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `bert-base-uncased` | Pretrained model |
| `MAX_SEQ_LENGTH` | 128 | Token sequence length |
| `EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size |
| `LEARNING_RATE_BERT` | 2e-5 | LR for BERT layers |
| `LEARNING_RATE_HEAD` | 1e-3 | LR for custom heads |
| `DROPOUT_RATE` | 0.3 | Dropout probability |

---

*Built with ❤️ using PyTorch and HuggingFace Transformers*
