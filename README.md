<div align="center">

# 📡 FinSight AI
### Financial News Sentiment & Risk Prediction System

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

<br/>

> **State-of-the-art BERT transformer** fine-tuned on financial corpora for real-time sentiment classification, calibrated risk scoring, and actionable investment recommendations — all wrapped in a professional interactive dashboard.

<br/>

</div>

---

## ✨ What It Does

FinSight AI ingests a plain-English financial news headline and returns **three simultaneous predictions** in milliseconds:

| Output | Description | Range |
|--------|-------------|-------|
| 🎭 **Sentiment** | Positive / Negative / Neutral with confidence % | 0 – 100% |
| ⚡ **Risk Score** | Continuous calibrated investment risk | 0.0 (safe) – 1.0 (critical) |
| 💼 **Recommendation** | Actionable decision: Invest / Accumulate / Hold / Avoid | — |

---

## 🏆 Model Performance

After fine-tuning on an expanded 200+ sample contrastive financial dataset with top-6 BERT layer unfreezing and a linear warmup scheduler:

| Metric | Before Fine-Tune | After Fine-Tune | Improvement |
|--------|:----------------:|:---------------:|:-----------:|
| Validation Accuracy | 66.67% | **92.59%** | ↑ 25.9 pp |
| Macro F1 Score | 0.4603 | **0.9014** | ↑ 95.8% |
| Risk Score MAE | 0.2852 | **0.0826** | ↓ 71.0% |

---

## 🏗️ Architecture

```
Financial News Headline (plain text)
           │
           ▼
  BERT Tokenizer (WordPiece, max 128 tokens)
           │
           ▼
  BERT Encoder — bert-base-uncased
  12 Transformer Layers · 768-dim hidden · 109.6M params
  (Top 6 layers unfrozen for fine-tuning → 14.9M trainable)
           │
           ▼
    [CLS] Token Embedding (768-dim)
           │
    ┌──────┴────────────────────────┐
    ▼                               ▼
Sentiment Head                 Risk Head
Dropout(0.3) → Linear(768→3)  Dropout(0.3) → Linear(768→256)
→ Softmax                      → ReLU → Linear(256→1)
→ {Positive, Negative, Neutral} → Sigmoid → risk ∈ [0, 1]
    │                               │
    └──────────┬────────────────────┘
               ▼
        Decision Engine
  ┌──────────────────────────────────────┐
  │  Positive + Risk < 0.40  → ✅ Invest  │
  │  Positive + Risk ≤ 0.50  → 📈 Accumulate / Buy on Dips │
  │  Neutral  + Risk < 0.60  → ⏸️ Hold   │
  │  Negative OR Risk ≥ 0.65 → 🛑 Avoid  │
  └──────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
financial_sentiment_system/
│
├── 🧠 Core ML
│   ├── config.py           # All hyperparameters, paths & device config
│   ├── dataset.py          # 200+ sample financial dataset with contrastive examples
│   ├── model.py            # BERT + dual-head architecture (sentiment + risk)
│   ├── train.py            # Multi-task training loop with warmup scheduler
│   ├── evaluate.py         # Accuracy, Macro F1, ROC-AUC, Risk MAE metrics
│   └── predict.py          # Inference pipeline & investment decision engine
│
├── 🖥️ Interface
│   ├── app.py              # Streamlit professional dashboard (FinSight AI UI)
│   └── main.py             # CLI pipeline orchestrator
│
├── 📊 Utilities
│   ├── visualize.py        # Loss curves, confusion matrix, risk distribution
│   └── explainability.py   # NLP & finance concept explainers
│
├── outputs/                # Auto-generated (gitignored)
│   ├── trained_model.pth   # Saved model checkpoint
│   └── plots/              # Generated visualizations
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Arjit11/financial_sentiment_system.git
cd financial_sentiment_system
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python main.py
```
This runs the full pipeline: **load data → train → evaluate → generate plots → demo predictions → save checkpoint**.

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🎛️ Configuration Reference

All hyperparameters live in [`config.py`](config.py) — change once, applies everywhere:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `MODEL_NAME` | `bert-base-uncased` | Pre-trained HuggingFace model |
| `MAX_SEQ_LENGTH` | `128` | Max token length per headline |
| `EPOCHS` | `15` | Training epochs |
| `BATCH_SIZE` | `8` | Samples per gradient step |
| `LEARNING_RATE_BERT` | `2e-5` | LR for BERT encoder layers |
| `LEARNING_RATE_HEAD` | `5e-4` | LR for sentiment & risk heads |
| `WARMUP_RATIO` | `0.1` | Linear warmup fraction of total steps |
| `WEIGHT_DECAY` | `0.01` | AdamW regularisation |
| `DROPOUT_RATE` | `0.3` | Dropout for both heads |
| `VALIDATION_SPLIT` | `0.2` | 80/20 train-val split |
| `SENTIMENT_LOSS_WEIGHT` | `1.0` | Cross-entropy loss weight |
| `RISK_LOSS_WEIGHT` | `0.5` | MSE risk loss weight |

---

## 💼 Investment Decision Logic

The decision engine combines both outputs into one actionable signal:

| Sentiment | Risk Score | Decision |
|:---------:|:----------:|:--------:|
| Positive | < 0.40 | ✅ **Invest** |
| Positive | 0.40 – 0.50 | 📈 **Accumulate / Buy on Dips** |
| Neutral | < 0.60 | ⏸️ **Hold** |
| Negative | any | 🛑 **Avoid Investment** |
| any | ≥ 0.65 | 🛑 **Avoid Investment** |

> Multi-horizon headlines (e.g. *"stock expected to fall this quarter but rebound next on new iPhone"*) are correctly classified as **Accumulate / Buy on Dips** thanks to contrastive training examples.

---

## 📊 Example Predictions

```
📰 Apple expected to fall this quarter, bounce back next on new iPhone
   🎭 Sentiment  : Positive  (99.9% confidence)
   ⚡ Risk Score : 0.2144   (Low)
   💼 Decision   : 📈 Accumulate / Buy on Dips

📰 Tesla reports record quarterly profits
   🎭 Sentiment  : Positive  (97.2% confidence)
   ⚡ Risk Score : 0.0923   (Low)
   💼 Decision   : ✅ Invest

📰 Bank faces massive fraud investigation
   🎭 Sentiment  : Negative  (94.5% confidence)
   ⚡ Risk Score : 0.8741   (High)
   💼 Decision   : 🛑 Avoid Investment

📰 Fed maintains current interest rates
   🎭 Sentiment  : Neutral   (81.3% confidence)
   ⚡ Risk Score : 0.4412   (Medium)
   💼 Decision   : ⏸️ Hold
```

---

## 🖥️ Dashboard Features

The **FinSight AI** Streamlit dashboard includes:

- 📡 **Live scrolling ticker** — real-time-style market prices strip
- 🦸 **Hero section** — live model performance stats (Accuracy, F1, MAE)
- 🎨 **Colour-coded recommendation banners** — with shimmer animation
- 📊 **4-column KPI cards** — Sentiment, Risk Score, Confidence, Dominant Class
- 📈 **Animated probability bars** — Positive / Negative / Neutral distribution
- 🎯 **Risk gauge** — spring-physics cursor across Low → Critical spectrum
- 🧠 **AI Summary box** — natural language explanation of each prediction
- 📋 **Analysis history table** — last 10 analyses with time column
- 🔘 **Quick headline buttons** — one-click example headlines in sidebar

---

## 📈 Visualizations Generated

After training, plots are saved to `outputs/plots/`:

- **Training / Validation Loss Curves** — total, sentiment, and risk losses
- **Accuracy Curves** — train vs. validation per epoch
- **Confusion Matrix** — per-class sentiment classification
- **Risk Score Distribution** — predicted vs. ground truth histogram
- **Sentiment Pie Chart** — class balance in dataset

---

## 🧩 Key Technical Highlights

- **Multi-task Learning** — a single BERT backbone shared between two prediction heads (classification + regression) trained jointly with weighted loss
- **Contrastive Dataset** — training examples deliberately include contradictory multi-horizon statements to teach cross-clause temporal reasoning
- **Calibrated Risk Scores** — continuous 0.0–1.0 risk labels grounded in real financial risk fundamentals, not just binary flags
- **Linear Warmup Scheduler** — 10% warmup steps then linear decay for stable BERT fine-tuning
- **Best Checkpoint Restoration** — training saves the epoch with lowest validation loss, not the last epoch
- **Apple Silicon (MPS) Compatible** — `attn_implementation="eager"` ensures compatibility with M1/M2/M3 Macs
- **Dual Deployment Modes** — full BERT mode locally, lightweight keyword engine on memory-constrained servers (e.g. Render free tier)

---

## 🔧 CLI Usage

```bash
# Full training pipeline
python main.py

# Interactive prediction prompt
python main.py --predict

# Print concept explanations
python main.py --explain
```

---

## 🏢 Real-World Use Cases

| Industry | Use Case |
|----------|----------|
| 📈 Trading Desks | Real-time sentiment signals for algorithmic trading strategies |
| 🏦 Banks | Loan risk assessment — evaluate news about borrowers & sectors |
| 🔱 Hedge Funds | Portfolio-wide news sentiment monitoring & alerting |
| 💰 Retail Finance Apps | Automated investment guidance for retail investors |
| 🔍 Fraud Detection | Flag suspicious news patterns about entities |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.0+ |
| Transformer | HuggingFace `bert-base-uncased` (109.6M params) |
| Tokenization | WordPiece (AutoTokenizer) |
| Metrics | scikit-learn |
| Visualization | matplotlib · seaborn |
| Dashboard | Streamlit |
| Accelerator | CUDA · Apple MPS · CPU (auto-detected) |
| Version Control | Git · GitHub |

---

## 📦 Requirements

```
torch>=2.0
transformers>=4.30
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
pandas>=2.0
streamlit>=1.28
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

<div align="center">

Built with ❤️ using **PyTorch** and **HuggingFace Transformers**

⭐ Star this repo if you found it useful!

</div>
