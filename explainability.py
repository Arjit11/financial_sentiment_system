"""
============================================================
Financial News Sentiment & Risk Prediction System
Explainability Module
============================================================
Beginner-friendly but technically accurate explanations of
the key concepts used in this system. Perfect for reports,
viva preparation, and understanding the architecture.
============================================================
"""


def print_section(title, content):
    """Helper to print a formatted explanation section."""
    print(f"\n{'='*60}")
    print(f"  📘 {title}")
    print(f"{'='*60}")
    print(content)
    print()


def explain_bert():
    """Explain what BERT is and why it's used."""
    print_section("What is BERT?", """
  BERT (Bidirectional Encoder Representations from Transformers)
  is a language model created by Google in 2018.

  Key Properties:
  ─────────────
  • Bidirectional: Unlike older models that read text left-to-right,
    BERT reads in BOTH directions simultaneously. This gives it
    much deeper understanding of context.

  • Pretrained: BERT was trained on massive text datasets (Wikipedia
    + BookCorpus = ~3.3 billion words). It already "knows" language.

  • Transformer-based: Uses the Transformer architecture with
    self-attention mechanisms to capture relationships between
    all words in a sentence.

  In Our System:
  ─────────────
  We use 'bert-base-uncased' which has:
    - 12 transformer layers
    - 768 hidden dimensions
    - 110 million parameters
    - Trained on lowercased English text

  BERT converts each headline into a rich 768-dimensional vector
  that captures the meaning, sentiment, and context of the text.
  We then pass this vector to our custom heads for prediction.
""")


def explain_fine_tuning():
    """Explain what fine-tuning is and why it matters."""
    print_section("What is Fine-Tuning?", """
  Fine-tuning is the process of taking a pretrained model and
  adapting it to a specific task using task-specific data.

  Analogy:
  ─────────
  Think of BERT as a university graduate who has broad knowledge.
  Fine-tuning is like giving them specialized job training.
  They don't start from scratch — they adapt their existing
  knowledge to the new domain (financial news).

  How It Works:
  ────────────
  1. Start with pretrained BERT (general language knowledge)
  2. Add custom layers on top (our sentiment & risk heads)
  3. Train on financial news data with a small learning rate
  4. BERT adjusts slightly while custom heads learn from scratch

  Why Differential Learning Rates:
  ────────────────────────────────
  • BERT layers: Very small LR (2e-5) — preserve pretrained knowledge
  • Custom heads: Larger LR (1e-3) — learn task quickly

  Benefits:
  ─────────
  ✓ Requires much less data than training from scratch
  ✓ Much faster training (minutes vs. weeks)
  ✓ Better performance due to transfer learning
  ✓ Works well even with small datasets
""")


def explain_transformers_vs_lstm():
    """Explain why Transformers are better than LSTMs."""
    print_section("Transformers vs. LSTM — Why Transformers Win", """
  LSTMs (Long Short-Term Memory) were the standard for NLP
  before Transformers. Here's why Transformers are superior:

  ┌──────────────────┬───────────────────┬──────────────────────┐
  │   Feature        │   LSTM            │   Transformer        │
  ├──────────────────┼───────────────────┼──────────────────────┤
  │ Processing       │ Sequential        │ Parallel             │
  │ Context          │ Limited window    │ Full sequence         │
  │ Long-range deps  │ Struggles         │ Excellent            │
  │ Training speed   │ Slow              │ Fast (parallelizable)│
  │ Bidirectional    │ Requires BiLSTM   │ Native (BERT)        │
  │ Scalability      │ Poor              │ Excellent            │
  └──────────────────┴───────────────────┴──────────────────────┘

  Key Advantage — Self-Attention:
  ───────────────────────────────
  Transformers use "self-attention" which lets every word
  attend to every other word in the sentence simultaneously.

  Example: "Tesla reports record profits despite chip shortage"
  
  • LSTM: Processes word by word. By the time it reaches
    "shortage", it may have partially forgotten "Tesla".
  
  • Transformer: Every word can directly attend to "Tesla",
    "profits", and "shortage" simultaneously, understanding
    the full context at once.

  For financial news, this is critical because keywords like
  "profits", "loss", "surge", "crash" need to be understood
  in context with the company and situation.
""")


def explain_tokenization():
    """Explain what tokenization is."""
    print_section("What is Tokenization?", """
  Tokenization is the process of converting raw text into
  numerical tokens that a model can process.

  BERT Tokenization Steps:
  ────────────────────────
  1. Input:  "Tesla reports record profits"
  2. Tokens: ["[CLS]", "tesla", "reports", "record", "profits", "[SEP]"]
  3. IDs:    [101, 26060, 4311, 2501, 11372, 102]

  Special Tokens:
  ──────────────
  • [CLS] (101) — Added at the start. Its final embedding is
    used as the "sentence representation" for classification.
  
  • [SEP] (102) — Added at the end. Marks sentence boundary.
  
  • [PAD] (0) — Added to make all sequences the same length.

  WordPiece Tokenization:
  ──────────────────────
  BERT uses WordPiece, which can split unknown words into
  known subwords:
  
  "cryptocurrency" → ["crypto", "##currency"]
  
  This means BERT can handle words it hasn't seen before
  by breaking them into familiar pieces.

  Attention Mask:
  ──────────────
  A binary mask that tells BERT which tokens are real (1)
  and which are padding (0), so it ignores padding tokens.
""")


def explain_sentiment_analysis():
    """Explain what sentiment analysis is."""
    print_section("What is Sentiment Analysis?", """
  Sentiment analysis is the task of determining the emotional
  tone or opinion expressed in a piece of text.

  In Financial Context:
  ────────────────────
  • Positive: "Tesla reports record profits" → Bullish signal
  • Negative: "Bank faces fraud investigation" → Bearish signal
  • Neutral:  "Fed maintains interest rates"  → No strong signal

  Why It Matters for Finance:
  ──────────────────────────
  Research shows that news sentiment is one of the strongest
  predictors of short-term stock price movements.

  • Positive sentiment → Prices tend to rise
  • Negative sentiment → Prices tend to fall
  • Speed matters — first to detect sentiment gains advantage

  Our Approach:
  ────────────
  1. BERT encodes the headline into a 768-dim vector
  2. This vector captures nuanced sentiment signals
  3. Our classification head maps it to 3 classes
  4. Softmax gives probability distribution over classes
  5. The class with highest probability is the prediction
  6. The probability value serves as confidence score

  Real-World Applications:
  ───────────────────────
  • Algorithmic trading systems
  • Market sentiment dashboards
  • News-based portfolio rebalancing
  • Regulatory risk monitoring
""")


def explain_risk_modeling():
    """Explain what risk modeling is."""
    print_section("What is Risk Modeling?", """
  Risk modeling estimates the probability or severity of
  a negative financial outcome based on available information.

  Our Risk Score (0 to 1):
  ───────────────────────
  • 0.0 – 0.4 → Low Risk    → Favorable conditions
  • 0.4 – 0.7 → Medium Risk → Mixed signals, caution needed
  • 0.7 – 1.0 → High Risk   → Significant danger signals

  How Our Model Computes Risk:
  ───────────────────────────
  1. BERT processes the headline
  2. The [CLS] embedding goes through our Risk Head:
     768 → Dropout → 128 → ReLU → 1 → Sigmoid
  3. Sigmoid ensures output is between 0 and 1
  4. The deeper architecture (768→128→1) allows the model
     to learn complex, nonlinear risk patterns

  Why Not Just Use Sentiment?
  ──────────────────────────
  Risk is different from sentiment:
  
  • "Company sells building to pay debts" 
    → Neutral sentiment, but HIGH risk
  
  • "New CEO appointed after board disagreement"
    → Neutral sentiment, but MEDIUM risk (uncertainty)
  
  The risk head learns these nuanced patterns that pure
  sentiment classification might miss.

  Investment Decision Integration:
  ───────────────────────────────
  Our system combines BOTH sentiment and risk score
  to make investment suggestions, similar to how
  professional analysts consider multiple factors.
""")


def show_all_explanations():
    """Display all educational explanations."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  📚 EXPLAINABILITY MODULE — Key Concepts Explained".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    
    explain_bert()
    explain_fine_tuning()
    explain_transformers_vs_lstm()
    explain_tokenization()
    explain_sentiment_analysis()
    explain_risk_modeling()
    
    print("\n" + "="*60)
    print("  ✅ All concepts explained!")
    print("  These explanations cover the theory behind this system.")
    print("="*60)


# ── Quick test ──
if __name__ == "__main__":
    show_all_explanations()
