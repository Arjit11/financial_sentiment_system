"""
============================================================
Financial News Sentiment & Risk Prediction System
Prediction & Decision Engine Module
============================================================
Contains:
  - Single headline prediction
  - Batch prediction
  - Investment decision logic
  - Confidence scoring
  - Risk level classification
  - Formatted output display
============================================================
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    DEVICE, MODEL_NAME, MAX_SEQ_LENGTH,
    SENTIMENT_LABELS, RISK_LEVELS
)


def classify_risk_level(risk_score):
    """
    Classify a numeric risk score into a risk level category.
    
    Returns:
        str: "Low", "Medium", or "High"
    """
    for level, (low, high) in RISK_LEVELS.items():
        if low <= risk_score < high:
            return level
    return "High"  # Default for edge case (risk = 1.0)


def get_investment_suggestion(sentiment_label, risk_score):
    """
    Investment Decision Engine.
    
    Decision Logic:
      ┌─────────────┬──────────────┬──────────────┐
      │  Condition   │  Risk Score  │  Suggestion  │
      ├─────────────┼──────────────┼──────────────┤
      │  Positive   │  < 0.4       │  ✅ Invest    │
      │  Neutral    │  < 0.6       │  ⏸  Hold     │
      │  Negative   │  any         │  ❌ Avoid    │
      │  any        │  > 0.7       │  ❌ Avoid    │
      │  otherwise  │  otherwise   │  ⏸  Hold     │
      └─────────────┴──────────────┴──────────────┘
    
    Args:
        sentiment_label: str ("Positive", "Negative", "Neutral")
        risk_score: float in [0, 1]
    
    Returns:
        str: Investment suggestion
    """
    if sentiment_label == "Negative" or risk_score > 0.7:
        return "❌ Avoid Investment"
    elif sentiment_label == "Positive" and risk_score < 0.4:
        return "✅ Invest"
    elif sentiment_label == "Neutral" and risk_score < 0.6:
        return "⏸️  Hold"
    else:
        return "⏸️  Hold"


def predict_single(model, tokenizer, headline):
    """
    Predict sentiment, risk, and investment suggestion for a single headline.
    
    Args:
        model:     Trained FinancialSentimentRiskModel
        tokenizer: BERT tokenizer
        headline:  str — financial news headline
    
    Returns:
        result: Dict with all prediction outputs
    """
    model.eval()
    model.to(DEVICE)
    
    # Tokenize the headline
    encoding = tokenizer(
        headline,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        sentiment_logits, risk_score = model(input_ids, attention_mask)
    
    # Sentiment prediction with confidence
    sentiment_probs = F.softmax(sentiment_logits, dim=1).squeeze(0)
    predicted_class = torch.argmax(sentiment_probs).item()
    confidence = sentiment_probs[predicted_class].item()
    sentiment_label = SENTIMENT_LABELS[predicted_class]
    
    # Risk score
    risk_value = risk_score.item()
    risk_level = classify_risk_level(risk_value)
    
    # Investment suggestion
    suggestion = get_investment_suggestion(sentiment_label, risk_value)
    
    result = {
        "headline": headline,
        "sentiment": sentiment_label,
        "sentiment_confidence": confidence,
        "sentiment_probabilities": {
            SENTIMENT_LABELS[i]: round(sentiment_probs[i].item(), 4)
            for i in range(len(SENTIMENT_LABELS))
        },
        "risk_score": round(risk_value, 4),
        "risk_level": risk_level,
        "suggestion": suggestion,
    }
    
    return result


def predict_batch(model, tokenizer, headlines):
    """
    Predict on a batch of headlines.
    
    Args:
        model:     Trained model
        tokenizer: BERT tokenizer
        headlines: List[str]
    
    Returns:
        results: List[Dict] — prediction for each headline
    """
    return [predict_single(model, tokenizer, h) for h in headlines]


def display_prediction(result):
    """
    Display a formatted prediction result.
    """
    print("\n" + "─" * 60)
    print(f"  📰 Headline: {result['headline']}")
    print(f"  ─────────────────────────────────")
    print(f"  🎭 Sentiment:    {result['sentiment']}")
    print(f"  🔒 Confidence:   {result['sentiment_confidence']:.2%}")
    print(f"  📊 Probabilities:")
    for label, prob in result["sentiment_probabilities"].items():
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"       {label:>8}: {bar} {prob:.4f}")
    print(f"  ⚠️  Risk Score:   {result['risk_score']:.4f}")
    print(f"  📈 Risk Level:   {result['risk_level']}")
    print(f"  💼 Suggestion:   {result['suggestion']}")
    print("─" * 60)


def display_batch_predictions(results):
    """
    Display predictions for a batch of headlines.
    """
    print("\n" + "=" * 60)
    print("  🏦 FINANCIAL NEWS ANALYSIS RESULTS")
    print("=" * 60)
    
    for result in results:
        display_prediction(result)
    
    print(f"\n  Total headlines analyzed: {len(results)}")
    
    # Summary statistics
    sentiments = [r["sentiment"] for r in results]
    avg_risk = sum(r["risk_score"] for r in results) / len(results)
    
    print(f"  Sentiment breakdown: "
          f"Positive={sentiments.count('Positive')}, "
          f"Negative={sentiments.count('Negative')}, "
          f"Neutral={sentiments.count('Neutral')}")
    print(f"  Average risk score: {avg_risk:.4f}")
    print("=" * 60)


def interactive_prediction(model, tokenizer):
    """
    Interactive prediction mode — type headlines to get instant analysis.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 60)
    print("  🔮 INTERACTIVE PREDICTION MODE")
    print("  Type a financial news headline to analyze it.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    while True:
        headline = input("\n  📝 Enter headline: ").strip()
        
        if headline.lower() in ("quit", "exit", "q"):
            print("  Exiting interactive mode. Goodbye!")
            break
        
        if not headline:
            print("  ⚠️  Please enter a headline.")
            continue
        
        result = predict_single(model, tokenizer, headline)
        display_prediction(result)


# ── Quick test ──
if __name__ == "__main__":
    from model import FinancialSentimentRiskModel
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = FinancialSentimentRiskModel()
    
    # Test with sample headlines (untrained model)
    test_headlines = [
        "Tesla reports record profits in Q4",
        "Bank faces massive fraud investigation",
        "Market trades sideways with no clear direction",
    ]
    
    results = predict_batch(model, tokenizer, test_headlines)
    display_batch_predictions(results)
