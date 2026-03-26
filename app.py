"""
============================================================
Financial News Sentiment & Risk Prediction System
✨ Premium Streamlit Web Interface ✨
============================================================
A stunning, state-of-the-art web dashboard for analyzing
financial news headlines with BERT-powered AI.

Run with:
  streamlit run app.py
============================================================
"""

import streamlit as st
import torch
import torch.nn.functional as F
import sys
import os
import time
import numpy as np
from datetime import datetime

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, MODEL_SAVE_PATH, SENTIMENT_LABELS, RISK_LEVELS, MODEL_NAME
)
from model import FinancialSentimentRiskModel
from predict import classify_risk_level, get_investment_suggestion

# ══════════════════════════════════════════════
# Page Configuration
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="FinSight AI — Financial Sentiment Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# Premium CSS — Complete Custom Theme
# ══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ══════ ROOT VARIABLES ══════ */
    :root {
        --bg-primary: #0B0D1A;
        --bg-secondary: #111328;
        --bg-card: #151833;
        --bg-glass: rgba(21, 24, 51, 0.7);
        --border-subtle: rgba(124, 77, 255, 0.1);
        --border-glow: rgba(124, 77, 255, 0.3);
        --accent-blue: #00D2FF;
        --accent-purple: #7C4DFF;
        --accent-pink: #FF6B9D;
        --accent-green: #00E676;
        --accent-red: #FF5252;
        --accent-amber: #FFAB40;
        --text-primary: #FFFFFF;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --text-muted: rgba(255, 255, 255, 0.35);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.4);
        --shadow-glow-blue: 0 0 30px rgba(0, 210, 255, 0.15);
        --shadow-glow-purple: 0 0 30px rgba(124, 77, 255, 0.15);
    }

    /* ══════ GLOBAL ══════ */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary);
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* ══════ SCROLLBAR ══════ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-blue)); 
        border-radius: 3px; 
    }

    /* ══════ HERO ══════ */
    .hero {
        position: relative;
        background: linear-gradient(145deg, #0d1028 0%, #151a42 35%, #1f1555 65%, #0d0f28 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-subtle);
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: 
            radial-gradient(ellipse 600px 400px at 20% 50%, rgba(0, 210, 255, 0.08) 0%, transparent 70%),
            radial-gradient(ellipse 500px 350px at 80% 30%, rgba(124, 77, 255, 0.08) 0%, transparent 70%),
            radial-gradient(ellipse 300px 300px at 60% 80%, rgba(255, 107, 157, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(124, 77, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(60px);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.2);
        color: var(--accent-blue);
        padding: 0.35rem 1rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0E0FF 30%, var(--accent-blue) 60%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        line-height: 1.6;
        max-width: 650px;
        position: relative;
        z-index: 1;
    }

    /* ══════ GLASS CARD ══════ */
    .glass-card {
        background: linear-gradient(160deg, rgba(21, 24, 51, 0.85), rgba(11, 13, 26, 0.95));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124, 77, 255, 0.3), transparent);
    }
    .glass-card:hover {
        border-color: var(--border-glow);
        box-shadow: var(--shadow-card), var(--shadow-glow-purple);
        transform: translateY(-2px);
    }

    /* ══════ METRIC CARD ══════ */
    .metric-card {
        background: linear-gradient(160deg, rgba(21, 24, 51, 0.85), rgba(11, 13, 26, 0.95));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.8rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124, 77, 255, 0.3), transparent);
    }
    .metric-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: var(--shadow-card), var(--shadow-glow-purple);
    }
    .metric-icon {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* ══════ COLORS ══════ */
    .c-green  { color: var(--accent-green); }
    .c-red    { color: var(--accent-red); }
    .c-amber  { color: var(--accent-amber); }
    .c-blue   { color: var(--accent-blue); }
    .c-purple { color: var(--accent-purple); }
    .c-pink   { color: var(--accent-pink); }
    .c-white  { color: var(--text-primary); }

    /* ══════ SUGGESTION BANNERS ══════ */
    .banner {
        border-radius: 16px;
        padding: 1.2rem 2rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .banner::before {
        content: '';
        position: absolute;
        inset: 0;
        opacity: 0.15;
    }
    .banner-invest {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.12), rgba(0, 200, 83, 0.08));
        border: 1px solid rgba(0, 230, 118, 0.25);
        color: var(--accent-green);
        box-shadow: 0 4px 25px rgba(0, 230, 118, 0.12);
    }
    .banner-hold {
        background: linear-gradient(135deg, rgba(255, 171, 64, 0.12), rgba(255, 152, 0, 0.08));
        border: 1px solid rgba(255, 171, 64, 0.25);
        color: var(--accent-amber);
        box-shadow: 0 4px 25px rgba(255, 171, 64, 0.12);
    }
    .banner-avoid {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.12), rgba(244, 67, 54, 0.08));
        border: 1px solid rgba(255, 82, 82, 0.25);
        color: var(--accent-red);
        box-shadow: 0 4px 25px rgba(255, 82, 82, 0.12);
    }

    /* ══════ PROBABILITY BARS ══════ */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }
    .prob-label {
        width: 72px;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: right;
    }
    .prob-track {
        flex: 1;
        height: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .prob-fill-green  { background: linear-gradient(90deg, #00C853, #00E676); }
    .prob-fill-red    { background: linear-gradient(90deg, #F44336, #FF5252); }
    .prob-fill-amber  { background: linear-gradient(90deg, #FF8F00, #FFAB40); }
    .prob-pct {
        width: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
    }

    /* ══════ RISK GAUGE ══════ */
    .gauge-wrap {
        padding: 1rem 0;
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .gauge-track {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, 
            #00E676 0%, #69F0AE 20%, 
            #FFD740 40%, #FFAB40 55%, 
            #FF6E40 70%, #FF5252 85%, 
            #D50000 100%
        );
        position: relative;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    .gauge-needle {
        position: absolute;
        top: -6px;
        width: 24px;
        height: 24px;
        background: white;
        border-radius: 50%;
        border: 3px solid var(--bg-primary);
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.5), 0 2px 8px rgba(0,0,0,0.3);
        transform: translateX(-50%);
        transition: left 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-value {
        text-align: center;
        margin-top: 1rem;
    }
    .gauge-number {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .gauge-level {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 2px;
    }

    /* ══════ SECTION HEADERS ══════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 1rem 0;
    }
    .section-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, rgba(124, 77, 255, 0.2), rgba(0, 210, 255, 0.1));
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border: 1px solid rgba(124, 77, 255, 0.15);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
    }

    /* ══════ HISTORY TABLE ══════ */
    .history-row {
        display: grid;
        grid-template-columns: 1fr 90px 80px 110px;
        gap: 12px;
        align-items: center;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .history-row:hover {
        background: rgba(124, 77, 255, 0.06);
        border-color: rgba(124, 77, 255, 0.15);
    }
    .h-headline {
        font-size: 0.82rem;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .h-cell {
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
    }
    .h-badge {
        font-size: 0.68rem;
        padding: 3px 10px;
        border-radius: 100px;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .badge-invest { background: rgba(0, 230, 118, 0.12); color: var(--accent-green); border: 1px solid rgba(0, 230, 118, 0.2); }
    .badge-hold   { background: rgba(255, 171, 64, 0.12); color: var(--accent-amber); border: 1px solid rgba(255, 171, 64, 0.2); }
    .badge-avoid  { background: rgba(255, 82, 82, 0.12); color: var(--accent-red); border: 1px solid rgba(255, 82, 82, 0.2); }

    /* ══════ ANALYSIS BOX ══════ */
    .analysis-box {
        background: linear-gradient(160deg, rgba(0, 210, 255, 0.05), rgba(124, 77, 255, 0.03));
        border: 1px solid rgba(0, 210, 255, 0.12);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.7;
    }
    .analysis-box b {
        color: var(--text-primary);
    }

    /* ══════ TECH STACK FOOTER ══════ */
    .tech-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .tech-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(124, 77, 255, 0.08);
        color: rgba(179, 136, 255, 0.9);
        padding: 0.4rem 1rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 3px;
        border: 1px solid rgba(124, 77, 255, 0.12);
        letter-spacing: 0.3px;
    }

    /* ══════ SIDEBAR ══════ */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }
    .sidebar-box {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0.8rem;
    }
    .sb-label {
        font-size: 0.62rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    .sb-value {
        font-size: 0.88rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-top: 2px;
    }
    .sb-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        animation: dot-pulse 2s ease-in-out infinite;
    }
    @keyframes dot-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .sb-dot-green  { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
    .sb-dot-amber  { background: var(--accent-amber); box-shadow: 0 0 6px var(--accent-amber); }

    /* ══════ INPUT AREA ══════ */
    .input-section {
        background: linear-gradient(160deg, rgba(21, 24, 51, 0.6), rgba(11, 13, 26, 0.8));
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-subtle);
        margin-bottom: 1.5rem;
    }
    .input-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    /* ══════ HIDE STREAMLIT DEFAULTS ══════ */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Session State Init
# ══════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "headline_input" not in st.session_state:
    st.session_state.headline_input = ""


# ══════════════════════════════════════════════
# Model Loading (cached)
# ══════════════════════════════════════════════
@st.cache_resource
def load_model_and_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = FinancialSentimentRiskModel()
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        status = "trained"
    else:
        status = "untrained"
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, status


def predict(model, tokenizer, headline):
    from config import MAX_SEQ_LENGTH
    encoding = tokenizer(
        headline, add_special_tokens=True, max_length=MAX_SEQ_LENGTH,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        sentiment_logits, risk_score = model(input_ids, attention_mask)
    probs = F.softmax(sentiment_logits, dim=1).squeeze(0)
    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()
    sentiment = SENTIMENT_LABELS[predicted_class]
    risk_val = risk_score.item()
    risk_level = classify_risk_level(risk_val)
    suggestion = get_investment_suggestion(sentiment, risk_val)
    return {
        "sentiment": sentiment, "confidence": confidence,
        "probabilities": {SENTIMENT_LABELS[i]: probs[i].item() for i in range(len(SENTIMENT_LABELS))},
        "risk_score": risk_val, "risk_level": risk_level, "suggestion": suggestion,
    }


# Load model
model, tokenizer, model_status = load_model_and_tokenizer()


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 1.6rem; font-weight: 800; 
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -0.5px;">
            FinSight AI
        </div>
        <div style="font-size: 0.7rem; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase;">
            Intelligence Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    dot_class = "sb-dot-green" if model_status == "trained" else "sb-dot-amber"
    status_txt = "Model Active" if model_status == "trained" else "Untrained — run main.py"

    st.markdown(f"""
    <div class="sidebar-box">
        <div class="sb-label">Status</div>
        <div class="sb-value"><span class="sb-dot {dot_class}"></span>{status_txt}</div>
    </div>
    <div class="sidebar-box">
        <div class="sb-label">Model</div>
        <div class="sb-value">BERT base uncased</div>
    </div>
    <div class="sidebar-box">
        <div class="sb-label">Device</div>
        <div class="sb-value">{str(DEVICE).upper()}</div>
    </div>
    <div class="sidebar-box">
        <div class="sb-label">Parameters</div>
        <div class="sb-value">109.6M total • 14.9M trainable</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted); 
         letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.6rem;">
        Decision Logic
    </div>
    <div class="sidebar-box" style="font-size: 0.8rem; color: var(--text-secondary); line-height: 2;">
        <span class="c-green">●</span> <b>Invest</b> — Positive & Risk &lt; 0.4<br>
        <span class="c-amber">●</span> <b>Hold</b> — Neutral & Risk &lt; 0.6<br>
        <span class="c-red">●</span> <b>Avoid</b> — Negative OR Risk &gt; 0.7
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted);
         letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.6rem;">
        Quick Headlines
    </div>
    """, unsafe_allow_html=True)

    examples = [
        ("Tesla reports record quarterly profits", "🟢"),
        ("Bank faces massive fraud investigation", "🔴"),
        ("Fed maintains current interest rates", "🟡"),
        ("Apple stock surges on strong earnings", "🟢"),
        ("Company shares fell after revenue miss", "🔴"),
        ("Market trades sideways with no direction", "🟡"),
        ("Tech layoffs accelerate across industry", "🔴"),
        ("Startup secures billion dollar funding", "🟢"),
    ]

    for text, dot in examples:
        if st.button(f"{dot} {text}", key=text, use_container_width=True):
            st.session_state.headline_input = text


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

# ── Hero Section ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ BERT-POWERED AI ENGINE</div>
    <div class="hero-title">Financial Sentiment<br>& Risk Analyzer</div>
    <div class="hero-desc">
        Analyze financial news headlines with state-of-the-art transformer AI. 
        Get instant sentiment classification, risk assessment, and 
        actionable investment recommendations.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Section ──
st.markdown("""
<div class="section-header">
    <div class="section-icon">📝</div>
    <div class="section-title">Analyze a Headline</div>
</div>
""", unsafe_allow_html=True)

headline = st.text_input(
    "headline",
    value=st.session_state.get("headline_input", ""),
    placeholder="Enter a financial news headline — e.g. 'Apple announces groundbreaking AI chip'",
    label_visibility="collapsed",
)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    analyze_clicked = st.button("🔍  Analyze with BERT", type="primary", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🗑️  Clear", use_container_width=True)

if clear_clicked:
    st.session_state.headline_input = ""
    st.session_state.history = []
    st.rerun()


# ══════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════
if analyze_clicked and headline.strip():
    with st.spinner("🧠 Processing with BERT transformer..."):
        result = predict(model, tokenizer, headline.strip())
        time.sleep(0.2)

    sentiment = result["sentiment"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    risk_score = result["risk_score"]
    risk_level = result["risk_level"]
    suggestion = result["suggestion"]

    # Add to history
    st.session_state.history.insert(0, {
        "headline": headline.strip(),
        "sentiment": sentiment,
        "risk": risk_score,
        "suggestion": suggestion,
        "time": datetime.now().strftime("%H:%M"),
    })
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]

    # Determine suggestion style
    if "Invest" in suggestion and "Avoid" not in suggestion:
        banner_cls, sugg_text, sugg_icon = "banner-invest", "INVEST", "✅"
    elif "Avoid" in suggestion:
        banner_cls, sugg_text, sugg_icon = "banner-avoid", "AVOID INVESTMENT", "🛑"
    else:
        banner_cls, sugg_text, sugg_icon = "banner-hold", "HOLD POSITION", "⏸️"

    # Color helpers
    sent_color = {"Positive": "c-green", "Negative": "c-red", "Neutral": "c-amber"}[sentiment]
    risk_color = {"Low": "c-green", "Medium": "c-amber", "High": "c-red"}[risk_level]

    st.markdown("---")

    # ── Suggestion Banner ──
    st.markdown(f"""
    <div class="banner {banner_cls}">
        {sugg_icon} &nbsp; RECOMMENDATION: {sugg_text}
    </div>
    """, unsafe_allow_html=True)

    # ── Three Metric Cards ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎭</div>
            <div class="metric-label">Sentiment</div>
            <div class="metric-value {sent_color}">{sentiment}</div>
            <div class="metric-sub">{confidence:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Risk Score</div>
            <div class="metric-value {risk_color}" style="font-family: 'JetBrains Mono', monospace;">{risk_score:.4f}</div>
            <div class="metric-sub">{risk_level} Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{sugg_icon}</div>
            <div class="metric-label">Action</div>
            <div class="metric-value" style="font-size: 1.5rem; color: var(--text-primary);">{sugg_text}</div>
            <div class="metric-sub">Sentiment + Risk based</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detailed Analysis: Probabilities + Risk Gauge ──
    col_left, col_right = st.columns(2)

    with col_left:
        prob_colors = {"Positive": ("c-green", "prob-fill-green"), "Negative": ("c-red", "prob-fill-red"), "Neutral": ("c-amber", "prob-fill-amber")}

        prob_html = """<div class="section-header" style="margin-top: 0;">
            <div class="section-icon">📊</div>
            <div class="section-title">Sentiment Probabilities</div>
        </div><div class="glass-card">"""
        for label, prob in probs.items():
            txt_cls, bar_cls = prob_colors[label]
            pct_width = f"{prob * 100:.1f}"
            pct_text = f"{prob:.1%}"
            prob_html += f"""<div class="prob-row"><span class="prob-label {txt_cls}">{label}</span><div class="prob-track"><div class="prob-fill {bar_cls}" style="width: {pct_width}%;"></div></div><span class="prob-pct {txt_cls}">{pct_text}</span></div>"""
        prob_html += "</div>"
        st.markdown(prob_html, unsafe_allow_html=True)

    with col_right:
        needle_pct = max(2, min(98, risk_score * 100))
        risk_html = f"""<div class="section-header" style="margin-top: 0;">
            <div class="section-icon">🎯</div>
            <div class="section-title">Risk Assessment</div>
        </div><div class="glass-card"><div class="gauge-wrap"><div class="gauge-labels"><span>Low Risk</span><span>Medium</span><span>High Risk</span></div><div class="gauge-track"><div class="gauge-needle" style="left: {needle_pct}%;"></div></div><div class="gauge-value"><div class="gauge-number {risk_color}">{risk_score:.4f}</div><div class="gauge-level {risk_color}">{risk_level} Risk</div></div></div></div>"""
        st.markdown(risk_html, unsafe_allow_html=True)

    # ── Analysis Summary ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="analysis-box">
        <b>🧠 AI Analysis Summary</b><br><br>
        The BERT transformer analyzed <b>"{headline.strip()}"</b> and classified it with 
        <b class="{sent_color}">{sentiment.lower()}</b> sentiment at 
        <b>{confidence:.1%}</b> confidence. The custom risk neural network assigned a 
        <b class="{risk_color}">{risk_level.lower()} risk</b> score of 
        <b style="font-family: 'JetBrains Mono', monospace;">{risk_score:.4f}</b>.
        Combined analysis recommends: <b>{sugg_text.lower()}</b>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════
if st.session_state.history:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div class="section-title">Analysis History</div>
    </div>
    """, unsafe_allow_html=True)

    # Header row
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 90px 80px 110px; gap: 12px; 
         padding: 0.5rem 1rem; font-size: 0.62rem; color: var(--text-muted); 
         text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
        <span>Headline</span>
        <span style="text-align:center;">Sentiment</span>
        <span style="text-align:center;">Risk</span>
        <span style="text-align:center;">Action</span>
    </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.history:
        s_color = {"Positive": "c-green", "Negative": "c-red", "Neutral": "c-amber"}[item["sentiment"]]
        r_val = item["risk"]
        r_color = "c-green" if r_val < 0.4 else ("c-amber" if r_val < 0.7 else "c-red")

        if "Invest" in item["suggestion"] and "Avoid" not in item["suggestion"]:
            badge_cls, badge_txt = "badge-invest", "INVEST"
        elif "Avoid" in item["suggestion"]:
            badge_cls, badge_txt = "badge-avoid", "AVOID"
        else:
            badge_cls, badge_txt = "badge-hold", "HOLD"

        st.markdown(f"""
        <div class="history-row">
            <div class="h-headline">{item['headline']}</div>
            <div class="h-cell {s_color}">{item['sentiment']}</div>
            <div class="h-cell {r_color}" style="font-family: 'JetBrains Mono', monospace;">{r_val:.3f}</div>
            <div><div class="h-badge {badge_cls}">{badge_txt}</div></div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="tech-footer">
    <div class="tech-tag">🤖 BERT Transformer</div>
    <div class="tech-tag">📊 Multi-Task Learning</div>
    <div class="tech-tag">⚡ Real-Time Analysis</div>
    <div class="tech-tag">🎯 Risk Neural Network</div>
    <div class="tech-tag">💼 Investment Engine</div>
    <div class="tech-tag">🔥 PyTorch</div>
    <div style="margin-top: 1rem; font-size: 0.65rem; color: var(--text-muted); letter-spacing: 1px;">
        FinSight AI • Built with PyTorch & HuggingFace Transformers
    </div>
</div>
""", unsafe_allow_html=True)
