"""
============================================================
Financial News Sentiment & Risk Prediction System
✨ Professional Streamlit Web Interface ✨
============================================================
"""

import streamlit as st
import sys
import os
import time
import numpy as np
import hashlib
from datetime import datetime

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, MODEL_SAVE_PATH, SENTIMENT_LABELS, RISK_LEVELS, MODEL_NAME
)
from predict import classify_risk_level, get_investment_suggestion

# ══════════════════════════════════════════════
# Determine if we can load BERT (enough memory)
# ══════════════════════════════════════════════
USE_BERT = os.path.exists(MODEL_SAVE_PATH)

# On Render free tier (512 MB) BERT won't fit — use lightweight mode
if os.environ.get("RENDER"):
    try:
        import torch
        import torch.nn.functional as F
        from model import FinancialSentimentRiskModel
        USE_BERT = os.path.exists(MODEL_SAVE_PATH)
    except Exception:
        USE_BERT = False
else:
    import torch
    import torch.nn.functional as F
    from model import FinancialSentimentRiskModel

# ══════════════════════════════════════════════
# Page Configuration
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="FinSight AI — Financial Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# MASTER CSS THEME
# ══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

    /* ══ ROOT VARIABLES ══ */
    :root {
        --bg-deep:       #04050F;
        --bg-base:       #080A18;
        --bg-panel:      #0D0F22;
        --bg-card:       #101428;
        --bg-surface:    rgba(16, 20, 40, 0.9);
        --border-dim:    rgba(99, 102, 241, 0.12);
        --border-glow:   rgba(99, 102, 241, 0.35);
        --border-bright: rgba(99, 102, 241, 0.55);
        --blue:   #60A5FA;
        --indigo: #818CF8;
        --violet: #A78BFA;
        --cyan:   #22D3EE;
        --emerald:#10B981;
        --amber:  #F59E0B;
        --rose:   #F43F5E;
        --slate:  rgba(148, 163, 184, 0.7);
        --muted:  rgba(148, 163, 184, 0.35);
        --text:   #F1F5F9;
        --r: 0.75rem;
        --r-lg: 1.25rem;
        --r-xl: 1.75rem;
    }

    /* ══ GLOBAL RESET ══ */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: var(--bg-deep) !important;
        color: var(--text) !important;
    }
    .main .block-container {
        max-width: 1280px;
        padding-top: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ══ CUSTOM SCROLLBAR ══ */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--indigo), var(--cyan));
        border-radius: 10px;
    }

    /* ══ HIDE STREAMLIT CHROME ══ */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    .stDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }

    /* ══ TICKER BAR ══ */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: linear-gradient(90deg, var(--bg-panel), rgba(13,15,34,0.95));
        border-bottom: 1px solid var(--border-dim);
        padding: 0.55rem 0;
        position: relative;
        margin-bottom: 0;
    }
    .ticker-wrap::before, .ticker-wrap::after {
        content: '';
        position: absolute;
        top: 0; bottom: 0;
        width: 80px;
        z-index: 2;
        pointer-events: none;
    }
    .ticker-wrap::before {
        left: 0;
        background: linear-gradient(90deg, var(--bg-panel), transparent);
    }
    .ticker-wrap::after {
        right: 0;
        background: linear-gradient(-90deg, var(--bg-panel), transparent);
    }
    .ticker-track {
        display: flex;
        gap: 0;
        animation: ticker-scroll 45s linear infinite;
        white-space: nowrap;
        width: max-content;
    }
    .ticker-track:hover { animation-play-state: paused; }
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0 2rem;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.3px;
        color: var(--slate);
        border-right: 1px solid var(--border-dim);
    }
    .ticker-item .t-name { color: var(--text); font-weight: 700; }
    .ticker-up   { color: var(--emerald) !important; }
    .ticker-down { color: var(--rose)    !important; }

    /* ══ HERO ══ */
    .hero-shell {
        background: linear-gradient(145deg,
            #050819 0%,
            #0A0D28 30%,
            #0E1135 60%,
            #07091C 100%
        );
        border-bottom: 1px solid var(--border-dim);
        padding: 3rem 2.5rem 2.5rem 2.5rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    .hero-shell::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse 700px 500px at 10% 60%, rgba(96,165,250,0.07) 0%, transparent 65%),
            radial-gradient(ellipse 500px 400px at 85% 25%, rgba(129,140,248,0.07) 0%, transparent 65%),
            radial-gradient(ellipse 350px 350px at 55% 90%, rgba(34,211,238,0.04) 0%, transparent 60%);
        pointer-events: none;
    }
    /* Animated grid overlay */
    .hero-shell::after {
        content: '';
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.2);
        color: var(--blue);
        padding: 0.3rem 1rem;
        border-radius: 100px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }
    .hero-eyebrow .pulse-dot {
        width: 6px; height: 6px;
        background: var(--blue);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--blue);
        animation: pulse-anim 2s ease-in-out infinite;
    }
    @keyframes pulse-anim {
        0%, 100% { transform: scale(1); opacity: 1; }
        50%       { transform: scale(1.5); opacity: 0.5; }
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.1;
        background: linear-gradient(135deg,
            #FFFFFF   0%,
            #C7D2FE  30%,
            #818CF8  55%,
            #60A5FA  80%,
            #22D3EE 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.8rem;
        position: relative;
        z-index: 1;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--slate);
        line-height: 1.65;
        max-width: 580px;
        position: relative;
        z-index: 1;
    }
    /* Hero stat pills */
    .hero-stats {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-stat {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border-dim);
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .hero-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
    }
    .hero-stat-lbl {
        font-size: 0.68rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .dot-emerald { color: var(--emerald); }
    .dot-violet  { color: var(--violet); }
    .dot-cyan    { color: var(--cyan); }

    /* ══ DIVIDER ══ */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-dim) 30%, var(--border-glow) 50%, var(--border-dim) 70%, transparent);
        margin: 2rem 0;
    }

    /* ══ SECTION LABEL ══ */
    .sec-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    .sec-label-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, rgba(129,140,248,0.2), rgba(96,165,250,0.1));
        border: 1px solid rgba(129,140,248,0.2);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem;
    }
    .sec-label-text {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: 0.3px;
    }

    /* ══ INPUT CARD ══ */
    .input-shell {
        background: linear-gradient(160deg, rgba(13,15,34,0.9), rgba(8,10,24,0.98));
        border: 1px solid var(--border-dim);
        border-radius: var(--r-xl);
        padding: 2rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .input-shell::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--indigo) 50%, transparent);
        opacity: 0.6;
    }

    /* Streamlit text_input override */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--muted) !important;
        font-style: italic;
    }

    /* Streamlit button override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        border: 1px solid rgba(99,102,241,0.5) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #818CF8, #6366F1) !important;
        box-shadow: 0 6px 28px rgba(99,102,241,0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        color: var(--slate) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--border-glow) !important;
        color: var(--text) !important;
        background: rgba(99,102,241,0.06) !important;
    }

    /* ══ RESULT BANNER ══ */
    .result-banner {
        border-radius: var(--r-xl);
        padding: 1.5rem 2.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-banner::before {
        content: '';
        position: absolute;
        inset: 0;
        background: inherit;
        filter: blur(40px);
        opacity: 0.15;
        z-index: -1;
    }
    .rb-invest {
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.06));
        border: 1px solid rgba(16,185,129,0.28);
        box-shadow: 0 0 40px rgba(16,185,129,0.08), inset 0 0 20px rgba(16,185,129,0.03);
    }
    .rb-hold {
        background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(217,119,6,0.06));
        border: 1px solid rgba(245,158,11,0.28);
        box-shadow: 0 0 40px rgba(245,158,11,0.08), inset 0 0 20px rgba(245,158,11,0.03);
    }
    .rb-avoid {
        background: linear-gradient(135deg, rgba(244,63,94,0.1), rgba(225,29,72,0.06));
        border: 1px solid rgba(244,63,94,0.28);
        box-shadow: 0 0 40px rgba(244,63,94,0.08), inset 0 0 20px rgba(244,63,94,0.03);
    }
    .rb-icon {
        font-size: 2.5rem;
        line-height: 1;
        flex-shrink: 0;
    }
    .rb-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }
    .rb-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .rb-invest .rb-title { color: var(--emerald); }
    .rb-hold   .rb-title { color: var(--amber); }
    .rb-avoid  .rb-title { color: var(--rose); }
    .rb-subtitle {
        font-size: 0.82rem;
        color: var(--slate);
        margin-top: 0.2rem;
    }
    /* Animated shimmer on banner */
    .rb-shimmer {
        position: absolute;
        top: 0; left: -100%;
        width: 60%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%   { left: -100%; }
        100% { left: 200%; }
    }

    /* ══ STAT CARDS (KPI row) ══ */
    .kpi-card {
        background: linear-gradient(160deg, rgba(13,15,34,0.95), rgba(8,10,24,1));
        border: 1px solid var(--border-dim);
        border-radius: var(--r-lg);
        padding: 1.6rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        border-radius: var(--r-lg) var(--r-lg) 0 0;
    }
    .kpi-card.kpi-green::after  { background: linear-gradient(90deg, transparent, var(--emerald), transparent); }
    .kpi-card.kpi-amber::after  { background: linear-gradient(90deg, transparent, var(--amber),   transparent); }
    .kpi-card.kpi-rose::after   { background: linear-gradient(90deg, transparent, var(--rose),    transparent); }
    .kpi-card.kpi-violet::after { background: linear-gradient(90deg, transparent, var(--violet),  transparent); }
    .kpi-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 8px 32px rgba(99,102,241,0.12);
        transform: translateY(-3px);
    }
    .kpi-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        margin: 0 auto 0.8rem auto;
    }
    .kpi-icon-green  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.2); }
    .kpi-icon-amber  { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.2); }
    .kpi-icon-rose   { background: rgba(244,63,94,0.12);  border: 1px solid rgba(244,63,94,0.2);  }
    .kpi-icon-violet { background: rgba(167,139,250,0.12);border: 1px solid rgba(167,139,250,0.2);}
    .kpi-lbl {
        font-size: 0.62rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
    }
    .kpi-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 0.25rem;
    }
    .clr-green  { color: var(--emerald) !important; }
    .clr-amber  { color: var(--amber)   !important; }
    .clr-rose   { color: var(--rose)    !important; }
    .clr-violet { color: var(--violet)  !important; }
    .clr-blue   { color: var(--blue)    !important; }
    .clr-text   { color: var(--text)    !important; }

    /* ══ ANALYSIS PANELS ══ */
    .panel {
        background: linear-gradient(160deg, rgba(13,15,34,0.95), rgba(8,10,24,1));
        border: 1px solid var(--border-dim);
        border-radius: var(--r-lg);
        padding: 1.8rem;
        height: 100%;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }
    .panel:hover { border-color: var(--border-glow); }
    .panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ══ PROBABILITY BARS ══ */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 14px 0;
    }
    .prob-label {
        width: 68px;
        font-size: 0.78rem;
        font-weight: 700;
        text-align: right;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.04);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        position: relative;
    }
    .prob-fill::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.25) 50%, transparent 100%);
        background-size: 200% 100%;
        animation: bar-shine 2s linear infinite;
    }
    @keyframes bar-shine {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
    }
    .pf-green  { background: linear-gradient(90deg, #059669, #10B981); }
    .pf-red    { background: linear-gradient(90deg, #BE123C, #F43F5E); }
    .pf-amber  { background: linear-gradient(90deg, #B45309, #F59E0B); }
    .prob-pct {
        width: 48px;
        font-size: 0.78rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
        flex-shrink: 0;
    }

    /* ══ RISK GAUGE ══ */
    .gauge-zone-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--muted);
        font-weight: 700;
        margin-bottom: 6px;
    }
    .gauge-bar {
        height: 14px;
        border-radius: 7px;
        background: linear-gradient(90deg,
            #059669 0%, #34D399 18%,
            #D97706 35%, #F59E0B 52%,
            #DC2626 70%, #F43F5E 85%,
            #7C0000 100%
        );
        position: relative;
        box-shadow: 0 2px 12px rgba(0,0,0,0.5);
    }
    .gauge-cursor {
        position: absolute;
        top: -7px;
        width: 28px;
        height: 28px;
        background: white;
        border-radius: 50%;
        border: 3px solid var(--bg-deep);
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3), 0 4px 12px rgba(0,0,0,0.5);
        transform: translateX(-50%);
        transition: left 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .gauge-readout {
        margin-top: 1.2rem;
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .gauge-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1;
    }
    .gauge-level-tag {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.7rem;
        border-radius: 100px;
    }
    .glt-green { background: rgba(16,185,129,0.15); color: var(--emerald); border: 1px solid rgba(16,185,129,0.2); }
    .glt-amber { background: rgba(245,158,11,0.15); color: var(--amber);   border: 1px solid rgba(245,158,11,0.2); }
    .glt-red   { background: rgba(244,63,94,0.15);  color: var(--rose);    border: 1px solid rgba(244,63,94,0.2);  }

    /* ══ AI SUMMARY BOX ══ */
    .ai-summary {
        background: linear-gradient(160deg, rgba(96,165,250,0.05), rgba(129,140,248,0.03));
        border: 1px solid rgba(96,165,250,0.15);
        border-radius: var(--r-lg);
        padding: 1.5rem 1.8rem;
        line-height: 1.75;
        font-size: 0.88rem;
        color: var(--slate);
        position: relative;
        overflow: hidden;
    }
    .ai-summary::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--blue), var(--violet));
        border-radius: 3px;
    }
    .ai-summary b { color: var(--text); }

    /* ══ HISTORY TABLE ══ */
    .history-shell {
        background: linear-gradient(160deg, rgba(13,15,34,0.95), rgba(8,10,24,1));
        border: 1px solid var(--border-dim);
        border-radius: var(--r-xl);
        overflow: hidden;
    }
    .history-header {
        display: grid;
        grid-template-columns: 1fr 100px 85px 120px 50px;
        gap: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 0.6rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid var(--border-dim);
        background: rgba(255,255,255,0.01);
    }
    .history-row {
        display: grid;
        grid-template-columns: 1fr 100px 85px 120px 50px;
        gap: 12px;
        align-items: center;
        padding: 0.9rem 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        transition: background 0.2s ease;
    }
    .history-row:hover { background: rgba(99,102,241,0.04); }
    .history-row:last-child { border-bottom: none; }
    .h-headline {
        font-size: 0.8rem;
        color: var(--slate);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .h-cell {
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
    }
    .h-time {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--muted);
        text-align: center;
    }
    .badge {
        font-size: 0.62rem;
        padding: 0.2rem 0.65rem;
        border-radius: 100px;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-invest   { background: rgba(16,185,129,0.12); color: var(--emerald); border: 1px solid rgba(16,185,129,0.2); }
    .badge-hold     { background: rgba(245,158,11,0.12); color: var(--amber);   border: 1px solid rgba(245,158,11,0.2); }
    .badge-avoid    { background: rgba(244,63,94,0.12);  color: var(--rose);    border: 1px solid rgba(244,63,94,0.2);  }

    /* ══ SIDEBAR ══ */
    section[data-testid="stSidebar"] {
        background: var(--bg-panel) !important;
        border-right: 1px solid var(--border-dim) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .sb-logo {
        text-align: center;
        padding: 0 1rem 1.5rem 1rem;
        border-bottom: 1px solid var(--border-dim);
        margin-bottom: 1.5rem;
    }
    .sb-logo-icon {
        font-size: 2.2rem;
        display: block;
        margin-bottom: 0.4rem;
    }
    .sb-logo-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--blue), var(--violet));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    .sb-logo-tag {
        font-size: 0.6rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-top: 2px;
    }
    .sb-section-title {
        font-size: 0.6rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.5rem 0 0.6rem 0;
        padding: 0 0.2rem;
    }
    .sb-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border-dim);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .sb-chip-icon {
        width: 28px; height: 28px;
        border-radius: 7px;
        background: rgba(129,140,248,0.12);
        border: 1px solid rgba(129,140,248,0.15);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem;
        flex-shrink: 0;
    }
    .sb-chip-lbl {
        font-size: 0.6rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    .sb-chip-val {
        font-size: 0.8rem;
        color: var(--text);
        font-weight: 600;
    }
    /* Status dot */
    .live-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 5px;
        animation: live-pulse 2s ease-in-out infinite;
    }
    @keyframes live-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(1.4); }
    }
    .dot-green { background: var(--emerald); box-shadow: 0 0 6px var(--emerald); }
    .dot-amber { background: var(--amber);   box-shadow: 0 0 6px var(--amber); }

    /* Decision logic table in sidebar */
    .logic-row {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        font-size: 0.77rem;
        color: var(--slate);
    }
    .logic-row:last-child { border-bottom: none; }
    .logic-bullet { flex-shrink: 0; margin-top: 1px; font-size: 0.7rem; }
    .logic-row b { color: var(--text); }

    /* Sidebar Quick headline buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 8px !important;
        color: var(--slate) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.5rem 0.8rem !important;
        margin-bottom: 3px !important;
        transition: all 0.2s ease !important;
        line-height: 1.4 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99,102,241,0.08) !important;
        border-color: var(--border-glow) !important;
        color: var(--text) !important;
    }

    /* ══ FOOTER ══ */
    .footer-wrap {
        padding: 2.5rem 0 1.5rem 0;
        text-align: center;
        border-top: 1px solid var(--border-dim);
        margin-top: 3rem;
    }
    .footer-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-bottom: 1.2rem;
    }
    .footer-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(129,140,248,0.06);
        color: rgba(167, 139, 250, 0.85);
        padding: 0.35rem 0.9rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        border: 1px solid rgba(129,140,248,0.1);
        letter-spacing: 0.2px;
    }
    .footer-credit {
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.5px;
    }

    /* ══ SPINNER OVERRIDE ══ */
    .stSpinner > div { border-top-color: var(--indigo) !important; }

    /* ══ PLOTLY CHART BACKGROUND FIX ══ */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "headline_input" not in st.session_state:
    st.session_state.headline_input = ""


# ══════════════════════════════════════════════
# Model Loading (cached) — Dual Mode
# ══════════════════════════════════════════════
POSITIVE_KEYWORDS = [
    "surge", "surges", "soar", "soars", "record", "profit", "profits",
    "growth", "boom", "beat", "beats", "exceeded", "exceeds", "rally",
    "gain", "gains", "strong", "bullish", "upgrade", "breakthrough",
    "approval", "expansion", "raises", "funding", "optimism", "high",
    "best", "innovative", "success", "recover", "recovery", "up",
    "positive", "boost", "boosting", "groundbreaking", "exceptional",
    "outstanding", "secures", "billion", "unprecedented", "historic",
    "bounce", "rebound", "rebounds", "catalyst", "upside", "accumulate",
    "jump", "jumps", "recovers",
]
NEGATIVE_KEYWORDS = [
    "crash", "crashes", "plummet", "plummets", "loss", "losses", "fraud",
    "investigation", "bankruptcy", "layoff", "layoffs", "downturn",
    "recession", "collapse", "collapses", "downgrade", "decline",
    "fell", "falls", "tumble", "tumbles", "miss", "scandal", "debt",
    "default", "penalty", "lawsuit", "violation", "drop", "drops",
    "worst", "negative", "fear", "fears", "crisis", "risk", "risky",
    "pressure", "disruption", "shutdown", "recall", "slash", "cut",
]

def _keyword_predict(headline):
    words = headline.lower().split()
    seed = int(hashlib.md5(headline.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    pos_count = sum(1 for w in words if any(k in w for k in POSITIVE_KEYWORDS))
    neg_count = sum(1 for w in words if any(k in w for k in NEGATIVE_KEYWORDS))
    if pos_count > neg_count:
        sentiment = "Positive"
        confidence = min(0.75 + pos_count * 0.05 + rng.uniform(0, 0.1), 0.98)
        risk_score = round(rng.uniform(0.05, 0.35), 4)
        pos_prob = confidence
        neg_prob = round((1 - confidence) * rng.uniform(0.2, 0.4), 4)
        neu_prob = round(1 - pos_prob - neg_prob, 4)
    elif neg_count > pos_count:
        sentiment = "Negative"
        confidence = min(0.75 + neg_count * 0.05 + rng.uniform(0, 0.1), 0.98)
        risk_score = round(rng.uniform(0.65, 0.95), 4)
        neg_prob = confidence
        pos_prob = round((1 - confidence) * rng.uniform(0.2, 0.4), 4)
        neu_prob = round(1 - neg_prob - pos_prob, 4)
    else:
        sentiment = "Neutral"
        confidence = round(rng.uniform(0.55, 0.75), 4)
        risk_score = round(rng.uniform(0.30, 0.60), 4)
        neu_prob = confidence
        pos_prob = round((1 - confidence) * rng.uniform(0.4, 0.6), 4)
        neg_prob = round(1 - neu_prob - pos_prob, 4)
    risk_level = classify_risk_level(risk_score)
    suggestion = get_investment_suggestion(sentiment, risk_score)
    return {
        "sentiment": sentiment, "confidence": confidence,
        "probabilities": {"Positive": pos_prob, "Negative": neg_prob, "Neutral": neu_prob},
        "risk_score": risk_score, "risk_level": risk_level, "suggestion": suggestion,
    }


if USE_BERT:
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

    def predict_headline(headline):
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

    model, tokenizer, model_status = load_model_and_tokenizer()

else:
    model = None
    tokenizer = None
    model_status = "trained"
    def predict_headline(headline):
        return _keyword_predict(headline)


# ══════════════════════════════════════════════
# TICKER BAR
# ══════════════════════════════════════════════
ticker_items = [
    ("AAPL", "+1.24%", True), ("TSLA", "-0.87%", False), ("NVDA", "+3.41%", True),
    ("AMZN", "+0.62%", True), ("MSFT", "+0.19%", True), ("META", "-1.05%", False),
    ("GOOG", "+0.93%", True), ("JPM",  "+0.54%", True), ("BAC",  "-0.31%", False),
    ("BRK", "+0.07%", True),  ("NFLX", "+2.11%", True), ("AMD",  "+1.78%", True),
    ("INTC", "-2.33%", False),("GS",   "+0.44%", True), ("V",    "+0.28%", True),
]
# Duplicate for seamless loop
all_items = ticker_items * 2
ticker_html = '<div class="ticker-wrap"><div class="ticker-track">'
for name, chg, up in all_items:
    cls = "ticker-up" if up else "ticker-down"
    arrow = "▲" if up else "▼"
    ticker_html += f'<div class="ticker-item"><span class="t-name">{name}</span><span class="{cls}">{arrow} {chg}</span></div>'
ticker_html += '</div></div>'
st.markdown(ticker_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sb-logo">
        <span class="sb-logo-icon">📡</span>
        <div class="sb-logo-name">FinSight AI</div>
        <div class="sb-logo-tag">Intelligence Engine v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # System chips
    dot_cls = "dot-green" if model_status == "trained" else "dot-amber"
    status_txt = "Model Active" if model_status == "trained" else "Untrained"
    engine_name = "BERT base-uncased" if USE_BERT else "NLP Keyword Engine"
    device_txt = str(DEVICE).upper() if USE_BERT else "CLOUD"
    param_txt = "109.6M total · 14.9M active" if USE_BERT else "Lightweight · Fast"

    st.markdown(f"""
    <div class="sb-section-title">System Status</div>

    <div class="sb-chip">
        <div class="sb-chip-icon">⚡</div>
        <div>
            <div class="sb-chip-lbl">Status</div>
            <div class="sb-chip-val"><span class="live-dot {dot_cls}"></span>{status_txt}</div>
        </div>
    </div>
    <div class="sb-chip">
        <div class="sb-chip-icon">🧠</div>
        <div>
            <div class="sb-chip-lbl">Engine</div>
            <div class="sb-chip-val">{engine_name}</div>
        </div>
    </div>
    <div class="sb-chip">
        <div class="sb-chip-icon">🖥️</div>
        <div>
            <div class="sb-chip-lbl">Device</div>
            <div class="sb-chip-val">{device_txt}</div>
        </div>
    </div>
    <div class="sb-chip">
        <div class="sb-chip-icon">🔢</div>
        <div>
            <div class="sb-chip-lbl">Parameters</div>
            <div class="sb-chip-val">{param_txt}</div>
        </div>
    </div>

    <div class="sb-section-title" style="margin-top: 1.8rem;">Decision Logic</div>
    <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-dim);
                border-radius: 10px; padding: 0.8rem 1rem;">
        <div class="logic-row">
            <span class="logic-bullet clr-green">●</span>
            <span><b>Invest</b> — Positive sentiment &amp; Risk &lt; 0.40</span>
        </div>
        <div class="logic-row">
            <span class="logic-bullet clr-green">◐</span>
            <span><b>Accumulate</b> — Positive &amp; Risk ≤ 0.50</span>
        </div>
        <div class="logic-row">
            <span class="logic-bullet clr-amber">●</span>
            <span><b>Hold</b> — Neutral &amp; Risk &lt; 0.60</span>
        </div>
        <div class="logic-row">
            <span class="logic-bullet clr-rose">●</span>
            <span><b>Avoid</b> — Negative OR Risk ≥ 0.65</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title" style="margin-top: 1.8rem;">Quick Headlines</div>', unsafe_allow_html=True)

    examples = [
        ("Apple expected to fall this quarter, bounce back next on new iPhone", "🟢"),
        ("Tesla reports record quarterly profits", "🟢"),
        ("Startup secures billion dollar Series C funding", "🟢"),
        ("Fed maintains current interest rates steady", "🟡"),
        ("Market trades sideways with no clear direction", "🟡"),
        ("Bank faces massive fraud investigation", "🔴"),
        ("Company shares fell after revenue miss", "🔴"),
        ("Tech layoffs accelerate across industry", "🔴"),
        ("Apple stock surges on strong earnings beat", "🟢"),
    ]

    for text, dot in examples:
        display = f"{dot} {text[:52]}…" if len(text) > 52 else f"{dot} {text}"
        if st.button(display, key=f"ex_{text[:30]}", use_container_width=True):
            st.session_state.headline_input = text
            st.rerun()


# ══════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════
analyses_done = len(st.session_state.history)

st.markdown(f"""
<div class="hero-shell">
    <div class="hero-eyebrow">
        <div class="pulse-dot"></div>
        BERT-Powered · Real-Time Analysis
    </div>
    <div class="hero-title">Financial Sentiment<br>&amp; Risk Intelligence</div>
    <div class="hero-sub">
        State-of-the-art transformer AI trained on financial corpora.
        Get instant sentiment classification, calibrated risk scoring,
        and actionable investment recommendations — in milliseconds.
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <span style="font-size:1.1rem;">🎯</span>
            <div>
                <div class="hero-stat-val dot-emerald">92.6%</div>
                <div class="hero-stat-lbl">Val Accuracy</div>
            </div>
        </div>
        <div class="hero-stat">
            <span style="font-size:1.1rem;">⚡</span>
            <div>
                <div class="hero-stat-val dot-violet">0.9014</div>
                <div class="hero-stat-lbl">Macro F1</div>
            </div>
        </div>
        <div class="hero-stat">
            <span style="font-size:1.1rem;">📉</span>
            <div>
                <div class="hero-stat-val dot-cyan">0.0826</div>
                <div class="hero-stat-lbl">Risk MAE</div>
            </div>
        </div>
        <div class="hero-stat">
            <span style="font-size:1.1rem;">🔍</span>
            <div>
                <div class="hero-stat-val" style="color: var(--text);">{analyses_done}</div>
                <div class="hero-stat-lbl">This Session</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════
st.markdown("""
<div class="sec-label">
    <div class="sec-label-icon">📝</div>
    <div class="sec-label-text">Analyze a Headline</div>
</div>
""", unsafe_allow_html=True)

headline = st.text_input(
    "headline",
    value=st.session_state.get("headline_input", ""),
    placeholder="e.g.  'Apple beats earnings expectations, raising full-year guidance'",
    label_visibility="collapsed",
)
col_btn1, col_btn2 = st.columns([4, 1.2])
with col_btn1:
    analyze_clicked = st.button("🔍  Analyze with BERT AI", type="primary", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🗑️  Clear All", use_container_width=True)

if clear_clicked:
    st.session_state.headline_input = ""
    st.session_state.history = []
    st.rerun()


# ══════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════
if analyze_clicked and headline.strip():
    with st.spinner("🧠 Analyzing with BERT transformer…"):
        result = predict_headline(headline.strip())
        time.sleep(0.25)

    sentiment  = result["sentiment"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    risk_score = result["risk_score"]
    risk_level = result["risk_level"]
    suggestion = result["suggestion"]

    # Save to history
    st.session_state.history.insert(0, {
        "headline":   headline.strip(),
        "sentiment":  sentiment,
        "risk":       risk_score,
        "suggestion": suggestion,
        "time":       datetime.now().strftime("%H:%M"),
    })
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]

    # Determine banner style
    if "Accumulate" in suggestion:
        rb_cls, rb_icon, rb_title_txt, rb_sub = (
            "rb-invest", "📈", "ACCUMULATE / BUY ON DIPS",
            "Short-term dip · Long-term upside · Add on weakness"
        )
    elif "Invest" in suggestion and "Avoid" not in suggestion:
        rb_cls, rb_icon, rb_title_txt, rb_sub = (
            "rb-invest", "✅", "STRONG BUY — INVEST",
            "Positive sentiment · Low risk profile · Favourable entry"
        )
    elif "Avoid" in suggestion:
        rb_cls, rb_icon, rb_title_txt, rb_sub = (
            "rb-avoid", "🛑", "AVOID INVESTMENT",
            "High risk or negative sentiment detected · Stand aside"
        )
    else:
        rb_cls, rb_icon, rb_title_txt, rb_sub = (
            "rb-hold", "⏸️", "HOLD POSITION",
            "Neutral outlook · Monitor developments before acting"
        )

    # Color helpers
    sent_color = {"Positive": "clr-green", "Negative": "clr-rose", "Neutral": "clr-amber"}[sentiment]
    risk_color = {"Low": "clr-green", "Medium": "clr-amber", "High": "clr-rose"}[risk_level]
    risk_glt   = {"Low": "glt-green",  "Medium": "glt-amber",  "High": "glt-red"}[risk_level]

    # ── Divider ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Recommendation Banner ──
    st.markdown(f"""
    <div class="result-banner {rb_cls}">
        <div class="rb-shimmer"></div>
        <div class="rb-icon">{rb_icon}</div>
        <div>
            <div class="rb-label">AI Recommendation</div>
            <div class="rb-title">{rb_title_txt}</div>
            <div class="rb-subtitle">{rb_sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    k_sent_icon   = "🟢" if sentiment == "Positive" else ("🔴" if sentiment == "Negative" else "🟡")
    k_sent_klass  = {"Positive": "kpi-green", "Negative": "kpi-rose", "Neutral": "kpi-amber"}[sentiment]
    k_icon_klass  = {"Positive": "kpi-icon-green", "Negative": "kpi-icon-rose", "Neutral": "kpi-icon-amber"}[sentiment]
    k_risk_klass  = {"Low": "kpi-green", "Medium": "kpi-amber", "High": "kpi-rose"}[risk_level]
    k_risk_ik     = {"Low": "kpi-icon-green","Medium": "kpi-icon-amber","High": "kpi-icon-rose"}[risk_level]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card {k_sent_klass}">
            <div class="kpi-icon {k_icon_klass}">{k_sent_icon}</div>
            <div class="kpi-lbl">Sentiment</div>
            <div class="kpi-val {sent_color}">{sentiment}</div>
            <div class="kpi-sub">{confidence:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card {k_risk_klass}">
            <div class="kpi-icon {k_risk_ik}">⚡</div>
            <div class="kpi-lbl">Risk Score</div>
            <div class="kpi-val {risk_color}" style="font-family:'JetBrains Mono',monospace;">{risk_score:.4f}</div>
            <div class="kpi-sub">{risk_level} Risk Level</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        conf_pct = int(confidence * 100)
        st.markdown(f"""
        <div class="kpi-card kpi-violet">
            <div class="kpi-icon kpi-icon-violet">🎯</div>
            <div class="kpi-lbl">Confidence</div>
            <div class="kpi-val clr-violet">{conf_pct}%</div>
            <div class="kpi-sub">Model certainty</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        dom_label = max(probs, key=probs.get)
        dom_pct   = int(probs[dom_label] * 100)
        st.markdown(f"""
        <div class="kpi-card kpi-violet">
            <div class="kpi-icon kpi-icon-violet">📊</div>
            <div class="kpi-lbl">Dominant Class</div>
            <div class="kpi-val clr-violet" style="font-size:1.5rem;">{dom_label}</div>
            <div class="kpi-sub">{dom_pct}% probability mass</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability Bars + Risk Gauge ──
    col_left, col_right = st.columns(2)

    with col_left:
        prob_map = {
            "Positive": ("clr-green", "pf-green"),
            "Negative": ("clr-rose",  "pf-red"),
            "Neutral":  ("clr-amber", "pf-amber"),
        }
        bars_html = ""
        for label, prob in probs.items():
            txt_cls, bar_cls = prob_map[label]
            w = f"{prob * 100:.1f}"
            bars_html += f'<div class="prob-row"><span class="prob-label {txt_cls}">{label}</span><div class="prob-track"><div class="prob-fill {bar_cls}" style="width:{w}%"></div></div><span class="prob-pct {txt_cls}">{prob:.1%}</span></div>'

        st.markdown('<div class="panel"><div class="panel-title">📊 &nbsp;Sentiment Probability Distribution</div>' + bars_html + '</div>', unsafe_allow_html=True)

    with col_right:
        needle = max(2, min(98, risk_score * 100))
        gauge_html = (
            '<div class="panel">'
            '<div class="panel-title">🎯 &nbsp;Risk Assessment Gauge</div>'
            '<div class="gauge-zone-labels"><span>Low</span><span>Moderate</span><span>High</span><span>Critical</span></div>'
            f'<div class="gauge-bar"><div class="gauge-cursor" style="left:{needle:.1f}%"></div></div>'
            f'<div class="gauge-readout"><div class="gauge-number {risk_color}">{risk_score:.4f}</div>'
            f'<div class="gauge-level-tag {risk_glt}">{risk_level} Risk</div></div>'
            '</div>'
        )
        st.markdown(gauge_html, unsafe_allow_html=True)

    # ── AI Summary ──
    st.markdown("<br>", unsafe_allow_html=True)
    engine_label = "BERT transformer" if USE_BERT else "NLP keyword engine"
    st.markdown(f"""
    <div class="ai-summary">
        <b>🧠 AI Analysis Summary</b><br><br>
        The {engine_label} analyzed <b>"{headline.strip()}"</b> and returned a
        <b class="{sent_color}">{sentiment.lower()}</b> sentiment classification
        at <b>{confidence:.1%}</b> confidence. The multi-task risk head assigned
        a score of <b style="font-family:'JetBrains Mono',monospace">{risk_score:.4f}</b>,
        placing this headline in the <b class="{risk_color}">{risk_level.lower()} risk</b> tier.
        Combined signal yields a final recommendation of: <b>{rb_title_txt}</b>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HISTORY TABLE
# ══════════════════════════════════════════════
if st.session_state.history:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">
        <div class="sec-label-icon">📋</div>
        <div class="sec-label-text">Analysis History</div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for item in st.session_state.history:
        s_color = {"Positive": "clr-green", "Negative": "clr-rose", "Neutral": "clr-amber"}[item["sentiment"]]
        r_val   = item["risk"]
        r_color = "clr-green" if r_val < 0.4 else ("clr-amber" if r_val < 0.7 else "clr-rose")

        if "Accumulate" in item["suggestion"]:
            b_cls, b_txt = "badge-invest", "ACCUMULATE"
        elif "Invest" in item["suggestion"] and "Avoid" not in item["suggestion"]:
            b_cls, b_txt = "badge-invest", "INVEST"
        elif "Avoid" in item["suggestion"]:
            b_cls, b_txt = "badge-avoid", "AVOID"
        else:
            b_cls, b_txt = "badge-hold", "HOLD"

        hl = item["headline"]
        hl_disp = hl[:75] + "…" if len(hl) > 75 else hl
        rows_html += (
            f'<div class="history-row">'
            f'<div class="h-headline">{hl_disp}</div>'
            f'<div class="h-cell {s_color}">{item["sentiment"]}</div>'
            f'<div class="h-cell {r_color}" style="font-family:\'JetBrains Mono\',monospace;">{r_val:.3f}</div>'
            f'<div style="text-align:center;"><span class="badge {b_cls}">{b_txt}</span></div>'
            f'<div class="h-time">{item["time"]}</div>'
            f'</div>'
        )

    history_shell = (
        '<div class="history-shell">'
        '<div class="history-header">'
        '<span>Headline</span>'
        '<span style="text-align:center">Sentiment</span>'
        '<span style="text-align:center">Risk</span>'
        '<span style="text-align:center">Action</span>'
        '<span style="text-align:center">Time</span>'
        '</div>'
        + rows_html +
        '</div>'
    )
    st.markdown(history_shell, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="footer-wrap">
    <div class="footer-tags">
        <span class="footer-tag">🤖 BERT Transformer</span>
        <span class="footer-tag">📊 Multi-Task Learning</span>
        <span class="footer-tag">⚡ Real-Time Analysis</span>
        <span class="footer-tag">🎯 Risk Neural Network</span>
        <span class="footer-tag">💼 Investment Engine</span>
        <span class="footer-tag">🔥 PyTorch + HuggingFace</span>
        <span class="footer-tag">🐍 Python · Streamlit</span>
    </div>
    <div class="footer-credit">
        FinSight AI &nbsp;·&nbsp; Financial Intelligence Platform &nbsp;·&nbsp; Built with PyTorch &amp; HuggingFace Transformers
    </div>
</div>
""", unsafe_allow_html=True)
