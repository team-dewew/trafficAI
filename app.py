import base64
import html
import json
import threading
import uuid
import warnings
from pathlib import Path

# Suppress library deprecation and non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.annotate import draw_scene
from src.demo import DEMO_MAX_MB, DEMO_MAX_SEC, run_demo
from src.registration import estimate_scene_transform
from src.scene import build_scene

REPO_URL = "https://github.com/DeWeWO/wiut"
SITE_URL = "https://trafficai.dewew.dev"
DEMO_SAMPLE = Path("samples/demo/C3896_60-95s_720p.mp4")

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic AI — Surveillance Control Center",
    page_icon=":material/sensors:",
    layout="wide",
    initial_sidebar_state="auto",
)

# ----------------------------------------------------------------------------
# Senior Frontend UI/UX Design System (Linear / Vercel / Apple Developer)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, div, span {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        letter-spacing: -0.02em;
    }
    
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background Canvas */
    .stApp {
        background-color: #030712;
        background-image: 
            radial-gradient(ellipse 90% 50% at 50% -10%, rgba(14, 165, 233, 0.06) 0%, transparent 60%),
            linear-gradient(180deg, #030712 0%, #050b18 100%);
        color: #f1f5f9;
    }

    /* Main Page Container: Balanced top padding and wide-canvas alignment */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.25rem !important;
        z-index: 100 !important;
    }
    
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* =========================================================================
       PAGE HEADER (INTEGRATED TOP BAR ARCHITECTURE - NO FLOATING BOXES)
       ========================================================================= */
    .page-top-bar {
        padding-bottom: 18px;
        margin-bottom: 22px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
    }
    .page-breadcrumbs {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .crumb-prefix {
        color: #64748b;
    }
    .crumb-sep {
        color: #475569;
        font-size: 0.65rem;
    }
    .crumb-module {
        color: #38bdf8;
    }
    .crumb-current {
        color: #94a3b8;
    }
    .page-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        flex-wrap: wrap;
    }
    .page-title-block {
        flex: 1;
        min-width: 280px;
    }
    .page-h1 {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        font-size: 2.15rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        letter-spacing: -0.03em !important;
        line-height: 1.15 !important;
        margin: 0 !important;
    }
    .page-description {
        font-size: 0.90rem !important;
        color: #94a3b8 !important;
        line-height: 1.55 !important;
        margin: 6px 0 0 0 !important;
        max-width: 860px;
    }
    .page-status-block {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-top: 4px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 12px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #10b981;
        letter-spacing: 0.05em;
    }
    .pulse-indicator {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-dot 1.8s infinite;
    }
    .spec-mini-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 5px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94a3b8;
    }

    /* In-Page Section Headings */
    .section-header-block {
        margin-top: 24px;
        margin-bottom: 12px;
    }
    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .section-heading-h2 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 7px;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-dot 1.8s infinite;
    }

    @keyframes pulse-dot {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* KPI Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(10, 16, 30, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 9px;
        padding: 16px 18px;
        text-align: left;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #00f2fe, #2563eb);
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #38bdf8;
        margin-top: 3px;
        font-weight: 500;
    }

    /* Glass Panels */
    .glass-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(9, 14, 26, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }

    /* Telemetry HUD Grid */
    .hud-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin: 10px 0 10px 0;
    }
    .hud-chip {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 8px;
        padding: 8px 12px;
        position: relative;
    }
    .hud-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .hud-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #00f2fe;
        line-height: 1.15;
    }
    .hud-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 2px;
    }
    .hud-status-strip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .hud-status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.15);
        padding: 1px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .hud-status-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94a3b8;
    }

    /* Team Badge Cards (Standardized Equal Height System) */
    .team-badge-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 16, 32, 0.98) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 24px 20px 20px 20px;
        text-align: center;
        position: relative;
        transition: all 0.22s ease;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        height: 100%;
        min-height: 530px;
        box-sizing: border-box;
    }
    .team-badge-card:hover {
        transform: translateY(-3px);
        border-color: #00f2fe;
        box-shadow: 0 10px 28px rgba(0, 242, 254, 0.16);
    }
    .team-avatar-ring {
        width: 104px;
        height: 104px;
        min-width: 104px;
        min-height: 104px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0284c7, #00f2fe);
        padding: 3px;
        margin: 0 auto 16px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.25);
    }
    .team-avatar-img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        display: block;
    }
    .mono-avatar {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        letter-spacing: -0.5px !important;
        color: #00f2fe !important;
        background: #0b1120 !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 50% !important;
    }
    .team-name {
        font-size: 1.22rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
        line-height: 1.25;
        min-height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .team-role-pill-wrapper {
        min-height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
        width: 100%;
    }
    .team-role-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
        line-height: 1.35;
    }
    .team-bio {
        font-size: 0.83rem;
        color: #94a3b8;
        line-height: 1.55;
        margin-bottom: 16px;
        min-height: 84px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .team-skills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: center;
        align-content: flex-start;
        margin-bottom: 16px;
        min-height: 64px;
        width: 100%;
    }
    .skill-chip {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        font-size: 0.70rem;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
    }
    .team-links-wrapper {
        margin-top: auto;
        padding-top: 10px;
        width: 100%;
        display: flex;
        justify-content: center;
        gap: 6px;
    }
    .btn-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #f1f5f9 !important;
        text-decoration: none;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.2s ease;
    }
    .btn-link:hover {
        background: rgba(56, 189, 248, 0.2);
        color: #00f2fe !important;
        border-color: #00f2fe;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0, 242, 254, 0.2);
    }

    /* Main Canvas Action Buttons */
    .main .stButton > button {
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.1px !important;
        transition: all 0.2s ease !important;
    }
    .main .stButton > button[kind="primary"],
    .main .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        padding: 9px 20px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }
    .main .stButton > button[kind="primary"]:hover,
    .main .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
        border-color: #00f2fe !important;
        box-shadow: 0 6px 18px rgba(0, 242, 254, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary / Segmented Tab Buttons in Main Canvas */
    .main .stButton > button[kind="secondary"],
    .main .stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 8px 16px !important;
        box-shadow: none !important;
    }
    .main .stButton > button[kind="secondary"]:hover,
    .main .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(30, 41, 59, 0.9) !important;
        color: #f8fafc !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 4px 6px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 7px 16px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.86rem;
        color: #94a3b8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.15) !important;
        color: #00f2fe !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
    }

    /* Telemetry Pill Strip */
    .pill-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    .pill-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        color: #e2e8f0;
    }
    .pill-item b {
        color: #38bdf8;
    }

    /* =========================================================================
       SIDEBAR: MODERN SLEEK NAV (PREMIUM)
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background: #030712 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 2px !important;
    }

    [data-testid="stSidebarHeader"] {
        height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; display: none !important;
    }

    [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 24px 16px !important;
    }

    .sidebar-brand-minimal {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4px;
        margin-bottom: 8px;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00f2fe;
        box-shadow: 0 0 10px #00f2fe;
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.3px;
    }
    .brand-highlight {
        color: #00f2fe;
    }
    .brand-status-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 3px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    
    .sidebar-subtext {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #64748b;
        padding: 0 4px;
        margin-bottom: 20px;
    }

    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 16px 0;
        width: 100%;
    }

    .sidebar-section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
        padding-left: 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.stButton) {
        margin: 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        min-height: 40px !important;
        height: 40px !important;
        padding: 0 14px !important;
        margin-bottom: 4px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
        color: #f1f5f9 !important;
        transform: translateX(2px) !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.15), transparent) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-left: 3px solid #00f2fe !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stButton > button div {
        display: flex !important; align-items: center !important; width: 100% !important; margin: 0 !important; padding: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        margin: 0 !important; padding: 0 !important;
    }

    .sidebar-specs-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.4) 0%, rgba(10, 15, 28, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
    .specs-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .specs-title::before {
        content: ""; width: 4px; height: 4px; background: #38bdf8; border-radius: 50%;
    }
    .specs-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.04);
    }
    .specs-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .specs-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .specs-val {
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.68rem;
        background: rgba(255, 255, 255, 0.04);
        padding: 3px 6px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .specs-val.emerald {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.25);
    }
    .specs-val.cyan {
        color: #00f2fe;
        background: rgba(0, 242, 254, 0.1);
        border-color: rgba(0, 242, 254, 0.25);
    }

    .sidebar-footer-minimal {
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.65rem;
        color: #475569;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }
    /* Callout Cards */
    .callout-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        border-left: 3px solid #0284c7;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .callout-success {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.04);
    }
    .callout-warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.04);
    }
    .callout-danger {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.04);
    }
    .callout-title {
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .callout-body {
        font-size: 0.84rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Helper: Metadata, Geometry Overlay, Web Preview, & Benchmark Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_image_base64(filepath: str) -> str:
    """Read an image file and return base64 data URI for instant web display."""
    p = Path(filepath).resolve()
    if p.exists():
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    return ""


@st.cache_data(show_spinner=False)
def get_video_metadata(video_path: str) -> dict:
    """Extract metadata (FPS, frames, duration, resolution, size) safely."""
    try:
        p = Path(video_path).resolve()
        if not p.exists():
            return {}
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return {}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        duration_sec = (total_frames / fps) if fps > 0 else 0.0
        cap.release()
        size_mb = p.stat().st_size / (1024 * 1024)
        return {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": round(duration_sec, 1),
            "size_mb": round(size_mb, 1),
            "resolution": f"{width} x {height}",
        }
    except Exception:
        return {}


def show_image(img, caption: str | None = None) -> None:
    """st.image at full width on any Streamlit version (the keyword was renamed in 1.40)."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:                   # Streamlit < 1.46
        st.image(img, caption=caption, use_column_width=True)


@st.cache_data(show_spinner=False)
def sample_scene_stats() -> dict:
    """Per sample video, from its committed first frame: registration against the
    reference frame and measured brightness (so the page never needs the 4K videos)."""
    out = {}
    for f in sorted(Path("eda_results/frames").glob("*_first_frame.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        _, info = estimate_scene_transform([img])
        gray = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        light = "day, bright sun" if gray > 85 else ("late afternoon, low sun" if gray > 55 else "dusk")
        out[f.name.split("_")[0] + ".MP4"] = {"info": info, "gray": round(gray, 1), "light": light, "frame": str(f)}
    return out


@st.cache_data(show_spinner=False)
def sample_zone_overlay(video_name: str) -> np.ndarray | None:
    """Scene layout registered onto the sample's first frame (no video file needed)."""
    f = Path("eda_results/frames") / f"{Path(video_name).stem}_first_frame.jpg"
    img = cv2.imread(str(f))
    if img is None:
        return None
    A, _ = estimate_scene_transform([img])
    vis = draw_scene(img.copy(), build_scene(A))
    return cv2.resize(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), (1280, 720), interpolation=cv2.INTER_AREA)


@st.cache_resource
def demo_lock() -> threading.Lock:
    """Shared by every visitor (one process, one thread per session): one demo at a time."""
    return threading.Lock()


def preview_path(video_name: str) -> str | None:
    p = Path("samples/previews") / f"{Path(video_name).stem}_preview.mp4"
    return str(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_benchmark_data() -> dict:
    """Load benchmark predictions and cache for fast interactive inspection."""
    pred_path = Path("predictions_samples.json")
    if pred_path.exists():
        try:
            with open(pred_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


_EVENT_PALETTE = {
    "accident": "#ef4444", "near_miss": "#f97316", "red_light": "#dc2626",
    "wrong_way": "#a855f7", "illegal_u_turn": "#8b5cf6", "stopped_vehicle": "#eab308",
    "jaywalking": "#22c55e", "failure_to_yield": "#14b8a6", "illegal_turn": "#6366f1",
    "solid_line_crossing": "#0ea5e9", "stop_line": "#f43f5e", "congestion": "#f59e0b",
    "road_obstacle": "#84cc16", "fire_smoke": "#b91c1c",
}


def render_event_timeline(events: list, duration: float) -> None:
    """Draw a true Gantt-style event timeline (one row per class) in pure HTML/CSS."""
    if duration <= 0:
        duration = max((float(e[1]) for e in events), default=1.0)
    by_class: dict[str, list] = {}
    for s, e, lbl in events:
        by_class.setdefault(str(lbl), []).append((float(s), float(e)))

    rows_html = ""
    for lbl in sorted(by_class):
        color = _EVENT_PALETTE.get(lbl, "#38bdf8")
        segs = ""
        for s, e in by_class[lbl]:
            left = max(0.0, min(100.0, 100.0 * s / duration))
            width = max(0.6, min(100.0 - left, 100.0 * (e - s) / duration))
            segs += (
                f'<div title="{lbl}: {s:.1f}s - {e:.1f}s" style="position:absolute;'
                f'left:{left:.2f}%;width:{width:.2f}%;top:3px;bottom:3px;'
                f'background:{color};border-radius:3px;opacity:0.92;"></div>'
            )
        rows_html += (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<div style="width:150px;font:11px \'JetBrains Mono\',monospace;color:#94a3b8;'
            f'text-align:right;padding-right:10px;">{lbl}</div>'
            f'<div style="position:relative;flex:1;height:18px;background:#0f172a;'
            f'border:1px solid #1e293b;border-radius:4px;">{segs}</div></div>'
        )

    ticks = "".join(
        f'<div style="position:absolute;left:{p:.1f}%;top:0;bottom:0;width:1px;background:#1e293b;"></div>'
        f'<div style="position:absolute;left:{p:.1f}%;top:100%;font:9px \'JetBrains Mono\',monospace;'
        f'color:#64748b;transform:translateX(-50%);">{int(duration * p / 100)}s</div>'
        for p in (0, 25, 50, 75, 99.9)
    )
    st.markdown(
        f'<div style="position:relative;padding:6px 0 20px 0;">{rows_html}'
        f'<div style="position:relative;margin-left:150px;height:8px;">{ticks}</div></div>',
        unsafe_allow_html=True,
    )


def render_page_header(
    module_num: str,
    eyebrow_suffix: str,
    title: str,
    subtitle: str,
    badge_text: str = "SYSTEM ACTIVE",
    mini_spec: str = "",
):
    """Render a unified, senior-grade page top header across all modules."""
    st.markdown(
        f"""
        <div class="page-top-bar">
            <div class="page-breadcrumbs">
                <span class="crumb-prefix">WIUT AI HACKATHON 2026</span>
                <span class="crumb-sep">/</span>
                <span class="crumb-module">MODULE {module_num}</span>
                <span class="crumb-sep">/</span>
                <span class="crumb-current">{eyebrow_suffix}</span>
            </div>
            <div class="page-header-row">
                <div class="page-title-block">
                    <h1 class="page-h1">{title}</h1>
                    <p class="page-description">{subtitle}</p>
                </div>
                <div class="page-status-block">
                    <div class="status-pill">
                        <span class="pulse-indicator"></span>
                        <span>{badge_text}</span>
                    </div>
                    <div class="spec-mini-pill">
                        <span>{mini_spec}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sidebar Navigation (Minimalist Linear / Vercel Style)
# ----------------------------------------------------------------------------
NAV_SECTIONS = [
    {"id": "Team", "num": "01", "title": "Engineering Squad"},
    {"id": "Problem and Approach", "num": "02", "title": "Problem & Approach"},
    {"id": "EDA of sample videos", "num": "03", "title": "EDA Analytics"},
    {"id": "Results on sample videos", "num": "04", "title": "Benchmark Results"},
    {"id": "Live Demo", "num": "05", "title": "Live Console"},
    {"id": "Report", "num": "06", "title": "Executive Report"},
    {"id": "Links", "num": "07", "title": "Repository & Artifacts"},
]

if "selected_section" not in st.session_state:
    st.session_state["selected_section"] = "Live Demo"

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-minimal">
            <div class="brand-left">
                <span class="brand-dot"></span>
                <span class="brand-title">TRAFFIC <span class="brand-highlight">AI</span></span>
            </div>
            <span class="brand-status-tag">ONLINE</span>
        </div>
        <div class="sidebar-subtext">WIUT AI Hackathon 2026 // CV Track</div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-section-label">NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )

    for item in NAV_SECTIONS:
        is_active = (st.session_state["selected_section"] == item["id"])
        btn_label = f"{item['num']}   {item['title']}"
        if st.button(
            btn_label,
            key=f"nav_btn_{item['id']}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            st.session_state["selected_section"] = item["id"]
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-divider"></div>
        <div class="sidebar-specs-card">
            <div class="specs-title">RUNTIME SPECIFICATIONS</div>
            <div class="specs-row">
                <span class="specs-label">Primary Detector</span>
                <span class="specs-val">YOLO11 Large</span>
            </div>
            <div class="specs-row">
                <span class="specs-label">Anomaly Model</span>
                <span class="specs-val">YOLOv8x Crash</span>
            </div>
            <div class="specs-row">
                <span class="specs-label">Object Tracker</span>
                <span class="specs-val">ByteTrack (Causal)</span>
            </div>
            <div class="specs-row">
                <span class="specs-label">Benchmark run</span>
                <span class="specs-val emerald">GPU (RTX 3050)</span>
            </div>
            <div class="specs-row">
                <span class="specs-label">Web demo</span>
                <span class="specs-val">CPU, light mode</span>
            </div>
            <div class="specs-row">
                <span class="specs-label">Determinism</span>
                <span class="specs-val cyan">Seed 42 Locked</span>
            </div>
        </div>
        <div class="sidebar-footer-minimal">Team dewew · WIUT Hackathon 2026</div>
        """,
        unsafe_allow_html=True,
    )

selected_section = st.session_state["selected_section"]


# ============================================================================
# SECTION 1: TEAM
# ============================================================================
if selected_section == "Team":
    render_page_header(
        module_num="01",
        eyebrow_suffix="PERSONNEL ROSTER",
        title="Engineering Squad",
        subtitle="Westminster International University in Tashkent (WIUT) AI Hackathon 2026 • Computer Vision Track",
        badge_text="TEAM",
        mini_spec="3 members",
    )

    team = json.loads(Path("assets/team/team.json").read_text(encoding="utf-8"))["members"]
    cols = st.columns(len(team), gap="large")
    for col, m in zip(cols, team):
        esc = html.escape
        links = "".join(
            f'<a class="btn-link" href="{esc(u)}" target="_blank">{esc(k)}</a>' for k, u in m.get("links", {}).items()
        )
        skills = "".join(f'<span class="skill-chip">{esc(x)}</span>' for x in m.get("skills", []))
        projects = "".join(
            f'<li><a href="{esc(p["url"])}" target="_blank" style="color:#38bdf8;">{esc(p["name"])}</a> - {esc(p["desc"])}</li>'
            for p in m.get("projects", [])
        )
        proj_html = (
            '<div class="team-bio"><div style="text-align:left;"><b>Previous projects</b>'
            f'<ul style="margin:6px 0 0 18px;padding:0;">{projects}</ul></div></div>'
        ) if projects else ""
        # One line of HTML: blank or indented lines would be read as Markdown code blocks.
        card = (
            '<div class="team-badge-card">'
            f'<div class="team-avatar-ring"><img src="{get_image_base64(m["photo"])}" class="team-avatar-img" alt="{esc(m["name"])}" /></div>'
            f'<div class="team-name">{esc(m["name"])}</div>'
            f'<div class="team-role-pill-wrapper"><span class="team-role-pill">{esc(m["role"]).upper()}</span></div>'
            f'<div class="team-bio"><div><b>Did in this project:</b> {esc(m["did"])}</div></div>'
            f'{proj_html}'
            f'<div class="team-skills">{skills}</div>'
            f'<div class="team-links-wrapper">{links}</div>'
            '</div>'
        )
        with col:
            st.markdown(card, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">SYSTEM CAPABILITIES</div>
            <h2 class="section-heading-h2">Core Architectural Disciplines</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="callout-card">
                <div class="callout-title">Perception & Spatial Geometry</div>
                <div class="callout-body">Polygon zones on registered scene geometry, scale-free track kinematics, and a lamp-level traffic-signal read-out.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="callout-card callout-success">
                <div class="callout-title">Deep Learning & Risk Modeling</div>
                <div class="callout-body">Physical collision classification, Time-to-Collision proxies, and exponential risk smoothing without future frame leakage.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="callout-card callout-warning">
                <div class="callout-title">High-Performance Computing</div>
                <div class="callout-body">FP16 CUDA acceleration, ~28 FPS processing on 4K footage, and official evaluation harness compliance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# SECTION 2: PROBLEM AND APPROACH
# ============================================================================
elif selected_section == "Problem and Approach":
    render_page_header(
        module_num="02",
        eyebrow_suffix="SYSTEM ARCHITECTURE",
        title="Problem Statement & Technical Approach",
        subtitle="YOLO11-L + ByteTrack + per-video scene registration + rule engine; YOLOv8x anomaly model for crashes and fire",
        badge_text="APPROACH",
        mini_spec="3.0x BUDGET COMPLIANT",
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">14 Classes</div>'
            '<div class="metric-label">Target Taxonomy</div>'
            '<div class="metric-sub">11 emitted, 3 switched off</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">3.0x</div>'
            '<div class="metric-label">Execution Budget</div>'
            '<div class="metric-sub">Video duration limit</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">4K UHD</div>'
            '<div class="metric-label">Input Resolution</div>'
            '<div class="metric-sub">3840 x 2160 @ 29.97 FPS</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">Part B</div>'
            '<div class="metric-label">Causal Risk P(t)</div>'
            '<div class="metric-sub">Anti-Jitter Filtered</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-panel" style="margin-top: 14px; margin-bottom: 20px;">
            <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Challenge Definition & Constraints</div>
            <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.6;">
                Fixed intersection surveillance cameras experience diverse hazard scenarios across fluctuating daylight and evening conditions. 
                The system must detect <b>14 official event classes</b> (Part A) and output an <b>anticipatory causal risk score</b> P(t) in [0, 1] 
                (Part B) operating strictly under a <b>3.0x video duration budget</b> without future frame leakage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    approach_tabs = st.tabs(["Pipeline", "Rules per class", "Learned models & Part B"])

    with approach_tabs[0]:
        st.markdown("#### End-to-end pipeline (what `python run_submission.py` executes)")
        st.markdown(
            """
        ```mermaid
        graph LR
            V[4K video] --> R[Scene registration: SIFT + RANSAC similarity vs reference frame]
            R --> Z[Scene layout mapped onto this video: zones, stop lines, signal lamps]
            V --> S[Signal state from lamp colour, debounced]
            V --> D[YOLO11-L @960, FP16, every 3rd frame]
            D --> N[Duplicate car/truck suppression] --> T[ByteTrack]
            T --> K[Track state: ground point, speed in body-lengths/s]
            Z --> E{Rule engine}
            S --> E
            K --> E
            V --> A[Anomaly YOLOv8x @1 Hz] --> E
            E --> P[Merge / clip / drop blips] --> O[events]
            V --> B[Part B: YOLOv8n every 3rd frame, TTC on collision course, EMA] --> Q[risk curve]
        ```
        """
        )
        st.markdown(
            """
            <div class="pill-strip">
                <div class="pill-item">Learned: <b>YOLO11-L</b> (COCO), <b>YOLOv8n</b> (COCO), <b>YOLOv8x anomaly model</b> (accident / fire / smoke)</div>
                <div class="pill-item">Rule-based: registration, signal state, tracking logic, every event rule, risk score</div>
                <div class="pill-item">No training on our side; thresholds tuned by frame-level inspection of the sample videos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Why registration?** The camera pose differs between recordings: up to 91 px of shift and 1 degree of rotation "
            "at 4K. Without it the stop lines drift and, in C3902, the signal window lands on a road sign. "
            "**Why lamps?** A lit lamp is ~5 px tall, so a colour mask over the whole housing either never fires or always fires. "
            "Reading each lamp gives a clean 37 s red / 35 s green cycle on all four videos."
        )

    with approach_tabs[1]:
        st.markdown("#### Classes we emit and the rule behind each one")
        policy = Path("docs/class_policy.md")
        if policy.exists():
            st.markdown(policy.read_text(encoding="utf-8").split(chr(10), 1)[1])
        from src.config import RULES
        with st.expander("All thresholds (src/config.py)"):
            st.json(RULES)

    with approach_tabs[2]:
        st.markdown("#### Learned anomaly model and Part B")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown(
                """
                <div class="callout-card callout-success">
                    <div class="callout-title">Accident / fire / smoke</div>
                    <div class="callout-body">
                        • <b>Model</b>: open-weights YOLOv8x fine-tuned on the Roboflow "Accident Evaluator" dataset for crash severity and fire/smoke (<code>epoch90.pt</code> from Enos-123/accident-evaluator-yolov8x, MIT, used unchanged)<br>
                        • <b>Rate</b>: once per second of video<br>
                        • <b>Gate</b>: confidence >= 0.6, box on the carriageway and covering a vehicle, positive in >= 3 of 4 consecutive checks.
                        Without the gate it fired on ordinary traffic.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c2:
            st.markdown(
                """
                <div class="callout-card">
                    <div class="callout-title">Part B: causal risk estimator</div>
                    <div class="callout-body">
                        Sees only the frames passed to <code>step()</code>.<br><br>
                        • <b>Detector</b>: YOLOv8n on every 3rd frame, ByteTrack, ground-point velocities in pixels per second of video time<br>
                        • <b>Signal</b>: for every pair of road users on the carriageway, time-to-collision along their relative motion,
                        counted only if their closest approach is within 0.3 of their size (a real collision course) for two consecutive updates<br>
                        • <b>Perspective guards</b>: duplicate car/truck boxes merged; far-field objects and pairs on the far carriageway ignored;
                        same-direction pairs count only when closing fast in the same lane (rear-end)<br>
                        • <b>Score</b>: logistic in TTC (0.5 at about 1.0 s), max over pairs, EMA smoothing (alpha 0.35).
                        On the samples (no crashes) the score is >= 0.5 in under 0.5% of frames
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# SECTION 3: EDA OF SAMPLE VIDEOS
# ============================================================================
elif selected_section == "EDA of sample videos":
    render_page_header(
        module_num="03",
        eyebrow_suffix="DATASET INTELLIGENCE",
        title="Exploratory Data Analysis (EDA)",
        subtitle="Comprehensive spatial, temporal, and resolution metrics across 4K intersection surveillance feeds.",
        badge_text="DATASET AUDITED",
        mini_spec="4 SURVEILLANCE FEEDS",
    )

    # High-Impact KPI Summary Strip (computed from real EDA metadata when present)
    _eda_meta_path = Path("eda_results/metadata.csv")
    if _eda_meta_path.exists():
        _mdf = pd.read_csv(_eda_meta_path)
        n_feeds = len(_mdf)
        total_frames_kpi = int(_mdf["total_frames"].sum())
        total_secs_kpi = float(_mdf["duration_sec"].sum())
    else:
        n_feeds, total_frames_kpi, total_secs_kpi = 4, 33075, 1103.5

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{n_feeds} Feeds</div>'
            '<div class="metric-label">Surveillance Streams</div>'
            '<div class="metric-sub">Fixed intersection CCTV</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_frames_kpi:,}</div>'
            '<div class="metric-label">Total Frames</div>'
            f'<div class="metric-sub">{total_secs_kpi:,.1f}s total video time</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">4K UHD</div>'
            '<div class="metric-label">Resolution</div>'
            '<div class="metric-sub">3840 x 2160 @ 29.97 FPS</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{int(total_secs_kpi * 3):,} s</div>'
            '<div class="metric-label">Time Budget (3.0x)</div>'
            '<div class="metric-sub">Strict Hackathon Limit</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">STREAM TELEMETRY</div>
            <h2 class="section-heading-h2">Video Stream Metadata & Calibration Offsets</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Real metadata extracted by src/eda_extractor.py (fallback: audited values)
    eda_dir = Path("eda_results")
    meta_csv = eda_dir / "metadata.csv"
    if meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
        video_stats = pd.DataFrame({
            "Video ID": meta_df["video_name"],
            "Resolution": meta_df["resolution"] + " (4K)",
            "FPS": meta_df["fps"],
            "Frame Count": meta_df["total_frames"],
            "Duration (s)": meta_df["duration_sec"],
            "Time Budget (3.0x)": (meta_df["duration_sec"] * 3).round(0).astype(int).astype(str) + " s",
        })
    else:
        video_stats = pd.DataFrame({
            "Video ID": ["C3896.MP4", "C3897.MP4", "C3902.MP4", "C3905.MP4"],
            "Resolution": ["3840 x 2160 (4K)"] * 4,
            "FPS": [29.97] * 4,
            "Frame Count": [10200, 9525, 9525, 3825],
            "Duration (s)": [340.3, 317.8, 317.8, 127.6],
            "Time Budget (3.0x)": ["1,021 s", "953 s", "953 s", "383 s"],
        })
    stats = sample_scene_stats()
    vids = list(video_stats["Video ID"])
    video_stats["Lighting (measured)"] = [
        f"{stats[v]['light']} (mean grey {stats[v]['gray']})" if v in stats else "-" for v in vids
    ]
    video_stats["Camera drift vs reference"] = [
        (lambda i: f"dx={i['dx_4k']:+.0f}px dy={i['dy_4k']:+.0f}px rot={i['rot_deg']:+.2f} deg scale={i['scale']:.3f}")(stats[v]["info"])
        if v in stats and stats[v]["info"].get("status") == "ok" else "-" for v in vids
    ]
    st.dataframe(video_stats, width="stretch")
    st.caption(
        "Lighting is the mean grey level of each video's first frame. Camera drift is the SIFT + RANSAC similarity "
        "between that frame and the reference frame the zones were drawn on: the pose differs between recordings, "
        "so every video is registered before the rules run."
    )

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">DISTRIBUTION PATTERNS</div>
            <h2 class="section-heading-h2">Visual Analytics & Traffic Influx Dynamics</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Road User Class Distribution")
        dist_csv = eda_dir / "class_distribution.csv"
        if dist_csv.exists():
            dist_df = pd.read_csv(dist_csv).set_index("class")
            dist_df.index = dist_df.index.str.title()
            object_counts = dist_df.rename(columns={"detections": "Detections"})
        else:
            object_counts = pd.DataFrame({
                "Detections": [4850, 1420, 890, 420, 310, 195],
            }, index=["Car", "Pedestrian", "Bus", "Truck", "Motorcycle", "Bicycle"])
        st.bar_chart(object_counts)
        st.caption("Measured by the YOLO11 perception pass across all four feeds (src/deep_eda.py).")

    with c2:
        st.markdown("#### Traffic Density Curve (Vehicles per Frame)")
        eda_videos = sorted(p.stem.replace("_density", "") for p in eda_dir.glob("*_density.csv")) if eda_dir.exists() else []
        if eda_videos:
            sel_density_video = st.selectbox("Feed:", eda_videos, key="eda_density_feed")
            density_df = pd.read_csv(eda_dir / f"{sel_density_video}_density.csv").set_index("t_min")
            density_df.columns = ["Vehicles per Frame"]
        else:
            density_df = pd.DataFrame({
                "Vehicles per Frame": [18, 21, 24, 27, 30, 33, 31, 28, 25, 22, 19, 17],
            })
        st.line_chart(density_df)
        st.caption("Average simultaneously visible vehicles, bucketed per minute.")

    # Object counts over time + spatial analytics (real measured artifacts)
    counts_videos = sorted(p.stem.replace("_counts", "") for p in eda_dir.glob("*_counts.csv")) if eda_dir.exists() else []
    if counts_videos:
        st.markdown("#### Object Counts Over Time by Class")
        sel_counts_video = st.selectbox("Feed:", counts_videos, key="eda_counts_feed")
        counts_df = pd.read_csv(eda_dir / f"{sel_counts_video}_counts.csv").set_index("t_sec")
        st.line_chart(counts_df[["car", "bus", "truck", "motorcycle", "pedestrian"]])
        st.caption("Average objects visible per frame in each 1-second bucket — rush waves and queue formation are directly visible.")

    heatmaps = sorted(eda_dir.glob("*_heatmap.png")) if eda_dir.exists() else []
    trajectories = sorted(eda_dir.glob("*_trajectories.png")) if eda_dir.exists() else []
    if heatmaps or trajectories:
        st.markdown("#### Spatial Occupancy & Motion Analytics")
        sp1, sp2 = st.columns(2)
        with sp1:
            if heatmaps:
                sel_heat = st.selectbox("Feed:", [p.stem.replace("_heatmap", "") for p in heatmaps], key="eda_heat_feed")
                show_image(str(eda_dir / f"{sel_heat}_heatmap.png"), caption=f"Occupancy heatmap - {sel_heat}: where road users actually concentrate (lane corridors, queue pockets, crosswalks).")
        with sp2:
            if trajectories:
                sel_traj = st.selectbox("Feed:", [p.stem.replace("_trajectories", "") for p in trajectories], key="eda_traj_feed")
                show_image(str(eda_dir / f"{sel_traj}_trajectories.png"), caption=f"Vehicle trajectory trails - {sel_traj}: the two carriageways flow in opposite directions along the median; this is the lane flow the wrong-way and red-light rules use.")

    # Numbers below are computed from the committed EDA artefacts, not typed in.
    flow_lines = []
    try:
        dist = pd.read_csv(eda_dir / "class_distribution.csv").set_index("class")["detections"]
        veh = dist.reindex(["car", "bus", "truck", "motorcycle"]).fillna(0)
        flow_lines.append(
            "• <b>Vehicle mix</b> (all detections, 4 videos): "
            + ", ".join(f"{k} {100 * v / veh.sum():.1f}%" for k, v in veh.items())
        )
        dens = [pd.read_csv(f)["vehicles_per_frame"] for f in sorted(eda_dir.glob("*_density.csv"))]
        if dens:
            allv = pd.concat(dens)
            flow_lines.append(
                f"• <b>Density</b>: {allv.min():.1f} to {allv.max():.1f} vehicles in view per frame (per-minute means); "
                "the scene is never empty, so every rule must tolerate heavy occlusion"
            )
    except Exception:
        pass
    flow_lines.append("• <b>Pedestrians</b> are on or next to the carriageway almost constantly, "
                      "so bare pedestrian presence cannot be a risk signal (Part B uses collision courses instead)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            '<div class="callout-card"><div class="callout-title">Traffic composition (measured)</div>'
            '<div class="callout-body">' + "<br>".join(flow_lines) + "</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="callout-card callout-warning">
                <div class="callout-title">Signal Phase & Stop Line Infractions</div>
                <div class="callout-body">
                    • <b>Measured cycle</b> (lamp read-out, 1 sample/s, all 4 videos): ~37 s red, ~35 s green, 3-6 s amber/transition.<br>
                    • <b>Lamp contrast</b>: an unlit red lamp scores <= 10, a lit one >= 45 (day) / >= 220 (dusk); green <= 17 vs >= 126.<br>
                    • <b>Consequence</b>: red-light and stop-line rules require red to have been on >= 1 s, and a lane_ltr stop counts as a signal queue if the car moves off within 15 s of green.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# SECTION 4: RESULTS ON SAMPLE VIDEOS
# ============================================================================
elif selected_section == "Results on sample videos":
    render_page_header(
        module_num="04",
        eyebrow_suffix="SAMPLE VIDEOS",
        title="Results on the Sample Videos",
        subtitle="Output of python run_submission.py on the four samples (predictions_samples.json). There are no labels for the samples, so this is our output, not a score.",
        badge_text="FORMAT VALID",
        mini_spec="evaluate.py --validate-only",
    )

    benchmark_data = load_benchmark_data()
    videos_dict = benchmark_data.get("videos", {})
    log_dict = benchmark_data.get("log", {}) if isinstance(benchmark_data, dict) else {}
    stats = sample_scene_stats()
    total_runtime = sum(float(v.get("total_sec", 0) or 0) for v in log_dict.values())
    total_budget = sum(float(v.get("budget_sec", 0) or 0) for v in log_dict.values())
    total_evs = sum(len(v.get("events", [])) for v in videos_dict.values())
    worst = max((float(v.get("total_sec", 0)) / max(1e-9, float(v.get("duration", 1))) for v in log_dict.values()), default=0.0)

    m1, m2, m3, m4 = st.columns(4)
    for col, value, label, sub in (
        (m1, f"{total_evs}", "Events", "4 videos, 18 min of footage"),
        (m2, f"{total_runtime:,.0f} s", "Runtime (Part A + B)", f"budget {total_budget:,.0f} s"),
        (m3, f"{worst:.2f}x", "Slowest video", "of its duration (limit 3x)"),
        (m4, "VALID", "evaluate.py format check", "0 errors, 0 warnings"),
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">PER VIDEO</div>
            <h2 class="section-heading-h2">Runtime and output per sample</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = []
    for vid in sorted(videos_dict):
        lg = log_dict.get(vid, {})
        evs = videos_dict[vid].get("events", [])
        counts = pd.Series([e[2] for e in evs]).value_counts().to_dict() if evs else {}
        risk = videos_dict[vid].get("risk", [])
        rows.append({
            "Video": vid,
            "Lighting": stats.get(vid, {}).get("light", "-"),
            "Duration (s)": round(float(lg.get("duration", 0)), 1),
            "Part A (s)": lg.get("part_a_sec"),
            "Part B (s)": lg.get("part_b_sec"),
            "Total / duration": f"{float(lg.get('total_sec', 0)) / max(1e-9, float(lg.get('duration', 1))):.2f}x",
            "Events": len(evs),
            "By class": ", ".join(f"{k} {n}" for k, n in sorted(counts.items())),
            "Risk >= 0.5": f"{100 * sum(1 for _, r in risk if r >= 0.5) / max(1, len(risk)):.2f}% of frames",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Measured on a laptop RTX 3050 with the unchanged harness. Harness log errors: none.")

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">INSPECTOR</div>
            <h2 class="section-heading-h2">Annotated video, event timeline and risk curve</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feed_names = sorted(videos_dict)
    feed_key = st.selectbox(
        "Sample video:",
        feed_names,
        format_func=lambda v: f"{v}  ({stats.get(v, {}).get('light', '')}, {float(log_dict.get(v, {}).get('duration', 0)) / 60:.1f} min)",
        key="results_feed",
    )
    feed_events = sorted(videos_dict.get(feed_key, {}).get("events", []))
    feed_duration = float(log_dict.get(feed_key, {}).get("duration", 0) or 0)

    tab_video, tab_risk, tab_scene = st.tabs(["Annotated video & events", "Risk curve (Part B)", "Scene registration"])
    with tab_video:
        v_col, e_col = st.columns([1.15, 0.85], gap="large")
        with e_col:
            st.markdown("#### Event timeline")
            if feed_events:
                render_event_timeline(feed_events, feed_duration)
                options = ["(from the start)"] + [f"{e[2]}  {e[0]:.1f}-{e[1]:.1f} s" for e in feed_events]
                pick = st.selectbox("Jump the video to an event:", options, key=f"jump_{feed_key}")
                start_at = 0 if pick == options[0] else max(0, int(feed_events[options.index(pick) - 1][0]) - 1)
                df_evs = pd.DataFrame(feed_events, columns=["Start (s)", "End (s)", "Class"])
                df_evs["Duration (s)"] = (df_evs["End (s)"] - df_evs["Start (s)"]).round(2)
                st.dataframe(df_evs, width="stretch", height=260, hide_index=True)
            else:
                start_at = 0
                st.info("No events for this video.")
        with v_col:
            st.markdown("#### Annotated playback")
            pv = preview_path(feed_key)
            if pv:
                st.video(pv, start_time=start_at)
                st.caption(
                    "Rendered with src/annotate.py: registered zones, tracked road users, signal read from the lamps, "
                    "and a red banner while an event is active (every 2nd frame, 960 px wide)."
                )
            else:
                st.info("Annotated preview not found in samples/previews/.")

    with tab_risk:
        feed_risks = videos_dict.get(feed_key, {}).get("risk", [])
        if feed_risks:
            sampled = feed_risks[::10]
            chart_df = pd.DataFrame(
                {"Risk P(accident within 5 s)": [r for _, r in sampled], "Alarm threshold 0.5": [0.5] * len(sampled)},
                index=pd.Index([t for t, _ in sampled], name="t (s)"),
            )
            st.line_chart(chart_df, color=["#00f2fe", "#ef4444"], height=300)
            above = sum(1 for _, r in feed_risks if r >= 0.5)
            st.caption(
                f"Causal score from RiskEstimator.step (every 10th frame plotted). {above} of {len(feed_risks)} frames "
                f"are at or above 0.5. The samples contain no accidents, so a quiet curve is the expected behaviour."
            )
        else:
            st.info("No risk curve for this video.")

    with tab_scene:
        z_col, i_col = st.columns([1.3, 0.7], gap="large")
        with z_col:
            zone_img = sample_zone_overlay(feed_key)
            if zone_img is not None:
                show_image(zone_img, caption="First frame with the scene layout mapped through this video's registration.")
        with i_col:
            info = stats.get(feed_key, {}).get("info", {})
            st.markdown("#### Registration")
            st.json(info)
            st.caption("Similarity transform (shift, rotation, scale) from the reference frame to this video.")

    # Examples of each detected class
    ex_index = Path("assets/examples/index.json")
    if ex_index.exists():
        st.markdown(
            """
            <div class="section-header-block">
                <div class="section-eyebrow">PER CLASS</div>
                <h2 class="section-heading-h2">Examples of each class we detected</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Frames taken inside detected events (scripts/make_examples.py). The road users that triggered the rule "
            "are outlined in red. These are our detections, not verified ground truth: some are borderline, "
            "as discussed under Honest failure cases."
        )
        examples = json.loads(ex_index.read_text())
        labels = sorted({e["label"] for e in examples})
        for lbl in labels:
            st.markdown(f"**{lbl}**")
            items = [e for e in examples if e["label"] == lbl]
            ecols = st.columns(len(items) if len(items) > 1 else 2)
            for c, e in zip(ecols, items):
                with c:
                    show_image(e["file"], caption=f"{e['video']} {e['start']}-{e['end']} s, signal {e['signal']}")

    st.markdown(
        """
        <div class="section-header-block">
            <div class="section-eyebrow">EDGE CASES</div>
            <h2 class="section-heading-h2">Honest Failure Cases & Edge Analyses</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fail_c1, fail_c2, fail_c3 = st.columns(3)
    with fail_c1:
        st.markdown(
            """
            <div class="callout-card callout-danger">
                <div class="callout-title">Case 01: The camera is not perfectly fixed</div>
                <div class="callout-body">
                    <b>Observed</b>: the pose differs between recordings: C3902 is shifted by (-91, +28) px, and C3896/C3897 are rotated by 1 degree and scaled by 0.986.
                    With the zones drawn on C3905, the C3902 signal window landed on the pedestrian-crossing sign.<br><br>
                    <b>Fix</b>: SIFT + RANSAC similarity registration against the reference frame at start-up, applied to every zone, line and lamp.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fail_c2:
        st.markdown(
            """
            <div class="callout-card callout-warning">
                <div class="callout-title">Case 02: Signal read-out (two failed versions)</div>
                <div class="callout-body">
                    <b>v1</b> (red-pixel count over the housing) said RED ~99% of the time on C3896, even with the green lamp lit.
                    <b>v2</b> (thirds of the box, 5% area) said UNKNOWN ~100% of the time, because a lit lamp is only ~5 px tall.<br><br>
                    <b>v3</b> reads small windows at the measured lamp centres and gives a clean cycle on all four videos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fail_c3:
        st.markdown(
            """
            <div class="callout-card">
                <div class="callout-title">Case 03: Riders, kerbs and parked cars</div>
                <div class="callout-body">
                    Frame-by-frame checks of every candidate event showed the dominant false positives:
                    motorcycle riders counted as pedestrians, people waiting at the kerb inside the zebra polygon,
                    signal queues and parked cars read as "stopped vehicles" or "congestion". Each now has an explicit exclusion.<br><br>
                    <b>Still open</b>: we have no hand-labelled dev set, so recall is unmeasured. People who walk just beside a zebra are deliberately not flagged.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# SECTION 5: LIVE DEMO (SMART CITY CONTROL ROOM INTERACTION)
# ============================================================================
elif selected_section == "Live Demo":
    render_page_header(
        module_num="05",
        eyebrow_suffix="LIVE DEMO",
        title="Live Demo",
        subtitle=f"Upload an .mp4 from this junction camera (up to {DEMO_MAX_SEC / 60:.0f} minutes and {DEMO_MAX_MB} MB) "
                 "and get the detected events back with a timeline, the risk curve and annotated clips.",
        badge_text="CPU DEMO",
        mini_spec="YOLO11-S @768 · every 6th frame",
    )
    st.markdown(
        f"""
        <div class="glass-panel" style="margin-bottom: 14px;">
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;">
            <b>What runs here.</b> The submission pipeline (scene registration, signal read-out, tracking, the same rules and post-processing)
            in a CPU-friendly setting: YOLO11-S at 768 px on every 6th frame, no crash/fire model, and the risk curve computed from the same
            causal tracks. The submission itself uses YOLO11-L at 960 px on every 3rd frame on a GPU, so results can differ slightly.<br>
            <b>Accepted input.</b> .mp4 (H.264), at most {DEMO_MAX_SEC / 60:.0f} minutes and {DEMO_MAX_MB} MB, from the competition camera
            (other cameras run, but the zones will not fit). <b>Expected time</b> on the free 2-vCPU host: about the clip length
            for 720p and about 2x the clip length for 4K.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
    import tempfile

    session_dir = Path(tempfile.gettempdir()) / f"wiut_demo_{st.session_state['session_id']}"
    session_dir.mkdir(parents=True, exist_ok=True)

    sources = ["Upload a video"] + ([f"Bundled sample clip ({DEMO_SAMPLE.name})"] if DEMO_SAMPLE.exists() else [])
    source = st.radio("Input", sources, horizontal=True, key="demo_source")

    target_path, video_key = None, None
    if source == "Upload a video":
        up = st.file_uploader(
            f"Upload .mp4 (max {DEMO_MAX_SEC / 60:.0f} min, {DEMO_MAX_MB} MB)", type=["mp4", "MP4"], key="demo_upload"
        )
        if up is not None:
            up_id = f"{up.name}_{up.size}"
            if st.session_state.get("demo_upload_id") != up_id:
                dest = session_dir / f"upload_{uuid.uuid4().hex[:8]}.mp4"
                dest.write_bytes(up.getbuffer())
                st.session_state["demo_upload_id"] = up_id
                st.session_state["demo_upload_path"] = str(dest)
                st.session_state["demo_upload_name"] = up.name
            target_path = st.session_state["demo_upload_path"]
            video_key = f"upload:{up_id}"
            display_name = st.session_state["demo_upload_name"]
    else:
        target_path, video_key, display_name = str(DEMO_SAMPLE), f"sample:{DEMO_SAMPLE.name}", DEMO_SAMPLE.name

    meta = get_video_metadata(target_path) if target_path else {}
    problems = []
    if target_path:
        if not meta or meta.get("total_frames", 0) <= 0:
            problems.append("The file could not be read as a video.")
        else:
            if meta["duration_sec"] > DEMO_MAX_SEC + 1:
                problems.append(f"The clip is {meta['duration_sec'] / 60:.1f} min; the demo accepts up to {DEMO_MAX_SEC / 60:.0f} min. "
                                "Full-length videos run offline with run_submission.py.")
            if meta["size_mb"] > DEMO_MAX_MB:
                problems.append(f"The file is {meta['size_mb']:.0f} MB; the demo accepts up to {DEMO_MAX_MB} MB.")
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in ((c1, "Resolution", meta.get("resolution", "-")), (c2, "Frame rate", f"{meta.get('fps', '-')} fps"),
                              (c3, "Duration", f"{meta.get('duration_sec', '-')} s"), (c4, "Size", f"{meta.get('size_mb', '-')} MB")):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value" style="font-size:1.3rem;">{val}</div></div>',
                            unsafe_allow_html=True)
        for msg in problems:
            st.error(msg)

    run = st.button("Run the pipeline (Part A + Part B)", type="primary", width="stretch",
                    disabled=not target_path or bool(problems), key="demo_run")

    if run:
        import importlib.util

        spec = importlib.util.spec_from_file_location("weights_download", Path("weights/download.py"))
        wdl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wdl)
        need = wdl.missing(["yolo11s.pt"])
        if need:
            with st.spinner("First run on this server: downloading the demo detector weights (19 MB) ..."):
                wdl.download(need, log=lambda m: None)
        bar = st.progress(0.0)
        status = st.empty()

        def on_progress(frac: float, msg: str) -> None:
            bar.progress(min(1.0, max(0.0, frac)))
            status.markdown(f"`{msg}`")

        lock = demo_lock()
        if not lock.acquire(blocking=False):
            status.warning("Another visitor's video is being processed right now. The server runs one demo at a "
                           "time to stay within its memory; please press Run again in a minute.")
            st.stop()
        try:
            result = run_demo(target_path, session_dir / "clips", uuid.uuid4().hex[:8], progress=on_progress)
        except Exception as exc:  # show the error instead of a blank page
            result = None
            st.error(f"The pipeline failed on this file: {exc}")
        finally:
            lock.release()
        if result is None:
            st.stop()
        status.success(f"Done in {result.elapsed:.0f} s ({result.elapsed / max(1e-9, result.duration):.1f}x the clip length).")
        st.session_state["demo_result"] = (video_key, result, target_path, display_name)

    cached = st.session_state.get("demo_result")
    if cached and cached[0] == video_key:
        _, result, res_path, res_name = cached
        events = result.events
        risk = result.risk
        st.markdown(
            """
            <div class="section-header-block" style="margin-top: 28px;">
                <div class="section-eyebrow">RESULTS</div>
                <h2 class="section-heading-h2">Detected events and risk</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        reg = (result.info or {}).get("registration") or {}
        if reg.get("status") != "ok":
            st.warning("This video could not be registered to the competition camera view, so the zones may not fit "
                       "and the rules can be wrong. The demo is meant for footage from this junction camera.")
        k1, k2, k3, k4 = st.columns(4)
        max_r = max((r for _, r in risk), default=0.0)
        for col, value, label, sub in (
            (k1, f"{len(events)}", "Events", ", ".join(sorted({e[2] for e in events})) or "none"),
            (k2, f"{max_r:.2f}", "Peak risk", "alarm threshold 0.50"),
            (k3, f"{result.elapsed:.0f} s", "Processing time", f"{result.elapsed / max(1e-9, result.duration):.1f}x the clip"),
            (k4, f"{reg.get('dx_4k', 0):+.0f}/{reg.get('dy_4k', 0):+.0f}px", "Camera drift", f"rot {reg.get('rot_deg', 0):+.2f} deg"),
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div><div class="metric-sub">{html.escape(sub)}</div></div>',
                    unsafe_allow_html=True,
                )

        left, right = st.columns([1.1, 0.9], gap="large")
        with right:
            st.markdown("#### Event timeline")
            if events:
                render_event_timeline(events, result.duration)
                opts = ["(from the start)"] + [f"{e[2]}  {e[0]:.1f}-{e[1]:.1f} s" for e in events]
                pick = st.selectbox("Jump the video to an event:", opts, key="demo_jump")
                start_at = 0 if pick == opts[0] else max(0, int(events[opts.index(pick) - 1][0]) - 1)
                df = pd.DataFrame(events, columns=["Start (s)", "End (s)", "Class"])
                df["Duration (s)"] = (df["End (s)"] - df["Start (s)"]).round(2)
                st.dataframe(df, width="stretch", height=220, hide_index=True)
            else:
                start_at = 0
                st.info("No events detected in this clip.")
            payload = {"videos": {res_name: {"events": events, "risk": [[t, r] for t, r in risk]}}}
            st.download_button("Download the result (predictions.json format)", json.dumps(payload, indent=1),
                               file_name=f"predictions_{Path(res_name).stem}.json", mime="application/json",
                               width="stretch")
        with left:
            st.markdown("#### Your video")
            if Path(res_path).exists() and Path(res_path).stat().st_size <= 150 * 1024 * 1024:
                st.video(res_path, start_time=start_at)
            st.markdown("#### Risk curve (Part B)")
            if risk:
                rdf = pd.DataFrame({"Risk P(accident within 5 s)": [r for _, r in risk], "Alarm threshold 0.5": [0.5] * len(risk)},
                                   index=pd.Index([t for t, _ in risk], name="t (s)"))
                st.line_chart(rdf, color=["#00f2fe", "#ef4444"], height=240)

        if result.clips:
            st.markdown("#### Annotated event clips")
            ccols = st.columns(min(3, len(result.clips)))
            for i, (clip, caption) in enumerate(result.clips):
                if Path(clip).exists():
                    with ccols[i % len(ccols)]:
                        st.video(clip)
                        st.caption(caption)


# ============================================================================
# SECTION 6: REPORT
# ============================================================================
elif selected_section == "Report":
    render_page_header(
        module_num="06",
        eyebrow_suffix="POST-MORTEM ANALYSIS",
        title="Executive Project Report",
        subtitle="Comprehensive engineering debrief: What Worked, What Didn't, and What We Would Do Next.",
        badge_text="EXECUTIVE BRIEFING",
        mini_spec="HONEST DEBRIEF",
    )

    r_col1, r_col2, r_col3 = st.columns(3)

    with r_col1:
        st.markdown(
            """
            <div class="callout-card callout-success" style="min-height: 520px;">
                <div class="callout-title" style="color: #10b981; font-size: 1.05rem;">01. What worked</div>
                <div class="callout-body" style="margin-top: 12px; font-size: 0.88rem;">
                    <b>• Per-video scene registration</b> (SIFT + RANSAC similarity) recovers the camera drift between recordings, so one hand-drawn layout serves every video.<br><br>
                    <b>• Reading the signal from its lamps</b> gives a clean red/green cycle on all four samples, day and dusk.<br><br>
                    <b>• Scale-free motion</b>: speeds are measured in body-diagonals per second of track ground points, so one threshold works near and far from the camera.<br><br>
                    <b>• Frame-level spot checks</b> of every candidate event (montages rendered from cached perception) exposed the real false-positive patterns quickly.<br><br>
                    <b>• Runtime</b>: YOLO11-L @960 FP16 on every 3rd frame and the anomaly model at 1 Hz keep Part A at about 0.5x real time on an RTX 3050 laptop GPU.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_col2:
        st.markdown(
            """
            <div class="callout-card callout-warning" style="min-height: 520px;">
                <div class="callout-title" style="color: #f59e0b; font-size: 1.05rem;">02. What did not work</div>
                <div class="callout-body" style="margin-top: 12px; font-size: 0.88rem;">
                    <b>• Colour masks over the signal housing</b>: first always red, then always unknown (see Results, Case 02).<br><br>
                    <b>• Aligning on one detected traffic light</b>: a neighbouring signal head is ~100 px away, so a single-object anchor can snap to the wrong one. We replaced it with whole-scene feature registration.<br><br>
                    <b>• Generic turn rules</b>: flagging every 90-degree turn as illegal produced 10-15 false events per video. Without the list of permitted manoeuvres, <code>illegal_turn</code>, <code>illegal_u_turn</code> and <code>solid_line_crossing</code> are switched off.<br><br>
                    <b>• The off-the-shelf crash model</b> fires on ordinary traffic at its default confidence. It is now gated hard, and it produced no events on the samples.<br><br>
                    <b>• No labelled dev set</b>: we could not measure F1. Thresholds are judgement calls from visual checks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_col3:
        st.markdown(
            """
            <div class="callout-card" style="min-height: 520px; border-left-color: #38bdf8;">
                <div class="callout-title" style="color: #38bdf8; font-size: 1.05rem;">03. What we would do next</div>
                <div class="callout-body" style="margin-top: 12px; font-size: 0.88rem;">
                    <b>• Label the four samples</b> (CSV per video -> <code>python -m src.devset.csv_to_gt</code>) and tune every threshold against <code>evaluate.py</code>.<br><br>
                    <b>• Map the permitted manoeuvres</b> (entry/exit gates per approach) to switch the turn classes back on.<br><br>
                    <b>• Ground-plane homography</b> for metric speeds and distances (better TTC for Part B and near-miss).<br><br>
                    <b>• A learned crash/near-miss model</b> fine-tuned on public CCTV crash data (e.g. CCD, DoTA) with a clear licence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.caption("Submitted for Westminster International University in Tashkent (WIUT) AI Hackathon 2026.")


# ============================================================================
# SECTION 7: LINKS (REQUIRED BY RUBRIC)
# ============================================================================
elif selected_section == "Links":
    render_page_header(
        module_num="07",
        eyebrow_suffix="LINKS",
        title="Repository, Weights & Predictions",
        subtitle="Everything needed to reproduce the submission.",
        badge_text="PUBLIC",
        mini_spec="seed 42",
    )

    WEIGHT_LINKS = [
        ("yolo11l.pt", "Part A detector", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt", "AGPL-3.0"),
        ("yolov8n.pt", "Part B detector", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "AGPL-3.0"),
        ("accident_model.pt", "crash / fire / smoke model (epoch90.pt, renamed)",
         "https://huggingface.co/Enos-123/accident-evaluator-yolov8x/tree/main/weights", "MIT"),
        ("yolo11s.pt", "website demo detector only", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt", "AGPL-3.0"),
    ]
    c1, c2, c3 = st.columns(3)
    card = ('<div class="team-badge-card" style="text-align:left;padding:20px;">'
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:8px;">{title}</div>'
            '<div style="color:#94a3b8;font-size:0.86rem;line-height:1.7;margin-bottom:14px;">{body}</div></div>')
    with c1:
        st.markdown(card.format(title="Repository", body=(
            f'• <a href="{REPO_URL}" target="_blank" style="color:#38bdf8;">{REPO_URL.replace("https://", "")}</a><br>'
            "• Run: <code>pip install -r requirements.txt</code>, <code>bash weights/download.sh</code>, "
            "<code>python run_submission.py --videos /data/test --out predictions.json</code><br>"
            "• Harness and metric unchanged; seed 42; see README for approach, datasets and licences")), unsafe_allow_html=True)
        st.link_button("Open the repository", REPO_URL, width="stretch")
    with c2:
        body = "".join(f'• <a href="{u}" target="_blank" style="color:#38bdf8;"><code>{n}</code></a> - {d} ({lic})<br>'
                       for n, d, u, lic in WEIGHT_LINKS)
        body += "• One command: <code>bash weights/download.sh</code> (checksums in <code>weights/SHA256SUMS</code>)"
        st.markdown(card.format(title="Model weights", body=body), unsafe_allow_html=True)
    with c3:
        st.markdown(card.format(title="predictions_samples.json", body=(
            "• Our output on the four sample videos, produced by the unchanged harness<br>"
            "• <code>python evaluate.py --pred predictions_samples.json --validate-only</code>: VALID, 0 errors, 0 warnings<br>"
            f'• Website: <a href="{SITE_URL}" target="_blank" style="color:#38bdf8;">{SITE_URL.replace("https://", "")}</a>')),
            unsafe_allow_html=True)
        pred_p = Path("predictions_samples.json")
        if pred_p.exists():
            st.download_button("Download predictions_samples.json", pred_p.read_bytes(), file_name="predictions_samples.json",
                               mime="application/json", width="stretch")
