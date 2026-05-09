"""
gui.py — MarineGuard  ·  WarRisk and Maritime Insurance
Production Streamlit dashboard: deep-ocean aesthetic, 3-D bathymetric
background illusion, glassmorphic cards, ortho globe, density heatmap,
sunburst, animated CSS, zero AI-slop.
"""

import os
import sys
import io
import contextlib
import base64
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insurance_premium import (
    calculate_premium,
    append_premium_columns,
    REGION_MULTIPLIERS,
    DEFAULT_INSURED_VALUE,
    DEFAULT_LGD,
    DEFAULT_THETA,
    DEFAULT_BASE_RATE,
)

from app import (
    load_data,
    preprocess_data,
    explore_data,
    engineer_features,
    prepare_X_y,
    build_pipeline,
    run_kfold_cv,
    fit_final_model,
    build_classification_pipeline,
    run_classification_cv,
    generate_attack_probability_column,
)


st.set_page_config(
    page_title="MarineGuard · WarRisk & Maritime Insurance",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FILEPATH    = os.path.join(BASE_DIR, "pirate_attacks_clean.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "pirate_attacks_with_probability.csv")
LOGO_PATH   = os.path.join(BASE_DIR, "assets", "logo.png")


PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG        = "rgba(0,0,0,0)"
FONT_FAMILY     = "'Syne', 'DM Sans', sans-serif"

# Ocean-grade palette — no purple gradients, no generic AI blue
OCEAN  = "#0b4f6c"        # deep trench
OCEAN2 = "#1a7a9c"        # mid-water
SURF   = "#00c6ff"        # surface glint
FOAM   = "#a8e6f0"        # seafoam
RUST   = "#c0392b"        # hull rust / danger
AMBER  = "#e67e22"        # nav light amber
KELP   = "#1abc9c"        # bioluminescent green

RISK_PALETTE = {
    "Low":      "#1abc9c",
    "Moderate": "#e67e22",
    "High":     "#e74c3c",
    "Critical": "#7b0d1e",
}
BAND_ORDER = ["Low", "Moderate", "High", "Critical"]


def logo_b64() -> str:
    try:
        with open(LOGO_PATH, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except FileNotFoundError:
        return ""


def inject_css():
    logo = logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo}" '
        f'style="width:52px;height:52px;object-fit:contain;" />'
        if logo else
        '<span style="font-size:2.6rem;line-height:1;">⚓</span>'
    )

    st.markdown(f"""
    <style>
    /* ── Typefaces ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* ── 3-D World-map background ─────────────────────────────────────────── */
    /*  Uses an SVG background that mimics ocean bathymetry depth lines plus a
        perspective grid to create the illusion of a curved, 3-D ocean floor.  */
    .stApp {{
        background-color: #020c14;
        background-image:
            /* Perspective latitude lines (3-D ocean grid illusion) */
            repeating-linear-gradient(
                to bottom,
                transparent 0px,
                transparent 58px,
                rgba(0,198,255,0.04) 58px,
                rgba(0,198,255,0.04) 60px
            ),
            /* Perspective longitude lines that converge toward horizon */
            repeating-linear-gradient(
                to right,
                transparent 0px,
                transparent 118px,
                rgba(0,198,255,0.03) 118px,
                rgba(0,198,255,0.03) 120px
            ),
            /* Depth-layer radial halos — bathymetric contour illusion */
            radial-gradient(ellipse 140% 70% at 50% 110%,
                #051d36 0%, #030f1e 35%, #020c14 65%, #020c14 100%),
            /* Deep trench undertow */
            radial-gradient(ellipse 60% 40% at 80% 120%,
                rgba(26,122,156,0.18) 0%, transparent 60%),
            radial-gradient(ellipse 55% 35% at 20% 110%,
                rgba(11,79,108,0.22) 0%, transparent 60%);
        min-height: 100vh;
    }}

    /* Pseudo-horizon glow at top */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 320px;
        background: linear-gradient(180deg,
            rgba(0,198,255,0.07) 0%,
            transparent 100%);
        pointer-events: none;
        z-index: 0;
    }}

    /* ── Hero banner ─────────────────────────────────────────────────────── */
    .mg-hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg,
            rgba(11,79,108,0.7) 0%,
            rgba(2,12,20,0.85) 60%,
            rgba(11,79,108,0.5) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(0,198,255,0.18);
        border-radius: 18px;
        padding: 36px 44px 30px 44px;
        margin-bottom: 26px;
        animation: heroRise 0.8s cubic-bezier(.22,.68,0,1.2) both;
        box-shadow:
            0 2px 0 0 rgba(0,198,255,0.25) inset,
            0 24px 60px rgba(0,0,0,0.5),
            0 0 80px rgba(0,198,255,0.06);
    }}
    /* Sonar-sweep animation */
    .mg-hero::after {{
        content: "";
        position: absolute;
        top: -80%; left: -30%;
        width: 40%; height: 260%;
        background: linear-gradient(
            105deg,
            transparent 0%,
            rgba(0,198,255,0.07) 45%,
            rgba(0,198,255,0.03) 55%,
            transparent 100%
        );
        animation: sonarSweep 7s ease-in-out infinite;
    }}
    @keyframes sonarSweep {{
        0%   {{ transform: translateX(-120%) skewX(-8deg); }}
        100% {{ transform: translateX(550%) skewX(-8deg); }}
    }}
    @keyframes heroRise {{
        from {{ opacity:0; transform:translateY(-22px) scale(0.98); }}
        to   {{ opacity:1; transform:translateY(0) scale(1); }}
    }}
    .mg-hero-wordmark {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 10px;
    }}
    .mg-hero-wordmark .logo-wrap {{
        flex-shrink: 0;
        width: 56px; height: 56px;
        border-radius: 14px;
        background: rgba(0,198,255,0.1);
        border: 1px solid rgba(0,198,255,0.25);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 20px rgba(0,198,255,0.15);
    }}
    .mg-title {{
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f0f9ff;
        line-height: 1;
        /* Subtle 3-D text-shadow depth */
        text-shadow:
            0 1px 0 rgba(0,198,255,0.4),
            0 2px 0 rgba(0,198,255,0.2),
            0 4px 12px rgba(0,0,0,0.5);
    }}
    .mg-title span {{ color: {SURF}; }}
    .mg-tagline {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: rgba(0,198,255,0.6);
        margin-top: 3px;
    }}
    .mg-sub {{
        color: rgba(168,230,240,0.55);
        font-size: 0.88rem;
        letter-spacing: 0.3px;
        margin-top: 6px;
    }}

    /* ── KPI cards ───────────────────────────────────────────────────────── */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin: 20px 0 24px 0;
    }}
    .kpi-card {{
        position: relative;
        background: linear-gradient(145deg,
            rgba(11,79,108,0.45) 0%,
            rgba(2,12,20,0.6) 100%);
        border: 1px solid rgba(0,198,255,0.14);
        border-radius: 14px;
        padding: 20px 18px 16px 18px;
        overflow: hidden;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: cardUp 0.55s cubic-bezier(.22,.68,0,1.1) both;
        /* 3-D pop */
        box-shadow:
            0 1px 0 rgba(0,198,255,0.2) inset,
            0 8px 32px rgba(0,0,0,0.45);
    }}
    .kpi-card:nth-child(1){{animation-delay:.04s}}
    .kpi-card:nth-child(2){{animation-delay:.10s}}
    .kpi-card:nth-child(3){{animation-delay:.16s}}
    .kpi-card:nth-child(4){{animation-delay:.22s}}
    .kpi-card:nth-child(5){{animation-delay:.28s}}
    @keyframes cardUp {{
        from{{opacity:0;transform:translateY(20px)}}
        to  {{opacity:1;transform:translateY(0)}}
    }}
    .kpi-card:hover {{
        transform: translateY(-5px) scale(1.015);
        box-shadow: 0 16px 48px rgba(0,198,255,0.14), 0 2px 0 rgba(0,198,255,0.25) inset;
        border-color: rgba(0,198,255,0.32);
    }}
    /* bottom-edge accent bar */
    .kpi-card::after {{
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        border-radius: 0 0 14px 14px;
    }}
    .kc-blue::after   {{ background: linear-gradient(90deg,{SURF},{OCEAN2}); }}
    .kc-teal::after   {{ background: linear-gradient(90deg,{KELP},#0d7a5e); }}
    .kc-amber::after  {{ background: linear-gradient(90deg,{AMBER},#c0611a); }}
    .kc-rust::after   {{ background: linear-gradient(90deg,{RUST},#7b0d1e); }}
    .kc-mid::after    {{ background: linear-gradient(90deg,{OCEAN2},{OCEAN}); }}
    .kpi-label {{
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: rgba(168,230,240,0.45);
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-family: 'Syne', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1;
        color: #e8f9ff;
        /* subtle depth */
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    .kpi-value.blue   {{ color: {SURF};  }}
    .kpi-value.teal   {{ color: {KELP};  }}
    .kpi-value.amber  {{ color: {AMBER}; }}
    .kpi-value.rust   {{ color: {RUST};  }}
    .kpi-value.mid    {{ color: {OCEAN2};}}
    .kpi-delta {{
        font-size: 0.74rem;
        color: rgba(168,230,240,0.35);
        margin-top: 5px;
        font-family: 'DM Mono', monospace;
    }}

    /* ── Section headers ─────────────────────────────────────────────────── */
    .sec-head {{
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 34px 0 18px 0;
    }}
    .sec-pip {{
        width: 3px;
        height: 22px;
        border-radius: 2px;
        background: linear-gradient(180deg, {SURF}, {OCEAN2});
        box-shadow: 0 0 8px rgba(0,198,255,0.5);
    }}
    .sec-head h3 {{
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #c8e8f5;
        margin: 0;
        letter-spacing: 0.3px;
    }}

    /* ── Tabs ─────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(11,79,108,0.22);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid rgba(0,198,255,0.1);
        gap: 3px;
        backdrop-filter: blur(8px);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: rgba(168,230,240,0.5) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600;
        font-size: 0.83rem;
        padding: 9px 18px;
        transition: all 0.2s ease;
        letter-spacing: 0.2px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg,
            rgba(0,198,255,0.18),
            rgba(26,122,156,0.22)) !important;
        color: {SURF} !important;
        border: 1px solid rgba(0,198,255,0.3) !important;
        box-shadow: 0 0 12px rgba(0,198,255,0.1);
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        animation: tabIn 0.35s ease-out;
    }}
    @keyframes tabIn {{
        from{{ opacity:0; transform:translateY(5px); }}
        to  {{ opacity:1; transform:translateY(0); }}
    }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg,
            rgba(5,20,35,0.96) 0%,
            rgba(2,12,20,0.98) 100%) !important;
        border-right: 1px solid rgba(0,198,255,0.1) !important;
        backdrop-filter: blur(20px);
    }}
    .sb-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px 4px 22px 4px;
        border-bottom: 1px solid rgba(0,198,255,0.08);
        margin-bottom: 20px;
    }}
    .sb-brand-logo {{
        width: 40px; height: 40px;
        border-radius: 10px;
        background: rgba(0,198,255,0.08);
        border: 1px solid rgba(0,198,255,0.2);
        display: flex; align-items:center; justify-content:center;
        font-size: 1.3rem;
    }}
    .sb-brand-name {{
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        color: #e8f9ff;
        letter-spacing: -0.3px;
    }}
    .sb-brand-tag {{
        font-size: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(0,198,255,0.4);
        margin-top: 1px;
    }}
    .sb-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,198,255,0.06);
        border: 1px solid rgba(0,198,255,0.12);
        border-radius: 7px;
        padding: 5px 11px;
        font-size: 0.75rem;
        color: rgba(168,230,240,0.6);
        margin: 3px 4px 3px 0;
        font-family: 'DM Mono', monospace;
    }}
    .sb-chip b {{ color: {SURF}; }}

    /* ── Divider ─────────────────────────────────────────────────────────── */
    .mg-div {{
        height: 1px;
        background: linear-gradient(90deg,
            transparent, rgba(0,198,255,0.18), transparent);
        margin: 28px 0;
    }}

    /* ── Metrics override ────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {{
        background: rgba(11,79,108,0.25);
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid rgba(0,198,255,0.1);
    }}
    div[data-testid="stMetricValue"] {{
        color: #e8f9ff !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: rgba(168,230,240,0.45) !important;
        font-size: 0.68rem !important;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }}

    /* ── Download button ─────────────────────────────────────────────────── */
    .stDownloadButton > button {{
        background: linear-gradient(135deg,
            rgba(0,198,255,0.14), rgba(26,122,156,0.16)) !important;
        border: 1px solid rgba(0,198,255,0.3) !important;
        color: {SURF} !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg,
            rgba(0,198,255,0.24), rgba(26,122,156,0.26)) !important;
        box-shadow: 0 4px 20px rgba(0,198,255,0.18) !important;
        transform: translateY(-1px);
    }}

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: #020c14; }}
    ::-webkit-scrollbar-thumb {{ background: #0b4f6c; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {OCEAN2}; }}

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    .stSpinner > div {{ border-top-color: {SURF} !important; }}

    /* ── Dataframe ───────────────────────────────────────────────────────── */
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

    /* ── Chart title ─────────────────────────────────────────────────────── */
    .ct {{
        font-family: 'Syne', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: rgba(0,198,255,0.55);
        margin-bottom: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)

   
    st.session_state["_logo_html"] = logo_html



def chart_layout(height=340, extra=None):
    base = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=8, r=8, t=28, b=8),
        height=height,
        font=dict(family=FONT_FAMILY, color="rgba(168,230,240,0.65)"),
        xaxis=dict(gridcolor="rgba(0,198,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,198,255,0.05)", zeroline=False),
    )
    if extra:
        base.update(extra)
    return base


def section(label, icon=""):
    st.markdown(
        f'<div class="sec-head"><div class="sec-pip"></div>'
        f'<h3>{icon}&nbsp;{label}</h3></div>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<div class="mg-div"></div>', unsafe_allow_html=True)


def ct(label):
    st.markdown(f'<div class="ct">{label}</div>', unsafe_allow_html=True)


def kpi_card(label, value, color="blue", delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card kc-{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {color}">{value}</div>'
        f'{delta_html}</div>'
    )


def kpi_row(cards):
    st.markdown(
        f'<div class="kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )



@st.cache_data(show_spinner=False)
def run_full_pipeline(filepath: str, output_path: str):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df_raw   = load_data(filepath)
        df_clean = preprocess_data(df_raw)
        explore_data(df_clean)

        df_eng, cat_cols = engineer_features(df_clean)
        y_clf      = df_eng["attack_occurred"].copy()
        df_for_mod = df_eng.drop(columns=["attack_occurred"])

        X, y, numeric_cols, cat_cols = prepare_X_y(
            df_for_mod, cat_cols=cat_cols, target="log_shore_distance"
        )
        pipe       = build_pipeline(numeric_cols, cat_cols)
        cv_metrics = run_kfold_cv(pipe, X, y, n_splits=10)
        final_pipe, importance_df = fit_final_model(pipe, X, y)

        clf_numeric_cols = (
            [c for c in numeric_cols if c != "attack_severity"]
            + ["log_shore_distance"]
        )
        clf_cat_cols = list(cat_cols)
        X_clf = df_for_mod[clf_numeric_cols + clf_cat_cols].copy()

        clf_pipe     = build_classification_pipeline(clf_numeric_cols, clf_cat_cols)
        clf_cv       = run_classification_cv(clf_pipe, X_clf, y_clf, n_splits=10)
        df_with_prob = generate_attack_probability_column(
            clf_pipe, X_clf, y_clf, df_clean, output_path
        )

    y_pred = final_pipe.predict(X)

    return {
        "df_clean":      df_clean,
        "df_with_prob":  df_with_prob,
        "importance_df": importance_df,
        "y_actual":      y.tolist(),
        "y_pred":        y_pred.tolist(),
        "clf_acc_train": clf_cv["acc_train"].tolist(),
        "clf_acc_test":  clf_cv["acc_test"].tolist(),
        "clf_brier":     clf_cv["brier_test"].tolist(),
        "n_numeric_reg": len(numeric_cols),
        "n_cat":         len(cat_cols),
        "n_numeric_clf": len(clf_numeric_cols),
    }


inject_css()


_splash = st.empty()
with _splash.container():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:62vh;gap:1.6rem;">
      <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:900;
                  color:#e8f9ff;letter-spacing:-1px;
                  text-shadow:0 1px 0 rgba(0,198,255,.45),
                              0 3px 0 rgba(0,198,255,.2),
                              0 6px 20px rgba(0,0,0,.6);">
        MarineGuard
      </div>
      <div style="font-size:0.72rem;letter-spacing:3.5px;text-transform:uppercase;
                  color:rgba(0,198,255,.5);">
        WarRisk &amp; Maritime Insurance
      </div>
      <div style="width:260px;height:4px;border-radius:99px;overflow:hidden;
                  background:rgba(0,198,255,.1);">
        <div style="width:100%;height:100%;
                    background:linear-gradient(90deg,transparent,#00c6ff,transparent);
                    animation:sonarSweep 1.8s linear infinite;"></div>
      </div>
      <div style="color:rgba(168,230,240,.35);font-size:.84rem;
                  font-family:'DM Mono',monospace;letter-spacing:.5px;">
        Initialising ML pipeline…
      </div>
    </div>
    """, unsafe_allow_html=True)

with st.spinner(""):
    results = run_full_pipeline(FILEPATH, OUTPUT_PATH)

_splash.empty()

df_clean      = results["df_clean"]
df_with_prob  = results["df_with_prob"]
importance_df = results["importance_df"]

n_folds = len(results["clf_acc_test"])
folds   = [f"F{i+1}" for i in range(n_folds)]

cv_clf_df = pd.DataFrame({
    "Fold":       folds,
    "Train Acc":  np.round(results["clf_acc_train"], 4),
    "Test Acc":   np.round(results["clf_acc_test"], 4),
    "Brier":      np.round(results["clf_brier"], 4),
})


logo_html = st.session_state.get("_logo_html", "⚓")
st.markdown(f"""
<div class="mg-hero">
  <div class="mg-hero-wordmark">
    <div class="logo-wrap">{logo_html}</div>
    <div>
      <div class="mg-title">Marine<span>Guard</span></div>
      <div class="mg-tagline">WarRisk &amp; Maritime Insurance</div>
    </div>
  </div>
  <div class="mg-sub">
    Global Maritime Piracy Analytics &nbsp;·&nbsp;
    Gradient Boosting ML &nbsp;·&nbsp;
    6 555 Incidents · 1994 – 2020
  </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-brand-logo">{logo_html}</div>
      <div>
        <div class="sb-brand-name">MarineGuard</div>
        <div class="sb-brand-tag">WarRisk &amp; Maritime</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Filters**")

    all_regions = sorted(df_with_prob["region"].unique().tolist())
    sel_regions = st.multiselect("Regions", options=all_regions, default=all_regions)

    year_min = int(df_with_prob["year"].min())
    year_max = int(df_with_prob["year"].max())
    sel_years = st.slider("Year Range", year_min, year_max, (year_min, year_max))

    sel_bands = st.multiselect("Risk Band", options=BAND_ORDER, default=BAND_ORDER)
    min_prob  = st.slider("Min Probability (%)", 0.0, 100.0, 0.0, step=1.0)

    st.markdown("---")
    total_recs = len(df_with_prob)
    st.markdown(
        f'<div class="sb-chip">Records&nbsp;<b>{total_recs:,}</b></div>'
        f'<div class="sb-chip">Regions&nbsp;<b>{len(all_regions)}</b></div>'
        f'<div class="sb-chip">Years&nbsp;<b>{year_min}–{year_max}</b></div>',
        unsafe_allow_html=True,
    )

mask = (
    df_with_prob["region"].isin(sel_regions)
    & df_with_prob["year"].between(sel_years[0], sel_years[1])
    & df_with_prob["risk_band"].isin(sel_bands)
    & (df_with_prob["attack_probability_pct"] >= min_prob)
)
df_f = df_with_prob[mask].copy()



tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  Dataset Overview  ",
    "  Attack Patterns  ",
    "  Regression Model  ",
    "  Classification  ",
    "  Probability Analysis  ",
    "  Insurance Premium  ",
])


with tab1:
    section("Dataset Overview")

    kpi_row([
        kpi_card("Total Incidents",  f"{len(df_with_prob):,}",                       "blue",  "⚓"),
        kpi_card("Year Range",       f"{year_min} – {year_max}",                     "mid",   "📅"),
        kpi_card("Countries",        str(df_with_prob["nearest_country"].nunique()),  "teal",  "🌍"),
        kpi_card("Regions",          str(df_with_prob["region"].nunique()),           "amber", "🗺️"),
    ])
    divider()

    c1, c2 = st.columns(2)
    with c1:
        ct("Attack Type Distribution")
        at_counts = df_with_prob["attack_type"].value_counts().reset_index()
        at_counts.columns = ["Attack Type", "Count"]
        fig_at = px.bar(
            at_counts, x="Count", y="Attack Type", orientation="h",
            color="Count", color_continuous_scale=[[0, OCEAN], [1, SURF]], text="Count",
        )
        fig_at.update_layout(**chart_layout(320), showlegend=False, coloraxis_showscale=False)
        fig_at.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_at, use_container_width=True)

    with c2:
        ct("Vessel Status Distribution")
        vs_counts = df_with_prob["vessel_status"].value_counts().reset_index()
        vs_counts.columns = ["Vessel Status", "Count"]
        fig_vs = px.bar(
            vs_counts, x="Count", y="Vessel Status", orientation="h",
            color="Count", color_continuous_scale=[[0, OCEAN], [1, KELP]], text="Count",
        )
        fig_vs.update_layout(**chart_layout(320), showlegend=False, coloraxis_showscale=False)
        fig_vs.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_vs, use_container_width=True)

    divider()
    ct("Sample Records")
    display_cols = [
        "year", "month", "region", "nearest_country", "attack_type",
        "vessel_status", "shore_distance", "attack_occurred",
        "attack_probability_pct", "risk_band",
    ]
    st.dataframe(df_f[display_cols].head(200), use_container_width=True, height=380)
    st.download_button(
        label="⬇  Download Filtered Data (CSV)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_pirate_attacks.csv",
        mime="text/csv",
    )



with tab2:
    section("Attack Patterns — Exploratory Analysis")

    ct("Annual Attack Count Trend")
    annual   = df_with_prob.groupby("year").size().reset_index(name="Incidents")
    peak_yr  = int(annual.loc[annual["Incidents"].idxmax(), "year"])
    peak_cnt = int(annual["Incidents"].max())
    fig_yr = go.Figure(go.Scatter(
        x=annual["year"], y=annual["Incidents"],
        mode="lines+markers",
        line=dict(color=SURF, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(0,198,255,0.10)",
        marker=dict(size=6, color=OCEAN2),
    ))
    fig_yr.add_annotation(
        x=peak_yr, y=peak_cnt,
        text=f"Peak {peak_yr}<br>{peak_cnt} incidents",
        showarrow=True, arrowhead=2, arrowcolor=AMBER,
        font=dict(color=AMBER, size=12),
        bgcolor="rgba(0,0,0,0.6)", bordercolor=AMBER, borderpad=5,
    )
    fig_yr.update_layout(**chart_layout(300), xaxis_title="Year", yaxis_title="Incidents")
    st.plotly_chart(fig_yr, use_container_width=True)

    divider()
    c1, c2 = st.columns(2)

    with c1:
        ct("Incidents by Region")
        reg_counts = df_with_prob["region"].value_counts().reset_index()
        reg_counts.columns = ["Region", "Count"]
        fig_reg = px.bar(
            reg_counts, x="Count", y="Region", orientation="h",
            color="Count", color_continuous_scale=[[0, OCEAN], [0.5, OCEAN2], [1, SURF]], text="Count",
        )
        fig_reg.update_layout(**chart_layout(340), showlegend=False, coloraxis_showscale=False)
        fig_reg.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_reg, use_container_width=True)

    with c2:
        ct("Year × Month Attack Heatmap")
        hm = df_with_prob.groupby(["year", "month"]).size().reset_index(name="Count")
        hm_pivot = hm.pivot(index="year", columns="month", values="Count").fillna(0)
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values,
            x=month_labels[:hm_pivot.shape[1]],
            y=hm_pivot.index.tolist(),
            colorscale=[[0, "#020c14"], [0.4, OCEAN], [0.7, OCEAN2], [1, SURF]],
        ))
        fig_hm.update_layout(**chart_layout(340), xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(fig_hm, use_container_width=True)

    divider()
    ct("Shore Distance by Region (Violin)")
    fig_vln = px.violin(
        df_with_prob, y="shore_distance", x="region", box=True,
        color="region",
        color_discrete_sequence=[SURF, OCEAN2, KELP, AMBER, RUST, FOAM, "#2980b9", "#16a085"],
        log_y=True,
    )
    fig_vln.update_layout(**chart_layout(380), showlegend=False,
                          xaxis_title="Region", yaxis_title="Shore Distance (km, log)")
    st.plotly_chart(fig_vln, use_container_width=True)

    divider()
    ct("Geographic Distribution — Orthographic Globe")
    geo_sample = df_with_prob.sample(n=min(3000, len(df_with_prob)), random_state=42)
    fig_geo = px.scatter_geo(
        geo_sample, lat="latitude", lon="longitude",
        color="region",
        hover_data=["year", "attack_type", "vessel_status", "shore_distance"],
        opacity=0.75, size_max=7, projection="orthographic",
        color_discrete_sequence=[SURF, OCEAN2, KELP, AMBER, RUST, FOAM, "#2980b9", "#16a085"],
    )
    # Geographic Distribution — Orthographic Globe
    geo_sample = df_with_prob.sample(n=min(3000, len(df_with_prob)), random_state=42)
    fig_geo = px.scatter_geo(
        geo_sample, lat="latitude", lon="longitude",
        color="region",
        hover_data=["year", "attack_type", "vessel_status", "shore_distance"],
        opacity=0.75, size_max=7, projection="orthographic",
        color_discrete_sequence=[SURF, OCEAN2, KELP, AMBER, RUST, FOAM, "#2980b9", "#16a085"],
    )
    
    fig_geo.update_layout(
        **chart_layout(460),
        geo=dict(
            bgcolor=CHART_BG,
            showland=True,  
            landcolor="#0d2233",
            showocean=True, 
            oceancolor="#051428",
            showcoastlines=True, 
            coastlinecolor=OCEAN2,
            showcountries=True, 
            countrycolor=OCEAN,
            # Removed showgraticules to prevent the ValueError
            lataxis=dict(showgrid=True, gridcolor="rgba(0,198,255,0.05)"),
            lonaxis=dict(showgrid=True, gridcolor="rgba(0,198,255,0.05)"),
            projection_type='orthographic'
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.15, 
            xanchor="center", 
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    st.plotly_chart(fig_geo, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regression Model
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    section("Regression Model — Shore Distance Prediction")

    divider()
    y_actual  = np.array(results["y_actual"])
    y_pred    = np.array(results["y_pred"])
    residuals = np.abs(y_pred - y_actual)

    ct("Predicted vs Actual (log shore distance)")
    fig_pa = go.Figure()
    fig_pa.add_trace(go.Scattergl(
        x=y_actual, y=y_pred, mode="markers",
        marker=dict(
            color=residuals, colorscale=[[0, OCEAN], [0.5, OCEAN2], [1, SURF]],
            size=4, opacity=0.55,
            colorbar=dict(title="Abs Residual", thickness=12,
                          tickfont=dict(color="rgba(168,230,240,.6)")),
        ),
        name="Predictions",
    ))
    _lo = float(min(y_actual.min(), y_pred.min()))
    _hi = float(max(y_actual.max(), y_pred.max()))
    fig_pa.add_trace(go.Scatter(
        x=[_lo, _hi], y=[_lo, _hi], mode="lines",
        line=dict(color=AMBER, dash="dash", width=2), name="Perfect fit",
    ))
    fig_pa.update_layout(**chart_layout(380),
                         xaxis_title="Actual (log km)", yaxis_title="Predicted (log km)",
                         legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_pa, use_container_width=True)

    divider()
    ct("Feature Importances (Top 25)")
    top_imp = importance_df.head(25).sort_values("importance")
    norm    = top_imp["importance"] / top_imp["importance"].max()
    # Ocean depth gradient: deep blue → surface cyan
    colors_imp = [
        f"rgba({int(11+215*v)},{int(79+119*v)},{int(108+147*v)},1)"
        for v in norm
    ]
    fig_imp = go.Figure(go.Bar(
        x=top_imp["importance"],
        y=top_imp["feature"],
        orientation="h",
        marker_color=colors_imp,
        marker_line_width=0,
        text=top_imp["importance"].round(4),
        textposition="outside",
    ))
    fig_imp.update_layout(**chart_layout(540), xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Classification Model
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    section("Classification Model — Attack Occurrence Prediction")

    acc_mean   = float(np.mean(results["clf_acc_test"]))
    brier_mean = float(np.mean(results["clf_brier"]))

    kpi_row([
        kpi_card("Mean Accuracy",      f"{acc_mean:.4f}",   "mid",   "✅"),
        kpi_card("Mean Brier Score",   f"{brier_mean:.4f}", "teal",  "📐"),
    ])
    divider()

    c1, c2 = st.columns(2)
    with c1:
        ct("Train vs Test Accuracy per Fold")
        fig_acc_fold = go.Figure()
        fig_acc_fold.add_trace(go.Bar(
            name="Train Accuracy", x=folds, y=results["clf_acc_train"],
            marker_color=OCEAN2, marker_line_width=0,
        ))
        fig_acc_fold.add_trace(go.Bar(
            name="Test Accuracy", x=folds, y=results["clf_acc_test"],
            marker_color=AMBER, marker_line_width=0,
        ))
        fig_acc_fold.update_layout(**chart_layout(300), barmode="group", yaxis_title="Accuracy",
                                   yaxis_range=[0.5, 1.0],
                                   legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_acc_fold, use_container_width=True)

    with c2:
        ct("Accuracy & Brier Score per Fold")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=folds, y=results["clf_acc_test"], name="Test Accuracy",
            mode="lines+markers", line=dict(color=KELP, width=2.5), marker=dict(size=8),
        ))
        fig_acc.add_trace(go.Scatter(
            x=folds, y=results["clf_brier"], name="Brier Score",
            mode="lines+markers", line=dict(color=RUST, width=2.5, dash="dot"),
            marker=dict(size=8), yaxis="y2",
        ))
        fig_acc.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            margin=dict(l=8, r=8, t=28, b=8), height=300,
            font=dict(color="rgba(168,230,240,.65)"),
            xaxis=dict(gridcolor="rgba(0,198,255,0.05)"),
            yaxis=dict(title="Accuracy", range=[0.5, 1.0], gridcolor="rgba(0,198,255,0.05)"),
            yaxis2=dict(title="Brier Score", overlaying="y", side="right",
                        range=[0, 0.5], gridcolor="rgba(0,198,255,0.05)"),
            legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    divider()
    c1, c2 = st.columns(2)
    with c1:
        ct("Class Balance (Donut)")
        cls_cnt = df_with_prob["attack_occurred"].value_counts().reset_index()
        cls_cnt.columns = ["Class", "Count"]
        cls_cnt["Class"] = cls_cnt["Class"].map({0: "Incomplete", 1: "Completed"})
        fig_cls = px.pie(
            cls_cnt, names="Class", values="Count", hole=0.5,
            color="Class",
            color_discrete_map={"Completed": RUST, "Incomplete": OCEAN2},
        )
        fig_cls.update_layout(**chart_layout(320))
        st.plotly_chart(fig_cls, use_container_width=True)

    with c2:
        ct("Completed vs Attempted by Region")
        reg_cls = (
            df_with_prob.groupby(["region", "attack_occurred"])
            .size().reset_index(name="Count")
        )
        reg_cls["Class"] = reg_cls["attack_occurred"].map({0: "Incomplete", 1: "Completed"})
        fig_rcls = px.bar(
            reg_cls, x="Count", y="region", color="Class", orientation="h",
            barmode="stack",
            color_discrete_map={"Completed": RUST, "Incomplete": OCEAN2},
        )
        fig_rcls.update_layout(**chart_layout(320), yaxis_title="", xaxis_title="Incidents",
                               legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rcls, use_container_width=True)

    divider()
    ct("10-Fold Stratified CV Detail")
    st.dataframe(cv_clf_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Probability Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    section("Attack Probability Analysis")

    avg_prob = df_f["attack_probability_pct"].mean() if len(df_f) else 0.0
    pct_high = (df_f["risk_band"].isin(["High","Critical"])).sum() / max(len(df_f),1) * 100
    pct_crit = (df_f["risk_band"]=="Critical").sum() / max(len(df_f),1) * 100

    kpi_row([
        kpi_card("Filtered Records", f"{len(df_f):,}",    "blue",  "🔍"),
        kpi_card("Avg Attack Prob",  f"{avg_prob:.1f}%",  "mid",   "🎲"),
        kpi_card("High or Critical", f"{pct_high:.1f}%",  "rust",  "⚠️"),
        kpi_card("Critical Only",    f"{pct_crit:.1f}%",  "amber", "🚨"),
    ])
    divider()

    c1, c2 = st.columns(2)
    with c1:
        ct("Risk Band Distribution (Donut)")
        bnd_cnt = (
            df_f["risk_band"].value_counts()
            .reindex(BAND_ORDER, fill_value=0).reset_index()
        )
        bnd_cnt.columns = ["Risk Band", "Count"]
        fig_bd = px.pie(
            bnd_cnt, names="Risk Band", values="Count", hole=0.45,
            color="Risk Band", color_discrete_map=RISK_PALETTE,
            category_orders={"Risk Band": BAND_ORDER},
        )
        fig_bd.update_layout(**chart_layout(340))
        st.plotly_chart(fig_bd, use_container_width=True)

    with c2:
        ct("Probability Histogram by Risk Band")
        fig_sh = px.histogram(
            df_f[df_f["risk_band"].isin(BAND_ORDER)],
            x="attack_probability_pct", color="risk_band",
            nbins=50, barmode="stack",
            color_discrete_map=RISK_PALETTE,
            category_orders={"risk_band": BAND_ORDER},
            labels={"attack_probability_pct": "Probability (%)"},
        )
        fig_sh.update_layout(**chart_layout(340),
                             xaxis_title="Probability (%)", yaxis_title="Count",
                             legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_sh, use_container_width=True)

    divider()
    ct("Probability Density Map")
    _geo_df = df_f.dropna(subset=["latitude","longitude"])
    if len(_geo_df) > 0:
        fig_dm = go.Figure(go.Densitymapbox(
            lat=_geo_df["latitude"], lon=_geo_df["longitude"],
            z=_geo_df["attack_probability_pct"],
            radius=14,
            colorscale=[[0, "#020c14"], [0.3, OCEAN], [0.6, OCEAN2], [1, SURF]],
            showscale=True,
            colorbar=dict(title="Prob (%)", tickfont=dict(color="rgba(168,230,240,.6)")),
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Prob: %{z:.1f}%<extra></extra>",
        ))
        fig_dm.update_layout(
            mapbox=dict(style="carto-darkmatter", zoom=1, center=dict(lat=10, lon=60)),
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            margin=dict(l=0, r=0, t=0, b=0), height=460,
        )
        st.plotly_chart(fig_dm, use_container_width=True)
    else:
        st.info("No data matches the current filters for the heatmap.")

    divider()
    ct("Risk Band → Attack Type (Sunburst)")
    _sb_df = df_f.groupby(["risk_band","attack_type"]).size().reset_index(name="Count")
    if len(_sb_df) > 0:
        fig_sb = px.sunburst(
            _sb_df, path=["risk_band","attack_type"], values="Count",
            color="risk_band", color_discrete_map=RISK_PALETTE, maxdepth=2,
        )
        fig_sb.update_layout(**chart_layout(460))
        fig_sb.update_traces(textfont_color="white", insidetextorientation="radial")
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.info("No data for sunburst chart with current filters.")

    divider()
    c1, c2 = st.columns(2)
    with c1:
        ct("Avg Probability by Region")
        avg_reg = (
            df_f.groupby("region")["attack_probability_pct"]
            .mean().sort_values().reset_index()
        )
        avg_reg.columns = ["Region", "Avg Prob (%)"]
        fig_ar = px.bar(
            avg_reg, x="Avg Prob (%)", y="Region", orientation="h",
            color="Avg Prob (%)",
            color_continuous_scale=[[0, KELP], [0.5, AMBER], [1, RUST]],
            text=avg_reg["Avg Prob (%)"].round(1).astype(str) + "%",
        )
        fig_ar.update_layout(**chart_layout(340), showlegend=False, coloraxis_showscale=False)
        fig_ar.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_ar, use_container_width=True)

    with c2:
        ct("Probability Trend by Year")
        yr_prob = df_f.groupby("year")["attack_probability_pct"].mean().reset_index()
        yr_prob.columns = ["Year","Avg Prob (%)"]
        fig_yrp = go.Figure(go.Scatter(
            x=yr_prob["Year"], y=yr_prob["Avg Prob (%)"],
            mode="lines+markers",
            line=dict(color=RUST, width=2.5),
            fill="tozeroy", fillcolor="rgba(192,57,43,0.1)",
            marker=dict(size=7),
        ))
        fig_yrp.update_layout(**chart_layout(340),
                              xaxis_title="Year", yaxis_title="Avg Prob (%)")
        st.plotly_chart(fig_yrp, use_container_width=True)

    divider()
    ct("Filtered Results (Top 50)")
    _show = [
        "year","month","region","nearest_country","attack_type",
        "vessel_status","shore_distance","attack_occurred",
        "attack_probability_pct","risk_band",
    ]
    _styled = df_f[_show].head(50).style.map(
        lambda v: (
            f"background-color: {RISK_PALETTE[v]}22; "
            f"color: {RISK_PALETTE[v]}; font-weight: 600;"
            if v in RISK_PALETTE else ""
        ),
        subset=["risk_band"],
    )
    st.dataframe(_styled, use_container_width=True, height=400)
    st.download_button(
        label="⬇  Download Results (CSV)",
        data=df_f[_show].to_csv(index=False).encode("utf-8"),
        file_name="attack_probability_results.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Insurance Premium Calculator
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    section("🛡️ Maritime Insurance Premium Calculator")
    st.markdown(
        '<p style="color:rgba(168,230,240,.5);font-size:.86rem;'
        'margin-top:-.5rem;margin-bottom:1.2rem;">'
        "Actuarially-grounded voyage premium using ML-predicted attack probability. "
        "Formula: <em>P = (p × V × LGD × λ<sub>region</sub> × (1+θ)) + V × r<sub>base</sub></em> "
        "— Bowers et al. (1997); Stopford (2009); ICC IMB (2020); Swiss Re (2011); IUMI (2019).</p>",
        unsafe_allow_html=True,
    )

    if "total_premium_usd" not in df_with_prob.columns:
        df_prem = append_premium_columns(df_with_prob)
    else:
        df_prem = df_with_prob

    divider()
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:.9rem;'
        'font-weight:700;color:#c8e8f5;margin-bottom:14px;">'
        'Single-Voyage Calculator</div>',
        unsafe_allow_html=True,
    )
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        calc_p = st.slider(
            "Attack probability (%)", 0.0, 100.0,
            float(df_prem["attack_probability_pct"].median()), step=0.5,
        )
        calc_region = st.selectbox(
            "Region", options=sorted(REGION_MULTIPLIERS.keys()),
            index=sorted(REGION_MULTIPLIERS.keys()).index("Somalia / East Africa")
            if "Somalia / East Africa" in REGION_MULTIPLIERS else 0,
        )
    with _c2:
        calc_value = st.number_input(
            "Insured value (USD M)", 0.5, 500.0,
            round(DEFAULT_INSURED_VALUE / 1_000_000, 1), step=0.5,
        ) * 1_000_000
        calc_lgd = st.slider(
            "Loss Given Default — LGD (%)", 5, 80,
            int(DEFAULT_LGD * 100), step=1,
        ) / 100.0
    with _c3:
        calc_theta = st.slider(
            "Safety loading — θ (%)", 0, 60,
            int(DEFAULT_THETA * 100), step=1,
        ) / 100.0
        calc_base_rate = st.slider(
            "Base rate (bps)", 1, 50,
            int(DEFAULT_BASE_RATE * 10_000), step=1,
        ) / 10_000.0

    prem_result = calculate_premium(
        p_attack=calc_p / 100.0,
        region=calc_region,
        insured_value=calc_value,
        lgd=calc_lgd,
        theta=calc_theta,
        base_rate=calc_base_rate,
    )
    divider()
    kpi_row([
        kpi_card("Total Premium",      f"${prem_result.total_premium_usd:,.0f}",  "blue"),
        kpi_card("Premium Rate",       f"{prem_result.premium_rate_pct:.4f}%",    "mid"),
        kpi_card("Expected Loss",      f"${prem_result.expected_loss_usd:,.0f}",  "amber"),
        kpi_card("Base Premium",       f"${prem_result.base_premium_usd:,.0f}",   "teal"),
        kpi_card("Region Multiplier",  f"×{prem_result.lambda_region:.2f}",       "rust"),
    ])

    divider()
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:.9rem;'
        'font-weight:700;color:#c8e8f5;margin-bottom:14px;">'
        'Portfolio Statistics (Full Dataset)</div>',
        unsafe_allow_html=True,
    )
    kpi_row([
        kpi_card("Avg Voyage Premium",
                 f"${df_prem['total_premium_usd'].mean():,.0f}", "blue",
                 f"σ = ${df_prem['total_premium_usd'].std():,.0f}"),
        kpi_card("Avg Premium Rate",
                 f"{df_prem['premium_rate_pct'].mean():.4f}%",  "mid",
                 f"max {df_prem['premium_rate_pct'].max():.4f}%"),
        kpi_card("Total Premium Pool",
                 f"${df_prem['total_premium_usd'].sum() / 1e9:.2f} B", "amber"),
        kpi_card("High/Critical Voyages",
                 f"{df_prem['risk_band'].isin(['High','Critical']).sum():,}",
                 "rust",
                 f"{100*df_prem['risk_band'].isin(['High','Critical']).mean():.1f}%"),
    ])

    divider()
    _r1c1, _r1c2 = st.columns(2)
    with _r1c1:
        ct("Avg Premium by Region")
        _reg_avg = (
            df_prem.groupby("region")["total_premium_usd"]
            .mean().reset_index().sort_values("total_premium_usd")
        )
        _reg_avg.columns = ["Region", "Avg Premium (USD)"]
        fig_rp = px.bar(
            _reg_avg, x="Avg Premium (USD)", y="Region", orientation="h",
            color="Avg Premium (USD)",
            color_continuous_scale=[[0, OCEAN], [0.5, OCEAN2], [1, SURF]],
            text=_reg_avg["Avg Premium (USD)"].apply(lambda v: f"${v:,.0f}"),
        )
        fig_rp.update_layout(**chart_layout(360), showlegend=False, coloraxis_showscale=False)
        fig_rp.update_traces(textposition="outside")
        st.plotly_chart(fig_rp, use_container_width=True)

    with _r1c2:
        ct("Premium Rate by Risk Band")
        fig_box = px.box(
            df_prem[df_prem["risk_band"].isin(BAND_ORDER)],
            x="risk_band", y="premium_rate_pct",
            category_orders={"risk_band": BAND_ORDER},
            color="risk_band", color_discrete_map=RISK_PALETTE,
            labels={"risk_band": "Risk Band", "premium_rate_pct": "Premium Rate (%)"},
        )
        fig_box.update_layout(**chart_layout(360), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    _r2c1, _r2c2 = st.columns(2)
    with _r2c1:
        ct("Attack Probability vs Total Premium")
        _samp = df_prem.sample(min(2000, len(df_prem)), random_state=42)
        fig_sc = px.scatter(
            _samp, x="attack_probability_pct", y="total_premium_usd",
            color="risk_band", opacity=0.6,
            category_orders={"risk_band": BAND_ORDER},
            color_discrete_map=RISK_PALETTE,
            labels={
                "attack_probability_pct": "Attack Probability (%)",
                "total_premium_usd": "Total Premium (USD)",
                "risk_band": "Risk Band",
            },
        )
        fig_sc.update_layout(**chart_layout(340))
        st.plotly_chart(fig_sc, use_container_width=True)

    with _r2c2:
        ct("Premium Distribution")
        fig_hist = px.histogram(
            df_prem, x="total_premium_usd", nbins=60,
            color_discrete_sequence=[OCEAN2],
            labels={"total_premium_usd": "Total Premium (USD)"},
        )
        fig_hist.update_layout(**chart_layout(340))
        st.plotly_chart(fig_hist, use_container_width=True)

    divider()
    _prem_cols = [
        "year", "month", "region", "nearest_country", "attack_type",
        "attack_probability_pct", "risk_band",
        "lambda_region", "expected_loss_usd",
        "risk_premium_usd", "base_premium_usd",
        "total_premium_usd", "premium_rate_pct",
    ]
    _prem_export = df_prem[[c for c in _prem_cols if c in df_prem.columns]]
    st.dataframe(
        _prem_export.head(50).style.map(
            lambda v: (
                f"background-color: {RISK_PALETTE[v]}22; "
                f"color: {RISK_PALETTE[v]}; font-weight: 600;"
                if v in RISK_PALETTE else ""
            ),
            subset=["risk_band"] if "risk_band" in _prem_export.columns else [],
        ),
        use_container_width=True, height=380,
    )
    st.download_button(
        label="⬇  Download Premium Data (CSV)",
        data=_prem_export.to_csv(index=False).encode("utf-8"),
        file_name="marine_guard_premiums.csv",
        mime="text/csv",
    )