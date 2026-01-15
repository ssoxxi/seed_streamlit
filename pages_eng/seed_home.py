# =============================================================================
# Home Page
# =============================================================================
from __future__ import annotations

import streamlit as st
from theme import apply_theme  # APP theme

# 1) set_page_config
st.set_page_config(
    page_title="Home | Team SEED Final Project",
    layout="wide",
    page_icon="🏠"
)

apply_theme()

# Head CSS
st.markdown(
    """
    <style>
    div.block-container { padding-top: 3.0rem !important; }
    header[data-testid="stHeader"] { height: 0px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Side Bar
# =============================================================================
if "sidebar_placeholder" in st.session_state:
    with st.session_state.sidebar_placeholder.container():
        st.header("👋 Welcome!")
        st.caption("Select a page from the left sidebar, or use the “Quick Start” button below.")
else:
    with st.sidebar:
        st.header("👋 Welcome!")
        st.caption("Select a page from the left sidebar!")

# =============================================================================
# Overview
# =============================================================================
st.markdown(
    """
    <div style="margin-top:0.2rem; margin-bottom:0.6rem;">
        <h1 style="margin-bottom:0.2rem;">🌱 Team SEED Final Project</h1>
        <div style="color:#6B7280; font-size:14px; line-height:1.6;">
            This is a decision-support dashboard that helps you quickly explore large-scale startup data 
            and <b>recommends priority candidates for initial review </b>
            through strategy simulations tailored to VC investment preferences.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Quick Start
# =============================================================================
st.divider()
st.subheader("🚀 Quick Start")

# Moving Pages Function
def go_to(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.info("Immediate navigation is restricted in the current environment. Please select a page from the sidebar.")

qs1, qs2, qs3 = st.columns([1.2, 1.2, 1.6])
with qs1:
    if hasattr(st, "page_link"):
        st.page_link("pages_eng/screening.py", label="Screening", icon="📊")
    else:
        st.button("📊 Screening", use_container_width=True, on_click=go_to, args=("pages_eng/screening.py",))
with qs2:
    if hasattr(st, "page_link"):
        st.page_link("pages_eng/recommendation.py", label="Simulator", icon="💰")
    else:
        st.button("💰 Simulator", use_container_width=True, on_click=go_to, args=("pages_eng/recommendation.py",))
with qs3:
    st.caption(
        """
        Recommended workflow:
        1) Narrow down the candidates in Startup Screening, then
        2) Compare strategy-specific results in Investment Strategy Simulation (success rate / time to exit / Top 10 recommendations).
        """
    )

# =============================================================================
# Short Introduction
# =============================================================================
st.divider()
st.subheader("📌What you can do in this app")
st.markdown(
        """
        <style>
        div[data-testid="stMetric"] label { font-size: 0.80rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Key Features", "2", help="Screening + Investment Strategy Simulation")
with c2:
    st.metric("Purpose", "Candidate Sourcing", help="Fast triage in the early-stage investment review process")
with c3:
    st.metric("Output", "Top 10 Recommendations", help="Provides a strategy-specific shortlist for priority review")

st.markdown(
    """
    - Based on **data analysis results and insights**, you can quickly apply commonly used investment review criteria (industry, country, stage, etc.).  
    - Using **ML-based prediction/recommendation results**, the app supports scenario comparison across strategies and helps set priorities.
    """
)

# =============================================================================
# Introduections with buttons
# =============================================================================
st.divider()
st.subheader("📑 Page Overview")

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 📊 Startup Screening")
        st.markdown(
            """
            **Purpose**: Source investment candidates and apply first-pass filtering  
            **How to use**: Select your criteria (industry, country, funding stage, etc.) → the follow-up review list is generated automatically  
            **Expected impact**: Explore large-scale data in seconds, shorten decision-prep time, and improve market-scanning efficiency
            """
        )
        if hasattr(st, "page_link"):
            st.page_link("pages_eng/screening.py", label="Go to the Screening page", icon="➡️")
        else:
            st.button("Go to the Screening page", use_container_width=True, on_click=go_to, args=("pages_eng/screening.py",))

with col2:
    with st.container(border=True):
        st.markdown("### 💰 Investment Strategy Simulation & Recommendations")
        st.markdown(
            """
            **Purpose**: Simulate expected success rates by VC investment strategy + recommend suitable startups  
            **How to use**: Choose industry / funding stage / VC investment preference → review expected success rate, average time to exit, and Top 10 recommendations  
            **Expected impact**: Speed up investment evaluation, strengthen decision rationale, compare scenarios by strategy, and set priorities
            """
        )
        if hasattr(st, "page_link"):
            st.page_link("pages_eng/recommendation.py", label="Go to the Simulation page", icon="➡️")
        else:
            st.button("Go to the Simulation page", use_container_width=True, on_click=go_to, args=("pages_eng/recommendation.py",))

# =============================================================================
# Team Introduction (Improved readability: columns instead of an info box)
# =============================================================================
st.divider()
st.header("🔄 About Team SEED")
st.markdown(
    """
    **Team name**: An abbreviation of *Startup Equity & Exit Data*, representing a team that works with startup equity and exit data.
    """
)

names = ["Suaa Bae", "Sohee Han", "Chaeyeon Hong", "Jaegyu Hwang"]
st.markdown(
    " ".join([
        f"<span style='display:inline-block; padding:6px 10px; margin:4px 6px 0 0; border-radius:999px; background:#F3F9FF; border:1px solid rgba(33,150,243,0.18);'>{n}</span>"
        for n in names
    ]),
    unsafe_allow_html=True
)
st.caption("Project period: 2025.11.24 ~ 2025.12.30")

# =============================================================================
# Getting Started
# =============================================================================
st.divider()
st.success("Get started using the left sidebar or the ‘Quick Start’ button.")
