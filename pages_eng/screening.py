# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from theme import apply_theme # 앱 theme

# =============================================================================
# 0) Page Configuration
# =============================================================================
st.cache_data.clear()
st.markdown(
    """
    <style>
    /* Fix top content being cut off: push the entire content downward */
    div.block-container{
        padding-top: 3.0rem !important;   /* Adjust between 2.5–4.0 if needed */
    }

    /* (When hiding the header) keep safe spacing even if the header is removed */
    header[data-testid="stHeader"]{
        height: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Analysis Process | Startup Screening", layout="wide", page_icon="📊")
st.title("📊 Startup Screening")
st.markdown(
    """
    <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
    ※ This is a first-pass screening tool designed to quickly shortlist startups that match key criteria (industry, country, funding stage, etc.) during the early phase of VC investment review, and to narrow down companies for further evaluation.
    </div>
    """,
    unsafe_allow_html=True
)

apply_theme()

# =============================================================================
# 1) Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]  # Adjust to match your project structure
DATA_DIR = BASE_DIR / "data"

SUCCESS_PATH = DATA_DIR / "s_master_distinct_startups.csv"
CLUSTER_PATH = DATA_DIR / "startup_ver.csv"
SHAP_PATH = DATA_DIR / "shap_local.csv"


# =============================================================================
# 2) Cluster (Startup Type) Labels (for display)
# =============================================================================
CLUSTER_LABEL = {
    0: "Early-Stage Experimental Startups",
    1: "Network-Driven Growth Startups",
    2: "Domain-Specialized Stable Startups",
    3: "Elite-Founder-Led Tech Startups",
    4: "Research-Driven Long-Term Growth Startups",
}


# =============================================================================
# 3) Loader
# =============================================================================
NEEDED_COLS = [
    "objects_cfpr_id",
    "founded_at",
    "country_code",
    "obj_city_fixed",
    "obj_category_filled",

    "funding_round_id",
    "funded_at",
    "raised_amount_usd",
    "is_first_round",
    "is_last_round",
    "funding_rounds",
    "funding_total_usd",
    "relationships",
    "round_tempo_months",

    "cat_fr_type",
    "num_fr_type",

    "acquisition_id",
    "acquired_at",
    "acquired_c_id",

    "ipo_id",
    "first_public_at",
    "ipos_c_id",

    "success_flag",
    "n_offices",
]

@st.cache_data(show_spinner=False)
def load_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if usecols is None:
        return pd.read_csv(path, low_memory=False)
    wanted = set(usecols)
    return pd.read_csv(path, low_memory=False, usecols=lambda c: c in wanted)

@st.cache_data(show_spinner=False)
def load_success_master(path: Path) -> pd.DataFrame:
    df = load_csv(path, usecols=NEEDED_COLS).copy()
    if df.empty:
        return df

    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)

    # Date
    for c in ["founded_at", "funded_at", "acquired_at", "first_public_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Numeric
    for c in ["raised_amount_usd", "funding_total_usd", "relationships", "num_fr_type", "funding_round_id", "funding_rounds", "n_offices"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["is_first_round", "is_last_round", "success_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def load_startup_ver(path: Path) -> pd.DataFrame:
    df = load_csv(path).copy()
    if df.empty:
        return df
    if "objects_cfpr_id" in df.columns:
        df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def load_shap_local(path: Path) -> pd.DataFrame:
    df = load_csv(path).copy()
    if df.empty:
        return df
    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)
    # 숫자형으로 변환(안 되는 건 NaN)
    for c in df.columns:
        if c != "objects_cfpr_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =============================================================================
# 4) Score Util
# =============================================================================
def winsor(s: pd.Series, p_lo=0.01, p_hi=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(p_lo)
    hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

def minmax_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = np.nanmin(s.values) if np.isfinite(np.nanmin(s.values)) else np.nan
    mx = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else np.nan
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(np.zeros(len(s)), index=s.index)


# =============================================================================
# 5) Build a company-level master table
# =============================================================================
@st.cache_data(show_spinner=True)
def build_company_master(success_path: Path, startup_ver_path: Path) -> pd.DataFrame:
    df = load_csv(success_path).copy()
    if df.empty:
        return df

    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)

    # ------------------------------------------------------------------
    # [CASE 1] If this is already a company-level dataset (1 row = 1 startup),
    #          e.g., an aggregated file like s_master_distinct.csv
    # ------------------------------------------------------------------
    is_distinct = {"industry", "country", "founded_year", "total_rounds", "invest_stage_last"}.issubset(df.columns)

    if is_distinct:
        # Standardize column names (to match what the dashboard expects)
        rename_map = {
            "industry": "obj_category_filled",
            "country": "country_code",
            "total_rounds": "round_cnt",
            "total_funding_usd": "funding_total_usd",
            "first_round_raised_usd": "first_raised_usd",
            "last_round_raised_usd": "last_raised_usd",
            "invest_stage_last": "cat_fr_type",
        }
        df_company = df.rename(columns=rename_map).copy()

        # Type cleanup
        for c in ["founded_year", "round_cnt", "funding_total_usd", "relationships",
                  "first_raised_usd", "last_raised_usd", "success_flag"]:
            if c in df_company.columns:
                df_company[c] = pd.to_numeric(df_company[c], errors="coerce")

        # If IPO/M&A already exist as 0/1, keep them; otherwise default to 0
        if "ipo_achieved" not in df_company.columns:
            df_company["ipo_achieved"] = 0
        if "mna_achieved" not in df_company.columns:
            df_company["mna_achieved"] = 0
        df_company["ipo_achieved"] = pd.to_numeric(df_company["ipo_achieved"], errors="coerce").fillna(0).astype(int)
        df_company["mna_achieved"] = pd.to_numeric(df_company["mna_achieved"], errors="coerce").fillna(0).astype(int)

        # Compute funding growth rate
        fr = pd.to_numeric(df_company.get("first_raised_usd"), errors="coerce")
        lr = pd.to_numeric(df_company.get("last_raised_usd"), errors="coerce")
        df_company["funding_growth_rate"] = np.where(
            (fr > 0) & np.isfinite(fr) & np.isfinite(lr),
            (lr - fr) / fr,
            np.nan
        )

        # Columns that may be missing (provide defaults if the UI expects them)
        if "obj_city_fixed" not in df_company.columns:
            df_company["obj_city_fixed"] = pd.NA
        if "n_offices" not in df_company.columns:
            df_company["n_offices"] = pd.NA

    # ------------------------------------------------------------------
    # [CASE 2] If this is the raw round-level success_master.csv:
    #          run the existing aggregation logic
    # ------------------------------------------------------------------
    else:
        # Create founded_year only if founded_at exists
        if "founded_at" in df.columns:
            df["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")
            df["founded_year"] = df["founded_at"].dt.year
        else:
            df["founded_year"] = pd.NA

        # Below: keep your existing logic as-is, but make sure any direct references
        # (e.g., df["founded_at"]) are guarded by column existence checks for safety.
        # (Omitted here: this branch is only used when running on the raw source file.)

        # --- At minimum, ensure the final output is stored in df_company ---
        # If you only need it for the raw source case, move your existing
        # build_company_master logic into this branch.
        df_company = df.groupby("objects_cfpr_id", as_index=False).first()

    # ------------------------------------------------------------------
    # cluster merge
    # ------------------------------------------------------------------
    sv = load_startup_ver(startup_ver_path)
    if (not sv.empty) and {"objects_cfpr_id", "cluster"}.issubset(sv.columns):
        df_company = df_company.merge(sv[["objects_cfpr_id", "cluster"]], on="objects_cfpr_id", how="left")
    else:
        df_company["cluster"] = pd.Series([pd.NA] * len(df_company), dtype="Int64")

    # Category optimization
    for c in ["obj_category_filled", "country_code", "cat_fr_type", "obj_city_fixed"]:
        if c in df_company.columns:
            df_company[c] = df_company[c].astype("category")

    return df_company

# =============================================================================
# 6) Load Data
# =============================================================================
if not SUCCESS_PATH.exists():
    st.error(f"Data file not found: {SUCCESS_PATH}")
    st.stop()

df_company = build_company_master(SUCCESS_PATH, CLUSTER_PATH)

startup_ver = load_startup_ver(CLUSTER_PATH) if CLUSTER_PATH.exists() else pd.DataFrame()
shap_local  = load_shap_local(SHAP_PATH) if SHAP_PATH.exists() else pd.DataFrame()

# =============================================================================
# 7) Left/Right Layout
# =============================================================================
col_left, col_right = st.columns([1.1, 3.2], gap="large")


# =============================================================================
# 8) Top-left: Filter Panel → Set Investment Criteria
# =============================================================================
with col_left:
    st.subheader("Set Investment Criteria")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ Select the startup types you are interested in to narrow down the candidate pool.
        </div>
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ <b>What is a startup cluster?</b> A group of startups with similar strategies, stages, and capabilities.<br> 
        Depending on the startup type you select, you can find companies that match your preferred style.
        </div>
        """,
        unsafe_allow_html=True
    )

    # (1) Cluster options
    cluster_options = []
    if (not startup_ver.empty) and ("cluster" in startup_ver.columns):
        cluster_ids = startup_ver["cluster"].dropna().astype(int).unique().tolist()
        cluster_ids = sorted(cluster_ids)
        cluster_options = [f"{cid} | {CLUSTER_LABEL.get(cid, f'Cluster {cid}')}" for cid in cluster_ids]

    # (2) Other options/ranges
    industry_opts = list(df_company["obj_category_filled"].cat.categories) if "obj_category_filled" in df_company.columns else []
    country_opts  = list(df_company["country_code"].cat.categories) if "country_code" in df_company.columns else []
    round_opts    = list(df_company["cat_fr_type"].cat.categories) if "cat_fr_type" in df_company.columns else []

    y_min = int(df_company["founded_year"].dropna().min()) if df_company["founded_year"].notna().any() else 1990
    y_max = int(df_company["founded_year"].dropna().max()) if df_company["founded_year"].notna().any() else 2025
    # s_min = float(df_company["success_score"].min()) if df_company["success_score"].notna().any() else 0.0
    # s_max = float(df_company["success_score"].max()) if df_company["success_score"].notna().any() else 100.0

    # Ensure defaults before creating widgets (first run only)
    st.session_state.setdefault("f_cluster_label", [])
    st.session_state.setdefault("f_industry", [])
    st.session_state.setdefault("f_country", [])
    st.session_state.setdefault("f_round", ["seed"])
    st.session_state.setdefault("f_year", (y_min, y_max))
    # st.session_state.setdefault("f_score", (float(s_min), float(s_max)))

    # Reset callback (only modify session_state via button on_click)
    def reset_filters():
        st.session_state["f_cluster_label"] = []
        st.session_state["f_industry"] = []
        st.session_state["f_country"] = []
        st.session_state["f_round"] = []
        st.session_state["f_year"] = (y_min, y_max)
        # st.session_state["f_score"] = (float(s_min), float(s_max)))

    # ---- Widgets ----
    sel_cluster_label = st.multiselect(
        "Select Startup Type",
        options=cluster_options,
        key="f_cluster_label",
    )
    sel_cluster_ids = [int(x.split("|")[0].strip()) for x in sel_cluster_label] if sel_cluster_label else []

    st.multiselect("Select Industry", options=industry_opts, key="f_industry")
    st.multiselect("Select Country", options=country_opts, key="f_country")
    st.multiselect("Funding Stage", options=round_opts, key="f_round")

    # ✅ Set initial f_year only if it doesn't exist
    if "f_year" not in st.session_state:
        st.session_state["f_year"] = (y_min, y_max)

    # ✅ Create slider: if session_state already has a value, do not pass value= (avoids warnings)
    if "f_year" in st.session_state:
        st.slider(
            "Year Founded",
            min_value=y_min,
            max_value=y_max,
            step=1,
            key="f_year",
        )
    else:
        st.slider(
            "Year Founded",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max),
            step=1,
            key="f_year",
        )

    # Use on_click (no need for st.rerun())
    st.button("Reset Filters", use_container_width=True, on_click=reset_filters)

# =============================================================================
# 9) Apply Filters
# =============================================================================
def apply_filters(df_: pd.DataFrame, cluster_ids: list[int]) -> pd.DataFrame:
    out = df_.copy()

    if cluster_ids and "cluster" in out.columns:
        out = out[out["cluster"].isin(cluster_ids)]

    if st.session_state.get("f_industry"):
        out = out[out["obj_category_filled"].isin(st.session_state["f_industry"])]

    if st.session_state.get("f_country"):
        out = out[out["country_code"].isin(st.session_state["f_country"])]

    if st.session_state.get("f_round"):
        out = out[out["cat_fr_type"].isin(st.session_state["f_round"])]

    y0, y1 = st.session_state.get("f_year", (y_min, y_max))
    out = out[out["founded_year"].between(y0, y1, inclusive="both") | out["founded_year"].isna()]

    # s0, s1 = st.session_state.get("f_score", (s_min, s_max))
    # out = out[out["success_score"].between(s0, s1, inclusive="both") | out["success_score"].isna()]

    return out

filtered = apply_filters(df_company, sel_cluster_ids)


# =============================================================================
# 11) Right side: KPIs + List + Details + Breakdown (SHAP)
# =============================================================================
with col_right:
            
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Companies", f"{len(df_company):,}")  # Added unit formatting
    with k2:
        st.metric("Companies After Filters", f"{len(filtered):,}")  # Added unit formatting
    with k3:
        ratio = (len(filtered) / len(df_company) * 100) if len(df_company) else 0
        st.metric("Filter Ratio", f"{ratio:.1f}%")

    st.divider()
    st.subheader("Startup List")

    # Search (optional)
    q = st.text_input("Search: Match by Company ID", value="", placeholder="e.g., c:10001")
    table_view = filtered.copy()
    if q.strip():
        table_view = table_view[table_view["objects_cfpr_id"].str.contains(q.strip(), case=False, na=False)]

    # show_cols = ["objects_cfpr_id", "obj_category_filled", "country_code", "cat_fr_type", "round_cnt", "success_score", "cluster"]
    show_cols = ["objects_cfpr_id", "obj_category_filled", "country_code", "cat_fr_type", "round_cnt", "cluster"]
    show_cols = [c for c in show_cols if c in table_view.columns]

    top_n = st.slider("Rows to Display", 50, 500, 200, step=50)
    table_df = table_view[show_cols].sort_values("cat_fr_type", ascending=False).head(top_n).copy()

    # Add cluster_label column for display (if available)
    if "cluster" in table_df.columns:
        table_df["cluster_label"] = table_df["cluster"].map(
            lambda x: f"{int(x)} | {CLUSTER_LABEL.get(int(x), f'Cluster {int(x)}')}" if pd.notna(x) else "—"
        )
        # Adjust display order
        display_cols = [c for c in show_cols if c != "cluster"] + ["cluster_label"]
    else:
        display_cols = show_cols

    # Column display names (EN)
    column_config = {
        "objects_cfpr_id": st.column_config.TextColumn("Company ID"),
        "obj_category_filled": st.column_config.TextColumn("Industry"),
        "country_code": st.column_config.TextColumn("Country"),
        "cat_fr_type": st.column_config.TextColumn("Funding Stage"),
        "round_cnt": st.column_config.NumberColumn("Total Rounds", format="%.0f"),
        # "success_score": st.column_config.NumberColumn("Success Score", format="%.1f"),
        "cluster_label": st.column_config.TextColumn("Startup Type"),
    }

    st.dataframe(table_df[display_cols], use_container_width=True, hide_index=True, column_config=column_config)

    st.subheader("Selected Company Details")

    # Selection options (based on the table)
    options = table_df["objects_cfpr_id"].dropna().astype(str).unique().tolist()
    if not options:
        st.info("No results match the current filters.")
        st.stop()

    STATE_KEY = "screening_selected_company"
    if (STATE_KEY not in st.session_state) or (st.session_state[STATE_KEY] not in options):
        st.session_state[STATE_KEY] = options[0]

    selected_id = st.selectbox("Select a Company", options=options, index=options.index(st.session_state[STATE_KEY]))
    st.session_state[STATE_KEY] = selected_id

    lookup = filtered.set_index("objects_cfpr_id", drop=False)
    row = lookup.loc[selected_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    # Top summary card
    cA, cB, cC, cD, cE = st.columns([2.4, 1, 1, 1, 1.2])
    with cA:
        st.markdown(f"### {row['objects_cfpr_id']}")
        st.write(
            f"Industry: {row.get('obj_category_filled', '—')} | "
            f"Country: {row.get('country_code', '—')} | "
            f"Stage: {row.get('cat_fr_type', '—')}"
        )
    with cB:
        st.metric("Founded Year", "—" if pd.isna(row.get("founded_year")) else int(row["founded_year"]))
    with cC:
        st.metric("Rounds", "—" if pd.isna(row.get("round_cnt")) else int(row["round_cnt"]))
    # with cD:
    #     st.metric("Success Score", f"{row.get('success_score', np.nan):.1f}")
    with cD:
        if pd.notna(row.get("cluster")):
            cid = int(row["cluster"])
            cl_txt = f"{cid} | {CLUSTER_LABEL.get(cid, f'Cluster {cid}')}"
        else:
            cl_txt = "—"

        # Smaller font + ellipsis for cluster text (prevents clipping)
        st.markdown(
            f"""
            <div style="text-align:center;">
            <div style="font-size:0.85rem; color: rgba(0,0,0,0.6); margin-bottom:0.25rem;">
                Startup Type
            </div>
            <div style="
                    font-size:14px;
                    font-weight:700;
                    line-height:1.2;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;">
                {cl_txt}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    growth_pct = row.get("funding_growth_rate")
    growth_txt = "Not Disclosed" if pd.isna(growth_pct) else f"{growth_pct*100:.1f}%"

    # Tabs: Summary / SHAP
    # tab1, tab2 = st.tabs(["Summary", "Key Drivers"])
    tabs = st.tabs(["Summary"])

    # =========================
    # ✅ Summary tab: convert NaN → "Not Disclosed" + unify formatting
    # =========================

    def _is_na(x) -> bool:
        try:
            return pd.isna(x)
        except Exception:
            return x is None

    def fmt_int(x) -> str:
        """Return 'Not Disclosed' if NaN; otherwise show as an integer."""
        if _is_na(x):
            return "Not Disclosed"
        try:
            return str(int(float(x)))
        except Exception:
            return str(x)

    def fmt_usd(x, digits=0, suffix=" USD") -> str:
        """Return 'Not Disclosed' if NaN; otherwise format with thousands separators + optional suffix."""
        if _is_na(x):
            return "Not Disclosed"
        try:
            x = float(x)
            s = f"{x:,.{digits}f}" if digits > 0 else f"{x:,.0f}"
            return s + suffix
        except Exception:
            return str(x)

    # ---- Precompute display text for the Summary tab ----
    ipo_txt = fmt_int(row.get("ipo_achieved", pd.NA))
    mna_txt = fmt_int(row.get("mna_achieved", pd.NA))

    first_raised_txt = fmt_usd(row.get("first_raised_usd", pd.NA))
    last_raised_txt  = fmt_usd(row.get("last_raised_usd", pd.NA))
    total_funding_txt = fmt_usd(row.get("funding_total_usd", pd.NA))

    rel_txt = fmt_int(row.get("relationships", pd.NA))
    success_txt = fmt_int(row.get("success_flag", pd.NA))

    # =========================
    #  SHAP key drivers: map feature names to labels
    # =========================

    SHAP_FEATURE_ENG = {
        "is_n_offices_missing": "Missing office count flag",
        "is_city_missing": "Missing city information flag",
        "subject_group": "Major group",
        "degree_level_filled": "Highest degree level (imputed)",
        "is_inst_missing": "Missing school information flag",
        "n_offices": "Number of offices",
        "inst_tier": "Institution tier",
        "log1p_n_founding": "Number of founding experiences (log1p)",
        "reinvest_rate_next": "Reinvestment rate in the next round",
        "city_grouped": "City group",
        "is_n_founding_missing": "Missing founding experience flag",
        "log1p_time_to_first_funding_days": "Time to first funding (log1p)",
        "first_participants": "Number of investors in the first round",
        "log1p_first_raised_amount": "First round amount raised (log1p)",
        "relationships": "Network size (relationships)",
    }

    with tabs[0]:
        st.write(
            f"- IPO Achieved: {ipo_txt}\n"
            f"- M&A Achieved: {mna_txt}\n"
            f"- Funding Growth Rate (First → Last): {growth_txt}\n"
            f"- First Round Raised (USD): {first_raised_txt}\n"
            f"- Last Round Raised (USD): {last_raised_txt}\n"
            f"- Total Funding (USD): {total_funding_txt}\n"
            f"- Network Size (relationships): {rel_txt}\n"
            f"- Success Flag (success_flag): {success_txt}"
        )

    # with tab2:
    #     # Breakdown based on shap_local.csv
    #     if shap_local.empty:
    #         st.info("Cannot display model contributions because shap_local.csv is not available.")
    #     else:
    #         # Prevent ID type mismatch
    #         shap_local["objects_cfpr_id"] = shap_local["objects_cfpr_id"].astype(str)
    #         sid = str(selected_id)

    #         shap_df = shap_local.loc[shap_local["objects_cfpr_id"] == sid].copy()

    #         if shap_df.empty:
    #             st.info("No SHAP information is available for the selected company.")
    #         else:
    #             # Wide (columns=feature) → long (feature, shap_value)
    #             shap_mat = shap_df.drop(columns=["objects_cfpr_id"], errors="ignore")

    #             # If multiple rows exist for the same company, use the mean as the representative value
    #             shap_mat = shap_mat.apply(pd.to_numeric, errors="coerce")
    #             shap_vec = shap_mat.mean(axis=0) if len(shap_mat) > 1 else shap_mat.iloc[0]

    #             shap_long = (
    #                 shap_vec.rename("shap_value")
    #                         .reset_index()
    #                         .rename(columns={"index": "feature"})
    #             )

    #             # Safety: if shap_value is missing, create it explicitly
    #             if "shap_value" not in shap_long.columns:
    #                 # Handle cases where reset_index yields ['index', 0]
    #                 if 0 in shap_long.columns:
    #                     shap_long = shap_long.rename(columns={0: "shap_value"})
    #                 else:
    #                     st.error("Failed to parse SHAP data columns. Please check the shap_local.csv structure.")
    #                     st.stop()

    #             shap_long["shap_value"] = pd.to_numeric(shap_long["shap_value"], errors="coerce")
    #             shap_long = shap_long.dropna(subset=["shap_value"])

    #             if shap_long.empty:
    #                 st.info("All SHAP values for the selected company are missing.")
    #             else:
    #                 shap_long["abs"] = shap_long["shap_value"].abs()
    #                 shap_long = shap_long.sort_values("abs", ascending=False).head(15)

    #                 st.caption("※ Positive values increase the predicted probability of success, negative values decrease it. Longer bars indicate stronger influence (use as priority review signals).")

    #                 shap_long["feature_eng"] = shap_long["feature"].map(SHAP_FEATURE_ENG).fillna(shap_long["feature"])
                    
    #                 fig = px.bar(
    #                     shap_long.sort_values("shap_value"),
    #                     x="shap_value",
    #                     y="feature_eng",
    #                     orientation="h",
    #                 )
    #                 fig.update_layout(
    #                     margin=dict(l=10, r=10, t=30, b=10),
    #                     xaxis_title="SHAP value",
    #                     yaxis_title="",
    #                 )
    #                 st.plotly_chart(fig, use_container_width=True)
