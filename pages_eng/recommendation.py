import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from theme import apply_theme  # app theme

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

# st.set_page_config(
#     page_title="Investment Strategy Simulation",
#     layout="wide",
# )
st.cache_data.clear()
# st.title("💰 Investment Strategy Simulation")

apply_theme()


# =========================
# (Added) VC cluster display names
# =========================
VC_CLUSTER_NAMES = {
    0: "Global Early-Stage VC",
    1: "Late-Stage Scale-up VC",
    2: "Early-to-Mid Stage VC",
    3: "Finance-Oriented Conservative VC",
    4: "Seed-Focused VC",
    5: "Growth-Validation Stage VC",
}

# =========================
# 0) Paths / loader
# =========================
ROOT = Path(__file__).resolve().parents[1]  # assume project root (located under pages_eng/)
DATA_DIR = ROOT / "data"

STARTUP_PATH = DATA_DIR / "startup_base.csv"
VC_PATH = DATA_DIR / "vc_base.csv"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# =========================
# 1) VC cluster rules (example: replace with your team’s final rules)
# =========================
VC_FILTER_RULES = {
    0: {  # Global diversified early VC
        # Early-stage: allow NaN + 0~1
        "num_fr_type_in": [0, 1, None],

        "n_founding_min": 1,
        "time_to_first_funding_days_max": 600,

        # Minimal preference matching
        "match_category": False,
        "match_city": False,
        "match_inst": False,
        "degree_min": None
    },

    1: {  # Late-stage scale-up VC
        "num_fr_type_in": [3, 4, 99],   # late stage only

        "relationships_min": 5,
        "reinvest_rate_next_min": 0.4,

        # Strong preference matching
        "match_category": True,
        "match_city": True,
        "match_inst": False,
        "degree_min": None
    },

    2: {  # Early-to-mid stage VC
        "num_fr_type_in": [0, 1, 2, None],

        "match_category": True,
        "match_city": False,
        "match_inst": False,
        "degree_min": 2   # Bachelor's or higher
    },

    3: {  # Finance-oriented conservative VC
        "num_fr_type_in": [2, 3, 4, 99],

        "relationships_min": 2,

        "match_category": False,
        "match_city": False,
        "match_inst": False,
        "degree_min": None
    },

    4: {  # Seed-focused VC
        "num_fr_type_in": [0, None],

        "n_founding_min": 2,

        "match_category": False,
        "match_city": False,
        # "match_inst": True,   # institution matters
        "degree_min": 2
    },

    5: {  # Growth-validation stage VC
        "num_fr_type_in": [1, 2],

        "relationships_min": 4,

        "match_category": True,
        "match_city": True,
        "match_inst": False,
        "degree_min": 2
    }
}


def apply_vc_filter(for_streamlit: pd.DataFrame, vc_row: pd.Series, vc_cluster: int) -> pd.DataFrame:
    rules = VC_FILTER_RULES[vc_cluster]
    df = for_streamlit.copy()

    def is_valid_pref(x):
        if x is None or pd.isna(x):
            return False
        s = str(x).strip().lower()
        return s not in {"", "unknown", "nan", "none", "<na>"}

    # 1) Round (stage) filter
    if "num_fr_type_in" in rules and "num_fr_type" in df.columns:
        allowed = rules["num_fr_type_in"]
        allow_nan = (None in allowed)
        allowed_vals = [x for x in allowed if x is not None]

        mask = df["num_fr_type"].isin(allowed_vals)
        if allow_nan:
            mask = mask | df["num_fr_type"].isna()
        df = df[mask]

    # 2) Numeric min/max conditions
    if "n_founding_min" in rules and "n_founding" in df.columns:
        df = df[df["n_founding"] >= rules["n_founding_min"]]

    if "relationships_min" in rules and "relationships" in df.columns:
        df = df[df["relationships"] >= rules["relationships_min"]]

    if "first_raised_amount_min" in rules and "first_raised_amount" in df.columns:
        df = df[df["first_raised_amount"] >= rules["first_raised_amount_min"]]

    if "first_participants_min" in rules and "first_participants" in df.columns:
        df = df[df["first_participants"] >= rules["first_participants_min"]]

    if "reinvest_rate_next_min" in rules and "reinvest_rate_next" in df.columns:
        df = df[df["reinvest_rate_next"] >= rules["reinvest_rate_next_min"]]

    if "time_to_first_funding_days_max" in rules and "time_to_first_funding_days" in df.columns:
        df = df[df["time_to_first_funding_days"] <= rules["time_to_first_funding_days_max"]]

    # 3) Match VC preferences
    match_category = rules.get("match_category_to_vc") or rules.get("match_category")
    match_city = rules.get("match_city_to_vc") or rules.get("match_city")
    match_inst = rules.get("match_inst_to_vc") or rules.get("match_inst")

    if match_category and "category" in df.columns and "startup_industry_top1" in vc_row.index:
        pref = vc_row["startup_industry_top1"]
        if is_valid_pref(pref):
            df = df[df["category"] == pref]

    if match_city and "city" in df.columns and "startup_city_top1" in vc_row.index:
        pref = vc_row["startup_city_top1"]
        if is_valid_pref(pref):
            df = df[df["city"] == pref]

    if match_inst and "inst" in df.columns and "founder_institution_top1" in vc_row.index:
        pref = vc_row["founder_institution_top1"]
        if is_valid_pref(pref):
            df = df[df["inst"] == pref]

    # 4) Degree matching
    degree_rule = rules.get("degree_min_from_vc_mean")
    if degree_rule and "degree_level_filled" in df.columns and "founder_degree_level_mean" in vc_row.index:
        deg_thr = vc_row["founder_degree_level_mean"]
        if pd.notna(deg_thr):
            df = df[df["degree_level_filled"] >= deg_thr]

    return df


# =========================
# 2) Build representative profiles by VC cluster
# =========================
def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().idxmax()


@st.cache_data(show_spinner=False)
def build_vc_cluster_profile(vc_df: pd.DataFrame) -> pd.DataFrame:
    grp = vc_df.groupby("cluster", dropna=False)

    prof = grp.agg(
        cluster_size=("investor_cfp_id", "count"),
        startup_industry_top1=("startup_industry_top1", _mode_or_nan),
        startup_city_top1=("startup_city_top1", _mode_or_nan),
        founder_institution_top1=("founder_institution_top1", _mode_or_nan),
        founder_degree_level_mean=("founder_degree_level_mean", "mean"),
    ).reset_index()

    return prof


# =========================
# 3) Startup stage buckets (based on current data)
# =========================
def stage_bucket_from_num_fr_type(x):
    if pd.isna(x):
        return "Unknown"
    try:
        x = float(x)
    except Exception:
        return "Unknown"

    if x == 99:
        return "Unknown"
    if x <= 1:
        return "Early"
    if x == 2:
        return "Mid"
    return "Late"


# =========================
# 4) UI
# =========================
startup_df = load_csv(STARTUP_PATH)
vc_df = load_csv(VC_PATH)
vc_prof = build_vc_cluster_profile(vc_df)

# Derived stage bucket (for display)
if "num_fr_type" in startup_df.columns:
    startup_df["stage_bucket"] = startup_df["num_fr_type"].apply(stage_bucket_from_num_fr_type)
else:
    startup_df["stage_bucket"] = "Unknown"

market_avg = float(startup_df["success_prob"].mean()) if "success_prob" in startup_df.columns else np.nan

# ---- Header
header_l, header_r = st.columns([4, 1])
with header_l:
    st.markdown("## 💰 Investment Strategy Simulation & Startup Recommendations")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ A decision-support tool that simulates expected success rates by VC investment strategy and recommends suitable startups.
        </div>
        """,
        unsafe_allow_html=True
    )

with header_r:
    # Show selectbox options as "0: Name"
    available = sorted([int(x) for x in vc_prof["cluster"].dropna().unique()])
    cluster_list = [c for c in VC_CLUSTER_NAMES.keys() if c in available] or available

    vc_cluster = st.selectbox(
        "VC Type",
        options=cluster_list,
        index=0,
        format_func=lambda c: f"{c}: {VC_CLUSTER_NAMES.get(int(c), 'Unknown')}",
        key="vc_cluster",
    )

# Selected cluster representative row (with safeguards)
sel = vc_prof.loc[vc_prof["cluster"] == vc_cluster]
if sel.empty:
    st.warning("The selected type does not exist in vc_base.csv. Please verify the data and type labels.")
    vc_row = pd.Series(dtype="object")
else:
    vc_row = sel.iloc[0]

st.divider()

# ---- Top 2 columns (left: parameters, right: results)
left, right = st.columns([1.2, 2.8])

with left:
    st.markdown("### Simulation Criteria")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ Selecting the conditions below will update the recommendations for startups that fit the strategy.<br>
        <b>※ Depending on your selections, there may be no results.</b><br>
        </div>
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ <b>What is a VC cluster?</b> A group of VCs clustered by similar investment preferences.<br>
        Settings are based on “startup characteristics that VCs of this type actually preferred.”
        </div>
        """,
        unsafe_allow_html=True
    )

    # use_vc_rules = st.toggle("Apply VC cluster rules", value=True)

    ind_col = "category" if "category" in startup_df.columns else None
    if ind_col:
        inds = sorted([x for x in startup_df[ind_col].dropna().unique()])
        sel_inds = st.multiselect("Industry (multi-select)", inds, default=[], key="sel_inds")
    else:
        sel_inds = []

    stages = sorted(startup_df["stage_bucket"].dropna().unique())
    sel_stages = st.multiselect("Funding Stage", stages, default=[], key="sel_stages")

    if "region" in startup_df.columns:
        regions = sorted([x for x in startup_df["region"].dropna().unique()])
        sel_regions = st.multiselect("Region", regions, default=[], key="sel_regions")
    else:
        sel_regions = []

    # “Reset filters” button
    def reset_filters(default_cluster: int):
        st.session_state["vc_cluster"] = default_cluster
        st.session_state["sel_inds"] = []
        st.session_state["sel_stages"] = []
        st.session_state["sel_regions"] = []

    default_cluster = cluster_list[0] if len(cluster_list) else 0

    st.button(
        "Reset Criteria",
        on_click=reset_filters,
        args=(default_cluster,),
        width="stretch",   # 대응: use_container_width warning
    )


with right:
    st.markdown("### Simulation Results")

    df = startup_df.copy()

    # Always apply VC cluster rules
    try:
        df = apply_vc_filter(df, vc_row, int(vc_cluster))
    except KeyError:
        st.error(f"No rule found in VC_FILTER_RULES for cluster={vc_cluster}. Please add it to the dictionary.")
        st.stop()
    except Exception as e:
        st.error(f"Error while applying VC rules: {e}")
        st.stop()

    if ind_col and sel_inds:
        df = df[df[ind_col].isin(sel_inds)]

    if sel_stages:
        df = df[df["stage_bucket"].isin(sel_stages)]

    if "region" in df.columns and sel_regions:
        df = df[df["region"].isin(sel_regions)]

    if len(df) == 0:
        st.warning("No results found. Please broaden your criteria.")
    else:
        exit_rate = float(df["success_prob"].mean()) if "success_prob" in df.columns else np.nan
        delta_pp = (exit_rate - market_avg) * 100 if (np.isfinite(exit_rate) and np.isfinite(market_avg)) else np.nan

        sel_cluster = vc_prof.loc[vc_prof["cluster"] == vc_cluster]
        if not sel_cluster.empty:
            cluster_name = VC_CLUSTER_NAMES.get(int(vc_cluster), "Unknown")
        else:
            cluster_name = "Unknown"

        if "exit_years_pred" in df.columns:
            hold_years = float(df["exit_years_pred"].mean())
            hold_label = "Expected Avg. Time to Exit"
        elif "time_to_first_funding_days" in df.columns:
            hold_years = float(df["time_to_first_funding_days"].mean()) / 365.0
            hold_label = "Expected Avg. Time to First Funding"
        else:
            hold_years = np.nan
            hold_label = "Expected Avg. Duration"

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            cluster_label = VC_CLUSTER_NAMES.get(int(vc_cluster), "Unknown")
            short = cluster_label.replace("VC", "").strip()
            st.metric("VC Type", f"{vc_cluster}: {short}")
        with kpi2:
            st.metric(
                "Expected Investment Success Rate",
                f"{exit_rate*100:.1f}%",
                delta=(f"{delta_pp:+.1f}%p" if np.isfinite(delta_pp) else None),
            )
        with kpi3:
            st.metric(hold_label, (f"{hold_years:.1f} years" if np.isfinite(hold_years) else "N/A"))

        st.markdown("### Top 10 Recommended Startups")
        st.markdown(
            """
            <div style="margin-top:-6px; margin-bottom:10px; color:#8a8a8a; font-size:12.5px; line-height:1.5;">
            <b>[Funding Stage]</b><br>
            • 0: angel, crowdfunding<br>
            • 1: series-a<br>
            • 2: series-b<br>
            • 3: series-c+<br>
            • 4: post-ipo (follow-on investment after IPO)<br>
            • 99: venture, grant (government), private-equity, debt_round, secondary_market
            </div>
            """,
            unsafe_allow_html=True
        )

        if "success_prob" not in df.columns:
            st.error("Missing 'success_prob'. Please attach ML_v2 results as 'success_prob' to compute Top 10 scores.")
        else:
            view = df.copy()
            view["Recommendation Score"] = (view["success_prob"] * 100).round(1)
            view["Funding Stage"] = view["num_fr_type"]
            view["Investor Count"] = view["first_participants"]
            view["Reinvestment Rate"] = view["reinvest_rate_next"].round(2)

            show_cols = []
            if "objects_cfpr_id" in view.columns:
                show_cols.append("objects_cfpr_id")
            if "name" in view.columns:
                show_cols.append("name")
            if ind_col and ind_col in view.columns:
                show_cols.append(ind_col)
            show_cols += ["Funding Stage", "Investor Count", "Reinvestment Rate", "Recommendation Score"]

            top10 = view.sort_values("success_prob", ascending=False).head(10).reset_index(drop=True)
            top10.insert(0, "Rank", np.arange(1, len(top10) + 1))

            # Display label mapping (EN)
            COL_EN = {
                "objects_cfpr_id": "Company ID",
                "category": "Industry",
                "name": "Company Name",
                "Funding Stage": "Funding Stage",
                "Investor Count": "Investor Count",
                "Reinvestment Rate": "Reinvestment Rate",
                "Recommendation Score": "Recommendation Score",
            }

            display_cols = ["Rank"] + show_cols
            display_df = top10[display_cols].rename(columns=COL_EN)

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download Top 10 CSV",
                data=display_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"top10_vc_cluster_{vc_cluster}.csv",
                mime="text/csv"
            )
