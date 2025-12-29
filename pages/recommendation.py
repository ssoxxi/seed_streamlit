import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
st.markdown(
    """
    <style>
    /* 상단이 잘리는 문제 해결: 전체 컨텐츠를 아래로 내림 */
    div.block-container{
        padding-top: 3.0rem !important;   /* 필요시 2.5~4.0 사이로 조정 */
    }

    /* (헤더 숨김 쓰는 경우) 헤더가 사라져도 안전하게 여백 확보 */
    header[data-testid="stHeader"]{
        height: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.set_page_config(
#     page_title="투자 전략 시뮬레이션",
#     layout="wide",
# )
st.cache_data.clear()
# st.title("💰 투자 전략 시뮬레이션")



# =========================
# (추가) VC 클러스터 표시명
# =========================
VC_CLUSTER_NAMES = {
    0: "글로벌 초기 투자형 vc",
    1: "후기 스케일업형 VC",
    2: "초기-중기 투자형 vc",
    3: "금융 중심 보수 투자형 vc",
    4: "seed 특화 투자형 vc",
    5: "성장 검증 단계 투자형 vc",
}

# =========================
# 0) 경로/로더
# =========================
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트 가정 (pages/ 아래에 위치)
DATA_DIR = ROOT / "data"

STARTUP_PATH = DATA_DIR / "startup_base.csv"
VC_PATH = DATA_DIR / "vc_base.csv"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# =========================
# 1) VC 클러스터 룰(예시: 실제 룰은 팀 룰로 교체)
# =========================
VC_FILTER_RULES = {
    0: {  # 글로벌 분산형 초기 VC
        # 초기라서 NaN 허용 + 0~1
        "num_fr_type_in": [0, 1, None],

        "n_founding_min": 1,
        "time_to_first_funding_days_max": 600,

        # 성향 매칭 거의 안 함
        "match_category": False,
        "match_city": False,
        "match_inst": False,
        "degree_min": None
    },

    1: {  # 후기 스케일업형 VC
        "num_fr_type_in": [3, 4, 99],   # 후기 단계만

        "relationships_min": 5,
        "reinvest_rate_next_min": 0.4,

        # 성향 강함
        "match_category": True,
        "match_city": True,
        "match_inst": False,
        "degree_min": None
    },

    2: {  # 초기~중기형 VC
        "num_fr_type_in": [0, 1, 2, None],

        "match_category": True,
        "match_city": False,
        "match_inst": False,
        "degree_min": 2   # 학사 이상
    },

    3: {  # 금융 중심 보수형 VC
        "num_fr_type_in": [2, 3, 4, 99],

        "relationships_min": 2,

        "match_category": False,
        "match_city": False,
        "match_inst": False,
        "degree_min": None
    },

    4: {  # SEED 특화형 VC
        "num_fr_type_in": [0, None],

        "n_founding_min": 2,

        "match_category": False,
        "match_city": False,
       # "match_inst": True,   # 학교 중요
        "degree_min": 2
    },

    5: {  # 성장 확신 단계 투자형 VC
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

    # 1) 라운드(단계) 필터
    if "num_fr_type_in" in rules and "num_fr_type" in df.columns:
        allowed = rules["num_fr_type_in"]
        allow_nan = (None in allowed)
        allowed_vals = [x for x in allowed if x is not None]

        mask = df["num_fr_type"].isin(allowed_vals)
        if allow_nan:
            mask = mask | df["num_fr_type"].isna()
        df = df[mask]

    # 2) 숫자형 min/max 조건
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

    # 3) VC 성향 매칭
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

    # 4) 학위 매칭
    degree_rule = rules.get("degree_min_from_vc_mean")
    if degree_rule and "degree_level_filled" in df.columns and "founder_degree_level_mean" in vc_row.index:
        deg_thr = vc_row["founder_degree_level_mean"]
        if pd.notna(deg_thr):
            df = df[df["degree_level_filled"] >= deg_thr]

    return df


# =========================
# 2) VC 클러스터 대표 프로필 생성
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
# 3) 스타트업 단계 버킷(현 데이터 기반)
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
# 4) 화면
# =========================
startup_df = load_csv(STARTUP_PATH)
vc_df = load_csv(VC_PATH)
vc_prof = build_vc_cluster_profile(vc_df)

# stage bucket 파생(표시용)
if "num_fr_type" in startup_df.columns:
    startup_df["stage_bucket"] = startup_df["num_fr_type"].apply(stage_bucket_from_num_fr_type)
else:
    startup_df["stage_bucket"] = "Unknown"

market_avg = float(startup_df["success_prob"].mean()) if "success_prob" in startup_df.columns else np.nan

# ---- 헤더
header_l, header_r = st.columns([4, 1])
with header_l:
    st.markdown("## 💰 투자 전략 시뮬레이션 및 스타트업 추천")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ VC의 투자 전략에 따라 예상 성공률을 시뮬레이션하고 적합한 스타트업을 추천하는 의사결정 지원 도구입니다.
        </div>
        """,
        unsafe_allow_html=True
    )    

with header_r:
    # (수정) selectbox 옵션을 "0: 한글명"으로 보이게
    available = sorted([int(x) for x in vc_prof["cluster"].dropna().unique()])
    cluster_list = [c for c in VC_CLUSTER_NAMES.keys() if c in available] or available

    vc_cluster = st.selectbox(
        "VC 유형",
        options=cluster_list,
        index=0,
        format_func=lambda c: f"{c}: {VC_CLUSTER_NAMES.get(int(c), 'Unknown')}",
         key="vc_cluster",
    )

# 선택 클러스터 대표 row (안전장치 포함)
sel = vc_prof.loc[vc_prof["cluster"] == vc_cluster]
if sel.empty:
    st.warning("선택한 유형이 vc_base.csv에 없습니다. 데이터/유형 라벨을 확인하세요.")
    vc_row = pd.Series(dtype="object")
else:
    vc_row = sel.iloc[0]

st.divider()

# ---- 상단 2컬럼(좌: 파라미터, 우: 결과)
left, right = st.columns([1.2, 2.8])

with left:
    st.markdown("### 투자 전략 시뮬레이션 조건 설정")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ 아래 조건을 선택하면 해당 전략에 적합한 스타트업 추천 결과가 업데이트됩니다.
        </div>
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ vc 클러스터란 ? 투자 성향이 비슷한 VC들을 분석하여 유형별로 묶은 그룹 <br>
        선택한 VC 유형에 따라 “해당 성향의 VC가 실제로 선호했던 스타트업 특징”을 기반으로 설정됨
        </div>
        """,
        unsafe_allow_html=True
    )

    # use_vc_rules = st.toggle("VC 클러스터 룰 적용", value=True)

    # ind_col = "category_4" if "category_4" in startup_df.columns else ("category" if "category" in startup_df.columns else None)
    ind_col = "category" if "category" in startup_df.columns else None
    if ind_col:
        inds = sorted([x for x in startup_df[ind_col].dropna().unique()])
        sel_inds = st.multiselect("산업 선택 (다중)", inds, default=[], key="sel_inds")
    else:
        sel_inds = []

    stages = sorted(startup_df["stage_bucket"].dropna().unique())
    sel_stages = st.multiselect("투자 단계", stages, default=[], key="sel_stages")

    if "region" in startup_df.columns:
        regions = sorted([x for x in startup_df["region"].dropna().unique()])
        sel_regions = st.multiselect("지역", regions, default=[], key="sel_regions")
    else:
        sel_regions = []
        
    # “조건 초기화” 버튼 구현
    def reset_filters(default_cluster: int):
        st.session_state["vc_cluster"] = default_cluster
        st.session_state["sel_inds"] = []
        st.session_state["sel_stages"] = []
        st.session_state["sel_regions"] = []

    default_cluster = cluster_list[0] if len(cluster_list) else 0

    st.button(
        "조건 초기화",
        on_click=reset_filters,
        args=(default_cluster,),
        width="stretch",   # use_container_width 경고 대응
    )


with right:
    st.markdown("### 시뮬레이션 결과")

    df = startup_df.copy()

     # 항상 VC 클러스터 룰 적용
    try:
        df = apply_vc_filter(df, vc_row, int(vc_cluster))
    except KeyError:
        st.error(f"VC_FILTER_RULES에 cluster={vc_cluster} 룰이 없습니다. 룰 딕셔너리를 추가하세요.")
        st.stop()
    except Exception as e:
        st.error(f"VC 룰 적용 중 오류: {e}")
        st.stop()

    if ind_col and sel_inds:
        df = df[df[ind_col].isin(sel_inds)]

    if sel_stages:
        df = df[df["stage_bucket"].isin(sel_stages)]

    if "region" in df.columns and sel_regions:
        df = df[df["region"].isin(sel_regions)]

    if len(df) == 0:
        st.warning("필터 결과가 0건입니다. 조건을 완화해 주세요.")
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
            hold_label = "예상 평균 회수기간"
        elif "time_to_first_funding_days" in df.columns:
            hold_years = float(df["time_to_first_funding_days"].mean()) / 365.0
            hold_label = "예상 평균 첫 투자까지 기간"
        else:
            hold_years = np.nan
            hold_label = "예상 평균 기간"

        # if "roi_multiple_pred" in df.columns:
        #     roi_val = float(df["roi_multiple_pred"].mean())
        #     roi_str = f"{roi_val:.2f}배수"
        # else:
        #     roi_str = "N/A"

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            cluster_label = VC_CLUSTER_NAMES.get(int(vc_cluster), "Unknown")
            short = cluster_label.replace("VC", "").strip()  # 필요시 더 축약

            st.metric("VC 유형", f"{vc_cluster}: {short}")

            # st.metric("VC 클러스터", f"{vc_cluster}: {cluster_name}")
            
        with kpi2:
            st.metric(
                "예상 투자 성공률",
                f"{exit_rate*100:.1f}%",
                delta=(f"{delta_pp:+.1f}%p" if np.isfinite(delta_pp) else None),
            )
        with kpi3:
            st.metric(hold_label, (f"{hold_years:.1f}년" if np.isfinite(hold_years) else "N/A"))
        # with kpi3:
        #     st.metric("예상 ROI(공개 금액 기준)", roi_str)

        # if roi_str == "N/A":
        #     st.caption("ROI는 현재 startup_base.csv에 ROI 예측/산출 컬럼이 없어 N/A로 표시됩니다.")

        st.markdown("### Top 10 추천 스타트업")
        st.markdown(
            """
            <div style="margin-top:-6px; margin-bottom:10px; color:#8a8a8a; font-size:12.5px; line-height:1.5;">
            <b>[투자 라운드 유형 코드]</b><br>
            • 0: angel, crowdfunding<br>
            • 1: series-a<br>
            • 2: series-b<br>
            • 3: series-c+<br>
            • 4: post-ipo (상장 이후 후속투자)<br>
            • 99: venture, grant(정부지원), private-equity, debt_round, secondary_market
            </div>
            """,
            unsafe_allow_html=True
        )


        if "success_prob" not in df.columns:
            st.error("success_prob 컬럼이 없어 Top10 추천점수 산출이 불가합니다. ML_v2 결과를 success_prob로 붙여주세요.")
        else:
            view = df.copy()
            view["추천점수"] = (view["success_prob"] * 100).round(1)
            view["라운드 단계"] = view["num_fr_type"]
            view["투자자 수"] = view["first_participants"]
            view["재투자율"] = view["reinvest_rate_next"].round(2)
            
            # def strength_row(r):
            #     parts = []
            #     if "num_fr_type" in r.index and pd.notna(r["num_fr_type"]):
            #         parts.append(f"라운드유형 {int(r['num_fr_type']) if float(r['num_fr_type']).is_integer() else r['num_fr_type']}")
            #     if "first_participants" in r.index and pd.notna(r["first_participants"]):
            #         parts.append(f"투자자 {int(r['first_participants'])}")
            #     if "reinvest_rate_next" in r.index and pd.notna(r["reinvest_rate_next"]):
            #         parts.append(f"재투자율 {r['reinvest_rate_next']:.2f}")
            #     return ", ".join(parts) if parts else "-"

            # view["주요 강점"] = view.apply(strength_row, axis=1)

            show_cols = []
            if "objects_cfpr_id" in view.columns:
                show_cols.append("objects_cfpr_id")
            if "name" in view.columns:
                show_cols.append("name")
            if ind_col and ind_col in view.columns:
                show_cols.append(ind_col)
            # show_cols += ["추천점수", "주요 강점"]
            show_cols += ["라운드 단계", "투자자 수", "재투자율", "추천점수"]

            top10 = view.sort_values("success_prob", ascending=False).head(10).reset_index(drop=True)
            top10.insert(0, "순위", np.arange(1, len(top10) + 1))

            st.dataframe(
                top10[["순위"] + show_cols],
                width="stretch",
                hide_index=True
            )


            st.download_button(
                "Top10 CSV 다운로드",
                data=top10.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"top10_vc_cluster_{vc_cluster}.csv",
                mime="text/csv"
            )
