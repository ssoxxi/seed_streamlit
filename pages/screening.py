# -*- coding: utf-8 -*-
"""
Streamlit | 스타트업 스크리닝 대시보드
- 좌측: 필터 패널 + 성공점수 히스토그램
- 우측: KPI + 리스트 + 선택 기업 상세 + (1) 점수 구성 (2) SHAP 로컬 설명

데이터:
- success_master.csv   : 회사/라운드 원천 + 성공여부 등
- startup_ver.csv      : 군집 결과(cluster) + (군집 학습에 썼던) 파생 피처
- shap_local.csv       : 기업별 로컬 기여도(피처별 SHAP 값)  ※ 'objects_cfpr_id' 키로 조인
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from theme import apply_theme # 앱 theme

# =============================================================================
# 0) 페이지 기본 설정
# =============================================================================
st.cache_data.clear()
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

st.set_page_config(page_title="분석과정 | 스타트업 스크리닝", layout="wide", page_icon="📊")
st.title("📊 스타트업 스크리닝")
st.markdown(
    """
    <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
    ※ VC의 투자 검토 초기 단계에서 관심 조건(산업·국가·투자 단계 등)에 맞는 스타트업을 빠르게 선별하고 후속 검토 대상 기업을 좁히기 위한 1차 스크리닝 도구입니다.
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# 1) 경로
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]  # 프로젝트 구조에 맞게 조정
DATA_DIR = BASE_DIR / "data"

SUCCESS_PATH = DATA_DIR / "s_master_distinct_startups.csv" 
CLUSTER_PATH = DATA_DIR / "startup_ver.csv"
SHAP_PATH = DATA_DIR / "shap_local.csv"


# =============================================================================
# 2) 클러스터(스타트업 유형) 라벨 (표시용)
# =============================================================================
CLUSTER_LABEL = {
    0: "초기 실험형 스타트업",
    1: "네트워크 기반 성장형 스타트업",
    2: "도메인 특화 안정형 스타트업",
    3: "엘리트 창업자 기반 기술 스타트업",
    4: "연구 중심 장기 성장형 스타트업",
}


# =============================================================================
# 3) 로더
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

    # 날짜 파싱
    for c in ["founded_at", "funded_at", "acquired_at", "first_public_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 숫자형 정리
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
# 4) 스코어 유틸
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
# 5) 회사 단위 마스터 생성
# =============================================================================
@st.cache_data(show_spinner=True)
def build_company_master(success_path: Path, startup_ver_path: Path) -> pd.DataFrame:
    df = load_csv(success_path).copy()
    if df.empty:
        return df

    df["objects_cfpr_id"] = df["objects_cfpr_id"].astype(str)

    # ------------------------------------------------------------------
    # [CASE 1] 이미 회사(1행=1스타트업) 집계본(s_master_distinct.csv)인 경우
    # ------------------------------------------------------------------
    is_distinct = {"industry", "country", "founded_year", "total_rounds", "invest_stage_last"}.issubset(df.columns)

    if is_distinct:
        # 컬럼명 통일(대시보드에서 기대하는 이름으로 맞추기)
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

        # 타입 정리
        for c in ["founded_year", "round_cnt", "funding_total_usd", "relationships",
                  "first_raised_usd", "last_raised_usd", "success_flag"]:
            if c in df_company.columns:
                df_company[c] = pd.to_numeric(df_company[c], errors="coerce")

        # IPO/M&A 이미 0/1로 들어있으면 그대로 사용(없으면 0)
        if "ipo_achieved" not in df_company.columns:
            df_company["ipo_achieved"] = 0
        if "mna_achieved" not in df_company.columns:
            df_company["mna_achieved"] = 0
        df_company["ipo_achieved"] = pd.to_numeric(df_company["ipo_achieved"], errors="coerce").fillna(0).astype(int)
        df_company["mna_achieved"] = pd.to_numeric(df_company["mna_achieved"], errors="coerce").fillna(0).astype(int)

        # 성장률 계산
        fr = pd.to_numeric(df_company.get("first_raised_usd"), errors="coerce")
        lr = pd.to_numeric(df_company.get("last_raised_usd"), errors="coerce")
        df_company["funding_growth_rate"] = np.where((fr > 0) & np.isfinite(fr) & np.isfinite(lr), (lr - fr) / fr, np.nan)

        # 없을 수 있는 컬럼(UI에서 쓰면 대비)
        if "obj_city_fixed" not in df_company.columns:
            df_company["obj_city_fixed"] = pd.NA
        if "n_offices" not in df_company.columns:
            df_company["n_offices"] = pd.NA

    # ------------------------------------------------------------------
    # [CASE 2] 원천(라운드 단위) success_master.csv인 경우: 기존 집계 로직 수행
    # ------------------------------------------------------------------
    else:
        # founded_at이 있을 때만 founded_year 만들기
        if "founded_at" in df.columns:
            df["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")
            df["founded_year"] = df["founded_at"].dt.year
        else:
            df["founded_year"] = pd.NA

        # 이하: 기존 로직(당신 코드)을 그대로 두되,
        # df["founded_at"] 직접 참조 같은 부분은 반드시 컬럼 존재 체크로 감싸야 안전합니다.
        # (여기는 생략: A안을 쓰거나, 원천 파일을 쓰는 경우에만 타는 분기라서)

        # --- 최소한 기존 코드의 결과물이 df_company에 들어오도록 구성 ---
        # 원천 파일을 쓰는 경우만 필요하면, 기존 build_company_master 내용을 여기로 옮기세요.
        df_company = df.groupby("objects_cfpr_id", as_index=False).first()

    # ------------------------------------------------------------------
    # cluster merge
    # ------------------------------------------------------------------
    sv = load_startup_ver(startup_ver_path)
    if (not sv.empty) and {"objects_cfpr_id", "cluster"}.issubset(sv.columns):
        df_company = df_company.merge(sv[["objects_cfpr_id", "cluster"]], on="objects_cfpr_id", how="left")
    else:
        df_company["cluster"] = pd.Series([pd.NA] * len(df_company), dtype="Int64")

    # 카테고리 최적화
    for c in ["obj_category_filled", "country_code", "cat_fr_type", "obj_city_fixed"]:
        if c in df_company.columns:
            df_company[c] = df_company[c].astype("category")

    return df_company



# =============================================================================
# 6) 데이터 로드
# =============================================================================
if not SUCCESS_PATH.exists():
    st.error(f"데이터 파일을 찾을 수 없습니다: {SUCCESS_PATH}")
    st.stop()

df_company = build_company_master(SUCCESS_PATH, CLUSTER_PATH)

startup_ver = load_startup_ver(CLUSTER_PATH) if CLUSTER_PATH.exists() else pd.DataFrame()
shap_local  = load_shap_local(SHAP_PATH) if SHAP_PATH.exists() else pd.DataFrame()

# st.write("df_company rows:", len(df_company))
# st.write("unique objects:", df_company["objects_cfpr_id"].nunique())
# st.write("duplicated objects:", df_company["objects_cfpr_id"].duplicated().sum())
# st.write("prefix counts:", df_company["objects_cfpr_id"].str[0].value_counts())

# =============================================================================
# 7) 좌/우 레이아웃
# =============================================================================
col_left, col_right = st.columns([1.1, 3.2], gap="large")


# =============================================================================
# 8) 좌측 상단: 필터 패널 → 투자 조건 설정
# =============================================================================
with col_left:
    st.subheader("투자 조건 설정")
    st.markdown(
        """
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ 원하는 스타트업 유형을 선택하여 후보군을 좁혀보세요.
        </div>
        <div style="color:#6B7280; font-size:12.8px; line-height:1.55; margin-top:-6px; margin-bottom:12px;">
        ※ 스타트업 클러스터란 ? 비슷한 전략,단계,역량을 가진 스타트업 그룹<br> 
        선택한 스타트업 유형에 따라 선호하는 스타일의 기업을 찾을 수 있음
        </div>
        """,
        unsafe_allow_html=True
    )

    # (1) 클러스터 옵션
    cluster_options = []
    if (not startup_ver.empty) and ("cluster" in startup_ver.columns):
        cluster_ids = startup_ver["cluster"].dropna().astype(int).unique().tolist()
        cluster_ids = sorted(cluster_ids)
        cluster_options = [f"{cid} | {CLUSTER_LABEL.get(cid, f'Cluster {cid}')}" for cid in cluster_ids]

    # (2) 기타 옵션/범위
    industry_opts = list(df_company["obj_category_filled"].cat.categories) if "obj_category_filled" in df_company.columns else []
    country_opts  = list(df_company["country_code"].cat.categories) if "country_code" in df_company.columns else []
    round_opts    = list(df_company["cat_fr_type"].cat.categories) if "cat_fr_type" in df_company.columns else []

    y_min = int(df_company["founded_year"].dropna().min()) if df_company["founded_year"].notna().any() else 1990
    y_max = int(df_company["founded_year"].dropna().max()) if df_company["founded_year"].notna().any() else 2025
    # s_min = float(df_company["success_score"].min()) if df_company["success_score"].notna().any() else 0.0
    # s_max = float(df_company["success_score"].max()) if df_company["success_score"].notna().any() else 100.0

    # 위젯 생성 전에 기본값 보장 (첫 실행 시만)
    st.session_state.setdefault("f_cluster_label", [])
    st.session_state.setdefault("f_industry", [])
    st.session_state.setdefault("f_country", [])
    st.session_state.setdefault("f_round", ["seed"])
    st.session_state.setdefault("f_year", (y_min, y_max))
    # st.session_state.setdefault("f_score", (float(s_min), float(s_max)))

    # 초기화 콜백(버튼 on_click에서만 session_state 수정)
    def reset_filters():
        st.session_state["f_cluster_label"] = []
        st.session_state["f_industry"] = []
        st.session_state["f_country"] = []
        st.session_state["f_round"] = []
        st.session_state["f_year"] = (y_min, y_max)
        # st.session_state["f_score"] = (float(s_min), float(s_max))

    # ---- 위젯들 ----
    sel_cluster_label = st.multiselect(
        "스타트업 유형 선택",
        options=cluster_options,
        key="f_cluster_label",
    )
    sel_cluster_ids = [int(x.split("|")[0].strip()) for x in sel_cluster_label] if sel_cluster_label else []

    st.multiselect("산업 선택", options=industry_opts, key="f_industry")
    st.multiselect("국가 선택", options=country_opts, key="f_country")
    st.multiselect("투자 단계", options=round_opts, key="f_round")

    # ✅ f_year 초기값은 '없는 경우에만' 세팅
    if "f_year" not in st.session_state:
        st.session_state["f_year"] = (y_min, y_max)

    # ✅ 슬라이더 생성: session_state에 값이 있으면 value=를 주지 않음 (경고 제거)
    if "f_year" in st.session_state:
        st.slider(
            "설립연도",
            min_value=y_min,
            max_value=y_max,
            step=1,
            key="f_year",
        )
    else:
        st.slider(
            "설립연도",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max),
            step=1,
            key="f_year",
        )


    # st.slider("성공점수", min_value=float(np.floor(s_min)), max_value=float(np.ceil(s_max)),
    #           value=st.session_state["f_score"], step=0.5, key="f_score")

    # on_click 사용 (st.rerun() 필요 없음)
    st.button("필터 초기화", use_container_width=True, on_click=reset_filters)

# =============================================================================
# 9) 필터 적용
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


# # =============================================================================
# # 10) 좌측 하단: 히스토그램
# # =============================================================================
# with col_left:
#     st.divider()
#     st.subheader("성공점수 분포")

#     score = filtered["success_score"].dropna()
#     if score.empty:
#         st.info("필터 결과에 성공점수 데이터가 없습니다.")
#     else:
#         fig = px.histogram(filtered.dropna(subset=["success_score"]), x="success_score", nbins=30)
#         fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#         st.caption(f"평균: {score.mean():.1f} / 중앙값: {score.median():.1f}")


# =============================================================================
# 11) 우측: KPI + 리스트 + 상세 + Breakdown(SHAP)
# =============================================================================
with col_right:
            
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("전체 기업 수", f"{len(df_company):,}")
    with k2:
        st.metric("필터링된 기업 수", f"{len(filtered):,}")
    with k3:
        ratio = (len(filtered) / len(df_company) * 100) if len(df_company) else 0
        st.metric("필터링 비율", f"{ratio:.1f}%")

    st.divider()
    st.subheader("스타트업 리스트")

    # 검색(선택)
    q = st.text_input("검색: 기업id 포함 검색", value="", placeholder="예) c:10001")
    table_view = filtered.copy()
    if q.strip():
        table_view = table_view[table_view["objects_cfpr_id"].str.contains(q.strip(), case=False, na=False)]

    # show_cols = ["objects_cfpr_id", "obj_category_filled", "country_code", "cat_fr_type", "round_cnt", "success_score", "cluster"]
    show_cols = ["objects_cfpr_id", "obj_category_filled", "country_code", "cat_fr_type", "round_cnt", "cluster"]
    show_cols = [c for c in show_cols if c in table_view.columns]

    top_n = st.slider("테이블 표시 행 수", 50, 500, 200, step=50)
    table_df = table_view[show_cols].sort_values("cat_fr_type", ascending=False).head(top_n).copy()

    # 표시용 cluster_label 컬럼(있으면)
    if "cluster" in table_df.columns:
        table_df["cluster_label"] = table_df["cluster"].map(lambda x: f"{int(x)} | {CLUSTER_LABEL.get(int(x), f'Cluster {int(x)}')}" if pd.notna(x) else "—")
        # 표시 순서 조정
        display_cols = [c for c in show_cols if c != "cluster"] + ["cluster_label"]
    else:
        display_cols = show_cols

    # 컬럼 표시명(KR)
    column_config = {
        "objects_cfpr_id": st.column_config.TextColumn("기업ID"),
        "obj_category_filled": st.column_config.TextColumn("산업"),
        "country_code": st.column_config.TextColumn("국가"),
        "cat_fr_type": st.column_config.TextColumn("투자 단계"),
        "round_cnt": st.column_config.NumberColumn("총 라운드 수", format="%.0f"),
        # "success_score": st.column_config.NumberColumn("성공 점수", format="%.1f"),
        "cluster_label": st.column_config.TextColumn("스타트업 유형"),
    }

    st.dataframe(table_df[display_cols], use_container_width=True, hide_index=True, column_config=column_config)

    st.subheader("선택한 기업 상세 카드")

    # 선택 옵션(테이블 기준)
    options = table_df["objects_cfpr_id"].dropna().astype(str).unique().tolist()
    if not options:
        st.info("필터 결과가 없습니다.")
        st.stop()

    STATE_KEY = "screening_selected_company"
    if (STATE_KEY not in st.session_state) or (st.session_state[STATE_KEY] not in options):
        st.session_state[STATE_KEY] = options[0]

    selected_id = st.selectbox("기업 선택", options=options, index=options.index(st.session_state[STATE_KEY]))
    st.session_state[STATE_KEY] = selected_id

    lookup = filtered.set_index("objects_cfpr_id", drop=False)
    row = lookup.loc[selected_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    # 상단 카드
    cA, cB, cC, cD, cE = st.columns([2.4, 1, 1, 1, 1.2])
    with cA:
        st.markdown(f"### {row['objects_cfpr_id']}")
        st.write(
            f"산업: {row.get('obj_category_filled', '—')} | "
            f"국가: {row.get('country_code', '—')} | "
            f"단계: {row.get('cat_fr_type', '—')}"
        )
    with cB:
        st.metric("설립연도", "—" if pd.isna(row.get("founded_year")) else int(row["founded_year"]))
    with cC:
        st.metric("라운드 수", "—" if pd.isna(row.get("round_cnt")) else int(row["round_cnt"]))
    # with cD:
    #     st.metric("성공점수", f"{row.get('success_score', np.nan):.1f}")
    with cD:
        if pd.notna(row.get("cluster")):
            cid = int(row["cluster"])
            cl_txt = f"{cid} | {CLUSTER_LABEL.get(cid, f'Cluster {cid}')}"
        else:
            cl_txt = "—"

        # 클러스터만 폰트 축소 + 말줄임(잘림 방지)
        st.markdown(
            f"""
            <div style="text-align:center;">
            <div style="font-size:0.85rem; color: rgba(0,0,0,0.6); margin-bottom:0.25rem;">
                스타트업 유형
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
    growth_txt = "—" if pd.isna(growth_pct) else f"{growth_pct*100:.1f}%"

    # 탭: 요약 /  SHAP
    tab1, tab2 = st.tabs(["요약", "모델 기여도(SHAP)"])

    with tab1:
        st.write(
            f"- IPO 달성: {int(row.get('ipo_achieved', 0))}\n"
            f"- M&A 달성: {int(row.get('mna_achieved', 0))}\n"
            f"- Funding 성장률(첫→마지막): {growth_txt}\n"
            f"- 첫 라운드 raised(USD): {row.get('first_raised_usd', np.nan)}\n"
            f"- 마지막 라운드 raised(USD): {row.get('last_raised_usd', np.nan)}\n"
            f"- 총 투자금(USD): {row.get('funding_total_usd', 0):,.0f}\n"
            f"- 관계규모(relationships): {row.get('relationships', 0):,.0f}\n"
            f"- 성공여부(success_flag): {row.get('success_flag', np.nan)}"
        )

    with tab2:
        # shap_local.csv 기반 breakdown
        if shap_local.empty:
            st.info("shap_local.csv가 없어서 모델 기여도를 표시할 수 없습니다.")
        else:
            # id 타입 불일치 방지
            shap_local["objects_cfpr_id"] = shap_local["objects_cfpr_id"].astype(str)
            sid = str(selected_id)

            shap_df = shap_local.loc[shap_local["objects_cfpr_id"] == sid].copy()

            if shap_df.empty:
                st.info("선택 기업의 SHAP 정보가 없습니다.")
            else:
                # wide(열=feature) → long(feature, shap_value)
                shap_mat = shap_df.drop(columns=["objects_cfpr_id"], errors="ignore")

                # 혹시 동일 기업이 여러 행이면 평균으로 대표값
                shap_mat = shap_mat.apply(pd.to_numeric, errors="coerce")
                shap_vec = shap_mat.mean(axis=0) if len(shap_mat) > 1 else shap_mat.iloc[0]

                shap_long = (
                    shap_vec.rename("shap_value")
                            .reset_index()
                            .rename(columns={"index": "feature"})
                )

                # 안전장치: shap_value가 없으면 강제로 생성
                if "shap_value" not in shap_long.columns:
                    # reset_index 결과가 ['index', 0] 형태일 때 대응
                    if 0 in shap_long.columns:
                        shap_long = shap_long.rename(columns={0: "shap_value"})
                    else:
                        st.error("SHAP 데이터 컬럼 파싱에 실패했습니다. shap_local.csv 구조를 확인해주세요.")
                        st.stop()

                shap_long["shap_value"] = pd.to_numeric(shap_long["shap_value"], errors="coerce")
                shap_long = shap_long.dropna(subset=["shap_value"])

                if shap_long.empty:
                    st.info("선택 기업의 SHAP 값이 전부 결측입니다.")
                else:
                    shap_long["abs"] = shap_long["shap_value"].abs()
                    shap_long = shap_long.sort_values("abs", ascending=False).head(15)

                    st.caption("값이 +이면 모델의 '성공' 예측을 올리는 방향, -이면 낮추는 방향으로 해석합니다.")

                    fig = px.bar(
                        shap_long.sort_values("shap_value"),
                        x="shap_value",
                        y="feature",
                        orientation="h",
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="SHAP value",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)