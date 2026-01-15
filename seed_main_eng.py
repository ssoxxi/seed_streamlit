# Terminal..
# Installation Libraries : pip install streamlit pandas seaborn matplotlib
# Run Streamlit App : streamlit run seed_main_eng.py

import streamlit as st


# =============================================================================
# Overall app setting(st.set_page_config)
# =============================================================================
st.set_page_config(
    page_title="[TEAM SEED] Final Project", # 브라우저 탭 제목
    page_icon="🌱",                         # 브라우저 탭 아이콘
    layout="wide",                          # 넓은 레이아웃
    initial_sidebar_state="expanded",       # 사이드바 기본 펼침
)


# =============================================================================
# Pages
# =============================================================================
home_page = st.Page(
    page="pages_eng/seed_home.py",
    title="Home",
    icon="🌱",
    default=True  # 기본 페이지로 설정
)

screening_page = st.Page(
    page="pages_eng/screening.py",
    title="Start-up screening",
    icon="📊"
)

recommend_page = st.Page(
    page="pages_eng/recommendation.py",
    title="simulation for the investment strategies",
    icon="💰"
)

# =============================================================================
# Navigation
# =============================================================================
pg = st.navigation({
    "Main": [home_page],
    "Analytics": [screening_page],
    "Recommendations": [recommend_page]
})

# =============================================================================
# 선택된 페이지 실행 (필수!)
# =============================================================================
pg.run()