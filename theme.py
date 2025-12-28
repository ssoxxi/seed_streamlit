import streamlit as st

def apply_theme():
    st.markdown(
        """
        <style>
        :root{
          --brand:#1E4EB5;          /* blue */
          --brand-2:#ACC6ECff;      /* skyblue */
          --bg:#FFFFFF;
          --bg-2:#F3F9FF;
          --text:#1F2937;
          --border: rgba(33,150,243,0.18);
          --shadow: 0 6px 18px rgba(15, 45, 90, 0.08);
          --radius: 14px;
          --sliderbar-color: #1E4EB5;
        }
        
        /* ---- Streamlit theme vars (중요) ---- */
        :root{
          --primary-color: var(--brand);
          --primaryColor: var(--brand);                 /* 일부 버전 호환 */
          --secondary-background-color: var(--bg-2);
          --background-color: var(--bg);
          --text-color: var(--text);
          --sliderbar-color: #1E4EB5;
        }

  

        /* ---- Toggle/Switch 파란색 ---- */
      



        /* ---- 상단 라벨 잘림 방지 ---- */

        /* columns(가로 블록)에서 라벨이 위로 잘리는 케이스 방지 */
        div[data-testid="stHorizontalBlock"]{
          overflow: visible !important;
        }

        /* 라벨 박스 자체 여백/라인하이트 보정 */
        div[data-testid="stWidgetLabel"]{
          overflow: visible !important;
          padding-top: 0.35rem !important;
          padding-bottom: 0.15rem !important;
        }
        div[data-testid="stWidgetLabel"] label{
          line-height: 1.45 !important;
          overflow: visible !important;
        }


        /* 전체 배경 */
        .stApp { background: var(--bg); color: var(--text); }

        /* 사이드바 */
        section[data-testid="stSidebar"]{
          background: var(--bg-2);
          border-right: 1px solid var(--border);
        }

        /* 컨테이너(카드 느낌) - 자주 쓰는 블록에 적용됨 */
        div[data-testid="stVerticalBlockBorderWrapper"]{
          background: #fff;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: 14px 14px 6px 14px;
        }

        /* 버튼 */
        .stButton > button{
          border-radius: 12px;
          border: 1px solid rgba(33,150,243,0.35);
          background: linear-gradient(180deg, var(--brand), #1E88E5);
          color: white;
          font-weight: 600;
          padding: 0.55rem 0.95rem;
        }
        .stButton > button:hover{
          filter: brightness(1.03);
          border-color: rgba(33,150,243,0.55);
        }

        /* 입력 위젯(셀렉트/텍스트/멀티셀렉트 등) 테두리 톤 */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div{
          border-radius: 12px !important;
          border-color: rgba(33,150,243,0.25) !important;
        }

        /* 탭 */
        button[data-baseweb="tab"]{
          border-radius: 12px 12px 0 0 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]{
          color: var(--brand) !important;
        }

        /* expander */
        details{
          background: #fff;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: 10px 12px;
        }

        /* 메트릭 카드 느낌(메트릭은 구조가 자주 바뀌어서 과하게 건드리지 않는 편이 안정적) */
        [data-testid="stMetric"]{
          background: #fff;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 12px;
          box-shadow: var(--shadow);
        }

        /* 기본 여백 살짝 정리 */
        .block-container{
          padding-top: 1.2rem;
          padding-bottom: 2rem;
        }

        /* Streamlit 기본 footer/menu 숨김(원하면 제거) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* ---- Widget label(라벨) 잘림 방지 ---- */
        div[data-testid="stWidgetLabel"]{
          overflow: visible !important;
          padding-top: 0.25rem !important;
          padding-bottom: 0.15rem !important;
        }

        label[data-testid="stWidgetLabel"]{
          line-height: 1.35 !important;
          overflow: visible !important;
        }

        /* ---- MultiSelect 선택 태그 색상(오렌지/레드 제거) ---- */
        span[data-baseweb="tag"]{
          background: linear-gradient(180deg, var(--brand-2), var(--brand)) !important;
          color: #fff !important;
          border: none !important;
          border-radius: 999px !important;
          font-weight: 600 !important;
        }

        span[data-baseweb="tag"] svg{
          color: #fff !important;   /* X 아이콘도 흰색 */
        }


        </style>
        """,
        unsafe_allow_html=True
    )