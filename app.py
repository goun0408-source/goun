"""
app.py — ScalpGuard 메인 진입점 (st.navigation 방식)
"""

import streamlit as st
from utils import GLOBAL_CSS, init_db, load_resources, render_sidebar

st.set_page_config(
    page_title="ScalpGuard — AI 암표 탐지",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS & 공통 초기화 (모든 페이지에 적용)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
init_db()
model, scaler = load_resources()
render_sidebar(model)

# 페이지 네비게이션 정의
pg = st.navigation([
    st.Page("pages/home.py",         title="홈",             icon="🏠", default=True),
    st.Page("pages/detect.py",       title="암표 탐지 & 신고", icon="🔍"),
    st.Page("pages/history.py",      title="신고 내역",       icon="📋"),
    st.Page("pages/stats.py",        title="통계 대시보드",    icon="📊"),
    st.Page("pages/model_info.py",   title="모델 분석",       icon="🤖"),
])

pg.run()
