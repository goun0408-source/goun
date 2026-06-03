"""pages/home.py — 홈 화면"""
import streamlit as st
from utils import GOVT_URL

st.markdown("""
<div class="hero-wrap">
  <div class="hero-chip">🎫 ScalpGuard · AI Anti-Scalping Platform</div>
  <div class="hero-title">암표1를 AI로 탐지하고<br>공식 신고까지 한 번에</div>
  <div class="hero-sub">
    티켓베이·중고나라·당근마켓 등에서 발견한 의심 매물을 입력하면<br>
    딥러닝이 4가지 신호를 분석해 암표 위험도를 즉시 판별합니다.
  </div>
  <div class="hero-stats">
    <div><div class="hero-stat-val">96.2%</div><div class="hero-stat-lbl">탐지 정확도</div></div>
    <div><div class="hero-stat-val">4가지</div><div class="hero-stat-lbl">분석 특징</div></div>
    <div><div class="hero-stat-val">&lt;1초</div><div class="hero-stat-lbl">분석 속도</div></div>
    <div><div class="hero-stat-val">26종</div><div class="hero-stat-lbl">의심 키워드</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="medium")
cards = [
    ("🔍", "암표 탐지 & 신고",  "의심 매물 정보 입력 →\nAI 위험도 즉시 분석"),
    ("📋", "신고 내역",         "접수 신고 조회 ·\n필터링 · CSV 내보내기"),
    ("📊", "통계 대시보드",     "신고 데이터 기반\n암표 현황 시각화"),
    ("🤖", "모델 분석",         "신경망 구조 &\n특징 중요도 확인"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:1.8rem 1rem;">
          <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
          <div style="font-size:0.95rem;font-weight:800;color:#1e293b;margin-bottom:0.4rem;">{title}</div>
          <div style="font-size:0.78rem;color:#64748b;line-height:1.7;white-space:pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="panel-info" style="margin-top:1.2rem;">
  📌 <strong>사용 방법</strong> — 왼쪽 사이드바 메뉴에서 원하는 기능을 선택하세요.
</div>
""", unsafe_allow_html=True)

with st.expander("🚀 빠른 시작 가이드"):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 📍 탐지 순서")
        for step, desc in [
            ("1단계", "사이드바에서 **🔍 암표 탐지 & 신고** 클릭"),
            ("2단계", "이벤트·플랫폼·가격 정보 입력"),
            ("3단계", "**AI 위험도 분석** 버튼 클릭"),
            ("4단계", "결과 확인 후 필요 시 신고 접수"),
        ]:
            st.markdown(f"**{step}** — {desc}")
    with col2:
        st.markdown("#### ⚠️ 주의사항")
        st.markdown(f"""
- AI 판정은 **참고 자료**이며 100% 정확하지 않습니다
- 최종 신고 여부는 **본인이 직접 판단**하세요
- 허위·과장 신고는 **법적 책임**이 발생합니다
- 공식 신고: [문화체육관광부 암표 신고 누리집]({GOVT_URL})
""")
