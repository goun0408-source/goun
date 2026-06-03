"""
proposal_app.py — AI 기반 노인·독거인 낙상 감지 & 응급 알림 시스템
사회를 AI로 개선하는 방법 기획 발표 Streamlit 앱 (v3 — 안정화 버전)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI | 낙상 감지 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — 최소한의 커스텀 스타일 (DOM 충돌 없음)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0e1117 !important;
}
.stApp { background: linear-gradient(160deg, #0e1117 60%, #111827 100%) !important; }

/* 사이드바 */
section[data-testid="stSidebar"] > div {
    background: #111827 !important;
    border-right: 1px solid #1f2937;
}

/* 구분선 */
hr { border-color: #1f2937 !important; margin: 1.5rem 0 !important; }

/* 전체 블록 패딩 */
.block-container { padding: 2rem 3rem !important; max-width: 1300px !important; }

/* 숨기기 */
#MainMenu, footer, header { visibility: hidden; }

/* 라디오 버튼 레이블 숨기기 */
.stRadio > label { display: none; }

/* Metric 카드 */
[data-testid="stMetric"] {
    background: #1f2937 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: 1px solid #374151 !important;
}
[data-testid="stMetricValue"] { color: #f9fafb !important; }
[data-testid="stMetricLabel"] { color: #9ca3af !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SafeGuard AI")
    st.caption("AI 기반 낙상 감지 시스템 기획서")
    st.divider()

    page = st.radio(
        "nav",
        options=[
            "🏠 개요 & 배경",
            "📊 문제의 심각성",
            "🤖 AI 기술 설계",
            "🔄 서비스 흐름",
            "💥 기대 임팩트",
            "🗺️ 실행 로드맵",
        ],
        label_visibility="hidden",
    )

    st.divider()
    st.caption("📌 기계학습 응용 기획 과제")
    st.caption("🎓 AI 기반 사회문제 해결")
    st.caption("🏥 노인·독거인 안전 향상")

# ─────────────────────────────────────────────────────────────────────────────
# 공통 헤더
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 50%, #0f2027 100%);
    border: 1px solid #2563eb44;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    text-align: center;
">
    <p style="color:#60a5fa; font-size:0.85rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase; margin:0 0 12px;">
        🛡️ AI FOR SOCIAL GOOD — SAFEGUARD AI
    </p>
    <h1 style="
        color:white; font-size:2.6rem; font-weight:900; margin:0 0 14px; line-height:1.2;
    ">낙상 한 번이 삶을 바꿉니다.<br><span style='color:#60a5fa;'>AI가 1초 만에 감지합니다.</span></h1>
    <p style="color:#9ca3af; font-size:1rem; margin:0; line-height:1.7; max-width:600px; display:inline-block;">
        혼자 사는 노인·장애인의 낙상을 카메라·웨어러블 AI로 실시간 감지하고,<br>
        골든타임 내 응급 알림을 자동 발송하는 스마트 안전 시스템
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 🏠 개요 & 배경
# ═══════════════════════════════════════════════════════
if page == "🏠 개요 & 배경":
    # KPI 카드 (네이티브 st.metric 사용)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👴 독거 노인 수 (2024)", "1,000만 명", "전년比 +3.2%")
    k2.metric("🚨 연간 낙상 경험률", "32%", "65세 이상")
    k3.metric("⏱️ 미발견 평균 시간", "73분", "낙상 후")
    k4.metric("💊 연간 의료비", "4.1조 원", "낙상 관련")

    st.divider()

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("😰 지금 이 순간에도 일어나는 일")
        st.markdown("""
        혼자 사는 노인이 화장실에서 넘어집니다.
        아무도 모릅니다. 휴대폰은 손에 닿지 않습니다.

        **73분이 지나서야** 가족이 연락이 안 된다며 경찰에 신고합니다.
        이 73분이 **사망과 생존을 가릅니다.**

        > ⚠️ **낙상은 65세 이상 응급실 입원 원인 1위** (전체 손상의 42.8%)
        > — 질병관리청

        낙상 사망자의 **68%** 는 빠른 발견만으로 막을 수 있었습니다.
        """)

    with col2:
        st.subheader("💡 SafeGuard AI 의 4단계 해법")
        steps = [
            ("1️⃣", "실내 카메라로 동작 감지", "AI가 자세 키포인트 17개를 실시간 추적"),
            ("2️⃣", "웨어러블 가속도계 융합", "손목 밴드 IMU 센서로 충격 패턴 분석"),
            ("3️⃣", "AI가 낙상 판별 (< 1초)", "LSTM + Vision Transformer 앙상블"),
            ("4️⃣", "가족·119 자동 알림 발송", "GPS 위치 포함 SMS/앱 푸시 즉시 발송"),
        ]
        for num, title, desc in steps:
            with st.container():
                st.markdown(f"**{num} {title}**")
                st.caption(desc)

    st.divider()
    st.subheader("🔬 핵심 기술 스택")
    t1, t2, t3, t4, t5 = st.columns(5)
    for col, (icon, name, desc) in zip([t1,t2,t3,t4,t5], [
        ("🦴", "Pose Estimation", "MediaPipe\n관절 좌표 추출"),
        ("🧠", "Vision Transformer", "낙상 동작\n시퀀스 분류"),
        ("📡", "IMU Fusion", "가속도·자이로\n데이터 결합"),
        ("☁️", "Edge + Cloud", "엣지 추론\n+ 클라우드 알림"),
        ("📱", "모바일 앱", "보호자 실시간\n모니터링"),
    ]):
        with col:
            st.markdown(f"**{icon} {name}**")
            st.caption(desc)

# ═══════════════════════════════════════════════════════
# 📊 문제의 심각성
# ═══════════════════════════════════════════════════════
elif page == "📊 문제의 심각성":
    st.header("📊 문제의 심각성 — 데이터로 보다")
    tab1, tab2, tab3 = st.tabs(["📈 고령화 현황", "🏥 낙상 피해 통계", "💰 사회적 비용"])

    with tab1:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            years = list(range(2010, 2045, 5))
            pct   = [11.0, 13.1, 15.7, 18.4, 20.4, 25.5, 30.5, 35.0, 38.0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years[:len(pct)], y=pct,
                mode="lines+markers",
                line=dict(color="#60a5fa", width=3),
                marker=dict(size=8),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
                name="65세+ 인구 비율 (%)",
            ))
            fig.add_vrect(x0=2025, x1=2044,
                          fillcolor="rgba(96,165,250,0.04)", layer="below", line_width=0,
                          annotation_text="예측 →", annotation_font_color="#60a5fa")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#d1d5db"),
                xaxis=dict(gridcolor="#1f2937"),
                yaxis=dict(title="비율 (%)", gridcolor="#1f2937"),
                showlegend=False, height=320,
                title=dict(text="한국 고령화 추이 (2010→2040)", font=dict(color="white", size=13)),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🔑 핵심 트렌드")
            st.error("🔴 **2025년** — 초고령사회 진입")
            st.warning("🟠 **2030년** — 독거노인 300만 돌파")
            st.info("🔵 **2035년** — 노인 인구 30% 초과")
            st.success("🟢 **2040년** — 4명 중 1명이 노인")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cats = ["즉시 발견\n(1시간↓)", "3시간 이내", "12시간 이내", "12시간↑"]
            vals = [3, 12, 28, 61]
            fig2 = go.Figure(go.Bar(
                x=cats, y=vals,
                marker_color=["#10b981", "#f59e0b", "#f97316", "#ef4444"],
                text=[f"{v}%" for v in vals],
                textposition="outside", textfont=dict(color="white"),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#d1d5db"),
                yaxis=dict(title="사망률 (%)", range=[0, 75], gridcolor="#1f2937"),
                xaxis=dict(gridcolor="#1f2937"),
                showlegend=False, height=320,
                title=dict(text="발견 시간별 낙상 사망률", font=dict(color="white", size=13)),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            labels = ["척추·골반 골절", "뇌진탕·두부 손상", "타박상·찰과상", "근육 파열", "기타"]
            values = [38, 22, 18, 14, 8]
            fig3 = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.5,
                marker=dict(colors=["#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#ddd6fe"],
                            line=dict(color="#0e1117", width=2)),
                textfont=dict(color="white"),
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d1d5db"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
                height=320,
                title=dict(text="낙상 부상 유형 분포", font=dict(color="white", size=13)),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("### 💰 낙상으로 인한 연간 사회적 비용")
        c1, c2, c3 = st.columns(3)
        c1.metric("💊 직접 의료비", "2.1조 원", "입원·수술·재활 치료비")
        c2.metric("🏠 장기 요양비", "1.5조 원", "낙상 후 요양원 이전")
        c3.metric("📉 생산성 손실", "0.5조 원", "보호자 간호 경제손실")

        st.divider()
        st.info("💡 **총 4.1조 원** — AI 조기 감지 도입 시 최대 **30% (약 1.2조 원)** 절감 가능")

# ═══════════════════════════════════════════════════════
# 🤖 AI 기술 설계
# ═══════════════════════════════════════════════════════
elif page == "🤖 AI 기술 설계":
    st.header("🤖 AI 기술 설계")
    tab1, tab2, tab3 = st.tabs(["🎥 비전 모델", "📡 센서 융합", "🎓 학습 전략"])

    with tab1:
        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.subheader("🦴 Pose Estimation + 분류 파이프라인")
            pipeline = [
                ("①", "영상 입력", "1080p CCTV / 스마트홈 캠 → 30fps", "#6366f1"),
                ("②", "Pose Estimation (MediaPipe)", "17개 관절 좌표 (x, y, confidence) 추출", "#8b5cf6"),
                ("③", "Temporal Sequence", "1초 슬라이딩 윈도우 → (30, 17×3) 텐서", "#10b981"),
                ("④", "LSTM + Attention 분류기", "낙상 / 앉기 / 서기 / 걷기 / 눕기 분류", "#f59e0b"),
                ("⑤", "낙상 감지 → 알림 트리거", "신뢰도 > 0.85 → 즉시 응급 알림 발송", "#ef4444"),
            ]
            for num, title, desc, color in pipeline:
                st.markdown(f"""
                <div style="border-left:3px solid {color}; padding:8px 14px; margin:6px 0;
                            background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0;">
                    <b style="color:white;">{num} {title}</b><br>
                    <span style="color:#9ca3af; font-size:0.85rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("📊 모델 성능 목표")

            # ✅ Plotly 수평 바 차트 사용 (커스텀 HTML 루프 제거 → DOM 에러 해결)
            metrics_names = ["낙상 감지 정확도", "정상 판별률", "처리 속도 (<1초)", "야간 환경", "부분 가림"]
            metrics_vals  = [96, 97, 99, 91, 88]
            metrics_colors = ["#ef4444", "#f59e0b", "#6366f1", "#10b981", "#8b5cf6"]

            fig_bar = go.Figure(go.Bar(
                x=metrics_vals,
                y=metrics_names,
                orientation="h",
                marker=dict(
                    color=metrics_colors,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{v}%" for v in metrics_vals],
                textposition="outside",
                textfont=dict(color="white", size=12),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#d1d5db"),
                xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
                yaxis=dict(gridcolor="#1f2937"),
                showlegend=False,
                height=280,
                margin=dict(l=0, r=60, t=10, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("⚡ 추론 성능")
            pa, pb = st.columns(2)
            pa.metric("처리 속도", "< 50ms")
            pa.metric("모델 크기", "~15MB")
            pb.metric("엣지 지원", "Jetson Nano")
            pb.metric("배터리", "< 3% / 시간")

    with tab2:
        st.subheader("📡 멀티모달 센서 융합 전략")
        s1, s2, s3 = st.columns(3)
        for col, (icon, name, items) in zip([s1,s2,s3],[
            ("🎥","카메라 (Vision)",["RGB 영상 분석","포즈 추정","행동 인식","엣지에서 처리 (개인정보 보호)"]),
            ("⌚","IMU 센서 (웨어러블)",["3축 가속도계","3축 자이로스코프","충격 패턴 감지","< 5mAh 소모"]),
            ("🌡️","환경 센서 (보조)",["기압 변화 감지","온도·습도 모니터","24시간 상시 작동","욕실 방수 지원"]),
        ]):
            with col:
                st.markdown(f"**{icon} {name}**")
                for item in items:
                    st.caption(f"• {item}")

        st.divider()
        st.info("🔗 **Late Fusion** — 카메라(0.6) + IMU(0.4) 가중 앙상블 → 이중 감지로 강건한 오류 보완 구조")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎓 전이학습 전략")
            st.markdown("""
            **베이스 모델**: VideoMAE (Kinetics-400 사전학습)
            → 일반 행동 인식 지식 활용

            **Phase 1 (5 에폭)**
            - Backbone 동결 → 분류 헤드만 학습
            - 빠른 수렴 · 과적합 방지

            **Phase 2 (20 에폭)**
            - 마지막 Transformer 블록 + 헤드 Fine-tuning
            - 차등 학습률: Backbone 1e-4 / Head 1e-3
            """)

        with col2:
            st.subheader("📦 학습 데이터셋")
            df_ds = pd.DataFrame({
                "데이터셋": ["UR Fall Detection", "URFD", "SisFall", "자체 수집"],
                "규모": ["1,000+ 시퀀스", "70명 피험자", "19개 낙상 유형", "CCTV 합성 데이터"],
                "특징": ["실내 환경", "5개 카메라 각도", "다양한 연령대", "야간·역광 포함"],
            })
            st.dataframe(df_ds, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 🔄 서비스 흐름
# ═══════════════════════════════════════════════════════
elif page == "🔄 서비스 흐름":
    st.header("🔄 사용자 경험 & 서비스 흐름")

    st.subheader("📱 3개 역할별 인터페이스")
    u1, u2, u3 = st.columns(3)
    for col, (icon, role, features, color_hex) in zip([u1,u2,u3], [
        ("👴", "노인·독거인", ["스마트밴드 착용", "앱 없어도 자동 감지", "응급버튼 하나로 SOS", "일상 활동 리포트"], "#6366f1"),
        ("👨‍👩‍👧", "가족·보호자", ["실시간 알림 수신", "위치 + 영상 확인", "주간 건강 리포트", "응급 연락망 설정"], "#10b981"),
        ("🚒", "응급기관", ["자동 위치 전송", "영상 증거 수신", "출동 우선순위 최적화", "이력 데이터 조회"], "#f59e0b"),
    ]):
        with col:
            st.markdown(f"**{icon} {role}**")
            for f in features:
                st.caption(f"✓ {f}")

    st.divider()
    st.subheader("⏱️ 낙상 감지 후 자동 대응 타임라인")

    timeline = [
        ("0초",   "낙상 발생",       "AI가 자세 붕괴 + IMU 충격 동시 감지",       "🔴"),
        ("< 1초", "AI 판별 완료",    "신뢰도 계산 → 임계값 초과 시 즉시 트리거",    "🟠"),
        ("3초",   "가족 앱 알림",    "GPS 위치 + 실시간 영상 링크 포함 푸시 알림",  "🟡"),
        ("5초",   "119 자동 신고",   "위치 좌표 + 개인정보 자동 전송 (설정 시)",    "🟢"),
        ("2~5분", "응급 출동",       "기존 평균 73분 → 골든타임 내 구조",           "🔵"),
    ]
    for i, (t, title, desc, dot) in enumerate(timeline):
        c_time, c_dot, c_content = st.columns([0.5, 0.2, 5])
        c_time.markdown(f"**{t}**")
        c_dot.markdown(dot)
        c_content.markdown(f"**{title}** — {desc}")

    st.divider()
    st.success("🏆 **기존 73분 → SafeGuard AI 5분 이내** — 발견 시간 93% 단축, 골든타임 내 구조 성공률 극적 향상")

# ═══════════════════════════════════════════════════════
# 💥 기대 임팩트
# ═══════════════════════════════════════════════════════
elif page == "💥 기대 임팩트":
    st.header("💥 기대 임팩트")

    st.subheader("📊 감지 방법별 성능 비교")
    methods  = ["기존 낙상 감지기", "스마트워치 단독", "CCTV AI 단독", "SafeGuard AI (융합)"]
    accuracy = [62, 78, 84, 96]

    fig_cmp = go.Figure(go.Bar(
        x=methods, y=accuracy,
        marker_color=["#374151","#4b5563","#6b7280","#6366f1"],
        text=[f"{v}%" for v in accuracy],
        textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig_cmp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#d1d5db"),
        yaxis=dict(range=[0,115], title="정확도 (%)", gridcolor="#1f2937"),
        xaxis=dict(gridcolor="#1f2937"),
        showlegend=False, height=320,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 레이더 분석")
        cats = ["생명 보호", "의료비 절감", "가족 안심", "응급 효율화", "삶의 질"]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=[30,25,40,35,45], theta=cats, fill="toself",
                                        name="도입 전", line_color="#374151", fillcolor="rgba(55,65,81,0.3)"))
        fig_r.add_trace(go.Scatterpolar(r=[95,82,93,88,91], theta=cats, fill="toself",
                                        name="SafeGuard AI",
                                        line_color="#6366f1", fillcolor="rgba(99,102,241,0.15)"))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,100], color="#4b5563"),
                angularaxis=dict(color="#9ca3af"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            height=350, margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.subheader("📈 사회적 임팩트")
        impacts = [
            ("🫀", "인명 피해 감소", "연간 낙상 사망자 약 30% 감소\n→ 약 600명 생명 보호"),
            ("💰", "의료비 절감",    "골든타임 구조로 중증 악화 방지\n→ 연간 약 1.2조 원 절감"),
            ("👨‍👩‍👧", "가족 부담 경감", "24시간 자동 모니터링\n→ 보호자 정신적 부담 해소"),
            ("🚒", "응급 효율화",   "오탐 감소로 불필요 출동 40% 절감\n→ 실제 응급 집중 가능"),
        ]
        for icon, title, desc in impacts:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc.replace("\n", " "))
            st.markdown("")

# ═══════════════════════════════════════════════════════
# 🗺️ 실행 로드맵
# ═══════════════════════════════════════════════════════
elif page == "🗺️ 실행 로드맵":
    st.header("🗺️ 실행 로드맵")

    phases = [
        ("Phase 1", "2025 Q1–Q2", "🔬 연구 & 프로토타입", [
            "공개 데이터셋 수집 (UR Fall, SisFall)",
            "Pose Estimation 파이프라인 구축",
            "LSTM + Attention 모델 학습 (목표 정확도 90%+)",
            "단일 방 환경 테스트",
        ]),
        ("Phase 2", "2025 Q3–Q4", "🏗️ 제품 개발", [
            "스마트밴드 IMU 모듈 연동",
            "엣지 디바이스 최적화 (Jetson Nano)",
            "모바일 앱 (iOS / Android) 개발",
            "5가구 파일럿 테스트",
        ]),
        ("Phase 3", "2026 Q1–Q2", "🚀 시범 서비스", [
            "지자체 독거노인 지원 사업 연계",
            "100가구 실증 실험",
            "119 연동 프로토콜 개발",
            "의료기관 EMR 연계",
        ]),
        ("Phase 4", "2026 Q3+", "🌐 전국 확산", [
            "복지부·소방청 정식 연동",
            "스마트홈 플랫폼 (네이버·카카오) 제휴",
            "일본·유럽 고령화 시장 수출",
            "구독 모델 (월 9,900원) 상용화",
        ]),
    ]

    for phase, period, title, items in phases:
        with st.expander(f"**{phase} — {title}** ({period})", expanded=(phase == "Phase 1")):
            for item in items:
                st.markdown(f"- {item}")

    st.divider()

    st.subheader("🎯 핵심 KPI")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("낙상 감지 정확도", "96% 이상")
    kpi2.metric("응급 알림 속도",   "5초 이내")
    kpi3.metric("오탐 허용률",      "5% 이하")
    kpi4.metric("목표 가입 가구",   "10만 가구")

    st.divider()
    st.markdown("""
    > **낙상은 예방할 수 없지만, 발견은 1초 만에 할 수 있습니다.**
    >
    > SafeGuard AI는 기술로 고독과 위험 사이의 간격을 메웁니다.
    > **혼자여도 안전한 사회** — AI가 만드는 복지 인프라입니다.
    """)
