"""pages/model_info.py — AI 모델 분석"""
import streamlit as st
import plotly.graph_objects as go

from utils import BASE, PLOTLY_LAYOUT, load_resources

model, scaler = load_resources()

st.markdown("""
<div class="hero-wrap" style="padding:2rem 2.5rem;">
  <div class="hero-chip">🤖 Model Intelligence</div>
  <div class="hero-title" style="font-size:1.8rem;">AI 모델 분석</div>
  <div class="hero-sub" style="font-size:0.88rem;">
    ScalpDetector 신경망의 구조, 학습 결과, 특징 중요도를 분석합니다.
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏗️ 모델 구조 & 설정", "📈 학습 결과", "⚖️ 특징 중요도"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-hd">🏗️ ScalpDetector 신경망 구조</div>', unsafe_allow_html=True)
            for name, size, color, note in [
                ("입력층",   "4개 특징 벡터",                           "#0f172a", "price_ratio · account_age_days · num_listings · keywords"),
                ("Hidden 1", "32 뉴런 + BatchNorm + ReLU + Dropout(0.3)", "#334155", "배치 정규화로 학습 안정화, 드롭아웃으로 과적합 방지"),
                ("Hidden 2", "16 뉴런 + BatchNorm + ReLU + Dropout(0.3)", "#475569", "특징을 압축하여 판별력 있는 표현 학습"),
                ("출력층",   "1 뉴런 + Sigmoid → 0~1 확률 출력",        "#ef4444", "≥ 0.70 이면 암표 의심으로 판정"),
            ]:
                st.markdown(f"""
                <div style="border-left:3px solid {color};padding:0.7rem 1rem;background:#f8fafc;border-radius:0 10px 10px 0;margin:0.5rem 0;">
                  <div style="font-weight:700;color:#1e293b;font-size:0.88rem;">{name}</div>
                  <div style="color:{color};font-size:0.8rem;font-weight:600;margin-top:0.15rem;">{size}</div>
                  <div style="color:#64748b;font-size:0.74rem;margin-top:0.1rem;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="card-hd">⚙️ 학습 하이퍼파라미터</div>', unsafe_allow_html=True)
            for k, v in [
                ("옵티마이저",    "Adam (lr=0.001, weight_decay=1e-4)"),
                ("손실 함수",    "Binary Cross-Entropy (BCELoss)"),
                ("학습률 스케줄", "StepLR (step=20, γ=0.5)"),
                ("에폭 / 배치",  "60 Epochs / 64"),
                ("히든 크기",    "32 → 16 뉴런"),
                ("드롭아웃",     "30%"),
                ("판정 임계값",  "0.70 (70%)"),
                ("학습:테스트",  "80% : 20%"),
                ("재현성 시드",  "42"),
            ]:
                st.markdown(f'<div class="sum-row"><span class="sum-k">{k}</span><span class="sum-v" style="font-family:monospace;font-size:0.78rem;">{v}</span></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-hd">📊 분석 특징 4가지</div>', unsafe_allow_html=True)
            for name, desc, color in [
                ("💸 가격배율",    "판매가÷정가 — 핵심 지표, 높을수록 위험",    "#ef4444"),
                ("📅 계정 나이",  "가입 후 경과 일수 — 짧을수록 위험",         "#f59e0b"),
                ("📦 동시매물수", "현재 올린 티켓 개수 — 많을수록 위험",        "#334155"),
                ("🔤 의심키워드", "26종 단어 포함 여부(0/1) — 있으면 위험 신호", "#0f172a"),
            ]:
                st.markdown(f"""
                <div style="display:flex;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid #f1f5f9;">
                  <div style="width:3px;background:{color};border-radius:2px;flex-shrink:0;"></div>
                  <div>
                    <div style="font-size:0.82rem;font-weight:700;color:#1e293b;">{name}</div>
                    <div style="font-size:0.74rem;color:#64748b;">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    results_dir = BASE / "results"
    has_img = False
    for fname, title in [
        ("training_curve.png",    "📈 PyTorch 학습 곡선 (Loss & Accuracy)"),
        ("metrics_comparison.png", "📊 3모델 성능 비교 (LR / Random Forest / PyTorch)"),
        ("confusion_matrices.png", "🎯 혼동 행렬 비교 (Confusion Matrix)"),
    ]:
        fpath = results_dir / fname
        if fpath.exists():
            has_img = True
            st.markdown(f'<div style="font-size:0.85rem;font-weight:700;color:#374151;margin:1rem 0 0.4rem;">{title}</div>', unsafe_allow_html=True)
            st.image(str(fpath), use_container_width=True)
            st.markdown("<hr>", unsafe_allow_html=True)
    if not has_img:
        st.markdown('<div class="panel-warn">⚠️ 학습 결과 이미지가 없습니다. <code>python train_models.py</code>를 실행하면 <strong>results/</strong> 폴더에 그래프가 저장됩니다.</div>', unsafe_allow_html=True)

with tab3:
    results_dir = BASE / "results"
    for fname, title in [
        ("feature_importance_combined.png",    "🔬 종합 특징 중요도 비교"),
        ("feature_importance_rf_mdi.png",      "🌲 Random Forest MDI 중요도"),
        ("feature_importance_pytorch.png",     "🧠 PyTorch 그래디언트 기반 중요도"),
        ("feature_importance_permutation.png", "🔀 Permutation Importance"),
    ]:
        fpath = results_dir / fname
        if fpath.exists():
            st.markdown(f'<div style="font-size:0.85rem;font-weight:700;color:#374151;margin:1rem 0 0.4rem;">{title}</div>', unsafe_allow_html=True)
            st.image(str(fpath), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    feats = ["가격배율\n(price_ratio)", "계정 나이\n(account_age)", "동시매물수\n(num_listings)", "의심 키워드\n(keywords)"]
    rf_imp = [0.52, 0.21, 0.18, 0.09]
    pt_imp = [0.48, 0.24, 0.19, 0.09]

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(name="Random Forest", x=rf_imp, y=feats, orientation="h",
                            marker=dict(color="rgba(245,158,11,0.8)",line=dict(color="#d97706",width=1)),
                            text=[f"{v:.0%}" for v in rf_imp], textposition="outside", textfont=dict(color="#374151")))
    fig_fi.add_trace(go.Bar(name="PyTorch NN", x=pt_imp, y=feats, orientation="h",
                            marker=dict(color="rgba(15,23,42,0.8)",line=dict(color="#0f172a",width=1)),
                            text=[f"{v:.0%}" for v in pt_imp], textposition="outside", textfont=dict(color="#374151")))
    fig_fi.update_layout(**PLOTLY_LAYOUT, barmode="group",
        xaxis=dict(title="중요도", tickformat=".0%", range=[0,0.72], gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151")), height=280)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('<div class="panel-info">💡 <strong>가격배율이 가장 중요한 특징</strong>입니다 (~50%). 정가 대비 판매가 비율이 높을수록 암표 확률이 급격히 상승합니다.</div>', unsafe_allow_html=True)
