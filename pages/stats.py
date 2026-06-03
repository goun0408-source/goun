"""pages/stats.py — 통계 대시보드"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils import THRESHOLD, PLOTLY_LAYOUT, load_reports, load_resources

model, scaler = load_resources()

st.markdown("""
<div class="hero-wrap" style="padding:2rem 2.5rem;">
  <div class="hero-chip">📊 Analytics Dashboard</div>
  <div class="hero-title" style="font-size:1.8rem;">통계 대시보드</div>
  <div class="hero-sub" style="font-size:0.88rem;">신고 데이터 기반 암표 현황을 시각적으로 분석합니다.</div>
</div>
""", unsafe_allow_html=True)

df = load_reports()
if df.empty:
    st.markdown('<div class="panel-info">📭 신고 데이터가 없습니다. <strong>🔍 암표 탐지 & 신고</strong>에서 먼저 신고를 접수해주세요.</div>', unsafe_allow_html=True)
    st.stop()

df["risk_pct"]    = (df["risk_score"]*100).round(1)
df["reported_dt"] = pd.to_datetime(df["reported_at"])
df["date"]        = df["reported_dt"].dt.date
high              = (df["risk_score"] >= THRESHOLD).sum()

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card kpi-indigo"><div class="kpi-lbl">총 신고</div><div class="kpi-val">{len(df)}</div></div>
  <div class="kpi-card kpi-rose"><div class="kpi-lbl">암표 의심 고위험</div><div class="kpi-val" style="color:#e11d48;">{high}</div></div>
  <div class="kpi-card kpi-amber"><div class="kpi-lbl">평균 위험도</div><div class="kpi-val" style="color:#d97706;">{df['risk_pct'].mean():.1f}%</div></div>
  <div class="kpi-card kpi-emerald"><div class="kpi-lbl">신고된 플랫폼</div><div class="kpi-val" style="color:#059669;">{df['platform'].nunique()}</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 플랫폼 & 분포", "💹 가격 분석", "🕒 시계열 추이"])

with tab1:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        plat_cnt = df["platform"].value_counts().reset_index()
        plat_cnt.columns = ["플랫폼","신고수"]
        fig1 = go.Figure(go.Bar(
            x=plat_cnt["플랫폼"], y=plat_cnt["신고수"],
            marker_color=["#6366f1","#8b5cf6","#a78bfa","#4f46e5","#7c3aed","#818cf8","#c4b5fd"],
            text=plat_cnt["신고수"], textposition="outside", textfont=dict(color="#374151"),
        ))
        fig1.update_layout(**PLOTLY_LAYOUT, title=dict(text="📊 플랫폼별 신고 건수", font=dict(color="#1e293b",size=13)),
                           showlegend=False, height=300, yaxis=dict(gridcolor="#e2e8f0"), xaxis=dict(gridcolor="#e2e8f0"))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        labels_bin = ["🟢 정상 (0~39%)", "🟡 주의 (40~69%)", "🔴 암표의심 (70%+)"]
        df["구간"] = pd.cut(df["risk_pct"], bins=[0,40,70,100], labels=labels_bin)
        dist = df["구간"].value_counts().reindex(labels_bin).fillna(0).reset_index()
        dist.columns = ["구간","건수"]
        fig2 = go.Figure(go.Pie(
            labels=dist["구간"], values=dist["건수"], hole=0.52,
            marker=dict(colors=["rgba(16,185,129,0.85)","rgba(245,158,11,0.85)","rgba(239,68,68,0.85)"],
                        line=dict(color="white",width=3)),
            textfont=dict(color="white",size=12),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title=dict(text="🎯 위험도 구간 분포", font=dict(color="#1e293b",size=13)),
                           legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151")), height=300)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        fig3 = go.Figure(go.Scatter(
            x=df["price_ratio"], y=df["risk_pct"], mode="markers",
            marker=dict(
                color=df["risk_pct"],
                colorscale=[[0,"#10b981"],[0.4,"#f59e0b"],[0.7,"#ef4444"],[1,"#dc2626"]],
                size=10, opacity=0.85, line=dict(color="white",width=1),
                showscale=True,
                colorbar=dict(title=dict(text="위험도%",font=dict(color="#374151")), tickfont=dict(color="#374151")),
            ),
            text=[f"{row['event_name']}<br>위험: {row['risk_pct']}%<br>배율: {row['price_ratio']:.2f}×" for _,row in df.iterrows()],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig3.add_vline(x=1.5, line=dict(color="#f59e0b",dash="dash",width=1.5), annotation_text="1.5×", annotation_font_color="#d97706")
        fig3.add_hline(y=70, line=dict(color="#ef4444",dash="dash",width=1.5), annotation_text="70% 기준", annotation_font_color="#dc2626")
        fig3.update_layout(**PLOTLY_LAYOUT, title=dict(text="💹 가격배율 vs 위험도 산포도", font=dict(color="#1e293b",size=13)),
                           xaxis=dict(title="가격배율",gridcolor="#e2e8f0"), yaxis=dict(title="AI 위험도 (%)",gridcolor="#e2e8f0"), height=300)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = go.Figure(go.Histogram(
            x=df["price_ratio"], nbinsx=20,
            marker=dict(color="rgba(99,102,241,0.75)", line=dict(color="#6366f1",width=1)),
        ))
        fig4.add_vline(x=df["price_ratio"].mean(), line=dict(color="#f59e0b",dash="dash"),
                       annotation_text=f"평균 {df['price_ratio'].mean():.2f}×", annotation_font_color="#d97706")
        fig4.update_layout(**PLOTLY_LAYOUT, title=dict(text="📊 가격배율 분포", font=dict(color="#1e293b",size=13)),
                           xaxis=dict(title="가격배율",gridcolor="#e2e8f0"), yaxis=dict(title="건수",gridcolor="#e2e8f0"),
                           showlegend=False, height=300)
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    daily = df.groupby("date").agg(신고수=("id","count"), 평균위험도=("risk_pct","mean")).reset_index()
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=daily["date"], y=daily["신고수"], name="신고 건수", marker_color="rgba(99,102,241,0.7)", yaxis="y"))
    fig5.add_trace(go.Scatter(x=daily["date"], y=daily["평균위험도"], name="평균 위험도(%)",
                              mode="lines+markers", line=dict(color="#ef4444",width=2.5), marker=dict(size=7,color="#ef4444"), yaxis="y2"))
    fig5.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="🕒 일별 신고 & 평균 위험도 추이", font=dict(color="#1e293b",size=13)),
        yaxis=dict(title="신고 건수", gridcolor="#e2e8f0"),
        yaxis2=dict(title="평균 위험도 (%)", overlaying="y", side="right", range=[0,100], showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151")),
        barmode="overlay", height=340,
    )
    st.plotly_chart(fig5, use_container_width=True)

    if "report_type" in df.columns:
        type_cnt = df["report_type"].value_counts().reset_index()
        type_cnt = type_cnt[type_cnt["report_type"].ne("")]
        if not type_cnt.empty:
            type_cnt.columns = ["신고유형","건수"]
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.88rem;font-weight:700;color:#1e293b;margin-bottom:0.6rem;">📋 신고 유형별 현황</div>', unsafe_allow_html=True)
            st.dataframe(type_cnt, use_container_width=True, hide_index=True)
