"""pages/history.py — 신고 내역"""
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils import PLATFORMS, THRESHOLD, PLOTLY_LAYOUT, load_reports, load_resources

model, scaler = load_resources()

st.markdown("""
<div class="hero-wrap" style="padding:2rem 2.5rem;">
  <div class="hero-chip">📋 Report History</div>
  <div class="hero-title" style="font-size:1.8rem;">신고 내역 관리</div>
  <div class="hero-sub" style="font-size:0.88rem;">접수된 암표 의심 신고를 조회·필터링·내보내기합니다.</div>
</div>
""", unsafe_allow_html=True)

df = load_reports()
if df.empty:
    st.markdown('<div class="panel-info">📭 아직 접수된 신고가 없습니다. <strong>🔍 암표 탐지 & 신고</strong> 페이지에서 신고를 접수해보세요.</div>', unsafe_allow_html=True)
    st.stop()

df["risk_pct"] = (df["risk_score"] * 100).round(1)
high      = (df["risk_score"] >= THRESHOLD).sum()
avg_r     = df["risk_score"].mean() * 100
avg_ratio = df["price_ratio"].mean()

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card kpi-indigo"><div class="kpi-lbl">총 신고 건수</div><div class="kpi-val">{len(df)}</div></div>
  <div class="kpi-card kpi-rose"><div class="kpi-lbl">고위험 신고 (70%+)</div><div class="kpi-val" style="color:#e11d48;">{high}</div></div>
  <div class="kpi-card kpi-amber"><div class="kpi-lbl">평균 위험 점수</div><div class="kpi-val" style="color:#d97706;">{avg_r:.1f}%</div></div>
  <div class="kpi-card kpi-emerald"><div class="kpi-lbl">평균 가격 배율</div><div class="kpi-val" style="color:#059669;">{avg_ratio:.2f}×</div></div>
</div>
""", unsafe_allow_html=True)

# 필터
with st.container(border=True):
    st.markdown('<div class="card-hd">🔎 필터 & 정렬</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4, gap="medium")
    with f1:
        sel_platform = st.multiselect("플랫폼", PLATFORMS, default=[], key="flt_plat")
    with f2:
        min_risk = st.slider("최소 위험 점수 (%)", 0, 100, 0, key="flt_risk")
    with f3:
        sel_kw = st.text_input("신고 유형 키워드", placeholder="가격, 허위...", key="flt_kw")
    with f4:
        sort_by = st.selectbox("정렬 기준", ["신고 시각 ↓", "위험 점수 ↓", "가격배율 ↓"], key="sort_by")

filtered = df.copy()
if sel_platform:
    filtered = filtered[filtered["platform"].isin(sel_platform)]
if min_risk > 0:
    filtered = filtered[filtered["risk_score"] * 100 >= min_risk]
if sel_kw:
    filtered = filtered[filtered["report_type"].str.contains(sel_kw, na=False)]
if sort_by == "위험 점수 ↓":
    filtered = filtered.sort_values("risk_score", ascending=False)
elif sort_by == "가격배율 ↓":
    filtered = filtered.sort_values("price_ratio", ascending=False)

if filtered.empty:
    st.markdown('<div class="panel-info">🔍 조건에 맞는 신고가 없습니다.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div style="font-size:0.82rem;color:#64748b;margin-bottom:0.6rem;">검색 결과 <strong style="color:#0f172a;">{len(filtered)}</strong>건 (전체 {len(df)}건)</div>', unsafe_allow_html=True)
    display = filtered[["id","reported_at","event_name","platform","seller_id","face_price","sell_price","price_ratio","account_age_days","num_listings","risk_score","report_type"]].copy()
    display["risk_score"]  = (display["risk_score"]*100).round(1).astype(str)+"%"
    display["price_ratio"] = display["price_ratio"].round(2).astype(str)+"배"
    display["face_price"]  = display["face_price"].apply(lambda x: f"{int(x):,}원")
    display["sell_price"]  = display["sell_price"].apply(lambda x: f"{int(x):,}원")
    display.columns = ["ID","신고시각","공연명","플랫폼","판매자ID","정가","판매가","배율","계정(일)","매물수","위험점수","신고유형"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=380)

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        csv = filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️  CSV 내보내기", csv, f"scalpguard_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# 미니 차트
st.markdown("<hr>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2, gap="medium")

with ch1:
    plat_cnt = df["platform"].value_counts().reset_index()
    plat_cnt.columns = ["플랫폼", "신고수"]
    fig_p = go.Figure(go.Bar(
        x=plat_cnt["플랫폼"], y=plat_cnt["신고수"],
        marker_color=["#0f172a","#334155","#475569","#0f172a","#1e293b","#334155","#94a3b8"],
        text=plat_cnt["신고수"], textposition="outside", textfont=dict(color="#374151"),
    ))
    fig_p.update_layout(**PLOTLY_LAYOUT, title=dict(text="플랫폼별 신고 건수", font=dict(color="#1e293b",size=13)),
                        showlegend=False, height=260, yaxis=dict(gridcolor="#e2e8f0"), xaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(fig_p, use_container_width=True)

with ch2:
    labels_bin = ["🟢 정상 (0~39%)", "🟡 주의 (40~69%)", "🔴 암표의심 (70%+)"]
    df["구간"] = pd.cut(df["risk_pct"], bins=[0,40,70,100], labels=labels_bin)
    dist = df["구간"].value_counts().reindex(labels_bin).fillna(0).reset_index()
    dist.columns = ["구간","건수"]
    fig_d = go.Figure(go.Pie(
        labels=dist["구간"], values=dist["건수"], hole=0.5,
        marker=dict(colors=["rgba(16,185,129,0.85)","rgba(245,158,11,0.85)","rgba(239,68,68,0.85)"],
                    line=dict(color="white", width=3)),
        textfont=dict(color="white", size=12),
    ))
    fig_d.update_layout(**PLOTLY_LAYOUT, title=dict(text="위험도 구간 분포", font=dict(color="#1e293b",size=13)),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151")), height=260)
    st.plotly_chart(fig_d, use_container_width=True)
