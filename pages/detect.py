"""pages/detect.py — 암표 탐지 & 신고"""
from datetime import datetime, date
import streamlit as st
import plotly.graph_objects as go

from utils import (
    PLATFORMS, SEAT_GRADES, REPORT_TYPES, GOVT_URL, PLOTLY_LAYOUT,
    load_resources, save_report, find_keywords, analyze_signals, risk_meta, predict,
)

model, scaler = load_resources()

st.markdown("""
<div class="hero-wrap" style="padding:2rem 2.5rem;">
  <div class="hero-chip">🔍 Detection & Report</div>
  <div class="hero-title" style="font-size:1.8rem;">암표 탐지 & 신고</div>
  <div class="hero-sub" style="font-size:0.88rem;">
    의심 매물 정보를 입력하면 AI가 4가지 신호를 분석해 위험도를 판별하고 신고서를 자동 생성합니다.
  </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown('<div class="panel-danger">⚠️ <strong>모델 파일이 없습니다.</strong> 터미널에서 <code>python train_models.py</code>를 실행해주세요.</div>', unsafe_allow_html=True)
    st.stop()

# ══ STEP 1 ══════════════════════════════════
st.markdown("""
<div class="sec-header">
  <div class="sec-num">1</div>
  <div class="sec-title">매물 정보 입력</div>
  <div class="sec-sub">발견한 의심 매물의 정보를 입력하세요</div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    with st.container(border=True):
        st.markdown('<div class="card-hd">🎪 이벤트 정보</div>', unsafe_allow_html=True)
        event_name = st.text_input("공연/행사명 *", placeholder="예: 아이유 콘서트 HEREH WORLD TOUR", key="ev_name")
        ev1, ev2 = st.columns(2)
        with ev1:
            event_date = st.date_input("행사 날짜", value=date.today(), key="ev_date")
        with ev2:
            seat_grade = st.selectbox("좌석 등급", SEAT_GRADES, key="seat_grade")
        venue = st.text_input("공연장", placeholder="예: 잠실종합운동장", key="venue")

    with st.container(border=True):
        st.markdown('<div class="card-hd">🌐 플랫폼 & 판매자</div>', unsafe_allow_html=True)
        platform = st.selectbox("발견 플랫폼 *", PLATFORMS, key="platform")
        p1, p2 = st.columns(2)
        with p1:
            seller_id = st.text_input("판매자 ID / 닉네임", placeholder="ticket_king99", key="seller_id")
        with p2:
            listing_url = st.text_input("매물 URL (선택)", placeholder="https://...", key="listing_url")

with col_r:
    with st.container(border=True):
        st.markdown('<div class="card-hd">💰 가격 정보</div>', unsafe_allow_html=True)
        pr1, pr2 = st.columns(2)
        with pr1:
            face_price = st.number_input("정가 (원) *", 1_000, 2_000_000, 110_000, 1_000, key="face_price")
        with pr2:
            sell_price = st.number_input("판매 요구가 (원) *", 1_000, 5_000_000, 350_000, 1_000, key="sell_price")

        if face_price > 0:
            ratio  = sell_price / face_price
            profit = sell_price - face_price
            bar_w  = min(ratio / 4 * 100, 100)
            clr, bg, bd = (
                ("#dc2626", "#fef2f2", "#fca5a5") if ratio >= 2.0 else
                ("#d97706", "#fffbeb", "#fcd34d") if ratio >= 1.3 else
                ("#16a34a", "#f0fdf4", "#86efac")
            )
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {bd};border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.4rem;">
              <div style="display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="font-size:1.9rem;font-weight:900;color:{clr};">{ratio:.2f}배</span>
                <span style="font-size:0.85rem;font-weight:700;color:{clr};">{"+" if profit>=0 else ""}{profit:,}원</span>
                <span style="font-size:0.72rem;color:#94a3b8;margin-left:auto;">가격 배율</span>
              </div>
              <div style="background:#e2e8f0;border-radius:99px;height:7px;overflow:hidden;">
                <div style="width:{bar_w}%;height:100%;background:{clr};border-radius:99px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#94a3b8;margin-top:0.25rem;">
                <span>1.0×</span><span>2.0×</span><span>3.0×</span><span>4.0×+</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-hd">👤 판매자 정보</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            account_age = st.number_input("계정 나이 (일) *", 1, 5000, 14, 1, help="가입일로부터 오늘까지의 일수", key="acc_age")
        with s2:
            num_listings = st.number_input("동시 매물 수 *", 1, 200, 15, 1, help="판매자가 현재 올린 티켓 매물 총 개수", key="num_list")

    with st.container(border=True):
        st.markdown('<div class="card-hd">📝 판매 설명글 (키워드 자동 탐지)</div>', unsafe_allow_html=True)
        description = st.text_area(
            "게시글 내용 붙여넣기",
            placeholder="급처합니다~ VIP 2연석 양도\n정가이상 웃돈 가능, 프리미엄 드립니다\n마지막 1장, 특석 급매!",
            height=105, key="desc",
        )

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_btn = st.button("🔍  AI 위험도 분석 시작", type="primary", use_container_width=True, key="analyze")

st.markdown("<hr>", unsafe_allow_html=True)

# ══ STEP 2 ══════════════════════════════════
if analyze_btn or "result" in st.session_state:
    if analyze_btn:
        if not event_name.strip():
            st.markdown('<div class="panel-warn">⚠️ 공연/행사명을 입력해주세요.</div>', unsafe_allow_html=True)
            st.stop()
        ratio_val = sell_price / face_price
        found_kws = find_keywords(description)
        risk      = predict(model, scaler, [ratio_val, account_age, num_listings, int(bool(found_kws))])
        signals   = analyze_signals(ratio_val, account_age, num_listings, found_kws, sell_price, face_price)
        st.session_state["result"] = {
            "event_name": event_name, "event_date": str(event_date),
            "venue": venue, "seat_grade": seat_grade, "platform": platform,
            "seller_id": seller_id, "listing_url": listing_url,
            "face_price": face_price, "sell_price": sell_price, "price_ratio": ratio_val,
            "account_age_days": account_age, "num_listings": num_listings,
            "has_keywords": int(bool(found_kws)), "description": description,
            "found_kws": found_kws, "risk_score": risk, "signals": signals,
            "ai_signals": " | ".join(f"[{s[2]}]" for s in signals if s[0]!="ok"),
            "report_type": "", "evidence_note": "", "submitted": False,
        }

    res = st.session_state["result"]
    risk = res["risk_score"]
    pct, risk_cls, verdict, risk_color = risk_meta(risk)

    st.markdown("""
    <div class="sec-header">
      <div class="sec-num" style="background:#059669;">2</div>
      <div class="sec-title">AI 분석 결과</div>
      <div class="sec-sub">딥러닝 4특징 기반 위험도 판정</div>
    </div>
    """, unsafe_allow_html=True)

    res1, res2, res3 = st.columns([1, 1.1, 1.4], gap="medium")

    with res1:
        st.markdown(f"""
        <div class="risk-box {risk_cls}">
          <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;margin-bottom:0.3rem;">AI 위험 점수</div>
          <div class="risk-pct">{pct}<span style="font-size:1.4rem;">%</span></div>
          <div class="risk-lbl">{verdict}</div>
          <div class="risk-track"><div class="risk-bar" style="width:{pct}%"></div></div>
          <div class="risk-scale"><span>안전</span><span>│70%</span><span>위험</span></div>
        </div>
        """, unsafe_allow_html=True)
        profit = res["sell_price"] - res["face_price"]
        st.markdown(f"""
        <div class="summary-card" style="margin-top:0.8rem;">
          <div class="sum-row"><span class="sum-k">공연명</span><span class="sum-v" style="font-size:0.78rem;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{res['event_name']}</span></div>
          <div class="sum-row"><span class="sum-k">좌석</span><span class="sum-v">{res['seat_grade']} · {res['event_date']}</span></div>
          <div class="sum-row"><span class="sum-k">플랫폼</span><span class="sum-v">{res['platform']}</span></div>
          <div class="sum-row"><span class="sum-k">정가</span><span class="sum-v">{res['face_price']:,}원</span></div>
          <div class="sum-row"><span class="sum-k">판매가</span><span class="sum-v" style="color:#dc2626;">{res['sell_price']:,}원</span></div>
          <div class="sum-row"><span class="sum-k">프리미엄</span><span class="sum-v" style="color:#dc2626;">+{profit:,}원 ({res['price_ratio']:.2f}×)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        lvl_map = {"danger": 90, "warn": 55, "ok": 15}
        cats = ["가격배율", "계정나이", "매물수", "키워드"]
        vals = [lvl_map.get(s[0], 15) for s in res["signals"]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            line=dict(color="#ef4444", width=2.5), fillcolor="rgba(239,68,68,0.15)",
        ))
        fig_r.update_layout(**{
            **PLOTLY_LAYOUT, "margin": dict(l=15,r=15,t=30,b=15),
            "polar": dict(
                bgcolor="rgba(248,250,252,0.8)",
                radialaxis=dict(visible=True, range=[0,100], color="#cbd5e1", gridcolor="#e2e8f0"),
                angularaxis=dict(color="#64748b"),
            ),
            "showlegend": False, "height": 230,
        })
        st.plotly_chart(fig_r, use_container_width=True)

        if res["found_kws"]:
            kw_html = "".join(f'<span class="kw-tag">🔍 {k}</span>' for k in res["found_kws"])
            st.markdown(f"""
            <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:0.7rem 0.9rem;">
              <div style="font-size:0.7rem;font-weight:700;color:#78350f;margin-bottom:0.4rem;">🔤 탐지된 의심 키워드</div>
              {kw_html}
            </div>
            """, unsafe_allow_html=True)

    with res3:
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;">항목별 위험 신호 분석</div>', unsafe_allow_html=True)
        lvl_css = {"danger":("sig-danger","sb-danger"), "warn":("sig-warn","sb-warn"), "ok":("sig-ok","sb-ok")}
        for lvl, icon, msg, badge_txt in res["signals"]:
            row_c, badge_c = lvl_css[lvl]
            st.markdown(f"""
            <div class="sig-item {row_c}">
              <span class="sig-icon">{icon}</span>
              <span class="sig-msg">{msg}</span>
              <span class="sig-badge {badge_c}">{badge_txt}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══ STEP 3 ══════════════════════════════
    st.markdown("""
    <div class="sec-header">
      <div class="sec-num" style="background:#dc2626;">3</div>
      <div class="sec-title">신고 접수</div>
      <div class="sec-sub">신고 유형 선택 후 공식 접수</div>
    </div>
    """, unsafe_allow_html=True)

    if risk < 0.70:
        st.markdown(f'<div class="panel-ok">✅ <strong>AI 판정: 정상 거래</strong> — 위험 점수 {pct}% (기준 70% 미만)<br><span>그래도 의심된다면 → <a href="{GOVT_URL}" target="_blank">문화체육관광부 암표 신고</a></span></div>', unsafe_allow_html=True)

    elif res.get("submitted"):
        st.markdown(f"""
        <div class="success-box">
          <div style="font-size:2.5rem;">✅</div>
          <div class="success-title">신고가 접수되었습니다!</div>
          <div class="success-sub">신고 내역 페이지에서 확인 가능합니다.<br>공식 신고도 함께 권장합니다 → <a href="{GOVT_URL}" target="_blank" style="color:#4f46e5;">신고 누리집</a></div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➕ 새 매물 분석하기", type="secondary", use_container_width=True):
            del st.session_state["result"]
            st.rerun()
    else:
        st.markdown('<div class="panel-warn">⚠️ AI 분석은 통계적 참고 자료입니다. <strong>최종 신고 여부는 본인이 판단</strong>하세요. 허위·과장 신고는 법적 책임이 발생합니다.</div>', unsafe_allow_html=True)

        with st.form("report_form"):
            st.markdown('<div class="report-box-title">🚨 신고 양식</div>', unsafe_allow_html=True)
            rf1, rf2 = st.columns([2, 1])
            with rf1:
                report_type = st.selectbox("신고 유형 *", REPORT_TYPES, key="rtype")
            with rf2:
                st.markdown(f'<div style="margin-top:1.6rem;text-align:center;background:#fef2f2;border:1.5px solid #fca5a5;border-radius:8px;padding:0.5rem;"><div style="font-size:0.68rem;color:#94a3b8;">AI 위험 점수</div><div style="font-size:1.6rem;font-weight:900;color:#dc2626;">{pct}%</div></div>', unsafe_allow_html=True)

            evidence_note = st.text_area("추가 증거 / 피해 상황", placeholder="예: 입금 후 연락 두절, 동일 계정에서 50장 이상 판매 중...", height=75, key="evidence")

            copy_text = "\n".join([
                "[ScalpGuard AI 암표 의심 신고]",
                f"신고 시각     : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"AI 위험 점수  : {pct}%  ({verdict})",
                "─" * 40,
                f"공연명        : {res['event_name']}",
                f"행사 날짜     : {res['event_date']}",
                f"공연장        : {res['venue'] or '미입력'}",
                f"좌석 등급     : {res['seat_grade']}",
                "─" * 40,
                f"발견 플랫폼   : {res['platform']}",
                f"판매자 ID     : {res['seller_id'] or '미입력'}",
                f"매물 URL      : {res['listing_url'] or '미입력'}",
                "─" * 40,
                f"정가          : {res['face_price']:,}원",
                f"판매 요구가   : {res['sell_price']:,}원",
                f"가격배율      : {res['price_ratio']:.2f}배  (+{res['sell_price']-res['face_price']:,}원)",
                f"판매자 계정   : {res['account_age_days']}일",
                f"동시 매물수   : {res['num_listings']}개",
                f"의심 키워드   : {', '.join(res['found_kws']) if res['found_kws'] else '없음'}",
            ])
            with st.expander("📋 신고서 미리보기 (복사해서 정부 신고처에 제출)", expanded=True):
                st.code(copy_text, language="")

            st.markdown(f'🔗 **공식 신고처**: [문화체육관광부 암표 신고 누리집]({GOVT_URL})')

            agreed = st.checkbox("✔ AI 판정은 참고용이며, 신고의 최종 판단은 제가 직접 합니다. 허위 신고의 법적 책임을 인지합니다.", key="agree")
            submitted = st.form_submit_button("🚨  신고 접수하기", type="primary", use_container_width=True)
            if submitted:
                if not agreed:
                    st.error("동의 체크박스를 체크해주세요.")
                else:
                    res["report_type"] = report_type
                    res["evidence_note"] = evidence_note
                    save_report(res)
                    st.session_state["result"]["submitted"] = True
                    st.rerun()
