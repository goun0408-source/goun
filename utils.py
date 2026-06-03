"""
utils.py — ScalpGuard 공통 유틸리티
모든 페이지에서 import하여 사용
"""

import sqlite3
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ══════════════════════════════════════════════
# 경로 & 상수
# ══════════════════════════════════════════════
BASE        = Path(__file__).parent
MODEL_PATH  = BASE / "model.pth"
SCALER_PATH = BASE / "scaler.pkl"
DB_PATH     = BASE / "reports.db"
THRESHOLD   = 0.70

PLATFORMS = ["티켓베이", "중고나라", "당근마켓", "네이버 카페", "번개장터", "인스타그램/SNS", "기타"]
SEAT_GRADES = ["VIP", "R석", "S석", "A석", "B석", "스탠딩", "기타"]
REPORT_TYPES = [
    "💰 가격 폭리 (정가 대비 과도한 프리미엄)",
    "🚫 허위 매물 (실제로 티켓 없음)",
    "📦 대량 매집 (다수 티켓 독점 구매 후 재판매)",
    "🎭 사기 의심 (입금 후 잠적 등)",
    "🤖 봇 구매 의심 (자동화 도구 사용)",
    "📋 기타 부정 거래",
]
SUSPICIOUS_WORDS = [
    "급처", "프리미엄", "정가이상", "웃돈", "급매", "고가", "희귀",
    "한정", "vip", "특가", "양도", "마지막", "최고가", "웃돈받고",
    "원가이상", "리셀", "resell", "피양도", "피양", "정가+",
    "최저가", "특석", "전석", "전체", "연석", "단석",
]
GOVT_URL = "https://www.culture.go.kr/singo"

# Plotly 공통 레이아웃 (라이트 테마)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.8)",
    font=dict(color="#374151", family="Noto Sans KR"),
    margin=dict(l=0, r=0, t=40, b=0),
)


# ══════════════════════════════════════════════
# 전역 CSS — 밝은 라이트 테마
# ══════════════════════════════════════════════
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
/* 팔레트 C — Slate + Amber 미니멀: 구조=slate(#0f172a), 포인트 1색=amber(#f59e0b) */

/* ── 기본 ── */
*, *::before, *::after {
    font-family: 'Noto Sans KR', sans-serif !important;
    box-sizing: border-box;
}

/* ── Material 아이콘 폰트 복구 (위 * 규칙이 덮어쓰는 문제 해결) ── */
[data-testid="stIconMaterial"],
.material-icons, .material-icons-outlined,
.material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
}

/* ── st.container(border=True) 를 카드 스타일로 ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    padding: 0.3rem 0.4rem;
}

/* ── 배경 ── */
.stApp { background: #f1f5f9 !important; }
.main .block-container {
    padding: 1.8rem 2.2rem 3rem !important;
    max-width: 1180px !important;
}

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }

/* ── 숨기기: 메뉴/배포 버튼만 숨기고 헤더·툴바 자체는 유지 ──
   (헤더나 툴바 전체를 숨기면 사이드바 펼치기 버튼까지 사라져 다시 못 펴게 됨) */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stMainMenu"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── 사이드바 접기/펼치기 버튼은 항상 보이게 (접혀도 다시 펼 수 있도록) ── */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * {
    visibility: visible !important;
}

/* ── 히어로 배너 ── */
.hero-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(15,23,42,0.18);
}
.hero-wrap::before {
    content: "";
    position: absolute; top: -60px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.hero-wrap::after {
    content: "";
    position: absolute; bottom: -50px; right: 120px;
    width: 150px; height: 150px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 999px; padding: 0.25rem 0.85rem;
    font-size: 0.72rem; font-weight: 700; color: white;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.2rem; font-weight: 900; color: white;
    line-height: 1.25; margin: 0 0 0.7rem;
}
.hero-sub {
    font-size: 0.95rem; color: rgba(255,255,255,0.85);
    line-height: 1.8; margin: 0; max-width: 560px;
}
.hero-stats {
    display: flex; gap: 2.5rem; margin-top: 1.8rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.2);
}
.hero-stat-val { font-size: 1.5rem; font-weight: 900; color: #fbbf24; }
.hero-stat-lbl { font-size: 0.72rem; color: rgba(255,255,255,0.7); margin-top: 0.1rem; }

/* ── 섹션 헤더 ── */
.sec-header {
    display: flex; align-items: center; gap: 0.8rem;
    margin: 1.8rem 0 1rem;
}
.sec-num {
    width: 30px; height: 30px; border-radius: 8px;
    background: #0f172a; color: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; font-weight: 800; flex-shrink: 0;
}
.sec-title { font-size: 1rem; font-weight: 700; color: #1e293b; }
.sec-sub   { font-size: 0.76rem; color: #94a3b8; margin-left: auto; white-space: nowrap; }

/* ── 카드 ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-color: #cbd5e1; }
.card-hd {
    font-size: 0.72rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid #f1f5f9;
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── 가격 인디케이터 ── */
.price-box {
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 0.6rem;
    display: flex; align-items: center; justify-content: space-between;
    border: 1.5px solid;
}
.price-ratio-num { font-size: 1.8rem; font-weight: 900; }
.price-profit    { font-size: 0.82rem; font-weight: 600; }

/* ── KPI 카드 ── */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-indigo::after { background: linear-gradient(90deg,#0f172a,#334155); }
.kpi-rose::after   { background: linear-gradient(90deg,#f43f5e,#fb7185); }
.kpi-emerald::after { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-amber::after  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-val { font-size: 1.9rem; font-weight: 900; color: #1e293b; margin-top: 0.3rem; }
.kpi-lbl { font-size: 0.7rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.07em; }

/* ── 리스크 게이지 ── */
.risk-box {
    border-radius: 16px; padding: 1.8rem;
    border: 2px solid; text-align: center;
    background: white;
}
.risk-safe   { border-color: #10b981; background: #f0fdf4; }
.risk-warn   { border-color: #f59e0b; background: #fffbeb; }
.risk-danger { border-color: #ef4444; background: #fef2f2; }
.risk-pct { font-size: 3.5rem; font-weight: 900; line-height: 1; }
.risk-safe   .risk-pct { color: #10b981; }
.risk-warn   .risk-pct { color: #f59e0b; }
.risk-danger .risk-pct { color: #ef4444; }
.risk-lbl { font-size: 1rem; font-weight: 700; margin-top: 0.4rem; }
.risk-safe   .risk-lbl { color: #059669; }
.risk-warn   .risk-lbl { color: #d97706; }
.risk-danger .risk-lbl { color: #dc2626; }
.risk-track { background: #e2e8f0; border-radius: 99px; height: 8px; margin: 1rem 0 0.3rem; overflow: hidden; }
.risk-bar   { height: 100%; border-radius: 99px; }
.risk-safe   .risk-bar { background: linear-gradient(90deg,#059669,#10b981); }
.risk-warn   .risk-bar { background: linear-gradient(90deg,#d97706,#f59e0b); }
.risk-danger .risk-bar { background: linear-gradient(90deg,#dc2626,#ef4444); }
.risk-scale { display:flex; justify-content:space-between; font-size:0.66rem; color:#94a3b8; }

/* ── 신호 항목 ── */
.sig-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.65rem 1rem; border-radius: 10px;
    border: 1px solid; margin: 0.35rem 0;
    background: white;
}
.sig-danger { border-color: #fca5a5; background: #fef2f2; }
.sig-warn   { border-color: #fcd34d; background: #fffbeb; }
.sig-ok     { border-color: #6ee7b7; background: #f0fdf4; }
.sig-icon   { font-size: 1rem; flex-shrink: 0; }
.sig-msg    { flex: 1; font-size: 0.84rem; color: #374151; line-height: 1.4; }
.sig-badge  { font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 5px; white-space: nowrap; }
.sb-danger  { background: #fee2e2; color: #dc2626; }
.sb-warn    { background: #fef3c7; color: #d97706; }
.sb-ok      { background: #dcfce7; color: #16a34a; }

/* ── 키워드 태그 ── */
.kw-tag {
    display: inline-flex; align-items: center;
    background: #fef3c7; border: 1px solid #fcd34d;
    border-radius: 6px; padding: 0.18rem 0.55rem;
    font-size: 0.74rem; font-weight: 600; color: #92400e;
    margin: 0.15rem;
}

/* ── 요약 패널 ── */
.summary-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.1rem 1.3rem;
}
.sum-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.38rem 0; font-size: 0.82rem;
    border-bottom: 1px solid #f1f5f9;
}
.sum-row:last-child { border-bottom: none; }
.sum-k { color: #64748b; }
.sum-v { color: #1e293b; font-weight: 600; }

/* ── 신고 폼 박스 ── */
.report-box {
    background: #fff5f5;
    border: 1.5px solid #fca5a5;
    border-radius: 16px; padding: 1.6rem 1.8rem;
    margin-top: 0.8rem;
}
.report-box-title {
    font-size: 1rem; font-weight: 800; color: #dc2626;
    display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;
}

/* ── 성공 박스 ── */
.success-box {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 14px; padding: 1.5rem;
    text-align: center;
}
.success-title { font-size: 1.1rem; font-weight: 800; color: #16a34a; margin-top: 0.4rem; }
.success-sub   { font-size: 0.84rem; color: #64748b; margin-top: 0.3rem; line-height: 1.6; }

/* ── 알림 패널 ── */
.panel-warn {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #78350f; margin: 0.7rem 0; line-height: 1.7;
}
.panel-info {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #1e40af; margin: 0.7rem 0; line-height: 1.7;
}
.panel-ok {
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #15803d; margin: 0.7rem 0; line-height: 1.7;
}
.panel-danger {
    background: #fef2f2; border: 1px solid #fca5a5;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.84rem; color: #991b1b; margin: 0.7rem 0; line-height: 1.7;
}

/* ── 분리선 ── */
hr { border: none !important; border-top: 1.5px solid #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* ── 버튼 ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#0f172a,#0f172a) !important;
    border: none !important; color: white !important; font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(15,23,42,0.18) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(15,23,42,0.28) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important; border: 1.5px solid #0f172a !important;
    color: #0f172a !important; font-weight: 700 !important; border-radius: 10px !important;
}

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 10px !important; padding: 0.2rem !important;
    gap: 0.2rem !important;
    border: 1px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-size: 0.82rem !important;
    font-weight: 600 !important; color: #64748b !important;
    padding: 0.4rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #b45309 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    border-bottom: 2px solid #f59e0b !important;
}

/* ── 입력 필드 ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important; color: #1e293b !important;
    font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #0f172a !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.18) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label {
    color: #374151 !important; font-size: 0.84rem !important; font-weight: 600 !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }

/* ── 사이드바 로고 영역 ── */
.sb-logo {
    padding: 1.3rem 1rem 1rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 0.8rem;
}
.sb-logo-title { font-size: 1.05rem; font-weight: 900; color: #1e293b; }
.sb-logo-sub   { font-size: 0.68rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; }

/* ── 모델 상태 ── */
.model-status {
    margin-top: 1.5rem; padding: 0.9rem;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px;
}
.ms-title { font-size: 0.68rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; letter-spacing: 0.08em; }
.ms-row   { display: flex; align-items: center; gap: 0.4rem; font-size: 0.76rem; color: #374151; }
.dot-ok   { width: 7px; height: 7px; border-radius: 50%; background: #10b981; flex-shrink: 0; }
.dot-err  { width: 7px; height: 7px; border-radius: 50%; background: #ef4444; flex-shrink: 0; }

/* ── legend 범주 ── */
.legend-row {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0; font-size: 0.78rem; color: #374151;
}
.legend-dot {
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
}
</style>
"""


# ══════════════════════════════════════════════
# PyTorch 모델
# ══════════════════════════════════════════════
class ScalpDetector(nn.Module):
    def __init__(self, input_dim=4, h1=32, h2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, 1),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(1)


@st.cache_resource(show_spinner="AI 모델 로딩 중…")
def load_resources():
    if not MODEL_PATH.exists():
        return None, None
    mdl = ScalpDetector()
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    mdl.eval()
    with open(SCALER_PATH, "rb") as f:
        sc = pickle.load(f)
    return mdl, sc


# ══════════════════════════════════════════════
# SQLite
# ══════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            reported_at      TEXT,
            event_name       TEXT,
            event_date       TEXT,
            venue            TEXT,
            seat_grade       TEXT,
            platform         TEXT,
            seller_id        TEXT,
            listing_url      TEXT,
            face_price       INTEGER,
            sell_price       INTEGER,
            price_ratio      REAL,
            account_age_days INTEGER,
            num_listings     INTEGER,
            has_keywords     INTEGER,
            description      TEXT,
            report_type      TEXT,
            evidence_note    TEXT,
            risk_score       REAL,
            ai_signals       TEXT
        )
    """)
    conn.commit(); conn.close()


def save_report(r: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO reports
          (reported_at, event_name, event_date, venue, seat_grade,
           platform, seller_id, listing_url, face_price, sell_price, price_ratio,
           account_age_days, num_listings, has_keywords, description,
           report_type, evidence_note, risk_score, ai_signals)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        r["event_name"], r["event_date"], r["venue"], r["seat_grade"],
        r["platform"], r["seller_id"], r["listing_url"],
        r["face_price"], r["sell_price"], r["price_ratio"],
        r["account_age_days"], r["num_listings"], r["has_keywords"], r["description"],
        r["report_type"], r["evidence_note"], r["risk_score"], r["ai_signals"],
    ))
    conn.commit(); conn.close()


def load_reports() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reports ORDER BY reported_at DESC", conn)
    conn.close()
    return df


# ══════════════════════════════════════════════
# AI 분석 로직
# ══════════════════════════════════════════════
def predict(mdl, sc, features):
    X = np.array([features], dtype=np.float32)
    t = torch.tensor(sc.transform(X), dtype=torch.float32)
    with torch.no_grad():
        return float(mdl(t).item())


def find_keywords(text: str) -> list:
    tl = text.lower()
    return [kw for kw in SUSPICIOUS_WORDS if kw in tl]


def analyze_signals(price_ratio, account_age, num_listings, found_kws, sell_price, face_price):
    signals = []
    profit = sell_price - face_price

    if price_ratio >= 3.0:
        signals.append(("danger", "💸", f"정가 대비 {price_ratio:.1f}배 — 초과 프리미엄 (+{profit:,}원)", "심각"))
    elif price_ratio >= 2.0:
        signals.append(("danger", "💸", f"정가 대비 {price_ratio:.1f}배 — 높은 프리미엄 (+{profit:,}원)", "위험"))
    elif price_ratio >= 1.5:
        signals.append(("warn",   "⚠️", f"정가 대비 {price_ratio:.1f}배 — 프리미엄 감지 (+{profit:,}원)", "주의"))
    elif price_ratio >= 1.15:
        signals.append(("warn",   "⚠️", f"정가 대비 {price_ratio:.1f}배 — 소폭 프리미엄", "경미"))
    else:
        signals.append(("ok",     "✅", f"가격배율 {price_ratio:.2f}배 — 정상 범위", "정상"))

    if account_age <= 7:
        signals.append(("danger", "🆕", f"계정 {account_age}일 — 신고 직전 개설 의심", "심각"))
    elif account_age <= 30:
        signals.append(("danger", "🆕", f"계정 {account_age}일 — 매우 신규 계정", "위험"))
    elif account_age <= 90:
        signals.append(("warn",   "🆕", f"계정 {account_age}일 — 신규 계정", "주의"))
    elif account_age <= 365:
        signals.append(("warn",   "📅", f"계정 {account_age}일 — 1년 미만", "경미"))
    else:
        signals.append(("ok",     "✅", f"계정 {account_age}일 ({account_age//365}년+) — 신뢰 계정", "정상"))

    if num_listings >= 20:
        signals.append(("danger", "📦", f"동시 매물 {num_listings}개 — 대량 매집 의심", "심각"))
    elif num_listings >= 10:
        signals.append(("danger", "📦", f"동시 매물 {num_listings}개 — 다량 보유", "위험"))
    elif num_listings >= 5:
        signals.append(("warn",   "📦", f"동시 매물 {num_listings}개 — 다수 매물", "주의"))
    else:
        signals.append(("ok",     "✅", f"동시 매물 {num_listings}개 — 정상 범위", "정상"))

    if len(found_kws) >= 4:
        signals.append(("danger", "🔤", f"의심 키워드 {len(found_kws)}개 탐지", "심각"))
    elif len(found_kws) >= 2:
        signals.append(("warn",   "🔤", f"의심 키워드 {len(found_kws)}개: {', '.join(found_kws)}", "주의"))
    elif len(found_kws) == 1:
        signals.append(("warn",   "🔤", f"의심 키워드: '{found_kws[0]}'", "경미"))
    else:
        signals.append(("ok",     "✅", "의심 키워드 없음", "정상"))

    return signals


def risk_meta(score):
    pct = int(score * 100)
    if score < 0.40:
        return pct, "risk-safe",   "정상 거래",   "#10b981"
    if score < THRESHOLD:
        return pct, "risk-warn",   "주의 필요",   "#f59e0b"
    return pct,     "risk-danger", "🚨 암표 의심", "#ef4444"


# ══════════════════════════════════════════════
# 공통 사이드바
# ══════════════════════════════════════════════
def render_sidebar(model):
    model_ok = model is not None
    dot_cls  = "dot-ok" if model_ok else "dot-err"
    dot_txt  = "PyTorch 모델 정상" if model_ok else "모델 파일 없음"

    st.sidebar.markdown(f"""
    <div class="sb-logo">
      <div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:10px;
                    width:36px;height:36px;display:flex;align-items:center;justify-content:center;
                    font-size:1.1rem;color:white;">🎫</div>
        <div>
          <div class="sb-logo-title">ScalpGuard</div>
          <div class="sb-logo-sub">AI ANTI-SCALPING</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="padding:0.4rem 0.5rem 0;">
      <div style="font-size:0.68rem;font-weight:700;color:#94a3b8;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:0.6rem;">탐지 기준</div>
      <div class="legend-row"><div class="legend-dot" style="background:#10b981;"></div>0~39% 정상 거래</div>
      <div class="legend-row"><div class="legend-dot" style="background:#f59e0b;"></div>40~69% 주의 필요</div>
      <div class="legend-row"><div class="legend-dot" style="background:#ef4444;"></div>70%+ 암표 의심</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="model-status">
      <div class="ms-title">📡 AI 모델 상태</div>
      <div class="ms-row"><div class="{dot_cls}"></div>{dot_txt}</div>
      <div style="font-size:0.7rem;color:#94a3b8;margin-top:0.3rem;">ScalpDetector v1.0</div>
    </div>
    <div style="margin-top:0.8rem;padding:0.8rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;">
      <div style="font-size:0.7rem;color:#64748b;line-height:1.7;">
        ⚠️ AI 판정은 통계적 참고입니다.<br>최종 신고 여부는 본인이 판단하세요.
      </div>
    </div>
    """, unsafe_allow_html=True)
