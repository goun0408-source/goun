"""
generate_data.py
────────────────
티켓 암표 탐지용 합성 데이터셋 생성기.

생성 특징(features):
  face_price          : 정가 (원)
  sell_price          : 판매가 (원)
  price_ratio         : 가격배율 = sell_price / face_price
  account_age_days    : 판매자 계정 나이 (일)
  num_listings        : 판매자의 동시 매물 수
  suspicious_keywords : 설명글 의심 키워드 포함 여부 (0/1)

레이블:
  label = 1 (암표) : 40%
  label = 0 (정상) : 60%

출력: ticket_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── 재현성 시드 ────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── 생성 파라미터 ──────────────────────────────────────────────
N_TOTAL   = 5000
N_SCALP   = int(N_TOTAL * 0.40)   # 암표 2,000건
N_NORMAL  = N_TOTAL - N_SCALP      # 정상 3,000건

# ── 정가 범위 (공통) ───────────────────────────────────────────
FACE_LOW, FACE_HIGH = 10_000, 150_000


def make_face_price(n: int) -> np.ndarray:
    """정가: 10,000~150,000원 사이 균등 분포."""
    return rng.integers(FACE_LOW, FACE_HIGH, size=n).astype(float)


# ──────────────────────────────────────────────────────────────
# 1) 정상 거래 생성 (label = 0)
# ──────────────────────────────────────────────────────────────
def generate_normal(n: int) -> pd.DataFrame:
    face   = make_face_price(n)

    # 판매가: 정가의 85~115% (소량 할인~소량 프리미엄)
    ratio  = rng.uniform(0.85, 1.15, size=n)
    # 노이즈: 표준편차 5% 추가
    ratio += rng.normal(0, 0.05, size=n)
    ratio  = np.clip(ratio, 0.5, 2.0)   # 비현실적 값 방지
    sell   = face * ratio

    # 계정 나이: 비교적 오래된 계정 (180~3000일)
    account_age = rng.integers(180, 3001, size=n).astype(float)
    account_age += rng.normal(0, 30, size=n)
    account_age = np.clip(account_age, 1, 5000)

    # 동시 매물 수: 1~4개 (소량)
    num_listings = rng.integers(1, 5, size=n).astype(float)
    num_listings += rng.normal(0, 0.5, size=n)
    num_listings = np.clip(num_listings, 1, 20).round()

    # 의심 키워드: 20% 확률로 포함 (가끔 정상 거래도 쓸 수 있음)
    suspicious = rng.random(size=n) < 0.20

    return pd.DataFrame({
        "face_price":          face.round(),
        "sell_price":          sell.round(),
        "price_ratio":         ratio.round(4),
        "account_age_days":    account_age.round().astype(int),
        "num_listings":        num_listings.astype(int),
        "suspicious_keywords": suspicious.astype(int),
        "label":               0,
    })


# ──────────────────────────────────────────────────────────────
# 2) 암표 거래 생성 (label = 1)
# ──────────────────────────────────────────────────────────────
def generate_scalp(n: int) -> pd.DataFrame:
    face   = make_face_price(n)

    # 판매가: 정가의 150~500% (강한 프리미엄)
    ratio  = rng.uniform(1.5, 5.0, size=n)
    # 노이즈: 표준편차 15% 추가
    ratio += rng.normal(0, 0.15, size=n)
    ratio  = np.clip(ratio, 1.1, 10.0)
    sell   = face * ratio

    # 계정 나이: 신규 계정 (1~90일)
    account_age = rng.integers(1, 91, size=n).astype(float)
    account_age += rng.normal(0, 10, size=n)
    account_age = np.clip(account_age, 1, 365)

    # 동시 매물 수: 다수 (5~30개)
    num_listings = rng.integers(5, 31, size=n).astype(float)
    num_listings += rng.normal(0, 2, size=n)
    num_listings = np.clip(num_listings, 1, 60).round()

    # 의심 키워드: 90% 확률로 포함
    suspicious = rng.random(size=n) < 0.90

    return pd.DataFrame({
        "face_price":          face.round(),
        "sell_price":          sell.round(),
        "price_ratio":         ratio.round(4),
        "account_age_days":    account_age.round().astype(int),
        "num_listings":        num_listings.astype(int),
        "suspicious_keywords": suspicious.astype(int),
        "label":               1,
    })


# ──────────────────────────────────────────────────────────────
# 3) 경계 샘플 (overlap noise) — 데이터를 너무 쉽지 않게
#    정상거래 5%를 암표처럼, 암표 5%를 정상처럼 섞음
# ──────────────────────────────────────────────────────────────
def add_overlap_noise(df: pd.DataFrame) -> pd.DataFrame:
    overlap_frac = 0.05
    n_overlap = int(len(df) * overlap_frac)

    normal_mask  = df["label"] == 0
    scalp_mask   = df["label"] == 1
    normal_idx   = df[normal_mask].sample(n_overlap, random_state=SEED).index
    scalp_idx    = df[scalp_mask].sample(n_overlap, random_state=SEED).index

    # 정상거래 일부 → 가격배율 올리고 계정 나이 낮춤
    df.loc[normal_idx, "price_ratio"]      = rng.uniform(1.3, 2.0, n_overlap).round(4)
    df.loc[normal_idx, "sell_price"]       = (
        df.loc[normal_idx, "face_price"] * df.loc[normal_idx, "price_ratio"]
    ).round()
    df.loc[normal_idx, "account_age_days"] = rng.integers(30, 180, n_overlap)

    # 암표 일부 → 가격배율 낮추고 계정 나이 높임
    df.loc[scalp_idx, "price_ratio"]       = rng.uniform(0.9, 1.3, n_overlap).round(4)
    df.loc[scalp_idx, "sell_price"]        = (
        df.loc[scalp_idx, "face_price"] * df.loc[scalp_idx, "price_ratio"]
    ).round()
    df.loc[scalp_idx, "account_age_days"]  = rng.integers(200, 1000, n_overlap)

    return df


# ──────────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  티켓 암표 합성 데이터 생성기")
    print("=" * 50)

    df_normal = generate_normal(N_NORMAL)
    df_scalp  = generate_scalp(N_SCALP)

    df = pd.concat([df_normal, df_scalp], ignore_index=True)
    df = add_overlap_noise(df)

    # 셔플
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 저장
    out_path = Path("ticket_data.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ── 통계 출력 ─────────────────────────────────────────────
    print(f"\n[완료] 저장 완료: {out_path.resolve()}")
    print(f"\n[데이터 요약] 총 {len(df):,}건")
    print(f"   정상(0): {(df['label']==0).sum():,}건 ({(df['label']==0).mean()*100:.1f}%)")
    print(f"   암표(1): {(df['label']==1).sum():,}건 ({(df['label']==1).mean()*100:.1f}%)")

    print("\n[특징별 기초 통계] label 기준:")
    for col in ["price_ratio", "account_age_days", "num_listings", "suspicious_keywords"]:
        g = df.groupby("label")[col].mean()
        print(f"  {col:25s} | 정상: {g[0]:.3f}  |  암표: {g[1]:.3f}")

    print("\n[샘플 데이터] 상위 5행:")
    print(df.head().to_string(index=False))
    print("\n생성 완료!")
