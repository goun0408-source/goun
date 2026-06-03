"""
feature_importance.py
─────────────────────
학습된 모델을 이용해 특징 중요도를 3가지 방법으로 분석하고
results/ 폴더에 PNG로 저장합니다.

1. Random Forest MDI (불순도 감소 기반)
2. Permutation Importance (테스트 세트 기반, 모델 무관)
3. PyTorch Gradient Saliency (입력 기울기 평균 절댓값)

출력:
  results/feature_importance_rf_mdi.png
  results/feature_importance_permutation.png
  results/feature_importance_pytorch.png
  results/feature_importance_combined.png   ← 3개 합친 종합 요약
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 한글 폰트 ────────────────────────────────────────────────
_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "sans-serif"]
_available  = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _candidates if f in _available), "sans-serif")
plt.rcParams["font.family"]       = _font
plt.rcParams["axes.unicode_minus"] = False

SEED     = 42
RESULTS  = Path("results")
RESULTS.mkdir(exist_ok=True)

FEATURES      = ["price_ratio", "account_age_days", "num_listings", "suspicious_keywords"]
FEATURE_KR    = ["가격배율", "계정나이(일)", "동시매물수", "의심키워드"]
PALETTE_BASE  = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726"]   # 빨·파·초·주황


# ══════════════════════════════════════════════════════════════
# 데이터 & 모델 로드
# ══════════════════════════════════════════════════════════════
class ScalpDetector(nn.Module):
    def __init__(self, input_dim=4, h1=32, h2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, 1),         nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(1)


def load_all():
    df = pd.read_csv("ticket_data.csv")
    X  = df[FEATURES].values.astype(np.float32)
    y  = df["label"].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # RF (재학습)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)

    # PyTorch
    pt = ScalpDetector()
    pt.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    pt.eval()

    return X_tr_sc, X_te_sc, y_tr, y_te, rf, pt, scaler


# ══════════════════════════════════════════════════════════════
# 1. RF MDI
# ══════════════════════════════════════════════════════════════
def compute_rf_mdi(rf):
    imp = rf.feature_importances_
    std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    order = np.argsort(imp)[::-1]
    return imp[order], std[order], [FEATURE_KR[i] for i in order], [PALETTE_BASE[i] for i in order]


def plot_rf_mdi(rf):
    imp, std, labels, colors = compute_rf_mdi(rf)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], imp[::-1], xerr=std[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8,
                   capsize=4, alpha=0.9)
    for bar, val in zip(bars, imp[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("MDI 중요도 (불순도 감소 기여)", fontsize=10)
    ax.set_title("Random Forest — Feature Importance (MDI)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, imp.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = RESULTS / "feature_importance_rf_mdi.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"[PNG] {p}"); plt.close()
    return imp, labels


# ══════════════════════════════════════════════════════════════
# 2. Permutation Importance
# ══════════════════════════════════════════════════════════════
def compute_permutation(rf, X_te, y_te):
    result = permutation_importance(
        rf, X_te, y_te, n_repeats=30, random_state=SEED, scoring="f1"
    )
    order = np.argsort(result.importances_mean)[::-1]
    return (result.importances_mean[order],
            result.importances_std[order],
            [FEATURE_KR[i] for i in order],
            [PALETTE_BASE[i] for i in order])


def plot_permutation(rf, X_te, y_te):
    imp, std, labels, colors = compute_permutation(rf, X_te, y_te)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], imp[::-1], xerr=std[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8,
                   capsize=4, alpha=0.9)
    for bar, val in zip(bars, imp[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("F1 감소량 (특징 무작위 셔플 시)", fontsize=10)
    ax.set_title("Permutation Importance (테스트 세트 기반)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(imp) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = RESULTS / "feature_importance_permutation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"[PNG] {p}"); plt.close()
    return imp, labels


# ══════════════════════════════════════════════════════════════
# 3. PyTorch Gradient Saliency
# ══════════════════════════════════════════════════════════════
def compute_gradient_saliency(pt, X_te_sc):
    pt.train()   # Dropout / BN을 활성화해야 grad 흐름
    X_t = torch.tensor(X_te_sc, dtype=torch.float32, requires_grad=True)
    out = pt(X_t)
    out.sum().backward()
    saliency = X_t.grad.abs().mean(dim=0).detach().numpy()
    saliency = saliency / saliency.sum()   # 합계=1 정규화
    pt.eval()
    order = np.argsort(saliency)[::-1]
    return (saliency[order],
            [FEATURE_KR[i] for i in order],
            [PALETTE_BASE[i] for i in order])


def plot_pytorch_saliency(pt, X_te_sc):
    imp, labels, colors = compute_gradient_saliency(pt, X_te_sc)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], imp[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8, alpha=0.9)
    for bar, val in zip(bars, imp[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("정규화 Gradient Saliency (기여도 비율)", fontsize=10)
    ax.set_title("PyTorch 신경망 — Gradient Saliency", fontsize=12, fontweight="bold")
    ax.set_xlim(0, imp.max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = RESULTS / "feature_importance_pytorch.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"[PNG] {p}"); plt.close()
    return imp, labels


# ══════════════════════════════════════════════════════════════
# 4. 종합 요약 (3-in-1)
# ══════════════════════════════════════════════════════════════
def plot_combined(rf, X_te_sc, y_te, pt):
    mdi_imp, mdi_std, mdi_labels, mdi_colors = compute_rf_mdi(rf)
    per_imp, per_std, per_labels, per_colors = compute_permutation(rf, X_te_sc, y_te)
    sal_imp, sal_labels, sal_colors          = compute_gradient_saliency(pt, X_te_sc)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.suptitle("Feature Importance — 3가지 방법 비교", fontsize=14, fontweight="bold", y=1.02)

    datasets = [
        (axes[0], mdi_imp,  mdi_std,  mdi_labels,  mdi_colors,
         "RF MDI 중요도",        "Random Forest MDI"),
        (axes[1], per_imp,  per_std,  per_labels,  per_colors,
         "F1 감소량",            "Permutation Importance"),
        (axes[2], sal_imp,  None,     sal_labels,  sal_colors,
         "Gradient Saliency",   "PyTorch Saliency"),
    ]

    for ax, imp, std, labels, colors, xlabel, title in datasets:
        xerr = std if std is not None else None
        bars = ax.barh(labels[::-1], imp[::-1],
                       xerr=(xerr[::-1] if xerr is not None else None),
                       color=colors[::-1], edgecolor="white",
                       linewidth=0.8, capsize=3, alpha=0.9)
        for bar, val in zip(bars, imp[::-1]):
            ax.text(bar.get_width() + imp.max()*0.03,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, imp.max() * 1.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = RESULTS / "feature_importance_combined.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"[PNG] {p}"); plt.close()


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Feature Importance 분석")
    print("=" * 55)

    X_tr_sc, X_te_sc, y_tr, y_te, rf, pt, scaler = load_all()

    print("\n[1/4] RF MDI ...")
    mdi_imp, mdi_labels = plot_rf_mdi(rf)

    print("[2/4] Permutation Importance ...")
    per_imp, per_labels = plot_permutation(rf, X_te_sc, y_te)

    print("[3/4] PyTorch Gradient Saliency ...")
    sal_imp, sal_labels = plot_pytorch_saliency(pt, X_te_sc)

    print("[4/4] 종합 비교 그래프 ...")
    plot_combined(rf, X_te_sc, y_te, pt)

    print("\n[결과 요약]")
    print(f"{'특징':<12} {'RF MDI':>10} {'Permutation':>13} {'PyTorch Saliency':>18}")
    print("-" * 56)
    for feat, kr in zip(FEATURES, FEATURE_KR):
        i = FEATURES.index(feat)
        print(f"{kr:<12} {mdi_imp[mdi_labels.index(kr)]:>10.4f} "
              f"{per_imp[per_labels.index(kr)]:>13.4f} "
              f"{sal_imp[sal_labels.index(kr)]:>18.4f}")

    print("\n[Done] results/ 폴더에 4개 PNG 저장 완료")
