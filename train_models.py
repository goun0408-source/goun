"""
train_models.py
───────────────
티켓 암표 탐지 — 3가지 모델 학습 & 비교 평가

모델:
  1. Logistic Regression  (sklearn 베이스라인)
  2. Random Forest        (sklearn 베이스라인)
  3. PyTorch 신경망       (입력5 → 32 → 16 → 1)

출력:
  model.pth              — PyTorch 모델 가중치
  scaler.pkl             — StandardScaler (추론 시 재사용)
  results/confusion_matrices.png  — 3개 혼동행렬 비교
  results/metrics_comparison.png  — 지표 막대 비교
  results/training_curve.png      — PyTorch 학습곡선
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # GUI 없이 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# 한글 폰트 설정 (Windows: 맑은 고딕, 기타: NanumGothic 시도)
import matplotlib.font_manager as fm
_korean_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "sans-serif"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _korean_candidates if f in _available), "sans-serif")
plt.rcParams["font.family"] = _font
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── 재현성 ─────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 결과 폴더 ──────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── 하이퍼파라미터 ─────────────────────────────────────────────
FEATURES = ["price_ratio", "account_age_days", "num_listings", "suspicious_keywords"]
# face_price, sell_price는 price_ratio에 이미 포함되어 있어 제외
# (포함하면 다중공선성 발생)
TEST_SIZE   = 0.2
LR          = 0.001
EPOCHS      = 60
BATCH_SIZE  = 64
HIDDEN1     = 32
HIDDEN2     = 16


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로딩 & 분할
# ══════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv("ticket_data.csv")
    X  = df[FEATURES].values.astype(np.float32)
    y  = df["label"].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # 스케일러 저장
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[OK] scaler.pkl 저장 완료")

    return X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, scaler


# ══════════════════════════════════════════════════════════════
# 2. sklearn 베이스라인 모델
# ══════════════════════════════════════════════════════════════
def train_lr(X_tr, X_te, y_tr, y_te):
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return model, y_pred, compute_metrics(y_te, y_pred)


def train_rf(X_tr, X_te, y_tr, y_te):
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return model, y_pred, compute_metrics(y_te, y_pred)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ══════════════════════════════════════════════════════════════
# 3. PyTorch 신경망
# ══════════════════════════════════════════════════════════════
class ScalpDetector(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_pytorch(X_tr_sc, X_te_sc, y_tr, y_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PyTorch] 학습 장치: {device}")

    # 텐서 변환
    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr,    dtype=torch.float32)
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32)
    y_te_t = torch.tensor(y_te,    dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = ScalpDetector(len(FEATURES), HIDDEN1, HIDDEN2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 ─────────────────────────────────────────────
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss    += loss.item() * len(xb)
            ep_correct += ((pred >= 0.5) == yb.bool()).sum().item()
            ep_total   += len(xb)

        scheduler.step()
        train_loss = ep_loss / ep_total
        train_acc  = ep_correct / ep_total

        # ── 검증 ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred  = model(X_te_t.to(device))
            val_loss  = criterion(val_pred, y_te_t.to(device)).item()
            val_acc   = ((val_pred >= 0.5) == y_te_t.to(device).bool()).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 최종 예측
    model.eval()
    with torch.no_grad():
        y_prob  = model(X_te_t.to(device)).cpu().numpy()
        y_pred  = (y_prob >= 0.5).astype(int)

    # 모델 저장
    torch.save(model.state_dict(), "model.pth")
    print("\n[OK] model.pth 저장 완료")

    return model, y_pred, compute_metrics(y_te, y_pred), history


# ══════════════════════════════════════════════════════════════
# 4. 시각화
# ══════════════════════════════════════════════════════════════
PALETTE = {
    "LR":       "#4F8EF7",
    "RF":       "#F7994F",
    "PyTorch":  "#4FC178",
}
CLASS_NAMES = ["정상(0)", "암표(1)"]


def plot_confusion_matrices(y_te, preds_dict: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("혼동 행렬 (Confusion Matrix) 비교", fontsize=14, fontweight="bold", y=1.02)

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, linecolor="gray",
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(name, fontsize=12, fontweight="bold", color=PALETTE[name])
        ax.set_xlabel("예측값", fontsize=10)
        ax.set_ylabel("실제값", fontsize=10)

    plt.tight_layout()
    path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PNG] 저장: {path}")
    plt.close()


def plot_metrics_comparison(metrics_dict: dict):
    metric_names  = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names   = list(metrics_dict.keys())
    colors        = [PALETTE[m] for m in model_names]

    x = np.arange(len(metric_names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [metrics_dict[name][m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                      edgecolor="white", linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("모델 성능 비교 (Logistic Regression / Random Forest / PyTorch)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PNG] 저장: {path}")
    plt.close()


def plot_training_curve(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PyTorch 신경망 학습 곡선 (Training Curve)", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], color="#4FC178", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   color="#E74C3C", label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("손실 (Loss)"); ax1.legend()
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], color="#4FC178", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   color="#E74C3C", label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("정확도 (Accuracy)"); ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PNG] 저장: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 5. 메인 실행
# ══════════════════════════════════════════════════════════════
def print_header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # ── 데이터 로드 ───────────────────────────────────────────
    print_header("데이터 로딩")
    X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, scaler = load_data()
    print(f"  학습 데이터: {len(X_tr):,}건  |  테스트 데이터: {len(X_te):,}건")
    print(f"  특징: {FEATURES}")

    # ── Logistic Regression ───────────────────────────────────
    print_header("1. Logistic Regression")
    lr_model, lr_pred, lr_metrics = train_lr(X_tr_sc, X_te_sc, y_tr, y_te)
    print(classification_report(y_te, lr_pred, target_names=CLASS_NAMES, digits=4))

    # ── Random Forest ─────────────────────────────────────────
    print_header("2. Random Forest")
    rf_model, rf_pred, rf_metrics = train_rf(X_tr_sc, X_te_sc, y_tr, y_te)
    print(classification_report(y_te, rf_pred, target_names=CLASS_NAMES, digits=4))

    # ── PyTorch 신경망 ────────────────────────────────────────
    print_header("3. PyTorch 신경망 (ScalpDetector)")
    pt_model, pt_pred, pt_metrics, history = train_pytorch(X_tr_sc, X_te_sc, y_tr, y_te)
    print(classification_report(y_te, pt_pred, target_names=CLASS_NAMES, digits=4))

    # ── 지표 요약 테이블 ──────────────────────────────────────
    print_header("[Result] 모델 성능 비교 요약")
    all_metrics = {"LR": lr_metrics, "RF": rf_metrics, "PyTorch": pt_metrics}
    header = f"{'모델':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<12} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

    # ── 시각화 ────────────────────────────────────────────────
    print_header("[Graph] 그래프 생성 및 저장")
    preds_dict = {"LR": lr_pred, "RF": rf_pred, "PyTorch": pt_pred}
    plot_confusion_matrices(y_te, preds_dict)
    plot_metrics_comparison(all_metrics)
    plot_training_curve(history)

    print("\n[Done] 1단계 완료! 생성된 파일:")
    print("   - model.pth")
    print("   - scaler.pkl")
    print("   - results/confusion_matrices.png")
    print("   - results/metrics_comparison.png")
    print("   - results/training_curve.png")
