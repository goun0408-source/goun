"""
train.py — MobileNetV2 전이학습 분류 스크립트
==============================================
Phase 1 : Backbone 고정 → 분류층만 학습 (Feature Extraction)
Phase 2 : 합성곱 뒷부분 해제 → 미세 조정 (Fine-tuning)

실행:
    python train.py
    python train.py --data ./data --epochs_head 5 --epochs_fine 20 --batch 32
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# 같은 폴더의 dataset.py 모듈 사용
from dataset import build_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
# CLI 인수
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="MobileNetV2 전이학습")
    p.add_argument("--data",         default="./data",  help="데이터 루트 폴더")
    p.add_argument("--output",       default="./outputs")
    p.add_argument("--model_path",   default="./model.pth")
    p.add_argument("--epochs_head",  type=int, default=5,  help="Phase1 에폭 수")
    p.add_argument("--epochs_fine",  type=int, default=20, help="Phase2 에폭 수")
    p.add_argument("--batch",        type=int, default=32)
    p.add_argument("--patience",     type=int, default=7,  help="조기 종료 patience")
    p.add_argument("--lr_head",      type=float, default=1e-3)
    p.add_argument("--lr_fine",      type=float, default=1e-4)
    p.add_argument("--val_ratio",    type=float, default=0.2)
    p.add_argument("--imbalance",    default="sampler",
                   choices=["sampler", "weight", "none"])
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--workers",      type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 공통 스타일
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0F0F1A"
PANEL_BG  = "#1A1A2E"
C_TRAIN   = "#FF7B7B"   # 학습 곡선 색
C_VAL     = "#7BFFB0"   # 검증 곡선 색
C_LR      = "#7BB8FF"
CLASS_PAL = {"fire": "#FF4D4D", "smoke": "#A0A0C0", "normal": "#4DFF91"}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor":  PANEL_BG,
        "text.color": "white",       "axes.labelcolor": "white",
        "xtick.color": "white",      "ytick.color":     "white",
        "axes.edgecolor": "#333355", "grid.color":      "#333355",
        "legend.facecolor": PANEL_BG,"legend.edgecolor":"#333355",
    })

# ─────────────────────────────────────────────────────────────────────────────
# 모델 빌더
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenet(num_classes: int) -> nn.Module:
    """
    MobileNetV2 사전학습 가중치 로드 후
    마지막 분류층을 num_classes 크기로 교체.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v2(weights=weights)

    # 원래 분류기: Linear(1280 → 1000)
    # 교체: Linear(1280 → 256) → ReLU → Dropout → Linear(256 → num_classes)
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_all_except_classifier(model: nn.Module):
    """Backbone(features) 전체 고정, 분류층만 학습."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int = 5):
    """
    MobileNetV2 features 마지막 n개 InvertedResidual 블록 해제.
    features는 총 19개 레이어(0~18).
    """
    total = len(model.features)   # 19
    for i, layer in enumerate(model.features):
        for param in layer.parameters():
            param.requires_grad = (i >= total - n)
    for param in model.classifier.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 검증 1 에폭
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, optimizer=None,
    scaler=None, device="cpu", desc="Train"
) -> tuple[float, float]:
    """1 에폭 실행. 학습/검증 모두 지원."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=f"  {desc}", ncols=85, leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{correct/total*100:.1f}%")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 학습 곡선 그래프
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, best_epoch: int, save_path: Path):
    set_dark_style()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("학습 곡선 (Training Curves)", fontsize=14, color="white", y=1.01)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=C_TRAIN, lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=C_VAL,   lw=2, label="Val Loss")
    ax.axvline(best_epoch, color="gold", ls="--", lw=1.2, alpha=0.8,
               label=f"Best ep {best_epoch}")
    ax.set_title("Loss"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [v*100 for v in history["train_acc"]], color=C_TRAIN, lw=2, label="Train Acc")
    ax.plot(epochs, [v*100 for v in history["val_acc"]],   color=C_VAL,   lw=2, label="Val Acc")
    ax.axvline(best_epoch, color="gold", ls="--", lw=1.2, alpha=0.8)
    ax.set_title("Accuracy (%)"); ax.set_xlabel("Epoch")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(epochs, history["lr"], color=C_LR, lw=2)
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(alpha=0.3)

    # Phase 구분선
    n_p1 = history.get("phase1_epochs", 0)
    if 0 < n_p1 < len(history["train_loss"]):
        for ax in axes:
            ax.axvline(n_p1 + 0.5, color="#FF9F43", ls=":", lw=1.5, alpha=0.6)
        axes[0].text(n_p1 + 0.7, axes[0].get_ylim()[1] * 0.95,
                     "Fine-tune↓", color="#FF9F43", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[저장] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 평가 & 시각화
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_visualize(
    model, loader, class_names: list[str], device, save_dir: Path
):
    """
    검증셋 전체 평가:
    - 전체 정확도
    - 클래스별 Precision / Recall / F1
    - 혼동 행렬 히트맵
    - fire·smoke → normal 미탐(Miss Detection) 비율 강조
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  평가 중", ncols=85, leave=False):
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    normal_idx = class_names.index("normal") if "normal" in class_names else -1

    # ── 전체 정확도 ───────────────────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n  전체 정확도 : {overall_acc:.2f}%")

    # ── 분류 리포트 ───────────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\n  클래스별 분류 리포트:")
    for line in report.split("\n"):
        print("   ", line)

    # ── 미탐(Miss Detection) 비율 ─────────────────────────────────────────────
    print("\n  ⚠️  미탐(Missed Detection) 분석 — 위험 클래스를 normal로 잘못 분류")
    if normal_idx != -1:
        for cls in ["fire", "smoke"]:
            if cls not in class_names:
                continue
            cls_idx     = class_names.index(cls)
            true_cls    = all_labels == cls_idx
            pred_normal = all_preds  == normal_idx
            miss_mask   = true_cls & pred_normal
            n_true      = true_cls.sum()
            n_miss      = miss_mask.sum()
            rate        = n_miss / n_true * 100 if n_true else 0.0
            print(f"    [{cls:5s}] 실제 {n_true}장 중 {n_miss}장을 normal로 오분류 "
                  f"→ 미탐률 {rate:.1f}%")
    else:
        print("    'normal' 클래스가 없어 미탐 분석을 건너뜁니다.")

    # ── 혼동 행렬 ────────────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # 행 정규화

    set_dark_style()
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # 좌: 절대 수치
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax1, linewidths=0.5, linecolor="#333355",
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax1.set_title("혼동 행렬 (Counts)", color="white", pad=10)
    ax1.set_xlabel("예측 레이블", color="white")
    ax1.set_ylabel("실제 레이블", color="white")
    ax1.tick_params(colors="white")

    # 우: 비율(%) — 미탐 셀 강조
    ax2 = fig.add_subplot(gs[1])
    annot_arr = np.array([[f"{v*100:.1f}%" for v in row] for row in cm_norm])

    # 미탐 셀 표시: fire/smoke 행의 normal 열
    if normal_idx != -1:
        for cls in ["fire", "smoke"]:
            if cls in class_names:
                ri = class_names.index(cls)
                annot_arr[ri][normal_idx] = (
                    annot_arr[ri][normal_idx] + "\n⚠️"
                )

    sns.heatmap(
        cm_norm, annot=annot_arr, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax2, linewidths=0.5, linecolor="#333355",
        vmin=0, vmax=1,
        annot_kws={"size": 11},
    )
    ax2.set_title("혼동 행렬 (비율) — ⚠️=미탐", color="white", pad=10)
    ax2.set_xlabel("예측 레이블", color="white")
    ax2.set_ylabel("실제 레이블", color="white")
    ax2.tick_params(colors="white")

    plt.suptitle(f"검증셋 평가  |  전체 정확도 {overall_acc:.2f}%",
                 fontsize=14, color="white", y=1.02)

    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n[저장] {cm_path}")

    # ── 클래스별 Precision / Recall 막대 그래프 ──────────────────────────────
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names))
    )
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.bar(x - width, prec * 100, width, label="Precision", color="#7BB8FF", alpha=0.9)
    ax.bar(x,         rec  * 100, width, label="Recall",    color="#FF9F7B", alpha=0.9)
    ax.bar(x + width, f1   * 100, width, label="F1-Score",  color="#B07BFF", alpha=0.9)

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - width, p * 100 + 1, f"{p*100:.1f}", ha="center", fontsize=9, color="white")
        ax.text(i,         r * 100 + 1, f"{r*100:.1f}", ha="center", fontsize=9, color="white")
        ax.text(i + width, f * 100 + 1, f"{f*100:.1f}", ha="center", fontsize=9, color="white")

    # 미탐 위험 클래스 배경 강조
    for cls in ["fire", "smoke"]:
        if cls in class_names:
            ci = class_names.index(cls)
            ax.axvspan(ci - 0.5, ci + 0.5, color="#FF4D4D", alpha=0.07)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("점수 (%)")
    ax.set_title("클래스별 Precision / Recall / F1-Score", color="white", pad=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    pr_path = save_dir / "class_metrics.png"
    plt.savefig(pr_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[저장] {pr_path}")

    return overall_acc, cm


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 재현성 ───────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 장치 ─────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    USE_AMP = device.type == "cuda"
    print(f"\n[장치] {device}  |  AMP={USE_AMP}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    imbalance = None if args.imbalance == "none" else args.imbalance
    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        data_dir    = args.data,
        batch_size  = args.batch,
        val_ratio   = args.val_ratio,
        imbalance   = imbalance,
        seed        = args.seed,
        num_workers = args.workers,
    )
    num_classes = len(class_names)

    # ── 모델 ─────────────────────────────────────────────────────────────────
    model = build_mobilenet(num_classes).to(device)
    print(f"\n[모델] MobileNetV2  |  클래스={class_names}")

    # 손실 함수: class_weight 사용 여부
    if imbalance == "weight" and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("  CrossEntropyLoss with class_weights")
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
        "phase1_epochs": args.epochs_head,
    }
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_cnt  = 0

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 1 — Feature Extraction (Backbone 고정)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 1 — Feature Extraction")
    print(f"  (Backbone 고정, 분류층만 학습, {args.epochs_head} 에폭)")
    print(f"{'='*60}")

    freeze_all_except_classifier(model)
    print(f"  학습 파라미터 수: {count_trainable(model):,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_head, eta_min=1e-5)

    for epoch in range(1, args.epochs_head + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, scaler, device, "Train")
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    None, None, device, "Val  ")
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        flag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), args.model_path)
            patience_cnt  = 0
            flag = " ← best ✅"
        else:
            patience_cnt += 1

        elapsed = time.time() - t0
        print(f"  [P1] Ep {epoch:02d}/{args.epochs_head}  "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc*100:.1f}%/{va_acc*100:.1f}%  "
              f"lr {lr_now:.2e}  {elapsed:.1f}s{flag}")

        if patience_cnt >= args.patience:
            print(f"  조기 종료 (patience={args.patience})")
            break

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 2 — Fine-tuning (마지막 5 블록 해제)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 2 — Fine-tuning")
    print(f"  (마지막 5 InvertedResidual 블록 + 분류층, {args.epochs_fine} 에폭)")
    print(f"{'='*60}")

    unfreeze_last_n_blocks(model, n=5)
    print(f"  학습 파라미터 수: {count_trainable(model):,}")

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(),   "lr": args.lr_head},
        {"params": [p for n, p in model.features.named_parameters()
                    if p.requires_grad],             "lr": args.lr_fine},
    ], weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs_fine, eta_min=1e-6)
    patience_cnt = 0

    for epoch in range(1, args.epochs_fine + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, scaler, device, "Train")
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    None, None, device, "Val  ")
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        ep_total = args.epochs_head + epoch
        flag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = ep_total
            torch.save(model.state_dict(), args.model_path)
            patience_cnt  = 0
            flag = " ← best ✅"
        else:
            patience_cnt += 1

        elapsed = time.time() - t0
        print(f"  [P2] Ep {ep_total:02d}/{args.epochs_head+args.epochs_fine}  "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc*100:.1f}%/{va_acc*100:.1f}%  "
              f"lr {lr_now:.2e}  {elapsed:.1f}s{flag}")

        if patience_cnt >= args.patience:
            print(f"  조기 종료 (에폭 {ep_total}, best={best_epoch})")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # 학습 곡선 저장
    # ─────────────────────────────────────────────────────────────────────────
    plot_training_curves(history, best_epoch,
                         save_path=OUTPUT_DIR / "training_curves.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 최종 평가 (best model 로드)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  최종 평가 — best model (에폭 {best_epoch}) 로드")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    evaluate_and_visualize(model, val_loader, class_names, device, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # 완료 요약
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ 학습 완료!")
    print(f"{'='*60}")
    print(f"  모델 저장  : {args.model_path}")
    print(f"  Best 에폭  : {best_epoch}  (val_loss={best_val_loss:.4f})")
    print(f"  출력 폴더  : {OUTPUT_DIR}/")
    print(f"    - training_curves.png")
    print(f"    - confusion_matrix.png")
    print(f"    - class_metrics.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
