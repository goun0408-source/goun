"""
Step 1: 데이터 탐색(EDA) & 전처리
- 클래스별 이미지 수 파악 및 분포 시각화
- 이미지 해상도·픽셀 통계 분석
- 클래스별 샘플 이미지 격자 표시
- Train / Val / Test Stratified Split
- 정규화 파라미터(평균·표준편차) 계산
"""

import os
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES    = ["fire", "smoke", "normal"]
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 색상·스타일 ────────────────────────────────────────────────────────────────
CLASS_COLORS = {"fire": "#FF4D4D", "smoke": "#A0A0C0", "normal": "#4DFF91"}
plt.rcParams.update({
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#1A1A2E",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "axes.edgecolor":   "#333355",
    "grid.color":       "#333355",
    "font.family":      "DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. 이미지 경로 수집
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Step 1 — 데이터 탐색 & 전처리")
print("=" * 60)

all_paths, all_labels = [], []
class_counts = {}

for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        print(f"  [경고] {cls_dir} 폴더가 없습니다 — 스킵")
        class_counts[cls] = 0
        continue

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
    all_paths.extend(paths)
    all_labels.extend([cls] * len(paths))
    class_counts[cls] = len(paths)
    print(f"  {cls:8s}: {len(paths):5d}장")

total = sum(class_counts.values())
print(f"\n  합계   : {total}장")

if total == 0:
    print("\n[오류] data 폴더에 이미지가 없습니다.")
    print("  data/fire/, data/smoke/, data/normal/ 에 이미지를 넣어주세요.")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 클래스 분포 시각화
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("클래스 분포 분석", fontsize=16, color="white", y=1.02)

# 막대 그래프
ax1 = axes[0]
bars = ax1.bar(
    class_counts.keys(),
    class_counts.values(),
    color=[CLASS_COLORS[c] for c in class_counts],
    width=0.5,
    edgecolor="white",
    linewidth=0.5,
)
for bar, (cls, cnt) in zip(bars, class_counts.items()):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.01,
        f"{cnt}\n({cnt/total*100:.1f}%)" if total else "0",
        ha="center", va="bottom", color="white", fontsize=10,
    )
ax1.set_title("클래스별 이미지 수", color="white")
ax1.set_ylabel("이미지 수")
ax1.set_ylim(0, max(class_counts.values()) * 1.2 if total else 1)
ax1.grid(axis="y", alpha=0.3)

# 파이 차트
ax2 = axes[1]
wedge_props = {"edgecolor": "#0F0F1A", "linewidth": 2}
wedges, texts, autotexts = ax2.pie(
    [max(v, 0.001) for v in class_counts.values()],
    labels=class_counts.keys(),
    colors=[CLASS_COLORS[c] for c in class_counts],
    autopct="%1.1f%%",
    wedgeprops=wedge_props,
    startangle=140,
)
for t in texts + autotexts:
    t.set_color("white")
ax2.set_title("클래스 비율", color="white")

plt.tight_layout()
out_path = OUTPUT_DIR / "class_distribution.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print(f"\n[저장] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 이미지 해상도 분포 & 픽셀 통계
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] 이미지 해상도 분석 중...")

widths, heights = [], []
pixel_means = {cls: [] for cls in CLASSES}
SAMPLE_LIMIT = 500  # 통계용 샘플 제한 (속도 절충)

sampled = list(zip(all_paths, all_labels))
random.shuffle(sampled)
sampled = sampled[:SAMPLE_LIMIT]

for path, cls in sampled:
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        widths.append(w)
        heights.append(h)

        arr = np.array(img, dtype=np.float32) / 255.0
        pixel_means[cls].append(arr.mean(axis=(0, 1)))  # (3,)
    except Exception:
        pass

# 해상도 산점도
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("이미지 해상도 분포", fontsize=16, color="white")

ax = axes[0]
ax.scatter(widths, heights, alpha=0.4, s=15, color="#7B9FFF", edgecolors="none")
ax.set_xlabel("너비 (px)")
ax.set_ylabel("높이 (px)")
ax.set_title("Width vs Height")
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.hist(widths,  bins=30, alpha=0.7, color="#FF7B7B", label="Width",  edgecolor="none")
ax2.hist(heights, bins=30, alpha=0.7, color="#7BFFB0", label="Height", edgecolor="none")
ax2.set_xlabel("픽셀 크기")
ax2.set_ylabel("빈도")
ax2.set_title("해상도 히스토그램")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path2 = OUTPUT_DIR / "resolution_distribution.png"
plt.savefig(out_path2, dpi=120, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print(f"[저장] {out_path2}")

if widths:
    print(f"  너비  — 평균: {np.mean(widths):.0f}px, 중앙값: {np.median(widths):.0f}px, 범위: {min(widths)}~{max(widths)}px")
    print(f"  높이  — 평균: {np.mean(heights):.0f}px, 중앙값: {np.median(heights):.0f}px, 범위: {min(heights)}~{max(heights)}px")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 클래스별 샘플 이미지 격자
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] 샘플 이미지 격자 시각화...")

N_COLS = 5
cls_sample_paths = {}
for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        cls_sample_paths[cls] = []
        continue
    paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    cls_sample_paths[cls] = random.sample(paths, min(N_COLS, len(paths)))

valid_cls = [c for c in CLASSES if cls_sample_paths.get(c)]
N_ROWS = len(valid_cls)

if N_ROWS > 0:
    fig = plt.figure(figsize=(N_COLS * 2.8, N_ROWS * 2.8))
    fig.patch.set_facecolor("#0F0F1A")
    fig.suptitle("클래스별 샘플 이미지", fontsize=16, color="white", y=1.01)

    gs = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig, hspace=0.05, wspace=0.05)

    for row_idx, cls in enumerate(valid_cls):
        for col_idx, path in enumerate(cls_sample_paths[cls]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#333")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(cls, color=CLASS_COLORS.get(cls, "white"), fontsize=12,
                              rotation=90, labelpad=5, va="center")
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)

    out_path3 = OUTPUT_DIR / "sample_grid.png"
    plt.savefig(out_path3, dpi=100, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
    print(f"[저장] {out_path3}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 정규화 파라미터 계산
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] 정규화 파라미터 계산 중...")

all_means_list = [v for vals in pixel_means.values() for v in vals]
if all_means_list:
    dataset_mean = np.mean(all_means_list, axis=0)
    dataset_std  = np.std(all_means_list, axis=0)
    print(f"  mean: R={dataset_mean[0]:.4f}, G={dataset_mean[1]:.4f}, B={dataset_mean[2]:.4f}")
    print(f"  std : R={dataset_std[0]:.4f},  G={dataset_std[1]:.4f},  B={dataset_std[2]:.4f}")
else:
    dataset_mean = np.array([0.485, 0.456, 0.406])
    dataset_std  = np.array([0.229, 0.224, 0.225])
    print("  → 샘플 부족: ImageNet 기본값 사용")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train / Val / Test Stratified Split & 저장
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Train/Val/Test 분할...")

str_paths = [str(p) for p in all_paths]

if total >= 6:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        str_paths, all_labels, test_size=0.1, stratify=all_labels, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.111, stratify=y_tmp, random_state=SEED
    )  # 0.111 * 0.9 ≈ 0.1 → 전체의 10%
else:
    X_train, y_train = str_paths, all_labels
    X_val,   y_val   = str_paths, all_labels
    X_test,  y_test  = str_paths, all_labels

split_info = {
    "train": {"paths": X_train, "labels": y_train},
    "val":   {"paths": X_val,   "labels": y_val},
    "test":  {"paths": X_test,  "labels": y_test},
    "class_to_idx": {cls: i for i, cls in enumerate(CLASSES)},
    "mean": dataset_mean.tolist(),
    "std":  dataset_std.tolist(),
}

split_path = OUTPUT_DIR / "split_info.json"
with open(split_path, "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print(f"  Train : {len(X_train)}장  ({Counter(y_train)})")
print(f"  Val   : {len(X_val)}장  ({Counter(y_val)})")
print(f"  Test  : {len(X_test)}장  ({Counter(y_test)})")
print(f"[저장] {split_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 클래스 가중치 계산 (불균형 대응)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] 클래스 가중치 계산...")
label_to_idx = split_info["class_to_idx"]
train_counts = Counter(y_train)
n_train = sum(train_counts.values())
class_weights = []
for cls in CLASSES:
    cnt = train_counts.get(cls, 1)
    w = n_train / (len(CLASSES) * cnt)
    class_weights.append(w)
    print(f"  {cls:8s}: {cnt}장 → weight={w:.4f}")

split_info["class_weights"] = class_weights
with open(split_path, "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 완료 요약
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  Step 1 완료!")
print("=" * 60)
print(f"  출력 파일:")
print(f"    - outputs/class_distribution.png")
print(f"    - outputs/resolution_distribution.png")
print(f"    - outputs/sample_grid.png")
print(f"    - outputs/split_info.json")
print("=" * 60)
