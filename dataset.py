"""
dataset.py — 재난 감지 데이터 전처리 & DataLoader 모듈
=========================================================
- ImageFolder 기반 로드
- 224×224 Resize + ImageNet 정규화
- 학습 데이터 증강 (좌우반전, 회전, 색상 지터 등)
- Train 80% / Val 20% Stratified Split
- 클래스 불균형 보정: WeightedRandomSampler 또는 class_weight 반환

Usage:
    from dataset import build_dataloaders

    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        data_dir   = "./data",
        batch_size = 32,
        imbalance  = "sampler",   # "sampler" | "weight" | None
        val_ratio  = 0.2,
        seed       = 42,
        num_workers= 0,
    )
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet 정규화 통계 (사전학습 모델 표준값)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224   # EfficientNet / ResNet 입력 크기


# ─────────────────────────────────────────────────────────────────────────────
# 변환(Transform) 정의
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """학습용 변환 — Augmentation 포함."""
    return transforms.Compose([
        # ── 크기 & 공간 변환 ──────────────────────────────────────────────────
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.7, 1.0),       # 원본의 70~100% 크기 영역을 랜덤 크롭
            ratio=(0.75, 1.33),     # 가로세로 비율 범위
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),          # 좌우 반전 50%
        transforms.RandomVerticalFlip(p=0.1),            # 상하 반전 10%
        transforms.RandomRotation(
            degrees=15,                                   # ±15° 회전
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),

        # ── 색상 & 밝기 변환 ──────────────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.4,   # 밝기 ±40%
            contrast=0.4,     # 대비 ±40%
            saturation=0.3,   # 채도 ±30%
            hue=0.05,         # 색조 ±5%
        ),
        transforms.RandomGrayscale(p=0.05),              # 5% 확률로 흑백 변환

        # ── 텐서 변환 & 정규화 ────────────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── 추가 정규화 증강 ──────────────────────────────────────────────────
        transforms.RandomErasing(
            p=0.2,            # 20% 확률로 이미지 일부 영역 마스킹 (Cutout 효과)
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
        ),
    ])


def get_val_transform() -> transforms.Compose:
    """검증/테스트용 변환 — Augmentation 없음."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stratified Split 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    targets: list[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    클래스 분포를 유지하면서 인덱스를 train / val 로 분리.

    Parameters
    ----------
    targets   : 각 샘플의 클래스 인덱스 리스트
    val_ratio : 검증 셋 비율 (0 < val_ratio < 1)
    seed      : 랜덤 시드

    Returns
    -------
    train_indices, val_indices
    """
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {}
    for idx, cls in enumerate(targets):
        class_indices.setdefault(cls, []).append(idx)

    train_idx, val_idx = [], []
    for cls, idxs in sorted(class_indices.items()):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(targets: list[int]) -> WeightedRandomSampler:
    """
    클래스 불균형 보정용 WeightedRandomSampler 생성.
    소수 클래스 샘플이 더 자주 선택되도록 가중치를 역빈도로 설정.
    """
    count = Counter(targets)
    n_classes = len(count)
    n_total   = len(targets)

    class_weight = {
        cls: n_total / (n_classes * cnt)
        for cls, cnt in count.items()
    }
    sample_weights = torch.tensor(
        [class_weight[t] for t in targets], dtype=torch.float
    )
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 클래스 가중치 (CrossEntropyLoss용)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(targets: list[int], n_classes: int) -> torch.Tensor:
    """
    CrossEntropyLoss의 weight 파라미터에 사용할 클래스 가중치 계산.
    (역빈도 방식 — sklearn의 compute_class_weight 'balanced'와 동일)
    """
    count = Counter(targets)
    n_total = len(targets)
    weights = [
        n_total / (n_classes * count.get(i, 1))
        for i in range(n_classes)
    ]
    return torch.tensor(weights, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 빌더 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir:    str | Path = "./data",
    batch_size:  int  = 32,
    val_ratio:   float = 0.2,
    seed:        int   = 42,
    num_workers: int   = 0,
    imbalance:   Literal["sampler", "weight", None] = "sampler",
) -> tuple[DataLoader, DataLoader, list[str], torch.Tensor | None]:
    """
    ImageFolder 기반 DataLoader 빌더.

    Parameters
    ----------
    data_dir    : 클래스 하위폴더가 있는 루트 경로 (./data/fire, ./data/smoke, ...)
    batch_size  : 배치 크기
    val_ratio   : 검증 셋 비율 (기본 0.2 = 20%)
    seed        : 재현성 시드
    num_workers : DataLoader 워커 수 (Windows는 0 권장)
    imbalance   : 클래스 불균형 처리 방법
                  - "sampler" : WeightedRandomSampler (학습 루프 변경 불필요)
                  - "weight"  : class_weights 텐서 반환 (loss에 직접 전달)
                  - None      : 보정 없음

    Returns
    -------
    train_loader  : 학습용 DataLoader
    val_loader    : 검증용 DataLoader
    class_names   : 클래스 이름 리스트 (폴더 이름 순)
    class_weights : 클래스 가중치 텐서 (imbalance="weight"일 때만 값, 나머지 None)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {data_dir.resolve()}\n"
            f"  data/fire/, data/smoke/, data/normal/ 폴더를 만들고 이미지를 넣어주세요."
        )

    # ── 전체 데이터셋 로드 (변환은 나중에 Subset별로 적용) ───────────────────
    # 처음에는 변환 없이 로드해서 인덱스·타깃만 얻음
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names  = full_dataset.classes
    targets      = [s[1] for s in full_dataset.samples]

    print(f"\n{'='*55}")
    print(f"  데이터 전처리 & DataLoader 빌더")
    print(f"{'='*55}")
    print(f"  데이터 경로 : {data_dir.resolve()}")
    print(f"  클래스     : {class_names}")
    print(f"  전체 이미지 : {len(full_dataset)}장")
    count = Counter(targets)
    for i, cls in enumerate(class_names):
        n = count.get(i, 0)
        print(f"    [{i}] {cls:10s}: {n}장 ({n/len(full_dataset)*100:.1f}%)")

    # ── Stratified Split ─────────────────────────────────────────────────────
    train_idx, val_idx = stratified_split(targets, val_ratio=val_ratio, seed=seed)
    print(f"\n  Split (seed={seed})")
    print(f"    Train : {len(train_idx)}장 (80%)")
    print(f"    Val   : {len(val_idx)}장 (20%)")

    # ── 각 Split에 맞는 변환 적용용 래퍼 클래스 ────────────────────────────
    class _TransformSubset(torch.utils.data.Dataset):
        """Subset에 독립적인 변환을 적용하는 래퍼."""
        def __init__(self, base_dataset, indices, transform):
            self.base      = base_dataset
            self.indices   = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            img, label   = self.base[original_idx]   # PIL Image
            if self.transform:
                img = self.transform(img)
            return img, label

        @property
        def targets(self):
            return [self.base.targets[i] for i in self.indices]

    # ImageFolder에서 PIL을 그대로 받으려면 transform=None
    full_dataset.transform = None

    train_dataset = _TransformSubset(full_dataset, train_idx, get_train_transform())
    val_dataset   = _TransformSubset(full_dataset, val_idx,   get_val_transform())

    # ── 불균형 보정 ──────────────────────────────────────────────────────────
    train_targets   = [targets[i] for i in train_idx]
    sampler         = None
    class_weight_t  = None
    shuffle_train   = True

    if imbalance == "sampler":
        sampler       = make_weighted_sampler(train_targets)
        shuffle_train = False    # sampler 사용 시 shuffle=False 필수
        print(f"\n  불균형 보정 : WeightedRandomSampler ✅")

    elif imbalance == "weight":
        class_weight_t = compute_class_weights(train_targets, n_classes=len(class_names))
        print(f"\n  불균형 보정 : class_weights = {class_weight_t.tolist()} ✅")
        print(f"    → criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 로 사용")

    else:
        print(f"\n  불균형 보정 : 없음")

    # ── DataLoader 생성 ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = shuffle_train,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,     # 마지막 불완전 배치 제거 (BatchNorm 안정화)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    print(f"\n  DataLoader")
    print(f"    batch_size  : {batch_size}")
    print(f"    train 배치 수: {len(train_loader)}")
    print(f"    val   배치 수: {len(val_loader)}")
    print(f"    num_workers : {num_workers}")
    print(f"{'='*55}\n")

    return train_loader, val_loader, class_names, class_weight_t


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행 시 동작 확인
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data"

    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        data_dir   = data_path,
        batch_size = 32,
        val_ratio  = 0.2,
        imbalance  = "sampler",   # "sampler" | "weight" | None
        seed       = 42,
        num_workers= 0,
    )

    # 배치 샘플 확인
    imgs, labels = next(iter(train_loader))
    print(f"배치 이미지 shape : {imgs.shape}")   # (32, 3, 224, 224)
    print(f"배치 레이블 shape : {labels.shape}")  # (32,)
    print(f"픽셀 min/max      : {imgs.min():.3f} / {imgs.max():.3f}")
    print(f"클래스 이름       : {class_names}")
    print(f"class_weights     : {class_weights}")

    # 시각화 (증강 결과 확인)
    import matplotlib.pyplot as plt

    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD  = torch.tensor([0.229, 0.224, 0.225])

    def denorm(t):
        return torch.clamp(t * STD[:, None, None] + MEAN[:, None, None], 0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.patch.set_facecolor("#0F0F1A")
    fig.suptitle("Train DataLoader 배치 샘플 (정규화 역변환)", color="white", fontsize=13)

    CLASS_COLORS = {"fire": "#FF4D4D", "smoke": "#A0A0C0", "normal": "#4DFF91"}
    for i, ax in enumerate(axes.flat):
        if i >= imgs.shape[0]:
            ax.axis("off")
            continue
        img_show = denorm(imgs[i]).permute(1, 2, 0).numpy()
        ax.imshow(img_show)
        cls_name = class_names[labels[i].item()]
        ax.set_title(cls_name, color=CLASS_COLORS.get(cls_name, "white"), fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./outputs/dataloader_sample.png", dpi=100,
                bbox_inches="tight", facecolor="#0F0F1A")
    print("\n[저장] outputs/dataloader_sample.png")
    plt.show()
