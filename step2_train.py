"""
Step 2: EfficientNet-B0 전이학습 (Transfer Learning)
- Freeze Backbone → Head 학습 (5 에폭)
- Unfreeze 일부 → Fine-tuning (15 에폭)
- AdamW + CosineAnnealingLR
- 조기 종료 (patience=5)
- 혼합 정밀도 (GPU 있을 경우 자동)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ── 경로 & 하이퍼파라미터 ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./outputs")
MODEL_DIR  = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
SPLIT_PATH = OUTPUT_DIR / "split_info.json"

CLASSES    = ["fire", "smoke", "normal"]
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42

EPOCHS_HEAD  = 5    # 헤드만 학습
EPOCHS_FINE  = 20   # 전체 fine-tuning
PATIENCE     = 5    # 조기 종료

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 장치 설정 ────────────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[장치] {device}")
USE_AMP = device.type == "cuda"

# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 & 데이터로더
# ─────────────────────────────────────────────────────────────────────────────
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_info = json.load(f)

mean = split_info["mean"]
std  = split_info["std"]
class_to_idx = split_info["class_to_idx"]
class_weights = torch.tensor(split_info["class_weights"], dtype=torch.float)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


class DisasterDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = [class_to_idx[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def make_loader(split_key, transform, shuffle=False):
    data = split_info[split_key]
    ds   = DisasterDataset(data["paths"], data["labels"], transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))


train_loader = make_loader("train", train_transform, shuffle=True)
val_loader   = make_loader("val",   val_transform)
test_loader  = make_loader("test",  val_transform)

print(f"[데이터] Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

# ─────────────────────────────────────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes=3):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # 분류 헤드 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


model = build_model(num_classes=len(CLASSES)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="  Train" if train else "  Val  ", leave=False, ncols=80):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out  = model(imgs)
                loss = criterion(out, labels)

            if train and optimizer:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: 헤드만 학습
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 1 — Head 학습 (Backbone Frozen)")
print("=" * 60)

freeze_backbone(model, freeze=True)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
best_val_loss = float("inf")
best_epoch    = 0
patience_cnt  = 0

for epoch in range(1, EPOCHS_HEAD + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, train=True)
    va_loss, va_acc = run_epoch(model, val_loader, train=False)
    scheduler.step()

    lr_now = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["lr"].append(lr_now)

    print(f"  Ep{epoch:02d}/{EPOCHS_HEAD} | "
          f"loss {tr_loss:.4f}/{va_loss:.4f} | "
          f"acc {tr_acc*100:.1f}%/{va_acc*100:.1f}% | "
          f"lr {lr_now:.2e} | {time.time()-t0:.1f}s")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("  조기 종료 (Phase 1)")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Fine-tuning (backbone 일부 unfreeze)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 2 — Fine-tuning (Partial Unfreeze)")
print("=" * 60)

freeze_backbone(model, freeze=False)

# 마지막 몇 개 블록만 학습 (features[6:])
for name, param in model.named_parameters():
    if "features" in name:
        block_num = name.split(".")[1]
        param.requires_grad = int(block_num) >= 5

optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "classifier" in n],       "lr": 1e-3},
    {"params": [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad], "lr": 1e-4},
], weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINE)
patience_cnt = 0

for epoch in range(1, EPOCHS_FINE + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, train=True)
    va_loss, va_acc = run_epoch(model, val_loader, train=False)
    scheduler.step()

    lr_now = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["lr"].append(lr_now)

    ep_total = EPOCHS_HEAD + epoch
    print(f"  Ep{ep_total:02d}/{EPOCHS_HEAD+EPOCHS_FINE} | "
          f"loss {tr_loss:.4f}/{va_loss:.4f} | "
          f"acc {tr_acc*100:.1f}%/{va_acc*100:.1f}% | "
          f"lr {lr_now:.2e} | {time.time()-t0:.1f}s")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_epoch    = ep_total
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        patience_cnt  = 0
        print("  ✅ best model 저장!")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  조기 종료 (에폭 {ep_total}, best={best_epoch})")
            break

# ─────────────────────────────────────────────────────────────────────────────
# 학습 곡선 저장
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F0F1A", "axes.facecolor": "#1A1A2E",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": "#333355", "grid.color": "#333355",
})

epochs_range = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("학습 결과", fontsize=14, color="white")

ax = axes[0]
ax.plot(epochs_range, history["train_loss"], label="Train", color="#FF7B7B")
ax.plot(epochs_range, history["val_loss"],   label="Val",   color="#7BFFB0")
ax.axvline(best_epoch, color="gold", linestyle="--", alpha=0.7, label=f"Best ep{best_epoch}")
ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train", color="#FF7B7B")
ax.plot(epochs_range, [a * 100 for a in history["val_acc"]],   label="Val",   color="#7BFFB0")
ax.set_title("Accuracy (%)"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(epochs_range, history["lr"], color="#7BB8FF")
ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_yscale("log"); ax.grid(alpha=0.3)

plt.tight_layout()
curve_path = OUTPUT_DIR / "training_curves.png"
plt.savefig(curve_path, dpi=120, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print(f"\n[저장] {curve_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 완료 요약
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  Step 2 완료!")
print("=" * 60)
print(f"  Best Val Loss : {best_val_loss:.4f}  (에폭 {best_epoch})")
print(f"  모델 저장     : models/best_model.pth")
print(f"  곡선 저장     : outputs/training_curves.png")
print("=" * 60)
