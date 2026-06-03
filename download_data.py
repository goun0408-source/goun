"""
download_data.py — Kaggle에서 화재/연기 데이터셋 자동 다운로드 & 정리
=======================================================================
사용 전 필수:
  1. https://www.kaggle.com/settings → API → Create New Token
  2. 다운로드된 kaggle.json을 C:\\Users\\<사용자명>\\.kaggle\\kaggle.json 로 이동

실행:
  python download_data.py
"""

import os
import shutil
import zipfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("./data")
TEMP_DIR  = Path("./data/_temp")

# 사용할 Kaggle 데이터셋 (slug 형식: owner/dataset-name)
# D-Fire: 6천 장 이상, fire/smoke/none 세 클래스로 구분된 고품질 데이터셋
DATASET   = "tharakan2023/wildfire-detection-image-data"
ALT_DATASET = "phylake1337/fire-dataset"   # 대체 데이터셋

# ─────────────────────────────────────────────────────────────────────────────
# kaggle.json 확인
# ─────────────────────────────────────────────────────────────────────────────
kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_json.exists():
    print("=" * 60)
    print("  ❌  kaggle.json 파일이 없습니다!")
    print("=" * 60)
    print("""
  1. https://www.kaggle.com/settings 접속 (Kaggle 로그인 필요)
  2. [API] 섹션 → [Create New Token] 클릭
  3. 다운로드된 kaggle.json을 아래 경로로 이동:

     C:\\Users\\{사용자명}\\.kaggle\\kaggle.json

  폴더가 없으면 직접 만드세요:
     mkdir C:\\Users\\{사용자명}\\.kaggle
""")
    exit(1)

import kaggle   # kaggle.json 확인 후 import

# ─────────────────────────────────────────────────────────────────────────────
# 다운로드 & 압축 해제
# ─────────────────────────────────────────────────────────────────────────────
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"  데이터셋 다운로드: {DATASET}")
print("=" * 60)

try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET,
        path=str(TEMP_DIR),
        unzip=True,
        quiet=False,
    )
    print("✅ 다운로드 완료")
except Exception as e:
    print(f"❌ 다운로드 실패: {e}")
    print(f"\n대체 데이터셋으로 재시도: {ALT_DATASET}")
    try:
        kaggle.api.dataset_download_files(
            ALT_DATASET,
            path=str(TEMP_DIR),
            unzip=True,
            quiet=False,
        )
        print("✅ 대체 데이터셋 다운로드 완료")
    except Exception as e2:
        print(f"❌ 대체 데이터셋도 실패: {e2}")
        print("\n수동 다운로드 안내:")
        print("  https://www.kaggle.com/datasets/phylake1337/fire-dataset")
        exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 폴더 구조 자동 정리 → data/fire, data/smoke, data/normal
# ─────────────────────────────────────────────────────────────────────────────
print("\n[폴더 정리] data/ 구조로 재배치 중...")

TARGET_CLASSES = {
    "fire":   DATA_DIR / "fire",
    "smoke":  DATA_DIR / "smoke",
    "normal": DATA_DIR / "normal",
}
for d in TARGET_CLASSES.values():
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 다운로드된 파일의 폴더 구조를 탐색하여 클래스 매핑
alias_map = {
    # 가능한 폴더 이름들 → 우리 클래스
    "fire":        "fire",
    "fire_images": "fire",
    "fires":       "fire",
    "smoke":       "smoke",
    "smoke_images":"smoke",
    "smokes":      "smoke",
    "normal":      "normal",
    "non_fire":    "normal",
    "no_fire":     "normal",
    "none":        "normal",
    "neutral":     "normal",
    "background":  "normal",
    "other":       "normal",
}

moved = {"fire": 0, "smoke": 0, "normal": 0}

for folder in TEMP_DIR.rglob("*"):
    if not folder.is_dir():
        continue
    folder_name = folder.name.lower()
    cls = alias_map.get(folder_name)
    if cls is None:
        continue

    images = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS]
    dest   = TARGET_CLASSES[cls]
    for img in images:
        dst = dest / img.name
        # 파일 이름 충돌 방지
        if dst.exists():
            stem = img.stem
            suffix = img.suffix
            counter = 1
            while dst.exists():
                dst = dest / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(img, dst)
        moved[cls] += 1

# 결과 출력
print("\n[결과]")
total = 0
for cls, cnt in moved.items():
    print(f"  {cls:8s}: {cnt}장 → data/{cls}/")
    total += cnt
print(f"  합계   : {total}장")

if total == 0:
    print("""
  ⚠️ 이미지를 자동으로 분류하지 못했습니다.
  다운로드된 파일 구조가 예상과 다를 수 있습니다.
  아래 경로를 확인하고 수동으로 이동해 주세요:
""")
    for p in TEMP_DIR.rglob("*"):
        if p.is_dir():
            n = len([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])
            if n > 0:
                print(f"    {p}  ({n}장)")
else:
    # 정리 성공 시 임시 폴더 삭제
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("\n✅ 데이터 준비 완료!")
    print("  다음 명령으로 학습을 시작하세요:")
    print("  python train.py")
