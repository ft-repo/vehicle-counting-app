"""
Merge SSD per-class labeled data into new_data/dataset/ for training.

Source structure on SSD:
  vehicle_dataset/labeled/<class>/images/  +  labels/

Run from project root:
    python model_compare/merge_dataset.py
"""

import os
import shutil
import sys
import platform
import random
from pathlib import Path

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATASET = BASE / "new_data/dataset"

CLASS_NAMES = ['person','car','bike','truck','bus','taxi','pickup','trailer','tuktuk','agri_truck','van']

# SSD path — override with env var if your drive is mounted elsewhere:
#   macOS  : /Volumes/Puen_SSD   (default)
#   Linux  : export SSD_PATH=/media/$USER/Puen_SSD
#   Windows: set SSD_PATH=D:\Puen_SSD
_default_ssd = {
    "Darwin":  "/Volumes/Puen_SSD",
    "Linux":   "/media/" + os.environ.get("USER", "user") + "/Puen_SSD",
    "Windows": "D:\\Puen_SSD",
}.get(platform.system(), "/Volumes/Puen_SSD")

SSD_ROOT    = Path(os.environ.get("SSD_PATH", _default_ssd))
SSD_LABELED = SSD_ROOT / "vehicle_dataset/labeled"

VAL_SPLIT   = 0.20
RANDOM_SEED = 42


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def make_link(src: Path, dst: Path) -> None:
    """Symlink on macOS/Linux, copy on Windows. Skip if already exists."""
    if dst.exists() or dst.is_symlink():
        return
    if platform.system() == "Windows":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def collect_pairs(cls_dir: Path):
    """Return (image_path, label_path) pairs from labeled/<class>/images+labels/."""
    pairs = []
    img_dir = cls_dir / "images"
    lbl_dir = cls_dir / "labels"
    if not img_dir.exists():
        return pairs
    for img in img_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MERGE DATASET")
    print("="*55)

    if not SSD_LABELED.exists():
        print(f"\n[ERROR] SSD not found: {SSD_LABELED}")
        print("        Plug in Puen_SSD and try again.")
        sys.exit(1)

    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Collect all pairs from every class folder ──
    print(f"\nScanning SSD labeled folders...")
    all_pairs = []
    for cls in CLASS_NAMES:
        cls_dir = SSD_LABELED / cls
        pairs   = collect_pairs(cls_dir)
        print(f"  {cls:12}: {len(pairs)}")
        all_pairs.extend(pairs)

    print(f"\n  Total: {len(all_pairs)} image/label pairs")

    if not all_pairs:
        print("[ERROR] No pairs found.")
        sys.exit(1)

    # ── Shuffle and split ──
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    split_idx   = int(len(all_pairs) * (1 - VAL_SPLIT))
    train_pairs = all_pairs[:split_idx]
    val_pairs   = all_pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # ── Symlink into dataset/ ──
    action = "Copying files" if platform.system() == "Windows" else "Creating symlinks"
    print(f"\n{action}...")
    created = {"train": 0, "val": 0}
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        for img_src, lbl_src in pairs:
            make_link(img_src, img_dir / img_src.name)
            make_link(lbl_src, lbl_dir / lbl_src.name)
            created[split] += 1
    print(f"  Train: {created['train']}  Val: {created['val']}")

    # ── Write data.yaml ──
    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(f"""path: {DATASET}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
    print(f"\nUpdated: {yaml_path}")

    # ── Summary ──
    total_train = sum(1 for f in (DATASET / "images/train").iterdir() if f.suffix in (".jpg", ".png", ".jpeg"))
    total_val   = sum(1 for f in (DATASET / "images/val").iterdir()   if f.suffix in (".jpg", ".png", ".jpeg"))

    print(f"\n{'='*55}")
    print(f"  DATASET READY")
    print(f"{'='*55}")
    print(f"  Train: {total_train}  Val: {total_val}  Total: {total_train + total_val}")
    print(f"\n  Next: train with")
    print(f"    ./model_compare/yolov8n/train.sh")
    print(f"    ./model_compare/yolo11n/train.sh")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
