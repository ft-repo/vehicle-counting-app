"""
Merge SSD dataset into new_data/dataset/ using symlinks.

Sources:
  - /Volumes/Puen_SSD/.../class7day/All/  → classes 0-7 (20K images)
  - new_data/dataset/images/train|val/    → existing tuktuk + multi-class images

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

# SSD path — override with env var if your drive is mounted elsewhere:
#   macOS  : /Volumes/Puen_SSD   (default)
#   Linux  : export SSD_PATH=/media/$USER/Puen_SSD
#   Windows: set SSD_PATH=D:\Puen_SSD
_default_ssd = {
    "Darwin":  "/Volumes/Puen_SSD",
    "Linux":   "/media/" + os.environ.get("USER", "user") + "/Puen_SSD",
    "Windows": "D:\\Puen_SSD",
}.get(platform.system(), "/Volumes/Puen_SSD")

SSD_ROOT = Path(os.environ.get("SSD_PATH", _default_ssd))
SSD_ALL  = SSD_ROOT / "CV/App_Yolo_Model_Training/images/class7day/All"

VAL_SPLIT   = 0.20   # 20% of SSD images go to val
RANDOM_SEED = 42


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def make_link(src: Path, dst: Path) -> None:
    """Create symlink (macOS/Linux) or copy (Windows), skip if already exists."""
    if dst.exists() or dst.is_symlink():
        return
    if platform.system() == "Windows":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def collect_ssd_pairs(folder: Path):
    """Return list of (image_path, label_path) from SSD flat folder."""
    pairs = []
    for img in folder.glob("*.jpg"):
        lbl = img.with_suffix(".txt")
        if lbl.exists() and img.stem != "classes":
            pairs.append((img, lbl))
    return pairs


def count_existing(img_dir: Path):
    """Count real files (not symlinks) in an image directory."""
    return sum(1 for f in img_dir.iterdir()
               if f.is_file() and not f.is_symlink() and f.suffix in (".jpg", ".png"))


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MERGE DATASET")
    print("="*55)

    # Check SSD is mounted
    if not SSD_ALL.exists():
        print(f"\n[ERROR] SSD not found: {SSD_ALL}")
        print("        Plug in Puen_SSD and try again.")
        sys.exit(1)

    # Create directory structure
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Collect SSD pairs ──
    print(f"\nScanning SSD dataset...")
    pairs = collect_ssd_pairs(SSD_ALL)
    print(f"  Found: {len(pairs)} image/label pairs")

    if not pairs:
        print("[ERROR] No pairs found in SSD folder.")
        sys.exit(1)

    # ── Shuffle and split ──
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    split_idx  = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # ── Create symlinks (copy on Windows) ──
    action = "Copying files" if platform.system() == "Windows" else "Creating symlinks"
    print(f"\n{action}...")
    created = {"train": 0, "val": 0}

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        for img_src, lbl_src in split_pairs:
            make_link(img_src, img_dir / img_src.name)
            make_link(lbl_src, lbl_dir / lbl_src.name)
            created[split] += 1

    print(f"  Train symlinks: {created['train']}")
    print(f"  Val   symlinks: {created['val']}")

    # ── Count existing real files (our new_data images) ──
    real_train = count_existing(DATASET / "images/train")
    real_val   = count_existing(DATASET / "images/val")
    print(f"\nExisting labeled images kept:")
    print(f"  Train: {real_train}  Val: {real_val}")

    # ── Update data.yaml ──
    yaml_path = DATASET / "data.yaml"
    yaml_content = f"""path: {DATASET}
train: images/train
val: images/val
nc: 10
names: ['person','car','bike','truck','bus','taxi','pickup','trailer','tuktuk','agri_truck']
"""
    yaml_path.write_text(yaml_content)
    print(f"\nUpdated: {yaml_path}")

    # ── Summary ──
    total_train = sum(1 for _ in (DATASET / "images/train").iterdir()
                      if _.suffix in (".jpg", ".png"))
    total_val   = sum(1 for _ in (DATASET / "images/val").iterdir()
                      if _.suffix in (".jpg", ".png"))

    print(f"\n{'='*55}")
    print(f"  DATASET READY")
    print(f"{'='*55}")
    print(f"  Train: {total_train} images")
    print(f"  Val:   {total_val} images")
    print(f"  Total: {total_train + total_val} images")
    print(f"\n  Classes 0-7: from SSD (car, bike, truck, etc.)")
    print(f"  Class 8:     tuktuk (your labeled images)")
    print(f"  Class 9:     agri_truck (add more labels when ready)")
    print(f"\n  Next: train with")
    print(f"    ./model_compare/yolov8n/train.sh")
    print(f"    ./model_compare/yolo11n/train.sh")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
