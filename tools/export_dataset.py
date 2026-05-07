"""
Build a self-contained, transfer-ready dataset copy with real files (no symlinks).
Intended for one-time transfer to the AI training PC.

Usage:
    python tools/export_dataset.py \
        --ssd /Volumes/Puen_SSD/vehicle_dataset \
        --out /Volumes/Puen_SSD/vehicle_dataset_export

Output structure:
    vehicle_dataset_export/
    ├── images/train/   ← real .jpg files
    ├── images/val/
    ├── labels/train/   ← real .txt files
    ├── labels/val/
    └── data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

CLASSES     = ['person','car','bike','truck','bus','taxi','pickup','trailer','tuktuk','agri_truck','van']
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp'}
VAL_SPLIT   = 0.20
RANDOM_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description='Export transfer-ready dataset copy')
    p.add_argument('--ssd', required=True, help='Path to vehicle_dataset/ root')
    p.add_argument('--out', required=True, help='Output folder for the export')
    return p.parse_args()


def collect_pairs(labeled_root: Path):
    pairs = []
    for cls in CLASSES:
        img_dir = labeled_root / cls / 'images'
        lbl_dir = labeled_root / cls / 'labels'
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + '.txt')
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def main():
    args    = parse_args()
    ssd     = Path(args.ssd)
    out     = Path(args.out)
    labeled = ssd / 'labeled'

    if not labeled.exists():
        print(f'[ERROR] Not found: {labeled}')
        return

    print(f'\n{"="*55}')
    print(f'  EXPORT DATASET')
    print(f'  → {out}')
    print(f'{"="*55}')

    # Create output dirs
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all pairs
    print('\nScanning labeled folders...')
    all_pairs = collect_pairs(labeled)
    print(f'  Found: {len(all_pairs)} image/label pairs')

    if not all_pairs:
        print('[ERROR] No pairs found. Run merge_dataset.py first or check the SSD path.')
        return

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    cut         = int(len(all_pairs) * (1 - VAL_SPLIT))
    train_pairs = all_pairs[:cut]
    val_pairs   = all_pairs[cut:]
    print(f'  Train: {len(train_pairs)}  Val: {len(val_pairs)}')

    # Copy files (real copies — no symlinks)
    print('\nCopying files...')
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        img_dir = out / 'images' / split
        lbl_dir = out / 'labels' / split
        for i, (img_src, lbl_src) in enumerate(pairs):
            if i % 2000 == 0 and i > 0:
                print(f'  {split}: {i}/{len(pairs)}...')
            shutil.copy2(str(img_src), str(img_dir / img_src.name))
            shutil.copy2(str(lbl_src), str(lbl_dir / lbl_src.name))

    # Write data.yaml with relative paths (works on any machine)
    yaml_path = out / 'data.yaml'
    yaml_path.write_text(f"""path: .
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
""")

    # Summary
    total_train = sum(1 for f in (out / 'images' / 'train').iterdir() if f.suffix in ('.jpg', '.png', '.jpeg'))
    total_val   = sum(1 for f in (out / 'images' / 'val').iterdir()   if f.suffix in ('.jpg', '.png', '.jpeg'))

    print(f'\n{"="*55}')
    print(f'  EXPORT COMPLETE')
    print(f'{"="*55}')
    print(f'  Train : {total_train}')
    print(f'  Val   : {total_val}')
    print(f'  Total : {total_train + total_val}')
    print(f'  Path  : {out}')
    print(f'\n  Transfer this folder to the AI PC.')
    print(f'  On the AI PC, train with:')
    print(f'    yolo train data=data.yaml model=yolov8n.pt imgsz=416')
    print(f'{"="*55}\n')


if __name__ == '__main__':
    main()
