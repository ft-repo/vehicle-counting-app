"""
Rename all files in vehicle_dataset/labeled/<class>/images+labels/
to a safe, transfer-ready format:

    <class>_<####>.jpg  +  <class>_<####>.txt

Dry-run by default — pass --execute to apply changes.

Usage:
    python tools/rename_dataset.py --ssd /Volumes/Puen_SSD/vehicle_dataset
    python tools/rename_dataset.py --ssd /Volumes/Puen_SSD/vehicle_dataset --execute
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES   = ['person','car','bike','truck','bus','taxi','pickup','trailer','tuktuk','agri_truck','van']
IMG_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp'}
SAFE_PAT  = re.compile(r'^[a-z_]+_\d{4}\.jpg$')   # already-renamed pattern


def parse_args():
    p = argparse.ArgumentParser(description="Rename dataset files to safe ASCII pattern")
    p.add_argument('--ssd',     required=True, help='Path to vehicle_dataset/ root')
    p.add_argument('--execute', action='store_true', help='Apply renames (default: dry-run)')
    return p.parse_args()


def rename_class(cls_dir: Path, cls_name: str, execute: bool) -> dict:
    img_dir = cls_dir / 'images'
    lbl_dir = cls_dir / 'labels'

    if not img_dir.exists():
        return {'skipped': 0, 'renamed': 0, 'no_label': 0}

    images = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in IMG_EXTS
    )

    stats = defaultdict(int)
    counter = 1

    for img in images:
        if SAFE_PAT.match(img.name):
            stats['skipped'] += 1
            continue

        lbl = lbl_dir / (img.stem + '.txt')

        new_stem  = f'{cls_name}_{counter:04d}'
        new_img   = img_dir / f'{new_stem}.jpg'
        new_lbl   = lbl_dir / f'{new_stem}.txt'

        # avoid collision with an already-renamed file
        while new_img.exists():
            counter += 1
            new_stem = f'{cls_name}_{counter:04d}'
            new_img  = img_dir / f'{new_stem}.jpg'
            new_lbl  = lbl_dir / f'{new_stem}.txt'

        print(f'  {img.name!r:50s} → {new_img.name}')

        if execute:
            img.rename(new_img)
            if lbl.exists():
                lbl.rename(new_lbl)
            else:
                stats['no_label'] += 1
        else:
            if not lbl.exists():
                stats['no_label'] += 1

        stats['renamed'] += 1
        counter += 1

    return stats


def main():
    args  = parse_args()
    root  = Path(args.ssd)
    labeled = root / 'labeled'

    if not labeled.exists():
        print(f'[ERROR] Not found: {labeled}')
        return

    mode = 'EXECUTE' if args.execute else 'DRY-RUN'
    print(f'\n{"="*55}')
    print(f'  RENAME DATASET  [{mode}]')
    print(f'{"="*55}')
    if not args.execute:
        print('  Pass --execute to apply changes.\n')

    total_renamed = 0
    total_skipped = 0

    for cls in CLASSES:
        cls_dir = labeled / cls
        if not cls_dir.exists():
            continue
        print(f'\n[{cls}]')
        stats = rename_class(cls_dir, cls, args.execute)
        print(f'  renamed: {stats["renamed"]}  skipped (already safe): {stats["skipped"]}  missing label: {stats["no_label"]}')
        total_renamed += stats['renamed']
        total_skipped += stats['skipped']

    print(f'\n{"="*55}')
    print(f'  Total renamed : {total_renamed}')
    print(f'  Already safe  : {total_skipped}')
    if not args.execute and total_renamed > 0:
        print(f'\n  Run with --execute to apply.')
    print(f'{"="*55}\n')


if __name__ == '__main__':
    main()
