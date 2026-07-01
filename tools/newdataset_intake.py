"""
newdataset_intake.py — gather the dataset into ONE organized folder.

RUN ON THE DGX. Copies images + their YOLO labels from the source folder(s)
into /home/admin/newdataset/{images,labels}, drops exact-duplicate files
(same content), then zips it to /home/admin/newdataset.zip.

    python tools/newdataset_intake.py                         # uses SOURCES below
    python tools/newdataset_intake.py /home/admin/<dataset>   # or pass folder(s)

Pure standard library.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

DEST = Path("/home/admin/newdataset")

# Where to gather from. Edit or pass folders on the command line.
SOURCES = [
    "/home/admin/cv_counting/exports/ls_p2_yolo_20260605_dir",  # labeled dataset
    "/home/admin/auto_capture",                                 # captured frames
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_label(img: Path):
    """YOLO .txt for an image: same-name sibling, or images/ → labels/."""
    sib = img.with_suffix(".txt")
    if sib.exists():
        return sib
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            return cand if cand.exists() else None
    return None


def main():
    sources = [Path(p) for p in (sys.argv[1:] or SOURCES)]
    (DEST / "images").mkdir(parents=True, exist_ok=True)
    (DEST / "labels").mkdir(parents=True, exist_ok=True)

    seen = {p.stem for p in (DEST / "images").iterdir()
            if p.suffix.lower() in IMG_EXTS}          # skip what's already there
    kept = dups = labeled = 0

    for root in sources:
        if not root.exists():
            print(f"skip (missing): {root}")
            continue
        for dirpath, _dirs, files in os.walk(root):
            here = Path(dirpath).resolve()
            if here == DEST.resolve() or DEST.resolve() in here.parents:
                continue                              # never re-scan our output
            for fn in files:
                if not fn.lower().endswith(IMG_EXTS):
                    continue
                img = Path(dirpath) / fn
                digest = md5(img)
                if digest in seen:
                    dups += 1
                    continue
                seen.add(digest)
                shutil.copy2(img, DEST / "images" / f"{digest}{img.suffix.lower()}")
                label = find_label(img)
                if label:
                    shutil.copy2(label, DEST / "labels" / f"{digest}.txt")
                    labeled += 1
                kept += 1
        print(f"gathered: {root}")

    print(f"\n{kept} images ({labeled} with labels), {dups} duplicates skipped → {DEST}")
    zip_path = shutil.make_archive(str(DEST), "zip",
                                   root_dir=str(DEST.parent), base_dir=DEST.name)
    print(f"zipped → {zip_path}")


if __name__ == "__main__":
    main()
