#!/usr/bin/env python3
"""
make_overview.py — visual sweep of the cameras we'll use, so you can eyeball the
whole selection one by one without opening 95 streams.

Builds, from the frames already captured in angle_discovery/frames/:
  - overview/cameras_overview.mp4   slideshow, ~2s/camera, ordered by diversity
                                    rank, annotated with split/priority/classes
  - overview/contact_sheet_NN.png   grid pages — the whole set at a glance

PRIVACY
    Local only. Annotations are ascii camera_index + ranking/priority/classes —
    NO stream URLs, IPs, or hostnames are ever drawn or written.

USAGE
    conda activate diabetes-ai
    cd ~/Desktop/vehicle-counting-app/camera_audit
    python make_overview.py
"""
from __future__ import annotations
import csv, glob
from pathlib import Path
import cv2
import numpy as np

FRAMES = Path("angle_discovery/frames")
OUT = Path("overview"); OUT.mkdir(exist_ok=True)
LISTS = [("labeling_list.csv", "TRAIN"), ("validation_holdout.csv", "VAL")]

VID_W, VID_H = 960, 540
SECS_PER_CAM, FPS = 5, 5
TH_W, TH_H, COLS, PER_PAGE = 384, 216, 5, 30
PRIO_COLOR = {"high": (80, 220, 80), "medium": (60, 200, 240)}  # BGR; else red


def prio_color(p):
    return PRIO_COLOR.get(p.split()[0], (80, 80, 230))


def load_cameras():
    cams = []
    for fn, split in LISTS:
        if not Path(fn).exists():
            continue
        for r in csv.DictReader(open(fn)):
            cams.append({**r, "split": split})
    # frame path by camera_index
    for c in cams:
        hits = glob.glob(str(FRAMES / f"cam_{int(c['camera_index']):04d}_*.jpg"))
        c["frame"] = hits[0] if hits else None
    cams = [c for c in cams if c["frame"]]
    # global order by diversity score desc
    cams.sort(key=lambda c: -float(c["score"]))
    return cams


def annotate(img, c, idx, total, thumb=False):
    img = cv2.resize(img, (TH_W, TH_H) if thumb else (VID_W, VID_H))
    h, w = img.shape[:2]
    col = prio_color(c["label_priority"])
    scale = 0.45 if thumb else 0.7
    th = 1 if thumb else 2
    # top bar
    bar = int(26 if thumb else 38)
    ov = img.copy(); cv2.rectangle(ov, (0, 0), (w, bar), (0, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
    top = f"{idx}/{total}  [{c['split']}]  cam{c['camera_index']}  {c['label_priority'].split()[0].upper()}"
    cv2.putText(img, top, (6, int(bar*0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    # bottom bar: classes seen + score
    sub = f"vp{c['viewpoint']} score={c['score']} conf={c['n_confused_classes']} rare={c['n_rare_classes']}"
    cls = c.get("classes", "")[: (40 if thumb else 90)]
    ov = img.copy(); cv2.rectangle(ov, (0, h-(int(bar*1.2))), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
    cv2.putText(img, sub, (6, h-int(bar*0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale*0.9, (230,230,230), th, cv2.LINE_AA)
    if not thumb or True:
        cv2.putText(img, cls, (6, h-int(bar*0.18)), cv2.FONT_HERSHEY_SIMPLEX, scale*0.8, (180,255,180), th, cv2.LINE_AA)
    cv2.rectangle(img, (0,0), (w-1,h-1), col, 2 if thumb else 3)  # priority border
    return img


def build_video(cams):
    path = str(OUT / "cameras_overview.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VID_W, VID_H))
    n = len(cams)
    for i, c in enumerate(cams, 1):
        img = cv2.imread(c["frame"])
        if img is None:
            continue
        frame = annotate(img, c, i, n, thumb=False)
        for _ in range(SECS_PER_CAM * FPS):
            vw.write(frame)
    vw.release()
    return path


def build_contact_sheets(cams):
    pages, paths = [cams[i:i+PER_PAGE] for i in range(0, len(cams), PER_PAGE)], []
    for pi, page in enumerate(pages, 1):
        rows = []
        for r in range(0, len(page), COLS):
            chunk = page[r:r+COLS]
            tiles = []
            for j, c in enumerate(chunk):
                img = cv2.imread(c["frame"])
                tiles.append(annotate(img, c, pages.index(page)*PER_PAGE + r + j + 1, len(cams), thumb=True))
            while len(tiles) < COLS:
                tiles.append(np.zeros((TH_H, TH_W, 3), np.uint8))
            rows.append(np.hstack(tiles))
        sheet = np.vstack(rows)
        p = str(OUT / f"contact_sheet_{pi:02d}.png")
        cv2.imwrite(p, sheet); paths.append(p)
    return paths


def main():
    cams = load_cameras()
    print(f"cameras with frames: {len(cams)}")
    vid = build_video(cams)
    sheets = build_contact_sheets(cams)
    print(f"\n=== OVERVIEW BUILT (local) ===")
    print(f"  video        : {vid}  ({len(cams)} cams x {SECS_PER_CAM}s)")
    print(f"  contact sheets: {len(sheets)} page(s) in {OUT}/")
    print(f"  green border=high priority, yellow=medium, red=low/empty")


if __name__ == "__main__":
    main()
