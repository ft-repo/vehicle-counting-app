#!/usr/bin/env python3
"""
build_lists.py — turn the diversity ranking into a leak-free train / validation
camera split for run7 labeling.

Reads  : diversity_scores.csv  (produced by score_diversity.py)
Writes : labeling_list.csv            (TRAIN — cameras to pull + label frames from)
         validation_holdout.csv       (VAL  — unseen cameras, labeled but never trained)

SELECTION LOGIC
    Cameras are ranked by class diversity + traffic density. We pull BOTH train
    and val from the *diverse* cameras (those that showed >=1 confused or rare
    class) so validation actually contains the confused classes; every 4th
    diverse camera (by rank) is reserved for val. Train is filled to --train-size
    highest-score first; low-signal cameras spill into val as low priority.
    The split is camera-level (no camera in both) so val measures generalization
    to unseen cameras.

PRIORITY TIERS (label top-down)
    high   : >=2 confused classes, or (>=1 confused AND >=1 rare)
    medium : >=1 confused or >=1 rare class
    low    : nothing seen yet — snapshot may have been a quiet moment; verify
             (run more score_diversity.py sessions to resolve these)

USAGE
    conda activate diabetes-ai
    cd ~/Desktop/vehicle-counting-app/camera_audit
    python build_lists.py                 # default train-size 60
    python build_lists.py --train-size 80 --val-every 4
"""
from __future__ import annotations
import argparse, csv
from collections import Counter
from pathlib import Path

SRC = "diversity_scores.csv"
TRAIN_OUT = "labeling_list.csv"
VAL_OUT = "validation_holdout.csv"
# redacted, safe-to-commit twins (no stream_url / camera_id)
TRAIN_MANIFEST = "labeling_manifest.csv"
VAL_MANIFEST = "validation_manifest.csv"
REDACT_DROP = {"stream_url", "camera_id", "labeled", "frames_pulled", "notes"}

LOW = "low (snapshot empty — verify)"


def tier(r):
    c, ra = int(r["n_confused_classes"]), int(r["n_rare_classes"])
    if c >= 2 or (c >= 1 and ra >= 1):
        return "high"
    if c >= 1 or ra >= 1:
        return "medium"
    return LOW


def is_diverse(r):
    return int(r["n_confused_classes"]) >= 1 or int(r["n_rare_classes"]) >= 1


def emit(fn, data, split):
    cols = ["rank_by_diversity", "label_priority", "viewpoint", "score",
            "n_confused_classes", "n_rare_classes", "avg_objs_per_frame",
            "frames_seen", "sessions", "camera_index", "camera_id", "classes",
            "stream_url", "split", "labeled", "frames_pulled", "notes"]
    data = sorted(data, key=lambda x: -float(x["score"]))
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, r in enumerate(data, 1):
            w.writerow({
                "rank_by_diversity": i, "label_priority": tier(r),
                "viewpoint": r.get("viewpoint", ""), "score": r["score"],
                "n_confused_classes": r["n_confused_classes"],
                "n_rare_classes": r["n_rare_classes"],
                "avg_objs_per_frame": r["avg_objs_per_frame"],
                "frames_seen": r.get("frames_seen", ""),
                "sessions": r.get("sessions", ""),
                "camera_index": r["camera_index"], "camera_id": r["camera_id"],
                "classes": r.get("classes", ""), "stream_url": r["stream_url"],
                "split": split, "labeled": "", "frames_pulled": "", "notes": "",
            })


def emit_manifest(fn, data, split):
    """Redacted, safe-to-commit version: drops stream_url / camera_id / tracking."""
    cols = ["rank_by_diversity", "label_priority", "viewpoint", "score",
            "n_confused_classes", "n_rare_classes", "avg_objs_per_frame",
            "frames_seen", "sessions", "camera_index", "classes", "split"]
    data = sorted(data, key=lambda x: -float(x["score"]))
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, r in enumerate(data, 1):
            w.writerow({
                "rank_by_diversity": i, "label_priority": tier(r),
                "viewpoint": r.get("viewpoint", ""), "score": r["score"],
                "n_confused_classes": r["n_confused_classes"],
                "n_rare_classes": r["n_rare_classes"],
                "avg_objs_per_frame": r["avg_objs_per_frame"],
                "frames_seen": r.get("frames_seen", ""),
                "sessions": r.get("sessions", ""),
                "camera_index": r["camera_index"],
                "classes": r.get("classes", ""), "split": split,
            })


def composition(data, label):
    t = Counter(tier(r) for r in data)
    withc = sum(1 for r in data if int(r["n_confused_classes"]) >= 1)
    withr = sum(1 for r in data if int(r["n_rare_classes"]) >= 1)
    print(f"  {label}: {len(data):3d} cams | high={t['high']} med={t['medium']} "
          f"low={t.get(LOW, 0)} | confused>=1: {withc} | rare>=1: {withr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-size", type=int, default=60,
                    help="max cameras in the training (labeling) list")
    ap.add_argument("--val-every", type=int, default=4,
                    help="reserve every Nth diverse camera (by rank) for validation")
    args = ap.parse_args()

    if not Path(SRC).exists():
        raise SystemExit(f"[FATAL] {SRC} not found — run score_diversity.py first.")
    rows = list(csv.DictReader(open(SRC)))   # already sorted by score desc
    if not rows:
        raise SystemExit(f"[FATAL] {SRC} is empty.")

    diverse = [r for r in rows if is_diverse(r)]
    val_ids = {r["camera_index"] for r in diverse[args.val_every - 1::args.val_every]}

    train, holdout = [], []
    for r in rows:
        (holdout if r["camera_index"] in val_ids else train).append(r)

    train.sort(key=lambda r: -float(r["score"]))
    keep, overflow = train[:args.train_size], train[args.train_size:]
    holdout += overflow
    train = keep

    emit(TRAIN_OUT, train, "TRAIN")
    emit(VAL_OUT, holdout, "VAL_HOLDOUT_unseen_camera")
    # redacted twins — safe to commit / share (no URLs or IDs)
    emit_manifest(TRAIN_MANIFEST, train, "TRAIN")
    emit_manifest(VAL_MANIFEST, holdout, "VAL_HOLDOUT_unseen_camera")

    ti = {r["camera_index"] for r in train}
    hi = {r["camera_index"] for r in holdout}
    sessions = max((int(r["sessions"]) for r in rows if r.get("sessions", "").isdigit()),
                   default=0)
    print(f"=== LISTS REBUILT FROM {SRC} "
          f"(diversity over {sessions} session(s)) ===")
    composition(train, f"TRAIN ({TRAIN_OUT})")
    composition(holdout, f"VAL   ({VAL_OUT})")
    print(f"  overlap (must be 0): {len(ti & hi)}")
    if any(tier(r) == LOW for r in train):
        print("  NOTE: 'low' cameras showed nothing yet — run more "
              "score_diversity.py busy-window sessions to resolve them.")


if __name__ == "__main__":
    main()
