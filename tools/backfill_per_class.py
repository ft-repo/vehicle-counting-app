"""Backfill per-class mAP50 / precision / recall into a synthesized val_results.json.

When `vehicle_counter.py` was first wired to write a val_results.json (2026-05-08),
the synth path only persisted aggregate metrics from `results.csv`. The new
`tools/level_gate.py` needs PER-CLASS mAP50 to decide which classes are
Level-2+ (eligible for SAM2 auto-labeling) — so we re-run validation once and
fill in the missing `classes` block.

Idempotent: reads the existing JSON, only writes new keys, preserves anything
that's already there.

Usage
-----
    python tools/backfill_per_class.py \\
        --weights runs/yolo26n/run1/weights/best.pt \\
        --data    /home/admin/cv_counting/data/data.yaml \\
        --out     runs/yolo26n/run1/val_results.json \\
        [--imgsz 416] [--device cpu]

After it runs, `level_gate.load_per_class_levels()` will return a populated
{class_id: 'level1'|'level2'} dict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data",    required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--imgsz",   type=int, default=416)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--split",   default="val")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed in this Python; activate yolo-env first.", file=sys.stderr)
        return 2

    weights = Path(args.weights)
    if not weights.exists():
        print(f"weights not found: {weights}", file=sys.stderr)
        return 2

    print(f"[backfill] loading {weights}")
    model = YOLO(str(weights))

    print(f"[backfill] validating against {args.data}  imgsz={args.imgsz}  device={args.device}")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        verbose=False,
    )

    # Per-class metrics: results.box.maps is a numpy array of mAP50-95 per class;
    # results.box.ap50 is mAP50 per class; results.names maps id -> name.
    box = results.box
    names: dict[int, str] = getattr(results, "names", None) or model.names

    classes: dict[str, dict] = {}
    for cls_id, name in names.items():
        idx = int(cls_id)
        classes[str(idx)] = {
            "name":      name,
            "map50":     float(box.ap50[idx]) if idx < len(box.ap50) else None,
            "map50_95":  float(box.maps[idx]) if idx < len(box.maps) else None,
            "precision": float(box.p[idx])    if hasattr(box, "p")    and idx < len(box.p)    else None,
            "recall":    float(box.r[idx])    if hasattr(box, "r")    and idx < len(box.r)    else None,
        }

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            print(f"[backfill] warning: {out_path} unreadable, starting fresh")

    # Aggregate metrics — refresh if the val results give us fresh numbers.
    existing.setdefault("metrics/mAP50(B)",     float(box.map50))
    existing.setdefault("metrics/mAP50-95(B)",  float(box.map))
    existing["classes"] = classes
    existing["per_class_backfilled_at"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[backfill] wrote {out_path}  ({len(classes)} classes)")
    for cls_id, m in sorted(classes.items(), key=lambda kv: int(kv[0])):
        print(f"  {cls_id:>2}  {m['name']:<12}  mAP50={m['map50']!s:>7}  mAP50-95={m['map50_95']!s:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
