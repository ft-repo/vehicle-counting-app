"""Class-maturity level gate.

Reads a populated `val_results.json` (see `tools/backfill_per_class.py`) and
returns a per-class maturity label:

    Level 1 (mAP50 < threshold)  ->  manual labelling only — DINO assist OK,
                                     SAM2 chain skipped, output goes to review.
    Level 2 (mAP50 >= threshold) ->  SAM2 chain enabled — combined-score
                                     filter applies (auto_accept / review).

The threshold defaults to 0.50 (per `wiki/concepts/lifecycle-workflow.md`
"Score-tier routing" Level 1 / Level 2 boundary).

Used by `tools/gdino_ls_backend.py` to decide whether to run SAM2 mask
refinement on each predicted box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_per_class_levels(
    val_json_path: str,
    threshold: float = 0.50,
) -> dict[int, str]:
    """Return {class_id: 'level1'|'level2'} for each class found in the JSON.

    Tolerant of an empty / missing `classes` block (returns {}). Caller decides
    fallback behaviour for unknown class IDs.
    """
    p = Path(val_json_path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}

    classes = payload.get("classes") or {}
    out: dict[int, str] = {}
    for cls_id, metrics in classes.items():
        try:
            cid = int(cls_id)
        except (TypeError, ValueError):
            continue
        m50 = metrics.get("map50") if isinstance(metrics, Mapping) else None
        if m50 is None:
            continue
        out[cid] = "level2" if float(m50) >= float(threshold) else "level1"
    return out


def is_level2(class_id: int, levels: Mapping[int, str]) -> bool:
    """True iff the class is known and at Level 2+. Unknown classes => False."""
    return levels.get(int(class_id)) == "level2"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: level_gate.py <val_results.json> [threshold]", file=sys.stderr)
        sys.exit(64)
    thr = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.50
    levels = load_per_class_levels(sys.argv[1], thr)
    if not levels:
        print(f"(no per-class metrics in {sys.argv[1]} — backfill needed)")
    else:
        for cid in sorted(levels):
            print(f"  {cid:>2}  {levels[cid]}")
