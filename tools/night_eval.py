"""
night_eval.py — macro per-class day-vs-night scorecard for the deployed model.

Answers "which classes go blind at night?" WITHOUT needing to know the night
class balance in advance:

  - scores each class independently and weights them EQUALLY (macro), so a flood
    of night cars/bikes can't paper over a class that is quietly failing;
  - leads with RECALL at the operating confidence (a miss = an undercount, which
    is the failure that matters for a counter);
  - flags any class with too few night instances to judge (so the unknown night
    balance never produces a misleading number).

You supply a small labelled NIGHT ground-truth set (and optionally a DAY set to
get the Δ column — the real decision driver):

    # on the DGX (where the .pt and a GPU live):
    python tools/night_eval.py --night-data night.yaml --day-data day.yaml
    python tools/night_eval.py --night-data night.yaml            # night-only
    python tools/night_eval.py --night-data night.yaml --model path/to/best.pt

--model defaults to the deployed model from models/model_registry.json.

Verdict per class:
    night_inst < --min-instances     → "too few — collect more"   (can't judge)
    Δrecall <= --blind-threshold     → "NIGHT-BLIND → train"      (needs night data)
    otherwise                        → "fine"
A night-only run (no --day-data) judges on absolute recall vs --weak-recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from model_compare.registry import deployed_pt   # noqa: E402


# ─────────────────────────────────────────
#  GT instance counts — read straight from the label files (version-independent)
# ─────────────────────────────────────────
def labels_dir_for(images_dir: Path) -> Path:
    """YOLO convention: .../images/<split> ↔ .../labels/<split>. Swap the last
    'images' path component for 'labels'; fall back to the images dir itself."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir


def count_instances(data_yaml: Path, split: str) -> dict[int, int]:
    """Per-class GT box counts for `split` in a YOLO data.yaml. {} if not found."""
    import yaml  # provided transitively by ultralytics
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    rel = cfg.get(split)
    if not rel:
        return {}
    img_dir = Path(rel) if Path(rel).is_absolute() else (base / rel)
    lbl_dir = labels_dir_for(img_dir)
    if not lbl_dir.exists():
        lbl_dir = img_dir              # labels alongside images?
    counts: Counter = Counter()
    for txt in lbl_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            line = line.strip()
            if line:
                counts[int(line.split()[0])] += 1
    return dict(counts)


# ─────────────────────────────────────────
#  Per-class metrics from a val run (mirrors tools/backfill_per_class.py indexing)
# ─────────────────────────────────────────
def eval_per_class(model, data_yaml: Path, split: str, imgsz: int,
                   conf: float, iou: float, device: str | None):
    results = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                        conf=conf, iou=iou, device=device, verbose=False)
    b = results.box
    names: dict[int, str] = getattr(results, "names", None) or model.names
    out: dict[int, dict] = {}
    for cls_id in names:
        idx = int(cls_id)
        out[idx] = {
            "recall": float(b.r[idx])     if hasattr(b, "r")     and idx < len(b.r)     else None,
            "map50":  float(b.ap50[idx])  if hasattr(b, "ap50")  and idx < len(b.ap50)  else None,
        }
    return out, {int(k): v for k, v in names.items()}


def fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def main() -> int:
    p = argparse.ArgumentParser(description="Macro per-class day-vs-night scorecard.")
    p.add_argument("--night-data", required=True, help="data.yaml for the night GT set")
    p.add_argument("--day-data",   default=None,   help="data.yaml for a day set (enables Δ)")
    p.add_argument("--model",      default=None,   help="model .pt (default: deployed, from registry)")
    p.add_argument("--split",      default="val")
    p.add_argument("--imgsz",      type=int,   default=416)
    p.add_argument("--conf",       type=float, default=0.2,  help="operating confidence")
    p.add_argument("--iou",        type=float, default=0.5)
    p.add_argument("--device",     default=None, help="cpu / 0 / mps (default: auto)")
    p.add_argument("--min-instances", type=int,   default=50,
                   help="below this many night instances a class is 'too few to judge'")
    p.add_argument("--blind-threshold", type=float, default=-0.15,
                   help="Δrecall (night-day) at/below which a class is NIGHT-BLIND")
    p.add_argument("--weak-recall", type=float, default=0.60,
                   help="night-only mode: recall below this is 'weak at night'")
    p.add_argument("--out", default=None, help="write the scorecard as JSON")
    args = p.parse_args()

    # Lazy imports so --help and unit tests don't require rich/ultralytics.
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    console = Console()

    model_path = Path(args.model) if args.model else deployed_pt()
    if not model_path.exists():
        console.print(f"[red]model not found: {model_path}[/red]")
        console.print("[dim]The deployed .pt lives on the DGX (training host). "
                      "Run there, or pass --model.[/dim]")
        return 2

    night_yaml = Path(args.night_data)
    day_yaml   = Path(args.day_data) if args.day_data else None
    for y in (night_yaml, day_yaml):
        if y and not y.exists():
            console.print(f"[red]data.yaml not found: {y}[/red]")
            return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        console.print("[red]ultralytics not installed in this Python; activate the yolo env.[/red]")
        return 2

    console.print(f"[dim]model:  {model_path}[/dim]")
    model = YOLO(str(model_path))

    night_inst = count_instances(night_yaml, args.split)
    console.print(f"[dim]night :  {night_yaml}  ({sum(night_inst.values())} boxes)[/dim]")
    night_m, names = eval_per_class(model, night_yaml, args.split, args.imgsz,
                                    args.conf, args.iou, args.device)
    day_m = None
    if day_yaml:
        console.print(f"[dim]day   :  {day_yaml}[/dim]")
        day_m, _ = eval_per_class(model, day_yaml, args.split, args.imgsz,
                                  args.conf, args.iou, args.device)

    # ── Build rows + verdicts ──
    rows, scored = [], []          # scored = classes with enough night instances
    for idx in sorted(names):
        name = names[idx]
        ninst = night_inst.get(idx, 0)
        nr = night_m.get(idx, {}).get("recall")
        dr = day_m.get(idx, {}).get("recall") if day_m else None
        delta = (nr - dr) if (nr is not None and dr is not None) else None

        if ninst < args.min_instances:
            verdict, style = "too few — collect more", "dim"
        elif delta is not None:
            if delta <= args.blind_threshold:
                verdict, style = "NIGHT-BLIND → train", "bold red"
            else:
                verdict, style = "fine", "green"
            scored.append((nr, dr, delta))
        else:  # night-only judgement
            if nr is not None and nr < args.weak_recall:
                verdict, style = "weak at night", "bold red"
            else:
                verdict, style = "ok", "green"
            scored.append((nr, dr, delta))

        rows.append((name, ninst, dr, nr, delta, verdict, style))

    # ── Render ──
    t = Table(box=rich_box.SIMPLE, header_style="bold white")
    t.add_column("class", style="white")
    t.add_column("night_inst", justify="right")
    if day_m:
        t.add_column("day_recall", justify="right")
    t.add_column("night_recall", justify="right")
    if day_m:
        t.add_column("Δ", justify="right")
    t.add_column("verdict")
    for name, ninst, dr, nr, delta, verdict, style in rows:
        cells = [name, str(ninst)]
        if day_m:
            cells.append(fmt(dr))
        cells.append(fmt(nr))
        if day_m:
            cells.append(f"[{'red' if (delta is not None and delta <= args.blind_threshold) else 'dim'}]"
                         f"{('%+.2f' % delta) if delta is not None else '—'}[/]")
        cells.append(f"[{style}]{verdict}[/{style}]")
        t.add_row(*cells)
    console.print()
    console.print(t)

    # ── Macro summary (each judged class weighted equally → balance-independent) ──
    judged = [s for s in scored]
    if judged:
        macro_nr = sum(s[0] for s in judged if s[0] is not None) / max(1, sum(1 for s in judged if s[0] is not None))
        line = f"[bold]macro night recall:[/bold] {macro_nr:.2f}"
        if day_m:
            ds = [s for s in judged if s[2] is not None]
            if ds:
                macro_dr = sum(s[1] for s in ds) / len(ds)
                macro_d  = sum(s[2] for s in ds) / len(ds)
                line += f"   day {macro_dr:.2f}   Δ {macro_d:+.2f}"
        console.print("\n" + line + f"   [dim](over {len(judged)} classes with ≥{args.min_instances} night inst)[/dim]")

    blind = [r[0] for r in rows if "BLIND" in r[5] or "weak" in r[5]]
    if blind:
        console.print(f"[bold red]priority (collect + retrain):[/bold red] {', '.join(blind)}")
    thin = [r[0] for r in rows if "too few" in r[5]]
    if thin:
        console.print(f"[dim]insufficient night sample (collect before judging): {', '.join(thin)}[/dim]")

    if args.out:
        payload = {
            "model": str(model_path), "conf": args.conf, "split": args.split,
            "classes": {
                names[idx]: {
                    "night_inst": night_inst.get(idx, 0),
                    "day_recall": (day_m.get(idx, {}).get("recall") if day_m else None),
                    "night_recall": night_m.get(idx, {}).get("recall"),
                } for idx in sorted(names)
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        console.print(f"\n[dim]wrote {args.out}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
