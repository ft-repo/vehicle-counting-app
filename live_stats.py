"""
Vehicle Detection System — Live Dashboard
iOS-inspired clean terminal UI.

Run alongside run_camera.sh in a separate terminal:
    python counting_app/live_stats.py
"""

import argparse
import csv
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

ROOT = Path(__file__).parent.parent

# ── iOS-inspired colour palette ───────────────────────────────────────────────
C_PRIMARY   = "white"
C_DIM       = "bright_black"      # muted gray  — "reduced opacity"
C_ACCENT    = "steel_blue1"       # iOS blue
C_GOOD      = "green"
C_WARN      = "yellow"
C_BAD       = "red"
C_METRIC    = "bold white"
C_BORDER    = "bright_black"      # subtle border

SPIN_FRAMES = ["·", "·", "·", "·", "·", "·", "·", "·"]   # subtle pulse
DOT_ON      = "●"
DOT_OFF     = "○"

_frame: int = 0
_fps_hist: deque = deque(maxlen=14)

VAL_RESULTS_PATH = ROOT / "val_results.json"

def load_class_accuracy() -> dict:
    """Load per-class mAP50 from val_results.json. Returns {} if not found."""
    try:
        if VAL_RESULTS_PATH.exists():
            data = json.loads(VAL_RESULTS_PATH.read_text())
            return data.get("classes", {}), data.get("overall_map50"), data.get("model", "")
    except Exception:
        pass
    return {}, None, ""

# ── Dataset targets ────────────────────────────────────────────────────────────
CLASSES_STATUS = {
    "person":  (45,    500,  30, 15),
    "car":     (26115, 26115, 60, 30),
    "bike":    (8596,  8596,  60, 30),
    "truck":   (2680,  2680,  60, 30),
    "bus":     (365,   500,  30, 15),
    "taxi":    (3520,  3520,  60, 30),
    "pickup":  (14620, 14620, 60, 30),
    "trailer": (690,   700,  50, 20),
    "tuktuk":  (37,    500,  30, 15),
    "agri":    (0,     500,  30, 15),
}

KNOWN_RESULTS = {
    "yolov8n run5":  ROOT / "runs/detect/model_compare/yolov8n/run5/results.csv",
    "yolov8n run6":  ROOT / "runs/detect/model_compare/yolov8n/run6/results.csv",
    "yolo11n run2":  ROOT / "runs/detect/model_compare/yolo11n/run2/results.csv",
    "yolo11n run":   ROOT / "runs/detect/model_compare/yolo11n/run/results.csv",
    "yolov4-tiny":   ROOT / "runs/detect/drr7_v1/results.csv",
    "yolov4":        ROOT / "runs/detect/drr7_v1/results.csv",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_str()  -> str: return datetime.now().strftime("%H:%M:%S")
def date_str() -> str: return datetime.now().strftime("%d %b %Y")

def fmt_uptime(s: int) -> str:
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

BAR_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(history: deque, width: int = 14) -> str:
    vals = list(history)
    if not vals: return "─" * width
    mx = max(vals) or 1
    return "".join(BAR_CHARS[min(int(v / mx * 8), 8)] for v in vals[-width:])

def ios_bar(value: float, max_val: float, width: int = 16) -> Text:
    """Clean iOS-style progress bar."""
    pct    = min(value / max(max_val, 1e-6), 1.0)
    filled = int(width * pct)
    bar    = "█" * filled + "─" * (width - filled)
    c = C_GOOD if pct >= 0.6 else (C_WARN if pct >= 0.3 else C_BAD)
    return Text(bar, style=f"dim {c}")

def pct_bar(current: int, target: int, width: int = 16) -> Text:
    pct    = min(current / max(target, 1), 1.0)
    filled = int(width * pct)
    bar    = "█" * filled + "─" * (width - filled)
    if pct >= 1.0:   c = C_GOOD
    elif pct >= 0.5: c = C_WARN
    elif pct >= 0.1: c = "orange3"
    else:            c = C_BAD
    return Text(bar, style=f"dim {c}")

def blink_dot() -> str:
    return DOT_ON if _frame % 2 == 0 else DOT_OFF

def read_stats(path: str) -> dict | None:
    try:
        d = json.load(open(path))
        return d if time.time() - d.get("ts", 0) < 10 else None
    except Exception:
        return None

def read_accuracy(path: Path) -> dict | None:
    try:
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append({k.strip(): v.strip() for k, v in row.items()})
        if not rows: return None
        last = rows[-1]
        return {
            "epoch":     last.get("epoch", "?"),
            "total":     len(rows),
            "map50":     float(last.get("metrics/mAP50(B)", 0)),
            "map50_95":  float(last.get("metrics/mAP50-95(B)", 0)),
            "precision": float(last.get("metrics/precision(B)", 0)),
            "recall":    float(last.get("metrics/recall(B)", 0)),
        }
    except Exception:
        return None

def find_results(model: str) -> Path | None:
    ml = model.lower().strip()
    for key, path in KNOWN_RESULTS.items():
        if key in ml or ml in key:
            if path.exists(): return path
    return None

def model_from_onnx(onnx: str) -> str:
    p = onnx.replace("\\", "/").lower()
    for arch in ("yolov8n", "yolov8s", "yolo11n", "yolo11s"):
        if arch in p:
            m = re.search(r'/(run\d+)/', p)
            return f"{arch} {m.group(1)}" if m else arch
    return "unknown"


# ── Panels ────────────────────────────────────────────────────────────────────

def _kv(label: str, value, label_w: int = 16) -> tuple:
    """Consistent key-value row."""
    return (Text(label, style=C_DIM, justify="left"),
            value if isinstance(value, Text) else Text(str(value), style=C_PRIMARY))


def build_header(stats: dict | None) -> Rule:
    dot    = blink_dot()
    online = stats is not None
    status = Text()
    status.append(f"  {dot} ", style=f"bold {C_GOOD}" if online else f"bold {C_BAD}")
    status.append("Online" if online else "Offline",
                  style=f"bold {C_GOOD}" if online else f"bold {C_BAD}")
    if online:
        status.append(f"   {stats.get('fps',0):.1f} fps", style=C_DIM)
    title = (f"[bold {C_ACCENT}]Vehicle Detection System[/bold {C_ACCENT}]"
             f"  [dim]{date_str()}  {now_str()}[/dim]")
    return Rule(title, style=C_DIM)


def build_feed_panel(stats: dict | None) -> Panel:
    g = Table.grid(padding=(0, 3))
    g.add_column(style=C_DIM,   width=18, no_wrap=True)
    g.add_column(style=C_PRIMARY, min_width=20)

    if stats is None:
        g.add_row("Status", Text(f"{DOT_OFF}  Offline", style=C_BAD))
        g.add_row("", Text("Start:  bash run_camera.sh", style=C_DIM))
    else:
        fps   = stats.get("fps", 0)
        spark = sparkline(_fps_hist)
        bar   = ios_bar(fps, 30)

        fps_text = Text()
        fps_text.append(f"{fps:.1f} fps", style=C_METRIC)
        fps_text.append(f"   {spark}", style=C_DIM)

        trk   = stats.get("active_tracks", 0)
        avg_c = stats.get("avg_conf", 0)
        model = stats.get("model", "—").upper()
        src   = stats.get("source", "")
        # Show only IP or last segment of URL
        m = re.search(r'(\d+\.\d+\.\d+\.\d+)', src)
        src_short = m.group(1) if m else src[-30:]

        g.add_row("Status",
                  Text(f"{DOT_ON}  Online", style=C_GOOD))
        g.add_row("FPS", fps_text)
        g.add_row("", bar)
        g.add_row("Active model",
                  Text(model, style=f"bold {C_ACCENT}"))
        g.add_row("Active tracks",
                  Text(str(trk), style=C_METRIC if trk > 0 else C_DIM))
        g.add_row("Avg confidence",
                  Text(f"{avg_c:.2f}",
                       style=C_GOOD if avg_c >= 0.7 else (C_WARN if avg_c >= 0.5 else C_BAD)))
        g.add_row("Uptime",
                  Text(fmt_uptime(stats.get("uptime_s", 0)), style=C_DIM))
        g.add_row("Source",
                  Text(src_short, style=C_DIM))

    return Panel(g, title=f"[{C_ACCENT}]Sensor Feed[/{C_ACCENT}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(1, 2))


def build_accuracy_panel(label: str, rcsv: Path | None) -> Panel:
    g = Table.grid(padding=(0, 3))
    g.add_column(style=C_DIM,   width=14, no_wrap=True)
    g.add_column(min_width=20)

    if rcsv is None or not rcsv.exists():
        g.add_row("Model", Text(label.upper(), style=f"bold {C_ACCENT}"))
        g.add_row("", Text("No results.csv found", style=C_DIM))
        g.add_row("", Text("Train a model first", style=C_DIM))
    else:
        acc = read_accuracy(rcsv)
        if not acc:
            g.add_row("", Text("Could not read results", style=C_DIM))
        else:
            m50  = acc["map50"]
            mc   = C_GOOD if m50 >= 0.6 else (C_WARN if m50 >= 0.4 else C_BAD)
            bar  = ios_bar(m50, 1.0, width=18)

            g.add_row("Model",     Text(label.upper(), style=f"bold {C_ACCENT}"))
            g.add_row("",          Text(""))
            g.add_row("mAP50",     Text(f"{m50:.3f}", style=f"bold {mc}"))
            g.add_row("",          bar)
            g.add_row("mAP50‑95",  Text(f'{acc["map50_95"]:.3f}', style=C_DIM))
            g.add_row("Precision", Text(f'{acc["precision"]:.3f}',
                                        style=C_GOOD if acc["precision"] >= 0.7 else C_WARN))
            g.add_row("Recall",    Text(f'{acc["recall"]:.3f}',
                                        style=C_GOOD if acc["recall"] >= 0.6 else C_WARN))
            g.add_row("Epochs",    Text(f'{acc["epoch"]} / {acc["total"]}',
                                        style=C_DIM))

    return Panel(g, title=f"[{C_ACCENT}]Model Accuracy[/{C_ACCENT}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(1, 2))


def build_comparison_panel(active_label: str) -> Panel:
    PRESETS = [
        ("1", "yolov4-tiny",  KNOWN_RESULTS.get("yolov4-tiny")),
        ("2", "yolov8n run5", KNOWN_RESULTS.get("yolov8n run5")),
        ("3", "yolo11n run2", KNOWN_RESULTS.get("yolo11n run2")),
    ]

    t = Table(box=None, show_header=True, header_style=C_DIM,
              padding=(0, 2))
    t.add_column("",         width=4,  style=C_DIM)
    t.add_column("Model",    width=14)
    t.add_column("mAP50",    width=8,  justify="right")
    t.add_column("Prec",     width=7,  justify="right")
    t.add_column("Recall",   width=8,  justify="right")
    t.add_column("Epochs",   width=8,  justify="right")
    t.add_column("",         width=12)   # status

    for key, model_key, rcsv_path in PRESETS:
        is_active = model_key.lower() in active_label.lower()
        acc       = read_accuracy(rcsv_path) if (rcsv_path and rcsv_path.exists()) else None

        key_t  = Text(f"[{key}]", style=C_ACCENT if is_active else C_DIM)
        name_t = Text(model_key, style=f"bold {C_PRIMARY}" if is_active else C_DIM)

        if is_active:
            status_t = Text(f"{DOT_ON} Running", style=C_GOOD)
        else:
            status_t = Text(f"{DOT_OFF} Idle",   style=C_DIM)

        if acc:
            m50 = acc["map50"]
            mc  = C_GOOD if m50 >= 0.6 else (C_WARN if m50 >= 0.4 else C_BAD)
            t.add_row(
                key_t, name_t,
                Text(f"{m50:.3f}", style=f"bold {mc}" if is_active else mc),
                Text(f'{acc["precision"]:.2f}', style=C_PRIMARY if is_active else C_DIM),
                Text(f'{acc["recall"]:.2f}',    style=C_PRIMARY if is_active else C_DIM),
                Text(str(acc["total"]),          style=C_DIM),
                status_t,
            )
        else:
            t.add_row(key_t, name_t,
                      Text("—", style=C_DIM), Text("—", style=C_DIM),
                      Text("—", style=C_DIM), Text("—", style=C_DIM),
                      status_t)

    subtitle = "[dim]Switch:  python counting_app/switch_model.py 1 | 2 | 3[/dim]"
    return Panel(t,
                 title=f"[{C_ACCENT}]Models[/{C_ACCENT}]  {subtitle}",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 2))


def build_counts_panel(stats: dict | None) -> Panel:
    t = Table(box=None, show_header=True, header_style=C_DIM,
              padding=(0, 2))
    t.add_column("Class",  width=10)
    t.add_column("Now",    width=5,  justify="right")
    t.add_column("In",     width=6,  justify="right")
    t.add_column("Out",    width=6,  justify="right")
    t.add_column("Total",  width=7,  justify="right")

    if not stats:
        t.add_row(Text("Waiting…", style=C_DIM), "—", "—", "—", "—")
    else:
        counts   = stats.get("class_counts", {})
        now_dets = stats.get("frame_dets", {})
        max_in   = max((v[0] for v in counts.values()), default=1)
        shown    = 0
        for cls in CLASSES_STATUS:
            inn, out = counts.get(cls, (0, 0))
            now_n    = now_dets.get(cls, 0)
            if inn == 0 and out == 0 and now_n == 0:
                continue
            total   = inn + out
            now_t   = Text(str(now_n), style=f"bold {C_WARN}") if now_n else Text("—", style=C_DIM)
            in_t    = Text(str(inn),   style=C_GOOD) if inn  else Text("0", style=C_DIM)
            out_t   = Text(str(out),   style=C_BAD)  if out  else Text("0", style=C_DIM)
            total_t = Text(str(total), style=C_METRIC)
            t.add_row(Text(cls, style=C_PRIMARY), now_t, in_t, out_t, total_t)
            shown += 1
        if shown == 0:
            t.add_row(Text("No vehicles detected", style=C_DIM), "—", "—", "—", "—")

        if stats.get("lanes"):
            t.add_section()
            for ln in stats["lanes"]:
                t.add_row(
                    Text(ln["name"], style=C_DIM),
                    Text("—",  style=C_DIM),
                    Text(str(ln["in"]),  style=C_GOOD),
                    Text(str(ln["out"]), style=C_BAD),
                    Text(str(ln["in"] + ln["out"]), style=C_METRIC),
                )

    return Panel(t, title=f"[{C_ACCENT}]Vehicle Counts[/{C_ACCENT}]  [dim]Now = in frame[/dim]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 2))


def build_events_panel(stats: dict | None) -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style=C_DIM,     width=9,  no_wrap=True)
    t.add_column(style=C_PRIMARY, width=9,  no_wrap=True)
    t.add_column(width=8,                   no_wrap=True)
    t.add_column(style=C_DIM,     width=10, no_wrap=True)

    events = (stats or {}).get("events", [])
    if not events:
        t.add_row("—", Text("No crossings yet", style=C_DIM), "", "")
    else:
        for i, ev in enumerate(events[:12]):
            arrow = Text("↑ In",  style=C_GOOD) if ev["dir"] == "IN" \
               else Text("↓ Out", style=C_BAD)
            name_style = C_PRIMARY if i == 0 else C_DIM
            t.add_row(ev["ts"],
                      Text(ev["cls"],  style=name_style),
                      arrow,
                      Text(ev["lane"], style=C_DIM))

    return Panel(t, title=f"[{C_ACCENT}]Crossings[/{C_ACCENT}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 2))


def build_dataset_panel() -> Panel:
    class_acc, overall_acc, acc_model = load_class_accuracy()

    t = Table(box=None, show_header=True, header_style=C_DIM,
              padding=(0, 2))
    t.add_column("Class",    width=10)
    t.add_column("Labeled",  width=8,  justify="right")
    t.add_column("Target",   width=8,  justify="right")
    t.add_column("Remaining",width=10, justify="right")
    t.add_column("Progress", width=20)
    t.add_column("Acc",      width=7,  justify="right")
    t.add_column("Status",   width=13)

    for cls, (current, target, day, night) in CLASSES_STATUS.items():
        need = max(0, target - current)
        bar  = pct_bar(current, target)

        if current >= target:
            status = Text("✓  Complete",   style=C_GOOD)
        elif current >= 50:
            status = Text("⚡  Auto-label", style=C_WARN)
        elif current > 0:
            status = Text("○  Need data",  style=C_BAD)
        else:
            status = Text("—  Empty",      style=C_DIM)

        # Accuracy cell
        ap = class_acc.get(cls)
        if ap is None:
            acc_cell = Text("—", style=C_DIM)
        elif ap >= 0.90:
            acc_cell = Text(f"{ap*100:.0f}%", style=C_GOOD)
        elif ap >= 0.60:
            acc_cell = Text(f"{ap*100:.0f}%", style=C_WARN)
        else:
            acc_cell = Text(f"{ap*100:.0f}%", style=C_BAD)

        t.add_row(
            Text(cls, style=C_PRIMARY),
            Text(f"{current:,}", style=C_METRIC if current >= target else C_DIM),
            Text(f"{target:,}",  style=C_DIM),
            Text(f"{need:,}" if need else "—", style=C_WARN if need else C_DIM),
            bar,
            acc_cell,
            status,
        )

    total_have   = sum(v[0] for v in CLASSES_STATUS.values())
    total_target = sum(v[1] for v in CLASSES_STATUS.values())

    # Overall accuracy cell
    if overall_acc is not None:
        oa = overall_acc
        oa_style = C_GOOD if oa >= 0.90 else (C_WARN if oa >= 0.60 else C_BAD)
        overall_acc_cell = Text(f"{oa*100:.0f}%", style=f"bold {oa_style}")
    else:
        overall_acc_cell = Text("—", style=C_DIM)

    t.add_section()
    t.add_row(
        Text("Total", style=C_DIM),
        Text(f"{total_have:,}",   style=C_METRIC),
        Text(f"{total_target:,}", style=C_DIM),
        Text(f"{max(0, total_target-total_have):,}" or "—", style=C_WARN),
        pct_bar(total_have, total_target),
        overall_acc_cell,
        Text(f"{total_have/total_target*100:.0f}%  labeled", style=C_DIM),
    )

    title_suffix = f" [{C_DIM}]({acc_model})[/{C_DIM}]" if acc_model else ""
    return Panel(t, title=f"[{C_ACCENT}]Dataset Progress[/{C_ACCENT}]{title_suffix}",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stats",   default="logs/live_stats.json")
    p.add_argument("--config",  default="counting_app/config/scene_config.json")
    p.add_argument("--results", default="")
    return p.parse_args()


def resolve_model(args, live_model: str | None) -> tuple[str, Path | None]:
    if args.results:
        p = Path(args.results)
        return (live_model or "custom", p if p.exists() else None)
    label = live_model
    if not label:
        try:
            cfg  = json.load(open(args.config))
            onnx = cfg.get("yolo", {}).get("onnx", "")
            if onnx: label = model_from_onnx(onnx)
        except Exception:
            pass
    label = label or "yolov8n run5"
    return label, find_results(label)


def main():
    global _frame

    args    = parse_args()
    console = Console()

    console.print()
    console.rule(f"[{C_ACCENT}]Vehicle Detection System[/{C_ACCENT}]", style=C_DIM)
    console.print(f"  [dim]Stats   [/dim] {args.stats}")
    console.print(f"  [dim]Config  [/dim] {args.config}")
    console.print(f"  [dim]Refresh [/dim] 0.5 s   ·   Ctrl+C to exit")
    console.print()
    time.sleep(0.5)

    last_label = "yolov8n run5"
    last_rcsv: Path | None = None

    with Live(console=console, refresh_per_second=2) as live:
        while True:
            stats = read_stats(args.stats)
            if stats:
                _fps_hist.append(stats.get("fps", 0))

            model_live  = (stats or {}).get("model")
            label, rcsv = resolve_model(args, model_live)
            if rcsv:
                last_label, last_rcsv = label, rcsv
            else:
                label, rcsv = last_label, last_rcsv

            # ── Layout ───────────────────────────────────────────────────────
            root = Layout()
            root.split_column(
                Layout(name="header",  size=1),
                Layout(name="row1",    size=12),
                Layout(name="models",  size=8),
                Layout(name="row2",    size=15),
                Layout(name="dataset", size=16),
            )

            root["header"].update(build_header(stats))

            root["row1"].split_row(
                Layout(build_feed_panel(stats),         ratio=3),
                Layout(build_accuracy_panel(label, rcsv), ratio=2),
            )

            root["models"].update(build_comparison_panel(label))

            root["row2"].split_row(
                Layout(build_counts_panel(stats),  ratio=3),
                Layout(build_events_panel(stats),  ratio=2),
            )

            root["dataset"].update(build_dataset_panel())

            live.update(root)
            _frame += 1
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print(f"\n[dim]Closed.[/dim]\n")
