"""
Real-Time Training Dashboard
Shows YOLOv8n and YOLO11n side by side with live batch-level progress.
Updates every second by reading the training log file.

Launched automatically by start_training.sh — or run manually:
    python model_compare/tracker_live.py
"""

import csv
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
BASE = Path(__file__).parent.parent

MODELS = {
    "YOLOv8n": {
        "csv":    BASE / "model_compare/yolov8n/run/results.csv",
        "log":    BASE / "model_compare/yolov8n/train.log",
        "epochs": 100,
        "color":  "dodger_blue1",
    },
    "YOLO11n": {
        "csv":    BASE / "model_compare/yolo11n/run/results.csv",
        "log":    BASE / "model_compare/yolo11n/train.log",
        "epochs": 100,
        "color":  "green3",
    },
}

REFRESH_SEC     = 1
NOTIFY_INTERVAL = 25
STALE_SEC       = 90

# Matches Ultralytics tqdm batch line (uses ── bars, not |█|):
# "[K  1/100  2.2G  1.227  4.616  1.003  97  416: 3% ──── 32/942 1.8s/it"
BATCH_RE = re.compile(
    r"(\d+)/(\d+)\s+[\d.]+G.*?:\s*(\d+)%[^0-9]+(\d+)/(\d+)"
)
# Validation phase (no GPU mem column):
VAL_RE = re.compile(
    r"(\d+)%[^0-9]+(\d+)/(\d+)"
)
# Strip ANSI escape codes from log
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\[K")


# ─────────────────────────────────────────
#  Log parsing — live batch progress
# ─────────────────────────────────────────
def parse_log(log_path: Path):
    """
    Read the last 8KB of the training log and extract the latest
    batch progress line. Returns dict or None.
    """
    if not log_path.exists():
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            raw = f.read()

        text = raw.decode("utf-8", errors="ignore")
        # Strip ANSI codes, split on \r and \n
        text  = ANSI_RE.sub("", text)
        lines = re.split(r"[\r\n]+", text)
        lines = [l.strip() for l in lines if l.strip()]

        # Scan from end for most recent batch progress line
        for line in reversed(lines):
            m = BATCH_RE.search(line)
            if m:
                ep_idx    = int(m.group(1))
                ep_total  = int(m.group(2)) + 1
                batch_pct = int(m.group(3))
                batch_cur = int(m.group(4))
                batch_tot = int(m.group(5))
                return {
                    "epoch":     ep_idx + 1,
                    "ep_total":  ep_total,
                    "batch_pct": batch_pct,
                    "batch_cur": batch_cur,
                    "batch_tot": batch_tot,
                    "phase":     "train",
                }
            v = VAL_RE.search(line)
            if v and "Class" not in line and "all" not in line:
                return {
                    "batch_pct": int(v.group(1)),
                    "batch_cur": int(v.group(2)),
                    "batch_tot": int(v.group(3)),
                    "phase":     "val",
                }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────
#  State per model
# ─────────────────────────────────────────
class ModelState:
    def __init__(self, name, cfg):
        self.name       = name
        self.cfg        = cfg
        self.color      = cfg["color"]
        self.total      = cfg["epochs"]
        self.rows       = []
        self.best_map50 = 0.0
        self.best_epoch = 0
        self.status     = "waiting"
        self.start_time = None
        self.last_mtime = 0
        self.last_notify_ep = 0
        self.notified_best  = -1.0
        # Live batch progress (from log file)
        self.batch_pct  = 0
        self.batch_cur  = 0
        self.batch_tot  = 0
        self.phase      = "train"   # "train" or "val"
        self.log_epoch  = 0        # epoch shown in log

    def refresh(self):
        # ── Read log for live batch progress ──
        log_info = parse_log(Path(self.cfg["log"]))
        if log_info:
            self.batch_pct = log_info.get("batch_pct", 0)
            self.batch_cur = log_info.get("batch_cur", 0)
            self.batch_tot = log_info.get("batch_tot", 0)
            self.phase     = log_info.get("phase", "train")
            if "epoch" in log_info:
                self.log_epoch = log_info["epoch"]
            if self.start_time is None:
                self.start_time = time.time()
                self.status = "training"

        # ── Read results.csv for epoch-level metrics ──
        csv_path = Path(self.cfg["csv"])
        if not csv_path.exists():
            return
        try:
            mtime = csv_path.stat().st_mtime
        except FileNotFoundError:
            return

        if self.start_time is None:
            self.start_time = time.time()
            self.status = "training"

        try:
            with open(csv_path) as f:
                rows = [{k.strip(): v.strip() for k, v in r.items()}
                        for r in csv.DictReader(f)]
            self.rows = [r for r in rows if r.get("epoch", "").strip()]
        except Exception:
            return

        if not self.rows:
            return

        latest = self.rows[-1]
        epoch  = self._int(latest.get("epoch", 0))
        map50  = self._float(latest.get("metrics/mAP50(B)", 0))

        if map50 > self.best_map50:
            self.best_map50 = map50
            self.best_epoch = epoch

        if epoch >= self.total:
            self.status = "done"
        elif mtime == self.last_mtime and self.status == "training":
            if time.time() - mtime > STALE_SEC:
                self.status = "stopped"
        self.last_mtime = mtime

    def latest(self):
        return self.rows[-1] if self.rows else {}

    def epoch(self):
        # Use log epoch if available (more real-time), else csv epoch
        return self.log_epoch or self._int(self.latest().get("epoch", 0))

    def elapsed(self):
        return time.time() - self.start_time if self.start_time else 0

    def eta(self):
        ep = self._int(self.latest().get("epoch", 0))  # use completed epochs for ETA
        if ep <= 0 or not self.start_time:
            return "--:--:--"
        per_ep    = self.elapsed() / ep
        remaining = per_ep * (self.total - ep)
        return str(timedelta(seconds=int(remaining)))

    @staticmethod
    def _float(v, d=0.0):
        try: return float(v)
        except: return d

    @staticmethod
    def _int(v, d=0):
        try: return int(float(v))
        except: return d


# ─────────────────────────────────────────
#  Notifications
# ─────────────────────────────────────────
def notify(title: str, msg: str, sound: str = "Glass") -> None:
    """Send desktop notification — macOS, Linux (notify-send), Windows (plyer fallback)."""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            script = f'display notification "{msg}" with title "{title}" sound name "{sound}"'
            subprocess.run(["osascript", "-e", script], capture_output=True)
        elif system == "Linux":
            subprocess.run(["notify-send", title, msg],
                           capture_output=True, timeout=3)
        elif system == "Windows":
            try:
                from plyer import notification as _n
                _n.notify(title=title, message=msg, timeout=5)
            except ImportError:
                print(f"  [{title}] {msg}")
    except Exception:
        pass


def check_notifications(states):
    for s in states.values():
        ep    = s._int(s.latest().get("epoch", 0))
        map50 = s._float(s.latest().get("metrics/mAP50(B)", 0))

        if map50 > s.notified_best and map50 > 0:
            s.notified_best = map50
            notify(f"{s.name} — New Best!", f"Epoch {ep}: mAP50 = {map50:.4f}", "Ping")

        if ep > 0 and ep >= s.last_notify_ep + NOTIFY_INTERVAL:
            s.last_notify_ep = ep
            notify(f"{s.name} — Epoch {ep}/{s.total}",
                   f"mAP50 {map50:.4f}  best {s.best_map50:.4f}")

        if s.status == "done" and s.last_notify_ep != -999:
            s.last_notify_ep = -999
            notify(f"{s.name} Training Complete!",
                   f"Best mAP50: {s.best_map50:.4f} at epoch {s.best_epoch}", "Glass")
        elif s.status == "stopped" and s.last_notify_ep != -998:
            s.last_notify_ep = -998
            notify(f"{s.name} — Early Stopped",
                   f"Best mAP50: {s.best_map50:.4f} at epoch {s.best_epoch}", "Bottle")


# ─────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────
STATUS_STYLE = {
    "waiting":  ("⏸  WAITING",  "dim"),
    "training": ("⚡ TRAINING", "bold yellow"),
    "done":     ("✅ COMPLETE", "bold green"),
    "stopped":  ("⏹  STOPPED",  "bold red"),
}


def make_bar(pct, width=30, color="white"):
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    return Text(f"[{bar}] {pct:3d}%", style=color)


def render_model_panel(s: ModelState) -> Panel:
    ep    = s.epoch()
    total = s.total
    r     = s.latest()
    f     = s._float

    map50   = f(r.get("metrics/mAP50(B)",    0))
    map95   = f(r.get("metrics/mAP50-95(B)", 0))
    box_tr  = f(r.get("train/box_loss",      0))
    cls_tr  = f(r.get("train/cls_loss",      0))
    box_val = f(r.get("val/box_loss",        0))
    cls_val = f(r.get("val/cls_loss",        0))

    status_label, status_style = STATUS_STYLE.get(s.status, ("?", "dim"))
    elapsed = str(timedelta(seconds=int(s.elapsed())))

    # Epoch-level progress %
    ep_pct = int(ep / max(total, 1) * 100)

    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="dim", no_wrap=True)
    t.add_column(justify="left",  no_wrap=True)

    def row(label, value, style="white"):
        t.add_row(label, Text(str(value), style=style))

    def blank():
        t.add_row("", "")

    if s.status == "waiting":
        t.add_row("", Text("Waiting for training to start…", style="dim"))
    else:
        # Epoch progress
        row("Epoch",   f"{ep} / {total}")
        t.add_row("Epochs", make_bar(ep_pct, width=26, color=s.color))

        blank()

        # Live batch progress
        if s.batch_tot > 0:
            phase_label = "Val batch" if s.phase == "val" else "Batch"
            row(phase_label, f"{s.batch_cur} / {s.batch_tot}")
            phase_color = "yellow" if s.phase == "val" else s.color
            t.add_row("", make_bar(s.batch_pct, width=26, color=phase_color))
        else:
            t.add_row("Batch", Text("waiting for log…", style="dim"))

        blank()
        row("Elapsed", elapsed)
        row("ETA",     s.eta())

        blank()
        row("mAP50",    f"{map50:.4f}" if map50 else "—", style="bold " + s.color)
        row("mAP50-95", f"{map95:.4f}" if map95 else "—")

        blank()
        row("Train box", f"{box_tr:.4f}"  if box_tr  else "—", style="dim")
        row("Train cls", f"{cls_tr:.4f}"  if cls_tr  else "—", style="dim")
        row("Val box",   f"{box_val:.4f}" if box_val else "—", style="dim")
        row("Val cls",   f"{cls_val:.4f}" if cls_val else "—", style="dim")

        blank()
        star = " ★ NEW" if map50 == s.best_map50 and map50 > 0 else ""
        row("Best mAP50",
            f"{s.best_map50:.4f}  (epoch {s.best_epoch}){star}",
            style="bold white")

    title    = Text(f"  {s.name}  ", style=f"bold {s.color}")
    subtitle = Text(f" {status_label} ", style=status_style)

    return Panel(t, title=title, subtitle=subtitle,
                 border_style=s.color if s.status == "training" else "dim")


def render_dashboard(states, wall_start):
    panels  = [render_model_panel(s) for s in states.values()]
    now     = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    elapsed = str(timedelta(seconds=int(time.time() - wall_start)))

    header = Table.grid(expand=True)
    header.add_column(justify="center")
    header.add_row(Text(
        f"YOLO Training Monitor    {now}    Total elapsed: {elapsed}",
        style="bold white"
    ))

    outer = Table.grid(padding=1)
    outer.add_row(header)
    outer.add_row(Columns(panels, equal=True, expand=True))
    outer.add_row(Text("  Refreshing every 1s — Ctrl+C to close", style="dim"))
    return outer


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    console    = Console()
    states     = {name: ModelState(name, cfg) for name, cfg in MODELS.items()}
    wall_start = time.time()

    notify("YOLO Tracker Started", "Watching YOLOv8n and YOLO11n", "Tink")

    with Live(console=console, refresh_per_second=1) as live:
        try:
            while True:
                for s in states.values():
                    s.refresh()
                check_notifications(states)
                live.update(render_dashboard(states, wall_start))

                finished = [s for s in states.values()
                            if s.status in ("done", "stopped")]
                if len(finished) == len(states):
                    time.sleep(3)
                    break

                time.sleep(REFRESH_SEC)
        except KeyboardInterrupt:
            pass

    console.print("\n[bold green]Tracker closed.[/bold green]")


if __name__ == "__main__":
    main()
