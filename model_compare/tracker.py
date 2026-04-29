"""
Live Training Tracker
Watches a results.csv and shows live progress in its own terminal window.
Launched automatically by train.sh — don't run manually.

Usage:
    python model_compare/tracker.py --model YOLOv8n --csv path/to/results.csv --epochs 100
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta


# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
POLL_SEC        = 8     # refresh interval
NOTIFY_INTERVAL = 25    # send macOS notification every N epochs
STALE_SEC       = 60    # if csv hasn't changed in this long → training done


def notify(title: str, message: str, sound: str = "Glass") -> None:
    """Send desktop notification — macOS, Linux (notify-send), Windows (plyer fallback)."""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            script = f'display notification "{message}" with title "{title}" sound name "{sound}"'
            subprocess.run(["osascript", "-e", script], capture_output=True)
        elif system == "Linux":
            subprocess.run(["notify-send", title, message],
                           capture_output=True, timeout=3)
        elif system == "Windows":
            try:
                from plyer import notification as _n
                _n.notify(title=title, message=message, timeout=5)
            except ImportError:
                print(f"  [{title}] {message}")
    except Exception:
        pass


def read_results(csv_path: Path):
    """Return list of row dicts from results.csv."""
    try:
        with open(csv_path) as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()}
                    for r in csv.DictReader(f)]
        return [r for r in rows if r.get("epoch", "").strip()]
    except Exception:
        return []


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def progress_bar(current, total, width=30):
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = current / max(total, 1) * 100
    return f"[{bar}] {pct:.0f}%"


def eta_str(elapsed_sec, current, total):
    if current <= 0:
        return "--:--:--"
    per_epoch = elapsed_sec / current
    remaining = per_epoch * (total - current)
    return str(timedelta(seconds=int(remaining)))


def clear():
    os.system("clear")


def draw(model_name, rows, total_epochs, start_time, best_map50, last_best_epoch):
    clear()
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    if not rows:
        print(f"\n  Waiting for {model_name} training to start...")
        print(f"  (results.csv not yet created)\n")
        return

    latest = rows[-1]
    epoch     = int(safe_float(latest.get("epoch", 0)))
    map50     = safe_float(latest.get("metrics/mAP50(B)", 0))
    map5095   = safe_float(latest.get("metrics/mAP50-95(B)", 0))
    box_loss  = safe_float(latest.get("train/box_loss", 0))
    cls_loss  = safe_float(latest.get("train/cls_loss", 0))
    val_box   = safe_float(latest.get("val/box_loss", 0))
    val_cls   = safe_float(latest.get("val/cls_loss", 0))

    bar = progress_bar(epoch, total_epochs)
    eta = eta_str(elapsed, epoch, total_epochs)

    # Header
    print()
    print(f"  ╔{'═'*50}╗")
    print(f"  ║  {model_name} — Live Training Tracker{' '*(50 - len(model_name) - 27)}║")
    print(f"  ╚{'═'*50}╝")
    print()

    # Progress
    print(f"  Epoch      {epoch} / {total_epochs}")
    print(f"  Progress   {bar}")
    print(f"  Elapsed    {elapsed_str}   ETA {eta}")
    print()

    # Metrics
    print(f"  ┌{'─'*36}┐")
    print(f"  │  Validation Metrics              │")
    print(f"  ├{'─'*36}┤")
    print(f"  │  mAP50      {map50:>8.4f}              │")
    print(f"  │  mAP50-95   {map5095:>8.4f}              │")
    print(f"  ├{'─'*36}┤")
    print(f"  │  Train Loss                      │")
    print(f"  │    box      {box_loss:>8.4f}              │")
    print(f"  │    cls      {cls_loss:>8.4f}              │")
    print(f"  │  Val Loss                        │")
    print(f"  │    box      {val_box:>8.4f}              │")
    print(f"  │    cls      {val_cls:>8.4f}              │")
    print(f"  └{'─'*36}┘")
    print()

    # Best so far
    star = " ← NEW BEST" if map50 == best_map50 and map50 > 0 else ""
    print(f"  Best mAP50   {best_map50:.4f}  (epoch {last_best_epoch}){star}")
    print()
    print(f"  Refreshing every {POLL_SEC}s — Ctrl+C to close tracker")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Model name (e.g. YOLOv8n)")
    parser.add_argument("--csv",    required=True, help="Path to results.csv")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    csv_path    = Path(args.csv)
    total       = args.epochs
    model_name  = args.model
    start_time  = time.time()

    best_map50      = 0.0
    last_best_epoch = 0
    last_notify_ep  = 0
    last_mod_time   = 0

    notify(f"{model_name} Training Started",
           f"Training {total} epochs — tracker is live", "Tink")

    print(f"\n  Tracker started for {model_name}")
    print(f"  Watching: {csv_path}\n")

    try:
        while True:
            rows = read_results(csv_path)

            if rows:
                latest    = rows[-1]
                epoch     = int(safe_float(latest.get("epoch", 0)))
                map50     = safe_float(latest.get("metrics/mAP50(B)", 0))

                # New best
                if map50 > best_map50:
                    best_map50      = map50
                    last_best_epoch = epoch
                    notify(f"{model_name} — New Best!",
                           f"Epoch {epoch}: mAP50 = {map50:.4f}", "Ping")

                # Milestone notification every N epochs
                if epoch > 0 and epoch >= last_notify_ep + NOTIFY_INTERVAL:
                    last_notify_ep = epoch
                    notify(f"{model_name} — Epoch {epoch}/{total}",
                           f"mAP50 {map50:.4f}  |  Best {best_map50:.4f}")

                # Check if training finished (last epoch reached)
                if epoch >= total:
                    draw(model_name, rows, total, start_time, best_map50, last_best_epoch)
                    print(f"\n  Training complete!")
                    notify(f"{model_name} Training Complete!",
                           f"Best mAP50: {best_map50:.4f} at epoch {last_best_epoch}", "Glass")
                    break

                # Check for stale file (early stopping or crash)
                try:
                    mod = csv_path.stat().st_mtime
                    if mod != last_mod_time:
                        last_mod_time = mod
                    elif time.time() - mod > STALE_SEC and epoch > 0:
                        draw(model_name, rows, total, start_time, best_map50, last_best_epoch)
                        print(f"\n  Training stopped (early stopping or finished).")
                        notify(f"{model_name} Training Stopped",
                               f"Best mAP50: {best_map50:.4f} at epoch {last_best_epoch}", "Glass")
                        break
                except FileNotFoundError:
                    pass

            draw(model_name, rows, total, start_time, best_map50, last_best_epoch)
            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\n\n  Tracker closed.")


if __name__ == "__main__":
    main()
