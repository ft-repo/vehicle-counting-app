"""
Auto-Label Tool
Runs YOLOv8n on a folder of unlabeled images and writes YOLO-format .txt labels.
Interns must review every label before using in training.

Run:
    python model_compare/auto_label.py --images path/to/images [--conf 0.40]

Output:
    A .txt file alongside each image (same name, .txt extension)
    One line per detection: class_id cx cy w h  (all 0-1 normalized)
    Images with no detections above threshold get NO .txt file — skip them.
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
DEFAULT_MODEL = "runs/detect/model_compare/yolov8n/run5/weights/best.pt"
DEFAULT_CONF  = 0.40   # discard detections below this
CLASS_NAMES   = ['person','car','bike','truck','bus','taxi','pickup','trailer','tuktuk','van','agri_truck']
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Auto-label images with YOLOv8n")
    p.add_argument("--images", required=True,
                   help="Folder of unlabeled images")
    p.add_argument("--model",  default=DEFAULT_MODEL,
                   help=f"Path to best.pt  (default: {DEFAULT_MODEL})")
    p.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                   help=f"Confidence threshold  (default: {DEFAULT_CONF})")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .txt label files")
    return p.parse_args()


def collect_images(folder: Path, overwrite: bool):
    """Return list of images that need labeling."""
    all_imgs = [f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not overwrite:
        pending = [f for f in all_imgs
                   if not f.with_suffix(".txt").exists()]
    else:
        pending = all_imgs
    return sorted(all_imgs), sorted(pending)


def write_label(txt_path: Path, results, conf_thresh: float) -> int:
    """Write YOLO label file; return number of boxes written."""
    lines = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        cx, cy, w, h = box.xywhn[0].tolist()   # normalized
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if lines:
        txt_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    args  = parse_args()
    root  = Path(os.getcwd())
    img_dir = Path(args.images)
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    if not img_dir.exists():
        console.print(f"[red]ERROR: image folder not found: {img_dir}[/red]")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        console.print(f"[red]ERROR: model not found: {model_path}[/red]")
        console.print("[dim]Run with --model path/to/best.pt[/dim]")
        sys.exit(1)

    all_imgs, pending = collect_images(img_dir, args.overwrite)
    skipped = len(all_imgs) - len(pending)

    console.print()
    console.rule("[bold white]AUTO-LABEL TOOL[/bold white]")
    console.print()

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="white")
    info.add_row("Image folder",  str(img_dir))
    info.add_row("Model",         str(model_path))
    info.add_row("Conf threshold",f"{args.conf}  (discard below this)")
    info.add_row("Total images",  str(len(all_imgs)))
    info.add_row("To label",      str(len(pending)))
    info.add_row("Already done",  str(skipped))
    console.print(Panel(info, title="[bold dodger_blue1]Config[/bold dodger_blue1]",
                        border_style="dodger_blue1"))
    console.print()

    if not pending:
        console.print("[green]All images already labeled. Use --overwrite to redo.[/green]")
        console.print()
        return

    # Load model
    console.print("[dim]Loading model...[/dim]")
    model = YOLO(str(model_path))
    console.print("[green]Model loaded.[/green]\n")

    # ── Run inference ──
    stats          = {name: 0 for name in CLASS_NAMES}
    labeled_count  = 0    # images that got ≥1 box
    total_boxes    = 0
    skipped_nodet  = 0    # images with 0 detections above threshold

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Auto-labeling...", total=len(pending))

        for img_path in pending:
            results = model(str(img_path), conf=args.conf, verbose=False)
            txt_path = img_path.with_suffix(".txt")
            n_boxes  = write_label(txt_path, results, args.conf)

            if n_boxes > 0:
                labeled_count += 1
                total_boxes   += n_boxes
                # tally per-class
                for box in results[0].boxes:
                    if float(box.conf[0]) >= args.conf:
                        cls_id = int(box.cls[0])
                        if cls_id < len(CLASS_NAMES):
                            stats[CLASS_NAMES[cls_id]] += 1
            else:
                skipped_nodet += 1

            progress.advance(task)

    # ── Summary ──
    console.print()
    console.rule("[bold white]RESULTS[/bold white]")
    console.print()

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    t.add_column("Class",       style="white",  width=12)
    t.add_column("Boxes",       style="cyan",   width=10, justify="right")
    t.add_column("Status",      width=18)

    for name in CLASS_NAMES:
        count = stats[name]
        if count == 0:
            status = "[dim]—[/dim]"
        elif count < 50:
            status = "[yellow]⚠ need more[/yellow]"
        else:
            status = "[green]✓ OK[/green]"
        t.add_row(name, str(count), status)

    console.print(t)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="white")
    summary.add_row("Images labeled",       str(labeled_count))
    summary.add_row("Images no detection",  str(skipped_nodet))
    summary.add_row("Total boxes written",  str(total_boxes))

    console.print(Panel(summary,
                        title="[bold green]Summary[/bold green]",
                        border_style="green"))
    console.print()

    console.print("[bold yellow]⚠  NEXT STEP: Intern must review every label in Label Studio before training![/bold yellow]")
    console.print()
    console.rule()
    console.print()


if __name__ == "__main__":
    main()
