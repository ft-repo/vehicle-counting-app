#!/usr/bin/env python3
"""
Cross-platform launcher for the Vehicle Detection System.
Works on macOS, Windows, and Linux — no shell-specific code needed.

Reads the camera source from config/scene_config.json automatically.

Usage:
    python run_camera.py
    python run_camera.py --config config/scene_config.json
    python run_camera.py --source rtsp://...
    python run_camera.py --gpu
    python run_camera.py --cpu
    python run_camera.py --nowin
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).parent.resolve()
PYTHON = sys.executable
SYSTEM = platform.system()   # "Darwin" | "Windows" | "Linux"


def open_dashboard(stats_path: str) -> None:
    """Open live_stats.py in a new terminal window (platform-specific)."""
    dashboard  = str(ROOT / "live_stats.py")
    stats_abs  = str(Path(stats_path).resolve()) if not Path(stats_path).is_absolute() else stats_path

    if SYSTEM == "Darwin":
        args_str = f'\\"{PYTHON}\\" \\"{dashboard}\\" --stats \\"{stats_abs}\\"'
        script = (
            'tell application "Terminal"\n'
            f'  do script "cd \\"{ROOT}\\" && {args_str}"\n'
            '  set custom title of front window to "DRR Vision HUD"\n'
            '  activate\n'
            'end tell'
        )
        subprocess.Popen(["osascript", "-e", script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elif SYSTEM == "Windows":
        inner = f'"{PYTHON}" "{dashboard}" --stats "{stats_abs}"'
        subprocess.Popen(
            f'start "DRR Vision HUD" cmd /k "cd /d "{ROOT}" && {inner}"',
            shell=True
        )

    else:
        # Linux — try common terminal emulators in order
        cmd = [PYTHON, dashboard, "--stats", stats_abs]
        terminals = [
            ["gnome-terminal", "--",          *cmd],
            ["xterm",          "-e",          *cmd],
            ["konsole",        "-e",          *cmd],
            ["xfce4-terminal", "-x",          *cmd],
            ["lxterminal",     "-e",          *cmd],
            ["mate-terminal",  "-x",          *cmd],
            ["tilix",          "-e",          *cmd],
            ["kitty",          "--",          *cmd],
            ["alacritty",      "-e",          *cmd],
            ["wezterm",        "start", "--", *cmd],
        ]
        for t in terminals:
            try:
                subprocess.Popen(t, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue
        print("[INFO] No terminal emulator found for dashboard.")
        print(f"       Open a new terminal and run:")
        print(f"       python live_stats.py --stats {stats_path}")


def main():
    p = argparse.ArgumentParser(
        description="Vehicle Detection System — cross-platform launcher"
    )
    p.add_argument("--config", default="config/scene_config.json",
                   help="scene_config.json (default: config/scene_config.json)")
    p.add_argument("--stats",  default="live_stats.json",
                   help="Live stats output JSON (default: live_stats.json)")
    p.add_argument("--source", default="",
                   help="Camera URL / file path (overrides config if set)")
    p.add_argument("--gpu",   action="store_true", help="Force GPU / CUDA backend")
    p.add_argument("--cpu",   action="store_true", help="Force CPU backend")
    p.add_argument("--nowin", action="store_true", help="Headless mode (no display window)")
    args = p.parse_args()

    config_path = ROOT / args.config

    # Resolve camera source
    source = args.source
    if not source and config_path.exists():
        try:
            cfg    = json.loads(config_path.read_text())
            source = cfg.get("camera", {}).get("source", "")
        except Exception:
            pass
    if not source:
        print("[ERROR] No camera source found.")
        print(f"        Set 'camera.source' in {config_path}  or use  --source <url>")
        sys.exit(1)

    print()
    print("  Vehicle Detection System")
    print("  " + "─" * 44)
    print(f"  Platform : {SYSTEM} ({platform.machine()})")
    print(f"  Python   : {PYTHON}")
    print(f"  Config   : {config_path}")
    print(f"  Source   : {source}")
    print(f"  Backend  : {'GPU (forced)' if args.gpu else 'CPU (forced)' if args.cpu else 'auto'}")
    print()
    print("  Model switcher (open a NEW terminal):")
    print("    python switch_model.py 1   (YOLOv4-tiny)")
    print("    python switch_model.py 2   (YOLOv8n run5)")
    print("    python switch_model.py 3   (YOLO11n run2)")
    print()

    # Open HUD in a new window
    open_dashboard(args.stats)
    time.sleep(0.8)

    # Build vehicle_counter command
    counter_cmd = [
        PYTHON, str(ROOT / "vehicle_counter.py"),
        source,
        "--config", str(config_path),
        "--stats",  args.stats,
    ]
    if args.gpu:    counter_cmd.append("--gpu")
    if args.cpu:    counter_cmd.append("--cpu")
    if args.nowin:  counter_cmd.append("--nowin")

    try:
        subprocess.run(counter_cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")


if __name__ == "__main__":
    main()
