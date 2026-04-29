"""
Frame Extractor
===============
Extracts frames from a video file or RTSP stream at a fixed time interval.
Deduplicates near-identical frames using perceptual hashing.

Usage:
    python tools/frame_extractor.py \\
        --source "rtsp://root:pass@100.115.149.76/axis-media/media.amp" \\
        --output raw_frames/tuk_tuk \\
        --interval 3 \\
        --max 1000

    python tools/frame_extractor.py \\
        --source traffic_video.mp4 \\
        --output raw_frames/tuk_tuk \\
        --interval 2 \\
        --max 500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from video or RTSP")
    p.add_argument("--source",   required=True,
                   help="RTSP URL or video file path")
    p.add_argument("--output",   required=True,
                   help="Output folder for saved frames")
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds between saved frames (default: 3)")
    p.add_argument("--max",      type=int,   default=1000,
                   help="Maximum frames to save (default: 1000)")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip perceptual hash deduplication")
    p.add_argument("--dedup-threshold", type=int, default=8,
                   help="Hash distance threshold for dedup (default: 8)")
    return p.parse_args()


def _try_import_imagehash():
    try:
        import imagehash
        from PIL import Image
        return imagehash, Image
    except ImportError:
        print("[WARN] imagehash not installed — dedup disabled")
        print("       Install with: pip install imagehash Pillow")
        return None, None


def main():
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dedup setup ──────────────────────────────────────────────────────────
    use_dedup = not args.no_dedup
    imagehash_mod, PIL_Image = (None, None)
    if use_dedup:
        imagehash_mod, PIL_Image = _try_import_imagehash()
        if imagehash_mod is None:
            use_dedup = False
    seen_hashes = {}

    # ── Open video source ─────────────────────────────────────────────────────
    print(f"[EXTRACT] Source:   {args.source}")
    print(f"[EXTRACT] Output:   {out_dir.resolve()}")
    print(f"[EXTRACT] Interval: {args.interval}s")
    print(f"[EXTRACT] Max:      {args.max} frames")
    print(f"[EXTRACT] Dedup:    {'yes (threshold=' + str(args.dedup_threshold) + ')' if use_dedup else 'no'}")
    print()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    # ── Determine if source is a file (use frame position) or stream (use time)
    is_file     = not args.source.startswith("rtsp://")
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip  = max(1, int(fps * args.interval))  # frames between saves (file mode)

    saved       = 0
    skipped_dup = 0
    frame_idx   = 0
    last_save_t = time.time() - args.interval   # allow saving immediately

    print("[EXTRACT] Starting...")

    while saved < args.max:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                print("[EXTRACT] End of video file.")
            else:
                print("[EXTRACT] Stream ended or connection lost.")
            break

        frame_idx += 1

        # ── Throttle: only consider saving every N frames (file) or N seconds (RTSP)
        if is_file:
            if frame_idx % frame_skip != 0:
                continue
        else:
            now = time.time()
            if now - last_save_t < args.interval:
                continue
            last_save_t = now

        # ── Dedup check ──────────────────────────────────────────────────────
        if use_dedup:
            pil_img  = PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h        = imagehash_mod.phash(pil_img)
            is_dup   = any(abs(h - prev) <= args.dedup_threshold for prev in seen_hashes.values())
            if is_dup:
                skipped_dup += 1
                continue
            seen_hashes[saved] = h

        # ── Save ─────────────────────────────────────────────────────────────
        fname = out_dir / f"frame_{saved:05d}.jpg"
        cv2.imwrite(str(fname), frame)
        saved += 1

        if saved % 100 == 0:
            print(f"  Saved {saved} frames  (duplicates skipped: {skipped_dup})")

    cap.release()

    print()
    print(f"[DONE] Saved {saved} frames to {out_dir.resolve()}")
    if use_dedup:
        print(f"[DONE] Duplicates removed: {skipped_dup}")
    print(f"[DONE] Copy this number to Google Sheet column C: {saved}")


if __name__ == "__main__":
    main()
