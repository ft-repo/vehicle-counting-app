#!/usr/bin/env python3
"""
score_diversity.py — rank counting cameras by CLASS DIVERSITY + TRAFFIC DENSITY,
so labeling targets cameras that actually show the confused/rare classes in busy
traffic (car/pickup/truck/bus/taxi + tuktuk/cone/trailer/van), not empty roads.

WHY
    Fixing run7's car↔pickup / truck↔bus / taxi↔car confusion needs frames where
    those classes co-occur. A different *viewpoint* on an empty rural road adds
    no labeled instances. So we select cameras by what their frames CONTAIN,
    scored with the run6 detector.

TIME-WINDOWED SAMPLING (important)
    A single snapshot badly undersamples traffic — a rush-hour road reads empty
    at 2pm. So this script ACCUMULATES into a persistent tally (diversity_tally.json):
    run it during several busy windows and the counts add up. Distinct-class
    presence and density then reflect ALL sessions, not one instant.

    Recommended: run once each morning + evening rush (Tailscale up), e.g.
        python score_diversity.py --rounds 3 --interval 120     # ~6 min/session
    Repeat at the next busy window — scores keep improving. Use --reset to start
    a fresh tally.

PRIVACY
    Local only. URLs read from viewpoint_assignments.csv, never printed; cameras
    referred to by index/hash. Output (tally + scores csv) stays local.

USAGE
    conda activate diabetes-ai
    cd ~/Desktop/vehicle-counting-app/camera_audit
    python score_diversity.py                       # one session, accumulates
    python score_diversity.py --rounds 3 --interval 120
    python score_diversity.py --reset               # wipe tally and start over
"""
from __future__ import annotations
import argparse, json, os, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000|max_delay;5000000")
import numpy as np
import cv2

MODEL   = "../eval/run6_best.onnx"
NAMES   = "../models/traffic14.names"
ASSIGN  = "angle_discovery/viewpoint_assignments.csv"
TALLY   = "diversity_tally.json"
OUT     = "diversity_scores.csv"

CONF, IMGSZ    = 0.35, 416   # run6 onnx is exported at fixed 416x416
FRAMES_PER_RND = 6
CONFUSED = {"car", "pickup", "truck", "bus", "taxi"}
RARE     = {"tuktuk", "cone", "trailer", "van", "ambulance", "bike"}
SKIP_CLS = {"agri_truck", "agri_vehicle"}


def load_names():
    return [l.strip() for l in Path(NAMES).read_text().splitlines() if l.strip()]


def grab_multi(url, k=FRAMES_PER_RND):
    """Open the HLS stream once, return up to k frames spaced across the buffer."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    except Exception:
        pass
    frames = []
    if not cap.isOpened():
        cap.release(); return frames
    read = 0
    while len(frames) < k and read < k * 30:
        ok, f = cap.read()
        if not ok or f is None:
            break
        read += 1
        if read % 20 == 0:
            frames.append(f)
    cap.release()
    return frames


def load_tally():
    if Path(TALLY).exists():
        return json.loads(Path(TALLY).read_text())
    return {}


def save_tally(t):
    Path(TALLY).write_text(json.dumps(t, ensure_ascii=False, indent=0))


def session_stamp():
    return time.strftime("%Y-%m-%d %H:%M")


def score_from(class_counts: dict, frames_seen: int):
    present = {c for c, n in class_counts.items() if n > 0}
    n_conf = len(present & CONFUSED)
    n_rare = len(present & RARE)
    total = sum(class_counts.values())
    density = total / frames_seen if frames_seen else 0.0
    score = n_conf * 4 + n_rare * 2 + min(density, 20) * 0.5
    return round(score, 2), n_conf, n_rare, round(density, 2)


def write_scores(tally):
    out = []
    for idx, rec in tally.items():
        cc = rec["class_counts"]
        fs = rec.get("frames_seen", 0)
        score, n_conf, n_rare, density = score_from(cc, fs)
        out.append({
            "camera_index": idx, "camera_id": rec.get("camera_id", ""),
            "viewpoint": rec.get("viewpoint", ""), "score": score,
            "n_confused_classes": n_conf, "n_rare_classes": n_rare,
            "avg_objs_per_frame": density, "frames_seen": fs,
            "sessions": rec.get("sessions", 0),
            "classes": "|".join(f"{k}:{v}" for k, v in
                                 Counter(cc).most_common() if v > 0),
            "stream_url": rec.get("url", ""),
        })
    out.sort(key=lambda x: (-x["score"], -x["n_confused_classes"], -x["avg_objs_per_frame"]))
    cols = ["rank", "camera_index", "camera_id", "viewpoint", "score",
            "n_confused_classes", "n_rare_classes", "avg_objs_per_frame",
            "frames_seen", "sessions", "classes", "stream_url"]
    with open(OUT, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, o in enumerate(out, 1):
            o["rank"] = i; w.writerow(o)
    return out


def main():
    import csv
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=1,
                    help="revisits per camera this session (spreads over time)")
    ap.add_argument("--interval", type=int, default=0,
                    help="seconds to wait between rounds")
    ap.add_argument("--reset", action="store_true", help="wipe the tally first")
    args = ap.parse_args()

    if args.reset and Path(TALLY).exists():
        Path(TALLY).unlink(); print("tally reset.")

    from ultralytics import YOLO
    names = load_names()
    model = YOLO(MODEL, task="detect")
    mnames = (model.names if isinstance(model.names, dict) and len(model.names) == len(names)
              else dict(enumerate(names)))

    cams = list(csv.DictReader(open(ASSIGN)))
    tally = load_tally()
    stamp = session_stamp()
    prior_sessions = max((rec.get("sessions", 0) for rec in tally.values()), default=0)
    print(f"session {stamp} | {len(cams)} cameras | rounds={args.rounds} "
          f"interval={args.interval}s | prior sessions={prior_sessions}")

    for rnd in range(1, args.rounds + 1):
        print(f"\n--- round {rnd}/{args.rounds} ---")
        for n, r in enumerate(cams, 1):
            idx = r["camera_index"]; url = r["stream_url"]
            rec = tally.setdefault(idx, {
                "url": url, "camera_id": r["camera_id"],
                "viewpoint": r["viewpoint_cluster"],
                "class_counts": {}, "frames_seen": 0, "sessions": 0,
            })
            for fr in grab_multi(url):
                res = model.predict(fr, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
                cls = [int(c) for c in res.boxes.cls.tolist()] if res.boxes is not None else []
                for c in cls:
                    lab = mnames.get(c)
                    if lab and lab not in SKIP_CLS:
                        rec["class_counts"][lab] = rec["class_counts"].get(lab, 0) + 1
                rec["frames_seen"] += 1
            if n % 20 == 0 or n == len(cams):
                print(f"  round {rnd}: {n}/{len(cams)} cameras")
        save_tally(tally)        # checkpoint after each round
        if args.interval and rnd < args.rounds:
            print(f"  waiting {args.interval}s before next round ...")
            time.sleep(args.interval)

    # mark this session on every camera that got frames this run
    for rec in tally.values():
        rec["sessions"] = rec.get("sessions", 0) + 1
    save_tally(tally)

    out = write_scores(tally)
    full = [o for o in out if o["n_confused_classes"] >= 4]
    total_sessions = max((rec.get("sessions", 0) for rec in tally.values()), default=0)
    print(f"\n=== DIVERSITY RANKING (cumulative over {total_sessions} session(s)) -> {OUT} ===")
    print(f"  cameras with >=4 of 5 confused classes: {len(full)}")
    print(f"  cameras with any rare class: {sum(1 for o in out if o['n_rare_classes'])}")
    print(f"  top 10:")
    for o in out[:10]:
        print(f"    #{o['rank']:2d} cam{o['camera_index']} score={o['score']:5.1f} "
              f"conf={o['n_confused_classes']} rare={o['n_rare_classes']} "
              f"dens={o['avg_objs_per_frame']} (seen {o['frames_seen']}f)")
    print(f"\n  Run again during the next busy window to accumulate. "
          f"Then rebuild lists from {OUT}.")


if __name__ == "__main__":
    main()
