"""
Model Comparison Script
Compares YOLOv4-tiny, YOLOv8n, and YOLO11n on the same val images.
Completely standalone — does not use vehicle_counter.py at all.

Usage:
    python model_compare/compare.py
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
#  Paths — adjust if needed
# ─────────────────────────────────────────
BASE       = Path(__file__).parent.parent
VAL_IMAGES = BASE / "new_data/dataset/images/val"
NAMES_FILE = BASE / "counting_app/old_win_code/models/coco_drr7.names"

MODELS = {
    "YOLOv4-tiny": {
        "type":    "darknet",
        "cfg":     BASE / "counting_app/old_win_code/models/yolov4-tiny-drr7-day.cfg",
        "weights": BASE / "counting_app/old_win_code/models/yolov4-tiny-drr7-day.weights",
        "imgsz":   416,
        "results": None,
    },
    "YOLOv8n": {
        "type":    "onnx",
        "onnx":    BASE / "model_compare/yolov8n/run/weights/best.onnx",
        "imgsz":   416,
        "results": BASE / "model_compare/yolov8n/run/results.csv",
    },
    "YOLO11n": {
        "type":    "onnx",
        "onnx":    BASE / "model_compare/yolo11n/run/weights/best.onnx",
        "imgsz":   416,
        "results": BASE / "model_compare/yolo11n/run/results.csv",
    },
}

CONF_THRESH = 0.25
NMS_THRESH  = 0.40
WARMUP_RUNS = 3
BENCH_RUNS  = 10


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_model(cfg):
    if cfg["type"] == "darknet":
        if not Path(cfg["cfg"]).exists() or not Path(cfg["weights"]).exists():
            return None
        net = cv2.dnn.readNetFromDarknet(str(cfg["cfg"]), str(cfg["weights"]))
    else:
        if not Path(cfg["onnx"]).exists():
            return None
        net = cv2.dnn.readNetFromONNX(str(cfg["onnx"]))

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def infer_darknet(net, frame, imgsz, conf_thresh, nms_thresh, nc):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for row in out:
            scores = row[5:]
            cls_id = int(np.argmax(scores))
            conf   = float(row[4]) * float(scores[cls_id])
            if conf < conf_thresh:
                continue
            cx = int(row[0] * w); cy = int(row[1] * h)
            bw = int(row[2] * w); bh = int(row[3] * h)
            boxes.append([cx - bw//2, cy - bh//2, bw, bh])
            confs.append(conf)
            class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((class_ids[i], confs[i], boxes[i]))
    return result


def infer_yolov8(net, frame, imgsz, conf_thresh, nms_thresh):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    predictions = outs[0][0].T   # (8400, 4+nc)
    scale_x = w / imgsz
    scale_y = h / imgsz
    boxes, confs, class_ids = [], [], []

    for row in predictions:
        scores = row[4:]
        cls_id = int(np.argmax(scores))
        conf   = float(scores[cls_id])
        if conf < conf_thresh:
            continue
        cx = row[0] * scale_x; cy = row[1] * scale_y
        bw = row[2] * scale_x; bh = row[3] * scale_y
        boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
        confs.append(conf)
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((class_ids[i], confs[i], boxes[i]))
    return result


def measure_fps(net, model_type, imgsz, frames):
    """Run inference on all frames and return average FPS."""
    # Warmup
    for frame in frames[:WARMUP_RUNS]:
        if model_type == "darknet":
            infer_darknet(net, frame, imgsz, CONF_THRESH, NMS_THRESH, 10)
        else:
            infer_yolov8(net, frame, imgsz, CONF_THRESH, NMS_THRESH)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        for frame in frames:
            if model_type == "darknet":
                infer_darknet(net, frame, imgsz, CONF_THRESH, NMS_THRESH, 10)
            else:
                infer_yolov8(net, frame, imgsz, CONF_THRESH, NMS_THRESH)
    elapsed = time.perf_counter() - t0
    total = BENCH_RUNS * len(frames)
    return total / elapsed


def read_map(results_csv):
    """Read best mAP50 and mAP50-95 from Ultralytics results.csv."""
    if not results_csv or not Path(results_csv).exists():
        return None, None
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    # Strip whitespace from keys
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
    map50    = float(best.get("metrics/mAP50(B)", 0))
    map5095  = float(best.get("metrics/mAP50-95(B)", 0))
    return map50, map5095


def file_size_mb(path):
    if path and Path(path).exists():
        return Path(path).stat().st_size / 1_000_000
    return None


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    print("\n" + "="*58)
    print("  MODEL COMPARISON")
    print("="*58)

    # Load class names
    if not NAMES_FILE.exists():
        print(f"[ERROR] Names file not found: {NAMES_FILE}")
        sys.exit(1)
    names = load_names(NAMES_FILE)

    # Load val images
    img_paths = list(VAL_IMAGES.glob("*.png")) + list(VAL_IMAGES.glob("*.jpg"))
    if not img_paths:
        print(f"[ERROR] No images found in {VAL_IMAGES}")
        sys.exit(1)

    frames = []
    for p in img_paths:
        f = cv2.imread(str(p))
        if f is not None:
            frames.append(f)
    print(f"\nVal images loaded: {len(frames)}")

    # ── Run each model ──
    results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'─'*58}")
        print(f"  {name}")
        print(f"{'─'*58}")

        net = load_model(cfg)
        if net is None:
            print(f"  [SKIP] Model file not found — train first")
            results[name] = None
            continue

        # FPS
        print(f"  Measuring FPS ({BENCH_RUNS} runs × {len(frames)} images)...")
        fps = measure_fps(net, cfg["type"], cfg["imgsz"], frames)
        print(f"  FPS: {fps:.1f}")

        # mAP from training results
        map50, map5095 = read_map(cfg.get("results"))

        # File sizes
        if cfg["type"] == "darknet":
            model_size = file_size_mb(cfg["weights"])
            onnx_size  = None
            tflite_path = None
        else:
            model_size = file_size_mb(cfg["onnx"])
            onnx_size  = model_size
            tflite_path = str(cfg["onnx"]).replace(".onnx", "_int8.tflite")
            tflite_path = tflite_path if Path(tflite_path).exists() else None

        tflite_size = file_size_mb(tflite_path)

        results[name] = {
            "fps":        fps,
            "map50":      map50,
            "map5095":    map5095,
            "model_size": model_size,
            "onnx_size":  onnx_size,
            "tflite_size":tflite_size,
        }

    # ── Summary table ──
    print(f"\n{'='*58}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*58}")
    print(f"{'Model':<14} {'mAP50':>7} {'mAP50-95':>9} {'FPS':>6} {'Size':>8} {'TFLite':>8}")
    print(f"{'─'*58}")

    for name, r in results.items():
        if r is None:
            print(f"{name:<14} {'(not trained yet)':>40}")
            continue
        map50    = f"{r['map50']:.3f}"    if r['map50']    is not None else "N/A"
        map5095  = f"{r['map5095']:.3f}"  if r['map5095']  is not None else "N/A"
        fps      = f"{r['fps']:.1f}"
        size     = f"{r['model_size']:.1f}MB" if r['model_size'] else "N/A"
        tflite   = f"{r['tflite_size']:.1f}MB" if r['tflite_size'] else "N/A"
        print(f"{name:<14} {map50:>7} {map5095:>9} {fps:>6} {size:>8} {tflite:>8}")

    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
