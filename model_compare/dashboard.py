"""
Model Comparison Dashboard
Reads training results and generates a visual comparison PNG.

Run AFTER all models are trained:
    python model_compare/dashboard.py
"""

import os
import sys
import csv
import time
import cv2
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed — saves to file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("[INFO] Installing matplotlib...")
    os.system(f"{sys.executable} -m pip install matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────
BASE       = Path(__file__).parent.parent
VAL_IMAGES = BASE / "new_data/dataset/images/val"
OUTPUT     = BASE / "model_compare/dashboard.png"

MODELS = {
    "YOLOv4-tiny": {
        "type":     "darknet",
        "cfg":      BASE / "counting_app/old_win_code/models/yolov4-tiny-drr7-day.cfg",
        "weights":  BASE / "counting_app/old_win_code/models/yolov4-tiny-drr7-day.weights",
        "imgsz":    416,
        "results":  None,
        "color":    "#E57373",
    },
    "YOLOv8n": {
        "type":     "onnx",
        "onnx":     BASE / "model_compare/yolov8n/run/weights/best.onnx",
        "imgsz":    416,
        "results":  BASE / "model_compare/yolov8n/run/results.csv",
        "color":    "#64B5F6",
    },
    "YOLO11n": {
        "type":     "onnx",
        "onnx":     BASE / "model_compare/yolo11n/run/weights/best.onnx",
        "imgsz":    416,
        "results":  BASE / "model_compare/yolo11n/run/results.csv",
        "color":    "#81C784",
    },
}

CONF_THRESH = 0.25
NMS_THRESH  = 0.40
BENCH_RUNS  = 5


# ─────────────────────────────────────────
#  Data collection
# ─────────────────────────────────────────
def load_model(cfg):
    if cfg["type"] == "darknet":
        if not Path(cfg["cfg"]).exists():
            return None
        net = cv2.dnn.readNetFromDarknet(str(cfg["cfg"]), str(cfg["weights"]))
    else:
        if not Path(cfg["onnx"]).exists():
            return None
        net = cv2.dnn.readNetFromONNX(str(cfg["onnx"]))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def measure_fps(net, model_type, imgsz, frames):
    dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
    blob  = cv2.dnn.blobFromImage(dummy, 1/255.0, (imgsz, imgsz), swapRB=True)
    net.setInput(blob)
    net.forward(net.getUnconnectedOutLayersNames())  # warmup

    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        for frame in frames:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (imgsz, imgsz), swapRB=True)
            net.setInput(blob)
            net.forward(net.getUnconnectedOutLayersNames())
    elapsed = time.perf_counter() - t0
    return (BENCH_RUNS * len(frames)) / elapsed


def read_map(results_csv):
    if not results_csv or not Path(results_csv).exists():
        return None, None
    with open(results_csv) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
    return float(best.get("metrics/mAP50(B)", 0)), float(best.get("metrics/mAP50-95(B)", 0))


def file_mb(path):
    p = Path(path) if path else None
    return round(p.stat().st_size / 1_000_000, 1) if p and p.exists() else None


def collect_data(frames):
    data = {}
    for name, cfg in MODELS.items():
        print(f"  [{name}]", end=" ", flush=True)
        net = load_model(cfg)

        fps = None
        if net and frames:
            fps = round(measure_fps(net, cfg["type"], cfg["imgsz"], frames), 1)
            print(f"FPS={fps}", end=" ", flush=True)
        else:
            print("model not found — skipping FPS", end=" ", flush=True)

        map50, map5095 = read_map(cfg.get("results"))
        if map50:
            print(f"mAP50={map50:.3f}", end=" ", flush=True)

        if cfg["type"] == "darknet":
            size  = file_mb(cfg["weights"])
            tflite = None
        else:
            size  = file_mb(cfg["onnx"])
            tp    = str(cfg["onnx"]).replace(".onnx", "_int8.tflite")
            tflite = file_mb(tp)

        print()
        data[name] = {
            "fps":       fps,
            "map50":     map50,
            "map5095":   map5095,
            "size_mb":   size,
            "tflite_mb": tflite,
            "color":     cfg["color"],
            "available": net is not None,
        }
    return data


# ─────────────────────────────────────────
#  Dashboard drawing
# ─────────────────────────────────────────
def bar_chart(ax, data, key, title, unit="", higher_better=True):
    names  = list(data.keys())
    values = [data[n][key] for n in names]
    colors = [data[n]["color"] for n in names]

    bars = []
    for i, (n, v, c) in enumerate(zip(names, values, colors)):
        if v is None:
            ax.bar(i, 0, color="#444444", alpha=0.3)
            ax.text(i, 0.01, "N/A", ha="center", va="bottom",
                    fontsize=9, color="#888888")
        else:
            b = ax.bar(i, v, color=c, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.text(i, v + (max(x for x in values if x) * 0.02),
                    f"{v}{unit}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="white")
            bars.append(b)

    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.set_facecolor("#1E1E1E")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="y", colors="#aaaaaa")

    arrow = "↑ higher better" if higher_better else "↓ lower better"
    ax.text(0.98, 0.98, arrow, transform=ax.transAxes,
            fontsize=7, color="#888888", ha="right", va="top")


def radar_chart(ax, data):
    metrics     = ["FPS", "mAP50", "mAP50-95", "Size (small=good)"]
    num_metrics = len(metrics)
    angles      = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles     += angles[:1]

    ax.set_facecolor("#1E1E1E")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8, color="white")
    ax.tick_params(colors="#555555")
    ax.spines["polar"].set_color("#555555")
    ax.yaxis.set_tick_params(labelcolor="#888888")

    # Normalize each metric to 0-1
    def norm(key, values, invert=False):
        vals = [v for v in values if v is not None]
        if not vals:
            return [0] * len(values)
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5 if v is not None else 0 for v in values]
        out = [(v - mn) / (mx - mn) if v is not None else 0 for v in values]
        return [1 - x for x in out] if invert else out

    names  = list(data.keys())
    fps_n   = norm("fps",     [data[n]["fps"]     for n in names])
    map50_n = norm("map50",   [data[n]["map50"]   for n in names])
    map95_n = norm("map5095", [data[n]["map5095"] for n in names])
    size_n  = norm("size_mb", [data[n]["size_mb"] for n in names], invert=True)

    for i, name in enumerate(names):
        vals = [fps_n[i], map50_n[i], map95_n[i], size_n[i]]
        vals += vals[:1]
        ax.plot(angles, vals, color=data[name]["color"], linewidth=2, linestyle="solid")
        ax.fill(angles, vals, color=data[name]["color"], alpha=0.15)

    ax.set_title("Overall Radar", fontsize=11, fontweight="bold",
                 color="white", pad=15)


def draw_summary_table(ax, data):
    ax.set_facecolor("#1E1E1E")
    ax.axis("off")

    headers = ["Model", "mAP50", "mAP50-95", "FPS", "ONNX", "TFLite"]
    rows = []
    for name, d in data.items():
        rows.append([
            name,
            f"{d['map50']:.3f}"    if d["map50"]    is not None else "N/A",
            f"{d['map5095']:.3f}"  if d["map5095"]  is not None else "N/A",
            f"{d['fps']:.1f}"      if d["fps"]       is not None else "N/A",
            f"{d['size_mb']} MB"   if d["size_mb"]   is not None else "N/A",
            f"{d['tflite_mb']} MB" if d["tflite_mb"] is not None else "N/A",
        ])

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#2A2A2A" if r == 0 else "#1E1E1E")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#555555")
        if r > 0:
            name = rows[r-1][0]
            if c == 0:
                cell.set_text_props(color=data[name]["color"], fontweight="bold")

    ax.set_title("Summary Table", fontsize=11, fontweight="bold",
                 color="white", pad=12)


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  GENERATING COMPARISON DASHBOARD")
    print("="*50)

    # Load val images
    img_paths = list(VAL_IMAGES.glob("*.png")) + list(VAL_IMAGES.glob("*.jpg"))
    frames    = [cv2.imread(str(p)) for p in img_paths]
    frames    = [f for f in frames if f is not None]
    print(f"\nVal images: {len(frames)}")
    print("\nCollecting data...")

    data = collect_data(frames)

    # ── Build dashboard ──
    fig = plt.figure(figsize=(16, 10), facecolor="#121212")
    fig.suptitle("Model Comparison Dashboard — Vehicle Detection",
                 fontsize=15, fontweight="bold", color="white", y=0.97)

    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.90, bottom=0.10, left=0.06, right=0.97)

    ax_fps    = fig.add_subplot(gs[0, 0])
    ax_map50  = fig.add_subplot(gs[0, 1])
    ax_map95  = fig.add_subplot(gs[0, 2])
    ax_radar  = fig.add_subplot(gs[0, 3], polar=True)
    ax_size   = fig.add_subplot(gs[1, 0])
    ax_tflite = fig.add_subplot(gs[1, 1])
    ax_table  = fig.add_subplot(gs[1, 2:])

    bar_chart(ax_fps,    data, "fps",       "FPS (Speed)",         unit=" fps", higher_better=True)
    bar_chart(ax_map50,  data, "map50",     "mAP50 (Accuracy)",    unit="",     higher_better=True)
    bar_chart(ax_map95,  data, "map5095",   "mAP50-95 (Accuracy)", unit="",     higher_better=True)
    bar_chart(ax_size,   data, "size_mb",   "Model Size",          unit=" MB",  higher_better=False)
    bar_chart(ax_tflite, data, "tflite_mb", "TFLite Size",         unit=" MB",  higher_better=False)

    radar_chart(ax_radar, data)
    draw_summary_table(ax_table, data)

    # Legend
    patches = [mpatches.Patch(color=cfg["color"], label=name)
               for name, cfg in MODELS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#1E1E1E", edgecolor="#555555",
               labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#121212")
    print(f"\n  Dashboard saved to:\n  {OUTPUT}")
    print("\n  Open it with: open model_compare/dashboard.png\n")


if __name__ == "__main__":
    main()
