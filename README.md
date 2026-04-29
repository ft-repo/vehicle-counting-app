# Vehicle Detection & Counting System

Real-time vehicle detection, tracking, and lane-crossing counting for CCTV and IP camera feeds. Supports YOLOv4-tiny (Darknet), YOLOv8n, and YOLO11n (ONNX) models on **macOS, Windows, and Linux**.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Validation](#validation)
- [Model Hot-Swap](#model-hot-swap)
- [File Structure](#file-structure)
- [Outputs](#outputs)
- [Supported Classes](#supported-classes)

---

## Features

- Multi-model support — YOLOv4-tiny (Darknet), YOLOv8n, YOLO11n via OpenCV DNN
- Centroid IoU tracker with configurable lost-frame timeout
- Per-lane directional counting (in / out) using cross-product line test
- Polygon ROI masking to restrict detection to the road area
- Day/Night model auto-switching by brightness and time of day
- Live model hot-swap without restarting the application
- Multi-page terminal dashboard with FPS, counts, mAP, confusion matrix, and dataset progress
- Per-crossing CSV event log
- Interactive ROI and lane editor directly in the video window
- Headless mode for edge/server deployment (`--nowin`)
- Auto-validation on startup when results are missing or stale

---

## Requirements

- Python 3.9+
- OpenCV 4.8+ — **must be `opencv-python` (not `opencv-python-headless`)**
- NumPy 1.24+
- Rich 13.0+
- Ultralytics 8.0+ *(validation only)*

> **Important:** The headless variant (`opencv-python-headless`) does not include GUI support and will crash on window creation. Always install `opencv-python`.
>
> If you see the error `The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support`, run:
> ```bash
> pip uninstall opencv-python-headless opencv-python -y
> pip install opencv-python
> ```

---

## Installation

### Windows

```bat
setup\install.bat
```

### macOS / Linux

```bash
bash setup/install.sh
```

### Manual

```bash
pip install -r requirements.txt
```

#### GPU (CUDA) — Windows / Linux

For CUDA-accelerated inference, install a CUDA-enabled OpenCV build:

```bash
# Via conda (recommended)
conda install -c conda-forge opencv cudatoolkit

# Or run the provided setup script which auto-detects your CUDA version
setup\install.bat      # Windows
bash setup/install.sh  # Linux
```

---

## Quick Start

```bash
# macOS / Linux
python run_camera.py

# Windows
python run_camera.py
# or double-click run_camera.bat
```

`run_camera.py` reads the camera source from `config/scene_config.json`, opens the terminal dashboard in a new window, and starts the counter.

```bash
# Override source or backend at launch
python run_camera.py --source rtsp://user:pass@192.168.1.100/axis-media/media.amp
python run_camera.py --gpu
python run_camera.py --cpu
python run_camera.py --nowin   # headless
```

---

## Usage

`vehicle_counter.py` can also be run directly for more control:

```bash
python vehicle_counter.py <source> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `source` | *(required)* | RTSP URL, HLS URL, video file path, or webcam index |
| `--config` | — | Path to `scene_config.json` |
| `--onnx` | — | YOLOv8 / YOLO11 `.onnx` model file |
| `--cfg` | — | YOLOv4 `.cfg` file |
| `--weights` | — | YOLOv4 `.weights` file |
| `--names` | — | Class names `.names` file |
| `--size` | `416` | YOLO input resolution |
| `--conf` | `0.35` | Detection confidence threshold |
| `--nms` | `0.40` | NMS threshold |
| `--csv` | `vehicle_counts.csv` | CSV output path |
| `--stats` | `live_stats.json` | Live stats JSON output path |
| `--out` | — | Save annotated video to file |
| `--nowin` | off | Headless mode — no display window |
| `--gpu` | off | Force CUDA GPU backend |
| `--cpu` | off | Force CPU backend |

### Examples

```bash
# ONNX model with scene config
python vehicle_counter.py "rtsp://user:pass@192.168.1.100/axis-media/media.amp" \
    --onnx   models/yolov8n.onnx \
    --names  models/coco_drr7.names \
    --config config/scene_config.json

# YOLOv4-tiny (Darknet)
python vehicle_counter.py "rtsp://..." \
    --cfg     models/yolov4-tiny-drr7-day.cfg \
    --weights models/yolov4-tiny-drr7-day.weights \
    --names   models/coco_drr7.names \
    --config  config/scene_config.json

# Local video file
python vehicle_counter.py recording.mp4 \
    --onnx  models/yolov8n.onnx \
    --names models/coco_drr7.names

# Headless (server / edge deployment)
python vehicle_counter.py "rtsp://..." \
    --onnx models/yolov8n.onnx --names models/coco_drr7.names --nowin
```

### Keyboard Controls

Click the video window to focus it before using shortcuts.

| Key | Action |
|---|---|
| `E` | Toggle Edit Mode |
| `R` | Draw new ROI polygon *(Edit Mode)* |
| `L` | Add a new lane |
| `D` | Delete selected lane *(Edit Mode)* |
| `C` | Clear ROI *(Edit Mode)* |
| `S` | Save `scene_config.json` |
| `Tab` | Cycle lane direction |
| `I` | Change camera source |
| `1` / `2` / `3` | Switch model preset |
| Arrow keys | Nudge selected ROI point or lane endpoint (3 px) |
| `ESC` | Cancel current sub-mode |
| `Q` | Quit |

**Edit Mode workflow:**
1. Press `E` — teal HUD appears top-right
2. Click a ROI vertex or lane endpoint to select it
3. Drag to reposition, or use arrow keys to nudge
4. Right-click a point to delete it
5. Press `S` to save

---

## Configuration

All camera, model, ROI, and lane settings are stored in `config/scene_config.json`.
Edit manually or use the interactive UI in the video window.

```json
{
  "camera": {
    "source": "https://proxy/live/10.0.0.1.stream/playlist.m3u8",
    "width": 1280,
    "height": 960
  },
  "yolo": {
    "onnx":       "models/yolov8n.onnx",
    "names":      "models/coco_drr7.names",
    "input_size": 416,
    "conf": 0.2,
    "nms":  0.4
  },
  "roi": {
    "enabled": true,
    "points": [[x1, y1], [x2, y2], "..."]
  },
  "lanes": [
    {
      "id": 1,
      "name": "Lane 1",
      "enabled": true,
      "color": "#FF8800",
      "line": { "x1": 100, "y1": 500, "x2": 900, "y2": 500 },
      "arrow_in": "top_to_bottom",
      "count_in": true,
      "count_out": true
    }
  ],
  "model_presets": [
    { "label": "YOLOv4-tiny", "cfg": "models/yolov4-tiny-drr7-day.cfg",
      "weights": "models/yolov4-tiny-drr7-day.weights", "names": "models/coco_drr7.names" },
    { "label": "YOLOv8n run5", "onnx": "model_compare/yolov8n/run5/weights/best.onnx",
      "names": "models/coco_drr7.names" },
    { "label": "YOLO11n run2", "onnx": "model_compare/yolo11n/run2/weights/best.onnx",
      "names": "models/coco_drr7.names" }
  ]
}
```

`arrow_in` — direction counted as **IN**: `top_to_bottom` · `bottom_to_top` · `left_to_right` · `right_to_left`

---

## Dashboard

The terminal dashboard launches automatically via `run_camera.py`.
To open it manually in a separate terminal:

```bash
python live_stats.py
```

Navigate pages with number keys:

| Page | Content |
|---|---|
| `1` | Live feed status, FPS sparkline, active tracks, per-class counts, recent crossing events, dataset progress |
| `2` | Per-class metrics (AP50, AP50-95, Precision, Recall, F1), confusion matrix, error analysis |
| `3` | PR curves, IoU distribution, confidence threshold analysis, inference speed benchmarks |
| `4` | Full dataset progress with labeled counts and accuracy targets |

---

## Validation

```bash
# Use default model (configured in run_val.py)
python run_val.py

# Specify a model
python run_val.py --model model_compare/yolov8n/run5/weights/best.pt \
                  --data  new_data/dataset/data.yaml \
                  --imgsz 416
```

Results are written to `val_results.json` and loaded automatically by the dashboard.

---

## Model Hot-Swap

Switch the active model without restarting the application.

Press `1`, `2`, or `3` in the video window, or from a separate terminal:

```bash
python switch_model.py 1   # YOLOv4-tiny
python switch_model.py 2   # YOLOv8n run5
python switch_model.py 3   # YOLO11n run2
```

Model presets are defined in `config/scene_config.json` under `model_presets`.

---

## File Structure

```
vehicle-counting-app/
├── run_camera.py              # Cross-platform launcher
├── run_camera.sh              # macOS / Linux shortcut
├── run_camera.bat             # Windows shortcut
├── vehicle_counter.py         # Core application
├── live_stats.py              # Terminal dashboard
├── switch_model.py            # Runtime model switcher
├── run_val.py                 # Validation runner
├── requirements.txt
│
├── config/
│   └── scene_config.json      # Camera, model, ROI, and lane configuration
│
├── models/                    # Model weight files
│   ├── coco_drr7.names
│   ├── yolov4-tiny-drr7-day.cfg / .weights
│   ├── yolov4-tiny-drr7-night.cfg / .weights
│   ├── yolov8n.onnx
│   └── yolo11n.onnx
│
├── setup/
│   ├── install.bat            # Windows first-time setup
│   ├── install.sh             # macOS / Linux first-time setup
│   ├── setup_gpu.bat          # Windows CUDA setup
│   └── check_gpu.py           # CUDA availability check
│
├── tools/
│   ├── frame_extractor.py     # Extract frames from RTSP or video file
│   └── gdino_ls_backend.py    # Grounding DINO — Label Studio ML backend
│
├── model_compare/
│   ├── tracker.py             # Single-model training progress tracker
│   ├── tracker_live.py        # Side-by-side training dashboard
│   ├── auto_label.py          # Auto-label images with YOLO
│   ├── merge_dataset.py       # Merge external dataset into train/val split
│   └── eval_dashboard.py      # Generate evaluation comparison PNG
│
├── run_this/
│   ├── extract.sh             # Pre-configured frame extractor (macOS / Linux)
│   └── extract.bat            # Pre-configured frame extractor (Windows)
│
└── logs/                      # Auto-created at runtime
    ├── vehicle_counts.csv
    └── live_stats.json
```

---

## Outputs

| File | Description |
|---|---|
| `vehicle_counts.csv` | Per-crossing log: `timestamp, track_id, class, direction, lane` |
| `live_stats.json` | Real-time stats snapshot updated every ~1 s |
| `val_results.json` | Validation metrics: mAP50, precision, recall, F1, confusion matrix |
| `*(optional)*` annotated video | Use `--out output.mp4` to save an annotated recording |

---

## Supported Classes

Defined in `models/coco_drr7.names`:

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | person | 5 | taxi |
| 1 | car | 6 | pickup |
| 2 | bike | 7 | trailer |
| 3 | truck | 8 | tuktuk |
| 4 | bus | 9 | agri_truck |

---

## License

Internal use only. Not for public distribution.
