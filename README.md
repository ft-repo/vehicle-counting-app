# Vehicle Detection & Counting System

Real-time vehicle detection, tracking, and lane-crossing counting for CCTV and IP camera feeds. Primary model is **YOLO26n** (NMS-free, 43 % faster on CPU than v8n, edge-optimised for Axis ACAP cameras); YOLOv4-tiny (Darknet), YOLOv8n, and YOLO11n (ONNX) are kept as legacy / bench presets. Runs on **macOS, Windows, and Linux**.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training Guide](#training-guide)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Auto-Capture](#auto-capture)
- [Dashboard](#dashboard)
- [Validation](#validation)
- [Model Hot-Swap](#model-hot-swap)
- [File Structure](#file-structure)
- [Outputs](#outputs)
- [Supported Classes](#supported-classes)

---

## Features

- Multi-model support — **YOLO26n** (primary, NMS-free), YOLOv4-tiny (Darknet), YOLOv8n, YOLO11n via OpenCV DNN
- Centroid IoU tracker with configurable lost-frame timeout
- Per-lane directional counting (in / out) using cross-product line test
- Polygon ROI masking to restrict detection to the road area
- Day/Night model auto-switching by brightness and time of day
- Live model hot-swap without restarting the application
- Multi-page terminal dashboard with FPS, counts, mAP, confusion matrix, and dataset progress
- On-screen realtime-performance readout — processed vs source frame rate with a REALTIME OK / BEHIND indicator
- Real-time playback pacing — file and HLS sources play at the camera's true frame rate instead of fast-forwarding (`--no-pace` to run unthrottled for offline export or benchmarking)
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
- onnxruntime 1.20+ *(required for YOLO26n inference; YOLOv8n / YOLO11n run via OpenCV DNN)*
  - Mac / CPU-only Linux: `pip install onnxruntime`
  - Windows / Linux with NVIDIA GPU: `pip install onnxruntime-gpu` *(do not install both — they conflict)*

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

## Training Guide

A step-by-step beginner guide (in Thai, for Windows) covering the full data-to-training
workflow — from opening a terminal through frame extraction, labeling, and starting a
training run: **[`docs/training-guide-th.pdf`](docs/training-guide-th.pdf)**.

To regenerate the PDF after editing the source (`docs/training-guide-th.md`):

```bash
pip install markdown weasyprint
python docs/build_guide_pdf.py
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
| `--onnx` | — | YOLO26 / YOLOv8 / YOLO11 `.onnx` model file |
| `--cfg` | — | YOLOv4 `.cfg` file |
| `--weights` | — | YOLOv4 `.weights` file |
| `--names` | — | Class names `.names` file |
| `--size` | `416` | YOLO input resolution |
| `--conf` | `0.35` | Detection confidence threshold |
| `--nms` | `0.40` | NMS threshold |
| `--csv` | `vehicle_counts.csv` | CSV output path |
| `--stats` | `logs/live_stats.json` | Live stats JSON output path |
| `--out` | — | Save annotated video to file |
| `--skip` | `1` | Process 1 out of every N frames (e.g. `--skip 2` halves CPU load) |
| `--nowin` | off | Headless mode — no display window |
| `--gpu` | off | Force CUDA GPU backend |
| `--cpu` | off | Force CPU backend |

### Examples

```bash
# YOLO26n (primary) with scene config
python vehicle_counter.py "rtsp://user:pass@192.168.1.100/axis-media/media.amp" \
    --onnx   model_compare/yolo26n/run4/weights/best.onnx \
    --names  models/traffic14.names \
    --config config/scene_config.json

# YOLOv4-tiny (Darknet, fallback)
python vehicle_counter.py "rtsp://..." \
    --cfg     models/yolov4-tiny-drr7-day.cfg \
    --weights models/yolov4-tiny-drr7-day.weights \
    --names   models/coco_drr7.names \
    --config  config/scene_config.json

# Local video file
python vehicle_counter.py recording.mp4 \
    --onnx  model_compare/yolo26n/run4/weights/best.onnx \
    --names models/traffic14.names

# Headless (server / edge deployment)
python vehicle_counter.py "rtsp://..." \
    --onnx model_compare/yolo26n/run4/weights/best.onnx --names models/traffic14.names --nowin
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
    "onnx":       "model_compare/yolo26n/run4/weights/best.onnx",
    "names":      "models/traffic14.names",
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
    { "label": "YOLO26n run4", "onnx": "model_compare/yolo26n/run4/weights/best.onnx",
      "names": "models/traffic14.names", "input_size": 416 },
    { "label": "YOLOv8n run5 (legacy bench)", "onnx": "model_compare/yolov8n/run5/weights/best.onnx",
      "names": "models/coco_drr7.names",  "input_size": 416 },
    { "label": "YOLO11n run2 (legacy bench)", "onnx": "model_compare/yolo11n/run2/weights/best.onnx",
      "names": "models/coco_drr7.names",  "input_size": 416 }
  ]
}
```

`arrow_in` — direction counted as **IN**: `top_to_bottom` · `bottom_to_top` · `left_to_right` · `right_to_left`

> **Which model is deployed** is tracked in one place: `models/model_registry.json`
> (the `deployed` block + hot-swap `presets`). Tools (`run_val.py`, `auto_label.py`,
> `backfill_per_class.py`) resolve their default model from it, and the `model_presets`
> above mirror its `presets`. When you promote a new run, update the registry.

---

## Auto-Capture

Captures frames automatically when the live model is uncertain, drops blurry frames, and (optionally) imports the survivors into Label Studio for review. Imported frames arrive pre-annotated with the model's boxes, so reviewers correct rather than draw from scratch.

Configured under the `auto_capture` section of `config/scene_config.json`:

```json
"auto_capture": {
  "enabled": true,
  "hot_dir": "~/auto_capture",
  "confidence_threshold": 0.6,
  "cooldown_s": 5.0,
  "max_per_hour": 200,
  "jpeg_quality": 90,
  "lapvar_threshold": 100,
  "glitch_min_std": 8.0,
  "retention_days": 7,
  "mature_classes": ["car", "bike", "truck", "bus", "taxi", "pickup"],
  "label_studio": {
    "enabled": false,
    "url": "https://label-studio.example.com",
    "project_id": 2,
    "token_env": "LS_API_TOKEN"
  }
}
```

| Key | Description |
|---|---|
| `confidence_threshold` | Save a frame if any detection's confidence falls below this value |
| `lapvar_threshold` | Laplacian-variance gate; frames below this are dropped as too blurry (default `100`) |
| `glitch_min_std` | Minimum global pixel std-dev; frames below this are dropped as solid-color / glitched / blackout (default `8.0`). Catches RTSP/HLS decode failures, codec corruption, and blackout frames that can slip past LapVar |
| `retention_days` | Auto-purge `<YYYY-MM-DD>/` subdirectories older than this (default `7`). Cleanup runs once at startup and once per calendar day on first save. Set to `0` to disable. Only directories matching the strict `YYYY-MM-DD` pattern are touched — nothing else in the hot folder is at risk |
| `mature_classes` | Only consider detections from these classes when scoring uncertainty |
| `cooldown_s` / `max_per_hour` | Rate limits to keep the hot folder bounded |
| `label_studio.enabled` | Set to `true` to POST every surviving frame into Label Studio |
| `label_studio.url` / `project_id` | Target LS server and project ID |
| `label_studio.token_env` | Environment variable holding the LS API token (never commit the token to JSON) |

Set the API token before launch:

```bash
export LS_API_TOKEN="<your-label-studio-token>"
python run_camera.py
```

If `label_studio.enabled` is `true` but any of `url`, `project_id`, or the token are missing, LS push is disabled at startup with a warning — capture and disk save continue unaffected. LS push failures during runtime are logged and never block capture.

Each saved frame writes a sidecar JSON next to the JPG with `lapvar`, `min_confidence`, and per-detection rows for downstream audit.

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
# Use the deployed model (resolved from models/model_registry.json)
python run_val.py

# Specify a model
python run_val.py --model model_compare/yolo26n/run4/weights/best.pt \
                  --data  new_data/dataset/data.yaml \
                  --imgsz 416
```

Results are written to `val_results.json` and loaded automatically by the dashboard.

### Counting Accuracy

Validation mAP measures detection quality; the metric that matters in the field is **counting accuracy**. Measure it directly against a clip with hand-counted crossings:

```bash
python tools/counting_eval.py --clip path/to/clip.mp4 --gt path/to/ground_truth.json
```

The tool runs the configured model over the clip and reports counting error (`|counted − truth| / truth`) per class, per lane, and overall. See `tools/counting_gt.example.json` for the ground-truth format. Results are written to `counting_eval_results.json`.

---

## Model Hot-Swap

Switch the active model without restarting the application.

Press `1`, `2`, or `3` in the video window, or from a separate terminal:

```bash
python switch_model.py 1   # YOLO26n run4   (primary, 14-class)
python switch_model.py 2   # YOLOv8n run5   (legacy / bench)
python switch_model.py 3   # YOLO11n run2   (legacy / bench)
```

Model presets are defined in `config/scene_config.json` under `model_presets` (and
mirror `models/model_registry.json → presets`). `switch_model.py` prints the live
list — run it with no argument. The hot-swap IPC file is `logs/model_cmd.txt`.

---

## File Structure

```
vehicle-counting-app/
├── run_camera.py              # Cross-platform launcher
├── run_camera.sh / .bat       # macOS / Linux / Windows shortcuts
├── vehicle_counter.py         # Core application
├── live_stats.py              # Terminal dashboard
├── switch_model.py            # Runtime model switcher (reads model_presets)
├── run_val.py                 # Validation runner (default model from the registry)
├── requirements.txt
│
├── config/
│   ├── scene_config.json      # Camera, model, ROI, lanes, model_presets
│   └── pipeline_config.yaml   # GDINO + SAM2 + class_prompts + level_gate (labeling)
│
├── models/
│   ├── model_registry.json    # SINGLE source of truth for the deployed model
│   ├── traffic14.names        # canonical 14-class schema
│   ├── traffic12.names        # prior 12-class (reference)
│   ├── coco_drr7.names        # legacy schema (bench presets)
│   ├── yolov4-tiny-drr7-day/night.cfg / .weights
│   └── yolov8n.onnx / yolo11n.onnx   # legacy / bench
│
├── model_compare/             # Training runs + dataset tooling
│   ├── yolo26n/run4/weights/best.{pt,onnx}  # DEPLOYED (14-class) — see registry
│   ├── yolov8n/run5, yolo11n/run2           # legacy / bench (onnx)
│   ├── registry.py            # resolves the deployed model from model_registry.json
│   ├── build_split.py         # canonical train/val/test split (flat LS export aware)
│   ├── auto_label.py          # auto-label images (class names read from the model)
│   └── tracker.py / tracker_live.py / compare.py / dashboard.py
│
├── tools/
│   ├── frame_extractor.py     # extract frames from RTSP / video
│   ├── gdino_ls_backend.py    # Grounding DINO — Label Studio ML backend
│   ├── export_approved.py     # export reviewer-approved labels from LS
│   ├── remap_ls_export.py     # remap LS export class ids → canonical by NAME
│   ├── backfill_per_class.py  # per-class mAP → val_results.json (level gate)
│   ├── counting_eval.py       # counting-accuracy harness
│   ├── rename_dataset.py      # normalise corpus filenames (classes from traffic14)
│   ├── active_learning.py     # low-confidence frame capture (prod)
│   └── auto_save.sh           # DGX-side rsync of active-learning frames
│
├── archive/                   # RETIRED (do not run): merge_dataset.py, export_dataset.py
├── tests/                     # test_build_split_smoke.py (offline)
├── docs/                      # training-guide-th.md, pipeline.md
├── setup/                     # install.sh / .bat, check_gpu.py
│
└── logs/                      # Auto-created at runtime
    ├── vehicle_counts.csv
    ├── live_stats.json
    └── model_cmd.txt          # transient model hot-swap IPC
```

---

## Outputs

| File | Description |
|---|---|
| `logs/vehicle_counts.csv` | Per-crossing log: `timestamp, track_id, class, direction, lane` |
| `logs/live_stats.json` | Real-time stats snapshot updated every ~1 s |
| `val_results.json` | Validation metrics: mAP50, precision, recall, F1, confusion matrix |
| `*(optional)*` annotated video | Use `--out output.mp4` to save an annotated recording |

---

## Supported Classes

The active class schema is set in `config/scene_config.json` under `yolo.names`, and the deployed model is tracked in `models/model_registry.json`. The default ships on the **14-class schema** (`models/traffic14.names`); the deployed weights (YOLO26n run4) run on it. `van` (id 10) is index-kept but its boxes are dropped from the current training split. The legacy YOLOv8n / YOLO11n bench presets remain on the older `models/coco_drr7.names` schema.

### 14-class schema — primary (`models/traffic14.names`)

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | car | 7 | person |
| 1 | bike | 8 | cone |
| 2 | truck | 9 | tuktuk |
| 3 | bus | 10 | van |
| 4 | taxi | 11 | agri_truck |
| 5 | pickup | 12 | agri_vehicle |
| 6 | trailer | 13 | ambulance |

### 12-class schema — prior (`models/traffic12.names`)

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | car | 6 | trailer |
| 1 | bike | 7 | person |
| 2 | truck | 8 | cone |
| 3 | bus | 9 | tuktuk |
| 4 | taxi | 10 | van |
| 5 | pickup | 11 | agriculture |

### 11-class schema — legacy (`models/coco_drr7.names`)

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | person | 5 | taxi |
| 1 | car | 6 | pickup |
| 2 | bike | 7 | trailer |
| 3 | truck | 8 | tuktuk |
| 4 | bus | 9 | agri_truck |
| | | 10 | van |

> Agricultural-vehicle classes are excluded from counting and on-screen display: `agri_truck` and `agri_vehicle` (14-class), `agriculture` (12-class), `agri_truck` (11-class). `ambulance` is counted and displayed.

---

## License

Internal use only. Not for public distribution.
