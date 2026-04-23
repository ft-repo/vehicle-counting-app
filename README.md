# Vehicle Counter

Real-time vehicle detection and counting system for CCTV/IP camera feeds. Built with OpenCV DNN and supports YOLOv4-tiny (Darknet) and YOLOv8/YOLO11 (ONNX) models.

---

## Features

- **Detection** — YOLOv4-tiny, YOLOv8n, YOLO11n via OpenCV DNN (CPU or CUDA GPU)
- **Tracking** — centroid-based IoU tracker with configurable lost-frame timeout
- **Lane crossing** — directional in/out counts per lane using cross-product side test
- **ROI masking** — polygon region of interest filters detections to road area only
- **Day/Night switching** — auto-detects brightness and loads the correct model variant
- **Live model swap** — press `1` / `2` / `3` or run `switch_model.py` to swap models without restart
- **Live dashboard** — `live_stats.py` renders a rich terminal UI fed by `live_stats.json`
- **CSV logging** — every crossing event is written to `vehicle_counts.csv`
- **Interactive editor** — draw/edit ROI and lanes in real time without touching the JSON

---

## Requirements

```
Python 3.9+
opencv-python      # cv2 with DNN module
numpy
rich               # for live_stats.py dashboard
ultralytics        # for run_val.py validation only
```

Install:

```bash
pip install opencv-python numpy rich ultralytics
```

For GPU (CUDA) acceleration, build OpenCV with CUDA support or install `opencv-contrib-python` and ensure CUDA toolkit + cuDNN are installed.

---

## File Structure

```
counting_app/
└── vehicle-counter/
    ├── vehicle_counter.py       # Main app — detection, tracking, counting
    ├── live_stats.py            # Terminal dashboard
    ├── switch_model.py          # Hot-swap model from another terminal
    ├── run_val.py               # Run validation on a dataset
    ├── run_camera.bat           # Windows launch script
    ├── config/
    │   └── scene_config.json    # Camera, model, ROI, and lane configuration
    ├── models/
    │   ├── coco_drr7.names
    │   ├── yolov4-tiny-drr7-day.cfg / .weights
    │   ├── yolov4-tiny-drr7-night.cfg / .weights
    │   ├── yolov8n.onnx
    │   └── yolo11n.onnx
    └── logs/
        ├── vehicle_counts.csv   # Crossing event log (auto-created)
        └── live_stats.json      # Live stats for dashboard (auto-created)
```

---

## Quick Start

### Run on RTSP camera (from scene_config.json)

```bash
cd vehicle-counter
python vehicle_counter.py "rtsp://user:pass@IP/axis-media/media.amp" \
    --onnx models/yolov8n.onnx \
    --names models/coco_drr7.names \
    --config config/scene_config.json
```

### Run on a local video file

```bash
python vehicle_counter.py my_video.mp4 \
    --onnx models/yolov8n.onnx \
    --names models/coco_drr7.names \
    --config config/scene_config.json
```

### Run with YOLOv4-tiny (Darknet)

```bash
python vehicle_counter.py "rtsp://..." \
    --cfg   models/yolov4-tiny-drr7-day.cfg \
    --weights models/yolov4-tiny-drr7-day.weights \
    --names models/coco_drr7.names \
    --config config/scene_config.json
```

### Headless mode (no window, server/edge deployment)

```bash
python vehicle_counter.py "rtsp://..." --onnx best.onnx --names coco_drr7.names --nowin
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `input` | *(required)* | Video file, RTSP URL, or webcam index (e.g. `0`) |
| `--cfg` | — | YOLOv4 `.cfg` file |
| `--weights` | — | YOLOv4 `.weights` file |
| `--names` | — | Class names `.names` file |
| `--onnx` | — | YOLOv8/YOLO11 ONNX model (replaces `--cfg`/`--weights`) |
| `--config` | — | `scene_config.json` path |
| `--size` | `416` | YOLO input image size |
| `--conf` | `0.35` | Detection confidence threshold |
| `--nms` | `0.40` | NMS threshold |
| `--csv` | `vehicle_counts.csv` | CSV output path |
| `--out` | — | Save annotated video to this path |
| `--stats` | `live_stats.json` | Live stats JSON output path |
| `--nowin` | off | Headless mode — no display window |
| `--gpu` | off | Force CUDA GPU (exits if unavailable) |
| `--cpu` | off | Force CPU backend |

---

## Keyboard Controls

Press keys in the OpenCV window (click the window first to focus it):

| Key | Action |
|---|---|
| `E` | Toggle Edit Mode |
| `R` | Draw new ROI (in Edit Mode) |
| `L` | Add lane |
| `D` | Delete selected lane (in Edit Mode) |
| `C` | Clear ROI (in Edit Mode) |
| `S` | Save `scene_config.json` |
| `Tab` | Cycle lane direction (both / in / out) |
| `1` / `2` / `3` | Switch model preset |
| Arrow keys | Nudge selected ROI point or lane endpoint (3 px) |
| `ESC` | Cancel current sub-mode |
| `Q` | Quit |

**Edit Mode workflow:**
1. Press `E` — enters Edit Mode (teal HUD appears top-right)
2. Click a ROI point or lane endpoint to select it
3. Drag to reposition, or use arrow keys to nudge
4. Press `S` to save

---

## scene_config.json

All camera, model, ROI, and lane settings live here. Edit it manually or use the interactive UI.

```json
{
  "version": "1.0",
  "camera": {
    "source": "rtsp://user:pass@IP/axis-media/media.amp",
    "width": 1280,
    "height": 960
  },
  "yolo": {
    "onnx":       "model_compare/yolov8n/run5/weights/best.onnx",
    "names":      "old_win_code/models/coco_drr7.names",
    "input_size": 416,
    "conf":       0.2,
    "nms":        0.4
  },
  "roi": {
    "enabled": true,
    "points": [[x1,y1], [x2,y2], ...]
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
    { "label": "YOLOv4-tiny", "cfg": "...", "weights": "...", "names": "...", "input_size": 416 },
    { "label": "YOLOv8n",     "onnx": "...", "names": "...", "input_size": 416 },
    { "label": "YOLO11n",     "onnx": "...", "names": "...", "input_size": 416 }
  ]
}
```

`arrow_in` sets what direction is counted as **in**. Options: `top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`.

---

## Live Dashboard

In a second terminal (while the main app is running):

```bash
cd vehicle-counter
python live_stats.py --stats live_stats.json --config config/scene_config.json
```

Displays live FPS, active tracks, per-class counts, lane counts, recent events, model accuracy, and dataset progress.

---

## Hot-Swap Model

Without stopping the main app:

```bash
python switch_model.py 1   # YOLOv4-tiny
python switch_model.py 2   # YOLOv8n
python switch_model.py 3   # YOLO11n
```

Or just press `1`, `2`, `3` in the OpenCV window.

---

## Run Validation

```bash
python run_val.py \
    --model model_compare/yolov8n/run5/weights/best.pt \
    --data  ../new_data/dataset/data.yaml \
    --imgsz 416
```

Saves `val_results.json` with per-class mAP50 scores.

---

## Outputs

| File | Description |
|---|---|
| `vehicle_counts.csv` | One row per lane crossing: `timestamp, track_id, class, direction, lane` |
| `live_stats.json` | Real-time JSON (updated every 1 s): FPS, counts, events, model info |
| *(optional)* annotated video | Pass `--out output.mp4` to save annotated frames |

---

## Supported Classes

Defined in `old_win_code/models/coco_drr7.names`:

```
0: person    1: car       2: bike      3: truck     4: bus
5: taxi      6: pickup    7: trailer   8: tuktuk    9: van
```

> `van` is excluded from counting by default (`SKIP_CLASSES`). Edit the constant in `vehicle_counter.py` to change this.

---

## Notes

- The RTSP stream uses TCP transport (`?tcp` or OpenCV flag) to reduce packet loss.
- Day/Night model auto-switch uses frame brightness with hysteresis (thresholds: 55 dark / 90 bright) and hard clock limits (09:00–16:00 always day, 00:00–05:00 always night). Only applies to Darknet models.
- Ported from the original Windows C++ app (`old_win_code/`). Windows shared-memory IPC to a C# UI is removed; CSV + JSON replace it.
