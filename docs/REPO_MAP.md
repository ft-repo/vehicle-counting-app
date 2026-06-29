# VCA repo map — what lives where (canonical vs archive)

> Updated 2026-06-29. Authoritative pointer for "where is X". If a file isn't
> here, it's incidental. Canonical = the thing actually used; archive = kept
> for reference, not run.

## Canonical (run / depend on these)
| Path | Role |
|---|---|
| `vehicle_counter.py` | Core: detect (YOLO26n via cv2.dnn/onnxruntime) → centroid-IoU track → lane-cross count → CSV. |
| `run_camera.py` | Launcher — reads `config/scene_config.json`, spawns dashboard. |
| `live_stats.py` | Rich terminal dashboard. |
| `run_val.py` | Ultralytics validation → `val_results.json`. |
| `config/scene_config.json` | Single source of truth: source, model presets, ROI, lanes. |
| `tools/counting_eval.py` | Counting-error gate (shells the counter over a clip vs GT). |
| `tools/night_eval.py` | Day-vs-night per-class scorecard. |
| `camera_audit/score_diversity.py` | Rank cameras by class diversity (run7 selection). |
| `tools/vca_store.py` | DuckDB store for counts + eval history (M0). |
| `models/` | ONNX/cfg weights (`*.pt`/`*.weights` gitignored). |
| `eval/run6_best.onnx` | Current production model. |

## Archive (reference only — do NOT run)
| Path | Why kept |
|---|---|
| `archive/` | Superseded copies (`dashboard.py`, `compare.py`, old `merge_dataset.py`, `export_dataset.py`). |
| `old_win_code/` | Original C++ port source. |
| `model_compare/yolo26n/run1..run4/` | Earlier training runs; weights for comparison. |

## Data (gitignored — local only)
| Path | Role |
|---|---|
| `data/counts.duckdb` | Crossing + eval history store (M0). |
| `eval/*.mp4`,`*.jpg` | Heavy eval clips/frames (regenerable). |
| `runs/`,`new_data/`,`raw_frames/`,`auto_labels/` | Training data + runs. |
