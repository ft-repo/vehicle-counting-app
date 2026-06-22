# Data & Model Pipeline (English)

The full loop, from camera to a redeployed model. The Thai intern guide
(`docs/training-guide-th.md`) covers the labelling steps in depth; this is the
engineer's map.

> **Canonical locations.** The training **corpus and all training run on the DGX**
> (`admin@edgexpert-14e2`, `/home/admin/vehicle-counting-app`, corpus at
> `/home/admin/vehicle_dataset/corpus`). The Mac is the dev / live-inference box and
> orchestrates over SSH. The SSD (`/Volumes/Puen_SSD`) is a **backup only**.
> "Which model is deployed" lives in **one** file: `models/model_registry.json`.

```
        ┌──────────────── LIVE OPS (prod box) ────────────────┐
 RTSP → │ run_camera.py → vehicle_counter.py → counts → logs/  │
        │            ↘ low-confidence frames captured          │
        └───────────────────────┬─────────────────────────────┘
                                 │ tools/auto_save.sh (cron, → DGX)
                                 ▼
 RETRAIN LOOP (on the DGX):  frames → Label Studio (GDINO auto-box)
        → human review → export_approved → remap_ls_export → build_split.py
        → train → run_val → export ONNX → update model_registry.json → redeploy ↺
```

## Stages

| # | Stage | Command (run on the DGX unless noted) |
|---|-------|----------------------------------------|
| 0 | Extract frames | `python tools/frame_extractor.py --source <rtsp\|mp4> --output raw_frames/ --interval 3` |
| 1 | Auto-label (LS ML backend) | `label-studio-ml start tools/gdino_ls_backend.py --port 9090` — reads `config/pipeline_config.yaml`; GDINO boxes appear for the intern to review |
| 2 | Export approved labels | `python tools/export_approved.py --url <ls-url> --project 2 --out approved_export` |
| 2.5 | Canonicalise class order | `python tools/remap_ls_export.py approved_export` (remaps by NAME → `models/traffic14.names`) |
| 3 | Build balanced split | `python model_compare/build_split.py` (defaults to `/home/admin/vehicle_dataset/corpus`, flat LS export, 70/20/10, `van` dropped) |
| 4 | Train | `yolo detect train data=new_data/dataset/data.yaml model=yolo26n.pt imgsz=416` (batch=8, workers=8 on the GB10) |
| 4.5 | Backfill per-class metrics | `python tools/backfill_per_class.py --data <data.yaml> --out runs/yolo26n/<run>/val_results.json` |
| 5 | Validate | `python run_val.py` (no `--model` → uses the registry's deployed model) |
| 5.5 | Counting accuracy | `python tools/counting_eval.py --clip <mp4> --gt <gt.json>` |
| 6 | Export ONNX | `yolo export model=runs/yolo26n/<run>/weights/best.pt format=onnx imgsz=416` |
| 7 | Promote | edit `models/model_registry.json` → point `deployed` at the new run; mirror the `.onnx` to the Mac |

## Run live (Mac / prod)

```bash
python run_camera.py                 # reads config/scene_config.json, opens dashboard
```
- Outputs land in `logs/` (`live_stats.json`, `vehicle_counts.csv`).
- Hot-swap models in the video window with keys `1`/`2`/`3`, or from another
  terminal: `python switch_model.py 2`. Presets come from
  `config/scene_config.json → model_presets` (the IPC file is `logs/model_cmd.txt`).

## Offline smoke test (no SSD/DGX needed)

```bash
python tests/test_build_split_smoke.py
```
Generates a tiny synthetic flat corpus, runs `build_split.py`, asserts a 14-class
split is produced.

## Notes

- **Class schema:** `models/traffic14.names` is canonical (14 classes; `van` is
  index-kept but boxes dropped this cycle). The legacy 11-class person-first order is
  retired — `build_split.py` remaps stale corpora by NAME and hard-fails on unknowns.
- **SAM2** is disabled (`config/pipeline_config.yaml`); see the in-file note for the
  re-enable condition.
