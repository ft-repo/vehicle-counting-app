# Camera Viewpoint Audit — run7 class-confusion fix

Pipeline to discover how many distinct **camera viewpoints** the vehicle-counting
cameras span, and to produce a leak-free **train / validation split** for run7.

## Why this exists

run7 shows worse inter-class confusion — **car↔pickup, truck↔bus, taxi↔car** —
which got worse after camera angles changed. The company runs 1000+ cameras
across Thailand for several solutions; only the **counting (`นับรถ`)** subset
matters here. We can't check 1000+ cameras one by one, so this pipeline samples
a representative set automatically and trains for viewpoint robustness rather
than tuning to any one angle.

## Privacy (important)

- Everything runs **locally**. The only outbound traffic is to your own cameras
  (over Tailscale → `*.enixma.net`) plus a one-time public model-weights download.
- Camera stream URLs / IPs are read from a local file, are **never printed**, and
  **never leave the machine**. Cameras are referred to by index + short hash.
- Terminal output is aggregates only. The per-camera CSVs carry URLs and **stay
  local** — do not commit or share them.

## Inputs

- Source of truth: Google Sheet **"Ip Analytics 69"** (54 site tabs).
  - Download as **Microsoft Excel (.xlsx)** → exports all tabs in one file.
  - Schema per tab: `การทำงาน` = function/solution, `RTSP` = HLS `.m3u8` URL.
  - Counting cameras = rows where `การทำงาน` contains `นับรถ`.

## Environment

```bash
conda activate diabetes-ai     # has cv2, sklearn, pandas, openpyxl, torch, torchvision
```

## Steps (reproduce when the sheet updates)

### 1. Extract counting-camera URLs
From the .xlsx, pull rows where `การทำงาน` contains `นับรถ`, take the `RTSP`
column's `.m3u8` URL, dedup → `counting_cameras.txt` (local).
(Last run: **118** counting cameras across 35 site tabs.)

### 2. Discover viewpoints
```bash
python discover_angles.py --cameras counting_cameras.txt --workers 10 --timeout 45
```
- Grabs one frame per camera (Tailscale must be up), embeds each with
  MobileNetV3, clusters into viewpoint archetypes (auto-k by silhouette).
- Outputs to `angle_discovery/`:
  - `frames/` — one jpg per reachable camera
  - `viewpoint_reps/` — representative thumbnail per cluster
  - `viewpoint_assignments.csv` — camera → cluster map (LOCAL, has URLs)

**Last run:** 95/118 reachable (23 persistently offline). Clusters into 3 *loose*
groups — silhouette ~0.06, i.e. a **continuum, not discrete angles**:
| Bucket | Scene | Cameras |
|---|---|---|
| VP00 | rural 2-lane | 26 |
| VP01 | highway 2-lane | 42 |
| VP02 | **multilane urban (priority)** | 27 |

Key finding: cameras are **geometrically consistent** (elevated, along-road,
slight downtilt) → a model *can* generalize. The real variance is lane count +
vehicle scale/distance + scene clutter, worst in **VP02**, where the confused
classes co-occur at distance.

### 3. Score cameras by class diversity + traffic density
Viewpoint spread is the wrong selector for class confusion — we need cameras
whose frames actually contain the confused classes (car/pickup/truck/bus/taxi)
and rare ones (tuktuk/cone/trailer/van) in busy traffic, not empty roads.
```bash
# run during a busy window (Tailscale up); accumulates across sessions
python score_diversity.py --reset --rounds 3 --interval 120   # morning rush
python score_diversity.py --rounds 3 --interval 120           # evening rush, adds on
```
- Uses the **run6** detector (`eval/run6_best.onnx`, imgsz 416) over frames
  grabbed from each reachable camera.
- **Accumulates** into `diversity_tally.json` — a single snapshot undersamples
  traffic (a rush-hour road reads empty at 2pm), so run it across several busy
  windows; distinct-class presence + density grow toward the truth.
- Writes `diversity_scores.csv` (ranked, local).

### 4. Build the train / validation lists
```bash
python build_lists.py --train-size 60
```
Both lists are drawn from the *diverse* cameras (so val also contains the
confused classes); every 4th diverse camera (by rank) is reserved for val.
Camera-level split (no camera in both) → val measures generalization to
**unseen cameras**. Priority tiers (label top-down): **high** (≥2 confused, or
confused+rare) → **medium** (≥1 confused or rare) → **low** (nothing seen yet —
quiet-snapshot, resolves as you run more scoring sessions).

Outputs: `labeling_list.csv` (TRAIN) + `validation_holdout.csv` (VAL).

## Intern handoff

1. For each camera in `labeling_list.csv`, pull training frames
   (`tools/frame_extractor.py` takes the stream URL).
2. **Weight the confused pairs** — choose frames with trucks/buses/pickups/taxis
   present, not empty road. Prioritize `label_priority = high` (VP02).
3. Auto-label → **intern review every label** → merge into the run7 train split.
4. Cameras in `validation_holdout.csv` get labeled too, but their frames are
   **never trained on** — they are the unseen-camera validation set.
5. Track progress in the `labeled` / `frames_pulled` / `notes` columns.

## run7 payoff check

Train on the 56, validate on the 39 unseen cameras, and watch the
**car↔pickup / truck↔bus / taxi↔car off-diagonals on the VP02 val subset** —
that number tells you whether run7 actually fixed the confusion.

## Files

| File | Contents | Share? |
|---|---|---|
| `discover_angles.py` | viewpoint-discovery pipeline | ✅ code only |
| `score_diversity.py` | rank cameras by class diversity + density (run6); accumulates across busy windows | ✅ code only |
| `build_lists.py` | turn the ranking into the train/val split | ✅ code only |
| `cluster_cameras.py` | metadata-based clustering (unused — sheet has no viewpoint columns) | ✅ code only |
| `counting_cameras.txt` | 118 counting-camera stream URLs | ❌ local only |
| `angle_discovery/` | frames, reps, assignments | ❌ local only |
| `diversity_tally.json` / `diversity_scores.csv` | cumulative detections + ranking (with URLs) | ❌ local only |
| `labeling_list.csv` / `validation_holdout.csv` | train/val camera lists (with URLs) | ❌ local only |
| `labeling_manifest.csv` / `validation_manifest.csv` | redacted twins — ranking/priority/class-counts, NO urls or ids | ✅ safe to commit |

## Notes

- 23 counting cameras were persistently unreachable across retries (~19%) —
  likely decommissioned / offline / stale URLs. Worth flagging to whoever
  maintains the sheet, but doesn't block run7.
- Re-running on an updated sheet only requires repeating steps 1–4.
