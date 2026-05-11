"""
Grounding DINO — Label Studio ML Backend
=========================================
Runs GDINO once at startup. Label Studio calls it every time an intern
opens an image — boxes appear automatically for the intern to review.

Level 1 rule enforced: ALL detections returned as suggestions.
No auto-accept. Human reviews every box.

HOW TO START:
    cd /path/to/vehicle-counting-app
    label-studio-ml start tools/gdino_ls_backend.py --port 9090

HOW TO CONNECT IN LABEL STUDIO:
    Project Settings → Machine Learning → Add Backend → http://localhost:9090

LABEL STUDIO PROJECT CONFIG (copy-paste into Settings → Labeling Interface):
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="person"/>
        <Label value="car"/>
        <Label value="bike"/>
        <Label value="truck"/>
        <Label value="bus"/>
        <Label value="taxi"/>
        <Label value="pickup"/>
        <Label value="trailer"/>
        <Label value="tuktuk"/>
        <Label value="agri_truck"/>
        <Label value="van"/>
      </RectangleLabels>
    </View>
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from urllib.parse import unquote, urlparse

from label_studio_ml.model import LabelStudioMLBase

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "pipeline_config.yaml"

# ── keyword → class label ───────────────────────────────────────────────────
# Built once from config. Maps DINO's returned phrases to your class names.
# e.g. "tuk-tuk" → "tuk_tuk", "pedestrian" → "person"
def _build_phrase_map(class_prompts: dict, class_names: dict | None = None) -> dict:
    """Build a phrase → class-name lookup.

    `class_names` is the YAML-driven id → name mapping (post-2026-05-08 it's
    12-class traffic12 order). If omitted, falls back to the legacy 11-class
    hardcoded mapping for backward compatibility — but new deployments should
    always pass it explicitly.
    """
    phrase_map: dict[str, str] = {}
    if not class_names:
        class_names = {
            0: "person", 1: "car",    2: "bike",      3: "truck",
            4: "bus",    5: "taxi",   6: "pickup",    7: "trailer",
            8: "tuktuk", 9: "agri_truck", 10: "van",
        }
    for class_id, prompt in class_prompts.items():
        label = class_names.get(int(class_id), f"class_{class_id}")
        for keyword in prompt.split(" . "):
            phrase_map[keyword.strip().lower()] = label
    return phrase_map


def _resolve_image_path(url: str, hostname: str = "", token: str = "") -> str:
    """
    Convert Label Studio image URL to an absolute file path.

    Handles:
      /data/local-files/?d=%2Fabsolute%2Fpath.jpg  →  /absolute/path.jpg
      /data/upload/<pid>/img.jpg                   →  $LS_MEDIA_ROOT/upload/<pid>/img.jpg (filesystem)
                                                      or HTTP-fetch via LS API as fallback
      /absolute/path.jpg                            →  as-is
    """
    if not url:
        return ""

    # Local files storage
    if "local-files" in url and "d=" in url:
        encoded = url.split("d=")[-1].split("&")[0]
        return unquote(encoded)

    # Already an absolute path
    if url.startswith("/") and not url.startswith("/data/"):
        return url

    # Uploaded file — try direct filesystem first if LS_MEDIA_ROOT is set
    # (works when the ML backend runs on the same host as LS).
    # Default points at the standard LS data dir on this DGX install.
    if url.startswith("/data/upload/"):
        media_root = os.environ.get(
            "LS_MEDIA_ROOT",
            "/home/admin/label-studio-data/media",
        )
        relative = url.split("/data/", 1)[1]   # "upload/2/<UUID>.jpg"
        candidate = os.path.join(media_root, relative)
        if os.path.exists(candidate):
            # resolve symlinks so downstream `os.path.exists` is happy
            return os.path.realpath(candidate)

        # Fallback: HTTP-fetch via LS API (works for remote LS server)
        if hostname:
            full_url = hostname.rstrip("/") + url
            try:
                import requests
                headers = {"Authorization": f"Token {token}"} if token else {}
                r = requests.get(full_url, headers=headers, timeout=10)
                if r.status_code == 200:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    tmp.write(r.content)
                    tmp.close()
                    return tmp.name
            except Exception:
                pass

    return url


class GDINOBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cfg = yaml.safe_load(CONFIG_PATH.read_text())

        # ── Device selection (auto-detect by platform) ─────────────────────
        dev = cfg.get("device", "auto")
        if dev == "auto":
            import platform as _pl
            if _pl.system() == "Darwin" and _pl.machine() == "arm64":
                dev = "mps" if torch.backends.mps.is_available() else "cpu"
            elif torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        if dev == "mps" and not torch.backends.mps.is_available():
            dev = "cpu"
            print("[GDINO] MPS not available, falling back to CPU")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
            print("[GDINO] CUDA not available, falling back to CPU")
        self.device = dev
        print(f"[GDINO] Device: {self.device}")

        # ── Load GDINO once — expand ~ in paths ────────────────────────────
        from groundingdino.util.inference import load_model
        gdino_cfg     = str(Path(cfg["gdino"]["config"]).expanduser())
        gdino_weights = str(Path(cfg["gdino"]["weights"]).expanduser())
        print("[GDINO] Loading model weights...")
        self.model = load_model(
            gdino_cfg,
            gdino_weights,
            device=self.device,
        )
        self.box_threshold  = cfg["gdino"]["box_threshold"]
        self.text_threshold = cfg["gdino"]["text_threshold"]

        # ── Build combined prompt (all classes in one DINO call) ────────────
        class_prompts = {int(k): v for k, v in cfg["class_prompts"].items()}
        self.class_names = {int(k): v for k, v in cfg.get("class_names", {}).items()}
        self.name_to_id  = {v: k for k, v in self.class_names.items()}
        self.prompt = " . ".join(class_prompts.values())
        self.phrase_map = _build_phrase_map(class_prompts, self.class_names)

        # ── Optional SAM2 chain (off by default; flip cfg.sam2.enabled to turn on) ─
        self.sam2 = None
        self.levels: dict[int, str] = {}
        sam2_cfg = cfg.get("sam2") or {}
        if sam2_cfg.get("enabled", False):
            try:
                from tools.sam2_refine import SAM2Refiner
                from tools.level_gate import load_per_class_levels
                weights = str(Path(sam2_cfg["weights"]).expanduser())
                sam2_dev = sam2_cfg.get("device", "cpu")
                self.sam2 = SAM2Refiner(sam2_cfg["config"], weights, device=sam2_dev)
                lg = cfg.get("level_gate") or {}
                self.levels = load_per_class_levels(
                    str(Path(lg.get("source", "")).expanduser()),
                    threshold=float(lg.get("level2_min_map50", 0.50)),
                )
                self.scoring = cfg.get("scoring") or {}
                self.review_thr = float(self.scoring.get("review", 0.45))
                print(f"[SAM2] enabled — device={sam2_dev}  level2 classes={sorted(c for c, l in self.levels.items() if l == 'level2')}")
            except (ImportError, FileNotFoundError, KeyError) as e:
                print(f"[SAM2] disabled — {e!r}")
                self.sam2 = None
                self.levels = {}

        print(f"[GDINO] Ready — {len(class_prompts)} classes  sam2={self.sam2 is not None}")
        print(f"[GDINO] Prompt: {self.prompt[:80]}...")

    # ── Main prediction method called by Label Studio ────────────────────────
    def predict(self, tasks, **kwargs):
        from groundingdino.util.inference import load_image, predict as gdino_predict

        results = []
        for task in tasks:
            url       = task.get("data", {}).get("image", "")
            img_path  = _resolve_image_path(url, self.hostname, self.access_token)

            if not img_path or not os.path.exists(img_path):
                print(f"[GDINO] Image not found: {img_path!r} (url={url!r})")
                results.append({"result": [], "score": 0.0})
                continue

            try:
                annotations = self._run_dino(img_path)
            except Exception as e:
                print(f"[GDINO] Inference error on {img_path}: {e}")
                annotations = []

            results.append({"result": annotations, "score": 0.0})

        return results

    @staticmethod
    def _nms(anns: list, iou_thresh: float = 0.5) -> list:
        """Greedy NMS over overlapping LS-format boxes — keep highest score."""
        def iou(a, b):
            ax1, ay1 = a["value"]["x"], a["value"]["y"]
            ax2, ay2 = ax1 + a["value"]["width"], ay1 + a["value"]["height"]
            bx1, by1 = b["value"]["x"], b["value"]["y"]
            bx2, by2 = bx1 + b["value"]["width"], by1 + b["value"]["height"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

        sorted_anns = sorted(anns, key=lambda a: -a["score"])
        kept = []
        while sorted_anns:
            best = sorted_anns.pop(0)
            kept.append(best)
            sorted_anns = [a for a in sorted_anns if iou(best, a) < iou_thresh]
        return kept

    @classmethod
    def _nms_per_class(cls, anns: list, iou_thresh: float = 0.5) -> list:
        """NMS applied within each class only.

        Keeps both candidates when DINO returns the same area as two different
        classes (e.g. tuktuk + truck on the same vehicle) — let the human picker
        choose. Within one class, dedupes overlapping boxes by score.
        """
        by_class: dict[str, list] = {}
        for a in anns:
            label = a["value"]["rectanglelabels"][0]
            by_class.setdefault(label, []).append(a)
        kept: list = []
        for group in by_class.values():
            kept.extend(cls._nms(group, iou_thresh))
        return kept

    @staticmethod
    def _combined_score(dino: float, fill: float, w_dino: float = 0.6, w_sam: float = 0.4) -> float:
        """Same `0.6·DINO + 0.4·SAM2_fill` formula as tools/sam2_refine.combined_score.
        Inlined to keep this file standalone (deployable at /home/admin/gdino_backend/
        without needing the tools/ package on sys.path)."""
        return max(0.0, min(1.0, w_dino * float(dino) + w_sam * float(fill)))

    def _run_dino(self, img_path: str) -> list:
        from groundingdino.util.inference import load_image, predict as gdino_predict

        image_source, image_tensor = load_image(img_path)

        boxes, logits, phrases = gdino_predict(
            model          = self.model,
            image          = image_tensor,
            caption        = self.prompt,
            box_threshold  = self.box_threshold,
            text_threshold = self.text_threshold,
            device         = self.device,
        )

        # Pre-load image into SAM2 once per frame so multiple boxes share the encoding cost.
        if self.sam2 is not None:
            self.sam2.reset()
            self.sam2.set_image(image_source)
        H, W = image_source.shape[:2]

        annotations = []
        for box, conf, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
            cx, cy, bw, bh = box

            # Convert DINO format (cx,cy,w,h normalized) → Label Studio (x,y,w,h %)
            x = max(0.0,   (cx - bw / 2) * 100)
            y = max(0.0,   (cy - bh / 2) * 100)
            w = min(100.0, bw * 100)
            h = min(100.0, bh * 100)

            # Skip tiny detections (likely noise)
            if w < 1.0 or h < 1.0:
                continue

            label = self._phrase_to_label(phrase)
            if label is None:
                # DINO returned a phrase we can't map to one of our classes.
                # Drop it rather than emit a junk label.
                continue

            score = round(float(conf), 3)

            # SAM2 refinement on Level-2+ classes only.
            if self.sam2 is not None:
                cls_id = self.name_to_id.get(label)
                if cls_id is not None and self.levels.get(cls_id) == "level2":
                    px1 = int(max(0, (cx - bw / 2) * W))
                    py1 = int(max(0, (cy - bh / 2) * H))
                    px2 = int(min(W, (cx + bw / 2) * W))
                    py2 = int(min(H, (cy + bh / 2) * H))
                    try:
                        fill = self.sam2.mask_fill_ratio(image_source, (px1, py1, px2, py2))
                        score = round(self._combined_score(conf, fill), 3)
                        # Drop noisy boxes below review threshold for mature classes.
                        if score < self.review_thr:
                            continue
                    except Exception as e:
                        print(f"[SAM2] {label} skipped — {e}")

            annotations.append({
                "from_name": "label",
                "to_name":   "image",
                "type":      "rectanglelabels",
                "value": {
                    "x":               x,
                    "y":               y,
                    "width":           w,
                    "height":          h,
                    "rotation":        0,
                    "rectanglelabels": [label],
                },
                "score": score,
            })

        # Class-aware NMS: dedupe within each class but keep cross-class
        # overlaps (so a single tuktuk that DINO also matches as "truck" still
        # surfaces both candidates for human review).
        return self._nms_per_class(annotations, iou_thresh=0.5)

    def _phrase_to_label(self, phrase: str) -> str | None:
        """Map DINO's returned phrase to our class name.

        Returns None when no match — caller should drop the box rather than
        emit a fake class. Handles three quirks:
          1. DINO returns multi-word fragments like "person person" or "tuk tuk".
          2. Dash/space variants ("tuk-tuk" vs "tuk tuk").
          3. Single-word matches when DINO only echoes one token of a multi-word
             keyword (e.g. "rickshaw" alone).
        """
        def norm(s: str) -> str:
            return s.lower().replace("-", " ").replace("_", " ").strip()

        p_norm   = norm(phrase)
        p_compact = p_norm.replace(" ", "")
        p_words  = set(p_norm.split())

        # Pass 1 — exact / substring match after normalisation
        for keyword, label in self.phrase_map.items():
            kw_norm = norm(keyword)
            kw_compact = kw_norm.replace(" ", "")
            if kw_norm and (kw_norm in p_norm or p_norm in kw_norm or kw_compact == p_compact):
                return label

        # Pass 2 — every word of the keyword appears somewhere in the phrase
        for keyword, label in self.phrase_map.items():
            kw_words = set(norm(keyword).split())
            if kw_words and kw_words.issubset(p_words):
                return label

        # Pass 3 — head-noun match (last word of the keyword).
        # Caveat from 2026-05-08 debug: matching ANY position-of-keyword word
        # caused bike→bus mislabels because DINO sometimes returns
        # "large motorbike" / "city scooter", and the generic adjective hit
        # the bus prompt's "large bus" / "city bus" entries via Pass 3.
        # Rules now:
        #   - Word must equal the LAST word of the keyword (the head noun).
        #   - Word must be >=6 chars (filters "large", "city", "tour" entirely).
        #   - The head noun must map to exactly ONE class across all keywords;
        #     ambiguous head nouns (e.g. "taxi" — used in both class 4 and
        #     class 9 prompts) are skipped here and the box is dropped.
        head_to_labels: dict[str, set[str]] = {}
        for kw, lbl in self.phrase_map.items():
            head = norm(kw).split()[-1] if norm(kw).split() else ""
            if len(head) >= 6:
                head_to_labels.setdefault(head, set()).add(lbl)

        for word in p_words:
            if len(word) < 6:
                continue
            labels = head_to_labels.get(word)
            if labels and len(labels) == 1:
                return next(iter(labels))

        # No match — signal caller to drop the box.
        return None

    def fit(self, tasks, workdir=None, **kwargs):
        # Training is handled externally (run_val.py → yolo train)
        # This backend is assist-only — no self-training
        return {}
