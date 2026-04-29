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
        <Label value="tuk_tuk"/>
        <Label value="agri_truck"/>
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
def _build_phrase_map(class_prompts: dict) -> dict:
    phrase_map = {}
    class_names = {
        0: "person", 1: "car",    2: "bike",      3: "truck",
        4: "bus",    5: "taxi",   6: "pickup",    7: "trailer",
        8: "tuk_tuk", 9: "agri_truck",
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
      /data/upload/1/img.jpg                        →  fetch via Label Studio API
      /absolute/path.jpg                            →  as-is
    """
    if not url:
        return ""

    # Local files storage (most common for our setup)
    if "local-files" in url and "d=" in url:
        encoded = url.split("d=")[-1].split("&")[0]
        return unquote(encoded)

    # Already an absolute path
    if url.startswith("/") and not url.startswith("/data/"):
        return url

    # Uploaded file — construct full URL and let get_local_path handle it
    # (requires hostname + token to be set on the backend)
    if url.startswith("/data/upload/") and hostname:
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

        # ── Build combined prompt (all 10 classes in one DINO call) ─────────
        class_prompts = {int(k): v for k, v in cfg["class_prompts"].items()}
        self.prompt = " . ".join(class_prompts.values())
        self.phrase_map = _build_phrase_map(class_prompts)

        print(f"[GDINO] Ready — {len(class_prompts)} classes")
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
                "score": round(float(conf), 3),
            })

        return annotations

    def _phrase_to_label(self, phrase: str) -> str:
        """Map DINO's returned phrase to our class name."""
        p = phrase.lower().strip()
        for keyword, label in self.phrase_map.items():
            if keyword in p:
                return label
        # Fallback: clean up the phrase itself
        return p.replace(" ", "_").replace("-", "_")

    def fit(self, tasks, workdir=None, **kwargs):
        # Training is handled externally (run_val.py → yolo train)
        # This backend is assist-only — no self-training
        return {}
