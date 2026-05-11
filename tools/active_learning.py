"""Low-confidence frame capturer for the active-learning loop.

When the deployed YOLO model emits a prediction with `confidence < threshold`,
the current frame is saved to a hot folder so it can re-enter Phase 2
labeling. Cooldown + hourly cap prevent runaway saves on a noisy stream.

Pure stdlib + cv2; runs on Mac/Windows/Linux. Imported by `vehicle_counter.py`.
A separate DGX-side `tools/auto_save.sh` rsyncs the hot folder into
`~/vehicle_dataset/raw/active_learning/<YYYY-MM-DD>/`.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _slug(s: str) -> str:
    """Filesystem-safe slug from a free-form source string."""
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(c if c in keep else "-" for c in (s or "src"))[:48]


class LowConfCapturer:
    def __init__(
        self,
        out_dir: str,
        threshold: float = 0.60,
        cooldown_s: float = 5.0,
        max_per_hour: int = 200,
        enabled: bool = True,
        jpeg_quality: int = 90,
    ) -> None:
        self.out_dir       = Path(os.path.expanduser(out_dir))
        self.threshold     = float(threshold)
        self.cooldown_s    = float(cooldown_s)
        self.max_per_hour  = int(max_per_hour)
        self.enabled       = bool(enabled)
        self.jpeg_quality  = int(jpeg_quality)
        self._last_save_at: float = 0.0
        self._save_history: deque[float] = deque(maxlen=max_per_hour + 1)
        self.saved_count: int = 0

        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _can_save(self) -> bool:
        now = time.monotonic()
        if now - self._last_save_at < self.cooldown_s:
            return False
        # Hourly cap: drop entries older than 3600s, then check len
        cutoff = now - 3600.0
        while self._save_history and self._save_history[0] < cutoff:
            self._save_history.popleft()
        return len(self._save_history) < self.max_per_hour

    @staticmethod
    def _min_confidence(dets: Iterable) -> tuple[float | None, list[dict]]:
        """Find the lowest confidence among detections; also return a serializable list."""
        lo: float | None = None
        rows: list[dict] = []
        for d in dets:
            conf = float(getattr(d, "confidence", 0.0))
            cls  = int(getattr(d, "class_id", -1))
            rect = getattr(d, "rect", None)
            row  = {"class_id": cls, "confidence": round(conf, 4)}
            if rect is not None and len(rect) == 4:
                row["bbox_xywh"] = [int(v) for v in rect]
            rows.append(row)
            lo = conf if lo is None else min(lo, conf)
        return lo, rows

    def maybe_save(
        self,
        frame: np.ndarray,
        dets: Iterable,
        frame_idx: int,
        source: str = "src",
    ) -> bool:
        """Save the frame if any detection is below `threshold`. Returns True iff saved."""
        if not self.enabled:
            return False

        lo, det_rows = self._min_confidence(dets)
        if lo is None or lo >= self.threshold:
            return False
        if not self._can_save():
            return False

        now_wall = datetime.now()
        date_dir = self.out_dir / now_wall.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        slug = _slug(source)
        stem = f"{now_wall.strftime('%H-%M-%S')}_{frame_idx:08d}_{slug}"
        jpg_path  = date_dir / f"{stem}.jpg"
        json_path = date_dir / f"{stem}.json"

        ok = cv2.imwrite(
            str(jpg_path), frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return False

        sidecar = {
            "saved_at":      now_wall.isoformat(timespec="seconds"),
            "frame_idx":     int(frame_idx),
            "source":        source,
            "threshold":     self.threshold,
            "min_confidence": round(float(lo), 4),
            "detections":    det_rows,
        }
        json_path.write_text(json.dumps(sidecar, indent=2))

        now_mono = time.monotonic()
        self._last_save_at = now_mono
        self._save_history.append(now_mono)
        self.saved_count += 1
        return True
