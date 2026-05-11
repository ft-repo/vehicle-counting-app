"""SAM2 mask refinement for Grounding DINO box predictions.

Used by `tools/gdino_ls_backend.py` to compute a `mask_fill_ratio` —
SAM2-predicted-mask-pixels divided by DINO-box-area — which feeds into the
combined score `0.6·DINO + 0.4·SAM2_fill` used to gate auto-accept vs.
human-review (per `wiki/concepts/lifecycle-workflow.md` "Score-tier routing").

Heavy: imports SAM2 lazily so cold-start of consumer scripts isn't blocked
by the model load if SAM2 isn't enabled.

Expected on the DGX: `pip install sam2==1.1.0` in the gdino-env, plus a
weights file (e.g. ~150 MB at `~/sam2_weights/sam2_small.pt`) and a matching
config name (`sam2_hiera_s.yaml`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def combined_score(
    dino_score: float,
    fill_ratio: float,
    w_dino: float = 0.6,
    w_sam: float = 0.4,
) -> float:
    """`0.6·DINO + 0.4·SAM2_fill` — clamped to [0, 1]."""
    s = w_dino * float(dino_score) + w_sam * float(fill_ratio)
    return max(0.0, min(1.0, s))


class SAM2Refiner:
    """Box-prompted SAM2 predictor + mask-fill-ratio computation.

    Catch FileNotFoundError / ImportError at the call site to gracefully
    fall back to DINO-only scoring when SAM2 isn't available.
    """

    def __init__(self, cfg_name: str, weights_path: str, device: str = "cpu") -> None:
        weights = Path(os.path.expanduser(weights_path)).resolve()
        if not weights.exists():
            raise FileNotFoundError(f"SAM2 weights not found: {weights}")

        # Lazy import — sam2 is heavy and only one consumer (the LS-ML backend) needs it.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.sam2 = build_sam2(cfg_name, str(weights), device=device)
        self.predictor = SAM2ImagePredictor(self.sam2)
        self._image_set: bool = False

    def set_image(self, image: "np.ndarray") -> None:
        """Pre-load an image so multiple boxes on the same frame share encoding cost."""
        self.predictor.set_image(image)
        self._image_set = True

    def mask_fill_ratio(
        self,
        image: "np.ndarray",
        box_xyxy_px: tuple[int, int, int, int],
    ) -> float:
        """Run SAM2 with `box_xyxy_px` as a box prompt; return mask_pixels / box_area.

        `image` is reused if already set via `set_image()` — cheap to re-pass.
        """
        import numpy as np

        if not self._image_set:
            self.set_image(image)

        x1, y1, x2, y2 = (int(v) for v in box_xyxy_px)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        box_area = float((x2 - x1) * (y2 - y1))
        if box_area <= 0:
            return 0.0

        masks, scores, _ = self.predictor.predict(
            box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
            multimask_output=False,
        )
        # masks: (1, 1, H, W) bool / float — pick the single mask, count pixels INSIDE the box.
        m = masks[0]
        if m.ndim == 3:
            m = m[0]
        mask_in_box = m[y1:y2, x1:x2]
        mask_pixels = float((mask_in_box > 0.5).sum())
        return max(0.0, min(1.0, mask_pixels / box_area))

    def reset(self) -> None:
        """Forget the cached image; call before processing a new frame."""
        self._image_set = False
