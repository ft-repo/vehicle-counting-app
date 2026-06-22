"""
Model registry — the SINGLE source of truth for "which model is deployed".

Every tool that needs the canonical model (run_val.py, auto_label.py,
backfill_per_class.py, and the runtime hot-swap presets) resolves it from
models/model_registry.json through this module, so the answer to "what model are
we running?" lives in exactly one place instead of being hardcoded in a dozen
files.

Paths in the registry are relative to the repo root; the helpers return absolute
Paths. The deployed .pt typically lives on the DGX (the training host); only the
.onnx is mirrored to the Mac for runtime inference — callers that need the .pt
(val / label) should expect it to be present only where training ran.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # repo root
REGISTRY_PATH = BASE / "models" / "model_registry.json"


def load() -> dict:
    if not REGISTRY_PATH.exists():
        raise SystemExit(f"[FATAL] model registry not found: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text())


def deployed() -> dict:
    """The deployed-model record (raw, paths relative to repo root)."""
    return load()["deployed"]


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else BASE / p


def deployed_pt() -> Path:
    """Absolute path to the deployed model's .pt (val / label / training host)."""
    return _abs(deployed()["pt"])


def deployed_onnx() -> Path:
    """Absolute path to the deployed model's .onnx (runtime inference)."""
    return _abs(deployed()["onnx"])


def deployed_names() -> Path:
    """Absolute path to the deployed model's class-names file."""
    return _abs(deployed()["names"])


def presets() -> list[dict]:
    """Hot-swap presets for the runtime video window (1 / 2 / 3 keys)."""
    return load().get("presets", [])
