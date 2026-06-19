"""
Switch the active model while vehicle_counter.py is running.

Writes the chosen preset number to repo-root/model_cmd.txt; vehicle_counter.py
polls that file once a second and hot-swaps without a restart. The preset list
is read from config/scene_config.json -> model_presets (the same source the
runtime uses), so the numbering here always matches what gets loaded.

Usage (run from the repo root, in any terminal):
    python switch_model.py 1     # first preset
    python switch_model.py 2     # ...
"""
import json
import os
import sys

ROOT  = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "config", "scene_config.json")

try:
    _presets = json.load(open(SCENE)).get("model_presets", [])
except Exception:
    _presets = []
LABELS = {str(i + 1): p.get("label", f"preset {i + 1}") for i, p in enumerate(_presets)}

if not LABELS:
    print("No model_presets found in config/scene_config.json — nothing to switch to.")
    sys.exit(1)

if len(sys.argv) < 2 or sys.argv[1] not in LABELS:
    print("Usage: python switch_model.py <n>")
    for k, v in LABELS.items():
        print(f"  {k} = {v}")
    sys.exit(1)

num = sys.argv[1]
cmd_path = os.path.join(ROOT, "logs", "model_cmd.txt")
os.makedirs(os.path.dirname(cmd_path), exist_ok=True)
with open(cmd_path, "w") as f:
    f.write(num)

print(f"[SWITCH] Sent → preset {num}: {LABELS[num]}")
print("[SWITCH] vehicle_counter.py will pick this up within 1 second.")
