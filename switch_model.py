"""
Switch the active model while vehicle_counter.py is running.

Usage (run in any terminal from Label Test folder):
    python counting_app/switch_model.py 1   → YOLOv4-tiny
    python counting_app/switch_model.py 2   → YOLOv8n run5
    python counting_app/switch_model.py 3   → YOLO11n run2
"""
import sys
import os

LABELS = {
    "1": "YOLOv4-tiny",
    "2": "YOLOv8n run5",
    "3": "YOLO11n run2",
}

if len(sys.argv) < 2 or sys.argv[1] not in LABELS:
    print("Usage: python counting_app/switch_model.py 1|2|3")
    for k, v in LABELS.items():
        print(f"  {k} = {v}")
    sys.exit(1)

num = sys.argv[1]
cmd_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cmd.txt")
with open(cmd_path, "w") as f:
    f.write(num)

print(f"[SWITCH] Sent → model {num}: {LABELS[num]}")
print(f"[SWITCH] vehicle_counter.py will pick this up within 1 second.")
