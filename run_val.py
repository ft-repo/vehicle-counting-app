"""
Run validation on best model and save per-class mAP50 to val_results.json.
Requires the SSD (/Volumes/Puen_SSD) to be connected.

Usage:
    python counting_app/run_val.py
    python counting_app/run_val.py --model runs/detect/model_compare/yolov8n/run5/weights/best.pt
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models/yolov8n.onnx"
DEFAULT_DATA  = ROOT / "new_data/dataset/data.yaml"
OUT_PATH      = ROOT / "logs/val_results.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--data",  default=str(DEFAULT_DATA))
    p.add_argument("--imgsz", type=int, default=416)
    args = p.parse_args()

    from ultralytics import YOLO
    print(f"[VAL] Model : {args.model}")
    print(f"[VAL] Data  : {args.data}")

    model   = YOLO(args.model)
    results = model.val(data=args.data, imgsz=args.imgsz)

    class_acc = {
        name: float(ap)
        for name, ap in zip(results.names.values(), results.box.ap50)
    }
    overall = float(results.box.map50)

    # Derive a short model label from the path
    parts = Path(args.model).parts
    try:
        idx   = parts.index("model_compare")
        label = f"{parts[idx+1]} {parts[idx+2]}"
    except (ValueError, IndexError):
        label = Path(args.model).stem

    payload = {
        "model":          label,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overall_map50":  overall,
        "precision":      float(results.box.mp),
        "recall":         float(results.box.mr),
        "classes":        class_acc,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n[VAL] Results saved to {OUT_PATH}")
    print(f"[VAL] Overall mAP50: {overall*100:.1f}%")
    print("\nPer-class mAP50:")
    for cls, ap in class_acc.items():
        bar   = "█" * int(ap * 20)
        flag  = "✓" if ap >= 0.90 else ("~" if ap >= 0.60 else "✗")
        print(f"  {flag} {cls:<10} {ap*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
