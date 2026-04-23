"""
Verify CUDA support in OpenCV.

Usage:
    python setup/check_gpu.py           # full output
    python setup/check_gpu.py --quiet   # silent, exit code 0=OK 1=fail
"""
import sys
import os

quiet = "--quiet" in sys.argv

def log(msg):
    if not quiet:
        print(msg)

log("=" * 44)
log(" CUDA Check for Vehicle Counter")
log("=" * 44)

# ── OpenCV ────────────────────────────────────────
try:
    import cv2
    log(f"\n  OpenCV version : {cv2.__version__}")
except ImportError:
    log("\n  [FAIL] OpenCV not installed.")
    log("         Run setup\\install.bat first.")
    sys.exit(1)

# ── CUDA device count ─────────────────────────────
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
except Exception:
    cuda_count = 0

log(f"  CUDA devices   : {cuda_count}")

if cuda_count == 0:
    if not quiet:
        print("\n  [FAIL] OpenCV has NO CUDA support.")
        print("         Current build is CPU-only.")
        print("         Run setup\\install.bat to install GPU support.")
        print()
        build = cv2.getBuildInformation()
        for line in build.splitlines():
            if "CUDA" in line or "NVIDIA" in line:
                print(f"  {line.strip()}")
    sys.exit(1)

# ── GPU info ──────────────────────────────────────
try:
    import numpy as np
    device = cv2.cuda.DeviceInfo(0)
    log(f"  GPU name       : {device.name()}")
    log(f"  Compute ver    : {device.majorVersion()}.{device.minorVersion()}")
    log(f"  Total memory   : {device.totalMemory() // (1024**2)} MB")
except Exception as e:
    log(f"  [WARN] Could not read GPU info: {e}")

# ── Inference probe ───────────────────────────────
log("\n  Running inference probe...")
try:
    import numpy as np
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(root, "models", "yolov8n.onnx")

    if os.path.exists(onnx_path):
        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        dummy = np.zeros((416, 416, 3), np.uint8)
        blob  = cv2.dnn.blobFromImage(dummy, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        net.forward(net.getUnconnectedOutLayersNames())
        log("  [OK] Inference probe passed.")
    else:
        log("  [SKIP] models/yolov8n.onnx not found — skipping inference probe.")
except Exception as e:
    if not quiet:
        print(f"  [FAIL] Inference probe failed: {e}")
    sys.exit(1)

log("")
log("=" * 44)
log(" CUDA is ready. GPU option is available.")
log("=" * 44)
log("")
sys.exit(0)
