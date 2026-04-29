#!/usr/bin/env bash
# install.sh — First-time setup for macOS and Linux
# Run from the project root:
#   bash setup/install.sh
#
# To use a specific Python:
#   PYTHON=python3.11 bash setup/install.sh

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

echo ""
echo "========================================="
echo "  Vehicle Counter -- Setup (Mac / Linux)"
echo "========================================="
echo ""

# ── 1. Check Python ────────────────────────────────────────────────────────
echo "[1/4] Checking Python..."
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] '$PYTHON' not found."
    echo "        macOS :  brew install python   (or use system Python 3.x)"
    echo "        Ubuntu:  sudo apt install python3 python3-pip"
    exit 1
fi
"$PYTHON" --version
echo ""

# ── 2. Detect OS & architecture ───────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)
echo "[2/4] Detected OS: $OS  Arch: $ARCH"
echo ""

# ── 3. Detect GPU ─────────────────────────────────────────────────────────
echo "[3/4] Detecting GPU..."
GPU_TYPE="none"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "       Apple Silicon (M-series) — MPS acceleration available."
    GPU_TYPE="apple"
elif command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    echo "       NVIDIA GPU: ${GPU_NAME:-detected}"
    GPU_TYPE="nvidia"
else
    echo "       No dedicated GPU detected — CPU only."
fi
echo ""

# ── 4. Install packages ───────────────────────────────────────────────────
echo "[4/4] Installing packages..."
echo ""
"$PYTHON" -m pip install --upgrade pip

if [[ "$GPU_TYPE" == "nvidia" ]]; then
    # NVIDIA: try conda first for CUDA-enabled OpenCV
    if command -v conda &>/dev/null; then
        echo "       Installing CUDA-enabled OpenCV via conda..."
        conda install -c conda-forge opencv cudatoolkit -y 2>/dev/null || {
            echo "       conda install failed — falling back to pip opencv."
            "$PYTHON" -m pip install opencv-python
        }
    else
        # pip wheel does not include CUDA support on Linux
        "$PYTHON" -m pip install opencv-python
        echo ""
        echo "       [NOTE] pip OpenCV does NOT include CUDA."
        echo "              For GPU acceleration install via conda:"
        echo "                conda install -c conda-forge opencv cudatoolkit"
    fi
else
    "$PYTHON" -m pip install opencv-python
fi

"$PYTHON" -m pip install numpy rich ultralytics

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Verifying installation..."
echo "========================================="
echo ""

"$PYTHON" -c "
import cv2, numpy, rich, ultralytics
print('  cv2        :', cv2.__version__)
print('  numpy      :', numpy.__version__)
print('  rich       : OK')
print('  ultralytics: OK')
"

"$PYTHON" -c "
import cv2
has_cuda = False
try:
    has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    pass
print('  CUDA       :', 'READY' if has_cuda else 'not available (CPU mode)')
try:
    import torch
    print('  MPS        :', 'READY' if torch.backends.mps.is_available() else 'not available')
except ImportError:
    pass
" 2>/dev/null || true

echo ""
echo "========================================="
echo "  Setup complete."
echo "  Start the app:  python run_camera.py"
echo "========================================="
echo ""
