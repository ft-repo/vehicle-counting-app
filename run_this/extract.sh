#!/usr/bin/env bash
# ============================================================
#  FRAME EXTRACTOR
#  Pre-configured by Person A. Do NOT modify unless instructed.
#  Just run this file and copy the final number to Google Sheet.
# ============================================================
#
#  HOW TO RUN:
#    1. Open Terminal (macOS / Linux)
#    2. Type:  bash run_this/extract.sh
#    3. Wait until you see [DONE]
#    4. Copy the number after "Copy this number to Google Sheet column C:"
#
#  On Windows: run  run_this\extract.bat  instead
# ============================================================
#  Person A: edit the lines marked EDIT below before each session
# ============================================================

set -e
cd "$(dirname "$0")/.."

# Auto-detect Python
if command -v python &>/dev/null; then
    PYTHON="python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "[ERROR] Python not found."
    exit 1
fi

# ── EDIT: change these before each collection session ──────────────────────
SOURCE="rtsp://root:pass@100.115.149.76/axis-media/media.amp?resolution=1280x960"
OUTPUT="raw_frames/tuk_tuk"   # folder name = class being collected
INTERVAL=3                    # seconds between frames  (3 s ≈ 333 frames/hour)
MAX=1000                      # max frames this session
# ───────────────────────────────────────────────────────────────────────────

echo "============================================"
echo "  FRAME EXTRACTOR"
echo "  Class:    $OUTPUT"
echo "  Camera:   $SOURCE"
echo "  Interval: ${INTERVAL}s"
echo "  Max:      $MAX frames"
echo "============================================"
echo ""

$PYTHON tools/frame_extractor.py \
    --source   "$SOURCE" \
    --output   "$OUTPUT" \
    --interval "$INTERVAL" \
    --max      "$MAX"
