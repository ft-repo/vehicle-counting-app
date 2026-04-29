@echo off
:: ============================================================
::  FRAME EXTRACTOR — Windows version
::  Pre-configured by Person A. Do NOT modify unless instructed.
::  Just run this file and copy the final number to Google Sheet.
:: ============================================================
::
::  HOW TO RUN:
::    1. Double-click this file  (or run from cmd: run_this\extract.bat)
::    2. Wait until you see [DONE]
::    3. Copy the number after "Copy this number to Google Sheet column C:"
::
::  On macOS / Linux: run  run_this/extract.sh  instead
:: ============================================================
::  Person A: edit the lines marked EDIT below before each session
:: ============================================================

cd /d "%~dp0\.."

:: ── EDIT: change these before each collection session ──────────────────
set SOURCE=rtsp://root:pass@100.115.149.76/axis-media/media.amp?resolution=1280x960
set OUTPUT=raw_frames\tuk_tuk
set INTERVAL=3
set MAX=1000
:: ────────────────────────────────────────────────────────────────────────

echo ============================================
echo   FRAME EXTRACTOR
echo   Class:    %OUTPUT%
echo   Camera:   %SOURCE%
echo   Interval: %INTERVAL%s
echo   Max:      %MAX% frames
echo ============================================
echo.

python tools\frame_extractor.py ^
    --source   "%SOURCE%" ^
    --output   "%OUTPUT%" ^
    --interval %INTERVAL% ^
    --max      %MAX%

pause
