@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo =======================================
echo   Vehicle Counter -- Backend Selection
echo =======================================
echo   [1] GPU  (CUDA)
echo   [2] CPU
echo =======================================
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="2" goto run_cpu

REM ── GPU selected: check if CUDA is available ──────────────────────────────
:run_gpu
echo.
echo [INFO] Checking CUDA availability...
python setup\check_gpu.py --quiet
if errorlevel 1 (
    echo.
    echo [WARN] GPU / CUDA is not available on this machine.
    echo        OpenCV is installed without CUDA support.
    echo.
    echo        To enable GPU:  run setup\install.bat
    echo.
    set /p FALLBACK="Continue with CPU instead? [Y/N]: "
    if /i "!FALLBACK!"=="Y" goto run_cpu
    if /i "!FALLBACK!"=="y" goto run_cpu
    echo.
    echo [INFO] Exiting. Run setup\install.bat then try again.
    pause & exit /b 0
)

echo [INFO] CUDA OK -- starting with GPU backend...
echo.
python vehicle_counter.py ^
    "rtsp://root:pass@axis-b8a44fe03000-rama3-10-207-200-7.tail8176dd.ts.net/axis-media/media.amp?resolution=1280x960" ^
    --config "config\scene_config.json" ^
    --stats  "logs\live_stats.json" ^
    --gpu
goto end

REM ── CPU ────────────────────────────────────────────────────────────────────
:run_cpu
echo.
echo [INFO] Starting with CPU backend...
echo.
python vehicle_counter.py ^
    "rtsp://root:pass@axis-b8a44fe03000-rama3-10-207-200-7.tail8176dd.ts.net/axis-media/media.amp?resolution=1280x960" ^
    --config "config\scene_config.json" ^
    --stats  "logs\live_stats.json" ^
    --cpu

:end
pause
