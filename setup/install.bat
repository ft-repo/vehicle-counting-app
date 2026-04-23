@echo off
cd /d "%~dp0\.."
setlocal enabledelayedexpansion

echo.
echo =========================================
echo   Vehicle Counter -- First Time Install
echo =========================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download from https://www.python.org/downloads/
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo        %%v
echo.

REM ── Detect GPU ────────────────────────────────────────────────────────────
echo [2/3] Detecting GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo        No NVIDIA GPU detected.
    echo        Installing CPU-only OpenCV.
    set GPU_AVAILABLE=0
) else (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
        echo        GPU found: %%g
    )
    set GPU_AVAILABLE=1
)
echo.

REM ── Install packages ──────────────────────────────────────────────────────
echo [3/3] Installing packages...
echo.

if "!GPU_AVAILABLE!"=="1" (
    echo        GPU detected -- installing CUDA-enabled OpenCV.
    echo        This allows both CPU and GPU options in run_camera.bat.
    echo.

    REM Try conda first
    where conda >nul 2>&1
    if not errorlevel 1 (
        echo        Using conda ^(recommended^)...
        conda install -c conda-forge opencv cudatoolkit numpy -y
        if not errorlevel 1 (
            pip install rich ultralytics
            goto done
        )
        echo        conda install failed, trying pip...
    )

    REM Pip fallback -- detect Python version for correct wheel
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
    for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
        set PY_MAJOR=%%a
        set PY_MINOR=%%b
    )
    set PYTAG=cp!PY_MAJOR!!PY_MINOR!

    echo        Detected Python !PY_MAJOR!.!PY_MINOR! ^(tag: !PYTAG!^)
    echo        Downloading CUDA-enabled OpenCV wheel...
    echo.

    set WHEEL_URL=https://github.com/cudawaredev/opencv-cuda-wheels/releases/download/4.10.0/opencv_contrib_python-4.10.0-!PYTAG!-!PYTAG!-win_amd64.whl

    pip install "!WHEEL_URL!" 2>nul
    if errorlevel 1 (
        echo.
        echo [WARN] Auto-download failed. Falling back to CPU-only OpenCV.
        echo        After setup, GPU option will not be available until you
        echo        manually install a CUDA wheel from:
        echo        https://github.com/cudawaredev/opencv-cuda-wheels/releases
        echo.
        pip install opencv-python numpy rich ultralytics
    ) else (
        pip install numpy rich ultralytics
    )

) else (
    echo        No GPU -- installing CPU-only packages.
    pip install opencv-python numpy rich ultralytics
)

:done
echo.
echo =========================================
echo   Verifying installation...
echo =========================================
echo.
python -c "import cv2, numpy; print('  cv2     :', cv2.__version__); print('  numpy   :', numpy.__version__)"

echo.
python setup\check_gpu.py --quiet
if errorlevel 1 (
    echo   CUDA   : NOT available  ^(CPU mode only^)
) else (
    echo   CUDA   : READY  ^(GPU mode available^)
)

echo.
echo =========================================
echo   Setup complete.
echo   Run run_camera.bat to start.
echo =========================================
echo.
pause
