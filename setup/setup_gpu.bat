@echo off
cd /d "%~dp0\.."
echo.
echo =========================================
echo   Vehicle Counter -- GPU Setup (CUDA)
echo =========================================
echo.

REM ── Check nvidia-smi ──────────────────────────────────────────────────────
echo [1/4] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found.
    echo         Make sure NVIDIA drivers are installed.
    echo         Download: https://www.nvidia.com/Download/index.aspx
    pause
    exit /b 1
)
echo [OK] NVIDIA GPU detected.
echo.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo.

REM ── Check CUDA version ────────────────────────────────────────────────────
echo [2/4] Detecting CUDA version...
for /f "tokens=*" %%i in ('nvidia-smi ^| findstr /i "CUDA Version"') do set CUDA_LINE=%%i
echo        %CUDA_LINE%
echo.

REM ── Remove CPU-only OpenCV ────────────────────────────────────────────────
echo [3/4] Removing CPU-only OpenCV...
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y 2>nul
echo [OK] Old OpenCV removed.
echo.

REM ── Install CUDA-enabled OpenCV ───────────────────────────────────────────
echo [4/4] Installing CUDA-enabled OpenCV...
echo.

REM Check if conda is available (preferred method)
where conda >nul 2>&1
if not errorlevel 1 (
    echo [INFO] conda found -- using conda-forge (recommended)
    echo.
    conda install -c conda-forge opencv cudatoolkit -y
    if errorlevel 1 (
        echo [ERROR] conda install failed. Trying pip fallback...
        goto pip_install
    )
    echo.
    echo [OK] Installed via conda.
    goto verify
)

:pip_install
REM Detect Python version for correct wheel
for /f "tokens=1,2 delims=." %%a in ('python --version 2^>^&1') do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
set PY_MAJOR=%PY_MAJOR:Python =%

echo [INFO] Python %PY_MAJOR%.%PY_MINOR% detected
echo [INFO] Installing CUDA OpenCV wheel via pip...
echo.

REM Try installing cuda-opencv from pip (unofficial CUDA builds)
pip install opencv-contrib-python==%PY_MAJOR%.%PY_MINOR%.* --extra-index-url https://download.pytorch.org/whl/cu121 2>nul

REM If above fails, download a known working wheel
if errorlevel 1 (
    echo.
    echo [INFO] Auto-install failed. Please manually download the correct wheel:
    echo.
    echo   Python 3.10:  https://github.com/cudawaredev/opencv-cuda-wheels/releases/download/4.10.0/opencv_contrib_python-4.10.0-cp310-cp310-win_amd64.whl
    echo   Python 3.11:  https://github.com/cudawaredev/opencv-cuda-wheels/releases/download/4.10.0/opencv_contrib_python-4.10.0-cp311-cp311-win_amd64.whl
    echo   Python 3.12:  https://github.com/cudawaredev/opencv-cuda-wheels/releases/download/4.10.0/opencv_contrib_python-4.10.0-cp312-cp312-win_amd64.whl
    echo.
    echo   Then run:  pip install [downloaded_file].whl
    echo.
    pause
    exit /b 1
)

:verify
echo.
echo =========================================
echo   Verifying CUDA support in OpenCV...
echo =========================================
echo.
python setup\check_gpu.py

pause
