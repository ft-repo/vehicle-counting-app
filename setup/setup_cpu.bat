@echo off
cd /d "%~dp0\.."
echo.
echo =========================================
echo   Vehicle Counter -- CPU Setup
echo =========================================
echo.

echo [1/2] Installing CPU requirements...
pip install -r setup\requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your Python/pip installation.
    pause
    exit /b 1
)

echo.
echo [2/2] Verifying installation...
python -c "import cv2, numpy; print('[OK] cv2', cv2.__version__); print('[OK] numpy', numpy.__version__)"
if errorlevel 1 (
    echo [ERROR] Import failed after install.
    pause
    exit /b 1
)

echo.
echo =========================================
echo   CPU setup complete.
echo   Run run_camera.bat and choose option 2.
echo =========================================
echo.
pause
