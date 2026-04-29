@echo off
:: Windows launcher — delegates to the cross-platform run_camera.py
cd /d "%~dp0"

echo.
echo =======================================
echo   Vehicle Detection System
echo =======================================
echo   [1] GPU  (CUDA)
echo   [2] CPU  (default)
echo =======================================
echo.
set /p CHOICE="Enter 1 for GPU or 2 for CPU (default: 2): "

if "%CHOICE%"=="1" (
    echo.
    echo [INFO] Checking CUDA availability...
    python setup\check_gpu.py --quiet
    if errorlevel 1 (
        echo.
        echo [WARN] GPU / CUDA not available on this machine.
        echo        Run setup\install.bat to enable GPU support.
        echo.
        set /p FALLBACK="Continue with CPU? [Y/N]: "
        if /i "!FALLBACK!"=="N" ( pause & exit /b 0 )
        python run_camera.py --cpu
    ) else (
        python run_camera.py --gpu
    )
) else (
    python run_camera.py --cpu
)

pause
