@echo off
REM Build a standalone Windows app from main.py using PyInstaller.
REM Output: dist/emotion-detector/ (includes a models/ folder copy).
REM Tip: drop a calibrated model (e.g. models/emotion_cnn_calibrated.pt)
REM next to the exe and pass --model to select it.
setlocal
cd /d "%~dp0"

python -m PyInstaller --noconfirm --clean --onedir ^
    --name emotion-detector ^
    --add-data "models;models" ^
    main.py

if errorlevel 1 (
    echo.
    echo Build FAILED. Make sure dev deps are installed: pip install -r requirements-dev.txt
    exit /b 1
)

echo.
echo Done. Run it from: dist\emotion-detector\emotion-detector.exe
echo Press any key to exit.
pause >nul