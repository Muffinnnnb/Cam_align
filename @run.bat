@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" calibration_gui.py
) else (
    python calibration_gui.py
)
if errorlevel 1 pause
