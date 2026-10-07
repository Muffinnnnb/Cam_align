@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" nsl_color_gui.py
) else (
    python nsl_color_gui.py
)
if errorlevel 1 pause
