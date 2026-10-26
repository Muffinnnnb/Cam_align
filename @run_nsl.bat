@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py --mode pointcloud
) else (
    python app.py --mode pointcloud
)
if errorlevel 1 pause
