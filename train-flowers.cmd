@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" run.py --mode train --epochs 200
pause
