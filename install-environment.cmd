@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo Environment ready. Run prepare.py to download data if needed.
pause
exit /b 0
:failed
echo Setup failed. Review the error above.
pause
exit /b 1
