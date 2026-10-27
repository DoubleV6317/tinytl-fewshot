@echo off
REM Continues the most recently updated run that has a checkpoint.
REM Use this after a crash, sleep or reboot instead of train-flowers.cmd,
REM which would start over from epoch 0.
cd /d "%~dp0"
set RUN=
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$r = Get-ChildItem runs -Directory ^| Where-Object { Test-Path (Join-Path $_.FullName 'checkpoint\checkpoint.pth.tar') } ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1; if ($r) { $r.Name }"`) do set RUN=%%i
if "%RUN%"=="" (
  echo No run with a checkpoint was found under runs\.
  pause
  exit /b 1
)
echo Resuming run: %RUN%
".venv\Scripts\python.exe" -u run.py --mode train --epochs 200 --resume-from "runs\%RUN%"
pause
