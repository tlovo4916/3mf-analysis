@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
echo Analysis dependencies are ready.
echo Optional SolidWorks build dependency: py -m pip install pywin32
pause
