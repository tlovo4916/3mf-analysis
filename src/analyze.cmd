@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo 请把一个或多个 3MF 文件拖到这个脚本上。
  pause
  exit /b 1
)
py assistant.py %* --out results
if errorlevel 1 echo 分析存在失败，请检查 results 目录中的 error.json。
pause
