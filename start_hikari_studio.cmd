@echo off
title Hikari Studio — 60 FPS AI VTuber Engine
cd /d "%~dp0"
echo Starting Hikari Studio at http://127.0.0.1:8800/ ...
start "" "http://127.0.0.1:8800/"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 server.py
) else (
  python server.py
)
pause
