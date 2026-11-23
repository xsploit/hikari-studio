@echo off
title Hikari Studio — 60 FPS AI VTuber Engine
cd /d "%~dp0"
echo Starting Hikari Studio at http://127.0.0.1:8800/ ...
start "" "http://127.0.0.1:8800/"
"C:\Users\SUBSECT\AppData\Local\Programs\Python\Python313\python.exe" server.py
pause
