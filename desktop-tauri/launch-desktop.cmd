@echo off
rem Hikari Studio desktop shell. Builds once if needed, then launches.
rem   launch-desktop.cmd          start (build first if no exe yet)
rem   launch-desktop.cmd build    force a rebuild, then start
setlocal
cd /d "%~dp0src-tauri"
set "EXE=%~dp0src-tauri\target\release\hikari-desktop.exe"
if /i "%~1"=="build" goto build
if exist "%EXE%" goto run
:build
echo Building Hikari Studio desktop shell (first build takes ~2 min)...
cargo build --release || (echo Build failed. & pause & exit /b 1)
:run
start "" "%EXE%"
