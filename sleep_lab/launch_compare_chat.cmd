@echo off
setlocal
set "PY=C:\Users\SUBSECT\AppData\Local\Temp\claude\e3af74eea852ab34\4dcdfb0b-e418-45b9-bb03-3131cbb3e3e3\scratchpad\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Users\SUBSECT\AppData\Local\Programs\Python\Python313\python.exe"
start "" http://127.0.0.1:8809/
"%PY%" "%~dp0sleep_compare_chat.py" --autoload %*
