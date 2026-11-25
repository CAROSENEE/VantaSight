@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found.
    echo Create it and install the project dependencies first.
    pause
    exit /b 1
)

echo This compatibility launcher now supports any numbered vd-N.mp4 video.
"%PYTHON%" "tools\open_dashboard_for_video.py"
if errorlevel 1 (
    echo.
    echo Dashboard could not be opened. Read the dialog or error above.
)
pause
