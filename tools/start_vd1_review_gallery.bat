@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found. Set up the project dependencies first.
    goto finish_error
)
echo This compatibility launcher now supports any numbered vd-N.mp4 video.
"%PYTHON%" "tools\open_review_gallery.py"
if errorlevel 1 (
    echo.
    echo Review gallery could not be opened. Read the dialog or error above.
    pause
    exit /b 1
)
