@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found. Set up the project dependencies first.
    pause
    exit /b 1
)

echo This compatibility launcher opens the local browser-based review gallery.
echo Review vd-1.mp4 and label each frame yourself.
"%PYTHON%" "tools\open_review_gallery.py" --video "data\input\videos\vd-1.mp4"
if errorlevel 1 (
    echo.
    echo Review gallery could not be opened. Read the dialog or error above.
    pause
    exit /b 1
)
