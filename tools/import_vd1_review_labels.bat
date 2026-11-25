@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found. Set up the project first.
    pause
    exit /b 1
)

echo Select the downloaded, human-reviewed vd-1 JSONL file in the dialog.
echo The importer validates the video and every sampled frame before saving.
"%PYTHON%" "tools\import_vd1_review_labels.py"
if errorlevel 1 (
    echo.
    echo Import did not complete. Read the dialog or error above.
    pause
    exit /b 1
)
