@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found. Set up the project first.
    pause
    exit /b 1
)

echo Select the downloaded, human-reviewed JSONL file in the dialog.
echo VantaSight will match its numbered video ID to the local source and shortlist.
"%PYTHON%" "tools\import_review_labels.py"
if errorlevel 1 (
    echo.
    echo Import did not complete. Read the dialog or error above.
    pause
    exit /b 1
)
