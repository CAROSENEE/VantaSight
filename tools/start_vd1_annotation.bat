@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo VantaSight virtual environment not found.
    echo Create it and install the project dependencies first.
    pause
    exit /b 1
)

if not exist "data\input\videos\vd-1.mp4" (
    echo data\input\videos\vd-1.mp4 was not found.
    echo Put the local video in that folder, then run this launcher again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "tools\annotate_video_frames.py" "data\input\videos\vd-1.mp4" --frames 0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000,6600 --prefill "data\annotations\vd-1_provisional_review.jsonl" --output "data\annotations\vd-1_user_verified.jsonl" --keep-on-top

if errorlevel 1 (
    echo.
    echo Annotation stopped with an error. Read the message above.
)
pause
