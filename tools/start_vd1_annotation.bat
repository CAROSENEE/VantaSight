@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO=data\input\videos\vd-1.mp4"
set "REPORT=reports\vd-1_dashboard_tracks_dense.json"
set "SHORTLIST=reports\vd-1_manual_review_shortlist.csv"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found.
    echo Create it and install the project dependencies first.
    pause
    exit /b 1
)

if not exist "%VIDEO%" (
    echo %VIDEO% was not found.
    echo Put the local video in that folder, then run this launcher again.
    pause
    exit /b 1
)

if not exist "%SHORTLIST%" (
    if not exist "%REPORT%" (
        echo %REPORT% was not found. Generate the dense tracking report first:
        echo .venv\Scripts\python.exe tools\track_video.py "data\input\videos\vd-1.mp4" --frame-step 30 --preserve-source-resolution --output reports\vd-1_dashboard_tracks_dense.csv --json-output reports\vd-1_dashboard_tracks_dense.json
        pause
        exit /b 1
    )
    echo Creating the local review shortlist from %REPORT%...
    "%PYTHON%" "tools\create_review_shortlist.py" "%REPORT%" --output "%SHORTLIST%"
    if errorlevel 1 goto finish
)

echo Starting a blank manual review. No provisional boxes will be loaded.
echo Look for a window titled "VantaSight annotation - vd-1.mp4 frame ...".
echo If the window does not appear, keep this console open and read any error below.
echo If the OpenCV window cannot be opened or found, run tools\open_review_gallery.bat for browser-based review.
echo Validating shortlist fingerprint and decoding the selected frames first...
"%PYTHON%" "tools\annotate_video_frames.py" "%VIDEO%" --shortlist-csv "%SHORTLIST%" --dry-run
if errorlevel 1 goto finish

"%PYTHON%" "tools\annotate_video_frames.py" "%VIDEO%" --shortlist-csv "%SHORTLIST%" --output "data\annotations\vd-1_user_verified.jsonl" --keep-on-top

if errorlevel 1 (
    echo.
    echo Annotation stopped with an error. Read the message above.
)
pause
