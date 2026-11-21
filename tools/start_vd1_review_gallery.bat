@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO=data\input\videos\vd-1.mp4"
set "REPORT=reports\vd-1_dashboard_tracks_dense.json"
set "SHORTLIST=reports\vd-1_manual_review_shortlist.csv"
set "GALLERY=data\previews\vd-1_review_gallery\index.html"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found. Set up the project dependencies first.
    goto finish_error
)
if not exist "%VIDEO%" (
    echo %VIDEO% was not found. Put the local vd-1 video there first.
    goto finish_error
)

if not exist "%SHORTLIST%" (
    if not exist "%REPORT%" (
        echo The dense tracking report is missing. Generate it first with:
        echo .venv\Scripts\python.exe tools\track_video.py "data\input\videos\vd-1.mp4" --frame-step 30 --preserve-source-resolution --output reports\vd-1_dashboard_tracks_dense.csv --json-output reports\vd-1_dashboard_tracks_dense.json
        goto finish_error
    )
    echo Creating the local manual-review shortlist...
    "%PYTHON%" tools\create_review_shortlist.py "%REPORT%" --output "%SHORTLIST%"
    if errorlevel 1 goto finish_error
)

echo Refreshing the local review gallery from the current video and template...
"%PYTHON%" tools\export_review_gallery.py "%VIDEO%" --shortlist-csv "%SHORTLIST%" --output-dir data\previews\vd-1_review_gallery
if errorlevel 1 goto finish_error

echo Opening the offline review gallery. No labels are created automatically.
start "" "%CD%\%GALLERY%"
if errorlevel 1 (
    echo Windows could not open the gallery automatically.
    goto finish_error
)
echo If no browser page appears, open this local file manually:
echo %CD%\%GALLERY%
echo Review frames yourself and press Save frame to create manual labels.
pause
exit /b 0

:finish_error
pause
exit /b 1
