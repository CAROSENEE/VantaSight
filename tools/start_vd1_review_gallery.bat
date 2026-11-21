@echo off
setlocal
cd /d "%~dp0\.."

set "GALLERY=data\previews\vd-1_review_gallery\index.html"

if not exist "%GALLERY%" (
    echo The local vd-1 review gallery was not found.
    echo Export it first with:
    echo .venv\Scripts\python.exe tools\export_review_gallery.py "data\input\videos\vd-1.mp4" --shortlist-csv "reports\vd-1_manual_review_shortlist.csv" --output-dir "data\previews\vd-1_review_gallery"
    pause
    exit /b 1
)

echo Opening the local offline gallery in your default browser.
echo No labels are created until you review and save frames yourself.
start "" "%CD%\%GALLERY%"
if errorlevel 1 (
    echo Windows could not open the gallery automatically.
    echo Open this file manually: %CD%\%GALLERY%
    pause
    exit /b 1
)

