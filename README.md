# VantaSight

### Visual analytics for recorded FPS footage

VantaSight is a computer-vision research project for analyzing screenshots and recorded gameplay. It aims to measure candidate detections, visible body-region estimates, tracking quality, and crosshair alignment. It does not control a game or automate player input.

The first milestone provides a small Python package for reading screenshots and videos and inspecting their metadata. Detection and analytics will be added only after the input pipeline is established.

## Requirements

- Python 3.11 or newer (the current development environment uses Python 3.13)
- The dependencies in `requirements.txt`

## Setup on Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If Python 3.13 is unavailable, create the environment with another supported Python version, for example `py -3.11 -m venv .venv`.

## Inspect an input file

```powershell
python -m vantasight "C:\path\to\screenshot.png"
python -m vantasight "C:\path\to\recording.mp4"
```

The command reports image dimensions or video dimensions, FPS, duration, and frame count. The reusable input API is under `src/vantasight/input/`.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the step-by-step roadmap, current progress, and plain-language notes about each component.

## Current scope

- PNG, JPEG, BMP, and WebP screenshots
- Video formats supported by the installed OpenCV build (commonly MP4, AVI, and MOV)
- Frame indices and timestamps are represented in the video input API
- No game-process access, overlays injected into a game, input automation, or aim control

## Research limitations

Detection and body-region estimates must be evaluated on representative footage. Model confidence is not hit probability, and alignment distance is only a geometric measurement. See [the project specification](Valorant_Visual_Analytics_Codex_Spec.md) for the research goals and evaluation plan.

## Author and license

Md. Moshiur Rahman Sajol. Licensed under GNU GPL v3.0; see `LICENSE`.
