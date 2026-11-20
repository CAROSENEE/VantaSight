# VantaSight

### Visual analytics for recorded FPS footage

VantaSight is a computer-vision research project. The first target is Valorant footage; the input and analytics pipeline should remain reusable for other FPS or online games later. It aims to measure candidate detections, visible body-region estimates, tracking quality, and crosshair alignment. It does not control a game or automate player input.

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

For person-candidate detection, install the matching PyTorch and TorchVision versions. The current RTX 3050 setup uses CUDA 12.8:

```powershell
python -m pip install -r requirements-vision.txt --index-url https://download.pytorch.org/whl/cu128
```

For a CPU-only environment, use PyTorch's CPU wheel index instead:

```powershell
python -m pip install -r requirements-vision.txt --index-url https://download.pytorch.org/whl/cpu
```

## Inspect an input file

```powershell
python -m vantasight "C:\path\to\screenshot.png"
python -m vantasight "C:\path\to\recording.mp4"
```

The command reports image dimensions or video dimensions, FPS, duration, and frame count. The reusable input API is under `src/vantasight/input/`.

## Local footage folders

- Put screenshots in `data/input/screenshots/`.
- Put recorded videos in `data/input/videos/`.
- Keep the IDs you assign in the filenames: `vd-1.mp4`, `vd-2.mp4`, `ss-1.png`, `ss-2.jpg`, and so on. Video IDs (`vd-N`) and screenshot IDs (`ss-N`) each have their own sequence. Preview filenames retain the video ID, for example `vd-1_frame_000120.png`.
- These input folders are ignored by Git, so your media will not be included in a normal commit or pushed to GitHub. Do not use `git add -f` on media unless you intentionally want to upload it.

## Save a frame preview

Extract a zero-based frame index from a video. By default, the preview is saved under `data/previews/`:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 120
```

Choose a custom output path if needed:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 120 --output "data/previews/vd-1_frame_000120.png"
```

## Run the baseline person detector

The optional detector reports COCO `person` candidates. It does not identify enemies or teams. The first detection run downloads the documented TorchVision COCO_V1 checkpoint; see [the model card](docs/MODEL_CARDS.md).

```powershell
python -m vantasight "data/input/screenshots/ss-1.png" --detect-persons
python -m vantasight "data/input/screenshots/ss-2.png" --detect-persons
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 3000 --detect-persons --max-side 2560 --exclude-region 0,620,530,960
```

Annotated previews go to `data/previews/` by default. `--max-side` controls internal detector resizing; reported bounding boxes map back to the original input resolution. Exclusion rectangles can be passed in source pixels with repeatable `--exclude-region x1,y1,x2,y2` options, for example to mask the `vd-1.mp4` webcam inset. The example region is specific to that video; other files need their own region coordinates.

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
