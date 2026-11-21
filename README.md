# VantaSight

### Visual analytics for recorded FPS footage

VantaSight is a computer-vision research project. The first target is Valorant footage; the input and analytics pipeline should remain reusable for other FPS or online games later. It aims to measure candidate detections, visible body-region estimates, tracking quality, and crosshair alignment. It does not control a game or automate player input.

The input foundation and frame preview are implemented. An optional baseline person-candidate detector is available for offline screenshots and selected video frames; it is a research baseline and does not identify enemies or teams.

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

Annotated previews go to `data/previews/` by default. `--max-side` controls internal detector resizing; reported bounding boxes map back to the original input resolution. Per-file exclusion rectangles are stored in the local, Git-ignored `data/annotations/source_regions.json`; the default file is loaded automatically by source filename. For one-off adjustments, repeat `--exclude-region x1,y1,x2,y2`. Region coordinates use source pixels, so each video or screenshot needs its own coordinates. These masks skip candidate boxes whose centers fall inside the marked areas; they do not crop or resize the source file.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the step-by-step roadmap, current progress, and plain-language notes about each component.

## Manually label opponent boxes

Use the local annotation tool to create review labels for selected recorded-video frames:

```powershell
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --frames 600,3000,4200,5400,6600
```

The preview is scaled only to fit the window; saved boxes use the original video pixel coordinates. Drag around each visible in-world opponent. Do not label the streamer inset, teammates, first-person hands/weapons, HUD, or replay silhouettes as opponents. Use `u` if the frame is ambiguous. Press `y` to mark an opponent visible, `n` for none visible, and `Enter` to save and continue. Use `z` to undo the latest box, `c` to clear, and `q` to stop; saved rows remain. Labels append to the local, Git-ignored `data/annotations/manual_enemy_boxes.jsonl` file.

After labeling, calculate offline detection metrics with:

```powershell
python tools/evaluate_detector.py "data/input/videos/vd-1.mp4" --confidence-threshold 0.25 --iou-threshold 0.50
```

The evaluator reports per-frame true positives, false positives, false negatives, precision, and recall. Uncertain frames are skipped. It requires manual annotations and writes its CSV under the Git-ignored `reports/` folder.

## Track candidates in a recorded video

Run short-term box association offline and export sampled detections with temporary IDs and image-plane velocity:

```powershell
python tools/track_video.py "data/input/videos/vd-1.mp4" --frame-step 30 --preview-frame 3000
```

Add experimental body-region estimates and geometric alignment measurements when desired:

```powershell
python tools/track_video.py "data/input/videos/vd-1.mp4" --frame-step 30 --estimate-body-regions --measure-alignment
```

Use `--crosshair x,y` to set a different reference in source-image pixels. Pose estimates are matched to candidate boxes by one-to-one IoU (`--pose-match-iou-threshold` changes the minimum overlap); unmatched pose observations remain unassociated. `--pose-person-threshold` filters pose-model person detections, while `--keypoint-score-threshold` filters raw keypoint heatmap values. These are distinct scores. The keypoint model is COCO-trained, and its raw keypoint scores are not calibrated confidence values. Alignment is a geometric offset, not hit probability or aim accuracy.

The frame step is in source frames. Use `--start-frame` and exclusive `--stop-frame` to analyze a segment. Results go to `reports/tracks.csv` and the structured `reports/tracks.json`; use `--output` and `--json-output` to choose other paths. The JSON report includes video metadata, analysis settings, per-frame detections, temporary IDs, image-plane velocity, and a runtime summary. An optional preview goes to `data/previews/`. IDs associate boxes only, can fragment when detections disappear or change shape, and do not identify players, teams, or enemies. Velocity is measured in source-image pixels per second, not game-world speed.

Open the saved report in the separate offline dashboard:

```powershell
python tools/open_dashboard.py reports/tracks.json
```

The dashboard loads the report and source video from local disk, lets you browse sampled frames, and draws the recorded candidate boxes and temporary track IDs. Pass `--video path\to\source.mp4` when the video is stored outside `data/input/videos/`. It does not capture the desktop or interact with a game.

## Experimental body-region estimate

The optional COCO Keypoint R-CNN experiment can be run on a screenshot or one selected video frame:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 3000 --estimate-body-regions --max-side 1920
```

It draws visible keypoints and coarse head/chest/lower-torso/leg centers to `data/previews/`. These are geometric estimates from COCO pose keypoints, not Valorant-trained anatomy labels. TorchVision's keypoint heatmap response is a raw score, not a probability or calibrated confidence; missing regions remain unreported. See [the model card](docs/MODEL_CARDS.md).

Measure alignment from the frame-center reference, or configure one in source-image pixels:

```powershell
python -m vantasight "data/input/screenshots/ss-1.png" --estimate-body-regions --measure-alignment
python -m vantasight "data/input/screenshots/ss-1.png" --estimate-body-regions --measure-alignment --crosshair 959,536
```

The CLI reports pixel distance and distance divided by the frame diagonal for each available estimated region. This is geometry only; it is not hit probability, enemy identification, or an instruction to move aim. A red cross in the preview marks the reference point.

Track sampled candidate boxes in an offline video with [the tracking tool](tools/track_video.py). Its IDs are temporary associations, and its speed values are measured in source-image pixels per second.

## Current scope

- PNG, JPEG, BMP, and WebP screenshots
- Video formats supported by the installed OpenCV build (commonly MP4, AVI, and MOV)
- Frame indices and timestamps are represented in the video input API
- No game-process access, overlays injected into a game, input automation, or aim control

## Research limitations

Detection and body-region estimates must be evaluated on representative footage. Model confidence is not hit probability, and alignment distance is only a geometric measurement. See [the project specification](Valorant_Visual_Analytics_Codex_Spec.md) for the research goals and evaluation plan.

## Author and license

Md. Moshiur Rahman Sajol. Licensed under GNU GPL v3.0; see `LICENSE`.
