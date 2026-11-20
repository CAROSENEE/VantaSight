# Valorant Visual Intelligence & Aim-Precision Analysis

## Research Project Specification for Codex

**Document type:** Software Requirements & Technical Implementation
Guide\
**Project category:** Computer Vision / AI / Real-Time Visual Analytics\
**Reference environment:** Valorant\
**Primary goal:** Analyze gameplay visuals and measure target-detection
and aim-precision metrics without controlling the game.

------------------------------------------------------------------------

## 1. Project Overview

Build a computer-vision research application that analyzes Valorant
gameplay footage and reports visual target information, body-region
estimates, tracking quality, and crosshair-to-target alignment metrics.

The application is an **analytics and research tool**, not a gameplay
automation tool.

### Core research question

How accurately and how quickly can a computer-vision pipeline detect
visible player characters, estimate visible body regions, track targets
across frames, and measure the distance between a target region and the
on-screen crosshair?

### Important operating boundary

The application must not: - Move the in-game crosshair or control the
player's aim. - Send mouse or keyboard inputs, click, or fire weapons. -
Automatically acquire, lock onto, or follow an opponent in the live
game. - Inject code into Valorant, hook the game process, read game
memory, inspect game files, or bypass anti-cheat. - Provide an unfair
gameplay advantage through automated interaction.

The application may: - Analyze user-provided screenshots and recorded
gameplay videos. - In a research-only mode, passively analyze pixels
captured from the display, without interacting with the game. - Run
experiments on an offline, self-created FPS simulation or permitted
benchmark footage. - Display analytics in a separate application window
after or alongside analysis, without drawing into or modifying the game.

**Recommended first milestone:** Start with recorded video and
screenshots. Add passive live-screen analysis only after the offline
pipeline is reliable and after checking applicable game/platform rules.

------------------------------------------------------------------------

## 2. Intended Features

### 2.1 Visual input

-   Load PNG/JPG screenshots.
-   Load MP4 gameplay recordings.
-   Allow the user to select a video file and configure analysis
    settings.
-   Optional passive desktop capture as a later milestone; no
    game-process hooks or injected overlays.

### 2.2 Player / character detection

-   Detect visible player-character candidates in each frame.
-   Return bounding boxes in pixel coordinates: `x1, y1, x2, y2`.
-   Return a model confidence score.
-   Keep detection confidence separate from aim-alignment score.
-   Do not assume every detected human-shaped object is an enemy. Team
    identity may be unknown unless visually and reliably supported by
    the input.

### 2.3 Body-region analysis

Estimate visible regions such as: - Head - Upper torso / chest - Lower
torso - Legs

Use a suitable object-detection or pose-estimation model. Valorant
character models, armor, animation, occlusion, camera perspective, and
visual effects may make exact anatomical keypoints unreliable. The UI
must label uncertain estimates as uncertain rather than inventing
precision.

### 2.4 Target tracking

-   Associate detections across consecutive frames.
-   Assign a temporary track ID to each visible candidate.
-   Estimate movement direction and image-plane velocity.
-   Handle temporary occlusion and missed detections.
-   Avoid claiming identity persistence when the tracker is uncertain.

### 2.5 Crosshair reference and alignment analytics

-   Estimate or configure the screen center / crosshair reference point.
-   Calculate the pixel distance from the crosshair reference to a
    detected target center or selected body-region center.
-   Normalize the distance by frame dimensions so results can be
    compared across resolutions.
-   Report alignment as a measurement only. Do not move the crosshair or
    issue game input.

### 2.6 Accuracy metrics

The project must distinguish these metrics:

**Detection confidence:** The model's confidence that a visual object
belongs to a detected class.

**Region-estimation confidence:** The model's confidence in a
body-region estimate.

**Alignment error:** Pixel or normalized distance between the crosshair
reference and a selected target/body-region point.

**Evaluation accuracy:** Measured performance against manually annotated
ground truth, such as precision, recall, F1-score, and localization
error.

Do not label model confidence as actual hit probability. A proposed
region score such as 20/50/80/100 is only a user-defined visualization
scale unless validated against ground-truth data.

Suggested display labels: - Head region: 100 points - Chest region: 80
points - Torso region: 50 points - Leg region: 20 points

These values are **illustrative research weights**, not real weapon
damage, hit probability, or guaranteed accuracy.

------------------------------------------------------------------------

## 3. System Workflow

``` mermaid
flowchart TD
    A[Input: Screenshot or Recorded Video] --> B[Frame Extraction]
    B --> C[Preprocessing]
    C --> D[Player-Character Detection]
    D --> E[Body-Region / Pose Estimation]
    E --> F[Multi-Object Tracking]
    F --> G[Crosshair Reference Point]
    G --> H[Alignment Error Calculation]
    F --> I[Detection and Tracking Evaluation]
    H --> J[Analytics Dashboard]
    I --> J
    J --> K[Export CSV / JSON / Report]
```

### Frame-by-frame processing

1.  Read an image or decode a video frame.
2.  Resize or normalize the frame while preserving coordinate mapping.
3.  Run the detection model.
4.  Apply confidence filtering and non-maximum suppression where
    appropriate.
5.  Estimate body regions or pose keypoints.
6.  Update the tracker with current detections.
7.  Calculate crosshair-to-region distances.
8.  Record metrics with frame number and timestamp.
9.  Render an analysis preview in the separate application window.
10. Export results for later evaluation.

------------------------------------------------------------------------

## 4. Suggested Technology Stack

Use a modular design. Do not install every library before confirming the
project needs it.

  -----------------------------------------------------------------------
  Area                    Suggested technology    Purpose
  ----------------------- ----------------------- -----------------------
  Main language           Python 3.11 or a        Research and
                          compatible supported    application logic
                          version                 

  Computer vision         OpenCV                  Image/video decoding
                                                  and preprocessing

  Detection               Ultralytics YOLO or     Player-character
                          another suitable        candidate detection
                          detector                

  Pose estimation         MediaPipe or a suitable Body keypoint
                          pose model              estimation where
                                                  feasible

  Tracking                ByteTrack or SORT       Association across
                                                  frames

  Numerical analysis      NumPy                   Coordinate and metric
                                                  calculations

  Data handling           Pandas                  Experiment results and
                                                  CSV export

  UI                      Streamlit or PySide6    Separate analytics
                                                  interface

  Evaluation              scikit-learn            Precision, recall, F1
                                                  and confusion matrix

  Testing                 pytest                  Unit and integration
                                                  tests
  -----------------------------------------------------------------------

**Model selection note:** Do not assume a general human-pose model will
work accurately on Valorant footage. Evaluate it on representative,
permitted footage first. A custom detector may require a carefully
annotated dataset.

------------------------------------------------------------------------

## 5. Proposed Project Structure

``` text
valorant-visual-analytics/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── settings.yaml
│
├── data/
│   ├── input/
│   ├── annotations/
│   └── samples/
│
├── models/
│   └── README.md
│
├── src/
│   ├── main.py
│   ├── input/
│   │   ├── image_loader.py
│   │   └── video_loader.py
│   ├── vision/
│   │   ├── preprocessing.py
│   │   ├── player_detector.py
│   │   └── body_region_estimator.py
│   ├── tracking/
│   │   └── object_tracker.py
│   ├── analytics/
│   │   ├── crosshair_reference.py
│   │   ├── alignment_metrics.py
│   │   └── evaluation.py
│   ├── visualization/
│   │   └── annotation_renderer.py
│   ├── export/
│   │   └── results_exporter.py
│   └── ui/
│       └── dashboard.py
│
├── tests/
│   ├── test_alignment_metrics.py
│   ├── test_video_loader.py
│   └── test_tracking.py
│
└── reports/
```

Codex may adjust this structure if the existing repository already has a
sound architecture. Do not overwrite existing user work.

------------------------------------------------------------------------

## 6. Data Model

Each analyzed detection should be representable as a structured record.

Example JSON:

``` json
{
  "frame_index": 125,
  "timestamp_seconds": 4.166,
  "track_id": 3,
  "bounding_box": {
    "x1": 610,
    "y1": 180,
    "x2": 730,
    "y2": 470
  },
  "detection_confidence": 0.91,
  "body_region": "chest",
  "region_confidence": 0.78,
  "region_center": {
    "x": 670,
    "y": 270
  },
  "crosshair_reference": {
    "x": 960,
    "y": 540
  },
  "alignment_error_pixels": 501.2,
  "alignment_error_normalized": 0.232
}
```

The coordinates above are example data only. The implementation must
calculate values from actual input frames.

------------------------------------------------------------------------

## 7. Alignment Mathematics

Let: - Crosshair reference point be (C=(c_x,c_y)). - Selected
target-region center be (T=(t_x,t_y)). - Frame width be (W), and frame
height be (H).

Pixel alignment error:

\[ d=`\sqrt{(t_x-c_x)^2+(t_y-c_y)^2}`{=tex} \]

Normalized alignment error:

\[ d_n=`\frac{d}{\sqrt{W^2+H^2}}`{=tex} \]

A lower value means the selected point is closer to the crosshair
reference. This is a geometric alignment measurement, not a prediction
that a shot would hit.

For a body-region visualization, report the region label,
region-estimation confidence, and alignment error as separate fields.

------------------------------------------------------------------------

## 8. Dashboard Requirements

The separate research dashboard should include:

### Input panel

-   Select image or video.
-   Show file name, resolution, frame rate, and duration when available.
-   Start, pause, and stop offline analysis.

### Analysis preview

-   Show the processed frame.
-   Draw detection bounding boxes.
-   Show temporary track IDs.
-   Mark estimated body regions/keypoints.
-   Draw the configured crosshair reference point.
-   Show alignment distance and confidence values.

### Metrics panel

-   Number of frames processed.
-   Number of detections.
-   Average processing time per frame.
-   Frames per second (analysis throughput).
-   Detection precision, recall and F1 when ground-truth annotations
    exist.
-   Tracking metrics when suitable annotated data exists.
-   Alignment-error distribution.

### Export panel

-   Export per-frame results as CSV.
-   Export structured results as JSON.
-   Save an annotated analysis video only when the user explicitly
    requests it.

------------------------------------------------------------------------

## 9. Performance and Reliability

Measure performance instead of promising a particular speed.

Record: - Input resolution and source FPS. - Analysis FPS. - Average and
95th-percentile frame-processing latency. - Detection confidence
distribution. - Missed detections and false positives. - Tracking ID
switches and track fragmentation. - Memory and GPU/CPU usage where
available.

Use a frame queue carefully if processing video in real time. If the
pipeline cannot keep up, report dropped or skipped frames; do not
silently present stale results as live results.

------------------------------------------------------------------------

## 10. Testing and Evaluation Plan

### Phase 1 --- Input pipeline

-   Load images successfully.
-   Decode videos correctly.
-   Preserve timestamps and frame indices.
-   Handle unsupported files and corrupt videos gracefully.

### Phase 2 --- Detection

-   Test on a small, manually reviewed dataset.
-   Measure precision, recall, and F1.
-   Review false positives and missed player characters.
-   Test different resolutions, lighting, effects, and occlusions.

### Phase 3 --- Body-region estimation

-   Annotate a representative evaluation set.
-   Measure region classification accuracy.
-   Track uncertainty and partially visible body regions.
-   Do not report a body-region estimate when evidence is insufficient.

### Phase 4 --- Tracking

-   Test track continuity across frames.
-   Measure ID switches and lost tracks.
-   Test temporary occlusion and fast movement.

### Phase 5 --- Alignment analytics

-   Verify crosshair coordinates at multiple resolutions.
-   Unit-test the distance formula.
-   Verify normalized values remain resolution-independent for
    equivalent relative positions.

### Phase 6 --- Dashboard and export

-   Confirm that preview labels match exported records.
-   Confirm CSV and JSON are valid.
-   Test long videos and graceful cancellation.

------------------------------------------------------------------------

## 11. Development Roadmap

  -----------------------------------------------------------------------
  Milestone                           Deliverable
  ----------------------------------- -----------------------------------
  M1                                  Python environment, repository
                                      structure, image/video input

  M2                                  Frame extraction and preprocessing

  M3                                  Player-character candidate detector

  M4                                  Body-region estimation experiment

  M5                                  Multi-object tracking

  M6                                  Crosshair reference and alignment
                                      metrics

  M7                                  Separate analytics dashboard

  M8                                  Evaluation report, tests, and
                                      documentation

  M9                                  Optional passive live-display
                                      analysis, subject to applicable
                                      rules
  -----------------------------------------------------------------------

**Recommended approach:** Complete M1--M3 before attempting real-time
analysis. A reliable offline baseline is essential.

------------------------------------------------------------------------

## 12. Instructions for Codex

You are acting as a senior computer-vision engineer helping implement
this research project.

Before writing code:

1.  Inspect the existing repository and identify its language,
    framework, dependencies, and current functionality.
2.  Do not delete, replace, or restructure existing files without
    explaining why.
3.  Compare the current repository with the proposed structure and
    prepare a short implementation plan.
4.  Identify hardware requirements and check whether CUDA/GPU
    acceleration is available; the application must still have a CPU
    fallback where practical.
5.  Start with a minimal working image/video pipeline.

Implementation rules:

-   Build small, testable Python modules.
-   Use type hints and clear docstrings.
-   Keep model inference separate from UI code.
-   Keep detection confidence, region confidence, and alignment error
    separate.
-   Make thresholds and model paths configurable.
-   Handle missing models, unsupported formats, and inference errors
    gracefully.
-   Never fabricate detections or metrics.
-   Add unit tests for mathematical calculations and data
    transformations.
-   Include exact installation and run commands in the README.
-   Pin compatible dependency versions after testing them.
-   Do not download datasets or model weights without documenting their
    source and license.
-   Do not use game-process injection, memory reading, input automation,
    automated aiming/firing, or anti-cheat evasion.

**First task:** Inspect the repository only. Report what already exists,
what is missing, likely dependency conflicts, and a step-by-step plan
for Milestone 1. Do not start a large implementation until the user
approves the plan.

------------------------------------------------------------------------

## 13. Definition of Done

The initial research prototype is complete when:

-   A user can load a screenshot or recorded video.
-   Frames are extracted with correct dimensions and timestamps.
-   The detector returns real, inspectable detections.
-   Body-region estimates include confidence and uncertainty.
-   Tracking produces frame-level track IDs.
-   Crosshair-to-region alignment error is mathematically correct.
-   The dashboard presents the analysis without controlling the game.
-   Results can be exported to CSV and JSON.
-   Tests pass and setup instructions work on a clean environment.
-   Known limitations and evaluation results are documented.

------------------------------------------------------------------------

## 14. Research Limitations

Valorant footage is a challenging computer-vision domain. Player models
may be partially hidden, visually similar to the environment, affected
by abilities and visual effects, or distorted by motion and perspective.
A general-purpose model may not distinguish teams or body regions
reliably.

The project must report measured performance on a documented test set.
It must not claim perfect detection, guaranteed head identification, or
real hit probability without appropriate ground-truth experiments.

**Final project principle:** Analyze visual information, quantify
uncertainty, and produce reproducible research measurements. Do not
automate gameplay actions.
