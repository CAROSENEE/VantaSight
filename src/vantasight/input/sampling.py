"""Sampling-rate helpers for offline video analysis."""

import math


def frame_step_for_sample_rate(source_fps: float, target_samples_per_second: float = 4.0) -> int:
    """Choose a source-frame interval near a target sample rate.

    At native rates below the target, every frame is selected. For example,
    120 FPS at four samples per second returns a frame step of 30.
    """

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source FPS must be a finite positive number")
    if not math.isfinite(target_samples_per_second) or target_samples_per_second <= 0:
        raise ValueError("Target sample rate must be a finite positive number")
    return max(1, round(source_fps / target_samples_per_second))
