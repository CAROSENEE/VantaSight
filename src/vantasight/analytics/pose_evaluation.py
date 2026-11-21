"""Ground-truth localization errors for manually marked body-region centers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class RegionLocalization:
    region_name: str
    ground_truth_xy: tuple[float, float]
    estimate_xy: tuple[float, float] | None
    error_pixels: float | None
    error_normalized: float | None


@dataclass(frozen=True)
class RegionLocalizationSummary:
    region_name: str
    annotated_points: int
    estimated_points: int
    coverage: float | None
    mean_error_pixels: float | None
    median_error_pixels: float | None
    mean_error_normalized: float | None
    median_error_normalized: float | None


def compare_body_region_points(
    expected: dict[str, tuple[float, float]],
    estimates: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> list[RegionLocalization]:
    """Compare expected centers with model centers; missing estimates remain missing."""

    if width < 1 or height < 1:
        raise ValueError("Frame width and height must be positive")
    diagonal = math.hypot(width, height)
    results: list[RegionLocalization] = []
    for region_name, expected_xy in expected.items():
        coordinates = (*expected_xy, *estimates.get(region_name, (0.0, 0.0)))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Body-region coordinates must be finite")
        if not (0 <= expected_xy[0] <= width and 0 <= expected_xy[1] <= height):
            raise ValueError(f"Ground-truth point {expected_xy} lies outside the frame")
        estimate_xy = estimates.get(region_name)
        if estimate_xy is None:
            results.append(RegionLocalization(region_name, expected_xy, None, None, None))
            continue
        if not (0 <= estimate_xy[0] <= width and 0 <= estimate_xy[1] <= height):
            raise ValueError(f"Estimated point {estimate_xy} lies outside the frame")
        error = math.hypot(estimate_xy[0] - expected_xy[0], estimate_xy[1] - expected_xy[1])
        results.append(
            RegionLocalization(region_name, expected_xy, estimate_xy, error, error / diagonal)
        )
    return results


def summarize_region_localizations(
    localizations: list[RegionLocalization],
) -> list[RegionLocalizationSummary]:
    """Summarize point coverage and pixel/normalized errors by region label."""

    names = sorted({item.region_name for item in localizations})
    summaries: list[RegionLocalizationSummary] = []
    for name in names:
        rows = [item for item in localizations if item.region_name == name]
        errors = [item.error_pixels for item in rows if item.error_pixels is not None]
        normalized = [item.error_normalized for item in rows if item.error_normalized is not None]
        summaries.append(
            RegionLocalizationSummary(
                region_name=name,
                annotated_points=len(rows),
                estimated_points=len(errors),
                coverage=len(errors) / len(rows) if rows else None,
                mean_error_pixels=statistics.mean(errors) if errors else None,
                median_error_pixels=statistics.median(errors) if errors else None,
                mean_error_normalized=statistics.mean(normalized) if normalized else None,
                median_error_normalized=statistics.median(normalized) if normalized else None,
            )
        )
    return summaries
