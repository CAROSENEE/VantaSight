"""Crosshair-to-region geometry; these values do not predict shot outcomes."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentMeasurement:
    """Pixel and frame-normalized distance from a reference to a region center."""

    region_name: str
    point_xy: tuple[float, float]
    crosshair_xy: tuple[float, float]
    distance_pixels: float
    distance_normalized: float


def center_crosshair(width: int, height: int) -> tuple[float, float]:
    """Return the geometric frame center in continuous pixel coordinates."""

    if width < 1 or height < 1:
        raise ValueError("Frame width and height must be positive")
    return width / 2.0, height / 2.0


def measure_alignment(
    point_xy: tuple[float, float],
    crosshair_xy: tuple[float, float],
    width: int,
    height: int,
    region_name: str = "unspecified",
) -> AlignmentMeasurement:
    """Measure Euclidean pixel distance and normalize by the frame diagonal."""

    if width < 1 or height < 1:
        raise ValueError("Frame width and height must be positive")
    coordinates = (*point_xy, *crosshair_xy)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Point and crosshair coordinates must be finite")
    for x, y in (point_xy, crosshair_xy):
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise ValueError("Point and crosshair coordinates must lie inside the frame")

    distance = math.hypot(point_xy[0] - crosshair_xy[0], point_xy[1] - crosshair_xy[1])
    diagonal = math.hypot(width, height)
    return AlignmentMeasurement(
        region_name=region_name,
        point_xy=point_xy,
        crosshair_xy=crosshair_xy,
        distance_pixels=distance,
        distance_normalized=distance / diagonal,
    )
