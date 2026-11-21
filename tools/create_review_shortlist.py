"""Select model-output frames for later manual review without creating labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from vantasight.dashboard import load_tracking_report


def _frame_candidate_heights(frame: dict[str, Any], source_height: int) -> list[float]:
    heights: list[float] = []
    for detection in frame["detections"]:
        _, y1, _, y2 = detection["box_xyxy"]
        heights.append(100.0 * (float(y2) - float(y1)) / source_height)
    return heights


def select_review_rows(report: dict[str, Any], bins: int) -> list[dict[str, object]]:
    """Choose the smallest candidate and an empty-candidate frame per time bin."""

    if bins < 1:
        raise ValueError("--bins must be at least 1")
    source = report["source"]
    source_height = int(source["height"])
    if source_height < 1:
        raise ValueError("Tracking report source height must be positive")
    frames = report["frames"]
    if not frames:
        raise ValueError("Tracking report contains no sampled frames")

    rows: list[dict[str, object]] = []
    bin_count = min(bins, len(frames))
    for bin_index in range(bin_count):
        start = bin_index * len(frames) // bin_count
        stop = (bin_index + 1) * len(frames) // bin_count
        chunk = frames[start:stop]
        nonempty: list[tuple[float, dict[str, Any]]] = []
        empty: list[tuple[int, dict[str, Any]]] = []
        for chunk_index, frame in enumerate(chunk):
            heights = _frame_candidate_heights(frame, source_height)
            if heights:
                nonempty.append((min(heights), frame))
            else:
                empty.append((chunk_index, frame))

        if nonempty:
            height, frame = min(
                nonempty,
                key=lambda item: (item[0], len(item[1]["detections"]), item[1]["frame_index"]),
            )
            rows.append(
                {
                    "source_video": source["file_name"],
                    "source_sha256": source.get("source_sha256", ""),
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": f"{float(frame['timestamp_seconds']):.6f}",
                    "candidate_count": len(frame["detections"]),
                    "smallest_candidate_height_pct": f"{height:.6f}",
                    "selection_reason": "smallest_model_candidate_in_time_bin",
                    "time_bin": f"{bin_index + 1}/{bin_count}",
                }
            )

        if empty:
            midpoint = (len(chunk) - 1) / 2
            _, frame = min(
                empty,
                key=lambda indexed_frame: (
                    abs(indexed_frame[0] - midpoint),
                    indexed_frame[1]["frame_index"],
                ),
            )
            rows.append(
                {
                    "source_video": source["file_name"],
                    "source_sha256": source.get("source_sha256", ""),
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": f"{float(frame['timestamp_seconds']):.6f}",
                    "candidate_count": 0,
                    "smallest_candidate_height_pct": "",
                    "selection_reason": "no_model_candidates_in_frame",
                    "time_bin": f"{bin_index + 1}/{bin_count}",
                }
            )

    return sorted(rows, key=lambda row: (float(row["timestamp_seconds"]), str(row["selection_reason"])))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select model-candidate and zero-candidate frames for later manual review."
    )
    parser.add_argument("report", type=Path, help="Validated offline tracking report JSON")
    parser.add_argument("--bins", type=int, default=6, help="Number of time bins (default: 6)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/manual_review_shortlist.csv"),
        help="CSV output path (default: reports/manual_review_shortlist.csv)",
    )
    args = parser.parse_args()

    try:
        report = load_tracking_report(args.report.expanduser())
        rows = select_review_rows(report, args.bins)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_video",
            "source_sha256",
            "frame_index",
            "timestamp_seconds",
            "candidate_count",
            "smallest_candidate_height_pct",
            "selection_reason",
            "time_bin",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"review shortlist: error: {exc}\n")

    print(f"Selected {len(rows)} frames from {report['source']['file_name']}")
    print(f"CSV: {output.resolve()}")
    print("Rows are model-output review suggestions, not enemy labels or evaluation results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
