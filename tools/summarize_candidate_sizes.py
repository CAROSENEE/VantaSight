"""Summarize predicted person-candidate box sizes by sampled video frame."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vantasight.dashboard import load_tracking_report


SIZE_GROUPS = (
    ("tiny", 0.0, 2.0),
    ("small", 2.0, 6.0),
    ("medium", 6.0, 16.0),
    ("large", 16.0, float("inf")),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize predicted person-candidate box heights in a tracking report."
    )
    parser.add_argument("report", type=Path, help="Offline tracking report JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/candidate_size_by_frame.csv"),
        help="CSV output path (default: reports/candidate_size_by_frame.csv)",
    )
    args = parser.parse_args()

    try:
        report = load_tracking_report(args.report.expanduser())
        source = report["source"]
        source_height = int(source["height"])
        source_name = str(source["file_name"])
        source_sha256 = str(source.get("source_sha256", ""))
        frames = report["frames"]
        if source_height < 1:
            raise ValueError("Tracking report source height must be positive")

        fieldnames = [
            "source_video",
            "source_sha256",
            "frame_index",
            "timestamp_seconds",
            "candidate_count",
            "tiny_candidates_lt_2pct",
            "small_candidates_2_6pct",
            "medium_candidates_6_16pct",
            "large_candidates_ge_16pct",
            "smallest_candidate_height_pct",
        ]
        rows: list[dict[str, object]] = []
        total_group_counts = {name: 0 for name, _, _ in SIZE_GROUPS}
        total_candidates = 0
        empty_candidate_frames = 0

        for frame in frames:
            detections = frame["detections"]
            counts = {name: 0 for name, _, _ in SIZE_GROUPS}
            candidate_heights: list[float] = []
            for detection in detections:
                _, y1, _, y2 = detection["box_xyxy"]
                height_pct = 100.0 * (float(y2) - float(y1)) / source_height
                candidate_heights.append(height_pct)
                for name, lower, upper in SIZE_GROUPS:
                    if lower <= height_pct < upper:
                        counts[name] += 1
                        total_group_counts[name] += 1
                        break

            total_candidates += len(detections)
            empty_candidate_frames += not detections
            rows.append(
                {
                    "source_video": source_name,
                    "source_sha256": source_sha256,
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": f"{float(frame['timestamp_seconds']):.6f}",
                    "candidate_count": len(detections),
                    "tiny_candidates_lt_2pct": counts["tiny"],
                    "small_candidates_2_6pct": counts["small"],
                    "medium_candidates_6_16pct": counts["medium"],
                    "large_candidates_ge_16pct": counts["large"],
                    "smallest_candidate_height_pct": (
                        f"{min(candidate_heights):.6f}" if candidate_heights else ""
                    ),
                }
            )

        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"candidate size summary: error: {exc}\n")

    print(f"Source video: {source_name}")
    print(f"Sampled frames: {len(frames)}; person candidates: {total_candidates}")
    print(f"Frames with no candidate: {empty_candidate_frames}")
    print("Candidate box height groups: " + ", ".join(
        f"{name}={total_group_counts[name]}" for name, _, _ in SIZE_GROUPS
    ))
    print(f"CSV: {output.resolve()}")
    print("These are model candidate sizes, not verified enemy-size recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
