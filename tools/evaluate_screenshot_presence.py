"""Evaluate person-candidate presence against locally labeled screenshots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vantasight.analytics import PresenceCounts, calculate_presence_metrics
from vantasight.cli import IMAGE_SUFFIXES, _load_excluded_regions
from vantasight.input import load_image
from vantasight.vision import TorchvisionPersonDetector


def _parse_bool(value: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid visible_in_game_enemy value on labels CSV line {line_number}: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure screenshot-level person-candidate presence against manual labels."
    )
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/screenshot_presence.csv"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/input/screenshots"))
    parser.add_argument("--regions-file", type=Path, default=Path("data/annotations/source_regions.json"))
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--output", type=Path, default=Path("reports/screenshot_presence_eval.csv"))
    args = parser.parse_args()

    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if args.max_side < 1:
        parser.error("--max-side must be positive")

    labels_path = args.labels.expanduser()
    input_dir = args.input_dir.expanduser()
    regions_path = args.regions_file.expanduser()
    output = args.output.expanduser()
    try:
        with labels_path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {"asset_id", "visible_in_game_enemy", "user_labeled"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Labels CSV must contain columns: {', '.join(sorted(required))}")
            records = list(reader)
        if not records:
            raise ValueError("Labels CSV contains no screenshot rows")

        detector = TorchvisionPersonDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
        )
        rows: list[dict] = []
        tp = fp = tn = fn = 0
        seen_ids: set[str] = set()
        for line_number, record in enumerate(records, start=2):
            asset_id = (record.get("asset_id") or "").strip()
            if not asset_id or asset_id in seen_ids:
                raise ValueError(f"Missing or duplicate asset_id on labels CSV line {line_number}")
            seen_ids.add(asset_id)
            if (record.get("user_labeled") or "").strip().lower() not in {"true", "1", "yes"}:
                raise ValueError(f"Screenshot {asset_id} is not marked user_labeled=true")
            visible = _parse_bool(record.get("visible_in_game_enemy") or "", line_number)
            candidates = [
                path for path in input_dir.glob(f"{asset_id}.*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one supported screenshot named {asset_id}.* in {input_dir}; "
                    f"found {len(candidates)}"
                )
            image_path = candidates[0]
            image = load_image(image_path)
            height, width = image.pixels.shape[:2]
            exclusions = _load_excluded_regions(regions_path, image_path.name)
            detector.excluded_regions = tuple(exclusions)
            detections = detector.detect(image.pixels)
            predicted = bool(detections)
            if predicted and visible:
                tp += 1
            elif predicted and not visible:
                fp += 1
            elif not predicted and visible:
                fn += 1
            else:
                tn += 1
            rows.append(
                {
                    "asset_id": asset_id,
                    "width": width,
                    "height": height,
                    "ground_truth_enemy_visible": str(visible).lower(),
                    "predicted_person_candidate_present": str(predicted).lower(),
                    "candidate_count": len(detections),
                    "max_candidate_confidence": max((item.confidence for item in detections), default=""),
                    "confidence_threshold": args.confidence_threshold,
                    "device": str(detector.device),
                }
            )

        counts = PresenceCounts(tp, fp, tn, fn)
        metrics = calculate_presence_metrics(counts)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as destination:
            fieldnames = list(rows[0])
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Device: {detector.device}")
        print(f"User-labeled screenshots: {counts.total}")
        print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
        for name, value in (
            ("Accuracy", metrics.accuracy),
            ("Precision", metrics.precision),
            ("Recall", metrics.recall),
            ("Specificity", metrics.specificity),
            ("F1", metrics.f1),
        ):
            print(f"{name}: {value:.3f}" if value is not None else f"{name}: n/a")
        print(f"Per-screenshot results: {output.resolve()}")
        print("These exploratory presence metrics are limited to the supplied labels and threshold.")
        print("Person-candidate presence is not enemy identity, box localization, or gameplay accuracy.")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"screenshot presence evaluation: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
