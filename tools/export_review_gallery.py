"""Export a fingerprint-checked shortlist as a local offline annotation gallery."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import cv2

from vantasight.input import VideoReader, file_sha256, frame_sha256


def _read_shortlist(path: Path, video: Path) -> list[dict[str, object]]:
    video_hash = file_sha256(video)
    required = {"source_video", "source_sha256", "frame_index"}
    records: list[dict[str, object]] = []
    seen: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Shortlist CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            if row.get("source_video") != video.name:
                raise ValueError(f"Shortlist video on CSV line {line_number} does not match {video.name}")
            source_hash = row.get("source_sha256")
            if not isinstance(source_hash, str) or source_hash.lower() != video_hash:
                raise ValueError(f"Shortlist source fingerprint on CSV line {line_number} does not match {video}")
            try:
                frame_index = int(row["frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame_index on shortlist CSV line {line_number}") from exc
            if frame_index < 0 or frame_index in seen:
                raise ValueError(f"Invalid or duplicate frame_index {frame_index} on CSV line {line_number}")
            seen.add(frame_index)
            records.append({
                "frame_index": frame_index,
                "shortlist_timestamp_seconds": row.get("timestamp_seconds", ""),
            })
    if not records:
        raise ValueError(f"Shortlist CSV contains no frames: {path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export shortlist frames to a local browser-based manual review gallery."
    )
    parser.add_argument("video", type=Path, help="Recorded source video")
    parser.add_argument(
        "--shortlist-csv", type=Path, required=True, help="Fingerprint-bound frame shortlist CSV"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/previews/review_gallery")
    )
    args = parser.parse_args()

    video = args.video.expanduser()
    shortlist = args.shortlist_csv.expanduser()
    output_dir = args.output_dir.expanduser()
    try:
        if not video.is_file():
            raise FileNotFoundError(f"Video does not exist: {video}")
        if not shortlist.is_file():
            raise FileNotFoundError(f"Shortlist CSV does not exist: {shortlist}")
        initial_video_hash = file_sha256(video)
        selected = _read_shortlist(shortlist, video)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[dict] = []
        with VideoReader(video) as reader:
            assert reader.info is not None
            for record in selected:
                frame_index = int(record["frame_index"])
                frame = reader.read_frame(frame_index)
                image_file = f"frame_{frame_index:06d}.jpg"
                image_path = output_dir / image_file
                if not cv2.imwrite(
                    str(image_path), frame.pixels, [cv2.IMWRITE_JPEG_QUALITY, 94]
                ):
                    raise OSError(f"Could not write review frame: {image_path}")
                exported.append({
                    "frame_index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "image_file": image_file,
                    "source_frame_sha256": frame_sha256(frame.pixels),
                })
            final_video_hash = file_sha256(reader.info.path)
            if final_video_hash != initial_video_hash:
                raise RuntimeError("Source video changed during review-frame export")
            manifest = {
                "source_video": reader.info.path.name,
                "source_sha256": final_video_hash,
                "source_width": reader.info.width,
                "source_height": reader.info.height,
                "frames": exported,
                "labels_saved": False,
            }

        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        gallery_data = base64.b64encode(
            json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        template_path = Path(__file__).with_name("review_gallery_template.html")
        template = template_path.read_text(encoding="utf-8")
        if "__GALLERY_DATA__" not in template:
            raise RuntimeError(f"Gallery template has no data placeholder: {template_path}")
        (output_dir / "index.html").write_text(
            template.replace("__GALLERY_DATA__", gallery_data), encoding="utf-8"
        )
        print(f"Exported {len(exported)} original-resolution review frames to {output_dir.resolve()}")
        print(f"Open this local gallery: {(output_dir / 'index.html').resolve()}")
        print("The gallery has no labels yet. Labels are created only when a reviewer saves them.")
        return 0
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        parser.exit(2, f"review gallery: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
