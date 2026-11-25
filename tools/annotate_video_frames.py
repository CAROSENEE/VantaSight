"""Interactively label visible opponents in selected offline video frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from vantasight.input import VideoFrame, VideoReader, file_sha256, frame_sha256


LABELS = {
    ord("y"): "opponent_visible",
    ord("n"): "no_opponent_visible",
    ord("u"): "uncertain",
}
BODY_REGIONS = {ord("1"): "head", ord("2"): "chest", ord("3"): "lower_torso", ord("4"): "legs"}
BODY_REGION_COLORS = {
    "head": (255, 190, 0),
    "chest": (0, 180, 255),
    "lower_torso": (0, 240, 240),
    "legs": (255, 80, 180),
}


def _parse_frames(value: str) -> list[int]:
    try:
        frames = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated frame indexes, e.g. 600,3000") from exc
    if not frames or any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError("Frame indexes must be zero or greater")
    if len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("Frame indexes must not repeat")
    return frames


def _read_shortlist(path: Path, video: Path) -> list[int]:
    """Read frame indexes only when the shortlist is for this exact source video."""
    video_hash = file_sha256(video)
    frames: list[int] = []
    seen: set[int] = set()
    required = {"source_video", "source_sha256", "frame_index"}
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Shortlist CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            if row.get("source_video") != video.name:
                raise ValueError(
                    f"Shortlist video on CSV line {line_number} does not match {video.name}"
                )
            source_hash = row.get("source_sha256")
            if not isinstance(source_hash, str) or source_hash.lower() != video_hash:
                raise ValueError(
                    f"Shortlist source fingerprint on CSV line {line_number} does not match {video}"
                )
            try:
                frame_index = int(row["frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame_index on shortlist CSV line {line_number}") from exc
            if frame_index < 0:
                raise ValueError(f"Negative frame_index on shortlist CSV line {line_number}")
            if frame_index in seen:
                raise ValueError(f"Duplicate frame_index {frame_index} in shortlist CSV")
            seen.add(frame_index)
            frames.append(frame_index)
    if not frames:
        raise ValueError(f"Shortlist CSV contains no frames: {path}")
    return frames


def _read_existing(output: Path) -> set[tuple[str, int]]:
    if not output.is_file():
        return set()
    completed: set[tuple[str, int]] = set()
    with output.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add((record["video"], int(record["frame_index"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid annotation at {output}:{line_number}: {exc}") from exc
    return completed


def _read_prefill(path: Path | None, video_name: str) -> dict[int, dict]:
    """Read provisional records as visual suggestions; never treat their labels as confirmed."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Prefill annotation file not found: {path}")
    suggestions: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or record.get("video") != video_name:
                continue
            frame_index = record.get("frame_index")
            if type(frame_index) is not int or frame_index < 0:
                raise ValueError(f"Invalid frame_index at {path}:{line_number}")
            if frame_index in suggestions:
                raise ValueError(f"Duplicate prefill for frame {frame_index} in {path}")
            boxes = record.get("opponent_boxes_xyxy")
            if not isinstance(boxes, list) or any(
                not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box)
                for box in boxes
            ):
                raise ValueError(f"Invalid provisional boxes at {path}:{line_number}")
            suggestions[frame_index] = record
    return suggestions


def _build_annotation_record(
    frame: VideoFrame,
    video_name: str,
    status: str,
    boxes: list[tuple[int, int, int, int]],
    region_points_by_box: dict[int, dict[str, tuple[int, int]]],
    suggestion: dict | None,
) -> dict:
    """Create a manually confirmed annotation row tied to its decoded source frame."""
    source_height, source_width = frame.pixels.shape[:2]
    return {
        "video": video_name,
        "frame_index": frame.index,
        "timestamp_seconds": round(frame.timestamp_seconds, 6),
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_sha256": frame_sha256(frame.pixels),
        "status": status,
        "opponent_boxes_xyxy": [list(box) for box in boxes],
        "reviewed_manually": True,
        "provisional_prefill_used": suggestion is not None,
        "opponent_body_regions_xy": [
            {
                "box_index": box_index,
                "points": {name: list(point) for name, point in points.items()},
            }
            for box_index, points in sorted(region_points_by_box.items())
            if points
        ],
    }


def _annotate_one(
    frame,
    video_name: str,
    max_width: int,
    max_height: int,
    suggestion: dict | None = None,
    review_position: int = 1,
    review_total: int = 1,
    keep_on_top: bool = False,
) -> dict | None:
    source_height, source_width = frame.pixels.shape[:2]
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    display_width = max(1, round(source_width * scale))
    display_height = max(1, round(source_height * scale))
    zoom = 1.0
    view_center_x = source_width / 2
    view_center_y = source_height / 2

    def view_bounds() -> tuple[int, int, int, int]:
        view_width = max(1, round(source_width / zoom))
        view_height = max(1, round(source_height / zoom))
        left = min(max(round(view_center_x - view_width / 2), 0), source_width - view_width)
        top = min(max(round(view_center_y - view_height / 2), 0), source_height - view_height)
        return left, top, left + view_width, top + view_height

    def to_source(x: int, y: int) -> tuple[int, int]:
        left, top, right, bottom = view_bounds()
        return (
            max(0, min(source_width, round(left + x * (right - left) / display_width))),
            max(0, min(source_height, round(top + y * (bottom - top) / display_height))),
        )

    def to_display(x: int, y: int) -> tuple[int, int]:
        left, top, right, bottom = view_bounds()
        return (
            round((x - left) * display_width / (right - left)),
            round((y - top) * display_height / (bottom - top)),
        )

    boxes: list[tuple[int, int, int, int]] = []
    if suggestion is not None:
        suggested_size = (suggestion.get("source_width"), suggestion.get("source_height"))
        if suggested_size != (source_width, source_height):
            raise ValueError(
                f"Prefill size {suggested_size[0]}x{suggested_size[1]} does not match "
                f"video {source_width}x{source_height} at frame {frame.index}"
            )
        for box in suggestion["opponent_boxes_xyxy"]:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
                raise ValueError(f"Invalid prefill box {box} at frame {frame.index}")
            boxes.append(tuple(box))
    region_points_by_box: dict[int, dict[str, tuple[int, int]]] = {}
    status: str | None = None
    notice: str | None = None
    active_region: str | None = None
    selected_box_index = 0
    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None
    pending_point: tuple[int, int] | None = None
    pan_start: tuple[int, int, float, float] | None = None
    redraw_required = True
    window_name = f"VantaSight annotation - {video_name} frame {frame.index}"

    def mouse_callback(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        nonlocal drag_start, drag_current, pending_point, notice
        nonlocal zoom, view_center_x, view_center_y, pan_start, redraw_required
        if event == cv2.EVENT_MOUSEWHEEL and y < display_height:
            left, top, right, bottom = view_bounds()
            anchor_x = left + x * (right - left) / display_width
            anchor_y = top + y * (bottom - top) / display_height
            wheel_delta = (flags >> 16) & 0xFFFF
            if wheel_delta > 32767:
                wheel_delta -= 65536
            next_zoom = min(8.0, zoom * 1.5) if wheel_delta > 0 else max(1.0, zoom / 1.5)
            if next_zoom != zoom:
                next_width = source_width / next_zoom
                next_height = source_height / next_zoom
                fraction_x = x / display_width
                fraction_y = y / display_height
                view_center_x = anchor_x + (0.5 - fraction_x) * next_width
                view_center_y = anchor_y + (0.5 - fraction_y) * next_height
                zoom = next_zoom
                redraw_required = True
            return
        if event == cv2.EVENT_RBUTTONDOWN and zoom > 1.0:
            pan_start = (x, y, view_center_x, view_center_y)
            return
        if event == cv2.EVENT_MOUSEMOVE and pan_start is not None:
            left, top, right, bottom = view_bounds()
            view_center_x = pan_start[2] - (x - pan_start[0]) * (right - left) / display_width
            view_center_y = pan_start[3] - (y - pan_start[1]) * (bottom - top) / display_height
            redraw_required = True
            return
        if event == cv2.EVENT_RBUTTONUP and pan_start is not None:
            pan_start = None
            return
        if event == cv2.EVENT_LBUTTONDOWN and y < display_height:
            location = (min(max(x, 0), display_width), min(max(y, 0), display_height))
            if active_region is None:
                drag_start = location
                drag_current = drag_start
                redraw_required = True
            else:
                pending_point = location
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_current = (min(max(x, 0), display_width), min(max(y, 0), display_height))
            redraw_required = True
        elif event == cv2.EVENT_LBUTTONUP and pending_point is not None and active_region is not None:
            if not (0 <= x < display_width and 0 <= y < display_height):
                notice = "click inside the video frame to mark a body point"
            elif selected_box_index >= len(boxes):
                notice = "draw an opponent box before marking body regions"
            else:
                source_point = to_source(x, y)
                x1, y1, x2, y2 = boxes[selected_box_index]
                if x1 <= source_point[0] <= x2 and y1 <= source_point[1] <= y2:
                    region_points_by_box.setdefault(selected_box_index, {})[active_region] = source_point
                    notice = f"{active_region} point set for box {selected_box_index + 1}"
                    redraw_required = True
                else:
                    notice = "body point must be inside the selected opponent box"
                    redraw_required = True
            pending_point = None
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            end = (min(max(x, 0), display_width), min(max(y, 0), display_height))
            left, right = sorted((drag_start[0], end[0]))
            top, bottom = sorted((drag_start[1], end[1]))
            if right - left >= 4 and bottom - top >= 4:
                box_left, box_top = to_source(left, top)
                box_right, box_bottom = to_source(right, bottom)
                box = (box_left, box_top, box_right, box_bottom)
                boxes.append(box)
                status = "opponent_visible"
                redraw_required = True
            drag_start = None
            drag_current = None
            redraw_required = True

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height + 180)
        if keep_on_top:
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error as exc:
                print(f"Could not keep annotation window on top; continuing normally: {exc}")
        cv2.setMouseCallback(window_name, mouse_callback)
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open the annotation window. Run this command in a desktop session "
            "with OpenCV GUI support. For browser-based manual review, run "
            "tools\\open_review_gallery.bat instead."
        ) from exc

    result: dict | None = None
    try:
        while True:
            if redraw_required:
                view_left, view_top, view_right, view_bottom = view_bounds()
                canvas = cv2.resize(
                    frame.pixels[view_top:view_bottom, view_left:view_right],
                    (display_width, display_height),
                )
                for box in boxes:
                    cv2.rectangle(canvas, to_display(box[0], box[1]), to_display(box[2], box[3]), (40, 230, 80), 2)
                if boxes:
                    x1, y1, x2, y2 = boxes[min(selected_box_index, len(boxes) - 1)]
                    cv2.rectangle(
                        canvas,
                        to_display(x1, y1),
                        to_display(x2, y2),
                        (255, 255, 0),
                        3,
                    )
                for box_index, points in region_points_by_box.items():
                    for region_name, (px, py) in points.items():
                        display_point = to_display(px, py)
                        color = BODY_REGION_COLORS[region_name]
                        cv2.drawMarker(canvas, display_point, color, cv2.MARKER_CROSS, 14, 2)
                        cv2.putText(canvas, region_name, (display_point[0] + 7, max(14, display_point[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                if drag_start is not None and drag_current is not None:
                    cv2.rectangle(canvas, drag_start, drag_current, (0, 220, 255), 2)

                panel = cv2.copyMakeBorder(
                    canvas, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
                )
                status_text = status or "not labeled"
                mode_text = f"box {selected_box_index + 1}" if active_region is None else f"{active_region} point for box {selected_box_index + 1}"
                notice_text = notice or "draw a box or mark optional body-region points"
                suggestion_text = (
                    f"PROVISIONAL SUGGESTIONS: {len(boxes)} box(es); inspect and confirm status"
                    if suggestion is not None
                    else ""
                )
                instructions = [
                    f"Review {review_position}/{review_total} | {video_name} | frame {frame.index} | "
                    f"{frame.timestamp_seconds:.3f}s | {source_width}x{source_height} | zoom {zoom:.1f}x",
                    f"Label: {status_text} | Opponent boxes: {len(boxes)} | Mode: {mode_text}",
                    f"Note: {notice_text}",
                    suggestion_text,
                    "Drag to add box. r: box mode | 1=head 2=chest 3=lower torso 4=legs, then click center inside selected box",
                    "[/] select box | y: confirm visible | n: none (clear boxes first) | u: unsure | z: undo | c: clear",
                    "Enter: save only after choosing y/n/u | q: quit | +/-: zoom | wheel: Windows zoom | right-drag: pan | 0: reset",
                ]
                for line_number, text in enumerate(instructions):
                    cv2.putText(
                        panel,
                        text,
                        (8, display_height + 18 + line_number * 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, panel)
                redraw_required = False
            key = cv2.waitKey(100) & 0xFF
            if key != 0xFF:
                redraw_required = True

            if key in LABELS:
                status = LABELS[key]
                notice = None
            elif key in BODY_REGIONS:
                active_region = BODY_REGIONS[key]
                notice = None
                if boxes:
                    selected_box_index = min(selected_box_index, len(boxes) - 1)
            elif key == ord("r"):
                active_region = None
                notice = None
            elif key == ord("0"):
                zoom = 1.0
                view_center_x = source_width / 2
                view_center_y = source_height / 2
            elif key in (ord("+"), ord("=")):
                zoom = min(8.0, zoom * 1.5)
            elif key in (ord("-"), ord("_")):
                zoom = max(1.0, zoom / 1.5)
            elif key == ord("]"):
                if boxes:
                    selected_box_index = (selected_box_index + 1) % len(boxes)
                    active_region = None
                    notice = None
            elif key == ord("["):
                if boxes:
                    selected_box_index = (selected_box_index - 1) % len(boxes)
                    active_region = None
                    notice = None
            elif key == ord("z"):
                if boxes:
                    removed_index = len(boxes) - 1
                    boxes.pop()
                    region_points_by_box.pop(removed_index, None)
                    selected_box_index = min(selected_box_index, max(0, len(boxes) - 1))
                if not boxes and status == "opponent_visible":
                    status = None
            elif key == ord("c"):
                boxes.clear()
                region_points_by_box.clear()
                selected_box_index = 0
                active_region = None
                status = None
                notice = None
            elif key in (10, 13):
                if status is None:
                    print(f"Frame {frame.index}: choose y, n, or u before saving.")
                    continue
                if status == "opponent_visible" and not boxes:
                    print(f"Frame {frame.index}: draw at least one opponent box, or choose u.")
                    continue
                if status == "no_opponent_visible" and boxes:
                    print(f"Frame {frame.index}: clear boxes before labeling no opponent.")
                    continue
                result = _build_annotation_record(
                    frame, video_name, status, boxes, region_points_by_box, suggestion
                )
                break
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(window_name)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review selected recorded-video frames and label visible opponents offline."
    )
    parser.add_argument("video", type=Path, help="Path to a recorded video")
    frame_group = parser.add_mutually_exclusive_group(required=True)
    frame_group.add_argument(
        "--frames",
        type=_parse_frames,
        help="Comma-separated zero-based frame indexes",
    )
    frame_group.add_argument(
        "--shortlist-csv",
        type=Path,
        help="Use frame indexes from a shortlist CSV, validating its source-video fingerprint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/manual_enemy_boxes.jsonl"),
        help="Local JSON Lines output file (default: data/annotations/manual_enemy_boxes.jsonl)",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum displayed frame width")
    parser.add_argument("--max-height", type=int, default=720, help="Maximum displayed frame height")
    parser.add_argument(
        "--prefill",
        type=Path,
        help="Optional JSONL with provisional boxes to display for manual review; labels are not pre-confirmed",
    )
    parser.add_argument(
        "--keep-on-top",
        action="store_true",
        help="Keep the annotation window above other windows so it is easy to find",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected video frames without opening a window or writing annotations",
    )
    args = parser.parse_args()

    if args.max_width < 1 or args.max_height < 1:
        parser.error("--max-width and --max-height must be positive")

    video = args.video.expanduser()
    output = args.output.expanduser()
    try:
        frames = args.frames or _read_shortlist(args.shortlist_csv.expanduser(), video)
        if args.dry_run:
            with VideoReader(video) as reader:
                assert reader.info is not None
                print(f"Video: {reader.info.path} ({reader.info.width}x{reader.info.height})")
                print(f"Review frames: {len(frames)}")
                for frame_index in frames:
                    frame = reader.read_frame(frame_index)
                    print(f"  frame {frame.index} at {frame.timestamp_seconds:.3f}s: decodable")
            print("Dry run complete; no annotation window was opened and no labels were written.")
            return 0

        completed = _read_existing(output)
        suggestions = _read_prefill(args.prefill.expanduser() if args.prefill else None, video.name)
        with VideoReader(video) as reader:
            for review_position, frame_index in enumerate(frames, start=1):
                if (video.name, frame_index) in completed:
                    print(f"Review {review_position}/{len(frames)}: skipping saved frame {frame_index}.")
                    continue
                frame = reader.read_frame(frame_index)
                record = _annotate_one(
                    frame,
                    video.name,
                    args.max_width,
                    args.max_height,
                    suggestions.get(frame_index),
                    review_position,
                    len(frames),
                    args.keep_on_top,
                )
                if record is None:
                    print("Stopped; saved annotations are preserved.")
                    break
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("a", encoding="utf-8", newline="\n") as destination:
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    destination.flush()
                completed.add((video.name, frame_index))
                print(
                    f"Review {review_position}/{len(frames)}: saved frame {frame_index}: "
                    f"{record['status']} ({len(record['opponent_boxes_xyxy'])} boxes)"
                )
        print(f"Annotations: {output.resolve()}")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError, csv.Error, cv2.error) as exc:
        parser.exit(2, f"annotation tool: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
