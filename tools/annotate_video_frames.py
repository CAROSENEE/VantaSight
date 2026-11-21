"""Interactively label visible opponents in selected offline video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from vantasight.input import VideoReader


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


def _draw_box(image, box: tuple[int, int, int, int], scale_x: float, scale_y: float) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(
        image,
        (round(x1 * scale_x), round(y1 * scale_y)),
        (round(x2 * scale_x), round(y2 * scale_y)),
        (40, 230, 80),
        2,
    )


def _annotate_one(frame, video_name: str, max_width: int, max_height: int) -> dict | None:
    source_height, source_width = frame.pixels.shape[:2]
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    display_width = max(1, round(source_width * scale))
    display_height = max(1, round(source_height * scale))
    scale_x = display_width / source_width
    scale_y = display_height / source_height

    boxes: list[tuple[int, int, int, int]] = []
    region_points_by_box: dict[int, dict[str, tuple[int, int]]] = {}
    status: str | None = None
    notice: str | None = None
    active_region: str | None = None
    selected_box_index = 0
    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None
    pending_point: tuple[int, int] | None = None
    window_name = f"VantaSight annotation - {video_name} frame {frame.index}"

    def mouse_callback(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        nonlocal drag_start, drag_current, pending_point, notice
        if event == cv2.EVENT_LBUTTONDOWN and y < display_height:
            location = (min(max(x, 0), display_width), min(max(y, 0), display_height))
            if active_region is None:
                drag_start = location
                drag_current = drag_start
            else:
                pending_point = location
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_current = (min(max(x, 0), display_width), min(max(y, 0), display_height))
        elif event == cv2.EVENT_LBUTTONUP and pending_point is not None and active_region is not None:
            if not (0 <= x < display_width and 0 <= y < display_height):
                notice = "click inside the video frame to mark a body point"
            elif selected_box_index >= len(boxes):
                notice = "draw an opponent box before marking body regions"
            else:
                source_point = (
                    max(0, min(source_width, round(x / scale_x))),
                    max(0, min(source_height, round(y / scale_y))),
                )
                x1, y1, x2, y2 = boxes[selected_box_index]
                if x1 <= source_point[0] <= x2 and y1 <= source_point[1] <= y2:
                    region_points_by_box.setdefault(selected_box_index, {})[active_region] = source_point
                    notice = f"{active_region} point set for box {selected_box_index + 1}"
                else:
                    notice = "body point must be inside the selected opponent box"
            pending_point = None
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            end = (min(max(x, 0), display_width), min(max(y, 0), display_height))
            left, right = sorted((drag_start[0], end[0]))
            top, bottom = sorted((drag_start[1], end[1]))
            if right - left >= 4 and bottom - top >= 4:
                box = (
                    max(0, min(source_width, round(left / scale_x))),
                    max(0, min(source_height, round(top / scale_y))),
                    max(0, min(source_width, round(right / scale_x))),
                    max(0, min(source_height, round(bottom / scale_y))),
                )
                boxes.append(box)
                status = "opponent_visible"
            drag_start = None
            drag_current = None

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height + 110)
        cv2.setMouseCallback(window_name, mouse_callback)
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open the annotation window. Run this command in a desktop session "
            "with OpenCV GUI support."
        ) from exc

    result: dict | None = None
    try:
        while True:
            canvas = cv2.resize(frame.pixels, (display_width, display_height))
            for box in boxes:
                _draw_box(canvas, box, scale_x, scale_y)
            if boxes:
                x1, y1, x2, y2 = boxes[min(selected_box_index, len(boxes) - 1)]
                cv2.rectangle(
                    canvas,
                    (round(x1 * scale_x), round(y1 * scale_y)),
                    (round(x2 * scale_x), round(y2 * scale_y)),
                    (255, 255, 0),
                    3,
                )
            for box_index, points in region_points_by_box.items():
                for region_name, (px, py) in points.items():
                    display_point = (round(px * scale_x), round(py * scale_y))
                    color = BODY_REGION_COLORS[region_name]
                    cv2.drawMarker(canvas, display_point, color, cv2.MARKER_CROSS, 14, 2)
                    cv2.putText(canvas, region_name, (display_point[0] + 7, max(14, display_point[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            if drag_start is not None and drag_current is not None:
                cv2.rectangle(canvas, drag_start, drag_current, (0, 220, 255), 2)

            panel = cv2.copyMakeBorder(
                canvas, 0, 110, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
            )
            status_text = status or "not labeled"
            mode_text = f"box {selected_box_index + 1}" if active_region is None else f"{active_region} point for box {selected_box_index + 1}"
            notice_text = notice or "draw a box or mark optional body-region points"
            instructions = [
                f"{video_name} | frame {frame.index} | {frame.timestamp_seconds:.3f}s | {source_width}x{source_height}",
                f"Label: {status_text} | Opponent boxes: {len(boxes)} | Mode: {mode_text}",
                f"Note: {notice_text}",
                "Drag to add box. r: box mode | 1=head 2=chest 3=lower torso 4=legs, then click center inside selected box",
                "[/] select opponent box | y: opponent visible | n: none | u: unsure | z: undo box | c: clear | Enter: save | q: quit",
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
            key = cv2.waitKey(25) & 0xFF

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
                result = {
                    "video": video_name,
                    "frame_index": frame.index,
                    "timestamp_seconds": round(frame.timestamp_seconds, 6),
                    "source_width": source_width,
                    "source_height": source_height,
                    "status": status,
                    "opponent_boxes_xyxy": [list(box) for box in boxes],
                    "opponent_body_regions_xy": [
                        {
                            "box_index": box_index,
                            "points": {name: list(point) for name, point in points.items()},
                        }
                        for box_index, points in sorted(region_points_by_box.items())
                        if points
                    ],
                }
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
    parser.add_argument("--frames", type=_parse_frames, required=True, help="Comma-separated zero-based frame indexes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/manual_enemy_boxes.jsonl"),
        help="Local JSON Lines output file (default: data/annotations/manual_enemy_boxes.jsonl)",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum displayed frame width")
    parser.add_argument("--max-height", type=int, default=720, help="Maximum displayed frame height")
    args = parser.parse_args()

    if args.max_width < 1 or args.max_height < 1:
        parser.error("--max-width and --max-height must be positive")

    video = args.video.expanduser()
    output = args.output.expanduser()
    try:
        completed = _read_existing(output)
        with VideoReader(video) as reader:
            for frame_index in args.frames:
                if (video.name, frame_index) in completed:
                    print(f"Skipping already labeled frame {frame_index}.")
                    continue
                frame = reader.read_frame(frame_index)
                record = _annotate_one(frame, video.name, args.max_width, args.max_height)
                if record is None:
                    print("Stopped; saved annotations are preserved.")
                    break
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("a", encoding="utf-8", newline="\n") as destination:
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    destination.flush()
                completed.add((video.name, frame_index))
                print(f"Saved frame {frame_index}: {record['status']} ({len(record['opponent_boxes_xyxy'])} boxes)")
        print(f"Annotations: {output.resolve()}")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f"annotation tool: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
