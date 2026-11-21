"""Browse a saved VantaSight analysis report and its source video offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from vantasight.dashboard import load_tracking_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a saved offline VantaSight tracking report.")
    parser.add_argument("report", type=Path, nargs="?", default=Path("reports/tracks.json"))
    parser.add_argument("--video", type=Path, help="Source video; defaults to data/input/videos/<report source>")
    args = parser.parse_args()

    report_path = args.report.expanduser()
    try:
        report = load_tracking_report(report_path)
        default_video = Path("data/input/videos") / str(report["source"].get("file_name", ""))
        video_path = (args.video or default_video).expanduser()
        if not video_path.is_file():
            raise ValueError(f"Source video does not exist: {video_path}; pass it with --video")
        frames = report["frames"]
        if not frames:
            raise ValueError("This report has no sampled frames to display")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {video_path}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"dashboard: error: {exc}\n")

    root = tk.Tk()
    root.title(f"VantaSight Offline Dashboard — {report['source'].get('file_name', video_path.name)}")
    root.geometry("1280x800")
    root.minsize(900, 600)

    summary = report["summary"]
    source = report["source"]
    header = ttk.Frame(root, padding=10)
    header.pack(fill="x")
    ttk.Label(header, text="VantaSight — Offline Analysis", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    summary_text = (
        f"Source: {source.get('file_name')}  |  {source.get('width')}×{source.get('height')}  |  "
        f"{source.get('fps', 0):.2f} FPS    •    Sampled frames: {summary.get('sampled_frames', 0)}    •    "
        f"Person candidates: {summary.get('candidate_detections', 0)}    •    "
        f"Mean detector latency: {summary.get('detector_latency_mean_ms', 0):.1f} ms"
    )
    if summary.get("pose_latency_mean_ms") is not None:
        summary_text += f"    •    Mean pose latency: {summary['pose_latency_mean_ms']:.1f} ms"
    ttk.Label(header, text=summary_text).pack(anchor="w", pady=(5, 0))
    ttk.Label(
        header,
        text="Offline COCO person candidates only. Boxes and temporary IDs do not verify enemy, team, or player identity.",
        foreground="#9a4d00",
    ).pack(anchor="w", pady=(4, 0))

    body = ttk.Panedwindow(root, orient="horizontal")
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    image_panel = ttk.Frame(body, padding=4)
    side_panel = ttk.Frame(body, padding=4, width=300)
    body.add(image_panel, weight=4)
    body.add(side_panel, weight=1)
    canvas = tk.Canvas(image_panel, background="#141820", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    ttk.Label(side_panel, text="Sampled frames", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    frame_list = tk.Listbox(side_panel, exportselection=False, height=18)
    frame_list.pack(fill="both", expand=True, pady=(5, 10))
    for frame in frames:
        frame_list.insert(
            "end",
            f"{frame.get('timestamp_seconds', 0):7.3f}s  |  frame {frame.get('frame_index')}  |  "
            f"{frame.get('detection_count', len(frame['detections']))} candidates",
        )
    detail = tk.Text(side_panel, height=10, wrap="word", state="disabled")
    detail.pack(fill="x")

    current_index = 0
    displayed_photo: tk.PhotoImage | None = None

    def show_frame(index: int) -> None:
        nonlocal current_index, displayed_photo
        current_index = max(0, min(index, len(frames) - 1))
        frame_record = frames[current_index]
        frame_index = int(frame_record["frame_index"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
        if not success:
            canvas.delete("all")
            canvas.create_text(20, 20, anchor="nw", fill="white", text=f"Could not decode source frame {frame_index}")
            return

        for item in frame_record["detections"]:
            x1, y1, x2, y2 = (int(value) for value in item["box_xyxy"])
            color = (0, 220, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            label = f"ID {item.get('track_id', '?')}  {item.get('detector_score', 0):.2f}"
            cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            for region in item.get("body_regions", []):
                rx, ry = (int(value) for value in region["center_xy"])
                region_color = (255, 190, 0) if region["name"] == "head" else (0, 190, 255)
                cv2.drawMarker(image, (rx, ry), region_color, cv2.MARKER_CROSS, 16, 2)
                cv2.putText(
                    image,
                    region["name"],
                    (rx + 6, max(16, ry - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    region_color,
                    1,
                    cv2.LINE_AA,
                )

        available_width = max(1, canvas.winfo_width())
        available_height = max(1, canvas.winfo_height())
        scale = min(available_width / image.shape[1], available_height / image.shape[0])
        display_width = max(1, round(image.shape[1] * scale))
        display_height = max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        encoded_ok, png = cv2.imencode(".png", resized)
        if not encoded_ok:
            raise RuntimeError("Could not encode the sampled video frame for dashboard display")
        displayed_photo = tk.PhotoImage(data=png.tobytes(), format="png")
        canvas.delete("all")
        canvas.create_image(available_width // 2, available_height // 2, image=displayed_photo, anchor="center")
        frame_list.selection_clear(0, "end")
        frame_list.selection_set(current_index)
        frame_list.see(current_index)
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        detail.insert("end", f"Frame: {frame_index}\nTimestamp: {frame_record.get('timestamp_seconds', 0):.3f}s\n")
        detail.insert("end", f"Person candidates: {len(frame_record['detections'])}\n")
        for item in frame_record["detections"]:
            velocity = item.get("velocity_xy_pixels_per_second")
            speed = "unavailable" if velocity is None else f"({velocity[0]:.1f}, {velocity[1]:.1f}) px/s"
            detail.insert("end", f"\nID {item.get('track_id', '?')} — score {item.get('detector_score', 0):.3f}\n")
            detail.insert("end", f"Box: {item.get('box_xyxy')}\nVelocity: {speed}\n")
            if "pose_status" in item:
                detail.insert("end", f"Pose association: {item['pose_status']}\n")
            if "pose_person_score" in item:
                detail.insert("end", f"Pose model person score: {item['pose_person_score']:.3f}\n")
            if item.get("pose_status") == "matched" and not item.get("body_regions"):
                detail.insert("end", "No body-region centers were returned for this pose.\n")
            for region in item.get("body_regions", []):
                detail.insert(
                    "end",
                    f"{region['name']} center: {region['center_xy']} | mean raw keypoint score: "
                    f"{region['mean_keypoint_score']:.3f}\n",
                )
                alignment = region.get("alignment")
                if alignment:
                    detail.insert(
                        "end",
                        f"  Crosshair offset: {alignment['distance_pixels']:.1f}px | "
                        f"normalized {alignment['distance_normalized']:.5f}\n",
                    )
        detail.configure(state="disabled")

    def on_select(_event: object = None) -> None:
        selection = frame_list.curselection()
        if selection:
            show_frame(selection[0])

    frame_list.bind("<<ListboxSelect>>", on_select)
    controls = ttk.Frame(image_panel, padding=(0, 8, 0, 0))
    controls.pack(fill="x")
    ttk.Button(controls, text="◀ Previous", command=lambda: show_frame(current_index - 1)).pack(side="left")
    ttk.Button(controls, text="Next ▶", command=lambda: show_frame(current_index + 1)).pack(side="left", padx=8)
    ttk.Label(controls, text=f"Report: {report_path}").pack(side="right")

    root.update_idletasks()
    show_frame(0)
    root.protocol("WM_DELETE_WINDOW", lambda: (capture.release(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
