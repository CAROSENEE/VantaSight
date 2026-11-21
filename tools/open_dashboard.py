"""Browse a saved VantaSight analysis report and its source video offline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from vantasight.dashboard import load_tracking_report, summarize_alignment_errors
from vantasight.dashboard.evaluation_summary import load_detector_evaluation, load_pose_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a saved offline VantaSight tracking report.")
    parser.add_argument("report", type=Path, nargs="?", default=Path("reports/tracks.json"))
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video", type=Path, help="Source video; defaults to data/input/videos/<report source>")
    source_group.add_argument("--image", type=Path, help="Source screenshot; defaults to data/input/screenshots/<report source>")
    parser.add_argument("--detector-evaluation", type=Path, help="Optional manual-label detector CSV")
    parser.add_argument("--pose-evaluation", type=Path, help="Optional manual-point pose CSV")
    args = parser.parse_args(argv)

    report_path = args.report.expanduser()
    try:
        report = load_tracking_report(report_path)
        source_is_image = report.get("analysis_type") == "offline_single_image_person_candidate_analysis"
        if source_is_image and args.video:
            raise ValueError("This is a screenshot report; pass its source with --image")
        if not source_is_image and args.image:
            raise ValueError("This is a video report; pass its source with --video")
        default_root = Path("data/input/screenshots" if source_is_image else "data/input/videos")
        default_source = default_root / str(report["source"].get("file_name", ""))
        source_path = (args.image if source_is_image else args.video) or default_source
        source_path = source_path.expanduser()
        if not source_path.is_file():
            raise ValueError(f"Source media does not exist: {source_path}; pass it with --image or --video")
        frames = report["frames"]
        if not frames:
            raise ValueError("This report has no sampled frames to display")
        detector_eval_path = args.detector_evaluation or Path("reports/detector_evaluation.csv")
        pose_eval_path = args.pose_evaluation or Path("reports/pose_region_evaluation.csv")
        detector_evaluation = load_detector_evaluation(detector_eval_path, report["source"]["file_name"])
        pose_evaluation = load_pose_evaluation(pose_eval_path, report["source"]["file_name"])
        source_image = None
        capture = None
        if source_is_image:
            source_image = cv2.imread(str(source_path))
            if source_image is None:
                raise ValueError(f"Cannot decode source screenshot: {source_path}")
        else:
            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open source video: {source_path}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"dashboard: error: {exc}\n")

    root = tk.Tk()
    root.title(f"VantaSight Offline Dashboard — {report['source'].get('file_name', source_path.name)}")
    root.geometry("1280x800")
    root.minsize(900, 600)

    summary = report["summary"]
    source = report["source"]
    alignment_summaries = summarize_alignment_errors(report)
    source_rate = "Screenshot" if source_is_image else f"{source.get('fps', 0):.2f} FPS"
    header = ttk.Frame(root, padding=10)
    header.pack(fill="x")
    ttk.Label(header, text="VantaSight — Offline Analysis", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    summary_text = (
        f"Source: {source.get('file_name')}  |  {source.get('width')}×{source.get('height')}  |  "
        f"{source_rate}    •    Sampled frames: {summary.get('sampled_frames', 0)}    •    "
        f"Person candidates: {summary.get('candidate_detections', 0)}    •    "
        f"Analysis throughput: {summary.get('sampled_frames_per_second', 0):.2f} sampled frames/s    •    "
        f"Mean detector latency: {summary.get('detector_latency_mean_ms', 0):.1f} ms"
    )
    if summary.get("analysis_paused_seconds", 0) > 0:
        summary_text += f"    •    User-paused: {summary['analysis_paused_seconds']:.1f}s (excluded from throughput)"
    if summary.get("pose_latency_mean_ms") is not None:
        summary_text += f"    •    Mean pose latency: {summary['pose_latency_mean_ms']:.1f} ms"
    ttk.Label(header, text=summary_text).pack(anchor="w", pady=(5, 0))
    ttk.Label(
        header,
        text="Offline COCO person candidates only. Boxes and temporary IDs do not verify enemy, team, or player identity.",
        foreground="#9a4d00",
    ).pack(anchor="w", pady=(4, 0))
    if detector_evaluation:
        metric_text = detector_evaluation
        detector_text = (
            f"Manual-label detector evaluation ({metric_text.scored_frames} frames): "
            f"TP={metric_text.counts.true_positives} FP={metric_text.counts.false_positives} "
            f"FN={metric_text.counts.false_negatives}"
        )
        detector_text += (
            f" | precision={metric_text.precision:.3f}"
            if metric_text.precision is not None
            else " | precision=n/a"
        )
        detector_text += (
            f" | recall={metric_text.recall:.3f}" if metric_text.recall is not None else " | recall=n/a"
        )
        detector_text += f" | F1={metric_text.f1:.3f}" if metric_text.f1 is not None else " | F1=n/a"
    else:
        detector_text = "Detector evaluation: not available for this source (manual labels required)."
    ttk.Label(header, text=detector_text, wraplength=1150).pack(anchor="w", pady=(4, 0))
    if pose_evaluation:
        pose_text = "Pose-region localization: " + " | ".join(
            f"{item.region_name} {item.estimated_points}/{item.annotated_points} coverage "
            f"({item.coverage:.2f}), median {item.median_error_pixels:.1f}px / "
            f"{item.median_error_normalized:.5f} normalized"
            if item.median_error_pixels is not None
            else f"{item.region_name} {item.estimated_points}/{item.annotated_points} coverage ({item.coverage:.2f}), median n/a"
            for item in pose_evaluation
        )
    else:
        pose_text = "Pose-region evaluation: not available (manual region points required)."
    ttk.Label(header, text=pose_text, wraplength=1150).pack(anchor="w", pady=(4, 0))
    if alignment_summaries:
        alignment_text = "Geometric alignment error: " + " | ".join(
            f"{item['region_name']} n={item['measurements']}, median={item['median_error_pixels']:.1f}px "
            f"(p95 {item['p95_error_pixels']:.1f}px)"
            for item in alignment_summaries
        )
    else:
        alignment_text = "Geometric alignment: not recorded in this report."
    ttk.Label(header, text=alignment_text, wraplength=1150).pack(anchor="w", pady=(4, 0))

    body = ttk.Panedwindow(root, orient="horizontal")
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    image_panel = ttk.Frame(body, padding=4)
    side_panel = ttk.Frame(body, padding=4, width=300)
    body.add(image_panel, weight=4)
    body.add(side_panel, weight=1)
    canvas = tk.Canvas(image_panel, background="#141820", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    analysis_controls = ttk.LabelFrame(side_panel, text="Analyze local media", padding=8)
    analysis_controls.pack(fill="x", pady=(0, 10))

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
    playback_after_id: str | None = None
    playback_running = False
    analysis_results: queue.Queue[tuple[int, str, Path, Path, bool]] = queue.Queue()
    analysis_process: dict[str, subprocess.Popen[str] | None] = {"process": None}
    analysis_cancel = threading.Event()
    reopen_dashboard: tuple[Path, Path, bool] | None = None
    active_pause_file: dict[str, Path | None] = {"path": None}

    def show_frame(index: int) -> None:
        nonlocal current_index, displayed_photo
        current_index = max(0, min(index, len(frames) - 1))
        frame_record = frames[current_index]
        frame_index = int(frame_record["frame_index"])
        if source_is_image:
            success, image = True, source_image.copy()
        else:
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
            raise RuntimeError("Could not encode the sampled media frame for dashboard display")
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
            stop_playback()
            show_frame(selection[0])

    def stop_playback() -> None:
        nonlocal playback_after_id, playback_running
        playback_running = False
        if playback_after_id is not None:
            root.after_cancel(playback_after_id)
            playback_after_id = None

    def advance_playback() -> None:
        nonlocal playback_after_id
        playback_after_id = None
        if not playback_running:
            return
        if current_index + 1 >= len(frames):
            stop_playback()
            return
        previous_timestamp = float(frames[current_index].get("timestamp_seconds", 0.0))
        next_timestamp = float(frames[current_index + 1].get("timestamp_seconds", previous_timestamp))
        show_frame(current_index + 1)
        delay_ms = max(25, min(2000, round(max(0.025, next_timestamp - previous_timestamp) * 1000)))
        playback_after_id = root.after(delay_ms, advance_playback)

    def play_sampled_frames() -> None:
        nonlocal playback_running
        if playback_running:
            return
        if current_index + 1 >= len(frames):
            show_frame(0)
        playback_running = True
        advance_playback()

    def close_dashboard() -> None:
        stop_playback()
        analysis_cancel.set()
        if active_pause_file["path"] is not None:
            active_pause_file["path"].unlink(missing_ok=True)
        if analysis_process["process"] is not None and analysis_process["process"].poll() is None:
            try:
                analysis_process["process"].terminate()
            except OSError:
                pass
        if capture is not None:
            capture.release()
        root.destroy()

    def poll_analysis() -> None:
        nonlocal reopen_dashboard
        try:
            return_code, output, selected_media, new_report, selected_is_image = analysis_results.get_nowait()
        except queue.Empty:
            root.after(150, poll_analysis)
            return
        analysis_process["process"] = None
        analyze_button.configure(state="normal")
        cancel_button.configure(state="disabled")
        pause_button.configure(state="disabled", text="Pause video analysis")
        if active_pause_file["path"] is not None:
            active_pause_file["path"].unlink(missing_ok=True)
            active_pause_file["path"] = None
        if return_code == 0 and new_report.is_file():
            analysis_status.set(f"Analysis complete. Opening {new_report.name}…")
            reopen_dashboard = (new_report, selected_media, selected_is_image)
            if capture is not None:
                capture.release()
            root.destroy()
        elif analysis_cancel.is_set():
            analysis_status.set("Analysis cancelled.")
            messagebox.showinfo("VantaSight", "Offline analysis was cancelled. No report was opened.", parent=root)
        else:
            analysis_status.set("Analysis failed. See the message for the runner output.")
            messagebox.showerror(
                "Analysis failed",
                (output.strip() or "The offline analysis runner returned an error.")[-5000:],
                parent=root,
            )

    def analyze_another_media() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select a screenshot or recorded video for offline analysis",
            filetypes=[("Supported media", "*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*.*")],
        )
        if not selected:
            return
        media_to_analyze = Path(selected).resolve()
        is_image = media_to_analyze.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        if alignment_enabled.get() and not pose_enabled.get():
            messagebox.showerror("Alignment setting", "Enable body-region estimation before measuring alignment.", parent=root)
            return

        if is_image:
            metadata_image = cv2.imread(str(media_to_analyze))
            if metadata_image is None:
                messagebox.showerror("Unreadable screenshot", "VantaSight could not decode that image file.", parent=root)
                return
            height, width = metadata_image.shape[:2]
            media_details.set(f"Selected: {media_to_analyze.name} | Screenshot | {width}×{height}")
        else:
            metadata_capture = cv2.VideoCapture(str(media_to_analyze))
            if not metadata_capture.isOpened():
                messagebox.showerror("Unreadable video", "VantaSight could not open that video file.", parent=root)
                return
            width = int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = metadata_capture.get(cv2.CAP_PROP_FPS)
            count = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata_capture.release()
            duration = count / fps if fps > 0 else 0
            media_details.set(f"Selected: {media_to_analyze.name} | {width}×{height} | {fps:.2f} FPS | {duration:.1f}s")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_output = (Path("reports") / f"{media_to_analyze.stem}_{stamp}_{'image' if is_image else 'tracks'}.json").resolve()
        csv_output = report_output.with_suffix(".csv")
        if is_image:
            active_pause_file["path"] = None
            runner = Path(__file__).resolve().parent / "analyze_image.py"
            command = [sys.executable, str(runner), str(media_to_analyze), "--output", str(csv_output), "--json-output", str(report_output)]
        else:
            frame_step_text = frame_step_entry.get().strip()
            try:
                frame_step = int(frame_step_text)
                if frame_step < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid sampling interval", "Frame step must be a positive whole number.", parent=root)
                return
            runner = Path(__file__).resolve().parent / "track_video.py"
            pause_file = report_output.with_suffix(".pause")
            pause_file.unlink(missing_ok=True)
            active_pause_file["path"] = pause_file
            command = [sys.executable, str(runner), str(media_to_analyze), "--frame-step", str(frame_step), "--output", str(csv_output), "--json-output", str(report_output), "--pause-file", str(pause_file)]
        if pose_enabled.get():
            command.append("--estimate-body-regions")
        if alignment_enabled.get():
            command.append("--measure-alignment")

        analysis_cancel.clear()
        analyze_button.configure(state="disabled")
        cancel_button.configure(state="normal")
        pause_button.configure(state="normal" if not is_image else "disabled", text="Pause video analysis")
        analysis_status.set(f"Analyzing {media_to_analyze.name} offline…")

        def run_worker() -> None:
            process: subprocess.Popen[str] | None = None
            output = ""
            return_code = 1
            try:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                analysis_process["process"] = process
                while True:
                    if analysis_cancel.is_set() and process.poll() is None:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    try:
                        output, _ = process.communicate(timeout=0.2)
                        return_code = process.returncode
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except OSError as exc:
                output = str(exc)
            analysis_results.put((return_code, output, media_to_analyze, report_output, is_image))

        threading.Thread(target=run_worker, name="vantasight-offline-analysis", daemon=True).start()
        root.after(150, poll_analysis)

    def cancel_analysis() -> None:
        analysis_cancel.set()
        if active_pause_file["path"] is not None:
            active_pause_file["path"].unlink(missing_ok=True)
        cancel_button.configure(state="disabled")
        pause_button.configure(state="disabled", text="Pause video analysis")
        analysis_status.set("Stopping offline analysis…")

    def toggle_analysis_pause() -> None:
        pause_file = active_pause_file["path"]
        if pause_file is None:
            return
        if pause_file.exists():
            pause_file.unlink(missing_ok=True)
            pause_button.configure(text="Pause video analysis")
            analysis_status.set("Video analysis resumed.")
        else:
            pause_file.parent.mkdir(parents=True, exist_ok=True)
            pause_file.touch()
            pause_button.configure(text="Resume video analysis")
            analysis_status.set("Video analysis paused; resume to continue.")

    media_details = tk.StringVar(value="Choose a local screenshot or video to inspect its metadata.")
    ttk.Label(analysis_controls, textvariable=media_details, wraplength=250).pack(anchor="w", pady=(0, 5))
    ttk.Label(analysis_controls, text="Video sampling: every N source frames").pack(anchor="w")
    frame_step_entry = ttk.Entry(analysis_controls, width=10)
    frame_step_entry.insert(0, "30")
    frame_step_entry.pack(anchor="w", pady=(2, 5))
    pose_enabled = tk.BooleanVar(value=False)
    alignment_enabled = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_controls, text="Estimate body regions (slower)", variable=pose_enabled).pack(anchor="w")
    ttk.Checkbutton(analysis_controls, text="Measure frame-center alignment", variable=alignment_enabled).pack(anchor="w")
    analyze_button = ttk.Button(analysis_controls, text="Choose image or video…", command=analyze_another_media)
    analyze_button.pack(fill="x", pady=(6, 2))
    cancel_button = ttk.Button(analysis_controls, text="Cancel analysis", command=cancel_analysis, state="disabled")
    cancel_button.pack(fill="x")
    pause_button = ttk.Button(
        analysis_controls,
        text="Pause video analysis",
        command=toggle_analysis_pause,
        state="disabled",
    )
    pause_button.pack(fill="x", pady=(4, 0))
    analysis_status = tk.StringVar(value="Analysis reads only the selected local media.")
    ttk.Label(analysis_controls, textvariable=analysis_status, wraplength=250).pack(anchor="w", pady=(5, 0))

    frame_list.bind("<<ListboxSelect>>", on_select)
    controls = ttk.Frame(image_panel, padding=(0, 8, 0, 0))
    controls.pack(fill="x")
    ttk.Button(controls, text="◀ Previous", command=lambda: (stop_playback(), show_frame(current_index - 1))).pack(side="left")
    ttk.Button(controls, text="Next ▶", command=lambda: (stop_playback(), show_frame(current_index + 1))).pack(side="left", padx=8)
    ttk.Button(controls, text="▶ Play samples", command=play_sampled_frames).pack(side="left", padx=(14, 4))
    ttk.Button(controls, text="Pause", command=stop_playback).pack(side="left", padx=4)
    ttk.Button(controls, text="Stop", command=lambda: (stop_playback(), show_frame(0))).pack(side="left", padx=4)
    ttk.Label(controls, text=f"Report: {report_path}").pack(side="right")
    ttk.Label(image_panel, text="Playback advances through analyzed samples only; it does not play every source-video frame.").pack(anchor="w")

    root.update_idletasks()
    show_frame(0)
    root.protocol("WM_DELETE_WINDOW", close_dashboard)
    root.mainloop()
    if reopen_dashboard is not None:
        next_report, next_source, is_image = reopen_dashboard
        return main([str(next_report), "--image" if is_image else "--video", str(next_source)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
