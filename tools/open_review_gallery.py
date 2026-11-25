"""Open a local review gallery for a selected numbered gameplay video."""

from __future__ import annotations

import re
import subprocess
import sys
import webbrowser
from pathlib import Path

from vantasight.input import VideoReader
from vantasight.input.sampling import frame_step_for_sample_rate


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:
        print(f"Could not load the Windows file dialog: {exc}", file=sys.stderr)
        return 2

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    project_root = Path(__file__).resolve().parents[1]
    videos_dir = project_root / "data" / "input" / "videos"
    video_name = filedialog.askopenfilename(
        parent=root,
        title="Select a numbered VantaSight source video",
        initialdir=videos_dir,
        filetypes=[("Numbered MP4 videos", "vd-*.mp4")],
    )
    if not video_name:
        root.destroy()
        return 0

    video = Path(video_name).resolve()
    if video.parent != videos_dir.resolve() or re.fullmatch(r"vd-[1-9][0-9]*\.mp4", video.name) is None:
        messagebox.showerror(
            "VantaSight video name is invalid",
            "Choose an MP4 named vd-N.mp4 from data/input/videos.",
            parent=root,
        )
        root.destroy()
        return 2

    stem = video.stem
    report = project_root / "reports" / f"{stem}_dashboard_tracks_dense.json"
    shortlist = project_root / "reports" / f"{stem}_manual_review_shortlist.csv"
    shortlist_tool = project_root / "tools" / "create_review_shortlist.py"
    exporter = project_root / "tools" / "export_review_gallery.py"
    python = Path(sys.executable)

    if not shortlist.is_file():
        if not report.is_file():
            try:
                with VideoReader(video) as reader:
                    if reader.info is None:
                        raise ValueError("Could not read video metadata")
                    frame_step = frame_step_for_sample_rate(reader.info.fps)
            except (OSError, ValueError) as exc:
                messagebox.showerror("VantaSight video metadata failed", str(exc), parent=root)
                root.destroy()
                return 2
            command = (
                f'.venv\\Scripts\\python.exe tools\\track_video.py '
                f'"data\\input\\videos\\{video.name}" --frame-step {frame_step} '
                f'--preserve-source-resolution --output '
                f'"reports\\{stem}_dashboard_tracks_dense.csv" --json-output '
                f'"reports\\{stem}_dashboard_tracks_dense.json"'
            )
            messagebox.showerror(
                "VantaSight tracking report is missing",
                f"Generate the offline tracking report first:\n\n{command}",
                parent=root,
            )
            root.destroy()
            return 2
        result = subprocess.run(
            [str(python), str(shortlist_tool), str(report), "--output", str(shortlist)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "Could not create the shortlist."
            messagebox.showerror("VantaSight shortlist failed", details, parent=root)
            root.destroy()
            return result.returncode

    output_dir = project_root / "data" / "previews" / f"{stem}_review_gallery"
    result = subprocess.run(
        [
            str(python),
            str(exporter),
            str(video),
            "--shortlist-csv",
            str(shortlist),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "Could not export the review gallery."
        messagebox.showerror("VantaSight gallery export failed", details, parent=root)
        root.destroy()
        return result.returncode

    gallery = output_dir / "index.html"
    opened = webbrowser.open(gallery.resolve().as_uri())
    messagebox.showinfo(
        "VantaSight offline review gallery",
        f"Gallery: {gallery}\n\n"
        + ("The browser was asked to open the local gallery." if opened else "If no browser page appears, open this local file manually."),
        parent=root,
    )
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
