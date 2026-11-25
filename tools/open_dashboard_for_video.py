"""Choose a numbered video and open its saved offline dashboard report."""

from __future__ import annotations

import re
import subprocess
import sys
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
    selected = filedialog.askopenfilename(
        parent=root,
        title="Select a numbered VantaSight source video",
        initialdir=videos_dir,
        filetypes=[("Numbered MP4 videos", "vd-*.mp4")],
    )
    if not selected:
        root.destroy()
        return 0

    video = Path(selected).resolve()
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

    viewer = project_root / "tools" / "open_dashboard.py"
    result = subprocess.run(
        [sys.executable, str(viewer), str(report)],
        cwd=project_root,
        check=False,
    )
    root.destroy()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
