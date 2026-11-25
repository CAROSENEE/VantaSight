"""Choose and validate downloaded browser-review labels for a numbered video."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _video_name_from_review(review_path: Path) -> str:
    video_names: set[str] = set()
    with review_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("video"), str):
                raise ValueError(f"Line {line_number} has no video filename")
            video_names.add(record["video"])

    if len(video_names) != 1:
        raise ValueError("The selected file must contain review rows for exactly one video")
    video_name = video_names.pop()
    if Path(video_name).name != video_name or not re.fullmatch(r"vd-[0-9]+\.mp4", video_name):
        raise ValueError("Review rows must name a numbered source video such as vd-1.mp4")
    return video_name


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
    importer = project_root / "tools" / "import_review_annotations.py"
    review_file = filedialog.askopenfilename(
        parent=root,
        title="Select the downloaded, manually reviewed JSONL file",
        initialdir=Path.home() / "Downloads",
        filetypes=[("JSON Lines review labels", "*.jsonl"), ("All files", "*.*")],
    )
    if not review_file:
        root.destroy()
        return 0

    review_path = Path(review_file)
    try:
        video_name = _video_name_from_review(review_path)
    except (OSError, ValueError) as exc:
        messagebox.showerror("VantaSight review file is invalid", str(exc), parent=root)
        root.destroy()
        return 2

    video = project_root / "data" / "input" / "videos" / video_name
    shortlist = project_root / "reports" / f"{Path(video_name).stem}_manual_review_shortlist.csv"
    output = project_root / "data" / "annotations" / f"{Path(video_name).stem}_user_verified.jsonl"
    missing = [
        str(path.relative_to(project_root))
        for path in (video, shortlist, importer)
        if not path.is_file()
    ]
    if missing:
        messagebox.showerror(
            "VantaSight import setup incomplete",
            "Required project files are missing:\n\n" + "\n".join(missing),
            parent=root,
        )
        root.destroy()
        return 2

    command = [
        sys.executable,
        str(importer),
        str(video),
        str(review_path),
        "--shortlist-csv",
        str(shortlist),
        "--output",
        str(output),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        messagebox.showerror("VantaSight import failed", str(exc), parent=root)
        root.destroy()
        return 2

    details = result.stdout.strip() or result.stderr.strip() or "No details were returned."
    if result.returncode == 0:
        messagebox.showinfo("VantaSight labels imported", details, parent=root)
    else:
        messagebox.showerror(
            "VantaSight could not import these labels",
            details + "\n\nNo labels were generated. Check the selected file and review all listed frames.",
            parent=root,
        )

    root.destroy()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
