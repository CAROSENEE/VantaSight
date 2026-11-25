"""Choose and validate a downloaded vd-1 browser-review JSONL offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    video = project_root / "data" / "input" / "videos" / "vd-1.mp4"
    shortlist = project_root / "reports" / "vd-1_manual_review_shortlist.csv"
    importer = project_root / "tools" / "import_review_annotations.py"
    output = project_root / "data" / "annotations" / "vd-1_user_verified.jsonl"

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

    review_file = filedialog.askopenfilename(
        parent=root,
        title="Select the downloaded vd-1 reviewed labels JSONL",
        initialdir=Path.home() / "Downloads",
        filetypes=[("JSON Lines review labels", "*.jsonl"), ("All files", "*.*")],
    )
    if not review_file:
        root.destroy()
        return 0

    command = [
        sys.executable,
        str(importer),
        str(video),
        review_file,
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
